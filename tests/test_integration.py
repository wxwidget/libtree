"""Public sklearn/SHAP contracts, including process boundaries and additivity."""
import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from libtree import C5Classifier, GBDTClassifier, GBDTRegressor

MODELS = [GBDTRegressor, GBDTClassifier, C5Classifier]


def data(cls):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(100, 3)).astype(np.float32)
    y = x[:, 0] + .3 * x[:, 1]
    if cls is not GBDTRegressor:
        y = np.where(y > 0, 'risk', 'safe')
    return x, y


@pytest.mark.parametrize('cls', MODELS)
def test_estimator_protocol_and_process_ensembles(cls):
    x, y = data(cls)
    model = cls(n_estimators=3, min_samples_leaf=1, n_jobs=2)
    assert is_regressor(model) == (cls is GBDTRegressor)
    assert is_classifier(model) == (cls is not GBDTRegressor)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(x, y)
    check_is_fitted(model)
    assert pipeline.score(x, y) > .4
    assert not clone(model).__sklearn_is_fitted__()
    regression = cls is GBDTRegressor
    voting = VotingRegressor if regression else VotingClassifier
    stacking = StackingRegressor if regression else StackingClassifier
    bagging = BaggingRegressor if regression else BaggingClassifier
    options = {} if regression else {'voting': 'soft'}
    ensembles = [voting([('a', clone(model)), ('b', clone(model))], n_jobs=2, **options),
                 stacking([('a', clone(model))], cv=2, n_jobs=2),
                 bagging(clone(model), n_estimators=2, n_jobs=2, random_state=7)]
    for ensemble in ensembles:
        ensemble.fit(x, y)
        prediction = ensemble.predict(x)
        assert prediction.shape == (len(x),)
        assert ensemble.score(x, y) > .3
        restored = pickle.loads(pickle.dumps(ensemble))
        np.testing.assert_array_equal(restored.predict(x), prediction)
    model.close()
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize('cls', MODELS)
def test_roundtrip_keeps_model_not_training_data(cls):
    x, y = data(cls)
    x[::13, 1] = np.nan
    with cls(n_estimators=4, min_samples_leaf=1, n_jobs=2) as model:
        model.fit(x, y)
        before = model.predict(x)
        inspection = model.dump_model(format='json')
        history = (model.training_loss_ if cls is GBDTRegressor or cls is GBDTClassifier
                   else model.estimator_errors_)
        model.set_params(max_depth=0, n_jobs=1)  # pending parameters do not rewrite trees
        for restored in (copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
            np.testing.assert_array_equal(restored.predict(x), before)
            assert restored.dump_model(format='json') == inspection
            np.testing.assert_array_equal(
                restored.estimator_errors_ if cls is C5Classifier else restored.training_loss_, history)
            assert restored.get_params() == model.get_params()
            if cls is C5Classifier:
                assert restored.dump_rules('json') == model.dump_rules('json')
            restored.close()
        model.close()
        restored = pickle.loads(pickle.dumps(model))
        assert not restored.__sklearn_is_fitted__()
    assert not pickle.loads(pickle.dumps(cls())).__sklearn_is_fitted__()


@pytest.mark.parametrize('cls', MODELS)
def test_shap_public_callable_and_factory_additivity(cls):
    import shap
    from libtree import shap_explainer
    x, y = data(cls)
    x[::13, 1] = np.nan
    with cls(n_estimators=3, min_samples_leaf=1) as model:
        model.fit(x, y)
        background = x[:8]
        rows = x[8:11]
        expected = model.predict(rows) if cls is GBDTRegressor else model.predict_proba(rows)
        explainers = [model.shap_explainer(background, algorithm='exact',
                                          feature_names=['income', 'debt', 'age']),
                      shap.Explainer(model, background, algorithm='exact'),
                      shap_explainer(model, background, algorithm='permutation', seed=7)]
        for explainer in explainers:
            values = explainer(rows, max_evals=32)
            np.testing.assert_allclose(values.base_values + values.values.sum(axis=1), expected,
                                       atol=2e-7, rtol=2e-7)
            assert np.isfinite(values.values).all()
        assert explainers[0](rows).feature_names == ['income', 'debt', 'age']
        if cls is not GBDTRegressor:
            assert explainers[0](rows).output_names == list(model.classes_)


def test_multiclass_shap_and_explained_ensemble():
    from libtree import shap_explainer
    x, _ = data(C5Classifier)
    y = np.where(x[:, 0] < -.5, 'low', np.where(x[:, 0] > .5, 'high', 'middle'))
    model = C5Classifier(n_estimators=3, min_samples_leaf=1).fit(x, y)
    values = model.shap_explainer(x[:6], algorithm='exact')(x[6:8])
    assert values.values.shape == (2, 3, 3)
    np.testing.assert_allclose(values.base_values + values.values.sum(1), model.predict_proba(x[6:8]))
    ensemble = VotingClassifier([('c5', clone(model))], voting='soft').fit(x, y)
    values = shap_explainer(ensemble, x[:6], algorithm='exact')(x[6:8])
    np.testing.assert_allclose(values.base_values + values.values.sum(1), ensemble.predict_proba(x[6:8]))
    model.close()


def test_analysis_validation_names_and_optional_dependency(monkeypatch):
    import builtins
    import pandas as pd
    from libtree import shap_explainer
    x, y = data(GBDTRegressor)
    model = GBDTRegressor(n_estimators=2)
    with pytest.raises(RuntimeError, match='not fitted'):
        model.shap_explainer(x[:4])
    model.fit(x, y)
    for background in ([], np.empty((0, 3)), [[np.inf, 1, 2]], [[1, 2]], [1, 2, 3]):
        with pytest.raises(ValueError, match='background'):
            shap_explainer(model, background)
    for names in ('abc', ['a'], ['a', 'b', 1]):
        with pytest.raises(ValueError, match='feature_names'):
            shap_explainer(model, x[:4], feature_names=names)
    with pytest.raises(ValueError, match='algorithm'):
        shap_explainer(model, x[:4], algorithm='tree')
    frame = pd.DataFrame(x[:4], columns=['income', 'debt', 'age'])
    values = shap_explainer(model, frame)(frame.iloc[:2])
    assert values.feature_names == list(frame.columns)
    np.testing.assert_allclose(values.base_values + values.values.sum(1), model.predict(frame.iloc[:2]))
    list_values = shap_explainer(model, x[:4].tolist(), algorithm='exact')(x[4:6].tolist())
    np.testing.assert_allclose(list_values.base_values + list_values.values.sum(1),
                               model.predict(x[4:6]))
    real_import = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == 'shap':
            raise ImportError('optional dependency absent')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_shap)
    with pytest.raises(ImportError, match=r'libtree-gbdt\[explain\]'):
        shap_explainer(model, x[:4])
    model.close()


@pytest.mark.parametrize('cls', MODELS)
def test_snapshot_failure_boundaries_preserve_ownership(cls, monkeypatch):
    import libtree
    x, y = data(cls)
    with cls(n_estimators=2, n_jobs=2) as model:
        model.fit(x, y)
        state = model.__getstate__()
        assert '_handle' not in state and '_finalizer' not in state
        assert all(not isinstance(value, np.ndarray) or key == 'classes_'
                   for key, value in state.items())
        before = model.predict(x)
        bad_state = dict(state, _native_snapshot=b'truncated')
        with pytest.raises(ValueError, match='invalid|truncated'):
            cls.__new__(cls).__setstate__(bad_state)
        lib = libtree._library()
        serialize_name = model._native_prefix + 'Serialize'
        serialize = getattr(lib, serialize_name)
        for fail_on in (1, 2):
            calls = 0

            def failing_serialize(*args):
                nonlocal calls
                calls += 1
                return -1 if calls == fail_on else serialize(*args)

            with monkeypatch.context() as patch:
                patch.setattr(lib, serialize_name, failing_serialize)
                with pytest.raises(RuntimeError):
                    pickle.dumps(model)
            np.testing.assert_array_equal(model.predict(x), before)
        assert model._more_tags()['allow_nan']


def test_classifier_label_validation_preserves_fitted_model():
    x, y = data(GBDTClassifier)
    with GBDTClassifier(n_estimators=3).fit(x, y) as model:
        before = model.predict(x)
        for bad in ([], [[0]] * len(x), [None] * len(x), [np.nan] * len(x),
                    np.arange(len(x)), ['one'] * len(x), ['a', 'b']):
            with pytest.raises(ValueError):
                model.fit(x, bad)
            np.testing.assert_array_equal(model.predict(x), before)
            assert model.classes_.tolist() == ['risk', 'safe']


def test_c5_weighted_adaboost_and_regression_adaboost():
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    x, y = data(C5Classifier)
    model = AdaBoostClassifier(C5Classifier(boosting='none', max_depth=1, min_samples_leaf=1),
                               n_estimators=3, random_state=7).fit(x, y)
    assert model.score(x, y) > .85
    rx, ry = data(GBDTRegressor)
    regression = AdaBoostRegressor(GBDTRegressor(n_estimators=20, min_samples_leaf=1),
                                   n_estimators=3, random_state=7).fit(rx, ry)
    assert regression.score(rx, ry) > .5


def test_sklearn_selection_and_permutation_importance():
    from sklearn.model_selection import GridSearchCV
    from sklearn.inspection import permutation_importance
    from sklearn.inspection import partial_dependence
    for cls in MODELS:
        x, y = data(cls)
        search = GridSearchCV(cls(n_estimators=3, min_samples_leaf=1),
                              {'max_depth': [1, 2]}, cv=2, n_jobs=2)
        search.fit(x, y)
        assert np.isfinite(search.best_score_)
        result = permutation_importance(search.best_estimator_, x, y, n_repeats=2,
                                         random_state=7, n_jobs=2)
        assert result.importances.shape == (3, 2)
        dependence = partial_dependence(search.best_estimator_, x, [0],
                                        method='brute', grid_resolution=5)
        assert dependence.average.shape[-1] == 5


def test_explain_pipeline_in_input_feature_space():
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from libtree import shap_explainer
    x, y = data(C5Classifier)
    frame = pd.DataFrame(x, columns=['income', 'debt', 'age'])
    preprocess = ColumnTransformer([('scale', StandardScaler(), ['income', 'debt'])])
    pipeline = make_pipeline(preprocess, C5Classifier(n_estimators=3)).fit(
        frame, pd.Series(y, dtype=object))
    values = shap_explainer(pipeline, frame.iloc[:5], algorithm='exact')(frame.iloc[5:7])
    np.testing.assert_allclose(values.base_values + values.values.sum(1),
                               pipeline.predict_proba(frame.iloc[5:7]))
    np.testing.assert_array_equal(values.values[:, 2], 0)  # Dropped input cannot affect output.
    pipeline[-1].close()


@pytest.mark.parametrize('cls', [GBDTClassifier, C5Classifier])
def test_dataframe_string_targets(cls):
    import pandas as pd
    x, y = data(cls)
    with cls(n_estimators=10, min_samples_leaf=1) as model:
        model.fit(pd.DataFrame(x), pd.Series(y, dtype=object))
        assert model.score(x, y) > .9
        assert model.classes_.tolist() == ['risk', 'safe']
