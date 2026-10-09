import gc
import numpy as np
import pytest
from libtree import GBDTRegressor, GBDTClassifier


def test_fit_predict_refit_and_ownership():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = (x[:, 0] > 20).astype(np.float32)
    original = x.copy()
    with GBDTRegressor(n_estimators=80, min_samples_leaf=1) as model:
        assert model.fit(x, y) is model
        assert model.score(x, y) > 0.999
        assert model.training_loss_[-1] < model.training_loss_[0]
        assert np.array_equal(x, original)
        assert model.predict(np.empty((0, 2))).shape == (0,)
        model.fit(x, np.full(20, 7))
        np.testing.assert_allclose(model.predict(x), 7)
        with pytest.raises(ValueError):
            model.fit(x, np.full(20, np.nan))
        np.testing.assert_allclose(model.predict(x), 7)
    with pytest.raises(RuntimeError):
        model.predict(x)
    model.close()


def test_classifier_and_noncontiguous_inputs():
    x = np.arange(40).reshape(10, 4)[:, ::2]
    y = np.array([0] * 5 + [1] * 5)
    model = GBDTClassifier(n_estimators=60, min_samples_leaf=1)
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x), y)
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1)
    assert model.score(x, y) == 1
    assert model.classes_.tolist() == [0, 1]
    del model
    gc.collect()


@pytest.mark.parametrize("x,y", [
    ([], []), ([[1]], [1, 2]), ([[1]], [[1]]),
    ([[np.inf]], [1]), ([[1]], [np.nan]), (np.empty((0, 1)), []),
])
def test_bad_data(x, y):
    with pytest.raises(ValueError):
        GBDTRegressor().fit(x, y)


@pytest.mark.parametrize("params", [
    {"n_estimators": 0}, {"n_estimators": 1.5}, {"max_depth": -1},
    {"max_bins": 65536}, {"learning_rate": np.inf}, {"reg_lambda": -1},
    {"min_gain": np.nan}, {"min_samples_leaf": 0},
])
def test_bad_parameters(params):
    with pytest.raises(ValueError):
        GBDTRegressor(**params).fit([[1]], [1])


def test_missing_and_validation():
    model = GBDTRegressor(min_samples_leaf=1)
    with pytest.raises(RuntimeError):
        _ = model.training_loss_
    model.fit([[np.nan], [np.nan], [1], [1]], [0, 0, 1, 1])
    assert model.predict([[np.nan]])[0] < 0.01
    with pytest.raises(ValueError):
        model.predict([[1, 2]])
    with pytest.raises(ValueError):
        model.set_params(unknown=1)
    assert model.set_params(max_depth=2).get_params()["max_depth"] == 2
    with pytest.raises(ValueError):
        GBDTClassifier().fit([[1], [2]], [0, 2])


def test_deterministic_and_clone():
    from sklearn.base import clone
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = x[:, 0] ** 2 + x[:, 1]
    first = GBDTRegressor(n_estimators=20).fit(x, y)
    second = clone(first).fit(x, y)
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
