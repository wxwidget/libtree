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


@pytest.mark.parametrize('cls', [GBDTRegressor, GBDTClassifier])
@pytest.mark.parametrize('categorical', [False, True])
def test_parallel_exact_and_concurrent(cls, categorical):
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.base import clone
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8193, 20)).astype(np.float32)
    if categorical:
        x[:, :12] = x[:, :12] > 1
    x[::31, 1] = np.nan
    y = x[:, 0] ** 2 + x[:, 2]
    if cls is GBDTClassifier:
        y = (y > 1).astype(np.float32)
    with cls(n_estimators=12, n_jobs=1) as serial:
        serial.fit(x, y)
        expected = serial.predict(x)
        for threads in (2, 4):
            with clone(serial).set_params(n_jobs=threads) as parallel:
                parallel.fit(x, y)
                np.testing.assert_array_equal(parallel.training_loss_, serial.training_loss_)
                with ThreadPoolExecutor(max_workers=4) as callers:
                    for prediction in callers.map(parallel.predict, [x] * 8):
                        np.testing.assert_array_equal(prediction, expected)
                parallel.set_params(n_jobs=1)
                np.testing.assert_array_equal(parallel.predict(x), expected)
                with pytest.raises(ValueError):
                    parallel.set_params(n_jobs=0)
                assert parallel.n_jobs == 1


@pytest.mark.parametrize('threads', [0, -1, 1.5, 257])
def test_invalid_threads(threads):
    with pytest.raises(ValueError):
        GBDTRegressor(n_jobs=threads).fit([[1]], [1])


def test_parallel_workers_released_and_failed_refit():
    from pathlib import Path
    def count():
        return len(list(Path('/proc/self/task').iterdir()))
    baseline = count()
    x = np.arange(40000, dtype=np.float32).reshape(2000, 20)
    y = x[:, 0] / 100
    for _ in range(15):
        with GBDTRegressor(n_estimators=3, n_jobs=4) as model:
            model.fit(x, y)
            before = model.predict(x)
            with pytest.raises(ValueError):
                model.fit(x, np.full(len(x), np.nan))
            np.testing.assert_array_equal(before, model.predict(x))
        assert count() == baseline
