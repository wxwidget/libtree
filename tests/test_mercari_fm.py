"""Exact pairwise FM arithmetic and race-free local SGD semantics."""
import pickle
import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('numba')
from benchmarks.mercari_fm import FactorizationMachine


def test_factorization_prediction_matches_explicit_feature_pairs():
    x = sparse.csr_matrix([[2., 0., 3.], [0., 0., 0.]])
    model = FactorizationMachine(features=3, rank=2, workers=2)
    model.factors_[:] = [[.2, .3], [.4, .5], [.6, .7]]
    model.linear_[:] = [1., 2., 3., 4.]
    expected = 4 + 2 + 9 + 6 * (.2 * .6 + .3 * .7)
    np.testing.assert_allclose(model.predict(x), [expected, 4.], rtol=1e-6)
    np.testing.assert_array_equal(model.predict(x), pickle.loads(pickle.dumps(model)).predict(x))


def test_local_sgd_learns_interactions_and_synchronizes_owned_weights():
    rng = np.random.default_rng(42)
    x = sparse.csr_matrix(rng.integers(0, 2, size=(200, 5)).astype(np.float32))
    y = 2 * x.toarray()[:, 0] * x.toarray()[:, 1] - .3
    model = FactorizationMachine(5, rank=4, workers=2, learning_rate=.05, chunk_size=40)
    before = np.mean((model.predict(x) - y)**2)
    for epoch in range(8):
        model.fit_epoch(x, y, seed=epoch)
    assert np.mean((model.predict(x) - y)**2) < before * .4
    np.testing.assert_array_equal(model.factors_[0], model.factors_[1])
    np.testing.assert_array_equal(model.linear_[0], model.linear_[1])
    with pytest.raises(ValueError):
        model.fit_epoch(x, [np.nan] * len(y))
    with pytest.raises(ValueError):
        model.predict(sparse.csr_matrix(np.ones((2, 4))))
    with pytest.raises(ValueError):
        FactorizationMachine(5, rank=0)


def test_field_aware_interactions_use_the_other_features_field_embedding():
    model = FactorizationMachine(3, rank=2, workers=1, fields=[0, 0, 1])
    model.factors_[0] = [[.2, .3, .7, .8], [.4, .5, .9, 1.], [.5, .6, 1.1, 1.2]]
    x = sparse.csr_matrix([[2., 0., 3.], [1., 1., 1.]])
    expected = [6 * (.7 * .5 + .8 * .6),
                .2 * .4 + .3 * .5 + .7 * .5 + .8 * .6 + .9 * .5 + 1. * .6]
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-6)
    model.fit_epoch(x, [1., 2.])
    assert np.isfinite(model.predict(x)).all()
    for fields in [[0], [-1, 0, 1], [0., .5, 1.]]:
        with pytest.raises(ValueError):
            FactorizationMachine(3, fields=fields)


def test_huber_gradient_and_frozen_inference_are_finite_and_immutable():
    x = sparse.csr_matrix([[1., 0., 0.], [0., 1., 1.]])
    model = FactorizationMachine(3, workers=1, error_clip=.5)
    model.fit_epoch(x, [1e5, -1e5])
    frozen = model.freeze()
    np.testing.assert_array_equal(model.predict(x), frozen.predict(x))
    before = frozen.predict(x)
    model.fit_epoch(x, [3., 4.])
    np.testing.assert_array_equal(frozen.predict(x), before)
    assert np.isfinite(before).all()
    with pytest.raises(ValueError):
        FactorizationMachine(3, error_clip=0)
    with pytest.raises(ValueError):
        frozen.predict(sparse.csr_matrix([[1., 2.]]))
