"""Numerical agreement with an independent dense Ridge solver."""
import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from benchmarks.mercari_solver import SparseRidge


def test_preconditioned_sparse_ridge_matches_dense_reference_and_serializes():
    import pickle
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 9)) * np.geomspace(.01, 100, 9)
    x[x < 0] = 0
    y = x @ rng.normal(size=9) + 7. + rng.normal(size=100)
    expected = Ridge(alpha=3., solver='cholesky').fit(x, y)
    fitted = SparseRidge(alpha=3., tol=1e-10).fit(sparse.csr_matrix(x), y)
    np.testing.assert_allclose(fitted.predict(sparse.csr_matrix(x)), expected.predict(x), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(fitted.coef_, expected.coef_, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fitted)).predict(x), fitted.predict(x))
    assert fitted.n_iter_ > 0


def test_solver_rejects_invalid_data_settings_and_nonconvergence():
    x = sparse.eye(4, format='csr')
    for settings in [dict(alpha=0), dict(tol=0), dict(max_iter=0)]:
        with pytest.raises(ValueError):
            SparseRidge(**settings).fit(x, [1., 2., 3., 4.])
    with pytest.raises(ValueError):
        SparseRidge().fit(x, [1., np.nan, 3., 4.])
    with pytest.raises(RuntimeError, match='converge'):
        SparseRidge(max_iter=1, tol=1e-15).fit(np.array([[1., 2.], [2., 8.], [4., -1.]]), [1., 5., 0.])
