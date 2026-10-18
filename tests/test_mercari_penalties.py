"""Sparse implicit centering must solve the same problem as sklearn."""

import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso

from benchmarks.mercari_penalties import SparseElasticNet, SparseLasso


@pytest.mark.parametrize(
    "reference,implementation", [(Lasso, SparseLasso), (ElasticNet, SparseElasticNet)]
)
@pytest.mark.parametrize("intercept", [True, False])
def test_matches_sklearn_on_correlated_nonzero_mean_and_constant_columns(
    reference, implementation, intercept
):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 12)) + 2
    x[:, 2] = x[:, 0] + rng.normal(scale=0.2, size=len(x))
    x[:, 3] = 1
    x[:, 4] = 0
    y = 7 + x[:, 0] * 3 - x[:, 1] * 2 + rng.normal(scale=0.01, size=len(x))
    kwargs = dict(alpha=0.03, fit_intercept=intercept, tol=1e-9, max_iter=5000)
    expected = reference(**kwargs).fit(x, y)
    actual = implementation(**kwargs).fit(sparse.csc_matrix(x), y)
    np.testing.assert_allclose(
        actual.predict(sparse.csr_matrix(x)), expected.predict(x), atol=2e-4
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_, atol=4e-4)
    assert actual.intercept_ == pytest.approx(expected.intercept_, abs=2e-4)
    assert (
        actual.dual_gap_ <= 1e-9 * np.var(y)
        if intercept
        else actual.dual_gap_ <= 1e-9 * np.mean(y * y)
    )
    np.testing.assert_array_equal(
        actual.predict(x), pickle.loads(pickle.dumps(actual)).predict(x)
    )


def test_duplicates_constant_target_and_nonconvergence_are_handled():
    x = sparse.csc_matrix(([1.0, 2.0, 1.0, 2.0], [0, 0, 1, 2], [0, 4]), shape=(3, 1))
    raw = x.data.copy()
    model = SparseLasso(alpha=0.01).fit(x, np.full(3, 4.0))
    np.testing.assert_array_equal(model.predict(x), [4.0, 4.0, 4.0])
    np.testing.assert_array_equal(raw, x.data)
    assert model.dual_gap_ == 0
    rng = np.random.default_rng(7)
    a = rng.normal(size=(80, 8))
    with pytest.warns(ConvergenceWarning):
        SparseElasticNet(alpha=1e-8, max_iter=1, tol=1e-15).fit(a, rng.normal(size=80))
    for kwargs in [
        dict(alpha=0),
        dict(selection="random"),
        dict(warm_start=True),
        dict(positive=True),
        dict(precompute=True),
        dict(l1_ratio=0),
    ]:
        with pytest.raises(ValueError):
            SparseElasticNet(**kwargs).fit(a, np.ones(80))
