"""Sparse Lasso/Elastic Net with an unpenalized intercept and implicit centering.

The objective and sklearn prediction API are preserved. Track the raw residual
and its sum; (X_j-mean_j).T @ r is X_j.T @ raw_r - mean_j*sum(raw_r).
Each coordinate touches only its nonzeros, rather than shifting every row.
Coordinate descent: Friedman et al. (2010), doi:10.18637/jss.v033.i01.
Primal/dual gap follows the Elastic Net dual construction (same reference and
sklearn 1.8 linear_model/_cd_fast.pyx, gap_enet_sparse). Independently implemented.
"""

import warnings

import numpy as np
from numba import njit
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.utils.validation import validate_data


@njit(cache=True, nogil=True)
def coordinate_descent(data, indices, indptr, y, l1, l2, tol, max_iter, intercept):
    n, p = len(y), len(indptr) - 1
    target_mean = np.mean(y) if intercept else 0.0
    centered = y - target_mean
    residual = centered.copy()
    residual_sum = np.sum(residual)
    mean = np.zeros(p)
    norm = np.zeros(p)
    coefficient = np.zeros(p)
    for j in range(p):
        column_sum, squared = 0.0, 0.0
        for k in range(indptr[j], indptr[j + 1]):
            column_sum += data[k]
            squared += data[k] * data[k]
        mean[j] = column_sum / n if intercept else 0.0
        norm[j] = max(0.0, squared - n * mean[j] * mean[j])
    tolerance = tol * np.dot(centered, centered)
    gap = np.inf
    for iteration in range(max_iter):
        for j in range(p):
            if norm[j] == 0.0:
                continue
            correlation = coefficient[j] * norm[j] - mean[j] * residual_sum
            for k in range(indptr[j], indptr[j + 1]):
                correlation += data[k] * residual[indices[k]]
            value = (
                np.sign(correlation) * max(0.0, abs(correlation) - l1) / (norm[j] + l2)
            )
            delta = value - coefficient[j]
            if delta != 0.0:
                coefficient[j] = value
                change_sum = 0.0
                for k in range(indptr[j], indptr[j + 1]):
                    change = delta * data[k]
                    residual[indices[k]] -= change
                    change_sum += change
                residual_sum -= change_sum
        if iteration % 5 == 0 or iteration + 1 == max_iter:
            shift = np.dot(mean, coefficient)
            actual = residual + shift
            actual_sum = np.sum(actual)
            dual_norm = 0.0
            for j in range(p):
                gradient = -mean[j] * actual_sum - l2 * coefficient[j]
                for k in range(indptr[j], indptr[j + 1]):
                    gradient += data[k] * actual[indices[k]]
                dual_norm = max(dual_norm, abs(gradient))
            scale = min(1.0, l1 / dual_norm) if dual_norm > 0.0 else 1.0
            gap = max(
                0.0,
                0.5 * (1.0 + scale * scale) * np.dot(actual, actual)
                + l1 * np.sum(np.abs(coefficient))
                - scale * np.dot(actual, centered)
                + 0.5 * l2 * (1.0 + scale * scale) * np.dot(coefficient, coefficient),
            )
            if gap <= tolerance:
                break
    bias = target_mean - np.dot(mean, coefficient)
    return coefficient, bias, iteration + 1, gap / n, gap <= tolerance


def fit_penalty(model, x, y):
    model._validate_params()
    if (
        model.alpha <= 0
        or not 0 < model.l1_ratio <= 1
        or model.tol <= 0
        or model.selection != "cyclic"
        or model.warm_start
        or model.positive
        or model.precompute is not False
    ):
        raise ValueError(
            "positive penalties/tolerance, cyclic fresh fits and no Gram precomputation required"
        )
    x, y = validate_data(
        model, x, y, accept_sparse="csc", dtype=np.float64, copy=True, y_numeric=True
    )
    x = sparse.csc_matrix(x)
    x.sum_duplicates()
    x.sort_indices()
    result = coordinate_descent(
        x.data,
        x.indices,
        x.indptr,
        y,
        model.alpha * model.l1_ratio * len(y),
        model.alpha * (1 - model.l1_ratio) * len(y),
        model.tol,
        model.max_iter,
        model.fit_intercept,
    )
    model.coef_, model.intercept_, model.n_iter_, model.dual_gap_, converged = result
    model.solver_ = "implicit_centering_coordinate_descent"
    if not converged:
        warnings.warn(
            f"Coordinate descent did not converge after {model.n_iter_} iterations; dual gap {model.dual_gap_:.6g}",
            ConvergenceWarning,
            stacklevel=2,
        )
    return model


class SparseLasso(Lasso):
    """Lasso objective with sparse, implicit-intercept coordinate updates."""

    def fit(self, x, y):
        return fit_penalty(self, x, y)


class SparseElasticNet(ElasticNet):
    """Elastic Net objective with sparse, implicit-intercept coordinate updates."""

    def fit(self, x, y):
        return fit_penalty(self, x, y)
