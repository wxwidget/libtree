"""Jacobi-preconditioned, matrix-free Ridge for large sparse text features.

Conjugate gradients: Hestenes & Stiefel (1952), J. Research NBS 49(6),
doi:10.6028/jres.049.044. Implicit centering avoids densifying the document
matrix; diagonal preconditioning preserves the original Ridge objective.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SparseRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=3., tol=1e-5, max_iter=300):
        self.alpha, self.tol, self.max_iter = alpha, tol, max_iter

    def fit(self, x, y):
        if self.alpha <= 0 or self.tol <= 0 or self.max_iter < 1:
            raise ValueError('alpha, tol and max_iter must be positive')
        x, y = check_X_y(x, y, accept_sparse='csr', dtype=np.float64, y_numeric=True)
        x = sparse.csr_matrix(x)
        mean = np.asarray(x.mean(axis=0)).ravel()
        target_mean = float(np.mean(y))
        n, self.n_features_in_ = x.shape
        diagonal = np.asarray(x.multiply(x).sum(axis=0)).ravel() - n * mean ** 2 + self.alpha

        def multiply(value):
            return np.asarray(x.T @ (x @ value)).ravel() - n * mean * (mean @ value) + self.alpha * value

        system = LinearOperator((x.shape[1], x.shape[1]), matvec=multiply, dtype=np.float64)
        preconditioner = LinearOperator(system.shape, matvec=lambda v: v / np.maximum(diagonal, self.alpha), dtype=np.float64)
        self.n_iter_ = 0

        def iteration(value):
            self.n_iter_ += 1

        self.coef_, info = cg(system, np.asarray(x.T @ (y - target_mean)).ravel(),
                              M=preconditioner, rtol=self.tol, atol=0., maxiter=self.max_iter, callback=iteration)
        if info:
            raise RuntimeError(f'sparse Ridge failed to converge after {self.n_iter_} iterations (status {info})')
        self.intercept_ = target_mean - float(mean @ self.coef_)
        return self

    def predict(self, x):
        check_is_fitted(self, 'intercept_')
        x = check_array(x, accept_sparse='csr', dtype=np.float64)
        return np.asarray(x @ self.coef_).ravel() + self.intercept_
