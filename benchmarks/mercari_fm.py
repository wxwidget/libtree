"""Sparse factorization expert with synchronous model averaging on CPUs.

FM arithmetic: Rendle (2010), https://doi.org/10.1109/ICDM.2010.127.
Field-aware factors: Juan et al. (2016), doi:10.1145/2959100.2959134.
Local SGD: Stich (2019), https://openreview.net/forum?id=S1g2JnRcFX.
Workers own independent arrays; synchronizing between chunks avoids shared
parameter writes and the data races of asynchronous Hogwild implementations.
Finite error_clip uses a Huber gradient (Huber, 1964,
doi:10.1214/aoms/1177703732); infinity retains squared error.
This benchmark expert is optional; the C++ LibTree core has no Numba dependency.
"""
import numpy as np
from numba import njit, prange, set_num_threads
from scipy import sparse
from sklearn.utils.validation import check_array


@njit(cache=True)
def _row(data, indices, begin, end, factors, linear, fields, field_count):
    rank = factors.shape[1] // field_count
    sums = np.zeros((field_count, field_count, rank), dtype=np.float32)
    square = 0.
    prediction = linear[-1]
    for position in range(begin, end):
        column, value = indices[position], data[position]
        source = fields[column]
        prediction += linear[column] * value
        for destination in range(field_count):
            for k in range(rank):
                v = factors[column, destination * rank + k] * value
                sums[source, destination, k] += v
                if source == destination:
                    square += v * v
    interaction = -.5 * square
    for source in range(field_count):
        for destination in range(source, field_count):
            weight = .5 if source == destination else 1.
            for k in range(rank):
                interaction += weight * sums[source, destination, k] * sums[destination, source, k]
    return prediction + interaction, sums


@njit(cache=True, parallel=True)
def _predict(data, indices, indptr, factors, linear, fields, field_count):
    result = np.empty(len(indptr) - 1, dtype=np.float32)
    for row in prange(len(result)):
        result[row] = _row(data, indices, indptr[row], indptr[row + 1], factors, linear, fields, field_count)[0]
    return result


@njit(cache=True, parallel=True)
def _chunk(data, indices, indptr, y, order, begin, end, factors, linear,
           accumulated, linear_accumulated, rate, l2, fields, field_count, error_clip):
    workers = len(factors)
    for worker in prange(workers):
        v, w = factors[worker], linear[worker]
        a, b = accumulated[worker], linear_accumulated[worker]
        for position in range(begin + worker, end, workers):
            row = order[position]
            prediction, sums = _row(data, indices, indptr[row], indptr[row + 1], v, w, fields, field_count)
            error = prediction - y[row]
            error = min(max(error, -error_clip), error_clip)
            b[-1] += error * error
            w[-1] -= rate * error / np.sqrt(b[-1])
            for p in range(indptr[row], indptr[row + 1]):
                column, value = indices[p], data[p]
                gradient = error * value + l2 * w[column]
                b[column] += gradient * gradient
                w[column] -= rate * gradient / np.sqrt(b[column])
                source = fields[column]
                rank = v.shape[1] // field_count
                for destination in range(field_count):
                    for k in range(rank):
                        component = destination * rank + k
                        other = sums[destination, source, k]
                        if source == destination:
                            other -= v[column, component] * value
                        gradient = error * value * other + l2 * v[column, component]
                        a[column, component] += gradient * gradient
                        v[column, component] -= rate * gradient / np.sqrt(a[column, component])


class FactorizationMachine:
    def __init__(self, features, rank=16, workers=2, learning_rate=.03, l2=.0001,
                 chunk_size=131072, seed=42, fields=None, error_clip=np.inf):
        if min(features, rank, workers, chunk_size) < 1 or learning_rate <= 0 or l2 < 0 or error_clip <= 0:
            raise ValueError('positive dimensions, workers, chunk size and learning rate required')
        self.features, self.rank = features, rank
        self.workers, self.learning_rate, self.l2 = workers, learning_rate, l2
        self.chunk_size = chunk_size
        self.error_clip = error_clip
        field_array = np.zeros(features, dtype=np.int32) if fields is None else np.asarray(fields)
        if field_array.shape != (features,) or (field_array < 0).any() or not np.array_equal(field_array, field_array.astype(np.int32)):
            raise ValueError('one nonnegative integer field per feature required')
        self.fields_ = field_array.astype(np.int32)
        self.field_count_ = int(self.fields_.max()) + 1
        set_num_threads(workers)
        rng = np.random.default_rng(seed)
        initial = rng.normal(0, .03, (features, rank * self.field_count_)).astype(np.float32)
        self.factors_ = np.repeat(initial[None, :, :], workers, axis=0)
        self.linear_ = np.zeros((workers, features + 1), dtype=np.float32)
        self.accumulated_ = np.full_like(self.factors_, .01)
        self.linear_accumulated_ = np.full_like(self.linear_, .01)

    def _matrix(self, x):
        x = sparse.csr_matrix(check_array(x, accept_sparse='csr', dtype=np.float32))
        if x.shape[1] != self.features:
            raise ValueError('FM feature count mismatch')
        x.sum_duplicates()
        return x

    def fit_epoch(self, x, y, seed=42):
        x = self._matrix(x)
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (x.shape[0],) or not np.isfinite(y).all():
            raise ValueError('finite labels must match FM training rows')
        order = np.random.default_rng(seed).permutation(x.shape[0])
        for begin in range(0, len(y), self.chunk_size):
            _chunk(x.data, x.indices, x.indptr, y, order, begin, min(begin + self.chunk_size, len(y)),
                   self.factors_, self.linear_, self.accumulated_, self.linear_accumulated_,
                   self.learning_rate, self.l2, self.fields_, self.field_count_, self.error_clip)
            for weights in (self.factors_, self.linear_, self.accumulated_, self.linear_accumulated_):
                weights[:] = weights.mean(axis=0)
        return self

    def predict(self, x):
        x = self._matrix(x)
        return _predict(x.data, x.indices, x.indptr, self.factors_[0], self.linear_[0], self.fields_, self.field_count_)

    def freeze(self):
        return FrozenFM(self.factors_[0].copy(), self.linear_[0].copy(), self.fields_.copy(), self.field_count_)


class FrozenFM:
    """Prediction state without training accumulators or worker copies."""

    def __init__(self, factors, linear, fields, field_count):
        self.factors, self.linear, self.fields, self.field_count = factors, linear, fields, field_count

    def predict(self, x):
        x = sparse.csr_matrix(check_array(x, accept_sparse='csr', dtype=np.float32))
        if x.shape[1] != len(self.factors):
            raise ValueError('FM feature count mismatch')
        x.sum_duplicates()
        return _predict(x.data, x.indices, x.indptr, self.factors, self.linear, self.fields, self.field_count)
