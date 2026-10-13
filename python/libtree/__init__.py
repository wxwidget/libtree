"""Deterministic histogram GBDT backed by a C++17 library."""

import ctypes as ct
import json
import os
from pathlib import Path
import weakref

import numpy as np

__version__ = "0.2.0"
_LIB = None
_FLOAT = ct.POINTER(ct.c_float)


def _library():
    global _LIB
    if _LIB is None:
        path = os.environ.get("LIBTREE_LIBRARY")
        if path is None:
            path = Path(__file__).with_name("_native.so")
            if not path.exists():
                path = Path(__file__).resolve().parents[2] / "build/libtree.so"
        lib = ct.CDLL(str(path))
        lib.LtCreate.argtypes = [ct.c_int] * 4 + [ct.c_double] * 3 + [ct.c_int]
        lib.LtCreate.restype = ct.c_void_p
        lib.LtSetNumThreads.argtypes = [ct.c_void_p, ct.c_int]
        lib.LtSetNumThreads.restype = ct.c_int
        lib.LtFree.argtypes = [ct.c_void_p]
        lib.LtFree.restype = None
        lib.LtFit.argtypes = [ct.c_void_p, _FLOAT, ct.c_size_t, ct.c_size_t, _FLOAT]
        lib.LtFit.restype = ct.c_int
        lib.LtPredict.argtypes = lib.LtFit.argtypes
        lib.LtPredict.restype = ct.c_int
        lib.LtLastError.argtypes = []
        lib.LtLastError.restype = ct.c_char_p
        lib.LtLossCount.argtypes = [ct.c_void_p]
        lib.LtLossCount.restype = ct.c_size_t
        lib.LtLoss.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_size_t]
        lib.LtLoss.restype = ct.c_int
        inspection_output = [ct.POINTER(ct.c_char_p), ct.c_size_t,
                             ct.c_char_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
        lib.LtDumpModel.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, *inspection_output]
        lib.LtDumpModel.restype = ct.c_int
        lib.LtExplainPrediction.argtypes = [ct.c_void_p, _FLOAT, ct.c_size_t,
                                           ct.c_int, *inspection_output]
        lib.LtExplainPrediction.restype = ct.c_int
        lib.LtC5Create.argtypes = [ct.c_int] * 4 + [ct.c_double] * 3 + [ct.c_int] * 2
        lib.LtC5Create.restype = ct.c_void_p
        lib.LtC5Free.argtypes = [ct.c_void_p]
        lib.LtC5Free.restype = None
        lib.LtC5Fit.argtypes = [ct.c_void_p, _FLOAT, ct.c_size_t, ct.c_size_t,
                                ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)]
        lib.LtC5Fit.restype = ct.c_int
        lib.LtC5PredictProba.argtypes = [ct.c_void_p, _FLOAT, ct.c_size_t, ct.c_size_t,
                                         ct.POINTER(ct.c_double)]
        lib.LtC5PredictProba.restype = ct.c_int
        for name in ('LtC5ClassCount', 'LtC5ErrorCount'):
            function = getattr(lib, name)
            function.argtypes = [ct.c_void_p]
            function.restype = ct.c_size_t
        lib.LtC5Errors.argtypes = [ct.c_void_p, ct.POINTER(ct.c_double), ct.c_size_t]
        lib.LtC5Errors.restype = ct.c_int
        lib.LtC5DumpRules.argtypes = [ct.c_void_p, ct.c_int, *inspection_output]
        lib.LtC5DumpRules.restype = ct.c_int
        lib.LtC5Explain.argtypes = lib.LtExplainPrediction.argtypes
        lib.LtC5Explain.restype = ct.c_int
        _LIB = lib
    return _LIB


def _matrix(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("X must have shape (rows, features), with features > 0")
    return x


def _pointer(array):
    return array.ctypes.data_as(_FLOAT)


class _OwnedModel:
    """Shared native ownership and caller-owned inspection buffers."""

    def _inspection(self, operation, arguments, format, feature_names):
        if self._handle is None:
            raise RuntimeError("model is not fitted or is closed")
        if format not in ('text', 'json'):
            raise ValueError("format must be text or json")
        names = None
        count = 0
        if feature_names is not None:
            if isinstance(feature_names, str):
                raise ValueError("feature_names must be a sequence of strings")
            labels = tuple(feature_names)
            if any(not isinstance(name, str) or '\0' in name for name in labels):
                raise ValueError("feature names must be strings without NUL")
            count = len(labels)
            names = (ct.c_char_p * count)(*(name.encode('utf-8') for name in labels))
        lib = _library()
        function = getattr(lib, operation)
        required = ct.c_size_t()
        arguments = arguments(0 if format == 'text' else 1)
        if function(self._handle, *arguments, names, count, None, 0, ct.byref(required)):
            raise ValueError(lib.LtLastError().decode())
        output = ct.create_string_buffer(required.value)
        if function(self._handle, *arguments, names, count, output, len(output), ct.byref(required)):
            raise RuntimeError(lib.LtLastError().decode())
        return output.value.decode('utf-8')

    def close(self):
        if self._finalizer is not None:
            self._finalizer()
        self._handle = None
        self._finalizer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GBDTRegressor(_OwnedModel):
    """Squared-error GBDT. fit replaces the model; predict returns float32.

    NaN values are supported. Inputs are not mutated. A model must not be fitted
    or closed concurrently with prediction. Distinct models are independent.
    """

    _binary = False

    def __init__(self, n_estimators=100, max_depth=4, max_bins=64,
                 min_samples_leaf=5, learning_rate=0.1, reg_lambda=1.0,
                 min_gain=0.0, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.min_gain = min_gain
        self.n_jobs = n_jobs
        self._handle = None
        self._finalizer = None

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in (
            "n_estimators", "max_depth", "max_bins", "min_samples_leaf",
            "learning_rate", "reg_lambda", "min_gain", "n_jobs")}

    def set_params(self, **params):
        for name in params:
            if name not in self.get_params():
                raise ValueError(f"unknown parameter: {name}")
        if 'n_jobs' in params and self._handle is not None:
            threads = params['n_jobs']
            if not isinstance(threads, (int, np.integer)) or not 1 <= threads <= 256:
                raise ValueError('n_jobs must be an integer in 1..256')
            lib = _library()
            if lib.LtSetNumThreads(self._handle, threads):
                raise ValueError(lib.LtLastError().decode())
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, x, y):
        x = _matrix(x)
        y = np.ascontiguousarray(y, dtype=np.float32)
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("y must be a vector matching X rows")
        lib = _library()
        for name in ("n_estimators", "max_depth", "max_bins", "min_samples_leaf", "n_jobs"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or not -(2**31) <= value < 2**31:
                raise ValueError(f"{name} must be a 32-bit integer")
        handle = lib.LtCreate(self.n_estimators, self.max_depth, self.max_bins,
                              self.min_samples_leaf, self.learning_rate,
                              self.reg_lambda, self.min_gain, self._binary)
        if not handle:
            raise ValueError(lib.LtLastError().decode())
        if lib.LtSetNumThreads(handle, self.n_jobs):
            error = lib.LtLastError().decode()
            lib.LtFree(handle)
            raise ValueError(error)
        if lib.LtFit(handle, _pointer(x), *x.shape, _pointer(y)):
            error = lib.LtLastError().decode()
            lib.LtFree(handle)
            raise ValueError(error)
        self.close()
        self._handle = handle
        self._finalizer = weakref.finalize(self, lib.LtFree, handle)
        self.n_features_in_ = x.shape[1]
        if self._binary:
            self.classes_ = np.array([0, 1])
        return self

    def _predict_values(self, x):
        if self._handle is None:
            raise RuntimeError("model is not fitted or is closed")
        x = _matrix(x)
        out = np.empty(len(x), dtype=np.float32)
        lib = _library()
        if lib.LtPredict(self._handle, _pointer(x), *x.shape, _pointer(out)):
            raise ValueError(lib.LtLastError().decode())
        return out

    def predict(self, x):
        return self._predict_values(x)

    def dump_model(self, format='text', tree_index=None, feature_names=None):
        """Return a text/JSON string; indices are zero-based, None exports all trees.

        Labels are supplied in input-column order; defaults are f0, f1, ...
        Leaf contribution is learning_rate * leaf_value on the margin scale.
        This is an inspection format, not a loadable model file.
        """
        index = -1
        if tree_index is not None:
            if (not isinstance(tree_index, (int, np.integer))
                    or not 0 <= tree_index < 2**31):
                raise ValueError("tree_index must be a nonnegative 32-bit integer")
            index = int(tree_index)
        return self._inspection('LtDumpModel', lambda format: (format, index),
                                format, feature_names)

    def explain(self, sample, feature_names=None, format='json'):
        """Explain one row: a dict for json, or a human-readable string for text.

        Input shape is (features,) or (1, features). Each path records the
        split comparison or missing-value route, plus the reached leaf.
        Binary margin is log-odds; prediction is P(class=1), not a class label.
        """
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        if sample.ndim != 2 or sample.shape[0] != 1:
            raise ValueError("explanation requires exactly one sample")
        sample = _matrix(sample)
        text = self._inspection('LtExplainPrediction',
                                lambda format: (_pointer(sample), sample.shape[1], format),
                                format, feature_names)
        return json.loads(text) if format == 'json' else text

    @property
    def training_loss_(self):
        if self._handle is None:
            raise RuntimeError("model is not fitted or is closed")
        lib = _library()
        out = np.empty(lib.LtLossCount(self._handle), dtype=np.float64)
        if lib.LtLoss(self._handle, out.ctypes.data_as(ct.POINTER(ct.c_double)), len(out)):
            raise RuntimeError(lib.LtLastError().decode())
        return out

    def score(self, x, y):
        y = np.asarray(y)
        pred = self.predict(x)
        if y.ndim != 1 or y.shape != pred.shape or len(y) == 0:
            raise ValueError("invalid score labels")
        residual = np.sum((y - pred) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total) if total else float(residual == 0)



class GBDTClassifier(GBDTRegressor):
    """Binary logistic GBDT; labels must be exactly 0 or 1."""

    _binary = True

    def predict_proba(self, x):
        positive = self._predict_values(x)
        return np.column_stack((1 - positive, positive))

    def predict(self, x):
        return (self._predict_values(x) >= 0.5).astype(np.int64)

    def score(self, x, y):
        y = np.asarray(y)
        pred = self.predict(x)
        if y.ndim != 1 or y.shape != pred.shape or len(y) == 0:
            raise ValueError("invalid score labels")
        return float(np.mean(pred == y))


from .c5 import C5Classifier  # noqa: E402
