"""Readable numeric classification rules with multiclass discrete boosting."""
import ctypes as ct
import json
import weakref

import numpy as np

from . import _OwnedModel, _library, _matrix, _pointer


class C5Classifier(_OwnedModel):
    """C5-inspired SAMME classifier, not a compatible C5.0 implementation.

    Arbitrary string/integer labels are mapped to sorted classes_. NaN features
    use learned routing. Probabilities are uncalibrated softmax of normalized
    votes. Fit/close must not run concurrently with prediction/inspection.
    """

    def __init__(self, n_estimators=10, max_depth=6, max_bins=64,
                 min_samples_leaf=5, learning_rate=1., pruning_alpha=.001,
                 min_gain=1e-9, criterion='gain_ratio', n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.pruning_alpha = pruning_alpha
        self.min_gain = min_gain
        self.criterion = criterion
        self.n_jobs = n_jobs
        self._handle = None
        self._finalizer = None

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in ('n_estimators', 'max_depth', 'max_bins',
                'min_samples_leaf', 'learning_rate', 'pruning_alpha', 'min_gain',
                'criterion', 'n_jobs')}

    def set_params(self, **params):
        if params.keys() - self.get_params().keys():
            raise ValueError('unknown C5 parameter')
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, x, y, sample_weight=None):
        x = _matrix(x)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x) or not len(y):
            raise ValueError('y must match nonempty X rows')
        if y.dtype.kind not in 'biufUS' or (y.dtype.kind == 'f' and not np.isfinite(y).all()):
            raise ValueError('labels must be finite numbers or strings')
        classes, indices = np.unique(y, return_inverse=True)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        weights = None
        if sample_weight is not None:
            weights = np.ascontiguousarray(sample_weight, dtype=np.float64)
            if weights.ndim != 1 or len(weights) != len(x):
                raise ValueError('sample_weight must match X rows')
        for k in ('n_estimators', 'max_depth', 'max_bins', 'min_samples_leaf', 'n_jobs'):
            if not isinstance(getattr(self, k), (int, np.integer)) or not -(2**31) <= getattr(self, k) < 2**31:
                raise ValueError(f'{k} must be a 32-bit integer')
        if self.criterion not in ('gain_ratio', 'entropy', 'gini'):
            raise ValueError('criterion must be gain_ratio, entropy or gini')
        lib = _library()
        handle = lib.LtC5Create(self.n_estimators, self.max_depth, self.max_bins,
                              self.min_samples_leaf, self.learning_rate,
                              self.pruning_alpha, self.min_gain,
                              ('gain_ratio', 'entropy', 'gini').index(self.criterion), self.n_jobs)
        if not handle:
            raise ValueError(lib.LtLastError().decode())
        if lib.LtC5Fit(handle, _pointer(x), *x.shape, indices.ctypes.data_as(ct.POINTER(ct.c_int)),
                      None if weights is None else weights.ctypes.data_as(ct.POINTER(ct.c_double))):
            error = lib.LtLastError().decode()
            lib.LtC5Free(handle)
            raise ValueError(error)
        self.close()
        self._handle = handle
        self._finalizer = weakref.finalize(self, lib.LtC5Free, handle)
        self.classes_ = classes
        self.n_features_in_ = x.shape[1]
        return self

    def predict_proba(self, x):
        if self._handle is None:
            raise RuntimeError('model is not fitted or is closed')
        x = _matrix(x)
        out = np.empty((len(x), len(self.classes_)), dtype=np.float64)
        lib = _library()
        if lib.LtC5PredictProba(self._handle, _pointer(x), *x.shape,
                               out.ctypes.data_as(ct.POINTER(ct.c_double))):
            raise ValueError(lib.LtLastError().decode())
        return out

    def predict(self, x):
        indices = self.predict_proba(x).argmax(axis=1)
        return self.classes_[indices]

    def score(self, x, y):
        prediction, y = self.predict(x), np.asarray(y)
        if y.ndim != 1 or y.shape != prediction.shape or not len(y):
            raise ValueError('invalid score labels')
        return float(np.mean(prediction == y))

    @property
    def estimator_errors_(self):
        if self._handle is None:
            raise RuntimeError('model is not fitted or is closed')
        lib = _library()
        out = np.empty(lib.LtC5ErrorCount(self._handle), dtype=np.float64)
        if lib.LtC5Errors(self._handle, out.ctypes.data_as(ct.POINTER(ct.c_double)), len(out)):
            raise RuntimeError(lib.LtLastError().decode())
        return out

    def dump_rules(self, format='text', feature_names=None):
        """Return text or a JSON string. Class indices refer to classes_."""
        text = self._inspection('LtC5DumpRules', lambda f: (f,), format, feature_names)
        return self._labelled(text, format)

    def _labelled(self, text, format):
        if format == 'text':
            return 'Classes (indices from 0): ' + json.dumps(self.classes_.tolist(), ensure_ascii=False) + '\n' + text
        doc = json.loads(text)
        doc['class_labels'] = self.classes_.tolist()
        if 'prediction' in doc:
            doc['predicted_label'] = self.classes_[doc['prediction']].item()
        return json.dumps(doc, ensure_ascii=False, allow_nan=False)

    def explain(self, sample, feature_names=None, format='json'):
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        if sample.ndim != 2 or sample.shape[0] != 1:
            raise ValueError('explanation requires exactly one sample')
        sample = _matrix(sample)
        text = self._inspection('LtC5Explain', lambda f: (_pointer(sample), sample.shape[1], f),
                                format, feature_names)
        text = self._labelled(text, format)
        return json.loads(text) if format == 'json' else text
