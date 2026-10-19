"""Train-only quantiles keep zero and identical prices in coherent classes."""

import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from benchmarks.mercari_price_buckets import (
    PriceBuckets,
    aligned_proba,
    classification_metrics,
    fit_classifier,
)


def test_positive_equal_frequency_excludes_zeros_and_keeps_boundary_ties():
    y = np.log1p(np.r_[np.zeros(40), np.arange(1, 91)])
    bins = PriceBuckets().fit(y)
    assert bins.n_classes_ == 10
    np.testing.assert_allclose(np.expm1(bins.log_cuts_), np.arange(10, 81, 10))
    np.testing.assert_array_equal(np.bincount(bins.transform(y)), [40] + [10] * 9)
    other = PriceBuckets().fit(np.log1p(np.r_[np.zeros(999), np.arange(1, 91)]))
    np.testing.assert_array_equal(bins.log_cuts_, other.log_cuts_)
    values = np.array(
        [0.0, bins.log_cuts_[0], np.nextafter(bins.log_cuts_[0], np.inf), 100.0]
    )
    np.testing.assert_array_equal(bins.transform(values), [0, 1, 2, 9])
    assert bins.describe()[0]["label"] == "zero"
    assert bins.describe()[-1]["upper_dollars"] is None
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(bins)).transform(y), bins.transform(y)
    )


def test_tied_constant_and_zero_only_targets_have_nonempty_positive_schema():
    for values in ([0, 10, 10, 10], [0, 0, 0], [10, 10]):
        bins = PriceBuckets().fit(np.log1p(values))
        assert bins.n_classes_ == 2
        np.testing.assert_array_equal(
            bins.transform(np.log1p([0, 0.01, 1000])), [0, 1, 1]
        )
    y = np.log1p([0] + [5] * 80 + [10] * 10 + [20] * 10)
    bins = PriceBuckets().fit(y)
    for price in (5, 10, 20):
        assert len(np.unique(bins.transform(np.full(4, np.log1p(price))))) == 1
    assert bins.n_classes_ < 10
    with pytest.raises(NotFittedError):
        PriceBuckets().transform([0])
    for invalid in ([], [np.nan], [-1], [[0, 1]]):
        with pytest.raises(ValueError):
            PriceBuckets().fit(invalid)
    for invalid in (0, 1.5):
        with pytest.raises(ValueError):
            PriceBuckets(invalid).fit([0, 1])


def test_metrics_match_sklearn_and_expose_missing_predictions_and_support():
    y = np.array([0, 1, 1, 2, 2, 3, 3])
    pred = np.array([1, 1, 2, 2, 1, 3, 3])
    m = classification_metrics(y, pred, 5)
    assert m["accuracy"] == accuracy_score(y, pred)
    assert m["balanced_accuracy"] == balanced_accuracy_score(y, pred)
    assert m["macro_f1"] == f1_score(
        y, pred, labels=np.arange(5), average="macro", zero_division=0
    )
    np.testing.assert_array_equal(
        m["confusion_matrix"], confusion_matrix(y, pred, labels=np.arange(5))
    )
    assert m["per_class"][0]["recall"] == 0
    assert m["per_class"][0]["precision"] is None
    assert m["per_class"][4]["recall"] is None
    for bad_y, bad_p in (
        ([], []),
        ([0], [10]),
        ([0.5], [0]),
        ([0], [np.nan]),
        ([0], [0, 1]),
    ):
        with pytest.raises(ValueError):
            classification_metrics(bad_y, bad_p, 5)


def test_actual_c5_and_lightgbm_fit_class_mapping_weights_and_pickle():
    for labels in ([0, 2, 4], [0, 4]):
        y = np.tile(labels, 90 // len(labels))
        x = y.astype(np.float32)[:, None]
        absent = np.setdiff1d(np.arange(6), labels)
        for engine in ("c5", "lightgbm"):
            for balanced in (False, True):
                model = fit_classifier(engine, balanced, x, y, x, y, rounds=4)
                p = aligned_proba(model, x, 6)
                assert p.shape == (90, 6) and not p[:, absent].any()
                np.testing.assert_allclose(p.sum(1), 1)
                assert classification_metrics(y, p.argmax(1), 6)["accuracy"] > 0.9
                restored = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(p, aligned_proba(restored, x, 6))
                for value in (model, restored):
                    if hasattr(value, "close"):
                        value.close()
    model = fit_classifier("lightgbm", False, x, np.zeros(len(x), int), x, y, rounds=2)
    np.testing.assert_array_equal(aligned_proba(model, x[:1], 6), [[1, 0, 0, 0, 0, 0]])
    with pytest.raises(ValueError):
        fit_classifier("invalid", False, x, y, x, y, rounds=1)
