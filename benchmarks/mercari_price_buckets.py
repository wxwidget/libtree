"""Zero plus train-only positive quantiles, and discrete price classification.

Quantile definitions: Hyndman & Fan (1996), doi:10.1080/00031305.1996.10473566.
Observed inverse-CDF quantiles retain ties. C5 SAMME: Zhu et al. (2009),
doi:10.4310/SII.2009.v2.n3.a8.
"""

import lightgbm as lgb
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

from libtree import C5Classifier


def log_targets(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("finite nonempty nonnegative log prices required")
    return y


class PriceBuckets(BaseEstimator):
    """Inputs are log1p prices; quantile ranks equal raw-price ranks.

    Zero has class 0. Positive intervals are right closed, with an open-ended
    last interval. Equal prices stay together; duplicate/maximal cuts collapse.
    Zero-only fitting retains one catch-all positive class for unseen targets.
    """

    def __init__(self, positive_bins=9):
        self.positive_bins = positive_bins

    def fit(self, log_y):
        y = log_targets(log_y)
        if (
            not isinstance(self.positive_bins, (int, np.integer))
            or self.positive_bins < 1
        ):
            raise ValueError("positive integer bin count required")
        positive = y[y > 0]
        cuts = np.array([], dtype=float)
        if len(positive):
            cuts = np.quantile(
                positive,
                np.arange(1, self.positive_bins) / self.positive_bins,
                method="inverted_cdf",
            )
            cuts = np.unique(cuts[cuts < positive.max()])
        self.log_cuts_ = cuts
        self.n_classes_ = len(cuts) + 2
        self.counts_ = np.bincount(self.transform(y), minlength=self.n_classes_)
        return self

    def transform(self, log_y):
        check_is_fitted(self, "log_cuts_")
        y = log_targets(log_y)
        return np.where(
            y == 0, 0, 1 + np.searchsorted(self.log_cuts_, y, side="left")
        ).astype(np.int32)

    def describe(self):
        check_is_fitted(self, "log_cuts_")
        edges = [0.0, *np.expm1(self.log_cuts_).tolist(), None]
        return [
            dict(class_id=0, label="zero", lower_dollars=0.0, upper_dollars=0.0)
        ] + [
            dict(
                class_id=i + 1,
                label="positive",
                lower_dollars=edges[i],
                upper_dollars=edges[i + 1],
            )
            for i in range(len(edges) - 1)
        ]


def classification_metrics(y, prediction, n_classes):
    y, p = np.asarray(y), np.asarray(prediction)
    if (
        y.ndim != 1
        or not len(y)
        or p.shape != y.shape
        or n_classes < 2
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or (y != np.floor(y)).any()
        or (p != np.floor(p)).any()
        or (y < 0).any()
        or (p < 0).any()
        or (y >= n_classes).any()
        or (p >= n_classes).any()
    ):
        raise ValueError(
            "matching nonempty integer class labels within schema required"
        )
    cm = np.bincount(
        n_classes * y.astype(int) + p.astype(int), minlength=n_classes**2
    ).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support > 0)
    f1 = np.divide(
        2 * tp,
        support + predicted,
        out=np.zeros(n_classes),
        where=(support + predicted) > 0,
    )
    return dict(
        rows=len(y),
        accuracy=float(tp.sum() / len(y)),
        balanced_accuracy=float(recall[support > 0].mean()),
        macro_f1=float(f1.mean()),
        confusion_matrix=cm.tolist(),
        per_class=[
            dict(
                class_id=i,
                support=int(support[i]),
                predicted=int(predicted[i]),
                correct=int(tp[i]),
                precision=float(precision[i]) if predicted[i] else None,
                recall=float(recall[i]) if support[i] else None,
                f1=float(f1[i]),
            )
            for i in range(n_classes)
        ],
    )


def fit_classifier(engine, balanced, x, y, vx, vy, rounds=150):
    if engine not in ("c5", "lightgbm"):
        raise ValueError("known classification engine required")
    if len(np.unique(y)) == 1:
        return DummyClassifier(strategy="prior").fit(x, y)
    leaf = min(80, max(2, len(y) // 20))
    if engine == "c5":
        return C5Classifier(
            n_estimators=min(20, rounds),
            max_depth=6,
            max_bins=64,
            min_samples_leaf=leaf,
            learning_rate=0.5,
            pruning_alpha=0.0001,
            criterion="gini",
            class_weight="balanced" if balanced else None,
            n_jobs=4,
        ).fit(x, y)
    model = lgb.LGBMClassifier(
        n_estimators=rounds,
        objective="multiclass",
        learning_rate=0.05,
        num_leaves=31,
        max_bin=127,
        min_child_samples=leaf,
        reg_lambda=10.0,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        class_weight="balanced" if balanced else None,
    )
    known = np.isin(vy, np.unique(y))
    options = {}
    if known.any():
        options = dict(
            eval_set=[(vx[known], vy[known])],
            eval_metric="multi_error",
            callbacks=[lgb.early_stopping(25, first_metric_only=True, verbose=False)],
        )
        # Early stopping must monitor classification error, not default log loss.
        model.set_params(metric="multi_error")
    return model.fit(x, y, **options)


def aligned_proba(model, x, n_classes):
    p = np.zeros((len(x), n_classes), dtype=float)
    p[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return p
