"""Shared, tie-safe binary risk metrics."""
import numpy as np


def ks_statistic(y, score, directional=False):
    """Two-sample KS over exact score groups; higher score means bad (y=1).

    Default is max |F_bad-F_good|. Directional KS is max(TPR-FPR), which
    avoids rewarding inverted risk scores. See Massey (1951),
    doi:10.1080/01621459.1951.10500769; ties cannot be split into fake thresholds.
    """
    y, score = np.asarray(y), np.asarray(score, dtype='f8')
    if (y.ndim != 1 or score.shape != y.shape or not len(y) or
            not np.isfinite(score).all() or set(np.unique(y)) != {0, 1}):
        raise ValueError('KS requires finite scores and both binary classes')
    order = np.argsort(-score, kind='stable')
    labels, values = y[order], score[order]
    ends = np.r_[values[1:] != values[:-1], True]
    difference = np.cumsum(labels)[ends] / sum(labels) - np.cumsum(1 - labels)[ends] / sum(1 - labels)
    return float(max(0, np.max(difference)) if directional else np.max(np.abs(difference)))

