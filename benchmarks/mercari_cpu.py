"""CPU lexical views and exact squared-RMSLE stacking, without pretrained data.

Feature/view motivation: Jankiewicz & Lopuhin's Mercari first-place solution,
https://github.com/pjankiewicz/mercari-solution (2018), presentation/5_modeling.tex.
Stacking: Wolpert (1992), doi:10.1016/S0893-6080(05)80023-1.
The code is an independent implementation, not a reproduced first-place score.
"""

import re

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer

from benchmarks.mercari_features import (
    ListingFeatures,
    make_ridge,
    normalize,
    stem_text,
)
from benchmarks.mercari_semantics import TextAttributes


def signals(value):
    """Bounded, label-free product identity, units and condition evidence."""
    text = "" if pd.isna(value) else str(value).lower()
    result = {
        f"unit_{number}{unit}"
        for number, unit in re.findall(
            r"(\d+(?:\.\d+)?)\s*(gb|tb|oz|ml|mm|cm|inch)\b", text
        )
    }
    result.update(
        "model_" + token for token in re.findall(r"\b[a-z]+\d+[a-z\d]*\b", text)
    )
    result.update(
        "flag_" + name
        for name, pattern in TextAttributes.flags.items()
        if re.search(pattern, text)
    )
    return " ".join(sorted(result)[:32]) or "missing_signal"


class CpuFeatures(ListingFeatures):
    """Exact and stemmed views share explicit bounded categorical/token crosses.

    A separate raw character channel retains model numbers erased by stemming.
    No quantities, dictionaries or target aggregates are fitted on evaluation rows.
    """

    def __init__(self, view="exact"):
        super().__init__()
        self.view = view

    def fit(self, frame):
        if self.view not in ("exact", "stemmed"):
            raise ValueError("view must be exact or stemmed")
        return super().fit(frame)

    def transform(self, frame):
        p = super().transform(frame)
        p["raw_name"] = frame.name.map(
            lambda value: normalize(value, split_alphanumeric=False)
        )
        p["signals"] = (
            frame.name.map(signals) + " " + frame.item_description.map(signals)
        )
        crosses = []
        for title, brand, category in zip(p.name, p.brand_name, p.category_2):
            tokens = sorted(set(title.split()[:24]))
            brand, category = brand.replace(" ", "_"), category.replace(" ", "_")
            crosses.append(
                " ".join(
                    [f"brand_{brand}__{token}" for token in tokens]
                    + [f"category_{category}__{token}" for token in tokens]
                )
            )
        p["crosses"] = crosses
        if self.view == "stemmed":
            p["name"] = frame.name.fillna("<missing>").astype(str).map(stem_text)
            p["item_description"] = (
                frame.item_description.fillna("<missing>").astype(str).map(stem_text)
            )
        # Keep field markers explicit; raw title and word crosses remain separate.
        p["item_description"] = (
            p.item_description
            + " fieldtitle "
            + p.name
            + " fieldbrand "
            + p.brand_name
            + " fieldcategory "
            + p.category_name
        )
        return p


def make_features(max_features=75000, char_features=40000, cross_features=50000):
    extractor = make_ridge(max_features=max_features, char_features=char_features)[
        "features"
    ]
    extractor.transformers = [
        (name, transformer, "raw_name" if name == "characters" else columns)
        for name, transformer, columns in extractor.transformers
    ]
    extractor.transformers += [
        (
            "crosses",
            CountVectorizer(
                min_df=3,
                max_features=cross_features,
                binary=True,
                token_pattern=r"(?u)\S+",
                dtype=np.float32,
            ),
            "crosses",
        ),
        (
            "signals",
            CountVectorizer(
                min_df=1, binary=True, token_pattern=r"(?u)\S+", dtype=np.float32
            ),
            "signals",
        ),
    ]
    extractor.transformer_weights.update(crosses=1.0, signals=1.0)
    return extractor


def value_gradient(parameters, prediction, log_y, prior, penalty):
    """Squared official RMSLE + shrinkage, and its analytic weight/bias gradient.

    Inputs are validated by RmsleBlend.fit. z=max(Pw+b,0); predicted price is
    expm1(z), so log1p(predicted price)=z exactly. The square root is monotone
    and does not change the unregularized minimizer. At zero choose subgradient
    zero. Huber/bin losses elsewhere are surrogates, not this exact objective.
    """
    raw = prediction @ parameters[:-1] + parameters[-1]
    error = np.maximum(raw, 0.0) - log_y
    difference = parameters[:-1] - prior
    value = np.mean(error**2) + penalty * (
        difference @ difference + parameters[-1] ** 2
    )
    slope = 2.0 * error * (raw > 0.0) / len(log_y)
    gradient = np.r_[
        prediction.T @ slope + 2.0 * penalty * difference,
        slope.sum() + 2.0 * penalty * parameters[-1],
    ]
    return float(value), gradient


class RmsleBlend:
    """Selection-only simplex blend with a bounded log-price intercept.

    The anchor/penalty and bias bound are declared before fitting. This class
    never selects features, reads confirmation labels or uses leaderboard scores.
    """

    def __init__(self, penalty=1e-4, bias_bound=0.25, anchor=0, max_iter=300):
        self.penalty, self.bias_bound = penalty, bias_bound
        self.anchor, self.max_iter = anchor, max_iter

    def fit(self, prediction, log_y):
        p, y = np.asarray(prediction, dtype=float), np.asarray(log_y, dtype=float)
        if (
            p.ndim != 2
            or y.ndim != 1
            or not y.size
            or len(p) != len(y)
            or not np.isfinite(p).all()
            or not np.isfinite(y).all()
            or (y < 0.0).any()
            or not 0 <= self.anchor < p.shape[1]
            or self.max_iter < 1
            or not np.isfinite(self.penalty)
            or self.penalty < 0.0
            or not np.isfinite(self.bias_bound)
            or self.bias_bound < 0.0
        ):
            raise ValueError(
                "finite matching log-price data and valid blend settings required"
            )
        prior = np.eye(p.shape[1])[self.anchor]
        # A wholly clipped anchor has zero data gradient. Start from the best
        # declared vertex/uniform mixture to avoid getting stuck on that plateau.
        starts = [*np.eye(p.shape[1]), np.full(p.shape[1], 1.0 / p.shape[1])]
        initial = min(
            starts,
            key=lambda w: value_gradient(np.r_[w, 0.0], p, y, prior, self.penalty)[0],
        )
        result = minimize(
            value_gradient,
            np.r_[initial, 0.0],
            args=(p, y, prior, self.penalty),
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * p.shape[1] + [(-self.bias_bound, self.bias_bound)],
            constraints=[
                dict(
                    type="eq",
                    fun=lambda w: w[:-1].sum() - 1.0,
                    jac=lambda w: np.r_[np.ones(len(w) - 1), 0.0],
                )
            ],
            options=dict(ftol=1e-12, maxiter=self.max_iter),
        )
        if not result.success:
            raise RuntimeError("direct RMSLE blend did not converge")
        self.weights_ = np.clip(result.x[:-1], 0.0, 1.0)
        self.weights_ /= self.weights_.sum()
        self.intercept_, self.n_iter_ = float(result.x[-1]), int(result.nit)
        self.objective_ = value_gradient(
            np.r_[self.weights_, self.intercept_], p, y, prior, self.penalty
        )[0]
        return self

    def predict(self, prediction):
        p = np.asarray(prediction, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self.weights_) or not np.isfinite(p).all():
            raise ValueError("finite prediction matrix must match fitted experts")
        return p @ self.weights_ + self.intercept_
