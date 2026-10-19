"""Market-aware heads for one base text expert, before any final ensemble.

Squared log loss elicits E[log(1+P)|X], without a Gaussian assumption:
Gneiting (2011), doi:10.1198/jasa.2011.r10138. Soft regimes use the law of total
expectation, not oracle routing by an unavailable test target. Mercari's winner
reported regression-via-classification and unsuccessful mixture experiments:
https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/5_modeling.tex
These bounded heads are independent adaptations, not that winning implementation.
"""

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

DAMAGE = re.compile(
    r"\b(?:broken|cracked|cracks|damaged|scratches|scratched|stains|stained|flaws|holes|rips|defects)\b"
)
NEGATION = re.compile(r"\b(?:no|not|without|never)\b(?:\s+\w+){0,3}\s*$")
NUMBERS = dict(
    one=1,
    two=2,
    three=3,
    four=4,
    five=5,
    six=6,
    seven=7,
    eight=8,
    nine=9,
    ten=10,
    eleven=11,
    twelve=12,
)
QUANTITY = re.compile(
    r"\b(?:lot|set|pack|bundle)\s+of\s+(\d+|"
    + "|".join(NUMBERS)
    + r")\b|\b(\d+)\s*(?:pcs|pieces|pairs|packs)\b"
)
FIELDS = (
    "damage",
    "damage_negated",
    "not_working",
    "accessory",
    "box_only",
    "bundle",
    "new",
    "quantity",
    "storage_gb",
    "redactions",
)


def market_cues(value):
    """Approximate lexical evidence, not verified product condition or identity."""
    text = "" if pd.isna(value) else str(value).lower()
    hits = list(DAMAGE.finditer(text))
    negated = [
        bool(NEGATION.search(text[max(0, m.start() - 45) : m.start()])) for m in hits
    ]
    broken = bool(
        re.search(
            r"\b(?:not working|does not work|doesn't work|for parts|non working|won't turn on)\b",
            text,
        )
    )
    counts = [
        NUMBERS.get(a, int(a) if a.isdigit() else 1) if a else int(b)
        for a, b in QUANTITY.findall(text)
    ]
    quantity = min(max(counts, default=2 if re.search(r"\bpair\b", text) else 1), 512)
    capacity = [
        float(v) * (1024 if unit == "tb" else 1)
        for v, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(gb|tb)\b", text)
    ]
    return dict(
        damage=float(broken or any(not v for v in negated)),
        damage_negated=float(any(negated)),
        not_working=float(broken),
        accessory=float(
            bool(
                re.search(r"\b(?:cases?|covers?|protectors?|chargers?|cables?)\b", text)
            )
        ),
        box_only=float(
            bool(
                re.search(
                    r"\b(?:(?:box|packaging) only|empty (?:box|bottle)|no (?:phone|device|console) included)\b",
                    text,
                )
            )
        ),
        bundle=float(
            bool(re.search(r"\b(?:bundle|lot of|set of|pack of|pair)\b", text))
        ),
        new=float(bool(re.search(r"\b(?:new|nwt|nwot|unused)\b", text))),
        quantity=float(np.log1p(quantity)),
        storage_gb=float(np.log1p(min(max(capacity, default=0), 1e6))),
        redactions=float(text.count("[rm]")),
    )


class MarketSignals:
    """Separate title/description cues; no fitting, price fields or query context."""

    def __init__(self):
        self.feature_names_ = [
            field + "_" + name for field in ("name", "description") for name in FIELDS
        ]

    def transform(self, frame):
        return np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(
                        [
                            [row[k] for k in FIELDS]
                            for row in frame[field].map(market_cues)
                        ],
                        dtype=np.float32,
                    )
                    for field in ("name", "item_description")
                ]
            ),
            dtype=np.float32,
        )


def expected_log_price(probability, conditional_means):
    """Bayes point for log-squared loss; exponentiate only AFTER this expectation."""
    p, m = (
        np.asarray(probability, dtype=float),
        np.asarray(conditional_means, dtype=float),
    )
    if (
        p.ndim != 2
        or not len(p)
        or p.shape != m.shape
        or p.shape[1] != 4
        or not np.isfinite(p).all()
        or not np.isfinite(m).all()
        or (p < 0).any()
        or (m < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "four finite nonnegative regime means and normalized probabilities required"
        )
    return np.sum(p * m, axis=1)


def arrays(x, y=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature matrix required")
    if y is None:
        return x
    y = np.asarray(y, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("matching finite nonnegative log prices required")
    return x, y


def regimes(y):
    return np.where(
        y == 0, 0, 1 + (y >= np.log1p(10)).astype(int) + (y >= np.log1p(50)).astype(int)
    )


class RegimePriceHead(BaseEstimator):
    """P(zero/low/main/high|X) times within-regime conditional log means.

    Gates use unweighted log loss: balancing classes would change the posterior.
    Rare/unseen heads fall back to their training-only conditional means. There
    is no true-price input to predict(), hard routing, target oversampling or
    correction of exp(log-price) to an arithmetic-dollar conditional mean.
    """

    def __init__(
        self,
        rounds=180,
        min_head_rows=200,
        min_child_samples=80,
        categorical_indices=(),
    ):
        self.rounds, self.min_head_rows, self.min_child_samples = (
            rounds,
            min_head_rows,
            min_child_samples,
        )
        self.categorical_indices = categorical_indices

    def fit(self, x, log_y, validation=None, budgets=None):
        x, y = arrays(x, log_y)
        if min(self.rounds, self.min_head_rows, self.min_child_samples) < 1:
            raise ValueError("positive model settings required")
        self.n_features_in_ = x.shape[1]
        if validation is not None:
            vx, vy = arrays(*validation)
            if vx.shape[1] != x.shape[1]:
                raise ValueError("validation features must match fitted columns")
        classes = regimes(y)
        self.classes_ = np.unique(classes)
        self.counts_ = np.bincount(classes, minlength=4)
        self.means_ = np.array(
            [y[classes == k].mean() if self.counts_[k] else 0 for k in range(4)]
        )
        self.heads_, self.gate_ = [], None
        self.budgets_ = dict(gate=0, heads=[0] * 4)
        common = dict(
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=self.min_child_samples,
            reg_lambda=20.0,
            n_jobs=2,
            random_state=42,
            verbosity=-1,
        )
        categorical = list(self.categorical_indices)
        if len(self.classes_) > 1 and (budgets is None or budgets["gate"] > 0):
            gate_rounds = self.rounds if budgets is None else budgets["gate"]
            self.gate_ = lgb.LGBMClassifier(n_estimators=gate_rounds, **common)
            opts = {}
            if (
                validation is not None
                and (known := np.isin(regimes(vy), self.classes_)).any()
            ):
                opts = dict(
                    eval_set=[(vx[known], regimes(vy)[known])],
                    callbacks=[lgb.early_stopping(25, verbose=False)],
                )
            self.gate_.fit(x, classes, categorical_feature=categorical, **opts)
            self.budgets_["gate"] = self.gate_.best_iteration_ or gate_rounds
        for k in range(4):
            head = None
            if (
                k
                and self.counts_[k] >= self.min_head_rows
                and (budgets is None or budgets["heads"][k] > 0)
            ):
                rounds = self.rounds if budgets is None else budgets["heads"][k]
                head = lgb.LGBMRegressor(n_estimators=rounds, **common)
                opts = {}
                if validation is not None and (select := regimes(vy) == k).any():
                    opts = dict(
                        eval_set=[(vx[select], vy[select])],
                        callbacks=[lgb.early_stopping(25, verbose=False)],
                    )
                head.fit(
                    x[classes == k],
                    y[classes == k],
                    categorical_feature=categorical,
                    **opts,
                )
                self.budgets_["heads"][k] = head.best_iteration_ or rounds
            self.heads_.append(head)
        return self

    def predict(self, x):
        check_is_fitted(self, "heads_")
        x = arrays(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("features must match fitted columns")
        probability = np.zeros((len(x), 4))
        if self.gate_ is None:
            probability[:] = self.counts_ / self.counts_.sum()
        else:
            probability[:, self.gate_.classes_] = self.gate_.predict_proba(x)
        means = np.tile(self.means_, (len(x), 1))
        lower = [0, 0, np.log1p(10), np.log1p(50)]
        upper = [0, np.log1p(10), np.log1p(50), np.inf]
        for k, head in enumerate(self.heads_):
            if head is not None:
                weight = self.counts_[k] / (self.counts_[k] + 500.0)
                means[:, k] = np.clip(
                    weight * head.predict(x) + (1 - weight) * self.means_[k],
                    lower[k],
                    upper[k],
                )
        return expected_log_price(probability, means)
