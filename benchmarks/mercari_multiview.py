"""Train-only product statistics and regularized CPU expert recombination.

Target smoothing: Micci-Barreca (2001), doi:10.1145/507533.507538.
Stacking: Wolpert (1992), doi:10.1016/S0893-6080(05)80023-1.
Global-to-local shrinkage is an independent, fixed-penalty adaptation, not an
implementation of a Bayesian posterior or a reproduced contest winner.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit

from benchmarks.mercari import listing_groups, prepare
from benchmarks.mercari_features import normalize
from benchmarks.mercari_cpu import RmsleBlend, value_gradient


class ProductStatistics:
    """Smoothed identity/brand/category/condition means, dispersion and support.

    Fitting may use base labels only. Unknown keys fall back to the base prior,
    zero support and base dispersion. No evaluated row contributes its label.
    """

    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing

    @staticmethod
    def keys(frame):
        p = prepare(frame)
        title = p.name.map(normalize)
        brand = p.brand_name.map(normalize)
        category = p.category_name.str.lower()
        stop = {
            "new",
            "nwt",
            "nwot",
            "authentic",
            "free",
            "shipping",
            "sale",
            "excellent",
            "condition",
            "with",
            "the",
            "a",
            "an",
        }
        tokens = title.map(
            lambda value: [word for word in value.split() if word not in stop]
        )
        identity = tokens.map(lambda words: " ".join(sorted(set(words))))
        prefix = tokens.map(lambda words: " ".join(words[:3]))
        numbers = tokens.map(
            lambda words: " ".join(
                sorted({w for w in words if any(c.isdigit() for c in w)})
            )
        )
        parent = brand + "|" + category
        return dict(
            title=parent + "|" + title,
            identity=parent + "|" + identity,
            prefix=parent + "|" + prefix,
            brand_condition=parent + "|" + p.item_condition_id,
            model=parent + "|" + numbers,
        )

    def fit(self, frame, log_y):
        y = np.asarray(log_y, dtype=float)
        if (
            y.shape != (len(frame),)
            or not len(y)
            or not np.isfinite(y).all()
            or (y < 0).any()
            or not np.isfinite(self.smoothing)
            or self.smoothing <= 0
        ):
            raise ValueError(
                "finite nonnegative log prices and positive smoothing required"
            )
        self.prior_, self.second_ = float(y.mean()), float(np.mean(y * y))
        self.maps_, self.feature_names_ = {}, []
        for name, key in self.keys(frame).items():
            table = (
                pd.DataFrame(dict(key=key.to_numpy(), y=y, y2=y * y))
                .groupby("key")
                .agg(total=("y", "sum"), second=("y2", "sum"), count=("y", "size"))
            )
            count = table["count"] + self.smoothing
            mean = (table.total + self.smoothing * self.prior_) / count
            second = (table.second + self.smoothing * self.second_) / count
            self.maps_[name] = pd.DataFrame(
                dict(
                    mean=mean,
                    support=np.log1p(table["count"]),
                    std=np.sqrt(np.maximum(second - mean * mean, 0.0)),
                )
            )
            self.feature_names_ += [
                name + "_" + suffix for suffix in ("mean", "support", "std")
            ]
        return self

    def transform(self, frame):
        columns = []
        fallback = [self.prior_, 0.0, np.sqrt(max(self.second_ - self.prior_**2, 0.0))]
        for name, key in self.keys(frame).items():
            table = self.maps_[name]
            for column, value in zip(table, fallback):
                if hasattr(self, "parents_") and column != "support":
                    parent = key.str.rsplit("|", n=1).str[0]
                    value = parent.map(self.parents_[column]).fillna(value)
                columns.append(key.map(table[column]).fillna(value).to_numpy())
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)

    def reparameterize(self, smoothing, hierarchical=False):
        """Reuse base-only sufficient statistics; never refit on query labels."""
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("positive smoothing required")
        raw = getattr(self, "sufficient_", None)
        if raw is None:
            raw = {}
            for name, table in self.maps_.items():
                count = np.rint(np.expm1(table.support))
                total = (
                    table["mean"] * (count + self.smoothing)
                    - self.smoothing * self.prior_
                )
                second = (table["std"] ** 2 + table["mean"] ** 2) * (
                    count + self.smoothing
                ) - self.smoothing * self.second_
                raw[name] = pd.DataFrame(dict(total=total, second=second, count=count))
        result = ProductStatistics(smoothing)
        result.prior_, result.second_ = self.prior_, self.second_
        result.feature_names_ = list(self.feature_names_)
        result.sufficient_, result.maps_ = raw, {}
        if hierarchical:
            table = raw["brand_condition"]
            parent = table.index.str.rsplit("|", n=1).str[0]
            pooled = table.groupby(parent).sum()
            mean = (pooled.total + 20 * self.prior_) / (pooled["count"] + 20)
            second = (pooled.second + 20 * self.second_) / (pooled["count"] + 20)
            result.parents_ = pd.DataFrame(
                dict(
                    mean=mean,
                    second=second,
                    std=np.sqrt(np.maximum(second - mean**2, 0.0)),
                )
            )
        for name, table in raw.items():
            mean_prior, second_prior = self.prior_, self.second_
            if hierarchical:
                parent = pd.Series(
                    table.index.str.rsplit("|", n=1).str[0], index=table.index
                )
                mean_prior = parent.map(result.parents_["mean"])
                second_prior = parent.map(result.parents_["second"])
            mean = (table.total + smoothing * mean_prior) / (table["count"] + smoothing)
            second = (table.second + smoothing * second_prior) / (
                table["count"] + smoothing
            )
            result.maps_[name] = pd.DataFrame(
                dict(
                    mean=mean,
                    support=np.log1p(table["count"]),
                    std=np.sqrt(np.maximum(second - mean**2, 0.0)),
                )
            )
        return result


def decompose(entry, frame):
    """Extract every fitted base expert once, sharing text preparation and CSR.

    The old corrected branch is reconstructed separately. It must never enter
    new calibration inputs: its correction already fitted calibration labels.
    """
    expert = entry.experts
    prepared = expert.listing.transform(frame)
    x = expert.extractor.transform(prepared)
    ridge = (
        np.full(len(frame), expert.mean)
        if expert.ridge is None
        else expert.ridge.predict(x)
    )
    fm = expert.mean + expert.fm.predict(
        x[:, expert.keep].multiply(expert.scales).tocsr()
    )
    neural = expert.neural
    members = getattr(neural, "members", [neural])
    outputs = np.column_stack(
        [member.predict(frame, prepared, x) for member in members]
    )
    weights = getattr(neural, "weights", np.ones(1))
    original = np.column_stack([ridge, fm, outputs @ weights])
    metadata = entry.encoder.transform(frame, prepared=prepared)
    uncorrected = original @ entry.config["weights"]
    correction = entry.native.predict(
        np.ascontiguousarray(np.column_stack([metadata, original]), dtype=np.float32)
    )
    return dict(
        names=[
            "mean_prior" if expert.ridge is None else "ridge",
            "ffm",
            *["neural_" + str(i) for i in range(len(members))],
        ],
        prediction=np.column_stack([ridge, fm, outputs]),
        metadata=metadata,
        uncorrected=uncorrected,
        reference=uncorrected + entry.config["correction_strength"] * correction,
    )


def disagreement(prediction):
    p = np.asarray(prediction)
    return np.column_stack(
        [
            p.std(axis=1),
            p.min(axis=1),
            p.max(axis=1),
            p[:, 0] - p[:, 1],
            p[:, -1] - p[:, 1],
        ]
    )


class RegularizedBlend(RmsleBlend):
    """Nonnegative affine stacking with a bounded coefficient sum and bias."""

    def __init__(
        self, penalty=0.001, bias_bound=0.25, mass_bounds=(0.8, 1.2), prior=None
    ):
        super().__init__(penalty=penalty, bias_bound=bias_bound)
        self.mass_bounds, self.prior = mass_bounds, prior

    def fit(self, prediction, log_y):
        p, y = np.asarray(prediction, dtype=float), np.asarray(log_y, dtype=float)
        bounds = np.asarray(self.mass_bounds, dtype=float)
        if (
            p.ndim != 2
            or y.ndim != 1
            or not y.size
            or p.shape[0] != len(y)
            or not p.shape[1]
            or not np.isfinite(p).all()
            or not np.isfinite(y).all()
            or (y < 0).any()
            or bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= bounds[1]
            or not np.isfinite(self.penalty)
            or self.penalty < 0
            or not np.isfinite(self.bias_bound)
            or self.bias_bound < 0
        ):
            raise ValueError("finite data and valid affine blend settings required")
        prior = (
            np.eye(p.shape[1])[0]
            if self.prior is None
            else np.asarray(self.prior, dtype=float)
        )
        if (
            prior.shape != (p.shape[1],)
            or not np.isfinite(prior).all()
            or (prior < 0).any()
            or not bounds[0] - 1e-8 <= prior.sum() <= bounds[1] + 1e-8
        ):
            raise ValueError("prior must be finite nonnegative and within mass bounds")
        starts = [prior, *np.eye(p.shape[1])]
        initial = min(
            starts,
            key=lambda w: value_gradient(np.r_[w, 0.0], p, y, prior, self.penalty)[0],
        )
        direction = np.r_[np.ones(p.shape[1]), 0.0]
        result = minimize(
            value_gradient,
            np.r_[initial, 0.0],
            args=(p, y, prior, self.penalty),
            jac=True,
            method="SLSQP",
            bounds=[(0.0, float(bounds[1]))] * p.shape[1]
            + [(-self.bias_bound, self.bias_bound)],
            constraints=[
                dict(
                    type="ineq",
                    fun=lambda w: w[:-1].sum() - bounds[0],
                    jac=lambda w: direction,
                ),
                dict(
                    type="ineq",
                    fun=lambda w: bounds[1] - w[:-1].sum(),
                    jac=lambda w: -direction,
                ),
            ],
            options=dict(ftol=1e-12, maxiter=300),
        )
        if not result.success:
            raise RuntimeError("affine blend did not converge")
        self.weights_, self.intercept_ = (
            np.maximum(result.x[:-1], 0.0),
            float(result.x[-1]),
        )
        self.n_iter_ = int(result.nit)
        return self


class ContextBlend:
    """Large-category weights shrink toward a global affine model."""

    def __init__(self, min_rows=2000, local_penalty=0.01):
        self.min_rows, self.local_penalty = min_rows, local_penalty

    def fit(self, prediction, log_y, groups):
        p, y, g = (
            np.asarray(prediction),
            np.asarray(log_y),
            np.asarray(groups, dtype=str),
        )
        if (
            g.shape != (len(p),)
            or self.min_rows < 1
            or not np.isfinite(self.local_penalty)
            or self.local_penalty <= 0
        ):
            raise ValueError("matching category groups and positive support required")
        self.global_ = RegularizedBlend().fit(p, y)
        self.locals_ = {}
        for name in np.unique(g):
            mask = g == name
            if mask.sum() >= self.min_rows:
                self.locals_[str(name)] = RegularizedBlend(
                    penalty=self.local_penalty, prior=self.global_.weights_
                ).fit(p[mask], y[mask])
        return self

    def predict(self, prediction, groups):
        p, g = np.asarray(prediction), np.asarray(groups, dtype=str)
        if g.shape != (len(p),):
            raise ValueError("prediction category groups must match rows")
        result = self.global_.predict(p)
        for name, model in self.locals_.items():
            mask = g == name
            if mask.any():
                result[mask] = model.predict(p[mask])
        return result


def context(frame):
    return prepare(frame).category_name.str.split("/").str[0].str.lower().to_numpy()


def predict(model, p, frame):
    return (
        model.predict(p, context(frame))
        if isinstance(model, ContextBlend)
        else model.predict(p)
    )


def snapshot(model):
    if isinstance(model, ContextBlend):
        return dict(
            kind="context",
            min_rows=model.min_rows,
            local_penalty=model.local_penalty,
            global_model=snapshot(model.global_),
            local_models={
                name: snapshot(local) for name, local in model.locals_.items()
            },
        )
    return dict(
        kind="affine" if isinstance(model, RegularizedBlend) else "simplex",
        weights=model.weights_.tolist(),
        intercept=model.intercept_,
        penalty=model.penalty,
        mass_bounds=list(model.mass_bounds)
        if isinstance(model, RegularizedBlend)
        else [1.0, 1.0],
    )


def fit_strategy(kind, p, y, frame):
    factories = dict(simplex=RmsleBlend, affine=RegularizedBlend, context=ContextBlend)
    model = factories[kind]()
    return model.fit(p, y, context(frame)) if kind == "context" else model.fit(p, y)


def select_strategy(p, y, frame):
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=193).split(
            frame, groups=listing_groups(frame)
        )
    )
    scores = {}
    for kind in ("simplex", "affine", "context"):
        model = fit_strategy(kind, p[a], y[a], frame.iloc[a])
        scores[kind] = float(
            np.sqrt(
                np.mean(
                    (np.maximum(predict(model, p[b], frame.iloc[b]), 0.0) - y[b]) ** 2
                )
            )
        )
    best = min(scores.values())
    chosen = next(name for name in scores if scores[name] <= best + 1e-5)
    return fit_strategy(chosen, p, y, frame), dict(
        chosen=chosen,
        validation=scores,
        fit_ids=frame.train_id.iloc[a].tolist(),
        validation_ids=frame.train_id.iloc[b].tolist(),
        tie_tolerance=1e-5,
        seed=193,
    )
