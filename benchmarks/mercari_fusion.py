"""Compare global linear, two-hidden-layer MLP and conditional linear stacking.

Stacked generalization: Wolpert (1992), doi:10.1016/S0893-6080(05)80023-1.
Adam: Kingma & Ba (2015), https://arxiv.org/abs/1412.6980.
The grouped fit/tune/check protocol is a local adaptation. Previously inspected
selection/audit data do not constitute a new unbiased leaderboard estimate.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import resource
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_cpu import RmsleBlend
from benchmarks.mercari_cpu_study import dump, read_data, score
from benchmarks.mercari_multiview import (
    ContextBlend,
    RegularizedBlend,
    context,
    snapshot,
)
from benchmarks.mercari_semantics import paired_comparison

SEEDS = (42, 43, 44)
MAX_EPOCHS = 100
PATIENCE = 12
KINDS = ("linear", "affine", "mlp", "conditional")


def arrays(prediction, groups, log_y=None, columns=None):
    p, g = np.asarray(prediction, dtype=float), np.asarray(groups, dtype=str)
    if p.ndim != 2 or not p.size or g.shape != (len(p),) or not np.isfinite(p).all():
        raise ValueError("finite prediction matrices and matching groups required")
    if columns is not None and p.shape[1] != columns:
        raise ValueError("prediction columns must match fitted experts")
    y = None if log_y is None else np.asarray(log_y, dtype=float)
    if y is not None and (
        y.shape != (len(p),) or not np.isfinite(y).all() or (y < 0).any()
    ):
        raise ValueError("finite nonnegative labels matching predictions required")
    return p, g, y


class MLPBlend(BaseEstimator):
    """Direct log-price stacking, fixed 32/16 ReLU layers and Adam defaults.

    Scaling and category vocabulary fit meta-fit rows only. Early stopping uses
    an explicit grouped tune window. A full-selection refit uses its frozen
    epoch count, with no validation/audit labels or new checkpoint selection.
    """

    def __init__(self, seed=42):
        self.seed = seed

    def features(self, p, g):
        return np.column_stack(
            [self.scaler_.transform(p), self.categories_.transform(g[:, None])]
        )

    def fit(self, prediction, log_y, groups, validation=None, epochs=None):
        p, g, y = arrays(prediction, groups, log_y)
        if epochs is None and validation is None:
            raise ValueError("explicit grouped validation or fixed epochs required")
        if epochs is not None and (not isinstance(epochs, int) or epochs < 1):
            raise ValueError("positive integer epochs required")
        v = (
            None
            if epochs is not None
            else arrays(validation[0], validation[2], validation[1], p.shape[1])
        )
        self.columns_ = p.shape[1]
        self.scaler_ = StandardScaler().fit(p)
        self.categories_ = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ).fit(g[:, None])
        self.mean_, self.scale_ = float(y.mean()), max(float(y.std()), 1e-6)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.1,
            batch_size=min(512, len(y)),
            learning_rate_init=0.001,
            random_state=self.seed,
            max_iter=1,
            shuffle=True,
        )
        x = self.features(p, g)
        vx = None if v is None else self.features(v[0], v[1])
        target = (y - self.mean_) / self.scale_
        self.history_, best, waiting = [], float("inf"), 0
        best_model = None
        for epoch in range(1, (MAX_EPOCHS if epochs is None else epochs) + 1):
            self.model_.partial_fit(x, target)
            row = dict(epoch=epoch, train_objective=float(self.model_.loss_))
            if v is not None:
                loss = score(v[2], self.mean_ + self.scale_ * self.model_.predict(vx))
                row["tune_rmsle"] = loss
                if loss < best - 1e-6:
                    best, waiting = loss, 0
                    best_model = copy.deepcopy(self.model_)
                    self.selected_epochs_ = epoch
                else:
                    waiting += 1
            self.history_.append(row)
            if waiting >= PATIENCE:
                break
        if epochs is None:
            self.model_ = best_model
        else:
            self.selected_epochs_ = epochs
        self.parameter_count_ = sum(
            a.size for a in self.model_.coefs_ + self.model_.intercepts_
        )
        return self

    def predict(self, prediction, groups):
        check_is_fitted(self, "parameter_count_")
        p, g, _ = arrays(prediction, groups, columns=self.columns_)
        return self.mean_ + self.scale_ * self.model_.predict(self.features(p, g))


def meta_split(frame):
    """60/20/20 listing groups: fit, checkpoint tune, strategy check."""
    groups = mercari.listing_groups(frame)
    fitting, check = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=703).split(
            frame, groups=groups
        )
    )
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=704).split(
            frame.iloc[fitting], groups=groups[fitting]
        )
    )
    return dict(fit=fitting[a], tune=fitting[b], check=check)


def load_candidates(manifest, frames):
    """Trusted local manifest; aggregate models fitted on selection are rejected."""
    specs = manifest["experts"]
    if not specs or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("nonempty unique candidate names required")
    if any(s["fit_window"] not in ("base", "calibration") for s in specs):
        raise ValueError(
            "selection-fitted aggregate candidates cannot train this comparison"
        )
    return load_predictions(specs, frames)


def load_predictions(specs, frames):
    """Validate prediction bytes/IDs; eligibility for training is a separate check."""
    archives, hashes, values = {}, {}, {k: [] for k in ("selection", "confirmation")}
    for spec in specs:
        path = Path(spec["path"])
        if path not in archives:
            hashes[path] = mercari.sha256(path)
            with np.load(path) as archive:
                archives[path] = dict(archive)
        if hashes[path] != spec["sha256"]:
            raise ValueError("candidate hash must match the frozen input manifest")
        archive = archives[path]
        for k in values:
            if (
                k not in archive
                or k + "_ids" not in archive
                or not np.array_equal(
                    archive[k + "_ids"], frames[k].train_id.to_numpy()
                )
            ):
                raise ValueError("candidate ordered IDs must match every window")
            vector = archive[k]
            if "column" in spec:
                column = spec["column"]
                if (
                    vector.ndim != 2
                    or not isinstance(column, int)
                    or not 0 <= column < vector.shape[1]
                ):
                    raise ValueError("valid packed candidate column required")
                vector = vector[:, column]
            if vector.shape != (len(frames[k]),) or not np.isfinite(vector).all():
                raise ValueError(
                    "finite candidate vectors matching feature rows required"
                )
            values[k].append(vector)
    return {k: np.column_stack(v) for k, v in values.items()}


def make_model(kind, seed):
    return {
        "linear": RmsleBlend,
        "affine": RegularizedBlend,
        "conditional": ContextBlend,
        "mlp": lambda: MLPBlend(seed),
    }[kind]()


def infer(model, p, groups):
    return (
        model.predict(p) if isinstance(model, RmsleBlend) else model.predict(p, groups)
    )


def fit_model(model, p, y, groups, validation=None, epochs=None):
    if isinstance(model, MLPBlend):
        return model.fit(p, y, groups, validation=validation, epochs=epochs)
    return model.fit(p, y) if isinstance(model, RmsleBlend) else model.fit(p, y, groups)


def compare_pool(p, y, groups, splits, output_dir, names, identifiers):
    """Return all frozen predictors; the primary MLP seed is predeclared as 42."""
    a, b, c = (splits[k] for k in ("fit", "tune", "check"))
    # Same best-fit reference column for global and conditional shrinkage priors.
    first = int(np.argmin(np.mean((p["selection"][a] - y[a, None]) ** 2, axis=0)))
    order = [first, *[i for i in range(len(names)) if i != first]]
    p = {k: v[:, order] for k, v in p.items()}
    ordered_names = [names[i] for i in order]
    metrics, predictors, predictions = {}, {}, {}
    for kind in KINDS:
        for seed in SEEDS if kind == "mlp" else (42,):
            name = kind if kind != "mlp" else f"mlp_{seed}"
            start = time.perf_counter()
            model = fit_model(
                make_model(kind, seed),
                p["selection"][a],
                y[a],
                groups["selection"][a],
                validation=(p["selection"][b], y[b], groups["selection"][b]),
            )
            fit_seconds = time.perf_counter() - start
            row = dict(
                check_rmsle=score(
                    y[c], infer(model, p["selection"][c], groups["selection"][c])
                ),
                meta_fit_seconds=fit_seconds,
                seed=seed,
            )
            if kind == "mlp":
                row.update(
                    selected_epochs=model.selected_epochs_,
                    history=model.history_,
                    parameter_count=model.parameter_count_,
                )
            start = time.perf_counter()
            final = fit_model(
                make_model(kind, seed),
                p["selection"],
                y,
                groups["selection"],
                epochs=model.selected_epochs_ if kind == "mlp" else None,
            )
            row["refit_seconds"] = time.perf_counter() - start
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                predictions[name] = {
                    k: infer(final, v, groups[k]) for k, v in p.items()
                }
                timings.append(time.perf_counter() - start)
            row.update(
                predict_seconds=float(np.mean(timings)),
                predict_seconds_samples=timings,
                predict_seconds_max=max(timings),
                predict_seconds_std=float(np.std(timings)),
                selection_rmsle=score(y, predictions[name]["selection"]),
            )
            row["model"] = (
                dict(
                    kind="mlp",
                    hidden_layers=[32, 16],
                    epochs=final.selected_epochs_,
                    parameter_count=final.parameter_count_,
                    alpha=0.1,
                    batch_size=min(512, len(y)),
                    learning_rate=0.001,
                )
                if kind == "mlp"
                else snapshot(final)
            )
            metrics[name], predictors[name] = row, final
            print(
                "FUSION_MODEL",
                output_dir.name,
                name,
                row["check_rmsle"],
                row["selection_rmsle"],
                flush=True,
            )
    primary = ["linear", "mlp_42", "conditional"]
    best = min(metrics[n]["check_rmsle"] for n in primary)
    chosen = next(
        n
        for n in ("linear", "conditional", "mlp_42")
        if metrics[n]["check_rmsle"] <= best + 1e-5
    )
    with (output_dir / "models.pkl").open("wb") as stream:
        pickle.dump(dict(names=ordered_names, models=predictors), stream)
    for name, values in predictions.items():
        np.savez_compressed(
            output_dir / (name + ".npz"),
            **values,
            **{k + "_ids": identifiers[k] for k in values},
        )
    return dict(
        names=ordered_names, chosen=chosen, models=metrics, primary=primary
    ), predictions


def run(args):
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames = read_data(args)
    manifest = json.loads(args.manifest.read_text())
    data_hashes = {k: mercari.sha256(args.data_dir / (k + ".tsv.gz")) for k in frames}
    if manifest["data_sha256"] != data_hashes:
        raise ValueError("fusion inputs must match the complete data protocol")
    p = load_candidates(manifest, frames)
    groups = {k: context(frames[k]) for k in p}
    y = np.log1p(frames["selection"].price.to_numpy())
    splits = meta_split(frames["selection"])
    anchor = load_predictions(
        [
            dict(
                name="anchor",
                path=str(args.anchor),
                sha256=mercari.sha256(args.anchor),
            )
        ],
        frames,
    )
    anchor = {k: v[:, 0] for k, v in anchor.items()}
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        manifest=manifest,
        anchor_sha256=mercari.sha256(args.anchor),
        source_sha256={
            n: mercari.sha256(Path(__file__).with_name(n))
            for n in ("mercari_fusion.py", "mercari_multiview.py", "mercari_cpu.py")
        },
        split_ids={
            k: frames["selection"].train_id.iloc[v].tolist() for k, v in splits.items()
        },
        seeds=list(SEEDS),
        primary_mlp_seed=42,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        frozen_hyperparameters=True,
        cached_base_cost_excluded=True,
        confirmation_previously_inspected=True,
        historical_candidate_selection_reused=True,
        anchor_used_as_training_feature=False,
        anchor_check_score_uses_historically_fitted_selection_weights=True,
    )
    dump(args.output_dir / "plan.json", plan)
    specs = manifest["experts"]
    pools = dict(full=list(range(len(specs))))
    without = [i for i, spec in enumerate(specs) if spec["family"] != "new_sparse"]
    if without and len(without) < len(specs):
        pools["without_new_sparse"] = without
    frozen_pools, predictions = {}, {}
    for name, indices in pools.items():
        out = args.output_dir / name
        out.mkdir()
        frozen_pools[name], predictions[name] = compare_pool(
            {k: v[:, indices] for k, v in p.items()},
            y,
            groups,
            splits,
            out,
            [specs[i]["name"] for i in indices],
            {k: frames[k].train_id.to_numpy() for k in p},
        )
    chosen = frozen_pools["full"]["chosen"]
    check = splits["check"]
    accepted = (
        frozen_pools["full"]["models"][chosen]["check_rmsle"]
        < score(y[check], anchor["selection"][check]) - 1e-5
    )
    candidate = predictions["full"][chosen]
    accepted = bool(
        accepted
        and score(y, candidate["selection"]) < score(y, anchor["selection"]) - 1e-5
    )
    frozen = dict(
        plan=plan,
        pools=frozen_pools,
        chosen=chosen,
        accepted=accepted,
        choice_and_refits_completed_before_confirmation=True,
        anchor_check_rmsle=score(y[check], anchor["selection"][check]),
    )
    dump(args.output_dir / "frozen.json", frozen)
    # All checkpoint epochs, strategy choices, refits and promotion precede labels.
    with np.load(args.data_dir / "evaluation.npz") as audit:
        if (
            not np.array_equal(
                audit["confirmation_ids"], frames["confirmation"].train_id.to_numpy()
            )
            or audit["confirmation"].shape != (len(frames["confirmation"]),)
            or not np.isfinite(audit["confirmation"]).all()
            or (audit["confirmation"] < 0).any()
        ):
            raise ValueError("audit labels and ordered IDs must match frozen rows")
        audit_y = audit["confirmation"]
    results = copy.deepcopy(frozen_pools)
    for pool, rows in results.items():
        for name, row in rows["models"].items():
            row["confirmation_rmsle"] = score(
                audit_y, predictions[pool][name]["confirmation"]
            )
        errors = [rows["models"][f"mlp_{seed}"]["confirmation_rmsle"] for seed in SEEDS]
        rows["mlp_stability"] = dict(
            mean=float(np.mean(errors)),
            std=float(np.std(errors)),
            worst=max(errors),
            seed_scores=errors,
            bag_rmsle=score(
                audit_y,
                np.mean(
                    [
                        predictions[pool][f"mlp_{seed}"]["confirmation"]
                        for seed in SEEDS
                    ],
                    axis=0,
                ),
            ),
            bag_diagnostic_only=True,
        )
        rows["versus_linear"] = {
            name: paired_comparison(
                audit_y,
                predictions[pool]["linear"]["confirmation"],
                predictions[pool][name]["confirmation"],
                repeats=300,
            )
            for name in ("mlp_42", "conditional")
        }
        rows["by_category"] = {
            str(group): dict(
                rows=int(np.sum(groups["confirmation"] == group)),
                rmsle={
                    name: score(
                        audit_y[groups["confirmation"] == group],
                        predictions[pool][name]["confirmation"][
                            groups["confirmation"] == group
                        ],
                    )
                    for name in rows["primary"]
                },
            )
            for group in np.unique(groups["confirmation"])
        }
    selected = candidate if accepted else anchor
    np.savez_compressed(
        args.output_dir / "ensemble.npz",
        **selected,
        **{k + "_ids": frames[k].train_id.to_numpy() for k in selected},
    )
    report = dict(
        frozen=frozen,
        results=results,
        anchor_confirmation_rmsle=score(audit_y, anchor["confirmation"]),
        candidate_confirmation_rmsle=score(audit_y, candidate["confirmation"]),
        confirmation_rmsle=score(audit_y, selected["confirmation"]),
        comparison=paired_comparison(
            audit_y, anchor["confirmation"], candidate["confirmation"], repeats=300
        ),
        seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        gpu_used=False,
        submission_status="not_submitted",
        confirmation_previously_inspected=True,
        row_counts={k: len(v) for k, v in frames.items()},
        meta_rows={k: len(v) for k, v in splits.items()},
    )
    dump(args.output_dir / "report.json", report)
    print(
        "FUSION_REPORT",
        chosen,
        report["candidate_confirmation_rmsle"],
        accepted,
        flush=True,
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "manifest", "anchor", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.rows, args.seed = 0, 42
    with threadpool_limits(2):
        run(args)


if __name__ == "__main__":
    from benchmarks.mercari_fusion import main as entrypoint

    entrypoint()
