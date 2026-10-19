"""Optimize the final Mercari prediction, with exact full-window metric gradients.

Official metric: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/overview/evaluation
L-BFGS-B: Byrd et al. (1995), doi:10.1137/0916069.
RMSLE and its square share unregularized minimizers. This experiment separates
clipping, optimizer changes and joint loss placement instead of assuming a new
metric name must improve the previous log-MSE fusion.
"""

import copy
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import resource
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_fusion import (
    MLPBlend,
    SEEDS,
    arrays,
    score,
    read_data,
    load_candidates,
    load_predictions,
    context,
    meta_split,
    make_model,
    fit_model,
    infer,
    dump,
)
from benchmarks.mercari_semantics import paired_comparison

LOSSES = ("log_mse", "squared_rmsle", "rmsle")
MAX_ITERATIONS = 20
PATIENCE = 6


def metric_gradient(prediction, log_y, loss):
    """Metric and dL/d(final log prediction); clip once after member averaging.

    The RMSLE denominator uses the complete fit window, never an average of
    minibatch roots. At a zero prediction use the zero clipping subgradient;
    at perfect RMSLE choose the zero norm subgradient.
    """
    p, y = np.asarray(prediction, dtype=float), np.asarray(log_y, dtype=float)
    if (
        p.ndim != 1
        or not p.size
        or y.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
        or (y < 0).any()
    ):
        raise ValueError("finite matching nonnegative log labels required")
    if loss not in LOSSES:
        raise ValueError("known final-prediction loss required")
    error = (p if loss == "log_mse" else np.maximum(p, 0.0)) - y
    slope = np.ones_like(p) if loss == "log_mse" else (p > 0.0)
    square = float(np.mean(error * error))
    if loss == "rmsle":
        value = float(np.sqrt(square))
        return value, error * slope / (len(p) * value) if value else np.zeros_like(p)
    return square, 2.0 * error * slope / len(p)


class NetworkObjective:
    """Transient training data and analytic backprop for 32/16 ReLU members.

    Each member retains its fit-only preprocessing and target scaling. Only
    dense fusion weights/biases are optimized; cached upstream experts are fixed.
    This object is never stored in a fitted prediction artifact.
    """

    def __init__(self, members, prediction, log_y, groups):
        p, g, self.y = arrays(prediction, groups, log_y)
        if not members:
            raise ValueError("at least one fitted MLP member required")
        self.members = members
        self.x = [m.features(p, g) for m in members]
        self.shapes = [
            a.shape for m in members for a in m.model_.coefs_ + m.model_.intercepts_
        ]
        self.offsets = np.r_[0, np.cumsum([np.prod(s) for s in self.shapes])].astype(
            int
        )
        self.initial = np.concatenate(
            [a.ravel() for m in members for a in m.model_.coefs_ + m.model_.intercepts_]
        )

    def tensors(self, parameters):
        theta = np.asarray(parameters, dtype=float)
        if theta.shape != self.initial.shape or not np.isfinite(theta).all():
            raise ValueError("finite parameters matching every member required")
        return [
            theta[a:b].reshape(shape)
            for a, b, shape in zip(self.offsets[:-1], self.offsets[1:], self.shapes)
        ]

    def forward(self, parameters):
        tensors, prediction, cache = self.tensors(parameters), [], []
        for i, (member, x) in enumerate(zip(self.members, self.x)):
            w0, w1, w2, b0, b1, b2 = tensors[6 * i : 6 * i + 6]
            a1 = np.maximum(x @ w0 + b0, 0.0)
            a2 = np.maximum(a1 @ w1 + b1, 0.0)
            prediction.append(member.mean_ + member.scale_ * (a2 @ w2 + b2).ravel())
            cache.append((x, a1, a2, w1, w2))
        return np.mean(prediction, axis=0), cache

    def predict(self, parameters):
        return self.forward(parameters)[0]

    def value_gradient(self, parameters, loss):
        prediction, cache = self.forward(parameters)
        value, slope = metric_gradient(prediction, self.y, loss)
        gradient = []
        for member, (x, a1, a2, w1, w2) in zip(self.members, cache):
            d2 = slope[:, None] * member.scale_ / len(self.members)
            d1 = (d2 @ w2.T) * (a2 > 0)
            d0 = (d1 @ w1.T) * (a1 > 0)
            gradient.extend(
                [
                    x.T @ d0,
                    a1.T @ d1,
                    a2.T @ d2,
                    d0.sum(axis=0),
                    d1.sum(axis=0),
                    d2.sum(axis=0),
                ]
            )
        return value, np.concatenate([a.ravel() for a in gradient])

    def assign(self, parameters):
        tensors = self.tensors(parameters)
        for i, member in enumerate(self.members):
            member.model_.coefs_ = [a.copy() for a in tensors[6 * i : 6 * i + 3]]
            member.model_.intercepts_ = [
                a.copy() for a in tensors[6 * i + 3 : 6 * i + 6]
            ]


class PredictionMean:
    """Mean log predictions; the metric is applied only after this operation."""

    def __init__(self, members):
        self.members = members

    def predict(self, prediction, groups):
        return np.mean([m.predict(prediction, groups) for m in self.members], axis=0)


class MetricBlend(BaseEstimator):
    """Full-window L-BFGS refinement, shared warm starts and no extra L2 term.

    Early stopping and a fixed iteration budget regularize the refinement.
    Squared/RMSLE objectives operate on the final member mean. The previous
    Adam warm start is shared across losses; a log-MSE refinement control
    distinguishes objective changes from using a new optimizer.
    """

    def __init__(self, loss="rmsle"):
        self.loss = loss

    def fit(
        self, prediction, log_y, groups, *, members, validation=None, iterations=None
    ):
        if iterations is None and validation is None:
            raise ValueError("explicit tune validation or frozen iterations required")
        if iterations is not None and (
            not isinstance(iterations, int) or iterations < 0
        ):
            raise ValueError("nonnegative integer iterations required")
        self.members_ = copy.deepcopy(members)
        objective = NetworkObjective(self.members_, prediction, log_y, groups)
        tune = (
            None
            if iterations is not None
            else NetworkObjective(self.members_, *validation[:2], validation[2])
        )
        initial, _ = objective.value_gradient(objective.initial, self.loss)
        best_theta, waiting = objective.initial.copy(), 0
        best = None if tune is None else score(tune.y, tune.predict(best_theta))
        self.history_ = [dict(iteration=0, train_objective=initial, tune_rmsle=best)]
        self.selected_iterations_ = 0

        def checkpoint(theta):
            nonlocal best_theta, best, waiting
            value = metric_gradient(objective.predict(theta), objective.y, self.loss)[0]
            row = dict(iteration=len(self.history_), train_objective=value)
            if tune is not None:
                row["tune_rmsle"] = score(tune.y, tune.predict(theta))
                if row["tune_rmsle"] < best - 1e-6:
                    best, waiting, best_theta = row["tune_rmsle"], 0, theta.copy()
                    self.selected_iterations_ = row["iteration"]
                else:
                    waiting += 1
            self.history_.append(row)
            if waiting >= PATIENCE:
                raise StopIteration

        budget = MAX_ITERATIONS if iterations is None else iterations
        if budget:
            result = minimize(
                objective.value_gradient,
                objective.initial,
                args=(self.loss,),
                jac=True,
                method="L-BFGS-B",
                callback=checkpoint,
                options=dict(maxiter=budget, maxls=20, ftol=1e-12, gtol=1e-7),
            )
            self.optimizer_ = dict(
                iterations=int(result.nit),
                evaluations=int(result.nfev),
                success=bool(result.success),
                status=int(result.status),
                message=str(result.message),
            )
            if tune is None:
                best_theta = result.x
                self.selected_iterations_ = int(result.nit)
        else:
            self.optimizer_ = dict(
                iterations=0,
                evaluations=1,
                success=True,
                status=0,
                message="Frozen zero-step warm start",
            )
        self.requested_iterations_ = budget
        self.initial_objective_ = initial
        self.final_objective_ = metric_gradient(
            objective.predict(best_theta), objective.y, self.loss
        )[0]
        self.negative_final_predictions_ = int(
            np.sum(objective.predict(best_theta) < 0)
        )
        objective.assign(best_theta)
        return self

    def predict(self, prediction, groups):
        check_is_fitted(self, "final_objective_")
        return PredictionMean(self.members_).predict(prediction, groups)


def run(args):
    """Identical base predictions, shared clean warm starts, frozen metric choices."""
    start = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    frame = read_data(args)
    manifest = json.loads(args.manifest.read_text())
    hashes = {k: mercari.sha256(args.data_dir / (k + ".tsv.gz")) for k in frame}
    if manifest["data_sha256"] != hashes:
        raise ValueError("objective study must match the complete data protocol")
    p = load_candidates(manifest, frame)
    groups = {k: context(frame[k]) for k in p}
    y = np.log1p(frame["selection"].price.to_numpy())
    splits = meta_split(frame["selection"])
    a, b, c = (splits[k] for k in ("fit", "tune", "check"))
    specs = manifest["experts"]
    first = int(np.argmin(np.mean((p["selection"][a] - y[a, None]) ** 2, axis=0)))
    order = [first, *[i for i in range(len(specs)) if i != first]]
    p = {k: v[:, order] for k, v in p.items()}
    names = [specs[i]["name"] for i in order]
    anchor = load_predictions(
        [dict(path=str(args.anchor), sha256=mercari.sha256(args.anchor))], frame
    )
    anchor = {k: v[:, 0] for k, v in anchor.items()}
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        manifest=manifest,
        source_sha256={
            n: mercari.sha256(Path(__file__).with_name(n))
            for n in (
                "mercari_objective.py",
                "mercari_fusion.py",
                "mercari_cpu.py",
                "mercari_multiview.py",
            )
        },
        anchor_sha256=mercari.sha256(args.anchor),
        seeds=list(SEEDS),
        names=names,
        split_ids={
            k: frame["selection"].train_id.iloc[v].tolist() for k, v in splits.items()
        },
        max_iterations=MAX_ITERATIONS,
        tune_patience=PATIENCE,
        fine_tune_L2=0.0,
        full_window_objective=True,
        losses=list(LOSSES),
        clip_after_member_mean=True,
        primary_single_seed=42,
        cached_base_cost_excluded=True,
        historical_selection_reused=True,
        confirmation_previously_inspected=True,
        joint_ensemble_initializations=1,
        anchor_used_as_training_feature=False,
    )
    dump(out / "plan.json", plan)
    models, meta_models, predictions, check_prediction, metrics = {}, {}, {}, {}, {}

    def record(name, meta, final, fit_seconds, refit_seconds, **extra):
        check_prediction[name] = infer(meta, p["selection"][c], groups["selection"][c])
        samples = []
        for _ in range(3):
            stamp = time.perf_counter()
            predictions[name] = {k: infer(final, v, groups[k]) for k, v in p.items()}
            samples.append(time.perf_counter() - stamp)
        row = dict(
            check_rmsle=score(y[c], check_prediction[name]),
            selection_rmsle=score(y, predictions[name]["selection"]),
            meta_fit_seconds=fit_seconds,
            refit_seconds=refit_seconds,
            predict_seconds=float(np.mean(samples)),
            predict_seconds_samples=samples,
            predict_seconds_max=max(samples),
            predict_seconds_std=float(np.std(samples)),
            negative_predictions={
                k: int(np.sum(v < 0)) for k, v in predictions[name].items()
            },
            **extra,
        )
        if isinstance(meta, MetricBlend):
            row.update(
                loss=meta.loss,
                selected_iterations=meta.selected_iterations_,
                history=meta.history_,
                meta_optimizer=meta.optimizer_,
                refit_optimizer=final.optimizer_,
                refit_requested_iterations=final.requested_iterations_,
                refit_actual_iterations=final.selected_iterations_,
                initial_train_objective=meta.initial_objective_,
                final_train_objective=meta.final_objective_,
            )
        models[name], meta_models[name], metrics[name] = final, meta, row
        np.savez_compressed(
            out / (name + ".npz"),
            **predictions[name],
            **{k + "_ids": frame[k].train_id.to_numpy() for k in p},
        )
        print(
            "OBJECTIVE_MODEL",
            name,
            row["check_rmsle"],
            row["selection_rmsle"],
            row.get("selected_iterations"),
            flush=True,
        )

    for kind in ("linear", "conditional"):
        stamp = time.perf_counter()
        meta = fit_model(
            make_model(kind, 42), p["selection"][a], y[a], groups["selection"][a]
        )
        fit_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        final = fit_model(make_model(kind, 42), p["selection"], y, groups["selection"])
        record(
            kind,
            meta,
            final,
            fit_seconds,
            time.perf_counter() - stamp,
            loss="squared_rmsle_plus_existing_shrinkage",
        )
    warm_meta, warm_final = [], []
    for seed in SEEDS:
        stamp = time.perf_counter()
        meta = MLPBlend(seed).fit(
            p["selection"][a],
            y[a],
            groups["selection"][a],
            validation=(p["selection"][b], y[b], groups["selection"][b]),
        )
        fit_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        final = MLPBlend(seed).fit(
            p["selection"], y, groups["selection"], epochs=meta.selected_epochs_
        )
        record(
            f"warm_seed{seed}",
            meta,
            final,
            fit_seconds,
            time.perf_counter() - stamp,
            loss="standardized_log_mse_plus_existing_L2",
            selected_epochs=meta.selected_epochs_,
        )
        warm_meta.append(meta)
        warm_final.append(final)
    record(
        "warm_mean3",
        PredictionMean(warm_meta),
        PredictionMean(warm_final),
        0.0,
        0.0,
        dependencies=[f"warm_seed{s}" for s in SEEDS],
        warm_training_cost_shared=True,
    )
    for loss in LOSSES:
        independent_meta, independent_final, dependencies = [], [], []
        for i, seed in enumerate(SEEDS):
            name = f"single_{loss}_{seed}"
            stamp = time.perf_counter()
            meta = MetricBlend(loss).fit(
                p["selection"][a],
                y[a],
                groups["selection"][a],
                members=[warm_meta[i]],
                validation=(p["selection"][b], y[b], groups["selection"][b]),
            )
            fit_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            final = MetricBlend(loss).fit(
                p["selection"],
                y,
                groups["selection"],
                members=[warm_final[i]],
                iterations=meta.selected_iterations_,
            )
            record(
                name,
                meta,
                final,
                fit_seconds,
                time.perf_counter() - stamp,
                members=1,
                seed=seed,
                warm_training_cost_shared=True,
            )
            independent_meta.append(meta)
            independent_final.append(final)
            dependencies.append(name)
        record(
            f"independent_{loss}_mean3",
            PredictionMean(independent_meta),
            PredictionMean(independent_final),
            0.0,
            0.0,
            loss=loss,
            dependencies=dependencies,
            training_cost_shared=True,
        )
        stamp = time.perf_counter()
        meta = MetricBlend(loss).fit(
            p["selection"][a],
            y[a],
            groups["selection"][a],
            members=warm_meta,
            validation=(p["selection"][b], y[b], groups["selection"][b]),
        )
        fit_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        final = MetricBlend(loss).fit(
            p["selection"],
            y,
            groups["selection"],
            members=warm_final,
            iterations=meta.selected_iterations_,
        )
        record(
            f"joint_{loss}_mean3",
            meta,
            final,
            fit_seconds,
            time.perf_counter() - stamp,
            members=3,
            warm_training_cost_shared=True,
        )
    primary = [
        "linear",
        "conditional",
        "warm_seed42",
        "warm_mean3",
        *[f"single_{loss}_42" for loss in LOSSES],
        *[f"independent_{loss}_mean3" for loss in LOSSES],
        *[f"joint_{loss}_mean3" for loss in LOSSES],
    ]
    best = min(metrics[n]["check_rmsle"] for n in primary)
    chosen = next(n for n in primary if metrics[n]["check_rmsle"] <= best + 1e-5)
    accepted = bool(
        metrics[chosen]["check_rmsle"] < score(y[c], anchor["selection"][c]) - 1e-5
        and metrics[chosen]["selection_rmsle"] < score(y, anchor["selection"]) - 1e-5
    )
    frozen = dict(
        plan=plan,
        models=copy.deepcopy(metrics),
        primary=primary,
        chosen=chosen,
        accepted=accepted,
        all_choices_frozen_before_confirmation=True,
        anchor_check_rmsle=score(y[c], anchor["selection"][c]),
    )
    for filename, bundle in (("models.pkl", models), ("meta-models.pkl", meta_models)):
        with (out / filename).open("wb") as stream:
            pickle.dump(dict(names=names, models=bundle), stream)
    np.savez_compressed(
        out / "meta-check.npz",
        **check_prediction,
        ids=frame["selection"].train_id.iloc[c].to_numpy(),
    )
    dump(out / "frozen.json", frozen)
    with np.load(args.data_dir / "evaluation.npz") as audit:
        audit_y = audit["confirmation"]
        if (
            not np.array_equal(
                audit["confirmation_ids"], frame["confirmation"].train_id.to_numpy()
            )
            or audit_y.shape != (len(frame["confirmation"]),)
            or not np.isfinite(audit_y).all()
            or (audit_y < 0).any()
        ):
            raise ValueError("audit labels and ordered IDs must match frozen windows")
    for name, row in metrics.items():
        row["confirmation_rmsle"] = score(audit_y, predictions[name]["confirmation"])
    stability = {
        loss: dict(
            seed_scores=[
                metrics[f"single_{loss}_{s}"]["confirmation_rmsle"] for s in SEEDS
            ],
            mean=float(
                np.mean(
                    [metrics[f"single_{loss}_{s}"]["confirmation_rmsle"] for s in SEEDS]
                )
            ),
            std=float(
                np.std(
                    [metrics[f"single_{loss}_{s}"]["confirmation_rmsle"] for s in SEEDS]
                )
            ),
            worst=max(
                metrics[f"single_{loss}_{s}"]["confirmation_rmsle"] for s in SEEDS
            ),
        )
        for loss in LOSSES
    }
    comparisons = {
        loss: {
            label: paired_comparison(
                audit_y,
                predictions[left]["confirmation"],
                predictions[right]["confirmation"],
                repeats=300,
            )
            for label, left, right in (
                ("single_vs_warm", "warm_seed42", f"single_{loss}_42"),
                ("independent_vs_warm", "warm_mean3", f"independent_{loss}_mean3"),
                ("joint_vs_warm", "warm_mean3", f"joint_{loss}_mean3"),
                (
                    "joint_vs_independent",
                    f"independent_{loss}_mean3",
                    f"joint_{loss}_mean3",
                ),
            )
        }
        for loss in LOSSES
    }
    selected = predictions[chosen] if accepted else anchor
    np.savez_compressed(
        out / "ensemble.npz",
        **selected,
        **{k + "_ids": frame[k].train_id.to_numpy() for k in p},
    )
    report = dict(
        frozen=frozen,
        models=metrics,
        stability=stability,
        comparisons=comparisons,
        anchor_confirmation_rmsle=score(audit_y, anchor["confirmation"]),
        candidate_confirmation_rmsle=metrics[chosen]["confirmation_rmsle"],
        confirmation_rmsle=score(audit_y, selected["confirmation"]),
        seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        row_counts={k: len(v) for k, v in frame.items()},
        meta_rows={k: len(v) for k, v in splits.items()},
        gpu_used=False,
        submission_status="not_submitted",
        confirmation_previously_inspected=True,
    )
    dump(out / "report.json", report)
    print(
        "OBJECTIVE_REPORT",
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
    from benchmarks.mercari_objective import main as entrypoint

    entrypoint()
