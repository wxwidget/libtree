"""Retain raw experts for calibration-only residual stacking and boosting.

Stacking: Wolpert (1992), doi:10.1016/S0893-6080(05)80023-1.
Functional-gradient boosting: Friedman (2001), doi:10.1214/aos/1013203451.
Independent weak experts have no boosting guarantee: their conditional residual
information must survive validation. Never fit this layer on calibration-fitted
anchor predictions. The anchor belongs only to the outer selection blend.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_cpu_study import WINDOWS, dump, read_data, score
from benchmarks.mercari_multiview import ProductStatistics
from benchmarks.mercari_semantics import TextAttributes


class ResidualCorrection(BaseEstimator):
    """A signed linear correction or sequential boosted residual trees.

    Fixed bounded defaults, no standalone accuracy/convergence gate. All input
    prediction columns must come from models fitted without calibration labels.
    Context statistics must also be fitted strictly on the original base split.
    """

    def __init__(self, kind="linear", threads=4):
        self.kind, self.threads = kind, threads

    @staticmethod
    def arrays(prediction, context=None):
        p = np.asarray(prediction, dtype=float)
        if p.ndim != 2:
            raise ValueError("finite prediction matrices and matching context required")
        c = (
            np.empty((len(p), 0))
            if context is None
            else np.asarray(context, dtype=float)
        )
        if (
            p.ndim != 2
            or min(p.shape) < 1
            or c.ndim != 2
            or c.shape[0] != len(p)
            or not np.isfinite(p).all()
            or not np.isfinite(c).all()
        ):
            raise ValueError("finite prediction matrices and matching context required")
        return p, c

    def features(self, p, c):
        base = p[:, self.base_index_]
        return np.column_stack([base, p - base[:, None], c])

    def fit(self, prediction, log_y, context=None):
        p, c = self.arrays(prediction, context)
        y = np.asarray(log_y, dtype=float)
        if y.shape != (len(p),) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("finite nonnegative labels matching calibration required")
        if self.kind not in ("linear", "boosting") or self.threads < 1:
            raise ValueError("valid correction kind and positive threads required")
        self.base_index_ = int(np.argmin(np.mean((p - y[:, None]) ** 2, axis=0)))
        self.input_columns_ = (p.shape[1], c.shape[1])
        if self.kind == "linear":
            self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        else:
            from libtree import GBDTRegressor

            self.model_ = GBDTRegressor(
                n_estimators=300,
                learning_rate=0.04,
                max_depth=4,
                max_bins=128,
                min_samples_leaf=100,
                reg_lambda=20.0,
                n_jobs=self.threads,
            )
        self.model_.fit(self.features(p, c), y - p[:, self.base_index_])
        return self

    def predict(self, prediction, context=None):
        check_is_fitted(self, "model_")
        p, c = self.arrays(prediction, context)
        if (p.shape[1], c.shape[1]) != self.input_columns_:
            raise ValueError("prediction/context columns must match calibration")
        return p[:, self.base_index_] + self.model_.predict(self.features(p, c))

    def close(self):
        if hasattr(getattr(self, "model_", None), "close"):
            self.model_.close()


def expand_candidates(frames, values, pools, output_dir, threads):
    """Augment each family ablation independently; never inspect audit labels."""
    stamp = time.perf_counter()
    statistics = ProductStatistics().fit(
        frames["train"], np.log1p(frames["train"].price.to_numpy())
    )
    context = {
        k: np.column_stack(
            [statistics.transform(frames[k]), TextAttributes().transform(frames[k])]
        )
        for k in WINDOWS
    }
    with (output_dir / "residual-context.pkl").open("wb") as stream:
        pickle.dump(statistics, stream)
    context_seconds = time.perf_counter() - stamp
    raw = {k: v for k, v in values.items() if k != "anchor"}
    pools["raw_only"] = list(pools["full"])
    metrics = {}
    for pool in ("full", "ridge_only", "without_lasso", "without_elasticnet"):
        # Corrected anchor deliberately excluded, even if its standalone score wins.
        names = [n for n in pools[pool] if n != "anchor"]
        p = {k: np.column_stack([raw[n][k] for n in names]) for k in WINDOWS}
        for kind in ("linear", "boosting"):
            name = f"residual_{pool}_{kind}"
            model = ResidualCorrection(kind, threads)
            start = time.perf_counter()
            try:
                model.fit(
                    p["calibration"],
                    np.log1p(frames["calibration"].price.to_numpy()),
                    context["calibration"],
                )
                fit_seconds = time.perf_counter() - start
                samples = []
                for _ in range(3):
                    start = time.perf_counter()
                    values[name] = {k: model.predict(p[k], context[k]) for k in WINDOWS}
                    samples.append(time.perf_counter() - start)
                metrics[name] = dict(
                    kind=kind,
                    experts=names,
                    base_expert=names[model.base_index_],
                    calibration_rows=len(p["calibration"]),
                    fit_seconds=fit_seconds,
                    predict_seconds=float(np.mean(samples)),
                    predict_seconds_samples=samples,
                    selection_rmsle=score(
                        np.log1p(frames["selection"].price.to_numpy()),
                        values[name]["selection"],
                    ),
                    features=model.features(
                        p["calibration"], context["calibration"]
                    ).shape[1],
                )
                with (output_dir / (name + ".pkl")).open("wb") as stream:
                    pickle.dump(model, stream)
                np.savez_compressed(
                    output_dir / (name + ".npz"),
                    **values[name],
                    **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
                )
                pools[pool].append(name)
                print("RESIDUAL_MODEL", name, metrics[name], flush=True)
            finally:
                model.close()
    return dict(
        models=metrics,
        context_seconds=context_seconds,
        retained_raw_experts=list(raw),
        standalone_quality_gate=False,
        correction_fit_window="calibration",
        anchor_used_as_calibration_feature=False,
        context_fit_window="train",
        settings=dict(
            rounds=300,
            depth=4,
            learning_rate=0.04,
            ridge_alpha=100.0,
            inference_repeats=3,
        ),
    )


def cached_study(args):
    """Reuse trusted base-only expert artifacts; no model or feature refit."""
    from benchmarks.mercari_top3 import finish

    start = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    frames = read_data(args)
    original_plan = json.loads((args.experts_dir / "plan.json").read_text())
    original_report = json.loads((args.experts_dir / "report.json").read_text())
    data_sha = {k: mercari.sha256(args.data_dir / (k + ".tsv.gz")) for k in frames}
    if original_plan["data_sha256"] != data_sha:
        raise ValueError(
            "cached experts must match the complete original data protocol"
        )
    metrics = original_report["models"]
    values, hashes = {}, {}
    paths = {
        "anchor": args.anchor,
        **{name: args.experts_dir / (name + ".npz") for name in metrics},
    }
    for name, path in paths.items():
        hashes[name] = mercari.sha256(path)
        with np.load(path) as archive:
            values[name] = {}
            for k in WINDOWS:
                if (
                    not np.array_equal(
                        archive[k + "_ids"], frames[k].train_id.to_numpy()
                    )
                    or archive[k].shape != (len(frames[k]),)
                    or not np.isfinite(archive[k]).all()
                ):
                    raise ValueError(
                        "finite cached predictions and identical ordered IDs required"
                    )
                values[name][k] = archive[k]
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        data_sha256=data_sha,
        candidate_sha256=hashes,
        original_plan=original_plan,
        cached_base_cost_excluded=True,
        source_sha256={
            name: mercari.sha256(Path(__file__).with_name(name))
            for name in (
                "mercari_residual.py",
                "mercari_top3.py",
                "mercari_multiview.py",
            )
        },
        confirmation_previously_inspected=True,
        minimum_selection_gain=args.minimum_gain,
    )
    dump(out / "plan.json", plan)
    return finish(args, frames, values, metrics, plan, start, 0.0, {})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "anchor", "experts-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.rows, args.seed, args.threads, args.minimum_gain = 0, 42, 4, 1e-5
    with threadpool_limits(args.threads):
        cached_study(args)


if __name__ == "__main__":
    from benchmarks.mercari_residual import main as entrypoint

    entrypoint()
