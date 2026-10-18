"""Budgeted CPU-only Mercari lexical/MLP study; never submits automatically.

Champion-inspired sparse views and heterogeneous objectives, not exhaustive
grid search. See mercari/CPU_STUDY.md for metric, lineage and budget contracts.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_cpu import CpuFeatures, RmsleBlend, make_features
from benchmarks.mercari_features import ListingFeatures, ResidualFeatures, make_ridge
from benchmarks.mercari_semantics import TextAttributes
from benchmarks.mercari_solver import SparseRidge

WINDOWS = ("calibration", "selection", "confirmation")


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def score(y, prediction):
    return float(np.sqrt(np.mean((np.maximum(prediction, 0.0) - y) ** 2)))


def read_data(args):
    frames, ids, groups = {}, set(), set()
    for name in ("train", *WINDOWS):
        part = pd.read_csv(args.data_dir / (name + ".tsv.gz"), sep="\t")
        if name == "train" and args.rows:
            part = part.sample(n=min(args.rows, len(part)), random_state=args.seed)
        if name == "confirmation" and "price" in part:
            raise ValueError("confirmation features must not contain prices")
        current = set(part.train_id)
        current_groups = set(mercari.listing_groups(part))
        if ids & current or groups & current_groups or len(current) != len(part):
            raise ValueError("IDs and listing groups must not overlap between windows")
        if name != "confirmation":
            mercari.rmsle(part.price, part.price)
        ids.update(current)
        groups.update(current_groups)
        frames[name] = part
    return frames


def reference_predictions(args, frames):
    if "price" in frames["confirmation"]:
        raise ValueError("confirmation features must not contain prices")
    archive = np.load(args.data_dir / "baseline.npz")
    predictions = {}
    for window in WINDOWS:
        if not np.array_equal(
            archive[window + "_ids"], frames[window].train_id.to_numpy()
        ):
            raise ValueError("cached reference IDs must match every window")
        predictions[window] = archive[window]
    return predictions


def run(args):
    start = time.perf_counter()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    frames = read_data(args)
    y = {
        k: np.log1p(frames[k].price.to_numpy())
        for k in ("train", "calibration", "selection")
    }
    predictions, metrics = {}, {}
    if args.reference:
        predictions["cached_cpu_baseline"] = reference_predictions(args, frames)
    for view in ("plain", "exact", "stemmed"):
        stamp = time.perf_counter()
        listing = (ListingFeatures() if view == "plain" else CpuFeatures(view)).fit(
            frames["train"]
        )
        prepared = {k: listing.transform(v) for k, v in frames.items()}
        extractor = (
            make_ridge(args.max_features, char_features=args.char_features)["features"]
            if view == "plain"
            else make_features(
                args.max_features, args.char_features, args.cross_features
            )
        )
        x = {"train": extractor.fit_transform(prepared["train"]).tocsr()}
        x.update({k: extractor.transform(prepared[k]).tocsr() for k in WINDOWS})
        preparation_seconds = time.perf_counter() - stamp
        with (out / (view + "-features.pkl")).open("wb") as stream:
            pickle.dump((listing, extractor), stream)
        print(
            "CPU_FEATURES",
            view,
            x["train"].shape,
            "seconds",
            preparation_seconds,
            flush=True,
        )
        trials = [("ridge3", "ridge", 3.0), ("ridge10", "ridge", 10.0)]
        if view != "plain":
            trials += [(loss, "mlp", loss) for loss in ("mse", "huber", "bins")]
        for index, (suffix, kind, setting) in enumerate(trials):
            stamp = time.perf_counter()
            name = view + "_" + suffix
            history = []
            if kind == "ridge":
                model = SparseRidge(alpha=setting, tol=3e-4, max_iter=500).fit(
                    x["train"], y["train"]
                )
            else:
                import tensorflow as tf
                from benchmarks.mercari_sparse import SparseExpert

                tf.config.set_visible_devices([], "GPU")
                model = SparseExpert(
                    width=args.width,
                    seed=args.seed + index,
                    loss=setting,
                    binary=setting != "mse",
                    optimizer="row_adam",
                )
                model.fit(
                    x["train"],
                    y["train"],
                    rates=[0.003, 0.003, 0.002][: args.epochs],
                    batches=[1024, 2048, 4096][: args.epochs],
                    validation=(x["selection"], y["selection"]),
                )
                history = model.history_
            fit_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            predictions[name] = {k: model.predict(x[k]) for k in WINDOWS}
            prediction_seconds = time.perf_counter() - stamp
            metrics[name] = dict(
                view=view,
                kind=kind,
                objective="log_mse" if kind == "ridge" else setting,
                fit_seconds=fit_seconds,
                prediction_seconds=prediction_seconds,
                history=history,
                selection_rmsle=score(y["selection"], predictions[name]["selection"]),
                features=x["train"].shape[1],
                feature_seconds=preparation_seconds,
            )
            with (out / (name + ".pkl")).open("wb") as stream:
                pickle.dump(model, stream)
            np.savez_compressed(
                out / (name + ".npz"),
                **predictions[name],
                **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
            )
            print("CPU_MODEL", name, metrics[name], flush=True)
            del model
            if kind == "mlp":
                tf.keras.backend.clear_session()
            gc.collect()
        del x, prepared, extractor, listing
        gc.collect()
    # The cached baseline's old corrector saw calibration labels, so that entire
    # branch is excluded from every new corrector's targets AND input features.
    base_names = [name for name in predictions if name != "cached_cpu_baseline"]
    base = {
        k: np.column_stack([predictions[n][k] for n in base_names]) for k in WINDOWS
    }
    mean = {k: base[k].mean(axis=1) for k in WINDOWS}
    encoder = ResidualFeatures().fit(frames["train"], y["train"])
    features = {
        k: np.ascontiguousarray(
            np.column_stack(
                [
                    encoder.transform(frames[k]),
                    TextAttributes().transform(frames[k]),
                    base[k],
                ]
            ),
            dtype=np.float32,
        )
        for k in WINDOWS
    }
    with (out / "residual-features.pkl").open("wb") as stream:
        pickle.dump(encoder, stream)
    from libtree import GBDTRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    common = dict(n_estimators=args.tree_rounds, learning_rate=0.04)
    models = {
        f"libtree{depth}": GBDTRegressor(
            **common,
            max_depth=depth,
            n_jobs=args.threads,
            max_bins=128,
            min_samples_leaf=100,
            reg_lambda=20.0,
        )
        for depth in (3, 5)
    }
    models.update(
        {
            f"lightgbm_{loss}": LGBMRegressor(
                **common,
                objective=loss,
                num_leaves=15,
                min_child_samples=100,
                reg_lambda=20.0,
                n_jobs=args.threads,
                random_state=args.seed,
                verbosity=-1,
            )
            for loss in ("regression_l2", "huber")
        }
    )
    models["xgboost4"] = XGBRegressor(
        **common,
        max_depth=4,
        min_child_weight=100,
        reg_lambda=20.0,
        n_jobs=args.threads,
        tree_method="hist",
        device="cpu",
        random_state=args.seed,
    )
    for name, model in models.items():
        stamp = time.perf_counter()
        model.fit(features["calibration"], y["calibration"] - mean["calibration"])
        fit_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        residuals = {k: model.predict(features[k]) for k in WINDOWS}
        for strength in (0.5, 1.0):
            predictions[name + "_" + str(strength)] = {
                k: mean[k] + strength * residuals[k] for k in WINDOWS
            }
        metrics[name] = dict(
            kind="residual",
            objective="huber" if name == "lightgbm_huber" else "log_mse",
            fit_seconds=fit_seconds,
            prediction_seconds=time.perf_counter() - stamp,
            calibration_features=features["calibration"].shape[1],
        )
        with (out / (name + ".pkl")).open("wb") as stream:
            pickle.dump(model, stream)
        if hasattr(model, "close"):
            model.close()
    names = list(predictions)
    matrices = {k: np.column_stack([predictions[n][k] for n in names]) for k in WINDOWS}
    stamp = time.perf_counter()
    selected = RmsleBlend().fit(matrices["selection"], y["selection"])
    no_bias = RmsleBlend(bias_bound=0.0).fit(matrices["selection"], y["selection"])
    base_only = RmsleBlend().fit(
        np.column_stack([predictions[n]["selection"] for n in base_names]),
        y["selection"],
    )
    abl = {}
    for label in ("mse", "huber", "bins", "engineered"):
        keep = [
            n
            for n in base_names
            if (
                n.startswith("plain_")
                if label == "engineered"
                else not n.endswith("_" + label)
            )
        ]
        local = RmsleBlend().fit(
            np.column_stack([predictions[n]["selection"] for n in keep]), y["selection"]
        )
        abl["without_" + label] = (keep, local)
    frozen = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        names=names,
        weights=selected.weights_.tolist(),
        intercept=selected.intercept_,
        penalty=selected.penalty,
        bias_bound=selected.bias_bound,
        seed=args.seed,
        threads=args.threads,
        selection_rmsle=score(y["selection"], selected.predict(matrices["selection"])),
        selection_chosen_before_confirmation=True,
        optimizer_seconds=time.perf_counter() - stamp,
        objective="mean((max(Pw+b,0)-log1p(price))**2) + penalty*(||w-anchor||**2+b**2)",
        ablations={
            key: dict(names=n, weights=m.weights_.tolist(), intercept=m.intercept_)
            for key, (n, m) in abl.items()
        },
        input_sha256={
            name: mercari.sha256(args.data_dir / (name + ".tsv.gz"))
            for name in ("train", *WINDOWS)
        },
        config={
            k: v for k, v in vars(args).items() if k not in ("data_dir", "output_dir")
        },
    )
    dump(out / "frozen.json", frozen)
    # No confirmation labels are accessed before the frozen choice is on disk.
    audit = np.load(args.data_dir / "evaluation.npz")
    if not np.array_equal(
        audit["confirmation_ids"], frames["confirmation"].train_id.to_numpy()
    ):
        raise ValueError("confirmation audit IDs must match frozen feature rows")
    confirmation_y = audit["confirmation"]
    comparisons = {
        n: dict(
            selection=score(y["selection"], matrices["selection"][:, i]),
            confirmation=score(confirmation_y, matrices["confirmation"][:, i]),
        )
        for i, n in enumerate(names)
    }
    ablations = {
        key: dict(
            selection=score(
                y["selection"],
                m.predict(np.column_stack([predictions[n]["selection"] for n in ns])),
            ),
            confirmation=score(
                confirmation_y,
                m.predict(
                    np.column_stack([predictions[n]["confirmation"] for n in ns])
                ),
            ),
        )
        for key, (ns, m) in abl.items()
    }
    final = {k: selected.predict(matrices[k]) for k in WINDOWS}
    np.savez_compressed(
        out / "ensemble.npz",
        **final,
        **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
    )
    report = dict(
        frozen=frozen,
        models=metrics,
        individual=comparisons,
        base_family_ablations=ablations,
        confirmation_rmsle=score(confirmation_y, final["confirmation"]),
        no_bias=dict(
            selection=score(y["selection"], no_bias.predict(matrices["selection"])),
            confirmation=score(
                confirmation_y, no_bias.predict(matrices["confirmation"])
            ),
        ),
        new_base_only=dict(
            selection=score(y["selection"], base_only.predict(base["selection"])),
            confirmation=score(confirmation_y, base_only.predict(base["confirmation"])),
        ),
        training_rows=len(frames["train"]),
        seconds=time.perf_counter() - start,
        gpu_used=False,
        external_pretrained_data=False,
        reference_cached_cost_excluded=args.reference,
        confirmation_previously_inspected=True,
        submission_status="not_submitted",
        timing_scope="Single incremental CPU run; cached reference fit/inference excluded. Epochs are successive training steps, not independent repetitions.",
    )
    dump(out / "report.json", report)
    print(
        "CPU_RESULT",
        report["confirmation_rmsle"],
        "seconds",
        report["seconds"],
        flush=True,
    )
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in [
        ("threads", 4),
        ("rows", 0),
        ("max-features", 75000),
        ("char-features", 40000),
        ("cross-features", 50000),
        ("width", 128),
        ("epochs", 3),
        ("tree-rounds", 300),
        ("seed", 42),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--max-seconds", type=float, default=1200.0)
    parser.add_argument("--no-reference", dest="reference", action="store_false")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        not np.isfinite(args.max_seconds)
        or args.max_seconds <= 0
        or args.rows < 0
        or not 1 <= args.epochs <= 3
        or min(
            args.threads,
            args.max_features,
            args.char_features,
            args.cross_features,
            args.width,
            args.tree_rounds,
        )
        < 1
    ):
        raise ValueError(
            "positive dimensions/budget, nonnegative rows and 1-3 epochs required"
        )
    if args.worker:
        with threadpool_limits(limits=args.threads):
            run(args)
        return 0
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        TF_CPP_MIN_LOG_LEVEL="3",
        TF_NUM_INTEROP_THREADS="1",
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
    ):
        env[name] = str(args.threads)
    try:
        return subprocess.run(
            [sys.executable, "-m", "benchmarks.mercari_cpu_study", *argv, "--worker"],
            env=env,
            timeout=args.max_seconds,
            check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dump(
            args.output_dir / "budget-exhausted.json",
            dict(
                max_seconds=args.max_seconds,
                submitted=False,
                status="incomplete_do_not_promote",
                reason="Hard wall-time budget reached; worker terminated",
            ),
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
