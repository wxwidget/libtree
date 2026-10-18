"""CPU-only product-feature and recombination study; no automatic submission."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_cpu_study import (
    read_data,
    reference_predictions,
    dump,
    score,
    WINDOWS,
)
from benchmarks.mercari_multiview import (
    ProductStatistics,
    decompose,
    disagreement,
    select_strategy,
    fit_strategy,
    predict,
    snapshot,
)
from benchmarks.mercari_semantics import TextAttributes, paired_comparison


def load_small(folder, frames):
    values = {}
    for name in ("stemmed_huber", "stemmed_bins"):
        path = folder / (name + ".npz")
        archive = np.load(path)
        for window in WINDOWS:
            if not np.array_equal(
                archive[window + "_ids"], frames[window].train_id.to_numpy()
            ):
                raise ValueError("small-expert IDs must match all windows")
        values[name] = {window: archive[window] for window in WINDOWS}
    return values


def compact_strategy(protocol):
    result = {k: v for k, v in protocol.items() if not k.endswith("_ids")}
    for key in ("fit_ids", "validation_ids"):
        result[key + "_count"] = len(protocol[key])
        result[key + "_sha256"] = hashlib.sha256(
            np.asarray(protocol[key], dtype=np.int64).tobytes()
        ).hexdigest()
    return result


def run(args):
    if getattr(args, "refine_from", None) is not None:
        from benchmarks.mercari_multiscale import run as refine

        return refine(args)
    start = time.perf_counter()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parent
    source_sha = {
        name: mercari.sha256(source_root / name)
        for name in (
            "mercari_multiview.py",
            "mercari_multiview_study.py",
            "mercari_cpu.py",
        )
    }
    frames = read_data(args)
    reference = reference_predictions(args, frames)
    baseline_sha = mercari.sha256(args.baseline)
    protocol_path = args.data_dir / "protocol.json"
    if (
        protocol_path.exists()
        and json.loads(protocol_path.read_text())["baseline_sha256"] != baseline_sha
    ):
        raise ValueError("baseline artifact must match the prepared fitting protocol")
    with args.baseline.open("rb") as stream:
        entry = pickle.load(stream)
    components = {}
    component_seconds = {}
    try:
        for window in WINDOWS:
            stamp = time.perf_counter()
            components[window] = decompose(entry, frames[window])
            component_seconds[window] = time.perf_counter() - stamp
            print("CPU_COMPONENTS", window, component_seconds[window], flush=True)
    finally:
        entry.close()
    replay = max(
        float(np.max(np.abs(components[k]["reference"] - reference[k])))
        for k in WINDOWS
    )
    if replay > 5e-6:
        raise ValueError("component reconstruction does not match the frozen reference")
    predictions = {
        "reference": reference,
        "uncorrected": {k: components[k]["uncorrected"] for k in WINDOWS},
    }
    for i, name in enumerate(components["selection"]["names"]):
        predictions[name] = {k: components[k]["prediction"][:, i] for k in WINDOWS}
    small = {} if args.small_experts is None else load_small(args.small_experts, frames)
    predictions.update(small)
    y = {
        k: np.log1p(frames[k].price.to_numpy())
        for k in ("train", "calibration", "selection")
    }
    stamp = time.perf_counter()
    product = ProductStatistics().fit(frames["train"], y["train"])
    product_x = {k: product.transform(frames[k]) for k in WINDOWS}
    with (out / "product-statistics.pkl").open("wb") as stream:
        pickle.dump(product, stream)
    feature_seconds = time.perf_counter() - stamp
    print("CPU_PRODUCT_FEATURES", feature_seconds, flush=True)
    features = {view: {} for view in ("basic", "attributes", "product")}
    for k in WINDOWS:
        base = np.column_stack(
            [
                components[k]["metadata"],
                components[k]["prediction"],
                disagreement(components[k]["prediction"]),
            ]
        )
        attributes = np.column_stack([base, TextAttributes().transform(frames[k])])
        features["basic"][k] = np.ascontiguousarray(base, dtype=np.float32)
        features["attributes"][k] = np.ascontiguousarray(attributes, dtype=np.float32)
        features["product"][k] = np.ascontiguousarray(
            np.column_stack([attributes, product_x[k]]), dtype=np.float32
        )
    from libtree import GBDTRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    common = dict(n_estimators=args.tree_rounds, learning_rate=0.04)
    models = {}
    for view in ("basic", "attributes", "product"):
        for depth in (3,) if view == "attributes" else (3, 5):
            models[view + "_libtree" + str(depth)] = (
                view,
                GBDTRegressor(
                    **common,
                    max_depth=depth,
                    max_bins=128,
                    min_samples_leaf=100,
                    reg_lambda=20.0,
                    n_jobs=args.threads,
                ),
            )
    for view in ("basic", "product"):
        models[view + "_lightgbm"] = (
            view,
            LGBMRegressor(
                **common,
                num_leaves=31,
                min_child_samples=100,
                reg_lambda=20.0,
                n_jobs=args.threads,
                verbosity=-1,
                random_state=args.seed,
            ),
        )
    models["product_lightgbm_huber"] = (
        "product",
        LGBMRegressor(
            **common,
            num_leaves=31,
            objective="huber",
            min_child_samples=100,
            reg_lambda=20.0,
            n_jobs=args.threads,
            verbosity=-1,
            random_state=args.seed,
        ),
    )
    models["product_xgboost"] = (
        "product",
        XGBRegressor(
            **common,
            max_depth=4,
            min_child_weight=100,
            reg_lambda=20.0,
            n_jobs=args.threads,
            device="cpu",
            tree_method="hist",
            random_state=args.seed,
        ),
    )
    metrics = {}
    base = {k: components[k]["uncorrected"] for k in WINDOWS}
    for name, (view, model) in models.items():
        stamp = time.perf_counter()
        try:
            model.fit(
                features[view]["calibration"], y["calibration"] - base["calibration"]
            )
            fit_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            correction = {k: model.predict(features[view][k]) for k in WINDOWS}
            for strength in (0.5, 1.0):
                predictions[name + "_" + str(strength)] = {
                    k: base[k] + strength * correction[k] for k in WINDOWS
                }
            metrics[name] = dict(
                view=view,
                fit_seconds=fit_seconds,
                prediction_seconds=time.perf_counter() - stamp,
                features=features[view]["selection"].shape[1],
                selection_rmsle=score(
                    y["selection"], predictions[name + "_1.0"]["selection"]
                ),
            )
            with (out / (name + ".pkl")).open("wb") as stream:
                pickle.dump(model, stream)
        finally:
            if hasattr(model, "close"):
                model.close()
        print("CPU_CORRECTOR", name, metrics[name], flush=True)
    names = list(predictions)
    matrices = {k: np.column_stack([predictions[n][k] for n in names]) for k in WINDOWS}
    model, strategy = select_strategy(
        matrices["selection"], y["selection"], frames["selection"]
    )
    kind = strategy["chosen"]
    masks = dict(
        without_product_statistics=np.array(
            [not n.startswith("product_") for n in names]
        ),
        without_all_new_features=np.array(
            [not n.startswith(("product_", "attributes_")) for n in names]
        ),
        without_small_experts=np.array([n not in small for n in names]),
    )
    ablations = {
        key: fit_strategy(
            kind, matrices["selection"][:, mask], y["selection"], frames["selection"]
        )
        for key, mask in masks.items()
    }
    frozen = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        names=names,
        model=snapshot(model),
        strategy=compact_strategy(strategy),
        selected_before_confirmation=True,
        selection_rmsle=score(
            y["selection"], predict(model, matrices["selection"], frames["selection"])
        ),
        ablations={
            key: dict(names=list(np.asarray(names)[masks[key]]), model=snapshot(local))
            for key, local in ablations.items()
        },
        source_sha256=source_sha,
        baseline_sha256=baseline_sha,
        data_sha256={
            k: mercari.sha256(args.data_dir / (k + ".tsv.gz"))
            for k in ("train", *WINDOWS)
        },
        small_expert_sha256={
            name: mercari.sha256(args.small_experts / (name + ".npz")) for name in small
        },
        config=dict(
            tree_rounds=args.tree_rounds,
            threads=args.threads,
            seed=args.seed,
            product_smoothing=20.0,
            affine_penalty=0.001,
            local_penalty=0.01,
            local_min_rows=2000,
            mass_bounds=[0.8, 1.2],
            bias_bound=0.25,
        ),
    )
    dump(out / "frozen.json", frozen)
    np.savez_compressed(
        out / "pool.npz",
        **matrices,
        **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
    )
    audit = np.load(args.data_dir / "evaluation.npz")
    if not np.array_equal(
        audit["confirmation_ids"], frames["confirmation"].train_id.to_numpy()
    ):
        raise ValueError("confirmation audit IDs must match frozen rows")
    truth = audit["confirmation"]
    final = {k: predict(model, matrices[k], frames[k]) for k in WINDOWS}
    report = dict(
        frozen=frozen,
        models=metrics,
        individual={
            name: dict(
                selection=score(y["selection"], matrices["selection"][:, i]),
                confirmation=score(truth, matrices["confirmation"][:, i]),
            )
            for i, name in enumerate(names)
        },
        feature_ablations={
            key: dict(
                selection=score(
                    y["selection"],
                    predict(
                        local, matrices["selection"][:, masks[key]], frames["selection"]
                    ),
                ),
                confirmation=score(
                    truth,
                    predict(
                        local,
                        matrices["confirmation"][:, masks[key]],
                        frames["confirmation"],
                    ),
                ),
            )
            for key, local in ablations.items()
        },
        confirmation_rmsle=score(truth, final["confirmation"]),
        paired_comparison=paired_comparison(
            truth, reference["confirmation"], final["confirmation"], repeats=300
        ),
        feature_names=product.feature_names_,
        training_rows=len(frames["train"]),
        calibration_rows=len(frames["calibration"]),
        component_seconds=component_seconds,
        product_feature_seconds=feature_seconds,
        component_replay_max_error=replay,
        seconds=time.perf_counter() - start,
        gpu_used=False,
        confirmation_previously_inspected=True,
        submission_status="not_submitted",
        timing_scope="Incremental inference, new feature statistics, calibration and blending; original fitted expert training excluded",
    )
    np.savez_compressed(
        out / "ensemble.npz",
        **final,
        **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
    )
    dump(out / "report.json", report)
    print(
        "CPU_MULTIVIEW_RESULT",
        report["confirmation_rmsle"],
        "seconds",
        report["seconds"],
        flush=True,
    )
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "baseline", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--small-experts", type=Path)
    parser.add_argument(
        "--refine-from",
        type=Path,
        help="Reuse a completed multiview pool and base product statistics",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tree-rounds", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.rows, args.seed = 0, 42
    if (
        min(args.threads, args.tree_rounds) < 1
        or not np.isfinite(args.max_seconds)
        or args.max_seconds <= 0
    ):
        raise ValueError("positive threads, rounds and time budget required")
    if args.worker:
        from numba import set_num_threads

        set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            run(args)
        return 0
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        TF_CPP_MIN_LOG_LEVEL="3",
        TF_NUM_INTEROP_THREADS="1",
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[key] = str(args.threads)
    try:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.mercari_multiview_study",
                *argv,
                "--worker",
            ],
            env=env,
            timeout=args.max_seconds,
            check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dump(
            args.output_dir / "budget-exhausted.json",
            dict(
                status="incomplete_do_not_promote",
                submitted=False,
                max_seconds=args.max_seconds,
            ),
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
