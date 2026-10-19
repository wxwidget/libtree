"""Evaluate zero plus positive equal-frequency price classes on grouped windows."""

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import pickle
import resource
import time

import numpy as np
import pandas as pd

from benchmarks import mercari
from benchmarks.mercari_frontier import dump
from benchmarks.mercari_price_buckets import (
    PriceBuckets,
    aligned_proba,
    classification_metrics,
    fit_classifier,
)

WINDOWS = ("train", "calibration", "selection", "confirmation")
EVALUATION = WINDOWS[1:]
ROUNDS = 150
CLASSIFIERS = ("c5", "lightgbm", "c5_balanced", "lightgbm_balanced")


def prepare(args):
    start = time.perf_counter()
    cache = json.loads((args.context_dir / "manifest.json").read_text())
    identity = cache["identity"]
    data_hashes = {w: mercari.sha256(args.data_dir / (w + ".tsv.gz")) for w in WINDOWS}
    if (
        data_hashes != identity["data_sha256"]
        or not cache["query_duplicate_group_excluded"]
        or not cache["all_priors_leave_group_out"]
    ):
        raise ValueError(
            "base-only, leave-group-out context and matching data required"
        )
    for name, digest in identity["source_sha256"].items():
        if mercari.sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("context source identity mismatch")
    for name, version in identity["versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("context library version mismatch")
    with args.encoder.open("rb") as stream:
        encoder = pickle.load(stream)
    prefix = encoder.categorical_indices_[0]
    names = encoder.feature_names_[:prefix] + cache["feature_names"]
    arrays, targets, ids, seen, groups = {}, {}, {}, [], {}
    for window in WINDOWS:
        frame = pd.read_csv(args.data_dir / (window + ".tsv.gz"), sep="\t")
        if window == "confirmation" and "price" in frame:
            raise ValueError("audit prices must remain sealed")
        current = np.unique(mercari.listing_groups(frame))
        if any(np.intersect1d(current, previous).size for previous in seen):
            raise ValueError("identical listing groups cannot cross windows")
        seen.append(current)
        groups[window] = len(current)
        loaded = {}
        for name in ("context", "ids"):
            path = args.context_dir / (window + "-" + name + ".npy")
            if mercari.sha256(path) != cache["files"][path.name]:
                raise ValueError("context file hash mismatch")
            loaded[name] = np.load(path, mmap_mode="r")
        ids[window] = frame.train_id.to_numpy()
        if not np.array_equal(loaded["ids"], ids[window]):
            raise ValueError("context ordered IDs mismatch")
        raw = frame.drop(columns="price", errors="ignore")
        x = np.ascontiguousarray(
            np.column_stack([encoder.transform(raw)[:, :prefix], loaded["context"]]),
            dtype=np.float32,
        )
        if x.shape != (len(frame), len(names)) or not np.isfinite(x).all():
            raise ValueError("finite matching feature columns required")
        np.save(args.output_dir / (window + "-features.npy"), x)
        arrays[window] = np.load(
            args.output_dir / (window + "-features.npy"), mmap_mode="r"
        )
        if window != "confirmation":
            targets[window] = np.log1p(frame.price.to_numpy())
        print(
            "BUCKET_INPUTS", window, len(frame), time.perf_counter() - start, flush=True
        )
    bins = PriceBuckets().fit(targets["train"])
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        zero_is_separate=True,
        positive_bins_requested=9,
        actual_classes=bins.n_classes_,
        bins=bins.describe(),
        train_class_counts=bins.counts_.tolist(),
        log_cuts=bins.log_cuts_.tolist(),
        quantile_fit_window="train",
        quantile_method="inverted_cdf",
        intervals="right closed; equal prices stay together",
        data_sha256=data_hashes,
        source_sha256={
            n: mercari.sha256(Path(__file__).with_name(n))
            for n in (
                "mercari_price_buckets.py",
                "mercari_price_bucket_study.py",
                "mercari_semantics.py",
            )
        },
        encoder_sha256=mercari.sha256(args.encoder),
        context_manifest_sha256=mercari.sha256(args.context_dir / "manifest.json"),
        context_identity=identity,
        feature_names=names,
        categorical_ids_excluded=True,
        rows={w: len(ids[w]) for w in WINDOWS},
        distinct_groups=groups,
        rounds=ROUNDS,
        c5_max_rounds=min(20, ROUNDS),
        threads=4,
        calibration_role="early stopping by unweighted multiclass error",
        selection_role="choose accuracy and macro-F1 classifiers",
        regression_baselines=[
            dict(path=str(p), sha256=mercari.sha256(p)) for p in args.regression
        ],
        historical_selection_and_audit_reused=True,
        final_fusion_modified=False,
        versions={
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scikit-learn", "lightgbm")
        },
        feature_preparation_seconds=time.perf_counter() - start,
    )
    dump(args.output_dir / "plan.json", plan)
    return bins, arrays, targets, ids, plan


def release(model):
    if hasattr(model, "close"):
        model.close()


def run(args):
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bins, x, y, ids, plan = prepare(args)
    k = bins.n_classes_
    classes = {w: bins.transform(values) for w, values in y.items()}
    majority = int(bins.counts_.argmax())
    predictions = {
        "majority": {
            w: np.full(len(ids[w]), majority, dtype=np.int32) for w in EVALUATION
        }
    }
    metadata = {"majority": dict(train_majority_class=majority)}
    for i, path in enumerate(args.regression):
        name = "regression_" + str(i)
        with np.load(path) as source:
            values = {}
            for w in EVALUATION:
                if w == "calibration" and w not in source:
                    continue
                if (
                    not np.array_equal(source[w + "_ids"], ids[w])
                    or source[w].shape != (len(ids[w]),)
                    or not np.isfinite(source[w]).all()
                ):
                    raise ValueError(
                        "regression baseline shape and ordered IDs must match"
                    )
                values[w] = bins.transform(np.maximum(source[w].astype(float), 0.0))
        predictions[name] = values
        metadata[name] = dict(
            path=str(path),
            sha256=mercari.sha256(path),
            historical_regression_comparator=True,
        )
    for name in CLASSIFIERS:
        balanced = name.endswith("_balanced")
        engine = name.removesuffix("_balanced")
        tick = time.perf_counter()
        model = fit_classifier(
            engine,
            balanced,
            x["train"],
            classes["train"],
            x["calibration"],
            classes["calibration"],
            rounds=ROUNDS,
        )
        fit_seconds = time.perf_counter() - tick
        probabilities = {w: aligned_proba(model, x[w], k) for w in EVALUATION}
        predictions[name] = {
            w: p.argmax(1).astype(np.int32) for w, p in probabilities.items()
        }
        samples = []
        for _ in range(3):
            tick = time.perf_counter()
            aligned_proba(model, x["calibration"], k)
            samples.append(time.perf_counter() - tick)
        path = args.output_dir / (name + ".pkl")
        with path.open("wb") as stream:
            pickle.dump(model, stream)
        metadata[name] = dict(
            engine=engine,
            balanced=balanced,
            fit_seconds=fit_seconds,
            rounds=len(model.estimator_errors_)
            if hasattr(model, "estimator_errors_")
            else int(
                getattr(model, "best_iteration_", 0)
                or getattr(model, "n_estimators", 1)
            ),
            inference_rows=len(ids["calibration"]),
            inference_seconds=float(np.mean(samples)),
            inference_samples=samples,
            inference_max=max(samples),
            inference_std=float(np.std(samples)),
            model_sha256=mercari.sha256(path),
        )
        release(model)
        with path.open("rb") as stream:
            restored = pickle.load(stream)
        replay = aligned_proba(restored, x["selection"][:512], k)
        metadata[name]["serialized_replay_max_error"] = float(
            np.max(np.abs(replay - probabilities["selection"][:512]))
        )
        release(restored)
        np.savez_compressed(
            args.output_dir / (name + ".npz"),
            **{w + "_class": predictions[name][w] for w in EVALUATION},
            **{w + "_probability": p for w, p in probabilities.items()},
            **{w + "_ids": ids[w] for w in EVALUATION},
        )
        print("BUCKET_MODEL", name, fit_seconds, flush=True)
    metrics = {
        name: dict(
            **metadata[name],
            **{
                w: classification_metrics(classes[w], p[w], k)
                for w in ("calibration", "selection")
                if w in p
            },
        )
        for name, p in predictions.items()
    }
    chosen = max(CLASSIFIERS, key=lambda n: metrics[n]["selection"]["accuracy"])
    macro = max(CLASSIFIERS, key=lambda n: metrics[n]["selection"]["macro_f1"])
    with (args.output_dir / "bins.pkl").open("wb") as stream:
        pickle.dump(bins, stream)
    frozen = dict(
        chosen_accuracy=chosen,
        chosen_macro_f1=macro,
        plan=plan,
        models=copy.deepcopy(metrics),
        choices_frozen_before_audit=True,
    )
    dump(args.output_dir / "frozen.json", frozen)
    with np.load(args.data_dir / "evaluation.npz") as labels:
        if not np.array_equal(labels["confirmation_ids"], ids["confirmation"]):
            raise ValueError("audit ordered IDs must match")
        truth = bins.transform(labels["confirmation"])
    for name, p in predictions.items():
        metrics[name]["confirmation"] = classification_metrics(
            truth, p["confirmation"], k
        )
    report = dict(
        plan=plan,
        frozen=frozen,
        models=metrics,
        chosen_accuracy=chosen,
        chosen_macro_f1=macro,
        seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        submission_status="not_submitted",
        classification_accuracy_is_not_rmsle=True,
    )
    dump(args.output_dir / "report.json", report)
    print(
        "BUCKET_RESULT", chosen, metrics[chosen]["confirmation"]["accuracy"], flush=True
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--context-dir", type=Path, required=True)
    p.add_argument("--regression", type=Path, nargs="*", default=[])
    p.add_argument("--output-dir", type=Path, required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    main()
