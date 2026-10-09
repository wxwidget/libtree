"""Paired native before/after timing on identical prepared benchmark inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["titanic", "insurance", "pima", "telco", "wine", "friedman_20k"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--candidate", type=Path, default=ROOT / "build/native_benchmark")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/optimization-ab-results.json")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repeats required")
    binaries = {"baseline": args.baseline.resolve(), "optimized": args.candidate.resolve()}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    rng = random.Random(42)
    results = []
    for name in args.datasets:
        data = ROOT / f"benchmarks/runs/{name}.bin"
        samples = {engine: [] for engine in binaries}
        predictions = {}
        order_log = []
        for repeat in range(args.repeats):
            order = list(binaries)
            rng.shuffle(order)
            order_log.append(order)
            for engine in order:
                output = ROOT / f"benchmarks/runs/ab-{name}-{engine}.f32"
                completed = subprocess.run(
                    [str(binaries[engine]), str(data), str(output),
                     str(int(name in ("titanic", "pima", "telco"))), "libtree"],
                    check=True, capture_output=True, text=True, env=env, timeout=120)
                measurement = json.loads(completed.stdout)
                if not all(np.isfinite(measurement[key]) and measurement[key] > 0
                           for key in ("fit_seconds", "predict_seconds", "peak_rss_kib")):
                    raise RuntimeError(f"invalid measurements: {name}/{engine}")
                samples[engine].append(measurement)
                prediction = np.fromfile(output, dtype=np.float32)
                if prediction.size == 0 or not np.isfinite(prediction).all():
                    raise RuntimeError(f"invalid predictions: {name}/{engine}")
                if engine in predictions:
                    np.testing.assert_array_equal(prediction, predictions[engine])
                predictions[engine] = prediction
            np.testing.assert_array_equal(predictions["baseline"], predictions["optimized"])
        row = {"dataset": name, "prepared_input_sha256": digest(data),
               "predictions_identical": True, "order": order_log, "raw_runs": samples}
        for engine, runs in samples.items():
            for key in ("fit_seconds", "predict_seconds", "peak_rss_kib"):
                row[f"{engine}_{key}"] = median(run[key] for run in runs)
        row["fit_speedup"] = row["baseline_fit_seconds"] / row["optimized_fit_seconds"]
        row["predict_speedup"] = row["baseline_predict_seconds"] / row["optimized_predict_seconds"]
        results.append(row)
        print(f"{name}: fit {row['fit_speedup']:.2f}x, predict {row['predict_speedup']:.2f}x; exact parity", flush=True)
    report = {"metadata": {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_ref": args.baseline_ref, "repeats": args.repeats, "threads": 1,
        "parameters": {"trees": 100, "depth": 4, "bins": 64, "min_leaf": 5, "rate": .1, "l2": 1, "min_gain": 0},
        "ordering": "Randomized paired order, seed 42; independent process per measurement",
        "binary_sha256": {engine: digest(path) for engine, path in binaries.items()},
        "source_sha256": {str(path): digest(ROOT / path) for path in (
            "src/gbdt.cc", "src/c_api.cc", "src/model.cc", "include/libtree/gbdt.h", "benchmarks/ab.py")},
        "cpu": next(line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name'))}, "results": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
