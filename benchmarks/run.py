"""Reproducible equal-thread C++ and Python comparisons; no Kaggle credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import struct
import subprocess
import sys
import time
import urllib.request

# Set before loading numerical libraries, including in fresh worker processes.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes, make_friedman1
from sklearn.metrics import (mean_squared_error, roc_auc_score, average_precision_score,
                             accuracy_score, log_loss, mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
from libtree import GBDTClassifier, GBDTRegressor

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "benchmarks/cache"
RUNS = ROOT / "benchmarks/runs"
SOURCES = {
    "pima": ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"),
    "telco": ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"),
    "wine": ("https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv", "4678927bac9ff54ac7431a10979a3a9b5ba014d20e14196b047073f2ceb75f4c"),
    "titanic": ("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/train.csv", "14769fb1850e2d26d8e6db0ee49c213878040432827e39b13caaa15603c6598f"),
    "insurance": ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv", "505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281"),
    "bank_marketing": ("https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv", "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"),
}


def peak_rss_kib():
    # VmHWM belongs to the current exec's address space; ru_maxrss can include
    # high-water usage inherited from the fork before exec on Linux.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise RuntimeError("Linux VmHWM unavailable")


def worker(args):
    data = np.load(args.worker)
    x, y, test = data["x"], data["y"], data["test"]
    binary = bool(data["binary"])
    if args.engine == "libtree":
        cls = GBDTClassifier if binary else GBDTRegressor
        model = cls(n_estimators=args.trees, max_depth=args.depth, max_bins=args.bins,
                    min_samples_leaf=args.min_leaf, learning_rate=args.rate,
                    reg_lambda=args.l2, min_gain=args.min_gain, n_jobs=args.threads)
    elif args.engine == "lightgbm":
        cls = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
        model = cls(n_estimators=args.trees, max_depth=args.depth, num_leaves=1 << args.depth,
                    max_bin=args.bins, min_child_samples=args.min_leaf, min_data_in_bin=1,
                    learning_rate=args.rate, reg_lambda=args.l2, min_split_gain=args.min_gain,
                    n_jobs=args.threads, random_state=42, data_random_seed=42, deterministic=True,
                    feature_pre_filter=False, force_col_wise=True, verbosity=-1)
    else:
        cls = xgb.XGBClassifier if binary else xgb.XGBRegressor
        model = cls(n_estimators=args.trees, max_depth=args.depth, max_bin=args.bins, learning_rate=args.rate,
                    reg_lambda=args.l2, gamma=args.min_gain, tree_method="hist", n_jobs=args.threads, random_state=42)
    start = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict_proba(test)[:, 1] if binary else model.predict(test)
    predict_seconds = time.perf_counter() - start
    train_pred = model.predict_proba(x)[:, 1] if binary else model.predict(x)
    np.asarray(pred, dtype=np.float32).tofile(args.predictions)
    np.asarray(train_pred, dtype=np.float32).tofile(str(args.predictions) + ".train")
    print(json.dumps({"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                      "peak_rss_kib": peak_rss_kib()}))


def source(name):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.csv"
    url, expected = SOURCES[name]
    if not path.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            contents = response.read()
        path.write_bytes(contents)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise RuntimeError(f"checksum mismatch: {name}")
    return pd.read_csv(path, header=None if name == "pima" else "infer",
                       sep=";" if name == "bank_marketing" else ","), {
        "url": url, "sha256": digest, "bytes": path.stat().st_size,
        "kaggle_url": {
            "pima": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
            "telco": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
            "wine": "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
            "titanic": "https://www.kaggle.com/competitions/titanic",
            "insurance": "https://www.kaggle.com/datasets/mirichoi0218/insurance",
            "bank_marketing": "https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing",
        }[name], "download_route": "public_mirror",
    }


def datasets(suite="all"):
    if suite in ("all", "original", "builtin"):
        yield from builtin_datasets()
    if suite == "builtin":
        return
    for dataset in kaggle_datasets():
        if suite != "original" or dataset[0] in ("titanic", "insurance"):
            yield dataset


def builtin_datasets():
    for name, loader in (("diabetes", load_diabetes), ("breast_cancer", load_breast_cancer)):
        x, y = loader(return_X_y=True)
        yield name, x, y, name == "breast_cancer", {"source": f"sklearn.datasets.{loader.__name__}"}
    for rows in (20000, 100000):
        x, y = make_friedman1(n_samples=rows, n_features=20, noise=1, random_state=42)
        yield f"friedman_{rows // 1000}k", x, y, False, {"source": "sklearn.make_friedman1", "seed": 42}


def kaggle_datasets():
    frame, provenance = source("bank_marketing")
    provenance["preprocessing"] = (
        "bank subscription target; drop call duration because it is only known "
        "after the outcome; remaining categoricals encoded from training fold only"
    )
    yield "bank_marketing", frame.drop(columns=["y", "duration"]), (
        frame["y"] == "yes").to_numpy(dtype=np.float32), True, provenance
    frame, provenance = source("titanic")
    x = frame[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    provenance["preprocessing"] = "7 selected features; remove ID/name/ticket/cabin"
    yield "titanic", x, frame["Survived"].to_numpy(), True, provenance
    frame, provenance = source("insurance")
    provenance["preprocessing"] = "all 6 non-target features; target charges unchanged"
    yield "insurance", frame.drop(columns="charges"), frame["charges"].to_numpy(), False, provenance
    frame, provenance = source("pima")
    frame.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
                     "bmi", "diabetes_pedigree", "age", "outcome"]
    x = frame.drop(columns="outcome").copy()
    columns = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
    x[columns] = x[columns].replace(0, np.nan)
    provenance["preprocessing"] = "physiologically invalid zero in glucose/BP/skin/insulin/BMI mapped to NaN; pregnancies zero retained"
    yield "pima", x, frame["outcome"].to_numpy(), True, provenance
    frame, provenance = source("telco")
    x = frame.drop(columns=["customerID", "Churn"]).copy()
    x["TotalCharges"] = pd.to_numeric(x["TotalCharges"], errors="coerce")
    provenance["preprocessing"] = "remove customerID; 11 blank TotalCharges mapped to NaN; churn Yes=1"
    yield "telco", x, (frame["Churn"] == "Yes").to_numpy(dtype=np.float32), True, provenance
    frame, provenance = source("wine")
    original_rows = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    provenance["preprocessing"] = "drop 240 identical rows before splitting to avoid duplicate train/test leakage; quality regression"
    provenance["original_rows"] = original_rows
    provenance["duplicates_removed"] = original_rows - len(frame)
    yield "wine", frame.drop(columns="quality"), frame["quality"].to_numpy(), False, provenance


def prepare(x, y, binary, seed=42):
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=.25, random_state=seed, stratify=y if binary else None)
    if isinstance(x, pd.DataFrame):
        categorical = x.select_dtypes(include=["object", "string"]).columns
        numeric = x.columns.difference(categorical)
        if len(categorical):
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            train_cat = encoder.fit_transform(train_x[categorical].fillna("missing"))
            test_cat = encoder.transform(test_x[categorical].fillna("missing"))
        else:
            train_cat = np.empty((len(train_x), 0))
            test_cat = np.empty((len(test_x), 0))
        train_x = np.column_stack((train_x[numeric], train_cat))
        test_x = np.column_stack((test_x[numeric], test_cat))
    return (np.ascontiguousarray(train_x, dtype=np.float32),
            np.ascontiguousarray(test_x, dtype=np.float32),
            np.asarray(train_y, dtype=np.float32), np.asarray(test_y, dtype=np.float32))


def measurement_order(dataset, seed, threads, engines):
    """Return a reproducible, seed-specific order for engine/interface jobs."""
    order = [(language, engine) for language in ("cpp", "python")
             for engine in engines]
    order_seed = (seed * 1_000_003 + threads * 97 +
                  sum((index + 1) * ord(char) for index, char in enumerate(dataset)))
    random.Random(order_seed).shuffle(order)
    return order


def main(args):
    RUNS.mkdir(exist_ok=True)
    results, sources = [], {}
    libraries = {"libtree": "-", "xgboost": str(Path(xgb.__file__).parent / "lib/libxgboost.so"),
                 "lightgbm": str(Path(lgb.__file__).parent / "lib/lib_lightgbm.so")}
    engines = args.engines
    parameters = ["--trees", str(args.trees), "--depth", str(args.depth), "--bins", str(args.bins),
                  "--min-leaf", str(args.min_leaf), "--rate", str(args.rate), "--l2", str(args.l2),
                  "--min-gain", str(args.min_gain), "--threads", str(args.threads)]
    for name, x, y, binary, provenance in datasets(args.suite):
        if args.datasets and name not in args.datasets:
            continue
        provenance["rows_after_cleaning"] = len(x)
        provenance["raw_feature_count"] = x.shape[1]
        provenance["feature_missing_values"] = int(x.isna().sum().sum()) if isinstance(x, pd.DataFrame) else int(np.isnan(x).sum())
        provenance["target_missing_values"] = int(np.isnan(y).sum())
        if binary:
            provenance["positive_fraction"] = float(np.mean(y))
        sources[name] = provenance
        x, test, y, truth = prepare(x, y, binary, args.seed)
        digest = hashlib.sha256()
        for array in (x, y, test, truth):
            digest.update(array.astype("<f4").tobytes())
        provenance["prepared_split_sha256"] = digest.hexdigest()
        npz, native = RUNS / f"{name}.npz", RUNS / f"{name}.bin"
        np.savez(npz, x=x, y=y, test=test, binary=binary)
        with native.open("wb") as stream:
            stream.write(struct.pack("<QQQ", len(x), len(test), x.shape[1]))
            for array in (x, y, test):
                stream.write(array.astype("<f4").tobytes())
        predictions = {}
        training_predictions = {}
        current_order = measurement_order(name, args.seed, args.threads, engines)
        for language, engine in current_order:
                runs = []
                output = RUNS / f"{name}-{language}-{engine}.f32"
                for repetition in range(args.repeats):
                    if language == "cpp":
                        command = [str(ROOT / "build/native_benchmark"), str(native), str(output),
                                   str(int(binary)), engine, libraries[engine], str(args.trees), str(args.depth),
                                   str(args.bins), str(args.min_leaf), str(args.rate), str(args.l2), str(args.min_gain), str(args.threads)]
                    else:
                        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(npz),
                                   "--engine", engine, "--predictions", str(output), *parameters]
                    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
                    runs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
                    pred = np.fromfile(output, dtype=np.float32)
                    train_pred = np.fromfile(str(output) + ".train", dtype=np.float32)
                    if pred.shape != truth.shape or not np.isfinite(pred).all():
                        raise RuntimeError("invalid benchmark predictions")
                    if train_pred.shape != y.shape or not np.isfinite(train_pred).all():
                        raise RuntimeError("invalid training predictions")
                    if repetition:
                        try:
                            np.testing.assert_allclose(
                                pred, predictions[(language, engine)],
                                atol=1e-6, rtol=1e-6)
                            np.testing.assert_allclose(
                                train_pred,
                                training_predictions[(language, engine)],
                                atol=1e-6, rtol=1e-6)
                        except AssertionError as error:
                            raise RuntimeError(
                                "nondeterministic prediction: "
                                f"dataset={name}, language={language}, "
                                f"engine={engine}, threads={args.threads}, "
                                f"seed={args.seed}, repetition={repetition + 1}") \
                                from error
                    predictions[(language, engine)] = pred
                    training_predictions[(language, engine)] = train_pred
                metric = float(roc_auc_score(truth, pred) if binary else
                               np.sqrt(mean_squared_error(truth, pred)))
                baseline = .5 if binary else float(np.sqrt(mean_squared_error(truth, np.full_like(truth, y.mean()))))
                if (binary and metric <= baseline) or (not binary and metric >= baseline):
                    raise RuntimeError(f"model does not beat baseline: {name}/{engine}")
                record = {"dataset": name, "language": language, "engine": engine,
                          "train_rows": len(x), "test_rows": len(test), "features": x.shape[1],
                          "metric_name": "roc_auc" if binary else "rmse", "metric": metric,
                          "baseline": baseline, "raw_runs": runs, "seed": args.seed,
                          "checks": {"beats_baseline": True, "finite_predictions": True,
                                     "deterministic_repeats": True}}
                if binary:
                    probability = np.clip(pred.astype(np.float64), 1e-7, 1-1e-7)
                    train_probability = np.clip(
                        train_pred.astype(np.float64), 1e-7, 1-1e-7)
                    record["metrics"] = {
                        "roc_auc": metric, "average_precision": float(average_precision_score(truth, pred)),
                        "accuracy": float(accuracy_score(truth, pred >= .5)),
                        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
                        "positive_fraction": float(truth.mean()),
                    }
                    record["train_metrics"] = {
                        "roc_auc": float(roc_auc_score(y, train_pred)),
                        "average_precision": float(average_precision_score(y, train_pred)),
                        "accuracy": float(accuracy_score(y, train_pred >= .5)),
                        "log_loss": float(log_loss(y, train_probability, labels=[0, 1])),
                    }
                else:
                    record["metrics"] = {"rmse": metric, "mae": float(mean_absolute_error(truth, pred)),
                                         "r2": float(r2_score(truth, pred))}
                    record["train_metrics"] = {
                        "rmse": float(np.sqrt(mean_squared_error(y, train_pred))),
                        "mae": float(mean_absolute_error(y, train_pred)),
                        "r2": float(r2_score(y, train_pred)),
                    }
                for key in ("fit_seconds", "predict_seconds", "peak_rss_kib"):
                    record[key] = float(np.median([run[key] for run in runs]))
                results.append(record)
                print(f"{name:15} {language:6} {engine:8} {metric:.4f} fit={record['fit_seconds']:.4f}s", flush=True)
        for engine in engines:
            np.testing.assert_allclose(predictions[("cpp", engine)], predictions[("python", engine)],
                                       rtol=2e-5, atol=2e-3)
            np.testing.assert_allclose(training_predictions[("cpp", engine)],
                                       training_predictions[("python", engine)],
                                       rtol=2e-5, atol=2e-3)
        for record in results[-2 * len(engines):]:
            record["checks"]["cpp_python_parity"] = True
    report = {"metadata": {"platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
              "xgboost": xgb.__version__, "lightgbm": lgb.__version__, "engines": engines, "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
              "repeats": args.repeats, "seed": args.seed, "suite": args.suite, "threads": args.threads,
              "available_cpus": len(os.sched_getaffinity(0)),
              "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
              "memory_measurement": "Linux /proc/self/status VmHWM after prediction (KiB); current exec address space",
              "parameters": {"trees": args.trees, "depth": args.depth, "bins": args.bins,
                             "min_leaf": args.min_leaf, "rate": args.rate, "l2": args.l2, "min_gain": args.min_gain},
              "cpu": next((line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
              "source_sha256": {str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                                for path in ("src/gbdt.cc", "src/parallel.h", "src/c_api.cc", "src/model.cc", "src/cli.cc", "include/libtree/gbdt.h", "python/libtree/__init__.py", "benchmarks/native.cc", "benchmarks/run.py", "CMakeLists.txt", "setup.py")},
              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              "sources": sources, "results": results}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=str(ROOT / "benchmarks/results.json"))
    parser.add_argument("--suite", choices=["all", "kaggle", "original", "builtin"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", choices=["titanic", "insurance", "pima", "telco", "wine", "bank_marketing",
                                                         "diabetes", "breast_cancer", "friedman_20k", "friedman_100k"])
    parser.add_argument("--worker")
    parser.add_argument("--engine", choices=["libtree", "xgboost", "lightgbm"])
    parser.add_argument("--engines", nargs="+", choices=["libtree", "xgboost", "lightgbm"], default=["libtree", "xgboost", "lightgbm"])
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--rate", type=float, default=.1)
    parser.add_argument("--l2", type=float, default=1.)
    parser.add_argument("--min-gain", type=float, default=0.)
    parser.add_argument("--predictions")
    args = parser.parse_args()
    if not 1 <= args.threads <= 256:
        parser.error("threads must be 1..256")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not 1 <= args.depth <= 20 or len(args.engines) != len(set(args.engines)):
        parser.error("benchmark depth must be 1..20 and engines must be distinct")
    if args.trees < 1 or not 2 <= args.bins <= 65535 or args.min_leaf < 1 or not 0 < args.rate <= 1 or not np.isfinite([args.rate, args.l2, args.min_gain]).all() or args.l2 < 0 or args.min_gain < 0:
        parser.error("invalid benchmark parameters")
    worker(args) if args.worker else main(args)
