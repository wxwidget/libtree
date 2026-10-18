"""Refit the frozen full Mercari ensemble and stream competition predictions.

All price experts train inside one Kernel. Independent worker processes release
TensorFlow/Torch GPU allocations between text views. No fitted competition model
is uploaded. Stacking weights and epoch counts come from the earlier selection
window; this command never selects parameters using final calibration labels.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_ensemble import combine
from benchmarks.mercari_frontier import dump

GROUPS = (
    "baseline",
    "lsa",
    "sequence",
    "separate",
    "joint",
    "stemmed",
    "vectors",
    "transformer",
)


def training(args):
    return tuple(
        pd.read_csv(args.output_dir / (name + ".tsv.gz"), sep="\t")
        for name in ("base", "calibration")
    )


def test_frames(args):
    for frame in pd.read_csv(
        args.data_dir / args.test_file,
        sep="\t",
        dtype={"test_id": str},
        chunksize=args.chunk_size,
    ):
        if "price" in frame:
            raise ValueError("test price labels must not be supplied")
        yield frame


def prepare(args):
    from sklearn.model_selection import GroupShuffleSplit

    frame = pd.read_csv(args.data_dir / "train.tsv", sep="\t")
    if (
        "train_id" not in frame
        or frame.train_id.isna().any()
        or frame.train_id.duplicated().any()
    ):
        raise ValueError("unique training IDs required")
    mercari.rmsle(frame.price, frame.price)
    indices = next(
        GroupShuffleSplit(n_splits=1, test_size=1 / 17, random_state=44).split(
            frame, groups=mercari.listing_groups(frame)
        )
    )
    base, calibration = (frame.iloc[index] for index in indices)
    for name, part in [("base", base), ("calibration", calibration)]:
        part.to_csv(args.output_dir / (name + ".tsv.gz"), sep="\t", index=False)
    dump(
        args.output_dir / "protocol.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            splits={
                k: mercari.audit_split(p)
                for k, p in [("base", base), ("calibration", calibration)]
            },
            train_sha256=mercari.sha256(args.data_dir / "train.tsv"),
            no_selection_or_confirmation_labels=True,
            method="Grouped seed-44 final 16/17 base, 1/17 calibration; every official label has exactly one role",
        ),
    )


def export(args, name, calibration, predictor):
    """Write bounded batches with exact sample IDs, including all upstream columns."""
    out = args.output_dir / name
    out.mkdir(exist_ok=False)
    sample = pd.read_csv(args.data_dir / args.sample_file, dtype={"test_id": str})
    if (
        list(sample.columns) != ["test_id", "price"]
        or sample.test_id.isna().any()
        or sample.test_id.duplicated().any()
    ):
        raise ValueError("sample requires unique IDs and test_id,price columns")
    for window, expected, frames in [
        (
            "calibration",
            calibration.train_id.to_numpy(dtype=str),
            (
                calibration.iloc[i : i + args.chunk_size]
                for i in range(0, len(calibration), args.chunk_size)
            ),
        ),
        ("test", sample.test_id.to_numpy(dtype=str), test_frames(args)),
    ]:
        if not len(expected):
            raise ValueError("positive prediction row count required")
        directory = out / window
        directory.mkdir()
        np.save(directory / "ids.npy", expected)
        arrays, seen = {}, 0
        for frame in frames:
            actual = (
                frame["train_id" if window == "calibration" else "test_id"]
                .astype(str)
                .to_numpy()
            )
            if not np.array_equal(actual, expected[seen : seen + len(frame)]):
                raise ValueError("prediction IDs must match the complete sample order")
            values = predictor(frame)
            if not values or (arrays and set(arrays) != set(values)):
                raise ValueError(
                    "all prediction batches require the same nonempty components"
                )
            for key, value in values.items():
                value = np.asarray(value)
                if (
                    value.ndim not in (1, 2)
                    or len(value) != len(frame)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        "each component must be finite with one prediction row per ID"
                    )
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        directory / (key + ".npy"),
                        mode="w+",
                        dtype=value.dtype,
                        shape=(len(expected), *value.shape[1:]),
                    )
                if arrays[key].shape[1:] != value.shape[1:]:
                    raise ValueError("component feature shape changed")
                arrays[key][seen : seen + len(frame)] = value
            seen += len(frame)
            print(
                "PREDICTION_PROGRESS",
                name,
                window,
                seen,
                "/",
                len(expected),
                flush=True,
            )
        if seen != len(expected):
            raise ValueError("test/sample row count mismatch")
        for array in arrays.values():
            array.flush()


def read_stage(root, name, window):
    return {
        p.stem: np.load(p, mmap_mode="r")
        for p in (Path(root) / name / window).glob("*.npy")
    }


def stack(groups, config):
    reference = groups["baseline"]["ids"]
    if any(not np.array_equal(block["ids"], reference) for block in groups.values()):
        raise ValueError("all stage IDs must have exactly the same order")
    values = {
        name: value
        for block in groups.values()
        for name, value in block.items()
        if name != "ids" and value.ndim == 1
    }
    inner = config["inner"]
    columns = [values[name] for name in inner["names"]]
    x = np.ascontiguousarray(
        np.column_stack(
            [
                groups["baseline"]["metadata"],
                groups["lsa"]["latent"],
                groups["baseline"]["attributes"],
                *columns,
            ]
        ),
        dtype=np.float32,
    )
    if x.shape[1] != config["feature_count"]:
        raise ValueError("frozen calibration feature count mismatch")
    return values, x, combine(values, inner)


def baseline_stage(args, config):
    from benchmarks.mercari_advanced import fit_entry
    from benchmarks.mercari_semantics import TextAttributes

    base, calibration = training(args)
    entry, times = fit_entry(base, calibration, config["baseline"])
    try:

        def prediction(frame):
            p, x, correction = entry.parts(frame)
            raw = p @ entry.config["weights"]
            return dict(
                baseline=raw + entry.config["correction_strength"] * correction,
                old_uncorrected=raw,
                metadata=x,
                attributes=TextAttributes().transform(frame),
            )

        export(args, "baseline", calibration, prediction)
    finally:
        entry.close()
    return times


def lsa_stage(args, config):
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    from benchmarks.mercari_semantics import LatentText

    base, calibration = training(args)
    y = np.log1p(base.price.to_numpy())
    encoder = LatentText(**config["lsa"]).fit(base)
    x = encoder.transform(base)
    prefix = encoder.categorical_indices_[0]
    models = {
        f"lsa_ridge_{a:g}": Ridge(alpha=a).fit(x[:, :prefix], y) for a in (10.0, 100.0)
    }
    for leaves in (31, 63):
        models[f"lsa_lgbm_{leaves}"] = LGBMRegressor(
            n_estimators=config["lsa_rounds"],
            num_leaves=leaves,
            learning_rate=0.04,
            min_child_samples=80,
            reg_lambda=10.0,
            colsample_bytree=0.85,
            n_jobs=args.threads,
            random_state=42,
            verbosity=-1,
        ).fit(x, y, categorical_feature=encoder.categorical_indices_)
    del x, base
    gc.collect()

    def prediction(frame):
        z = encoder.transform(frame)
        return dict(
            latent=z[:, :prefix],
            **{
                n: m.predict(z[:, :prefix] if "ridge" in n else z)
                for n, m in models.items()
            },
        )

    export(args, "lsa", calibration, prediction)
    return dict(rounds=config["lsa_rounds"], components=encoder.components)


def sequence_stage(args, config):
    import tensorflow as tf
    from benchmarks.mercari_features import ListingFeatures
    from benchmarks.mercari_neural import Tokens, build_model, dataset, predict

    base, calibration = training(args)
    listing = ListingFeatures().fit(base)
    prepared = listing.transform(base)
    tokens = Tokens(
        max_words=config["max_features"],
        title_length=24,
        description_length=96,
        unknown_tokens=True,
    ).fit(prepared)
    encoded = tokens.transform(prepared)
    y = np.log1p(base.price.to_numpy()).astype(np.float32)
    mean = float(np.mean(y))
    models = {}
    histories = {}
    for name, architecture, width, loss in [
        ("cnn128_mse", "cnn", 128, "mse"),
        ("gru96_huber", "gru", 96, "huber"),
    ]:
        model = build_model(
            tokens,
            seed=42,
            width=config.get("tiny_width", width),
            loss=tf.keras.losses.Huber(delta=0.5) if loss == "huber" else loss,
            architecture=architecture,
        )
        history = []
        for rate in [0.003, 0.002, 0.001, 0.0005, 0.00025][: config["epochs"][name]]:
            model.optimizer.learning_rate.assign(rate)
            result = model.fit(
                dataset(encoded, y - mean, batch_size=1024, shuffle=True),
                epochs=1,
                verbose=2,
                shuffle=False,
            )
            history.append(float(result.history["loss"][0]))
        models[name] = tf.keras.Model(model.inputs, model.outputs)
        histories[name] = history
    del encoded, prepared, base, model
    gc.collect()

    def prediction(frame):
        z = tokens.transform(listing.transform(frame))
        return {n: mean + predict(m, z, batch_size=2048) for n, m in models.items()}

    export(args, "sequence", calibration, prediction)
    return dict(losses=histories)


def sparse_stage(args, config, view):
    from benchmarks.mercari_features import (
        ListingFeatures,
        JointListingFeatures,
        StemmedListingFeatures,
        make_ridge,
        RICH_TOKEN_PATTERN,
    )
    from benchmarks.mercari_sparse import SparseExpert, select_columns

    base, calibration = training(args)
    y = np.log1p(base.price.to_numpy()).astype(np.float32)
    listing = {
        "separate": ListingFeatures,
        "joint": JointListingFeatures,
        "stemmed": StemmedListingFeatures,
    }[view]().fit(base)
    prepared = listing.transform(base)
    extractor = make_ridge(
        max_features=config["max_features"], char_features=config["char_features"]
    )["features"]
    if view == "stemmed":
        extractor.transformers = [
            (n, t, "raw_name" if n == "characters" else columns)
            for n, t, columns in extractor.transformers
        ]
        for n, t, _ in extractor.transformers:
            if n in ("name", "description"):
                t.set_params(token_pattern=RICH_TOKEN_PATTERN)
    x = extractor.fit_transform(prepared)
    blocks = {k: v for k, v in extractor.output_indices_.items() if k != "remainder"}
    keep = select_columns(
        x,
        blocks,
        word_limit=config["max_features"],
        char_limit=config["char_features"],
        min_df=5,
    )
    x = x[:, keep].tocsr()
    models = {}
    histories = {}
    width = 256 if view == "separate" else 512
    prefix = ("stemmed_" if view == "stemmed" else "") + "sparse" + str(width)
    for suffix, binary, loss in [
        ("tfidf_mse", False, "mse"),
        ("binary_huber", True, "huber"),
        ("binary_bins", True, "bins"),
    ]:
        name = prefix + "_" + suffix
        model = SparseExpert(
            width=config.get("tiny_width", width),
            seed=117 if loss == "bins" else 77 if binary else 43,
            loss=loss,
            binary=binary,
            optimizer="row_adam",
        )
        epochs = config["epochs"][name]
        model.fit(
            x,
            y,
            rates=[0.003, 0.003, 0.002, 0.001][:epochs],
            batches=[1024, 2048, 4096, 8192][:epochs],
        )
        models[name] = model
        histories[name] = model.history_
    del x, base, prepared
    gc.collect()

    def prediction(frame):
        z = extractor.transform(listing.transform(frame))[:, keep]
        return {n: m.predict(z) for n, m in models.items()}

    export(args, view, calibration, prediction)
    return dict(histories=histories, features=len(keep))


def correctors(config):
    from libtree import GBDTRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge

    common = dict(n_estimators=config["tree_rounds"], learning_rate=0.04)
    return dict(
        cal_meta_ridge10=Ridge(alpha=10.0, positive=True),
        cal_meta_ridge100=Ridge(alpha=100.0, positive=True),
        cal_libtree3=GBDTRegressor(
            **common,
            max_depth=3,
            max_bins=128,
            min_samples_leaf=100,
            reg_lambda=20.0,
            n_jobs=config["threads"],
        ),
        cal_libtree5=GBDTRegressor(
            **common,
            max_depth=5,
            max_bins=128,
            min_samples_leaf=100,
            reg_lambda=20.0,
            n_jobs=config["threads"],
        ),
        cal_lightgbm=LGBMRegressor(
            **common,
            num_leaves=15,
            min_child_samples=200,
            reg_lambda=20.0,
            n_jobs=config["threads"],
            random_state=42,
            verbosity=-1,
        ),
        cal_xgboost=XGBRegressor(
            **common,
            max_depth=4,
            min_child_weight=100,
            reg_lambda=20.0,
            n_jobs=config["threads"],
            tree_method="hist",
            random_state=42,
        ),
    )


def finish(args, config):
    _, calibration = training(args)
    groups = {name: read_stage(args.output_dir, name, "calibration") for name in GROUPS}
    values, x, base = stack(groups, config)
    if not np.array_equal(
        groups["baseline"]["ids"], calibration.train_id.astype(str).to_numpy()
    ):
        raise ValueError("calibration label IDs must match prediction rows")
    target = np.log1p(calibration.price.to_numpy())
    y = target - base
    models = correctors(config)
    times = {}
    try:
        for name, model in models.items():
            start = time.perf_counter()
            model.fit(
                x[:, -len(config["inner"]["names"]) :], target
            ) if "meta_ridge" in name else model.fit(x, y)
            times[name] = time.perf_counter() - start
        test = {name: read_stage(args.output_dir, name, "test") for name in GROUPS}
        sample = pd.read_csv(args.data_dir / args.sample_file, dtype={"test_id": str})
        seen = 0
        out = args.output_dir / "submission.csv"
        for begin in range(0, len(sample), args.chunk_size):
            end = min(begin + args.chunk_size, len(sample))
            part = {
                name: {k: v[begin:end] for k, v in block.items()}
                for name, block in test.items()
            }
            if not np.array_equal(
                part["baseline"]["ids"], sample.test_id.iloc[begin:end].to_numpy()
            ):
                raise ValueError("final test IDs must match sample order")
            values, x, base = stack(part, config)
            for name, model in models.items():
                values[name] = (
                    model.predict(x[:, -len(config["inner"]["names"]) :])
                    if "meta_ridge" in name
                    else base + config["correction_strengths"][name] * model.predict(x)
                )
            prediction = mercari.prices(combine(values, config["outer"]))
            pd.DataFrame(
                dict(test_id=sample.test_id.iloc[begin:end], price=prediction)
            ).to_csv(
                out, index=False, mode="w" if begin == 0 else "a", header=begin == 0
            )
            seen += len(prediction)
        if seen != len(sample) or not seen:
            raise ValueError("positive complete submission required")
        report = dict(
            rows=seen,
            sha256=mercari.sha256(out),
            status="generated_not_submitted",
            correction_fit_seconds=times,
            outer=config["outer"],
            frozen_config_sha256=mercari.sha256(args.config),
            no_final_parameter_selection=True,
        )
        dump(args.output_dir / "submission-report.json", report)
        print("FINAL_SUBMISSION_REPORT", json.dumps(report), flush=True)
    finally:
        for model in models.values():
            if hasattr(model, "close"):
                model.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["prepare", *GROUPS, "finish", "all"], default="all"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--test-file", default="test_stg2.tsv")
    parser.add_argument("--sample-file", default="sample_submission_stg2.csv")
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--torch-python",
        default=sys.executable,
        help="Optional isolated interpreter for vector/transformer workers",
    )
    args = parser.parse_args(argv)
    if min(args.chunk_size, args.threads) < 1:
        parser.error("positive resource settings required")
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        arguments = list(sys.argv[1:] if argv is None else argv)
        if "--stage" in arguments:
            index = arguments.index("--stage")
            del arguments[index : index + 2]
        for stage in ["prepare", *GROUPS, "finish"]:
            executable = (
                args.torch_python
                if stage in ("vectors", "transformer")
                else sys.executable
            )
            subprocess.run(
                [
                    executable,
                    "-u",
                    "-m",
                    "benchmarks.mercari_final",
                    *arguments,
                    "--stage",
                    stage,
                ],
                check=True,
            )
        return
    start = time.perf_counter()
    if args.stage in ("baseline", "sequence", "separate", "joint", "stemmed"):
        import tensorflow as tf

        for device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(device, True)
    with threadpool_limits(args.threads):
        if args.stage == "prepare":
            prepare(args)
            result = {}
        elif args.stage == "finish":
            result = finish(args, config)
        elif args.stage in ("vectors", "transformer"):
            from benchmarks import mercari_final_torch

            result = getattr(mercari_final_torch, args.stage + "_stage")(args, config)
        elif args.stage in ("separate", "joint", "stemmed"):
            result = sparse_stage(args, config, args.stage)
        else:
            workers = {
                "baseline": baseline_stage,
                "lsa": lsa_stage,
                "sequence": sequence_stage,
            }
            result = workers[args.stage](args, config)
    report = dict(stage=args.stage, seconds=time.perf_counter() - start, result=result)
    dump(args.output_dir / (args.stage + "-report.json"), report)
    print("STAGE_REPORT", json.dumps(report), flush=True)


if __name__ == "__main__":
    from benchmarks.mercari_final import main as entrypoint

    entrypoint()
