"""Jointly fine-tune a pretrained text body with observable product/comparable cues.

Deep averaging/CNN text features: Iyyer et al. (2015), P15-1162; Kim (2014),
D14-1181, ACL Anthology. Conditional log-mean target: Gneiting (2011),
doi:10.1198/jasa.2011.r10138. The zero-initialized context residual preserves the
base forecast initially, while ALL original text weights remain trainable.
"""

import argparse
import gc
import copy
import json
from pathlib import Path
import resource
import time

import numpy as np
import tensorflow as tf

from benchmarks import mercari
from benchmarks.mercari_frontier import dump, score
from benchmarks.mercari_neural import MaskedPool, predict
from benchmarks.mercari_retrieval_inputs import IDENTITY_BUCKETS, WINDOWS, prepare, load
from benchmarks.mercari_market_study import bucket_metrics
from benchmarks.mercari_semantics import paired_comparison

EPOCHS = 2
BATCH = 4096
SEEDS = (42, 43)
MIN_BUCKET_ROWS = 200
GUARD_FACTOR = 1.005


def build_model(base, seed=42):
    tf.keras.utils.set_random_seed(seed)
    base.trainable = True
    inputs = {
        v.name.split(":")[0]: tf.keras.Input(
            shape=v.shape[1:], dtype=v.dtype, name=v.name.split(":")[0]
        )
        for v in base.inputs
    }
    body = tf.keras.Model(
        base.inputs, [base.outputs[0], base.layers[-1].input], name="text_trunk"
    )
    raw, latent = body([inputs[v.name.split(":")[0]] for v in base.inputs])
    inputs["identity"] = tf.keras.Input((16,), dtype="int32", name="identity")
    inputs["context"] = tf.keras.Input((28,), dtype="float32", name="context")
    identity = MaskedPool()(
        tf.keras.layers.Embedding(IDENTITY_BUCKETS + 1, 16, mask_zero=True)(
            inputs["identity"]
        )
    )
    hidden = tf.keras.layers.Concatenate()([latent, identity, inputs["context"]])
    hidden = tf.keras.layers.Dense(64, activation="relu")(hidden)
    hidden = tf.keras.layers.Dropout(0.1)(hidden)
    residual = tf.keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros"
    )(hidden)
    return tf.keras.Model(inputs, tf.keras.layers.Add()([raw, residual]))


def view_inputs(encoded, mean, scale, variant):
    if variant not in ("control", "market", "retrieval"):
        raise ValueError("known text/retrieval view required")
    result = {
        k: v for k, v in encoded.items() if k not in ("ids", "y", "identity", "context")
    }
    context = np.clip((encoded["context"] - mean) / scale, -8.0, 8.0).astype(np.float32)
    if variant == "control":
        context[:] = 0
    if variant == "market":
        context[:, 20:] = 0
    result["context"] = context
    result["identity"] = (
        np.zeros_like(encoded["identity"])
        if variant == "control"
        else encoded["identity"]
    )
    return result


def guardrail(y, candidate, reference):
    result = dict(rmsle=score(y, candidate), reference_rmsle=score(y, reference))
    eligible = result["rmsle"] < result["reference_rmsle"] - 1e-5
    for name, use in [
        ("low", y < np.log1p(10)),
        ("high", (y >= np.log1p(100)) & (y < np.log1p(500))),
    ]:
        n = int(use.sum())
        row = dict(rows=n, sufficient=n >= MIN_BUCKET_ROWS)
        if n:
            row.update(
                rmsle=score(y[use], candidate[use]),
                reference_rmsle=score(y[use], reference[use]),
            )
        if n >= MIN_BUCKET_ROWS:
            eligible = (
                eligible and row["rmsle"] <= row["reference_rmsle"] * GUARD_FACTOR
            )
        result[name] = row
    result["eligible"] = bool(eligible)
    return result


def save_model(model, path):
    tf.keras.Model(model.inputs, model.outputs).save(path)


def run(args):
    start = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    cache, encoded = load(args)
    if mercari.sha256(cache["reference"]["path"]) != cache["reference"]["sha256"]:
        raise ValueError("frozen CNN prediction hash mismatch")
    means = np.load(args.cache_dir / "context-mean.npy")
    scales = np.load(args.cache_dir / "context-scale.npy")
    center = cache["mean"]
    variants = [
        ("control", SEEDS[0]),
        ("market", SEEDS[0]),
        ("retrieval", SEEDS[0]),
        *[(v, s) for s in SEEDS[1:] for v in ("control", "retrieval")],
    ]
    plan = dict(
        cache_sha256=mercari.sha256(args.cache_dir / "manifest.json"),
        source_sha256=mercari.sha256(__file__),
        cache_identity=cache["identity"],
        variants=variants,
        epochs=EPOCHS,
        batch_size=BATCH,
        guard_factor=GUARD_FACTOR,
        minimum_bucket_rows=MIN_BUCKET_ROWS,
        train_rows=cache["rows"]["train"],
        checkpoint_window="calibration",
        strategy_window="selection",
        all_original_text_weights_trainable=True,
        final_fusion_modified=False,
        price_reweighting=False,
        zero_prices_retained=True,
        historical_selection_audit_reused=True,
        primary_seed=SEEDS[0],
        secondary_seeds_diagnostic_only=True,
        original_reference=cache["reference"],
    )
    dump(out / "plan.json", plan)
    base = tf.keras.models.load_model(
        args.model_dir / "cnn128_mse.keras", compile=False
    )
    raw_x = {
        w: {k: v for k, v in e.items() if k not in ("context", "identity", "ids", "y")}
        for w, e in encoded.items()
        if w != "train"
    }
    raw = {w: center + predict(base, x, batch_size=4096) for w, x in raw_x.items()}
    with np.load(cache["reference"]["path"]) as archive:
        replay = max(float(np.max(np.abs(raw[w] - archive[w]))) for w in raw)
        if replay > 2e-4:
            raise ValueError(
                "CNN preprocessing/CPU replay differs from the frozen base reference"
            )
    y = encoded["calibration"]["y"]
    selection_y = encoded["selection"]["y"]
    metrics = {
        "original": dict(
            calibration_rmsle=score(y, raw["calibration"]),
            selection_rmsle=score(selection_y, raw["selection"]),
            reference=True,
        )
    }
    predictions = {"original": raw}
    cal_reference = {}
    selection_reference = {}
    del base
    tf.keras.backend.clear_session()
    gc.collect()
    for variant, seed in variants:
        name = f"{variant}_seed{seed}"
        folder = out / name
        folder.mkdir()
        x = {w: view_inputs(e, means, scales, variant) for w, e in encoded.items()}
        tf.keras.utils.set_random_seed(seed)
        base = tf.keras.models.load_model(
            args.model_dir / "cnn128_mse.keras", compile=False
        )
        model = build_model(base, seed)
        body = model.get_layer("text_trunk")
        before = [v.numpy().copy() for v in body.trainable_weights]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
            loss="mse",
            jit_compile=False,
        )
        reference = raw["calibration"] if variant == "control" else cal_reference[seed]
        initial = center + predict(model, x["calibration"], batch_size=4096)
        best = score(y, initial)
        diagnostic_best = best
        epoch_selected = 0
        history = []
        save_model(model, folder / "selected.keras")
        save_model(model, folder / "diagnostic.keras")
        stamp = time.perf_counter()
        rng = np.random.default_rng(seed)
        for epoch in range(EPOCHS):
            model.optimizer.learning_rate.assign(1e-4 * (0.5**epoch))
            order = rng.permutation(cache["rows"]["train"])
            model.reset_metrics()
            epoch_start = time.perf_counter()
            for step, begin in enumerate(range(0, len(order), BATCH)):
                rows = order[begin : begin + BATCH]
                loss = model.train_on_batch(
                    {k: v[rows] for k, v in x["train"].items()},
                    encoded["train"]["y"][rows] - center,
                )
                if step % 64 == 0:
                    print("TRUNK_BATCH", name, epoch + 1, step, float(loss), flush=True)
            candidate = center + predict(model, x["calibration"], batch_size=4096)
            row = dict(
                epoch=epoch + 1,
                loss=float(loss),
                seconds=time.perf_counter() - epoch_start,
                **guardrail(y, candidate, reference),
            )
            history.append(row)
            if row["rmsle"] < diagnostic_best:
                diagnostic_best = row["rmsle"]
                save_model(model, folder / "diagnostic.keras")
            if row["eligible"] and row["rmsle"] < best - 1e-5:
                best = row["rmsle"]
                epoch_selected = epoch + 1
                save_model(model, folder / "selected.keras")
            print("TRUNK_EPOCH", name, row, flush=True)
        fit_seconds = time.perf_counter() - stamp
        changed = sum(
            not np.array_equal(a, b.numpy())
            for a, b in zip(before, body.trainable_weights)
        )
        for kind in ("selected", "diagnostic"):
            fitted = tf.keras.models.load_model(
                folder / (kind + ".keras"), compile=False
            )
            tag = name if kind == "selected" else "diagnostic_" + name
            values = {w: center + predict(fitted, x[w], batch_size=4096) for w in raw}
            predictions[tag] = values
            row = dict(
                variant=variant,
                seed=seed,
                selected_epoch=epoch_selected if kind == "selected" else None,
                calibration_rmsle=score(y, values["calibration"]),
                selection_rmsle=score(selection_y, values["selection"]),
                cal_guardrail=guardrail(y, values["calibration"], reference),
                history=history,
                fit_seconds=fit_seconds,
                trained_body_tensors_changed=changed,
                checkpoint_body_tensors_changed=sum(
                    not np.array_equal(a, b.numpy())
                    for a, b in zip(
                        before, fitted.get_layer("text_trunk").trainable_weights
                    )
                ),
                body_trainable_tensors=len(before),
                parameters=fitted.count_params(),
                diagnostic_only=kind != "selected" or seed != SEEDS[0],
            )
            if kind == "selected":
                ref = (
                    raw["selection"]
                    if variant == "control"
                    else selection_reference[seed]
                )
                row["selection_guardrail"] = guardrail(
                    selection_y, values["selection"], ref
                )
                samples = []
                for _ in range(3):
                    tick = time.perf_counter()
                    predict(fitted, x["calibration"], batch_size=4096)
                    samples.append(time.perf_counter() - tick)
                row.update(
                    inference_rows=len(y),
                    inference_seconds=float(np.mean(samples)),
                    inference_samples=samples,
                    inference_max=max(samples),
                    inference_std=float(np.std(samples)),
                )
                if variant == "control":
                    cal_reference[seed] = values["calibration"]
                    selection_reference[seed] = values["selection"]
            metrics[tag] = row
            np.savez_compressed(
                out / (tag + ".npz"),
                **values,
                **{w + "_ids": encoded[w]["ids"] for w in raw},
            )
        del model, base, body, fitted, before, x
        tf.keras.backend.clear_session()
        gc.collect()
    primary = [
        "original",
        *[f"{v}_seed{SEEDS[0]}" for v in ("control", "market", "retrieval")],
    ]
    eligible = [
        n
        for n in primary
        if n == "original"
        or (
            metrics[n]["cal_guardrail"]["eligible"]
            and metrics[n]["selection_guardrail"]["eligible"]
        )
    ]
    best = min(metrics[n]["selection_rmsle"] for n in eligible)
    chosen = next(n for n in eligible if metrics[n]["selection_rmsle"] <= best + 1e-5)
    np.savez_compressed(
        out / "original.npz", **raw, **{w + "_ids": encoded[w]["ids"] for w in raw}
    )
    frozen = dict(
        plan=plan,
        models=copy.deepcopy(metrics),
        primary=primary,
        eligible=eligible,
        chosen=chosen,
        choices_frozen_before_audit=True,
    )
    dump(out / "frozen.json", frozen)
    with np.load(args.data_dir / "evaluation.npz") as audit:
        truth = audit["confirmation"]
        if (
            not np.array_equal(
                audit["confirmation_ids"], encoded["confirmation"]["ids"]
            )
            or truth.shape != (cache["rows"]["confirmation"],)
            or not np.isfinite(truth).all()
            or (truth < 0).any()
        ):
            raise ValueError("audit labels and ordered IDs must match frozen inputs")
    for n, row in metrics.items():
        row["confirmation_rmsle"] = score(truth, predictions[n]["confirmation"])
        row["buckets"] = bucket_metrics(truth, predictions[n]["confirmation"])
    comparisons = {
        n: paired_comparison(truth, raw["confirmation"], p["confirmation"], repeats=300)
        for n, p in predictions.items()
        if n != "original"
    }
    report = dict(
        frozen=frozen,
        models=metrics,
        comparisons=comparisons,
        chosen=chosen,
        original_replay_max_absolute_error=replay,
        seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        input_preparation_seconds=cache["preparation_seconds"],
        final_fusion_modified=False,
        submission_status="not_submitted",
    )
    dump(out / "report.json", report)
    print("TRUNK_RESULT", chosen, metrics[chosen]["confirmation_rmsle"], flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/mercari/fusion-candidates.json"),
    )
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args(argv)
    if args.prepare:
        return prepare(args)
    if args.output_dir is None:
        p.error("--output-dir is required for training")
    return run(args)


if __name__ == "__main__":
    main()
