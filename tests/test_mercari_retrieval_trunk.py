"""Joint text-body updates and validation-only price-bucket safeguards."""

import json
import pickle
import runpy
import sys

import numpy as np
import pytest
import tensorflow as tf

from benchmarks import mercari_retrieval_trunk as trunk
from benchmarks.mercari_neural import build_model as old_model
from test_mercari_retrieval_inputs import fixture


def test_joint_model_starts_at_base_and_updates_original_text_weights(tmp_path):
    args = fixture(tmp_path)
    with (args.model_dir / "tokens.pkl").open("rb") as f:
        _, tokens, _ = pickle.load(f)
    base = old_model(tokens, width=4)
    model = trunk.build_model(base, seed=42)
    from benchmarks.mercari_retrieval_inputs import prepare, load

    prepare(args)
    _, encoded = load(args)
    mean = np.zeros(28, np.float32)
    scale = np.ones(28, np.float32)
    x = trunk.view_inputs(encoded["train"], mean, scale, "retrieval")
    raw = base({k: x[k] for k in base.input}, training=False).numpy()
    np.testing.assert_allclose(model(x, training=False).numpy(), raw, atol=1e-6)
    before = [a.numpy().copy() for a in model.get_layer("text_trunk").trainable_weights]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", jit_compile=False
    )
    model.train_on_batch(x, np.asarray(encoded["train"]["y"]))
    assert any(
        not np.array_equal(a, b.numpy())
        for a, b in zip(before, model.get_layer("text_trunk").trainable_weights)
    )
    path = tmp_path / "joint.keras"
    tf.keras.Model(model.inputs, model.outputs).save(path)
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(
        model(x, training=False), restored(x, training=False), atol=1e-6
    )
    for view in ["control", "market"]:
        values = trunk.view_inputs(encoded["train"], mean, scale, view)
        assert not values["context"][:, 20:].any()
    assert not trunk.view_inputs(encoded["train"], mean, scale, "control")[
        "identity"
    ].any()
    with pytest.raises(ValueError):
        trunk.view_inputs(encoded["train"], mean, scale, "bad")


def test_guardrails_use_metric_and_both_sufficient_price_segments(monkeypatch):
    monkeypatch.setattr(trunk, "MIN_BUCKET_ROWS", 1)
    y = np.log1p([5.0, 6.0, 20.0, 25.0, 100.0, 200.0])
    base = y + 0.1
    assert trunk.guardrail(y, y, base)["eligible"]
    bad = y.copy()
    bad[:2] += 0.2
    assert not trunk.guardrail(y, bad, base)["eligible"]
    bad = y.copy()
    bad[-2:] += 0.2
    assert not trunk.guardrail(y, bad, base)["eligible"]
    assert not trunk.guardrail(y, base, base)["eligible"]
    small = trunk.guardrail(y[2:4], y[2:4], base[2:4])
    assert small["low"]["rows"] == 0 and small["high"]["rows"] == 0


def test_cli_help_and_prepare_route(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["trunk", "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("benchmarks.mercari_retrieval_trunk", run_name="__main__")
    assert stopped.value.code == 0
    monkeypatch.setattr(trunk, "prepare", lambda args: "prepared")
    assert (
        trunk.main(
            [
                "--prepare",
                "--data-dir",
                "data",
                "--model-dir",
                "model",
                "--cache-dir",
                "cache",
            ]
        )
        == "prepared"
    )
    with pytest.raises(SystemExit):
        trunk.main(
            ["--data-dir", "data", "--model-dir", "model", "--cache-dir", "cache"]
        )


def training_fixture(tmp_path, real_reference=True):
    from benchmarks.mercari_retrieval_inputs import prepare
    from benchmarks.mercari_neural import predict
    from benchmarks import mercari
    import pandas as pd

    args = fixture(tmp_path)
    args.output_dir = tmp_path / "run"
    with (args.model_dir / "tokens.pkl").open("rb") as stream:
        listing, tokens, center = pickle.load(stream)
    tf.keras.utils.set_random_seed(11)
    base = old_model(tokens, width=4)
    tf.keras.Model(base.inputs, base.outputs).save(args.model_dir / "cnn128_mse.keras")
    ref = tmp_path / "reference.npz"
    if real_reference:
        frames = {
            w: pd.read_csv(args.data_dir / (w + ".tsv.gz"), sep="\t")
            for w in ("train", "calibration", "selection", "confirmation")
        }
        predictions = {
            w: center + predict(base, tokens.transform(listing.transform(f)))
            for w, f in frames.items()
        }
        np.savez_compressed(
            ref,
            **predictions,
            **{w + "_ids": f.train_id.to_numpy() for w, f in frames.items()},
        )
        m = json.loads(args.manifest.read_text())
        m["experts"][0]["sha256"] = mercari.sha256(ref)
        args.manifest.write_text(json.dumps(m))
    np.savez_compressed(
        args.data_dir / "evaluation.npz",
        confirmation=np.log1p(10.0 + np.arange(36) % 20),
        confirmation_ids=np.arange(36) + 300,
    )
    prepare(args)
    return args


def test_full_runner_joint_training_checkpoints_frozen_before_audit(
    tmp_path, monkeypatch
):
    args = training_fixture(tmp_path)
    monkeypatch.setattr(trunk, "EPOCHS", 1)
    monkeypatch.setattr(trunk, "BATCH", 16)
    monkeypatch.setattr(trunk, "SEEDS", (42,))
    original = np.load

    def audited(path, *a, **kw):
        from pathlib import Path

        if Path(path).name == "evaluation.npz":
            frozen = json.loads((args.output_dir / "frozen.json").read_text())
            assert frozen["choices_frozen_before_audit"]
            assert all(
                "confirmation_rmsle" not in row for row in frozen["models"].values()
            )
        return original(path, *a, **kw)

    monkeypatch.setattr(trunk.np, "load", audited)
    report = trunk.run(args)
    assert len(report["models"]) == 7
    assert report["final_fusion_modified"] is False
    for name, row in report["models"].items():
        if name == "original":
            continue
        assert row["trained_body_tensors_changed"] > 0
        assert row["parameters"] > 0 and len(row["history"]) == 1
        if row["selected_epoch"] == 0:
            assert row["checkpoint_body_tensors_changed"] == 0
        elif row["selected_epoch"] is not None:
            assert row["checkpoint_body_tensors_changed"] > 0
        assert (args.output_dir / (name + ".npz")).exists()
    assert all(
        "confirmation_rmsle" not in row for row in report["frozen"]["models"].values()
    )
    monkeypatch.setattr(trunk, "run", lambda parsed: parsed)
    parsed = trunk.main(
        [
            "--data-dir",
            str(args.data_dir),
            "--model-dir",
            str(args.model_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    assert parsed.output_dir == args.output_dir


def test_runner_rejects_reference_replay_and_late_audit_id_corruption(
    tmp_path, monkeypatch
):
    a = tmp_path / "wrong"
    a.mkdir()
    args = training_fixture(a, real_reference=False)
    with pytest.raises(ValueError, match="replay"):
        trunk.run(args)
    b = tmp_path / "corrupt"
    b.mkdir()
    args = training_fixture(b)
    ref = b / "reference.npz"
    ref.write_bytes(b"corrupted source")
    with pytest.raises(ValueError, match="hash"):
        trunk.run(args)
    c = tmp_path / "audit"
    c.mkdir()
    args = training_fixture(c)
    np.savez_compressed(
        args.data_dir / "evaluation.npz",
        confirmation=np.ones(36),
        confirmation_ids=np.arange(36)[::-1],
    )
    monkeypatch.setattr(trunk, "EPOCHS", 0)
    monkeypatch.setattr(trunk, "SEEDS", (42,))
    with pytest.raises(ValueError, match="audit"):
        trunk.run(args)
    frozen = json.loads((args.output_dir / "frozen.json").read_text())
    for name, row in frozen["models"].items():
        if name != "original":
            assert row["trained_body_tensors_changed"] == 0
            assert row["checkpoint_body_tensors_changed"] == 0
