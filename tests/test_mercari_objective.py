"""Exact global metric and joint-gradient tests before the experiment runner."""

import pickle
import copy
import json
import runpy
import sys

import numpy as np
import pytest

from benchmarks import mercari_objective as target
from benchmarks.mercari_fusion import MLPBlend
from sklearn.exceptions import NotFittedError
from test_mercari_fusion import settings


def members(rows=30, seeds=(42, 43)):
    rng = np.random.default_rng(4)
    p = 3 + rng.uniform(-1, 1, (rows, 3))
    g = np.where(np.arange(rows) % 2, "a", "b")
    y = 0.7 * p[:, 0] + 0.3 * p[:, 1]
    models = [MLPBlend(s).fit(p, y, g, epochs=2) for s in seeds]
    return models, p, y, g


def test_final_loss_is_global_rmsle_with_one_final_clip():
    z, y = np.array([-1.0, 1.0, 4.0]), np.array([0.0, 1.0, 2.0])
    value, gradient = target.metric_gradient(z, y, "rmsle")
    official = np.sqrt(np.mean((np.log1p(np.expm1(np.maximum(z, 0))) - y) ** 2))
    assert value == official
    square, slope = target.metric_gradient(z, y, "squared_rmsle")
    assert square == pytest.approx(value**2)
    np.testing.assert_allclose(gradient, slope / (2 * value))
    assert gradient[0] == 0
    assert target.metric_gradient([1, 3], [1, 1], "rmsle")[0] == pytest.approx(
        np.sqrt(2)
    )
    assert target.metric_gradient([1, 1], [1, 1], "rmsle")[0] == 0
    np.testing.assert_array_equal(
        target.metric_gradient([1, 1], [1, 1], "rmsle")[1], [0, 0]
    )
    for kind in ("log_mse", "squared_rmsle", "rmsle"):
        for i in range(3):
            plus, minus = z.copy(), z.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numeric = (
                target.metric_gradient(plus, y, kind)[0]
                - target.metric_gradient(minus, y, kind)[0]
            ) / 2e-6
            assert target.metric_gradient(z, y, kind)[1][i] == pytest.approx(
                numeric, abs=1e-7
            )
    with pytest.raises(ValueError, match="loss"):
        target.metric_gradient(z, y, "accuracy")


@pytest.mark.parametrize("kind", ["log_mse", "squared_rmsle", "rmsle"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_full_network_directional_gradient_includes_every_member(kind, count):
    models, p, y, g = members(seeds=tuple(range(42, 42 + count)))
    objective = target.NetworkObjective(models, p, y, g)
    prediction = np.mean([m.predict(p, g) for m in models], axis=0)
    np.testing.assert_allclose(
        objective.predict(objective.initial), prediction, atol=1e-14
    )
    value, gradient = objective.value_gradient(objective.initial, kind)
    assert value == pytest.approx(target.metric_gradient(prediction, y, kind)[0])
    rng = np.random.default_rng(8)
    for _ in range(6):
        direction = rng.normal(size=len(gradient))
        direction /= np.linalg.norm(direction)
        delta = 1e-5
        numeric = (
            objective.value_gradient(objective.initial + delta * direction, kind)[0]
            - objective.value_gradient(objective.initial - delta * direction, kind)[0]
        ) / (2 * delta)
        assert gradient @ direction == pytest.approx(numeric, abs=3e-7)


def test_joint_objective_applies_metric_after_averaging_member_predictions():
    models, p, _, g = members()
    for model, output in zip(models, (-1.0, 3.0)):
        for a in model.model_.coefs_ + model.model_.intercepts_:
            a.fill(0)
        model.model_.intercepts_[-1][:] = (output - model.mean_) / model.scale_
    objective = target.NetworkObjective(models, p, np.ones(len(p)), g)
    assert objective.value_gradient(objective.initial, "rmsle")[0] == pytest.approx(
        0, abs=1e-14
    )
    # Clipping individual predictions first would give 1.5 and nonzero error.
    assert np.mean([np.maximum(m.predict(p, g), 0) for m in models], axis=0)[
        0
    ] == pytest.approx(1.5)


def test_fine_tuning_improves_training_metric_without_mutating_warm_start():
    models, p, y, g = members(rows=120)
    before = [m.predict(p, g).copy() for m in models]
    baseline = target.metric_gradient(np.mean(before, axis=0), y, "rmsle")[0]
    model = target.MetricBlend("rmsle").fit(p, y, g, members=models, iterations=5)
    assert target.metric_gradient(model.predict(p, g), y, "rmsle")[0] < baseline
    for m, old in zip(models, before):
        np.testing.assert_array_equal(m.predict(p, g), old)
    np.testing.assert_array_equal(
        model.predict(p, g), pickle.loads(pickle.dumps(model)).predict(p, g)
    )
    assert not hasattr(model, "x") and not hasattr(model, "y")
    zero = target.MetricBlend().fit(p, y, g, members=models, iterations=0)
    np.testing.assert_allclose(zero.predict(p, g), np.mean(before, axis=0), atol=1e-14)


def test_invalid_objective_data_and_frozen_iteration_guards():
    for p, y in (([], []), ([[1]], [1]), ([1], [1, 2]), ([np.nan], [1]), ([1], [-1])):
        with pytest.raises(ValueError, match="finite"):
            target.metric_gradient(p, y, "rmsle")
    models, p, y, g = members()
    with pytest.raises(ValueError, match="member"):
        target.NetworkObjective([], p, y, g)
    objective = target.NetworkObjective(models, p, y, g)
    for bad in (objective.initial[:1], objective.initial * np.nan):
        with pytest.raises(ValueError, match="parameters"):
            objective.predict(bad)
    with pytest.raises(NotFittedError):
        target.MetricBlend().predict(p, g)
    with pytest.raises(ValueError, match="validation"):
        target.MetricBlend().fit(p, y, g, members=models)
    with pytest.raises(ValueError, match="iterations"):
        target.MetricBlend().fit(p, y, g, members=models, iterations=-1)
    assert target.metric_gradient([0], [1], "rmsle")[1][0] == 0


def test_tune_plateau_retains_iteration_zero_and_stops_refinement(monkeypatch):
    models, p, y, g = members(rows=80)
    monkeypatch.setattr(target, "score", lambda *args: 1.0)
    monkeypatch.setattr(target, "PATIENCE", 1)
    model = target.MetricBlend().fit(p, y, g, members=models, validation=(p, y, g))
    assert model.selected_iterations_ == 0
    assert model.optimizer_["status"] == 99
    np.testing.assert_allclose(
        model.predict(p, g),
        np.mean([m.predict(p, g) for m in models], axis=0),
        atol=1e-14,
    )


def test_full_loss_study_freezes_models_and_matches_saved_check_predictions(
    tmp_path, monkeypatch
):
    args, frames, _ = settings(tmp_path)
    original = target.dump
    freezes = []

    def observe(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            freezes.append(True)
            assert "confirmation_rmsle" not in json.dumps(payload)
            assert (args.output_dir / "meta-models.pkl").exists()

    monkeypatch.setattr(target, "dump", observe)
    report = target.run(args)
    assert freezes == [True]
    assert len(report["models"]) == 21
    assert len(report["frozen"]["primary"]) == 13
    assert report["frozen"]["accepted"]
    assert report["frozen"] == json.loads((args.output_dir / "frozen.json").read_text())
    assert report["frozen"]["plan"]["fine_tune_L2"] == 0
    assert report["frozen"]["plan"]["full_window_objective"]
    assert report["frozen"]["plan"]["joint_ensemble_initializations"] == 1
    assert set(report["comparisons"]) == set(target.LOSSES)
    with (args.output_dir / "meta-models.pkl").open("rb") as stream:
        meta = pickle.load(stream)
    candidates = target.load_candidates(json.loads(args.manifest.read_text()), frames)
    source = ["strong", "weak"]
    indices = [source.index(n) for n in meta["names"]]
    check_ids = report["frozen"]["plan"]["split_ids"]["check"]
    check_rows = [list(frames["selection"].train_id).index(i) for i in check_ids]
    with np.load(args.output_dir / "meta-check.npz") as saved:
        for name, model in meta["models"].items():
            np.testing.assert_allclose(
                saved[name],
                target.infer(
                    model,
                    candidates["selection"][check_rows][:, indices],
                    target.context(frames["selection"].iloc[check_rows]),
                ),
                atol=1e-14,
            )
    for name, row in report["models"].items():
        assert row["predict_seconds"] == np.mean(row["predict_seconds_samples"])
        with np.load(args.output_dir / (name + ".npz")) as saved:
            np.testing.assert_array_equal(
                saved["confirmation_ids"], frames["confirmation"].train_id
            )
    assert report["submission_status"] == "not_submitted"
    assert json.loads((args.output_dir / "report.json").read_text()) == report


def test_study_rejects_wrong_lineage_falls_back_exactly_and_checks_audit_ids(
    tmp_path, monkeypatch
):
    args, frames, manifest = settings(tmp_path)
    bad = copy.deepcopy(manifest)
    bad["data_sha256"]["train"] = "wrong"
    args.manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="protocol"):
        target.run(args)
    args.manifest.write_text(json.dumps(manifest))
    arrays = {}
    for k, frame in frames.items():
        if k == "train":
            continue
        arrays[k] = np.log1p(
            frame.price if "price" in frame else 10 + frame.train_id.to_numpy() % 20
        )
        arrays[k + "_ids"] = frame.train_id.to_numpy()
    np.savez(args.anchor, **arrays)
    args.output_dir = tmp_path / "fallback"
    report = target.run(args)
    assert not report["frozen"]["accepted"]
    with np.load(args.output_dir / "ensemble.npz") as saved:
        for k in ("selection", "confirmation"):
            np.testing.assert_array_equal(saved[k], arrays[k])
    args.output_dir = tmp_path / "bad-audit"
    original = target.dump

    def poison(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            np.savez(
                args.data_dir / "evaluation.npz",
                confirmation=np.ones(30),
                confirmation_ids=np.arange(30),
            )

    monkeypatch.setattr(target, "dump", poison)
    with pytest.raises(ValueError, match="audit"):
        target.run(args)


def test_cli_uses_fixed_loss_protocol_and_importable_classes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(target, "run", lambda args: calls.append(args))
    argv = [
        s
        for name in ("data-dir", "manifest", "anchor", "output-dir")
        for s in ("--" + name, str(tmp_path / name))
    ]
    target.main(argv)
    monkeypatch.setattr(sys, "argv", ["mercari_objective", *argv])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("benchmarks.mercari_objective", run_name="__main__")
    assert len(calls) == 2 and calls[0].rows == 0 and calls[0].seed == 42
