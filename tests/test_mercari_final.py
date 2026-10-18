"""Final refit: disjoint roles, streamed IDs, frozen stacking and fail-closed output."""

from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from benchmarks import mercari_final as final
from test_mercari import listings


def inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    listings(60).to_csv(data / "train.tsv", sep="\t", index=False)
    test = listings(7).drop(columns=["price", "train_id"]).assign(test_id=np.arange(7))
    test.to_csv(data / "test.tsv", sep="\t", index=False)
    pd.DataFrame({"test_id": np.arange(7), "price": 0.0}).to_csv(
        data / "sample.csv", index=False
    )
    return Namespace(
        data_dir=data,
        output_dir=tmp_path / "run",
        test_file="test.tsv",
        sample_file="sample.csv",
        chunk_size=3,
        threads=1,
    )


def test_final_split_uses_every_label_once_and_never_test_labels(tmp_path):
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    final.prepare(args)
    base, calibration = final.training(args)
    assert len(base) + len(calibration) == 60
    assert not set(base.train_id) & set(calibration.train_id)
    assert not set(final.test_frames(args).__next__().columns) & {"price", "train_id"}
    report = json.loads((args.output_dir / "protocol.json").read_text())
    assert report["no_selection_or_confirmation_labels"]
    assert sum(p["rows"] for p in report["splits"].values()) == 60


def test_streamed_exports_validate_each_component_and_round_trip_ids(tmp_path):
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    final.prepare(args)
    _, calibration = final.training(args)
    final.export(
        args,
        "unit",
        calibration,
        lambda frame: dict(
            value=np.full(len(frame), 3.0), features=np.ones((len(frame), 2))
        ),
    )
    block = final.read_stage(args.output_dir, "unit", "test")
    assert block["features"].shape == (7, 2)
    np.testing.assert_array_equal(block["ids"], np.arange(7).astype(str))
    np.testing.assert_array_equal(block["value"], 3.0)
    with pytest.raises(ValueError, match="finite"):
        final.export(
            args,
            "bad",
            calibration,
            lambda frame: dict(value=np.full(len(frame), np.nan)),
        )
    sample = pd.read_csv(args.data_dir / "sample.csv").iloc[::-1]
    sample.to_csv(args.data_dir / "sample.csv", index=False)
    with pytest.raises(ValueError, match="IDs"):
        final.export(
            args,
            "reordered",
            calibration,
            lambda frame: dict(value=np.full(len(frame), 3.0)),
        )


def test_submission_rejects_labeled_test_and_count_mismatch(tmp_path):
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    final.prepare(args)
    _, calibration = final.training(args)
    path = args.data_dir / "test.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.assign(price=10).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="price"):
        list(final.test_frames(args))
    frame.iloc[:3].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="count"):
        final.export(
            args, "short", calibration, lambda frame: dict(value=np.ones(len(frame)))
        )


def test_stack_reconstructs_frozen_features_and_weights_and_checks_ids():
    ids = np.arange(3).astype(str)
    groups = {
        "baseline": dict(
            ids=ids,
            baseline=np.ones(3) * 3,
            old_uncorrected=np.ones(3) * 2,
            metadata=np.ones((3, 2)),
            attributes=np.ones((3, 1)),
        ),
        "lsa": dict(ids=ids, latent=np.ones((3, 2)), lsa_ridge_10=np.ones(3) * 4),
    }
    config = dict(
        inner=dict(names=["old_uncorrected", "lsa_ridge_10"], weights=[0.25, 0.75]),
        feature_count=7,
    )
    values, x, base = final.stack(groups, config)
    assert x.shape == (3, 7)
    np.testing.assert_array_equal(base, 3.5)
    assert "baseline" in values
    groups["lsa"]["ids"] = ids[::-1]
    with pytest.raises(ValueError, match="IDs"):
        final.stack(groups, config)
    groups["lsa"]["ids"] = ids
    with pytest.raises(ValueError, match="feature"):
        final.stack(groups, dict(config, feature_count=20))


def tiny_config():
    import copy

    c = json.loads(Path("benchmarks/mercari/full-ensemble-config.json").read_text())
    c.update(
        tiny_width=8,
        max_features=50,
        char_features=30,
        tree_rounds=3,
        threads=1,
        lsa_rounds=3,
        vector_rounds=3,
        vector_components=2,
    )
    c["lsa"]["components"] = 2
    c["lsa"]["max_features"] = 50
    c["epochs"] = {n: 1 for n in c["epochs"]}
    c["transformer"] = dict(epochs=1, batch_size=10, length=16, width=16)
    c["feature_count"] = 88
    c["baseline"].update(
        max_features=50,
        char_features=30,
        max_words=50,
        neural_members=[],
        neural_architecture="bag",
        neural_width=8,
        neural_loss="mse",
        neural_rates=[0.003],
        neural_batches=[16],
        fm_rank=2,
        fm_epochs=1,
    )
    c["baseline"]["native"].update(n_estimators=3, min_samples_leaf=2, n_jobs=1)
    return c


@pytest.mark.parametrize(
    "stage", ["baseline", "lsa", "sequence", "separate", "joint", "stemmed"]
)
def test_actual_tiny_training_exports_complete_stage_predictions(stage, tmp_path):
    if stage != "lsa":
        pytest.importorskip("tensorflow")
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    final.prepare(args)
    c = tiny_config()
    if stage in ("separate", "joint", "stemmed"):
        final.sparse_stage(args, c, stage)
    else:
        getattr(final, stage + "_stage")(args, c)
    for window in ["calibration", "test"]:
        block = final.read_stage(args.output_dir, stage, window)
        assert len(block) > 1
        for key, value in block.items():
            if key != "ids":
                assert np.isfinite(value).all()
        assert len(block["ids"]) == (4 if window == "calibration" else 7)


def test_export_rejects_changing_schema_and_empty_batches(tmp_path):
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    final.prepare(args)
    _, calibration = final.training(args)
    with pytest.raises(ValueError, match="components"):
        final.export(args, "empty", calibration, lambda _: {})
    counter = [0]

    def different(frame):
        counter[0] += 1
        return {str(counter[0]): np.ones(len(frame))}

    with pytest.raises(ValueError, match="components"):
        final.export(args, "different", calibration, different)
    counter = [0]

    def different_shape(frame):
        counter[0] += 1
        return dict(x=np.ones((len(frame), counter[0])))

    with pytest.raises(ValueError, match="shape"):
        final.export(args, "shape", calibration, different_shape)
    empty = pd.read_csv(args.data_dir / "sample.csv").iloc[:0]
    empty.to_csv(args.data_dir / "sample.csv", index=False)
    with pytest.raises(ValueError, match="count"):
        final.export(
            args, "zero", calibration, lambda frame: dict(x=np.ones(len(frame)))
        )


def stage_arrays(args, config):
    _, calibration = final.training(args)
    ids = {
        "calibration": calibration.train_id.to_numpy(dtype=str),
        "test": np.arange(7).astype(str),
    }
    inner = config["inner"]["names"]
    for stage in final.GROUPS:
        for window, index in ids.items():
            directory = args.output_dir / stage / window
            directory.mkdir(parents=True)
            np.save(directory / "ids.npy", index)
            if stage == "baseline":
                np.save(directory / "metadata.npy", np.ones((len(index), 2)))
                np.save(directory / "attributes.npy", np.ones((len(index), 1)))
                np.save(directory / "baseline.npy", np.full(len(index), 3.0))
                np.save(directory / "old_uncorrected.npy", np.full(len(index), 2.0))
            if stage == "lsa":
                np.save(directory / "latent.npy", np.ones((len(index), 2)))
                np.save(directory / "lsa_ridge_10.npy", np.full(len(index), 4.0))


def test_finish_fits_real_correctors_and_exports_valid_submission(tmp_path):
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    final.prepare(args)
    config = dict(
        inner=dict(names=["old_uncorrected", "lsa_ridge_10"], weights=[0.25, 0.75]),
        feature_count=7,
        outer=dict(
            names=[
                "baseline",
                "cal_libtree3",
                "cal_libtree5",
                "cal_lightgbm",
                "cal_xgboost",
                "cal_meta_ridge10",
                "cal_meta_ridge100",
            ],
            weights=[0.1, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
        ),
        tree_rounds=3,
        threads=1,
        correction_strengths={
            n: 1.0
            for n in ["cal_libtree3", "cal_libtree5", "cal_lightgbm", "cal_xgboost"]
        },
    )
    args.config = tmp_path / "config.json"
    args.config.write_text(json.dumps(config))
    stage_arrays(args, config)
    report = final.finish(args, config)
    submission = pd.read_csv(args.output_dir / "submission.csv")
    assert list(submission) == ["test_id", "price"] and len(submission) == 7
    assert np.isfinite(submission.price).all() and (submission.price >= 0).all()
    assert report["no_final_parameter_selection"] and report["rows"] == 7
    np.save(args.output_dir / "baseline/calibration/ids.npy", np.array(["invalid"] * 4))
    for stage in final.GROUPS:
        np.save(
            args.output_dir / stage / "calibration/ids.npy", np.array(["invalid"] * 4)
        )
    with pytest.raises(ValueError, match="label IDs"):
        final.finish(args, config)
    _, cal = final.training(args)
    for stage in final.GROUPS:
        np.save(
            args.output_dir / stage / "calibration/ids.npy",
            cal.train_id.to_numpy(dtype=str),
        )
    for stage in final.GROUPS:
        np.save(
            args.output_dir / stage / "test/ids.npy", np.arange(7)[::-1].astype(str)
        )
    with pytest.raises(ValueError, match="test IDs"):
        final.finish(args, config)
    pd.DataFrame(columns=["test_id", "price"]).to_csv(
        args.data_dir / "sample.csv", index=False
    )
    with pytest.raises(ValueError, match="complete"):
        final.finish(args, config)


def test_invalid_train_and_sample_ids_are_rejected(tmp_path):
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    path = args.data_dir / "train.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame.assign(train_id=1).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="training IDs"):
        final.prepare(args)
    frame.to_csv(path, sep="\t", index=False)
    final.prepare(args)
    _, cal = final.training(args)
    pd.DataFrame({"test_id": [1, 1], "price": [0.0, 0.0]}).to_csv(
        args.data_dir / "sample.csv", index=False
    )
    with pytest.raises(ValueError, match="unique IDs"):
        final.export(args, "badids", cal, lambda f: dict(x=np.ones(len(f))))


def test_main_dispatches_frozen_workers_and_isolated_torch(tmp_path, monkeypatch):
    import sys, runpy
    from benchmarks import mercari_final_torch

    args = inputs(tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(tiny_config()))
    base = [
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--config",
        str(config),
        "--pretrained",
        str(tmp_path),
        "--test-file",
        "test.tsv",
        "--sample-file",
        "sample.csv",
    ]
    calls = []
    monkeypatch.setattr(
        final.subprocess, "run", lambda command, **kw: calls.append(command)
    )
    final.main(base + ["--stage", "all", "--torch-python", "isolated-python"])
    assert [c[-1] for c in calls] == ["prepare", *final.GROUPS, "finish"]
    assert all(
        c[0] == "isolated-python" for c in calls if c[-1] in ["vectors", "transformer"]
    )
    final.main(base + ["--stage", "prepare"])
    import tensorflow as tf
    device = object()
    growth_calls = []
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [device] if kind == "GPU" else [])
    monkeypatch.setattr(tf.config.experimental, "set_memory_growth", lambda current, enabled: growth_calls.append((current, enabled)))
    for stage in [*final.GROUPS, "finish"]:
        if stage in ["vectors", "transformer"]:
            monkeypatch.setattr(
                mercari_final_torch, stage + "_stage", lambda *a: {"verified": True}
            )
        elif stage in ["separate", "joint", "stemmed"]:
            monkeypatch.setattr(final, "sparse_stage", lambda *a: {"verified": True})
        elif stage == "finish":
            monkeypatch.setattr(final, "finish", lambda *a: {"verified": True})
        else:
            monkeypatch.setattr(final, stage + "_stage", lambda *a: {"verified": True})
        final.main(base + ["--stage", stage])
    assert growth_calls == [(device, True)] * 5
    with pytest.raises(SystemExit):
        final.main(base + ["--chunk-size", "0"])
    monkeypatch.setattr(sys, "argv", ["mercari_final", *base, "--stage", "all"])
    final.main()
    runpy.run_module("benchmarks.mercari_final", run_name="__main__")


def test_submission_configuration_preserves_the_better_full_study_exactly():
    c = json.loads(Path("benchmarks/mercari/full-ensemble-config.json").read_text())
    report = json.loads(Path("benchmarks/mercari/frontier-study.json").read_text())
    frozen = report["final"]["frozen"]
    cal = report["model_reports"]["calibration-final/report.json"]
    assert c["outer"]["names"] == frozen["names"]
    assert c["outer"]["weights"] == frozen["weights"]
    assert c["inner"] == dict(names=cal["expert_names"], weights=cal["weights"])
    assert (
        c["source_confirmation_rmsle"]
        == report["final"]["comparison"]["candidate_rmsle"]
    )
    assert c["source_confirmation_rmsle"] < 0.381
    assert (
        c["feature_count"] == 212 and sum(w > 1e-15 for w in c["outer"]["weights"]) == 9
    )
    for folder in ["t4/gpu", "a100/separate", "a100/joint", "stemmed/stemmed"]:
        for name, metrics in report["model_reports"][folder + "/report.json"][
            "models"
        ].items():
            if not isinstance(metrics, dict) or "history" not in metrics:
                continue
            history = metrics["history"]
            key = "selection_rmsle" if "selection_rmsle" in history[0] else "rmsle"
            assert c["epochs"][name] == min(history, key=lambda r: r[key])["epoch"]
