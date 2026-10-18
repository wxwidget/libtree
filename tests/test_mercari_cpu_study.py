"""A real CPU study plus lineage, budget and audit-boundary regressions."""

from argparse import Namespace
import json
import runpy
import sys

import numpy as np
import pytest

from benchmarks import mercari_cpu_study as study
from test_mercari import listings


def inputs(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    frame = listings(200)
    parts = dict(
        train=frame.iloc[:120],
        calibration=frame.iloc[120:145],
        selection=frame.iloc[145:170],
        confirmation=frame.iloc[170:],
    )
    for name, part in parts.items():
        (part.drop(columns="price") if name == "confirmation" else part).to_csv(
            root / (name + ".tsv.gz"), sep="\t", index=False
        )
    np.savez(
        root / "evaluation.npz",
        confirmation=np.log1p(parts["confirmation"].price),
        confirmation_ids=parts["confirmation"].train_id,
    )
    np.savez(
        root / "baseline.npz",
        **{k: np.full(len(v), 2.9) for k, v in parts.items() if k != "train"},
        **{k + "_ids": v.train_id for k, v in parts.items() if k != "train"},
    )
    return root


def settings(tmp_path, **overrides):
    args = Namespace(
        data_dir=inputs(tmp_path),
        output_dir=tmp_path / "result",
        rows=0,
        threads=1,
        max_features=40,
        char_features=20,
        cross_features=20,
        width=8,
        epochs=1,
        tree_rounds=2,
        seed=42,
        reference=True,
    )
    return Namespace(**(vars(args) | overrides))


def test_real_cpu_experiment_freezes_before_audit_and_retains_predictions(tmp_path):
    args = settings(tmp_path)
    report = study.run(args)
    assert report["gpu_used"] is False and report["external_pretrained_data"] is False
    assert (
        report["training_rows"] == 120 and report["confirmation_previously_inspected"]
    )
    assert len(report["models"]) == 12 + 5
    assert len(report["frozen"]["names"]) == 23
    assert report["frozen"]["selection_chosen_before_confirmation"] is True
    assert report["reference_cached_cost_excluded"]
    assert set(report["base_family_ablations"]) == {
        "without_mse",
        "without_huber",
        "without_bins",
        "without_engineered",
    }
    frozen = json.loads((args.output_dir / "frozen.json").read_text())
    assert frozen == report["frozen"]
    assert (args.output_dir / "frozen.json").stat().st_mtime_ns < (
        args.output_dir / "report.json"
    ).stat().st_mtime_ns
    prediction = np.load(args.output_dir / "ensemble.npz")
    np.testing.assert_array_equal(prediction["confirmation_ids"], np.arange(170, 200))
    assert np.isfinite(prediction["confirmation"]).all()
    assert report["submission_status"] == "not_submitted"
    assert report["seconds"] > 0.0
    assert json.loads((args.output_dir / "report.json").read_text()) == report
    with pytest.raises(ValueError, match="empty"):
        study.run(args)


def test_data_guards_reject_leakage_and_wrong_reference_ids(tmp_path):
    args = settings(tmp_path)
    frame = listings(200)
    frame.iloc[:10].to_csv(args.data_dir / "selection.tsv.gz", sep="\t", index=False)
    with pytest.raises(ValueError, match="overlap"):
        study.read_data(args)
    frame.iloc[145:170].assign(price=-1).to_csv(
        args.data_dir / "selection.tsv.gz", sep="\t", index=False
    )
    with pytest.raises(ValueError, match="nonnegative"):
        study.read_data(args)
    frame.iloc[145:170].to_csv(
        args.data_dir / "selection.tsv.gz", sep="\t", index=False
    )
    data = study.read_data(args)
    data["confirmation"] = data["confirmation"].assign(price=123.0)
    with pytest.raises(ValueError, match="confirmation"):
        study.reference_predictions(args, data)
    data["confirmation"] = data["confirmation"].drop(columns="price")
    data["selection"] = data["selection"].iloc[::-1]
    with pytest.raises(ValueError, match="IDs"):
        study.reference_predictions(args, data)
    frame.iloc[170:].to_csv(
        args.data_dir / "confirmation.tsv.gz", sep="\t", index=False
    )
    with pytest.raises(ValueError, match="confirmation"):
        study.read_data(args)


def test_sampled_run_without_cached_reference_rejects_misaligned_audit(
    tmp_path, monkeypatch
):
    args = settings(tmp_path, rows=60, reference=False)
    original = study.dump

    def poison_after_freeze(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            np.savez(
                args.data_dir / "evaluation.npz",
                confirmation=np.zeros(30),
                confirmation_ids=np.arange(170, 200)[::-1],
            )

    monkeypatch.setattr(study, "dump", poison_after_freeze)
    with pytest.raises(ValueError, match="audit IDs"):
        study.run(args)
    assert (args.output_dir / "frozen.json").exists()
    assert not (args.output_dir / "report.json").exists()


def test_cli_sets_cpu_limits_and_enforces_hard_timeout(tmp_path, monkeypatch):
    import subprocess

    observed = []

    def process(command, **kwargs):
        observed.append((command, kwargs))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        return Namespace(returncode=0)

    monkeypatch.setattr(study.subprocess, "run", process)
    argv = [
        "--data-dir",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "run"),
        "--threads",
        "2",
        "--max-seconds",
        "3",
    ]
    assert study.main(argv) == 0
    assert observed[0][1]["timeout"] == 3.0

    def expired(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 3)

    monkeypatch.setattr(study.subprocess, "run", expired)
    assert study.main(argv) == 124
    assert (
        json.loads((tmp_path / "run" / "budget-exhausted.json").read_text())[
            "submitted"
        ]
        is False
    )
    with pytest.raises(ValueError):
        study.main(argv + ["--max-seconds", "0"])
    with pytest.raises(ValueError):
        study.main(argv + ["--threads", "0"])
    monkeypatch.setattr(study, "run", lambda args: observed.append(args))
    assert study.main(argv + ["--worker", "--no-reference", "--rows", "15"]) == 0
    assert observed[-1].reference is False and observed[-1].rows == 15
    monkeypatch.setattr(sys, "argv", ["mercari_cpu_study", *argv, "--worker"])
    assert study.main() == 0
    monkeypatch.setattr(sys, "argv", ["mercari_cpu_study", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("benchmarks.mercari_cpu_study", run_name="__main__")
    assert exc.value.code == 0


def test_real_launcher_timeout_leaves_no_successful_score(tmp_path):
    result = study.main(
        [
            "--data-dir",
            str(tmp_path / "unused"),
            "--output-dir",
            str(tmp_path / "limited"),
            "--max-seconds",
            "0.001",
            "--threads",
            "1",
        ]
    )
    assert result == 124
    record = json.loads((tmp_path / "limited" / "budget-exhausted.json").read_text())
    assert record["status"] == "incomplete_do_not_promote"
    assert not (tmp_path / "limited" / "report.json").exists()
