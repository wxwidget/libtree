"""Real CPU expert reuse, ablations, lineage and bounded CLI execution."""

from argparse import Namespace
import json
import pickle
import runpy
import sys

import numpy as np
import pytest

from benchmarks import mercari_advanced as advanced
from benchmarks import mercari_multiview_study as study
from test_mercari_advanced import tiny_config
from test_mercari_cpu_study import inputs


def test_real_model_recombination_and_feature_views_are_frozen_before_audit(
    tmp_path, monkeypatch
):
    import pandas as pd

    data = inputs(tmp_path)
    frames = {
        k: pd.read_csv(data / (k + ".tsv.gz"), sep="\t")
        for k in ("train", "calibration", "selection", "confirmation")
    }
    model, _ = advanced.fit_entry(frames["train"], frames["calibration"], tiny_config())
    path = tmp_path / "trusted.pkl"
    with path.open("wb") as stream:
        pickle.dump(model, stream)
    np.savez(
        data / "baseline.npz",
        **{k: model.predict(frames[k]) for k in study.WINDOWS},
        **{k + "_ids": frames[k].train_id for k in study.WINDOWS},
    )
    model.close()
    small = tmp_path / "small"
    small.mkdir()
    for name in ("stemmed_huber", "stemmed_bins"):
        np.savez(
            small / (name + ".npz"),
            **{k: np.ones(len(frames[k])) * 3 for k in study.WINDOWS},
            **{k + "_ids": frames[k].train_id for k in study.WINDOWS},
        )
    args = Namespace(
        data_dir=data,
        baseline=path,
        output_dir=tmp_path / "study",
        small_experts=small,
        threads=1,
        seed=42,
        rows=0,
        tree_rounds=2,
    )
    report = study.run(args)
    assert len(report["models"]) == 9
    assert report["gpu_used"] is False and report["training_rows"] == 120
    assert report["component_replay_max_error"] < 1e-5
    assert set(report["feature_ablations"]) == {
        "without_product_statistics",
        "without_all_new_features",
        "without_small_experts",
    }
    assert report["frozen"]["selected_before_confirmation"]
    assert report["frozen"]["strategy"]["chosen"] in ("simplex", "affine", "context")
    assert (args.output_dir / "frozen.json").stat().st_mtime_ns < (
        args.output_dir / "report.json"
    ).stat().st_mtime_ns
    assert json.loads((args.output_dir / "report.json").read_text()) == report
    args.refine_from = args.output_dir
    args.output_dir = tmp_path / "refined"
    refined = study.run(args)
    assert len(refined["models"]) == 7
    assert refined["parent_statistics_reused"]
    assert refined["frozen"]["promotion_decided_before_confirmation"]
    assert refined["confirmation_rmsle"] >= 0
    assert set(refined["ablations"]) == {"without_hierarchical_views"}
    with pytest.raises(ValueError, match="empty"):
        study.run(args)
    old_path = args.baseline
    args.baseline = tmp_path / "other.pkl"
    args.baseline.write_bytes(b"not a trusted model")
    args.output_dir = tmp_path / "refine-wrong-model"
    with pytest.raises(ValueError, match="baseline"):
        study.run(args)
    args.baseline = old_path
    from benchmarks import mercari_multiscale as multiscale

    read_csv = multiscale.pd.read_csv

    def reversed_ids(path, **kwargs):
        result = read_csv(path, **kwargs)
        return result.iloc[::-1] if path.name == "selection.tsv.gz" else result

    monkeypatch.setattr(multiscale.pd, "read_csv", reversed_ids)
    args.output_dir = tmp_path / "refine-wrong-rows"
    with pytest.raises(ValueError, match="feature rows"):
        study.run(args)

    def leaked_prices(path, **kwargs):
        result = read_csv(path, **kwargs)
        return (
            result.assign(price=123.0) if path.name == "confirmation.tsv.gz" else result
        )

    monkeypatch.setattr(multiscale.pd, "read_csv", leaked_prices)
    args.output_dir = tmp_path / "refine-leaked-labels"
    with pytest.raises(ValueError, match="confirmation prices"):
        study.run(args)
    monkeypatch.setattr(multiscale.pd, "read_csv", read_csv)
    raw_evaluation = (data / "evaluation.npz").read_bytes()
    original_refine_dump = multiscale.dump

    def corrupt_refine_audit(path, payload):
        original_refine_dump(path, payload)
        if path.name == "frozen.json":
            np.savez(
                data / "evaluation.npz",
                confirmation=np.zeros(30),
                confirmation_ids=np.arange(170, 200)[::-1],
            )

    monkeypatch.setattr(multiscale, "dump", corrupt_refine_audit)
    args.output_dir = tmp_path / "refine-wrong-audit"
    with pytest.raises(ValueError, match="audit IDs"):
        study.run(args)
    (data / "evaluation.npz").write_bytes(raw_evaluation)
    args.refine_from = None
    args.output_dir = tmp_path / "study"
    with pytest.raises(ValueError, match="empty"):
        study.run(args)
    frames["selection"] = frames["selection"].iloc[::-1]
    with pytest.raises(ValueError, match="IDs"):
        study.load_small(small, frames)
    args.output_dir = tmp_path / "wrong-protocol"
    (data / "protocol.json").write_text(json.dumps({"baseline_sha256": "wrong"}))
    with pytest.raises(ValueError, match="protocol"):
        study.run(args)
    (data / "protocol.json").unlink()
    original = study.decompose
    monkeypatch.setattr(
        study, "decompose", lambda entry, frame: dict(reference=np.zeros(len(frame)))
    )
    args.output_dir = tmp_path / "wrong-replay"
    with pytest.raises(ValueError, match="reconstruction"):
        study.run(args)
    monkeypatch.setattr(study, "decompose", original)
    original_dump = study.dump

    def poison(path, payload):
        original_dump(path, payload)
        if path.name == "frozen.json":
            np.savez(
                data / "evaluation.npz",
                confirmation=np.ones(30),
                confirmation_ids=np.arange(170, 200)[::-1],
            )

    monkeypatch.setattr(study, "dump", poison)
    args.output_dir = tmp_path / "wrong-audit"
    args.small_experts = None
    with pytest.raises(ValueError, match="audit IDs"):
        study.run(args)
    assert (args.output_dir / "frozen.json").exists()
    assert not (args.output_dir / "report.json").exists()


def test_cli_cpu_limits_timeout_and_worker_dispatch(tmp_path, monkeypatch):
    import subprocess

    args = [
        "--data-dir",
        str(tmp_path / "data"),
        "--baseline",
        str(tmp_path / "model"),
        "--output-dir",
        str(tmp_path / "out"),
        "--max-seconds",
        "1",
        "--threads",
        "2",
    ]
    calls = []

    def child(command, **kwargs):
        calls.append((command, kwargs))
        return Namespace(returncode=0)

    monkeypatch.setattr(study.subprocess, "run", child)
    assert study.main(args) == 0
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert calls[0][1]["env"]["NUMBA_NUM_THREADS"] == "2"

    def expired(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 1)

    monkeypatch.setattr(study.subprocess, "run", expired)
    assert study.main(args) == 124
    assert (tmp_path / "out" / "budget-exhausted.json").exists()
    monkeypatch.setattr(study, "run", lambda a: calls.append(a))
    assert study.main(args + ["--worker"]) == 0
    with pytest.raises(ValueError):
        study.main(args + ["--max-seconds", "0"])
    with pytest.raises(ValueError):
        study.main(args + ["--tree-rounds", "0"])
    monkeypatch.setattr(sys, "argv", ["multiview", *args, "--worker"])
    assert study.main() == 0
    monkeypatch.setattr(sys, "argv", ["multiview", "--help"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("benchmarks.mercari_multiview_study", run_name="__main__")
    assert caught.value.code == 0
