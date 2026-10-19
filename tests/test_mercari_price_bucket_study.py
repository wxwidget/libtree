"""Real classifier integration and frozen-before-audit protocol guards."""

import importlib.metadata
import json
import pickle
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import mercari
from benchmarks.mercari_semantics import LatentText
from benchmarks import mercari_price_bucket_study as study
from test_mercari import listings


def fixture(tmp_path):
    data, context = tmp_path / "data", tmp_path / "context"
    data.mkdir()
    context.mkdir()
    frames = {}
    for i, window in enumerate(study.WINDOWS):
        f = listings(100).assign(
            train_id=np.arange(100) + 1000 * i,
            price=np.tile([0.0, 3.0, 5.0, 7.0, 9.0, 12.0, 16.0, 22.0, 35.0, 70.0], 10),
        )
        f["item_description"] += " window " + str(i)
        frames[window] = f
        (f.drop(columns="price") if window == "confirmation" else f).to_csv(
            data / (window + ".tsv.gz"), sep="\t", index=False
        )
        np.save(context / (window + "-ids.npy"), f.train_id.to_numpy())
        np.save(
            context / (window + "-context.npy"),
            np.column_stack([np.arange(100) % 10, np.ones(100)]).astype(np.float32),
        )
    source = Path(study.__file__).with_name("mercari_retrieval.py")
    manifest = dict(
        identity=dict(
            data_sha256={
                w: mercari.sha256(data / (w + ".tsv.gz")) for w in study.WINDOWS
            },
            source_sha256={source.name: mercari.sha256(source)},
            versions={"numpy": importlib.metadata.version("numpy")},
        ),
        query_duplicate_group_excluded=True,
        all_priors_leave_group_out=True,
        feature_names=["comparable_a", "comparable_b"],
        files={p.name: mercari.sha256(p) for p in context.iterdir()},
    )
    (context / "manifest.json").write_text(json.dumps(manifest))
    np.savez_compressed(
        data / "evaluation.npz",
        confirmation=np.log1p(frames["confirmation"].price),
        confirmation_ids=frames["confirmation"].train_id,
    )
    encoder = tmp_path / "encoder.pkl"
    encoder.write_bytes(
        pickle.dumps(
            LatentText(components=2, max_features=100, min_df=1).fit(frames["train"])
        )
    )
    regression = tmp_path / "regression.npz"
    np.savez_compressed(
        regression,
        **{w: np.log1p(frames[w].price.to_numpy()) + 0.03 for w in study.EVALUATION},
        **{w + "_ids": frames[w].train_id.to_numpy() for w in study.EVALUATION},
    )
    return SimpleNamespace(
        data_dir=data,
        context_dir=context,
        encoder=encoder,
        regression=[regression],
        output_dir=tmp_path / "run",
    )


def test_real_runner_uses_base_quantiles_and_freezes_before_audit(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    monkeypatch.setattr(study, "ROUNDS", 3)
    thin = tmp_path / "historical_fusion.npz"
    with np.load(args.regression[0]) as source:
        np.savez_compressed(
            thin,
            **{k: source[k] for k in source.files if not k.startswith("calibration")},
        )
    args.regression.append(thin)
    original = np.load

    def checked(path, *a, **kw):
        if Path(path).name == "evaluation.npz":
            frozen = json.loads((args.output_dir / "frozen.json").read_text())
            assert frozen["choices_frozen_before_audit"]
            assert all("confirmation" not in m for m in frozen["models"].values())
        return original(path, *a, **kw)

    monkeypatch.setattr(study.np, "load", checked)
    report = study.run(args)
    assert report["models"]["majority"]["confirmation"]["rows"] == 100
    assert len(report["models"]) == 7
    assert "calibration" not in report["models"]["regression_1"]
    assert report["plan"]["zero_is_separate"]
    assert report["plan"]["quantile_fit_window"] == "train"
    for name in study.CLASSIFIERS:
        row = report["models"][name]
        assert row["serialized_replay_max_error"] == 0
        assert 1 <= row["rounds"] <= 3
        assert len(row["inference_samples"]) == 3
        with original(args.output_dir / (name + ".npz")) as archive:
            assert "confirmation_probability" in archive
            assert "confirmation" not in archive
        assert sum(row["confirmation"]["confusion_matrix"][0]) == 10
    assert all("confirmation" not in m for m in report["frozen"]["models"].values())


@pytest.mark.parametrize(
    "bad",
    [
        "data",
        "source",
        "cache",
        "ids",
        "audit_price",
        "groups",
        "flags",
        "versions",
        "features",
        "regression",
        "regression_ids",
        "audit_ids",
    ],
)
def test_runner_rejects_bad_provenance_and_label_protocol(tmp_path, monkeypatch, bad):
    args = fixture(tmp_path)
    manifest = json.loads((args.context_dir / "manifest.json").read_text())
    if bad == "data":
        manifest["identity"]["data_sha256"]["train"] = "wrong"
    if bad == "source":
        manifest["identity"]["source_sha256"]["mercari_retrieval.py"] = "wrong"
    if bad == "cache":
        manifest["files"]["train-context.npy"] = "wrong"
    if bad == "flags":
        manifest["query_duplicate_group_excluded"] = False
    if bad == "versions":
        manifest["identity"]["versions"]["numpy"] = "wrong"
    if bad == "features":
        manifest["feature_names"].append("unexpected")
    if bad in ("audit_price", "groups"):
        import pandas as pd

        path = args.data_dir / "confirmation.tsv.gz"
        frame = pd.read_csv(path, sep="\t")
        if bad == "audit_price":
            frame["price"] = 1.0
        else:
            frame["item_description"] = frame.item_description.str.replace(
                "window 3", "window 0"
            )
        frame.to_csv(path, sep="\t", index=False)
        manifest["identity"]["data_sha256"]["confirmation"] = mercari.sha256(path)
    if bad == "ids":
        path = args.context_dir / "train-ids.npy"
        np.save(path, np.arange(100)[::-1])
        manifest["files"][path.name] = mercari.sha256(path)
    if bad == "regression":
        args.regression[0].write_bytes(b"invalid archive")
    if bad == "regression_ids":
        np.savez_compressed(
            args.regression[0],
            **{w: np.ones(100) for w in study.EVALUATION},
            **{w + "_ids": np.arange(100)[::-1] for w in study.EVALUATION},
        )
    if bad == "audit_ids":
        np.savez_compressed(
            args.data_dir / "evaluation.npz",
            confirmation=np.ones(100),
            confirmation_ids=np.arange(100),
        )
        monkeypatch.setattr(study, "ROUNDS", 1)
    (args.context_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        study.run(args)


def test_cli_help_and_actual_argument_mapping(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["price-buckets", "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("benchmarks.mercari_price_bucket_study", run_name="__main__")
    assert stopped.value.code == 0
    monkeypatch.setattr(study, "run", lambda a: a)
    parsed = study.main(
        [
            "--data-dir",
            "data",
            "--encoder",
            "e.pkl",
            "--context-dir",
            "context",
            "--output-dir",
            "output",
        ]
    )
    assert parsed.regression == [] and parsed.context_dir == Path("context")
