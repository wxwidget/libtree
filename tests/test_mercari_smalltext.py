"""Real CPU learning and offline, bounded-worker contract tests."""

import json
import runpy
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from benchmarks import mercari_smalltext as small
from benchmarks import mercari_transformer as neural
from test_mercari import listings
from test_mercari_transformer import TinyTokenizer, backbone


def test_bag_learns_padding_invariant_replay_and_parameter_cap():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    frame = listings(20)
    inputs = neural.ListingInputs().fit(frame)
    x = inputs.transform(frame, TinyTokenizer(), length=12)
    model = small.SubwordBag(
        32, inputs.cardinalities_, x["numeric"].shape[1], 8, 16, 32
    )
    counts = small.parameter_counts(model, 100000)
    assert counts["total"] == counts["trainable"] < 100000
    with pytest.raises(ValueError, match="parameter budget"):
        small.parameter_counts(model, 1)
    batches = neural.loader(x, np.linspace(-0.3, 0.3, 20), batch_size=10)
    opt = neural.optimizer(model, backbone_rate=0.01, head_rate=0.01)
    first = neural.train_epoch(model, batches, opt, torch.device("cpu"), "mse")
    for _ in range(5):
        last = neural.train_epoch(model, batches, opt, torch.device("cpu"), "mse")
    assert last < first
    pred = neural.predict(model, x, torch.device("cpu"), 10)
    x["input_ids"][x["attention_mask"] == 0] = 31
    np.testing.assert_allclose(
        pred, neural.predict(model, x, torch.device("cpu"), 10), atol=1e-6
    )
    x["attention_mask"][:] = 0
    assert np.isfinite(neural.predict(model, x, torch.device("cpu"), 10)).all()
    clone = small.SubwordBag(
        32, inputs.cardinalities_, x["numeric"].shape[1], 8, 16, 32
    )
    clone.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(
        neural.predict(model, x, torch.device("cpu"), 10),
        neural.predict(clone, x, torch.device("cpu"), 10),
    )


@pytest.mark.parametrize("architecture", ["bag", "bert"])
def test_worker_freezes_on_selection_retains_replay_and_ids(
    tmp_path, monkeypatch, architecture
):
    data = tmp_path / "data"
    data.mkdir()
    for i, name in enumerate(("train", "calibration", "selection", "confirmation")):
        frame = listings(20).assign(train_id=np.arange(20) + i * 100)
        if name == "confirmation":
            frame = frame.drop(columns="price")
        frame.to_csv(data / (name + ".tsv.gz"), sep="\t", index=False)
    monkeypatch.setattr(
        small,
        "load_pretrained",
        lambda *_: (TinyTokenizer(), backbone(), "offline-revision"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = [
        "--data-dir",
        str(data),
        "--output-dir",
        str(tmp_path / "out"),
        "--architecture",
        architecture,
        "--epochs",
        "2",
        "--batch-size",
        "10",
        "--length",
        "12",
        "--width",
        "16",
        "--embedding-dim",
        "8",
        "--buckets",
        "32",
        "--threads",
        "1",
        "--max-parameters",
        "100000",
    ]
    with pytest.raises(RuntimeError, match="GPU"):
        small.main(args)
    small.main(args + ["--allow-cpu"])
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["confirmation_labels_loaded"] is False and report["gpu"] is None
    assert report["parameters"]["total"] < 100000
    assert report["revision"] == "offline-revision"
    assert len(report["history"]) == 2 and report["prediction_seconds"] > 0
    assert (tmp_path / "out/checkpoint.pt").is_file() and (
        tmp_path / "out/inputs.pkl"
    ).is_file()
    with np.load(tmp_path / "out" / (architecture + "_mse.npz")) as p:
        np.testing.assert_array_equal(p["confirmation_ids"], np.arange(20) + 300)
    with pytest.raises(ValueError, match="new output"):
        small.main(args + ["--allow-cpu"])
    with pytest.raises(SystemExit):
        small.main(args + ["--epochs", "0"])
    # Reject accidental confirmation labels even when all other inputs are valid.
    listings(20).to_csv(data / "confirmation.tsv.gz", sep="\t", index=False)
    with pytest.raises(ValueError, match="confirmation labels"):
        small.main([*args, "--output-dir", str(tmp_path / "bad"), "--allow-cpu"])


def test_module_help(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mercari_smalltext", "--help"])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("benchmarks.mercari_smalltext", run_name="__main__")
    assert error.value.code == 0
