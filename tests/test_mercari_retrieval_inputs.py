"""Verified train-only inputs for joint text-trunk and comparable learning."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import mercari, mercari_retrieval_inputs as inputs
from benchmarks.mercari_features import ListingFeatures
from benchmarks.mercari_neural import Tokens
from test_mercari import listings


def fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    frames = {
        k: listings(36).assign(train_id=np.arange(36) + i * 100)
        for i, k in enumerate(inputs.WINDOWS)
    }
    for k, f in frames.items():
        (f.drop(columns="price") if k == "confirmation" else f).to_csv(
            data / (k + ".tsv.gz"), sep="\t", index=False
        )
    listing = ListingFeatures().fit(frames["train"])
    tokens = Tokens(max_words=100, title_length=8, description_length=8).fit(
        listing.transform(frames["train"])
    )
    (model / "tokens.pkl").write_bytes(
        pickle.dumps((listing, tokens, float(np.log1p(frames["train"].price).mean())))
    )
    (model / "cnn128_mse.keras").write_bytes(b"trusted model hash fixture")
    ref = tmp_path / "reference.npz"
    np.savez_compressed(
        ref,
        **{k: np.full(len(f), 3.0) for k, f in frames.items()},
        **{k + "_ids": f.train_id.to_numpy() for k, f in frames.items()},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                data_sha256={k: mercari.sha256(data / (k + ".tsv.gz")) for k in frames},
                experts=[
                    dict(
                        name="frontier_cnn128_mse",
                        path=str(ref),
                        sha256=mercari.sha256(ref),
                        fit_window="base",
                    )
                ],
            )
        )
    )
    return SimpleNamespace(
        data_dir=data, model_dir=model, manifest=manifest, cache_dir=tmp_path / "cache"
    )


def test_identity_preserves_model_and_units_without_price_or_query_context():
    f = listings(3).assign(
        name=["iPhone6S 64 GB", "iphone6s 64gb", "plain shirt"], item_description=""
    )
    a = inputs.identity_tokens(f)
    np.testing.assert_array_equal(a[0], a[1])
    assert a.shape == (3, 16) and a[0].max() > 0 and a[2].max() == 0
    np.testing.assert_array_equal(
        a, inputs.identity_tokens(f.assign(price=-999, train_id=-1))
    )


def test_cache_hashes_order_and_train_only_scaling(tmp_path):
    args = fixture(tmp_path)
    plan = inputs.prepare(args)
    loaded, arrays = inputs.load(args)
    assert plan == loaded and loaded["rows"]["train"] == 36
    assert arrays["train"]["context"].shape == (36, 28)
    assert "y" not in arrays["confirmation"] and "y" in arrays["train"]
    np.testing.assert_array_equal(arrays["train"]["ids"], np.arange(36))
    np.testing.assert_array_equal(
        np.load(args.cache_dir / "context-mean.npy"), arrays["train"]["context"].mean(0)
    )
    with pytest.raises(FileExistsError):
        inputs.prepare(args)
    model_file = args.model_dir / "cnn128_mse.keras"
    original = model_file.read_bytes()
    model_file.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="identity"):
        inputs.load(args)
    model_file.write_bytes(original)
    path = args.cache_dir / "train-identity.npy"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        inputs.load(args)


@pytest.mark.parametrize("bad", ["data", "provenance", "audit_price", "ids"])
def test_prepare_rejects_mismatched_protocol(tmp_path, bad):
    args = fixture(tmp_path)
    m = json.loads(args.manifest.read_text())
    if bad == "data":
        m["data_sha256"]["train"] = "wrong"
    if bad == "provenance":
        m["experts"][0]["fit_window"] = "selection"
    if bad == "audit_price":
        import pandas as pd

        p = args.data_dir / "confirmation.tsv.gz"
        f = pd.read_csv(p, sep="\t").assign(price=100)
        f.to_csv(p, sep="\t", index=False)
        m["data_sha256"]["confirmation"] = mercari.sha256(p)
    if bad == "ids":
        p = args.manifest.parent / "reference.npz"
        with np.load(p) as a:
            v = {k: a[k] for k in a.files}
        v["calibration_ids"] = v["calibration_ids"][::-1]
        np.savez_compressed(p, **v)
        m["experts"][0]["sha256"] = mercari.sha256(p)
    args.manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        inputs.prepare(args)
