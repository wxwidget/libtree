"""Immutable, verified input reuse; never cache fitted price regressors."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
from benchmarks import mercari_text_cache as cache
from benchmarks.mercari_transformer import ListingInputs
from test_mercari import listings
from test_mercari_transformer import TinyTokenizer


def test_real_input_cache_replays_exactly_and_rejects_mismatch_corruption_and_partial(
    tmp_path,
):
    data = tmp_path / "data"
    data.mkdir()
    frames = {}
    for name in ("train", "calibration", "selection", "confirmation"):
        frame = listings(8)
        if name == "confirmation":
            frame = frame.drop(columns="price")
        frame.to_csv(data / (name + ".tsv.gz"), sep="\t", index=False)
        frames[name] = frame
    key = cache.identity(data, "offline-bert", "fixed-revision", 12)
    inputs = ListingInputs().fit(frames["train"])
    encoded = {k: inputs.transform(v, TinyTokenizer(), 12) for k, v in frames.items()}
    store = cache.TextCache(tmp_path / "cache", key)
    assert store.load() is None
    store.save(inputs, encoded)
    restored, values = store.load()
    for window, arrays in encoded.items():
        for name, original in arrays.items():
            np.testing.assert_array_equal(values[window][name], original)
            assert values[window][name].flags.writeable
        for name, original in restored.transform(
            frames[window], TinyTokenizer(), 12
        ).items():
            np.testing.assert_array_equal(values[window][name], original)
    original_first = int(values["train"]["input_ids"][0, 0])
    values["train"]["input_ids"][0, 0] = original_first + 1
    assert int(store.load()[1]["train"]["input_ids"][0, 0]) == original_first
    with pytest.raises(FileExistsError):
        store.save(inputs, encoded)
    with pytest.raises(ValueError, match="identity"):
        cache.TextCache(store.directory, {**key, "length": 13}).load()
    changed = cache.identity(data, "offline-bert", "new-revision", 12)
    assert changed != key
    (data / "selection.tsv.gz").write_bytes(b"changed")
    assert cache.identity(data, "offline-bert", "fixed-revision", 12) != key
    p = store.directory / "confirmation-input_ids.npy"
    p.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        store.load()
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        cache.TextCache(partial, key).load()
