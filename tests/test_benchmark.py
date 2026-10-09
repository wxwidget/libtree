"""Behavioral guards for leakage-safe benchmark preparation and cache integrity."""
import numpy as np
import pandas as pd
import pytest
from benchmarks import run


def test_memory_counter_uses_current_address_space():
    assert run.peak_rss_kib() > 0


def test_numeric_only_preparation_and_seed():
    x = pd.DataFrame({"value": np.arange(80), "missing": [np.nan] * 80})
    y = np.arange(80, dtype=np.float32)
    first = run.prepare(x, y, False, seed=42)
    second = run.prepare(x, y, False, seed=42)
    other = run.prepare(x, y, False, seed=2024)
    assert first[0].shape == (60, 2)
    assert first[1].shape == (20, 2)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[2], other[2])
    assert np.isnan(first[0]).sum() == 60


def test_test_only_category_is_not_learned():
    x = pd.DataFrame({"category": ["common"] * 40})
    y = np.arange(40)
    _, _, _, test_y = run.prepare(x, y, False, seed=42)
    x.loc[test_y[0], "category"] = "test-only"
    train, test, _, _ = run.prepare(x, y, False, seed=42)
    assert train.shape[1] == test.shape[1] == 1
    assert (test == 0).any()


def test_corrupt_cached_dataset_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE", tmp_path)
    (tmp_path / "pima.csv").write_text("corrupted data")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run.source("pima")
