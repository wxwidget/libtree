"""TDD for observable market cues and probabilistic pre-fusion price heads."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from benchmarks import mercari_market as market
from test_mercari import listings


def test_market_cues_preserve_negation_units_and_bundle_quantity():
    healthy = market.market_cues("No cracks or scratches. Set of three cases 64 GB")
    damaged = market.market_cues("Cracked phone does not work; box only, pack of 4")
    assert healthy["damage"] == 0 and healthy["damage_negated"] == 1
    assert healthy["accessory"] == 1 and healthy["quantity"] == pytest.approx(
        np.log1p(3)
    )
    assert healthy["storage_gb"] == pytest.approx(np.log1p(64))
    assert damaged["damage"] == damaged["not_working"] == damaged["box_only"] == 1
    assert damaged["quantity"] == pytest.approx(np.log1p(4))
    assert market.market_cues(None)["quantity"] == pytest.approx(np.log1p(1))
    assert market.market_cues("pair of socks")["quantity"] == pytest.approx(np.log1p(2))
    assert market.market_cues("lot of 999999 pieces")["quantity"] == pytest.approx(
        np.log1p(512)
    )
    frame = listings(12)
    encoder = market.MarketSignals()
    x = encoder.transform(frame)
    assert x.shape == (12, len(encoder.feature_names_)) and x.dtype == np.float32
    np.testing.assert_array_equal(
        x, encoder.transform(frame.assign(price=-999, train_id=-1))
    )
    np.testing.assert_array_equal(x[:1], encoder.transform(frame.iloc[:1]))
    np.testing.assert_array_equal(
        x, pickle.loads(pickle.dumps(encoder)).transform(frame)
    )


def test_log_mean_uses_soft_probabilities_not_raw_price_mean():
    probabilities = np.array([[0.25, 0.75, 0, 0], [0, 0, 0.5, 0.5]])
    means = np.array([[0, np.log1p(5), np.log1p(20), np.log1p(100)]] * 2)
    actual = market.expected_log_price(probabilities, means)
    np.testing.assert_allclose(actual, (probabilities * means).sum(1))
    assert np.expm1(actual[0]) != pytest.approx(0.75 * 5)
    for p, m in [
        (probabilities[:1], means),
        (probabilities * np.nan, means),
        (-probabilities, means),
        (probabilities * 2, means),
        (probabilities, -means),
    ]:
        with pytest.raises(ValueError):
            market.expected_log_price(p, m)


def toy():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(240, 5)).astype(np.float32)
    y = np.choose(
        np.arange(len(x)) % 4, [0.0, np.log1p(5), np.log1p(25), np.log1p(100)]
    )
    x[:, 0] = np.arange(len(x)) % 4
    return x, y


def test_regime_model_is_fitted_soft_routed_serializable_and_target_free_at_predict():
    x, y = toy()
    model = market.RegimePriceHead(rounds=8, min_head_rows=5, min_child_samples=3)
    with pytest.raises(NotFittedError):
        model.predict(x)
    model.fit(x, y, validation=(x, y))
    result = model.predict(x)
    assert np.sqrt(np.mean((result - y) ** 2)) < 0.9 * np.std(y)
    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(model.predict(x[:1]), result[:1])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(x), result)
    assert not any(
        isinstance(v, np.ndarray) and v.shape == x.shape for v in vars(model).values()
    )
    for data, target in [
        (x, y[:-1]),
        (x, y * np.nan),
        (x, -np.ones(len(y))),
        (x[:, 0], y),
    ]:
        with pytest.raises(ValueError):
            market.RegimePriceHead().fit(data, target)
    for data in (x[:, :2], x * np.nan):
        with pytest.raises(ValueError):
            model.predict(data)
    for kwargs in [dict(rounds=0), dict(min_head_rows=0), dict(min_child_samples=0)]:
        with pytest.raises(ValueError):
            market.RegimePriceHead(**kwargs).fit(x, y)
    for v in [(x, y[:-1]), (x[:, :2], y)]:
        with pytest.raises(ValueError):
            market.RegimePriceHead().fit(x, y, validation=v)


def test_absent_and_rare_regimes_have_finite_fallback_and_frozen_budgets():
    x, y = toy()
    for target in (np.zeros(len(y)), np.full(len(y), np.log1p(5))):
        m = market.RegimePriceHead(rounds=3).fit(x, target)
        np.testing.assert_allclose(m.predict(x), target)
    m = market.RegimePriceHead(rounds=4, min_head_rows=999).fit(x, y, validation=(x, y))
    assert all(head is None for head in m.heads_)
    replay = market.RegimePriceHead(rounds=4, min_head_rows=999).fit(
        x, y, budgets=m.budgets_
    )
    np.testing.assert_allclose(replay.predict(x), m.predict(x))
    # Tune rows may contain a class absent in training; never encode it as a known class.
    m = market.RegimePriceHead(rounds=3, min_head_rows=5, min_child_samples=3).fit(
        x[y < 3.5], y[y < 3.5], validation=(x, y)
    )
    assert np.isfinite(m.predict(x)).all()


def test_frozen_zero_step_heads_do_not_grow_on_full_refit():
    x, y = toy()
    model = market.RegimePriceHead(rounds=3, min_head_rows=5).fit(
        x, np.full(len(y), np.log1p(5))
    )
    budget = dict(gate=0, heads=[0] * 4)
    full = market.RegimePriceHead(rounds=3, min_head_rows=5).fit(x, y, budgets=budget)
    assert full.gate_ is None and all(v is None for v in full.heads_)
    np.testing.assert_allclose(full.predict(x), y.mean())
    assert full.budgets_ == budget


def tiny_study(tmp_path):
    from benchmarks import mercari
    from benchmarks.mercari_semantics import LatentText
    import json

    data = tmp_path / "data"
    data.mkdir()
    frames = {
        k: listings(140).assign(train_id=np.arange(140) + i * 1000)
        for i, k in enumerate(("train", "calibration", "selection", "confirmation"))
    }
    for k, frame in frames.items():
        (frame.drop(columns="price") if k == "confirmation" else frame).to_csv(
            data / (k + ".tsv.gz"), sep="\t", index=False
        )
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
    archive = tmp_path / "base.npz"
    np.savez_compressed(
        archive,
        **{
            k: np.log1p(frames[k].price.to_numpy()) + 0.1 * np.sin(np.arange(140))
            for k in ("calibration", "selection", "confirmation")
        },
        **{
            k + "_ids": frames[k].train_id.to_numpy()
            for k in ("calibration", "selection", "confirmation")
        },
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                data_sha256={k: mercari.sha256(data / (k + ".tsv.gz")) for k in frames},
                experts=[
                    dict(
                        name="one_text",
                        path=str(archive),
                        sha256=mercari.sha256(archive),
                        fit_window="base",
                    )
                ],
            )
        )
    )
    return SimpleNamespace(
        data_dir=data,
        manifest=manifest,
        base_predictions=archive,
        encoder=encoder,
        output_dir=tmp_path / "output",
    )


def test_real_study_freezes_before_audit_and_retains_all_heads(tmp_path, monkeypatch):
    from benchmarks import mercari_market_study as study
    import json

    args = tiny_study(tmp_path)
    monkeypatch.setattr(study, "ROUNDS", 3)
    monkeypatch.setattr(study, "REGIME_ROUNDS", 3)
    original = np.load

    def checked(path, *a, **kw):
        if Path(path).name == "evaluation.npz":
            frozen = json.loads((args.output_dir / "frozen.json").read_text())
            assert frozen["all_choices_frozen_before_audit"]
            assert all(
                "confirmation_rmsle" not in row for row in frozen["models"].values()
            )
        return original(path, *a, **kw)

    from pathlib import Path

    monkeypatch.setattr(study.np, "load", checked)
    result = study.run(args)
    assert len(result["models"]) == 8 and len(result["comparisons"]) == 7
    assert result["final_fusion_modified"] is False
    assert result["frozen"]["plan"]["base_experts_per_input"] == 1
    assert result["frozen"]["plan"]["comparable_statistics_fit_window"] == "base"
    assert (
        result["models"]["comparables_l2"]["features"]
        == result["models"]["market_l2"]["features"] + 15
    )
    for name, row in result["models"].items():
        with original(args.output_dir / (name + ".npz")) as f:
            assert f["confirmation"].shape == (140,)
        assert sum(b["n"] for b in row["confirmation_buckets"]) == 140
    assert (
        study.bucket_metrics(np.zeros(2), np.zeros(2))[0]["log_squared_error_share"]
        == 0
    )
    # Normal CLI path executes the same governed runner, without model tuning flags.
    saved = study.run
    monkeypatch.setattr(study, "run", lambda options: options)
    parsed = study.main(
        [
            "--data-dir",
            str(args.data_dir),
            "--manifest",
            str(args.manifest),
            "--base-predictions",
            str(args.base_predictions),
            "--encoder",
            str(args.encoder),
            "--output-dir",
            str(args.output_dir),
        ]
    )
    assert parsed.base_predictions == args.base_predictions
    monkeypatch.setattr(study, "run", saved)


@pytest.mark.parametrize(
    "corruption",
    [
        "data_hash",
        "base_provenance",
        "confirmation_price",
        "base_ids",
        "base_values",
        "audit_ids",
    ],
)
def test_study_rejects_untrusted_or_misaligned_inputs(
    tmp_path, monkeypatch, corruption
):
    from benchmarks import mercari_market_study as study
    from benchmarks import mercari
    import json

    args = tiny_study(tmp_path)
    monkeypatch.setattr(study, "ROUNDS", 1)
    monkeypatch.setattr(study, "REGIME_ROUNDS", 1)
    manifest = json.loads(args.manifest.read_text())
    if corruption == "data_hash":
        manifest["data_sha256"]["train"] = "wrong"
    if corruption == "base_provenance":
        manifest["experts"][0]["fit_window"] = "selection"
    if corruption == "confirmation_price":
        part = pd.read_csv(args.data_dir / "confirmation.tsv.gz", sep="\t").assign(
            price=5
        )
        part.to_csv(args.data_dir / "confirmation.tsv.gz", sep="\t", index=False)
        manifest["data_sha256"]["confirmation"] = mercari.sha256(
            args.data_dir / "confirmation.tsv.gz"
        )
    if corruption in ("base_ids", "base_values"):
        with np.load(args.base_predictions) as f:
            archive = {k: f[k] for k in f.files}
        if corruption == "base_ids":
            archive["calibration_ids"] = archive["calibration_ids"][::-1]
        else:
            archive["calibration"] = np.full(len(archive["calibration"]), np.nan)
        np.savez_compressed(args.base_predictions, **archive)
        manifest["experts"][0]["sha256"] = mercari.sha256(args.base_predictions)
    if corruption == "audit_ids":
        with np.load(args.data_dir / "evaluation.npz") as f:
            archive = {k: f[k] for k in f.files}
        archive["confirmation_ids"] = archive["confirmation_ids"][::-1]
        np.savez_compressed(args.data_dir / "evaluation.npz", **archive)
    args.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        study.run(args)


def test_cli_module_help(monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["mercari-market-study", "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("benchmarks.mercari_market_study", run_name="__main__")
    assert stopped.value.code == 0
