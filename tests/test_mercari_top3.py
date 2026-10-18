"""Top-three ideas: interactions, numeric meaning and a frozen audit boundary."""

from argparse import Namespace
import json
import pickle
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from benchmarks import mercari_top3 as top3
from test_mercari_cpu_study import inputs


def test_lasso_and_elastic_net_have_builtin_sample_normalized_penalties():
    from sklearn.linear_model import ElasticNet, Lasso

    models = top3.sparse_penalty_models(1000)
    assert set(models) == {"lasso", "elasticnet"}
    assert isinstance(models["lasso"], Lasso)
    assert isinstance(models["elasticnet"], ElasticNet)
    assert models["lasso"].alpha == 0.02
    assert models["elasticnet"].alpha == 0.04
    assert models["elasticnet"].l1_ratio == 0.5
    larger = top3.sparse_penalty_models(2000)
    assert larger["lasso"].alpha == models["lasso"].alpha / 2
    for rows in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            top3.sparse_penalty_models(rows)


def test_sparse_penalties_learn_select_features_and_reload_without_densifying(
    monkeypatch,
):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(500, 12)).astype(np.float32)
    x = sparse.csc_matrix(raw)
    y = 10 + 4 * raw[:, 0] - 2 * raw[:, 1]
    heldout = rng.normal(size=(30, 12)).astype(np.float32)
    query = sparse.csr_matrix(heldout)
    expected = 10 + 4 * heldout[:, 0] - 2 * heldout[:, 1]

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse regularization must not densify the text matrix")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    for model in top3.sparse_penalty_models(len(y)).values():
        model.fit(x, y)
        prediction = model.predict(query)
        assert np.sqrt(np.mean((prediction - expected) ** 2)) < 0.3
        assert np.count_nonzero(model.coef_) == 2
        np.testing.assert_array_equal(
            prediction, pickle.loads(pickle.dumps(model)).predict(query)
        )


def test_canonical_text_and_bounded_order_invariant_pairs():
    assert top3.tokens("CAFÉ iPhone6+ w/ 2.5oz ❤") == [
        "cafe",
        "iphone",
        "6",
        "plus",
        "with",
        "2.5",
        "oz",
        "heart",
    ]
    assert top3.tokens(None) == top3.tokens(np.nan) == []
    assert top3.tokens("one two three", limit=2) == ["one", "two"]
    a = top3.pairs(["red", "shoe", "red"], ["new", "extra"], 1, 20)
    b = top3.pairs(["shoe", "red"], ["new", "ignored"], 1, 20)
    assert a == b == [("new\x1fred", 1.0), ("new\x1fshoe", 1.0), ("red\x1fshoe", 1.0)]
    assert len(top3.pairs(list("abcdef"), [], 0, 3)) == 3
    assert top3.pairs([], [], 0, 3) == []


def test_quantity_features_keep_amount_and_word_together():
    result = dict(
        top3.quantities(top3.tokens("10 cards 5 dolls 2.5 oz 0 shoes 201 toys 3"))
    )
    assert set(result) == {"cards", "dolls", "oz"}
    assert result["cards"] == np.log1p(10.0)
    assert result["dolls"] == np.log1p(5.0)
    assert result["oz"] == np.log1p(2.5)


def test_numeric_meaning_generalizes_to_an_unseen_amount():
    from sklearn.linear_model import Ridge

    frame = pd.DataFrame(
        {
            "name": ["2 cards", "5 cards", "9 cards"],
            "item_description": "",
            "brand_name": "x",
            "category_name": "a/b/c",
            "item_condition_id": 1,
            "shipping": 0,
        }
    )
    encoder = top3.InteractionFeatures(n_features=4096)
    x = encoder.fit_transform(frame)["quantities"]
    model = Ridge(alpha=1e-6, solver="lsqr", tol=1e-8).fit(x, np.log1p([2.0, 5.0, 9.0]))
    query = frame.iloc[:1].assign(name="3 cards")
    assert model.predict(encoder.transform(query)["quantities"])[0] == pytest.approx(
        np.log1p(3.0), abs=1e-5
    )


def test_sparse_blocks_ignore_labels_and_other_prediction_rows():
    frame = pd.DataFrame(
        {
            "name": ["10 red cards", None, "iphone 6 plus"],
            "item_description": ["5 blue cards", "", "CAFÉ w/ ❤"],
            "brand_name": ["x", None, "apple"],
            "category_name": ["a/b/c", None, "d/e/f"],
            "item_condition_id": [1, 2, 3],
            "shipping": [0, 1, 0],
            "price": [5, 6, 7],
            "train_id": [1, 2, 3],
        }
    )
    model = top3.InteractionFeatures(n_features=128)
    with pytest.raises(NotFittedError):
        model.transform(frame)
    blocks = model.fit_transform(frame)
    model.fit(frame)
    assert set(blocks) == {"lexical", "pairs", "quantities"}
    restored = pickle.loads(pickle.dumps(model))
    for name, block in blocks.items():
        assert sparse.isspmatrix_csr(block) and block.shape == (3, 128)
        assert block.dtype == np.float32 and np.isfinite(block.data).all()
        np.testing.assert_array_equal(
            block.toarray(),
            restored.transform(frame.assign(price=1e9, train_id=-1))[name].toarray(),
        )
        np.testing.assert_array_equal(
            block[:1].toarray(), model.transform(frame.iloc[:1])[name].toarray()
        )
    assert blocks["quantities"].getnnz(axis=1).tolist() == [2, 0, 1]
    for kwargs in (
        {"n_features": 0},
        {"max_tokens": 0},
        {"pair_tokens": 1},
        {"description_prefix": -1},
    ):
        with pytest.raises(ValueError, match="positive"):
            top3.InteractionFeatures(**kwargs).fit(frame)


def settings(tmp_path):
    data = inputs(tmp_path)
    return Namespace(
        data_dir=data,
        anchor=data / "baseline.npz",
        output_dir=tmp_path / "study",
        n_features=256,
        max_tokens=60,
        pair_tokens=20,
        description_prefix=5,
        alpha=20.0,
        max_iter=80,
        tol=0.002,
        threads=1,
        rows=0,
        seed=42,
        minimum_gain=1e-5,
        external_pretrained_data=False,
    )


def test_real_factorial_experiment_freezes_before_reading_confirmation(
    tmp_path, monkeypatch
):
    args = settings(tmp_path)
    original = top3.dump
    frozen_seen = []

    def observe(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            frozen_seen.append(True)
            # Data loader opens the label archive only after this callback.
            assert "confirmation_rmsle" not in json.dumps(payload)

    monkeypatch.setattr(top3, "dump", observe)
    report = top3.run(args)
    assert frozen_seen == [True]
    assert set(report["models"]) == {
        "lexical",
        "pairs",
        "quantities",
        "both",
        "lasso_pairs",
        "elasticnet_pairs",
    }
    assert report["models"]["lasso_pairs"]["family"] == "lasso"
    assert report["models"]["elasticnet_pairs"]["family"] == "elasticnet"
    for model in report["models"].values():
        assert model["predict_seconds"] == np.mean(model["predict_seconds_samples"])
        assert len(model["predict_seconds_samples"]) == 3
        assert 0 <= model["active_coefficients"] <= model["features"]
    assert set(report["ablations"]) == {
        "anchor",
        "without_pairs",
        "without_quantities",
        "full",
        "ridge_only",
        "without_lasso",
        "without_elasticnet",
        "raw_only",
    }
    assert report["training_rows"] == 120 and report["gpu_used"] is False
    assert report["confirmation_previously_inspected"]
    residual = report["residual"]
    assert set(residual["retained_raw_experts"]) == set(report["models"])
    assert not residual["standalone_quality_gate"]
    assert not residual["anchor_used_as_calibration_feature"]
    assert residual["correction_fit_window"] == "calibration"
    assert len(residual["models"]) == 8
    for pool, excluded in (
        ("without_lasso", "lasso_pairs"),
        ("without_elasticnet", "elasticnet_pairs"),
    ):
        for kind in ("linear", "boosting"):
            assert (
                excluded not in residual["models"][f"residual_{pool}_{kind}"]["experts"]
            )
    assert report["frozen"]["promotion_decided_before_confirmation"]
    assert report["submission_status"] == "not_submitted"
    assert json.loads((args.output_dir / "report.json").read_text()) == report
    assert (args.output_dir / "frozen.json").stat().st_mtime_ns < (
        args.output_dir / "report.json"
    ).stat().st_mtime_ns
    with np.load(args.output_dir / "ensemble.npz") as archive:
        assert archive["confirmation"].shape == (30,)
    with pytest.raises(ValueError, match="empty"):
        top3.run(args)


def test_study_rejects_wrong_anchor_and_misaligned_audit(tmp_path, monkeypatch):
    args = settings(tmp_path)
    raw = args.anchor.read_bytes()
    np.savez(
        args.anchor,
        **{k: np.zeros(1) for k in top3.WINDOWS},
        **{k + "_ids": np.zeros(1) for k in top3.WINDOWS},
    )
    with pytest.raises(ValueError, match="anchor"):
        top3.run(args)
    args.anchor.write_bytes(raw)
    args.output_dir = tmp_path / "bad-audit"
    original = top3.dump

    def poison(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            np.savez(
                args.data_dir / "evaluation.npz",
                confirmation=np.zeros(30),
                confirmation_ids=np.arange(30)[::-1],
            )

    monkeypatch.setattr(top3, "dump", poison)
    with pytest.raises(ValueError, match="audit"):
        top3.run(args)


def test_failed_selection_gate_preserves_the_exact_anchor(tmp_path):
    args = settings(tmp_path)
    args.minimum_gain = 10.0
    report = top3.run(args)
    assert not report["frozen"]["accepted"]
    with np.load(args.anchor) as old, np.load(args.output_dir / "ensemble.npz") as new:
        for window in top3.WINDOWS:
            np.testing.assert_array_equal(old[window], new[window])


def test_coordinate_descent_nonconvergence_is_retained_in_the_report(
    tmp_path, monkeypatch
):
    original = top3.sparse_penalty_models

    def one_pass(rows):
        models = original(rows)
        for model in models.values():
            model.set_params(alpha=1e-12, max_iter=1, tol=1e-12)
        return models

    monkeypatch.setattr(top3, "sparse_penalty_models", one_pass)
    report = top3.run(settings(tmp_path))
    for name in ("lasso_pairs", "elasticnet_pairs"):
        assert not report["models"][name]["converged"]
        assert report["models"][name]["iteration_limit_reached"]
        assert report["models"][name]["fit_warnings"]


def test_cli_validation_and_executable_entry(tmp_path, monkeypatch):
    for option in ("--lasso-alpha", "--l1-ratio", "--elasticnet-alpha"):
        with pytest.raises(SystemExit):
            top3.main(
                [
                    "--data-dir",
                    str(tmp_path),
                    "--anchor",
                    "x",
                    "--output-dir",
                    "y",
                    option,
                    ".1",
                ]
            )
    for option in [
        "--alpha",
        "--threads",
        "--tol",
        "--max-iter",
        "--n-features",
        "--max-tokens",
    ]:
        with pytest.raises(SystemExit):
            top3.main(
                [
                    "--data-dir",
                    str(tmp_path),
                    "--anchor",
                    "x",
                    "--output-dir",
                    "y",
                    option,
                    "0",
                ]
            )
    args = settings(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mercari_top3",
            "--data-dir",
            str(args.data_dir),
            "--anchor",
            str(args.anchor),
            "--output-dir",
            str(args.output_dir),
            "--n-features",
            "256",
            "--threads",
            "1",
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("benchmarks.mercari_top3", run_name="__main__")
    assert (args.output_dir / "report.json").exists()
    with (args.output_dir / "encoder.pkl").open("rb") as stream:
        assert type(pickle.load(stream)) is top3.InteractionFeatures
