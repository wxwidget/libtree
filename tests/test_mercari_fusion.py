"""Three fusion strategies share predictions and separate meta-data roles."""

from argparse import Namespace
import copy
import json
import pickle
import runpy
import sys

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from benchmarks import mercari_fusion as fusion
from benchmarks.mercari_multiview import ContextBlend, RegularizedBlend
from test_mercari_cpu_study import inputs


def test_mlp_and_conditional_weights_learn_complementary_category_experts():
    rng = np.random.default_rng(17)
    p = 3 + rng.uniform(-1, 1, (4000, 2))
    g = np.where(np.arange(len(p)) % 2, "a", "b")
    y = np.where(g == "a", p[:, 0], p[:, 1])
    fit, tune, check = slice(0, 2500), slice(2500, 3200), slice(3200, None)
    global_model = RegularizedBlend().fit(p[fit], y[fit])
    conditional = ContextBlend(min_rows=100).fit(p[fit], y[fit], g[fit])
    mlp = fusion.MLPBlend(seed=42).fit(
        p[fit], y[fit], g[fit], validation=(p[tune], y[tune], g[tune])
    )
    assert np.sqrt(np.mean((global_model.predict(p[check]) - y[check]) ** 2)) > 0.3
    assert (
        np.sqrt(np.mean((conditional.predict(p[check], g[check]) - y[check]) ** 2))
        < 0.08
    )
    assert np.sqrt(np.mean((mlp.predict(p[check], g[check]) - y[check]) ** 2)) < 0.12
    assert mlp.model_.hidden_layer_sizes == (32, 16)
    assert mlp.selected_epochs_ <= len(mlp.history_)
    assert mlp.scaler_.mean_.shape == (2,)
    np.testing.assert_allclose(mlp.scaler_.mean_, p[fit].mean(axis=0))
    assert mlp.mean_ == pytest.approx(y[fit].mean())
    assert set(mlp.categories_.categories_[0]) == {"a", "b"}
    np.testing.assert_array_equal(
        mlp.predict(p[check], g[check]),
        pickle.loads(pickle.dumps(mlp)).predict(p[check], g[check]),
    )
    assert np.isfinite(mlp.predict([[3, 3]], ["unseen"])).all()


def test_mlp_fixed_epoch_refit_and_input_guards():
    model = fusion.MLPBlend()
    with pytest.raises(NotFittedError):
        model.predict([[3]], ["a"])
    for p, y, groups in (
        ([], [], []),
        ([[np.nan]], [3], ["a"]),
        ([[3]], [-1], ["a"]),
        ([[3]], [3, 4], ["a"]),
        ([[3]], [3], []),
    ):
        with pytest.raises(ValueError, match="finite|matching"):
            model.fit(p, y, groups, epochs=1)
    with pytest.raises(ValueError, match="validation"):
        model.fit([[3]], [3], ["a"])
    with pytest.raises(ValueError, match="epochs"):
        model.fit([[3]], [3], ["a"], epochs=0)
    with pytest.raises(ValueError, match="columns"):
        model.fit([[3]], [3], ["a"], validation=([[3, 4]], [3], ["a"]))
    model.fit([[2], [3]], [3, 3], ["a", "b"], epochs=2)
    assert model.selected_epochs_ == 2 and len(model.history_) == 2
    with pytest.raises(ValueError, match="columns"):
        model.predict([[3, 4]], ["a"])


def test_mlp_restores_best_checkpoint_before_stopping_on_deterioration(monkeypatch):
    def worsening(self, x):
        return np.full(len(x), max(0, self.t_ / 20 - 1))

    monkeypatch.setattr(fusion.MLPRegressor, "predict", worsening)
    p, y, g = np.ones((20, 2)), np.full(20, 3.0), np.full(20, "a")
    model = fusion.MLPBlend().fit(p, y, g, validation=(p, y, g))
    assert model.selected_epochs_ == 1 and len(model.history_) == 1 + fusion.PATIENCE
    np.testing.assert_array_equal(model.predict(p, g), y)


def settings(tmp_path):
    data = inputs(tmp_path)
    args = Namespace(
        data_dir=data,
        anchor=data / "baseline.npz",
        manifest=tmp_path / "candidates.json",
        output_dir=tmp_path / "result",
        rows=0,
        seed=42,
    )
    frames = fusion.read_data(args)
    packed = tmp_path / "packed.npz"
    np.savez(
        packed,
        **{
            k: np.column_stack(
                [
                    np.log1p(
                        v.price if "price" in v else 10 + v.train_id.to_numpy() % 20
                    )
                    + 0.01,
                    np.full(len(v), 2.9),
                ]
            )
            for k, v in frames.items()
            if k != "train"
        },
        **{
            k + "_ids": v.train_id.to_numpy() for k, v in frames.items() if k != "train"
        },
    )
    manifest = dict(
        data_sha256={k: fusion.mercari.sha256(data / (k + ".tsv.gz")) for k in frames},
        experts=[
            dict(
                name=name,
                path=str(packed),
                sha256=fusion.mercari.sha256(packed),
                column=i,
                family=family,
                fit_window="base",
            )
            for i, (name, family) in enumerate(
                (("strong", "old"), ("weak", "new_sparse"))
            )
        ],
    )
    args.manifest.write_text(json.dumps(manifest))
    return args, frames, manifest


def test_grouped_meta_roles_do_not_split_duplicate_listings(tmp_path):
    import pandas as pd

    args, frames, _ = settings(tmp_path)
    frame = pd.concat(
        [frames["selection"], frames["selection"].assign(train_id=-1, price=9999)]
    )
    splits = fusion.meta_split(frame)
    groups = fusion.mercari.listing_groups(frame)
    sets = [set(groups[v]) for v in splits.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(frame)))


def test_real_three_strategy_study_freezes_all_refits_before_audit(
    tmp_path, monkeypatch
):
    args, frames, _ = settings(tmp_path)
    original = fusion.dump
    frozen_seen = []

    def observe(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            frozen_seen.append(True)
            assert "confirmation_rmsle" not in json.dumps(payload)
            for pool in payload["pools"]:
                assert (args.output_dir / pool / "models.pkl").exists()

    monkeypatch.setattr(fusion, "dump", observe)
    report = fusion.run(args)
    assert frozen_seen == [True]
    assert set(report["results"]) == {"full", "without_new_sparse"}
    assert report["frozen"]["accepted"]
    assert not report["frozen"]["plan"]["anchor_used_as_training_feature"]
    assert report["frozen"] == json.loads((args.output_dir / "frozen.json").read_text())
    assert sum(report["meta_rows"].values()) == len(frames["selection"])
    for pool, data in report["results"].items():
        assert data["primary"] == ["linear", "mlp_42", "conditional"]
        assert len(data["models"]) == 6
        assert sum(row["rows"] for row in data["by_category"].values()) == len(
            frames["confirmation"]
        )
        assert set(data["versus_linear"]) == {"mlp_42", "conditional"}
        with (args.output_dir / pool / "models.pkl").open("rb") as stream:
            saved = pickle.load(stream)
        p = fusion.load_candidates(json.loads(args.manifest.read_text()), frames)
        indices = [0 if name == "strong" else 1 for name in saved["names"]]
        for name, model in saved["models"].items():
            row = data["models"][name]
            assert row["predict_seconds"] == np.mean(row["predict_seconds_samples"])
            assert len(row["predict_seconds_samples"]) == 3
            with np.load(args.output_dir / pool / (name + ".npz")) as archive:
                for k in p:
                    np.testing.assert_array_equal(
                        archive[k + "_ids"], frames[k].train_id
                    )
                    np.testing.assert_array_equal(
                        archive[k],
                        fusion.infer(
                            model, p[k][:, indices], fusion.context(frames[k])
                        ),
                    )
    assert report["submission_status"] == "not_submitted"
    assert json.loads((args.output_dir / "report.json").read_text()) == report


def test_loader_rejects_wrong_provenance_hash_ids_and_invalid_vectors(tmp_path):
    args, frames, original = settings(tmp_path)
    for specs in ([], [original["experts"][0]] * 2):
        with pytest.raises(ValueError, match="unique"):
            fusion.load_candidates(dict(experts=specs), frames)
    for field, value, message in (
        ("fit_window", "selection", "aggregate"),
        ("sha256", "wrong", "hash"),
        ("column", -1, "column"),
        ("column", 2, "column"),
    ):
        manifest = copy.deepcopy(original)
        manifest["experts"][0][field] = value
        with pytest.raises(ValueError, match=message):
            fusion.load_candidates(manifest, frames)
    reference = {k: v for k, v in original["experts"][0].items() if k != "fit_window"}
    assert fusion.load_predictions([reference], frames)["selection"].shape == (25, 1)
    path = args.manifest.parent / "bad.npz"
    with np.load(original["experts"][0]["path"]) as archive:
        valid = dict(archive)
    for mode in ("ids", "missing", "nonfinite", "shape", "matrix"):
        arrays = dict(valid)
        if mode == "ids":
            arrays["selection_ids"] = valid["selection_ids"][::-1]
        elif mode == "missing":
            arrays.pop("confirmation_ids")
        elif mode == "nonfinite":
            arrays["selection"] = valid["selection"] * np.nan
        elif mode == "shape":
            arrays["selection"] = valid["selection"][:1]
        np.savez(path, **arrays)
        spec = dict(
            original["experts"][0], path=str(path), sha256=fusion.mercari.sha256(path)
        )
        if mode == "matrix":
            spec.pop("column")
        with pytest.raises(ValueError, match="IDs|finite|matching"):
            fusion.load_candidates(dict(experts=[spec]), frames)
    manifest = copy.deepcopy(original)
    manifest["data_sha256"]["train"] = "wrong"
    args.manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="data protocol"):
        fusion.run(args)


def test_exact_anchor_fallback_and_misaligned_audit(tmp_path, monkeypatch):
    args, frames, manifest = settings(tmp_path)
    for spec in manifest["experts"]:
        spec["family"] = "old"
    args.manifest.write_text(json.dumps(manifest))
    perfect = {}
    for k, v in frames.items():
        if k == "train":
            continue
        perfect[k] = np.log1p(
            v.price if "price" in v else 10 + v.train_id.to_numpy() % 20
        )
        perfect[k + "_ids"] = v.train_id.to_numpy()
    np.savez(args.anchor, **perfect)
    report = fusion.run(args)
    assert not report["frozen"]["accepted"]
    assert set(report["results"]) == {"full"}
    with np.load(args.anchor) as old, np.load(args.output_dir / "ensemble.npz") as new:
        for k in ("selection", "confirmation"):
            np.testing.assert_array_equal(old[k], new[k])
    args.output_dir = tmp_path / "bad-audit"
    original = fusion.dump

    def poison(path, payload):
        original(path, payload)
        if path.name == "frozen.json":
            np.savez(
                args.data_dir / "evaluation.npz",
                confirmation=np.ones(30),
                confirmation_ids=np.arange(30),
            )

    monkeypatch.setattr(fusion, "dump", poison)
    with pytest.raises(ValueError, match="audit"):
        fusion.run(args)


def test_cli_defaults_and_importable_checkpoint_class(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fusion, "run", lambda args: calls.append(args))
    argv = [
        part
        for name in ("data-dir", "manifest", "anchor", "output-dir")
        for part in ("--" + name, str(tmp_path / name))
    ]
    fusion.main(argv)
    monkeypatch.setattr(sys, "argv", ["mercari_fusion", *argv])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("benchmarks.mercari_fusion", run_name="__main__")
    assert len(calls) == 2 and calls[0].rows == 0 and calls[0].seed == 42
