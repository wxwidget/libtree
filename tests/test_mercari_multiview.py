"""Product-statistic leakage, shared expert views and regularized meta-learning."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from benchmarks import mercari_multiview as views
from test_mercari import listings


def test_product_statistics_shrink_to_training_prior_and_ignore_query_labels():
    frame = listings(60)
    frame["name"] = ["new iphone6 64 GB", "iphone6 64gb"] * 30
    frame["brand_name"] = "Apple"
    frame["category_name"] = "Electronics/Phones/Smartphones"
    y = np.linspace(1.0, 5.0, 60)
    model = views.ProductStatistics(smoothing=20.0).fit(frame, y)
    x = model.transform(frame)
    assert x.shape == (60, len(model.feature_names_)) and np.isfinite(x).all()
    index = model.feature_names_.index("identity_mean")
    assert x[0, index] == pytest.approx(y.mean())
    title_index = model.feature_names_.index("title_mean")
    # Three copies of the 30-row group plus all 60 base labels independently
    # represent group weight 30 and prior weight 20, including prior variance.
    pooled = np.r_[np.tile(y[::2], 3), y]
    assert x[0, title_index] == pytest.approx(pooled.mean(), abs=1e-7)
    assert x[0, title_index + 1] == pytest.approx(np.log1p(30))
    assert x[0, title_index + 2] == pytest.approx(pooled.std(), abs=1e-7)
    np.testing.assert_array_equal(
        x, model.transform(frame.assign(price=9999.0, train_id=-1))
    )
    np.testing.assert_array_equal(x[:1], model.transform(frame.iloc[:1]))
    np.testing.assert_array_equal(x, pickle.loads(pickle.dumps(model)).transform(frame))
    query = frame.iloc[:1].assign(
        name="never observed 17tb", brand_name="unknown", category_name="unknown"
    )
    unknown = model.transform(query)
    for name in model.maps_:
        offset = model.feature_names_.index(name + "_mean")
        assert unknown[0, offset] == pytest.approx(y.mean())
        assert unknown[0, offset + 1] == 0.0
    assert (
        views.ProductStatistics.keys(frame)["identity"].iloc[0]
        == views.ProductStatistics.keys(frame)["identity"].iloc[1]
    )
    for settings in [dict(smoothing=0), dict(smoothing=np.nan)]:
        with pytest.raises(ValueError):
            views.ProductStatistics(**settings).fit(frame, y)
    for bad in [y[:-1], y * np.nan, -y]:
        with pytest.raises(ValueError):
            views.ProductStatistics().fit(frame, bad)


def test_decomposition_reconstructs_original_prediction_without_repeating_extraction():
    calls = []

    class Model:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.full(x.shape[0], self.value)

    class Member:
        def __init__(self, value):
            self.value = value

        def predict(self, frame, prepared, x):
            return np.full(len(frame), self.value)

    class Extractor:
        def transform(self, p):
            calls.append(len(p))
            return sparse.csr_matrix(np.ones((len(p), 3)))

    class Encoder:
        def transform(self, frame, prepared):
            return np.ones((len(frame), 2))

    entry = SimpleNamespace(
        experts=SimpleNamespace(
            listing=SimpleNamespace(transform=lambda x: x),
            extractor=Extractor(),
            ridge=Model(2.0),
            fm=Model(1.0),
            mean=2.0,
            keep=np.arange(3),
            scales=np.ones(3),
            neural=SimpleNamespace(
                members=[Member(4.0), Member(5.0)], weights=np.array([0.3, 0.7])
            ),
        ),
        encoder=Encoder(),
        native=Model(0.2),
        config=dict(weights=[0.1, 0.2, 0.7], correction_strength=0.5),
    )
    result = views.decompose(entry, listings(4))
    np.testing.assert_array_equal(
        result["prediction"], np.tile([2.0, 3.0, 4.0, 5.0], (4, 1))
    )
    np.testing.assert_allclose(result["uncorrected"], 0.1 * 2 + 0.2 * 3 + 0.7 * 4.7)
    np.testing.assert_allclose(result["reference"], result["uncorrected"] + 0.1)
    assert calls == [4] and result["metadata"].shape == (4, 2)
    entry.experts.ridge = None
    entry.experts.neural = Member(6.0)
    assert views.decompose(entry, listings(2))["names"][0] == "mean_prior"
    assert views.disagreement(result["prediction"]).shape == (4, 5)


def test_product_reparameterization_matches_independent_refit_and_parent_backoff():
    frame = listings(60).assign(
        brand_name="Apple", category_name="Electronics/Phones/Smartphones"
    )
    y = np.linspace(1.0, 5.0, 60)
    original = views.ProductStatistics().fit(frame, y)
    before = original.transform(frame)
    changed = original.reparameterize(5.0)
    independently = views.ProductStatistics(5.0).fit(frame, y)
    np.testing.assert_allclose(
        changed.transform(frame), independently.transform(frame), atol=2e-6
    )
    np.testing.assert_array_equal(before, original.transform(frame))
    hierarchical = original.reparameterize(20.0, hierarchical=True)
    query = frame.iloc[:1].assign(name="neverobservedidentity 512tb")
    parent_prior = (
        views.ProductStatistics.keys(frame)["brand_condition"]
        .str.rsplit("|", n=1)
        .str[0]
    )
    # This corpus has one parent, so its posterior mean is the global prior.
    assert parent_prior.nunique() == 1
    result = hierarchical.transform(query)
    assert result[0, 0] == pytest.approx(y.mean())
    assert result[0, 1] == 0.0
    np.testing.assert_array_equal(
        result, pickle.loads(pickle.dumps(hierarchical)).transform(query)
    )
    np.testing.assert_allclose(
        hierarchical.reparameterize(5.0).transform(frame),
        independently.transform(frame),
        atol=2e-6,
    )
    unknown = query.assign(brand_name="unseen", category_name="unseen")
    assert hierarchical.transform(unknown)[0, 0] == pytest.approx(y.mean())
    with pytest.raises(ValueError):
        original.reparameterize(0.0)
    varied = frame.assign(brand_name=["Apple"] * 30 + ["Other"] * 30)
    targets = np.r_[np.full(30, 5.0), np.ones(30)]
    parents = (
        views.ProductStatistics()
        .fit(varied, targets)
        .reparameterize(20.0, hierarchical=True)
    )
    novel = varied.iloc[:1].assign(name="unseen 1024tb identity")
    assert parents.transform(novel)[0, 0] == pytest.approx(
        (30 * 5 + 20 * 3) / 50, abs=1e-6
    )


def test_affine_blend_learns_scale_beyond_the_simplex_and_regularizes(monkeypatch):
    rng = np.random.default_rng(42)
    p = rng.uniform(1.0, 5.0, (200, 3))
    y = p @ np.array([0.2, 0.4, 0.5]) + 0.08
    model = views.RegularizedBlend(penalty=0.0).fit(p, y)
    np.testing.assert_allclose(model.predict(p), y, atol=5e-6)
    assert model.weights_.sum() == pytest.approx(1.1, abs=5e-6)
    prior = np.array([0.8, 0.1, 0.1])
    shrunk = views.RegularizedBlend(penalty=100.0, prior=prior).fit(p, y)
    assert shrunk.weights_[0] > 0.7
    np.testing.assert_array_equal(
        model.predict(p), pickle.loads(pickle.dumps(model)).predict(p)
    )
    for settings in [
        dict(penalty=-1),
        dict(bias_bound=-1),
        dict(mass_bounds=(1.2, 0.8)),
        dict(mass_bounds=(0.0, 1.0)),
        dict(prior=[0.5, 0.5]),
        dict(prior=[1.0, 0.0, np.nan]),
    ]:
        with pytest.raises(ValueError):
            views.RegularizedBlend(**settings).fit(p, y)
    monkeypatch.setattr(
        views, "minimize", lambda *a, **k: SimpleNamespace(success=False)
    )
    with pytest.raises(RuntimeError, match="converge"):
        views.RegularizedBlend().fit(p, y)


def test_gated_blend_shrinks_sparse_groups_and_falls_back_for_unknowns():
    rng = np.random.default_rng(32)
    p = rng.uniform(1.0, 4.0, (200, 2))
    g = np.array(["a"] * 100 + ["b"] * 90 + ["rare"] * 10)
    y = np.where(g == "a", p[:, 0], p[:, 1])
    model = views.ContextBlend(min_rows=40, local_penalty=0.001).fit(p, y, g)
    assert set(model.locals_) == {"a", "b"}
    assert (
        np.mean((model.predict(p, g) - y) ** 2)
        < np.mean((model.global_.predict(p) - y) ** 2) * 0.1
    )
    unknown = np.full(len(p), "unknown")
    np.testing.assert_array_equal(model.predict(p, unknown), model.global_.predict(p))
    np.testing.assert_array_equal(
        model.predict(p, g), pickle.loads(pickle.dumps(model)).predict(p, g)
    )
    assert views.snapshot(model)["kind"] == "context"
    for bad in [g[:-1], np.array("scalar")]:
        with pytest.raises(ValueError):
            views.ContextBlend().fit(p, y, bad)
    with pytest.raises(ValueError):
        model.predict(p, g[:-1])
    with pytest.raises(ValueError):
        views.ContextBlend(min_rows=0).fit(p, y, g)


def test_numeric_groups_and_roundoff_at_affine_mass_bounds():
    rng = np.random.default_rng(42)
    p = rng.uniform(1.0, 3.0, (100, 2))
    g = np.repeat([1, 2], 50)
    y = np.where(g == 1, p[:, 0], p[:, 1])
    model = views.ContextBlend(min_rows=30).fit(p, y, g)
    assert np.mean((model.predict(p, g) - y) ** 2) < 0.001
    rounded_prior = np.array([np.nextafter(0.8, 0.0), 0.0])
    fitted = views.RegularizedBlend(prior=rounded_prior).fit(p, p[:, 0])
    assert np.isfinite(fitted.predict(p)).all()


def test_meta_strategy_uses_a_separate_grouped_window_and_freezes_serializable_state():
    rng = np.random.default_rng(42)
    p = rng.uniform(1.0, 4.0, (200, 3))
    y = 0.2 * p[:, 0] + 0.8 * p[:, 1]
    frame = listings(200)
    selected, report = views.select_strategy(p, y, frame)
    assert set(report["validation"]) == {"simplex", "affine", "context"}
    assert set(report["fit_ids"]).isdisjoint(report["validation_ids"])
    assert len(report["fit_ids"]) + len(report["validation_ids"]) == 200
    assert views.snapshot(selected)["kind"] == report["chosen"]
    assert np.mean((views.predict(selected, p, frame) - y) ** 2) < 1e-5
    import json

    json.dumps(views.snapshot(selected), allow_nan=False)
