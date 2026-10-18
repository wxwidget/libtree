"""CPU price features and direct metric optimization, with independent oracles."""

import pickle

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from benchmarks import mercari, mercari_cpu as cpu
from test_mercari import listings


def test_objective_is_squared_official_rmsle_and_has_correct_gradient():
    p = np.array([[-2.0, -0.5], [0.2, 0.8], [3.0, 4.0], [1.0, 3.0]])
    y = np.log1p([0.0, 0.0, 14.0, 5.0])
    prior = np.array([1.0, 0.0])
    parameters = np.array([0.3, 0.7, 0.1])
    value, gradient = cpu.value_gradient(parameters, p, y, prior, 0.0)
    predicted_price = mercari.prices(p @ parameters[:-1] + parameters[-1])
    assert value == pytest.approx(mercari.rmsle(np.expm1(y), predicted_price) ** 2)
    for penalty in (0.0, 0.3):
        value, gradient = cpu.value_gradient(parameters, p, y, prior, penalty)
        numerical = approx_derivative(
            lambda w: cpu.value_gradient(w, p, y, prior, penalty)[0], parameters
        )
        np.testing.assert_allclose(gradient, numerical.ravel(), atol=1e-8)
    # The chosen subgradient at the clipping boundary is zero.
    value, gradient = cpu.value_gradient(np.zeros(3), p, y, prior, 0.0)
    assert value == pytest.approx(np.mean(y * y))
    np.testing.assert_array_equal(gradient, np.zeros(3))


def test_direct_blend_recovers_known_weights_and_bias_and_serializes():
    rng = np.random.default_rng(17)
    p = rng.uniform(1.0, 5.0, (200, 3))
    expected = [0.2, 0.3, 0.5]
    y = p @ expected + 0.12
    model = cpu.RmsleBlend(penalty=0.0).fit(p, y)
    np.testing.assert_allclose(model.weights_, expected, atol=2e-6)
    assert model.intercept_ == pytest.approx(0.12, abs=2e-6)
    np.testing.assert_allclose(model.predict(p), y, atol=3e-6)
    np.testing.assert_array_equal(
        model.predict(p), pickle.loads(pickle.dumps(model)).predict(p)
    )
    assert model.n_iter_ > 0 and model.objective_ < 1e-10
    fixed = cpu.RmsleBlend(penalty=0.0, bias_bound=0.0).fit(p, p @ expected)
    assert fixed.intercept_ == 0.0
    regularized = cpu.RmsleBlend(penalty=100.0).fit(p, y)
    assert regularized.weights_[0] > model.weights_[0]


@pytest.mark.parametrize(
    "settings",
    [
        dict(penalty=-1),
        dict(penalty=np.nan),
        dict(bias_bound=-1),
        dict(bias_bound=np.inf),
        dict(anchor=3),
        dict(max_iter=0),
    ],
)
def test_blend_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        cpu.RmsleBlend(**settings).fit(np.ones((5, 2)), np.ones(5))


def test_blend_rejects_malformed_data_and_failed_optimization(monkeypatch):
    p, y = np.ones((5, 2)), np.ones(5)
    for matrix, target in [
        (p, y[:-1]),
        (p * np.nan, y),
        (p, y * -1),
        (p, y * np.inf),
        (np.ones(5), y),
        (p, 1.0),
        (np.empty((0, 2)), []),
        (np.empty((5, 0)), y),
    ]:
        with pytest.raises(ValueError):
            cpu.RmsleBlend().fit(matrix, target)
    with pytest.raises(ValueError):
        cpu.RmsleBlend().fit(p, y).predict(np.ones((2, 3)))
    with pytest.raises(ValueError):
        cpu.RmsleBlend().fit(p, y).predict([[np.nan, 1.0]])
    monkeypatch.setattr(
        cpu, "minimize", lambda *a, **k: type("Result", (), {"success": False})()
    )
    with pytest.raises(RuntimeError, match="converge"):
        cpu.RmsleBlend().fit(p, y)


def test_blend_escapes_a_clipped_anchor_with_zero_gradient():
    p = np.array([[-2.0, 1.0], [-1.0, 2.0], [-3.0, 4.0]])
    fitted = cpu.RmsleBlend(penalty=0.0, bias_bound=0.0).fit(p, p[:, 1])
    np.testing.assert_allclose(fitted.weights_, [0.0, 1.0], atol=1e-7)


def test_product_signals_preserve_units_models_and_price_changing_phrases():
    assert cpu.signals("iPhone6 64 GB case, unlocked") == cpu.signals(
        "iphone6 64gb case, unlocked"
    )
    tokens = cpu.signals("IPHONE6 64 GB cracked screen, box only, lot of 3").split()
    assert {
        "model_iphone6",
        "unit_64gb",
        "flag_damaged",
        "flag_box_only",
        "flag_bundle",
    } <= set(tokens)
    assert cpu.signals(None) == "missing_signal"
    assert cpu.signals("plain clothing") == "missing_signal"
    assert len(cpu.signals(" ".join(f"model{i}" for i in range(200))).split()) <= 32


def test_engineered_views_are_train_only_label_free_and_batch_independent():
    frame = listings(60)
    frame.loc[0, ["name", "item_description", "brand_name"]] = None
    frame.loc[1, "name"] = "iPhone6 64 GB case bundle"
    frame.loc[1, "brand_name"] = "Apple"
    for view in ("exact", "stemmed"):
        model = cpu.CpuFeatures(view=view).fit(frame)
        p = model.transform(frame)
        assert "model_iphone6" in p.loc[1, "signals"]
        assert "brand_apple__iphone" in p.loc[1, "crosses"]
        assert "iphone6" in p.loc[1, "raw_name"]
        altered = model.transform(frame.assign(price=1e7, train_id=-1))
        assert p.equals(altered)
        assert p.iloc[:1].equals(model.transform(frame.iloc[:1]))
        assert p.equals(pickle.loads(pickle.dumps(model)).transform(frame))
        extractor = cpu.make_features(
            max_features=60, char_features=30, cross_features=40
        )
        matrix = extractor.fit_transform(p)
        assert matrix.shape[0] == 60 and np.isfinite(matrix.data).all()
        columns = extractor.named_transformers_["crosses"].get_feature_names_out()
        assert (
            "brand_apple__iphone" not in columns
        )  # min_df excludes singleton crossings
        assert matrix.dtype == np.float32
        q = (
            frame.iloc[:2]
            .copy()
            .assign(name="neverobservedword", brand_name="newbrand")
        )
        before = extractor.transform(model.transform(q))
        np.testing.assert_array_equal(
            before[:1].toarray(),
            extractor.transform(model.transform(q.iloc[:1])).toarray(),
        )
    with pytest.raises(ValueError, match="view"):
        cpu.CpuFeatures(view="unknown").fit(frame)
