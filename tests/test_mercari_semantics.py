"""Train-only latent text, semantic adapters and honest ensemble contracts."""
import pickle

import numpy as np
import pytest

from benchmarks import mercari_semantics as semantics
from test_mercari import listings


def test_latent_features_freeze_vocabulary_categories_and_exclude_labels():
    frame = listings(50)
    frame.loc[0, ['name', 'item_description', 'brand_name']] = None
    encoder = semantics.LatentText(components=4, max_features=100, min_df=1).fit(frame)
    before = encoder.transform(frame.iloc[:3])
    assert before.shape == (3, len(encoder.feature_names_))
    assert before.dtype == np.float32 and np.isfinite(before).all()
    np.testing.assert_array_equal(before, encoder.transform(frame.iloc[:3].assign(price=1e8, train_id=-1)))
    query = frame.iloc[:2].copy()
    query['name'] = 'entirelyunseenword'
    query['brand_name'] = 'unseen brand'
    query['category_name'] = 'unknown/category'
    result = encoder.transform(query)
    np.testing.assert_array_equal(result[:1], encoder.transform(query.iloc[:1]))
    np.testing.assert_array_equal(result, pickle.loads(pickle.dumps(encoder)).transform(query))
    assert np.isfinite(result).all()
    assert 'entirelyunseenword' not in encoder.fields_['name'][0].vocabulary_
    assert encoder.categorical_indices_
    topics = encoder.topics(terms=3)
    assert set(topics) == {'name', 'item_description'}
    assert all(len(terms) <= 3 for field in topics.values() for terms in field)


def test_latent_settings_and_tiny_corpus_boundaries():
    for settings in [dict(components=0), dict(max_features=0), dict(min_df=0)]:
        with pytest.raises(ValueError):
            semantics.LatentText(**settings).fit(listings())
    with pytest.raises(ValueError, match='two'):
        semantics.LatentText(components=4, min_df=1).fit(listings(1))
    with pytest.raises(ValueError, match='two distinct'):
        semantics.LatentText(components=4, min_df=1).fit(listings(4).assign(name='one'))
    frame = listings(4).assign(name='one two', item_description='two three')
    encoder = semantics.LatentText(components=20, min_df=1).fit(frame)
    assert np.isfinite(encoder.transform(frame)).all()


def test_semantic_adapter_never_receives_prices_and_normalizes_embeddings():
    calls = []

    class Encoder:
        def encode(self, text, **kwargs):
            calls.append((text, kwargs))
            return np.column_stack([np.arange(len(text)), np.ones(len(text))]).astype(np.float32)

    encoder = semantics.SentenceVectors(Encoder(), batch_size=8)
    frame = listings(3)
    matrix = encoder.transform(frame)
    assert matrix.shape == (3, 4)
    np.testing.assert_array_equal(matrix, encoder.transform(frame.assign(price=1e8, train_id=-1)))
    assert all(c[1]['normalize_embeddings'] for c in calls)
    assert all(c[1]['batch_size'] == 8 for c in calls)
    assert calls[0][0] == frame.name.tolist()
    with pytest.raises(ValueError):
        semantics.SentenceVectors(Encoder(), batch_size=0)


def test_simplex_blend_has_closed_form_solution_and_rejects_invalid_input(monkeypatch):
    p = np.array([[1., 3.], [3., 1.], [2., 2.]])
    y = p @ [.25, .75]
    w = semantics.blend_weights(p, y, anchor=0, penalty=0.)
    np.testing.assert_allclose(w, [.25, .75], atol=1e-7)
    assert np.sum(w) == pytest.approx(1.) and min(w) >= 0
    regularized = semantics.blend_weights(p, y, anchor=0, penalty=10.)
    assert regularized[0] > w[0]
    for x, target, settings in [(p, [1.], {}), (p * np.nan, y, {}),
                                 (p, y, dict(anchor=3)), (p, y, dict(penalty=-1.)),
                                 (p, y * np.nan, {}), (np.empty((0, 2)), [], {}), (p, 1., {})]:
        with pytest.raises(ValueError):
            semantics.blend_weights(x, target, **settings)
    monkeypatch.setattr(semantics, 'minimize', lambda *a, **k: type('Result', (), {'success': False})())
    with pytest.raises(RuntimeError, match='converge'):
        semantics.blend_weights(p, y)


def test_paired_uncertainty_uses_identical_rows_and_preserves_bad_outcomes():
    y = np.linspace(1., 4., 50)
    a, b = y + .2, y + .1
    report = semantics.paired_comparison(y, a, b, repeats=100, seed=42)
    assert report['rmsle_delta'] == pytest.approx(-.1)
    assert report['ci95'][1] < 0
    assert report['candidate_rmsle'] == pytest.approx(.1)
    assert report == semantics.paired_comparison(y, a, b, repeats=100, seed=42)
    worse = semantics.paired_comparison(y, b, a, repeats=20)
    assert worse['ci95'][0] > 0
    for args in [([], [], []), (y, a[:-1], b), (y, a, b * np.nan)]:
        with pytest.raises(ValueError):
            semantics.paired_comparison(*args)
    with pytest.raises(ValueError):
        semantics.paired_comparison(y, a, b, repeats=0)


def test_text_attributes_distinguish_capacity_accessories_and_broken_bundles():
    frame = listings(3)
    frame['name'] = ['iPhone6 64 GB unlocked', 'iPhone 6 case bundle', None]
    frame['item_description'] = ['1 TB disk 2.5 oz unused', 'cracked screen, box only, lot of 3', None]
    encoder = semantics.TextAttributes()
    x = encoder.transform(frame)
    index = {name: i for i, name in enumerate(encoder.feature_names_)}
    assert x[0, index['name_storage_gb']] == pytest.approx(np.log1p(64))
    assert x[0, index['description_storage_gb']] == pytest.approx(np.log1p(1024))
    assert x[0, index['name_unlocked']] == 1
    assert x[1, index['name_accessory']] == 1 and x[1, index['description_damaged']] == 1
    assert x[1, index['description_box_only']] == 1 and x[1, index['name_bundle']] == 1
    assert np.all(x[2] == 0)
    assert x.dtype == np.float32 and np.isfinite(x).all()
    np.testing.assert_array_equal(x, encoder.transform(frame.assign(price=1e8, train_id=-1)))
    np.testing.assert_array_equal(x[:1], encoder.transform(frame.iloc[:1]))
