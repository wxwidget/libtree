"""GPU-compatible right padding and real recurrent regression contracts."""
import pickle

import numpy as np
import pytest

pytest.importorskip('tensorflow')
from benchmarks.mercari_features import ListingFeatures
from benchmarks.mercari_neural import Tokens, build_model, dataset, predict
from test_mercari import listings


def test_unknown_tokens_do_not_make_internal_mask_holes_for_cudnn():
    frame = listings(12)
    prepared = ListingFeatures().fit(frame).transform(frame)
    tokens = Tokens(max_words=100, unknown_tokens=True).fit(prepared)
    query = prepared.iloc[:2].copy()
    query['name'] = ['item neverseenword shirt', '']
    query['item_description'] = ['neverseenword good item', '']
    encoded = tokens.transform(query)
    for name in ('title', 'description'):
        for row in encoded[name]:
            n = np.count_nonzero(row)
            assert n > 0 and np.all(row[:n] > 0) and np.all(row[n:] == 0)
    assert tokens.words_['<unknown>'] in encoded['title'][0]
    restored = pickle.loads(pickle.dumps(tokens))
    np.testing.assert_array_equal(encoded['title'], restored.transform(query)['title'])


def test_gru_trains_predicts_and_preserves_saved_model():
    import tensorflow as tf
    frame = listings(20)
    prepared = ListingFeatures().fit(frame).transform(frame)
    tokens = Tokens(max_words=100, title_length=8, description_length=12, unknown_tokens=True).fit(prepared)
    x = tokens.transform(prepared)
    model = build_model(tokens, width=8, architecture='gru', loss='mse')
    model.fit(dataset(x, np.log1p(frame.price), batch_size=10), epochs=1, verbose=0)
    result = predict(model, x)
    assert result.shape == (len(frame),) and np.isfinite(result).all()
    assert any(isinstance(layer, tf.keras.layers.GRU) for layer in model.layers)
    clone = pickle.loads(pickle.dumps(tf.keras.Model(model.inputs, model.outputs)))
    np.testing.assert_allclose(result, predict(clone, x), atol=1e-6)
