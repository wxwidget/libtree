"""Contracts for a CPU-only, training-vocabulary neural text expert."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('tensorflow')
from benchmarks import mercari_neural as neural
from benchmarks.mercari_features import ListingFeatures
from test_mercari import listings


def test_sequences_are_frozen_padded_and_exclude_targets():
    frame = ListingFeatures().fit(listings()).transform(listings())
    encoder = neural.Tokens(max_words=100, title_length=8, description_length=10).fit(frame)
    x = encoder.transform(frame.iloc[:4])
    query = frame.iloc[:4].copy()
    query['name'] = 'unseenword'
    query['brand_name'] = 'neverknown'
    changed = encoder.transform(query.assign(price=1e10, train_id=-1))
    assert not changed['title'].any() and not changed['brand_name'].any()
    assert x['title'].shape == (4, 8) and x['description'].shape == (4, 10)
    before = encoder.transform(frame.iloc[:1])
    for name in x:
        np.testing.assert_array_equal(before[name], x[name][:1])
    assert np.isfinite(x['numeric']).all()
    assert '1' in encoder.words_
    del encoder.token_pattern_
    legacy = encoder.transform(frame.assign(name='1'))
    assert not legacy['title'].any()
    prior = neural.Tokens(max_words=100, version=1).fit(frame)
    assert '1' not in prior.words_


def test_masked_pool_padding_and_model_roundtrip(tmp_path):
    import tensorflow as tf
    values = tf.constant([[[-2., 3.], [7., -4.], [99., 99.]], [[3., 4.], [9., 9.], [9., 9.]]])
    mask = tf.constant([[True, True, False], [False, False, False]])
    pooled = neural.MaskedPool()(values, mask=mask).numpy()
    np.testing.assert_allclose(pooled[0], [2.5, -.5, 7., 3.])
    np.testing.assert_array_equal(pooled[1], [0., 0., 0., 0.])
    unmasked = neural.MaskedPool()(values).numpy()
    np.testing.assert_allclose(unmasked[0, :2], np.mean(values.numpy()[0], axis=0))
    assert np.isfinite(neural.MaskedConvolution()(values).numpy()).all()
    frame = ListingFeatures().fit(listings(40)).transform(listings(40))
    encoder = neural.Tokens(max_words=100, title_length=8, description_length=10).fit(frame)
    x = encoder.transform(frame)
    model = neural.build_model(encoder, seed=42, width=8)
    y = np.linspace(-1, 1, len(frame), dtype=np.float32)
    before = neural.predict(model, x, batch_size=16)
    history = neural.train(model, x, y, x, y, epochs=2, batch_size=16)
    after = neural.predict(model, x, batch_size=16)
    assert len(history['val_loss']) == 2 and np.mean((after-y)**2) < np.mean((before-y)**2)
    model.save(tmp_path / 'expert.keras')
    restored = tf.keras.models.load_model(tmp_path / 'expert.keras')
    np.testing.assert_array_equal(after, neural.predict(restored, x, batch_size=16))
    robust = neural.build_model(encoder, seed=42, width=8, loss=tf.keras.losses.Huber(delta=.5))
    assert isinstance(robust.loss, tf.keras.losses.Huber)
    convolution = neural.build_model(encoder, seed=42, width=8, architecture='cnn')
    assert np.isfinite(neural.predict(convolution, x, batch_size=16)).all()
    convolution.train_on_batch({k: v[:16] for k, v in x.items()}, y[:16])
    convolution.save(tmp_path / 'convolution.keras')
    clone = tf.keras.models.load_model(tmp_path / 'convolution.keras')
    np.testing.assert_array_equal(neural.predict(convolution, x), neural.predict(clone, x))
    with pytest.raises(ValueError):
        neural.build_model(encoder, architecture='unknown')

