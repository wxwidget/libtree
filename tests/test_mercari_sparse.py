"""Sparse nonlinear regression without materializing a dense document matrix."""
import pickle
import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('tensorflow')
from benchmarks import mercari_sparse as neural


def test_sparse_conversion_and_predictions_match_dense_reference():
    import tensorflow as tf
    x = sparse.csr_matrix([[0., 2., 0.], [1., 0., 3.]], dtype=np.float32)
    tensor = neural.tensor(x)
    np.testing.assert_array_equal(tf.sparse.to_dense(tensor).numpy(), x.toarray())
    binary = neural.tensor(x, binary=True)
    np.testing.assert_array_equal(tf.sparse.to_dense(binary).numpy(), x.toarray() > 0)
    model = neural.build_model(3, width=4, seed=42, loss='mse')
    np.testing.assert_allclose(model(tensor).numpy(), model(x.toarray()).numpy(), atol=1e-7)


def test_sparse_training_remains_supported_when_backend_prefers_auto_xla(monkeypatch):
    import tensorflow as tf
    monkeypatch.setattr(tf.keras.Model, '_resolve_auto_jit_compile', lambda self: True)
    model = neural.build_model(3, width=4, seed=42, loss='mse', optimizer='row_adam')
    assert model.jit_compile is False
    x = sparse.csr_matrix([[1., 0., 1.], [0., 1., 0.]], dtype=np.float32)
    loss = model.train_on_batch(neural.tensor(x), np.array([1., 2.], dtype=np.float32))
    assert np.isfinite(loss)


def test_sparse_expert_uses_training_scale_and_replays_without_optimizer():
    rng = np.random.default_rng(42)
    x = sparse.csr_matrix(rng.integers(0, 2, (60, 8)).astype(np.float32))
    y = 2 + x.toarray()[:, 0] * x.toarray()[:, 1]
    expert = neural.SparseExpert(width=8, seed=42, loss='huber', binary=True)
    expert.fit(x, y, rates=[.01, .005], batches=[16, 32], validation=(x, y))
    prediction = expert.predict(x, batch_size=16)
    clone = pickle.loads(pickle.dumps(expert))
    np.testing.assert_allclose(prediction, clone.predict(x, batch_size=16), atol=1e-7)
    assert not getattr(expert.model_, 'optimizer', None)
    assert expert.mean_ == pytest.approx(np.mean(y)) and len(expert.history_) == 2
    fixed = neural.SparseExpert(width=4, loss='mse').fit(x, y, rates=[.01], batches=[16])
    assert np.isfinite(fixed.predict(x)).all()
    with pytest.raises(ValueError):
        expert.fit(x, [np.nan] * len(y), rates=[.01], batches=[16])
    with pytest.raises(ValueError):
        expert.fit(x, y, rates=[], batches=[])


def test_column_pruning_is_train_only_and_keeps_stable_indices():
    x = sparse.csr_matrix([[1, 0, 2, 0, 1], [1, 1, 0, 0, 1], [1, 1, 2, 0, 1]])
    blocks = {'name': slice(0, 2), 'description': slice(2, 4), 'categories': slice(4, 5)}
    keep = neural.select_columns(x, blocks, word_limit=1, char_limit=0, min_df=2)
    np.testing.assert_array_equal(keep, [0, 2, 4])
    blocks['characters'] = slice(4, 5)
    assert len(neural.select_columns(x, blocks, word_limit=10, char_limit=1, min_df=2)) >= 3


def test_row_sparse_adam_aggregates_duplicates_and_does_not_update_absent_rows():
    import tensorflow as tf
    variable = tf.Variable(np.zeros((4, 2), dtype=np.float32))
    optimizer = neural.RowSparseAdam(learning_rate=.01)
    gradient = tf.IndexedSlices(tf.constant([[1., 2.], [3., 4.]]), tf.constant([1, 1]), tf.constant([4, 2]))
    optimizer.apply_gradients([(gradient, variable)])
    np.testing.assert_allclose(variable.numpy()[1], [-.01, -.01], atol=1e-6)
    previous = variable.numpy()[1].copy()
    optimizer.apply_gradients([(tf.IndexedSlices(tf.ones((1, 2)), [2], [4, 2]), variable)])
    np.testing.assert_array_equal(variable.numpy()[1], previous)
    np.testing.assert_array_equal(variable.numpy()[[0, 3]], np.zeros((2, 2)))
    a, b = tf.Variable([1., 2.]), tf.Variable([1., 2.])
    neural.RowSparseAdam(.01).apply_gradients([(tf.constant([.3, -.4]), a)])
    tf.keras.optimizers.Adam(.01).apply_gradients([(tf.constant([.3, -.4]), b)])
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    with pytest.raises(ValueError):
        neural.RowSparseAdam(amsgrad=True)


def test_row_sparse_model_trains_and_replays():
    import tensorflow as tf
    x = sparse.csr_matrix(np.tile(np.eye(4, dtype=np.float32), (8, 1)))
    y = np.tile([1., 2., 3., 4.], 8)
    model = neural.SparseExpert(width=8, optimizer='row_adam').fit(x, y, [.01, .01], [8, 16], validation=(x, y))
    assert np.isfinite(model.predict(x)).all()
    np.testing.assert_array_equal(model.predict(x), pickle.loads(pickle.dumps(model)).predict(x))
    probe = neural.build_model(4, width=8, optimizer='row_adam')
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(probe(neural.tensor(x[:4])))
    gradient = tape.gradient(loss, probe.trainable_variables)[0]
    assert isinstance(gradient, tf.IndexedSlices) and not probe.losses


def test_price_distribution_targets_are_normalized_stable_and_train_only():
    target = neural.soft_targets(np.array([-1000., 0., 1000.]), np.array([-1., 0., 1.]), .1)
    np.testing.assert_allclose(target.sum(axis=1), 1.)
    assert np.isfinite(target).all() and target[0, 0] == 1 and target[-1, -1] == 1
    x = sparse.csr_matrix(np.tile(np.eye(4, dtype=np.float32), (8, 1)))
    y = np.tile([1., 2., 3., 4.], 8)
    model = neural.SparseExpert(width=8, optimizer='row_adam', loss='bins', bins=8)
    model.fit(x, y, [.01, .005], [8, 16], validation=(x, y))
    prediction = model.predict(x)
    assert np.isfinite(prediction).all() and len(model.centers_) == 8
    np.testing.assert_array_equal(prediction, pickle.loads(pickle.dumps(model)).predict(x))
