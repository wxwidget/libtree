"""Bounded-memory sparse MLP for price regression.

Architecture and experiment ideas: Jankiewicz & Lopuhin, Mercari first-place
technical presentation (2018),
https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/5_modeling.tex.
Independent Keras 3 implementation; CSR batches stay sparse, and the saved
prediction graph excludes optimizer states. Huber: doi:10.1214/aoms/1177703732.
"""
import time

import numpy as np
from scipy import sparse
from scipy.special import softmax
import tensorflow as tf


def tensor(x, binary=False):
    coo = x.tocoo()
    values = np.ones(len(coo.data), np.float32) if binary else coo.data.astype(np.float32)
    return tf.SparseTensor(np.column_stack([coo.row, coo.col]).astype(np.int64), values,
                           np.asarray(coo.shape, np.int64))


def select_columns(x, blocks, word_limit=100000, char_limit=50000, min_df=5):
    counts = np.asarray(x.getnnz(axis=0)).ravel()
    selected = []
    for name, block in blocks.items():
        available = np.arange(block.start, block.stop)
        available = available[counts[available] >= min_df]
        limit = word_limit if name in ('name', 'description') else char_limit if name == 'characters' else len(available)
        if len(available) > limit:
            available = available[np.argsort(-counts[available], kind='stable')[:limit]]
        selected.extend(available)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


@tf.keras.utils.register_keras_serializable(package='LibTreeMercari')
class RowSparseAdam(tf.keras.optimizers.Adam):
    """Lazy Adam moments for present rows; dense parameters use standard Adam.

    Adam: Kingma & Ba (2015), https://arxiv.org/abs/1412.6980.
    Sparse optimization motivation: Chenglong Chen's Mercari fourth-place
    report, https://github.com/ChenglongChen/tensorflow-XNN.
    """

    def __init__(self, learning_rate=.004, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        if self.amsgrad:
            raise ValueError('row-sparse Adam does not support AMSGrad')

    def update_step(self, gradient, variable, learning_rate):
        if not isinstance(gradient, tf.IndexedSlices):
            return super().update_step(gradient, variable, learning_rate)
        indices, mapping = tf.unique(gradient.indices)
        values = tf.math.unsorted_segment_sum(gradient.values, mapping, tf.shape(indices)[0])
        index = self._get_variable_index(variable)
        m, v = self._momentums[index].value, self._velocities[index].value
        first = self.beta_1 * tf.gather(m, indices) + (1 - self.beta_1) * values
        second = self.beta_2 * tf.gather(v, indices) + (1 - self.beta_2) * tf.square(values)
        step = tf.cast(self.iterations + 1, variable.dtype)
        scale = tf.cast(learning_rate, variable.dtype) * tf.sqrt(1 - self.beta_2 ** step) / (1 - self.beta_1 ** step)
        m.scatter_update(tf.IndexedSlices(first, indices))
        v.scatter_update(tf.IndexedSlices(second, indices))
        variable.scatter_sub(tf.IndexedSlices(scale * first / (tf.sqrt(second) + self.epsilon), indices))


def soft_targets(target, centers, width=.1):
    """Gaussian label distributions; subtract-max softmax handles distant tails."""
    return softmax(-.5 * ((np.asarray(target)[:, None] - centers[None, :]) / width) ** 2,
                   axis=1).astype(np.float32)


def build_model(features, width=128, seed=42, loss='huber', optimizer='adam', bins=64):
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input((features,), sparse=True, dtype='float32')
    regularizer = None if optimizer == 'row_adam' else tf.keras.regularizers.L2(1e-6)
    value = tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=regularizer)(inputs)
    value = tf.keras.layers.Dense(64, activation='relu')(value)
    value = tf.keras.layers.Dense(32, activation='relu')(value)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(bins if loss == 'bins' else 1)(value))
    criterion = tf.keras.losses.Huber(delta=1.) if loss == 'huber' else loss
    if loss == 'bins':
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    algorithm = RowSparseAdam(.004) if optimizer == 'row_adam' else tf.keras.optimizers.Adam(.004)
    # Keras may auto-enable XLA on GPUs, but sparse gather/segment and scatter
    # optimizer operations do not have a complete XLA lowering (TF 2.21).
    model.compile(optimizer=algorithm, loss=criterion, jit_compile=False)
    return model


class SparseExpert:
    def __init__(self, width=128, seed=42, loss='huber', binary=True, optimizer='adam', bins=64):
        self.width, self.seed, self.loss, self.binary = width, seed, loss, binary
        self.optimizer = optimizer
        self.bins = bins

    def _prediction(self, model, x, batch_size):
        values = []
        for i in range(0, x.shape[0], batch_size):
            output = np.asarray(model(tensor(x[i:i + batch_size], self.binary), training=False))
            values.append(softmax(output, axis=1) @ self.centers_ if self.loss == 'bins' else output.ravel())
        return np.concatenate(values) * self.scale_ + self.mean_

    def fit(self, x, log_y, rates, batches, validation=None):
        x = sparse.csr_matrix(x, dtype=np.float32)
        log_y = np.asarray(log_y, dtype=np.float32)
        if (log_y.shape != (x.shape[0],) or not np.isfinite(log_y).all() or not len(rates)
                or len(rates) != len(batches) or min(batches) < 1 or min(rates) <= 0):
            raise ValueError('finite labels and matching positive learning-rate/batch schedules required')
        self.mean_, self.scale_ = float(np.mean(log_y)), max(float(np.std(log_y)), 1e-6)
        target = (log_y - self.mean_) / self.scale_
        if self.loss == 'bins':
            edges = np.quantile(target, np.linspace(0., 1., self.bins + 1))
            self.centers_ = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
            target = soft_targets(target, self.centers_)
        model = build_model(x.shape[1], self.width, self.seed, self.loss, self.optimizer, self.bins)
        self.history_, best_weights, best_score = [], None, float('inf')
        rng = np.random.default_rng(self.seed)
        for epoch, (rate, batch_size) in enumerate(zip(rates, batches)):
            stamp = time.perf_counter()
            model.optimizer.learning_rate.assign(rate)
            model.reset_metrics()
            order = rng.permutation(x.shape[0])
            for begin in range(0, len(order), batch_size):
                rows = order[begin:begin + batch_size]
                loss = model.train_on_batch(tensor(x[rows], self.binary), target[rows])
            row = dict(epoch=epoch + 1, rate=rate, batch_size=batch_size,
                       train_loss=float(loss), seconds=time.perf_counter() - stamp)
            if validation is not None:
                prediction = self._prediction(model, validation[0], batch_size=8192)
                score = float(np.sqrt(np.mean((np.maximum(prediction, 0.) - validation[1]) ** 2)))
                row['rmsle'] = score
                if score < best_score:
                    best_weights, best_score = model.get_weights(), score
            self.history_.append(row)
            print('Sparse MLP:', row, flush=True)
        if best_weights is not None:
            model.set_weights(best_weights)
        self.model_ = tf.keras.Model(model.inputs, model.outputs)
        return self

    def predict(self, x, batch_size=8192):
        return self._prediction(self.model_, sparse.csr_matrix(x, dtype=np.float32), batch_size)
