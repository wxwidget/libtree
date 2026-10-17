"""CPU deep averaging expert for Mercari's nonlinear text interactions.

Iyyer et al. (2015), Deep Unordered Composition Rivals Syntactic Methods for
Text Classification, https://aclanthology.org/P15-1162/. This is an adaptation
to log-price regression, with masked max pooling and categorical embeddings.
TensorFlow is optional and is not imported by the core LibTree package.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from benchmarks.mercari_features import TOKEN_PATTERN

CATEGORICAL = ('brand_name', 'category_name', 'category_0', 'category_1', 'category_2',
               'category_condition', 'brand_category', 'brand_condition')


class Tokens:
    """Fixed training dictionaries; ID zero means padding or unknown."""

    def __init__(self, max_words=100000, title_length=20, description_length=80, version=2):
        self.max_words = max_words
        self.title_length, self.description_length = title_length, description_length
        self.version = version

    def fit(self, frame):
        self.token_pattern_ = TOKEN_PATTERN if self.version >= 2 else r'(?u)\b\w[\w.]+\b'
        texts = list(frame.name) + list(frame.item_description)
        vectorizer = CountVectorizer(max_features=self.max_words, min_df=2, token_pattern=self.token_pattern_)
        vectorizer.fit(texts)
        self.words_ = {word: index + 1 for word, index in vectorizer.vocabulary_.items()}
        self.categories_ = {name: {value: i + 1 for i, value in enumerate(sorted(frame[name].unique()))}
                            for name in CATEGORICAL}
        return self

    def transform(self, frame):
        result = {}
        pattern = re.compile(getattr(self, 'token_pattern_', r'(?u)\b\w[\w.]+\b'))
        for column, key, width in [('name', 'title', self.title_length),
                                    ('item_description', 'description', self.description_length)]:
            encoded = np.zeros((len(frame), width), dtype=np.int32)
            for row, text in enumerate(frame[column]):
                ids = [self.words_.get(word, 0) for word in pattern.findall(text)[:width]]
                encoded[row, :len(ids)] = ids
            result[key] = encoded
        for name in CATEGORICAL:
            result[name] = frame[name].map(self.categories_[name]).fillna(0).to_numpy(dtype=np.int32)[:, None]
        result['numeric'] = np.column_stack([
            frame.shipping.astype(float), frame.item_condition_id.astype(float) / 5,
            np.log1p(frame.name.str.len()) / 5, np.log1p(frame.item_description.str.len()) / 7,
            np.log1p(frame.name.str.split().str.len()) / 3,
            np.log1p(frame.item_description.str.split().str.len()) / 5]).astype(np.float32)
        return result


@tf.keras.utils.register_keras_serializable(package='LibTreeMercari')
class MaskedPool(tf.keras.layers.Layer):
    """Mean and max of real tokens; an entirely unknown title maps to zero."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            mask = tf.ones(tf.shape(inputs)[:2], dtype=tf.bool)
        valid = tf.cast(mask[:, :, None], inputs.dtype)
        count = tf.reduce_sum(valid, axis=1)
        mean = tf.reduce_sum(inputs * valid, axis=1) / tf.maximum(count, 1.)
        maximum = tf.reduce_max(tf.where(mask[:, :, None], inputs, -1e9), axis=1)
        maximum = tf.where(count > 0, maximum, 0.)
        return tf.concat([mean, maximum], axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


@tf.keras.utils.register_keras_serializable(package='LibTreeMercari')
class MaskedConvolution(tf.keras.layers.Layer):
    """Three-token title phrases, with zero contribution from padding inputs.

    Kim (2014), CNNs for Sentence Classification, https://aclanthology.org/D14-1181/.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.convolution = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')

    def build(self, input_shape):
        self.convolution.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs * tf.cast(mask[:, :, None], inputs.dtype)
        return tf.reduce_max(self.convolution(inputs), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None


def build_model(encoder, seed=42, width=64, loss='mse', architecture='bag'):
    if architecture not in ('bag', 'cnn'):
        raise ValueError('neural architecture must be bag or cnn')
    tf.keras.utils.set_random_seed(seed)
    inputs, vectors = {}, []
    embedding = tf.keras.layers.Embedding(len(encoder.words_) + 1, width, mask_zero=True)
    for name, length in [('title', encoder.title_length), ('description', encoder.description_length)]:
        inputs[name] = tf.keras.Input((length,), dtype='int32', name=name)
        embedded = embedding(inputs[name])
        vectors.append(MaskedPool()(embedded))
        if name == 'title' and architecture == 'cnn':
            vectors.append(MaskedConvolution()(embedded))
    for name in CATEGORICAL:
        inputs[name] = tf.keras.Input((1,), dtype='int32', name=name)
        value = tf.keras.layers.Embedding(len(encoder.categories_[name]) + 1, 16)(inputs[name])
        vectors.append(tf.keras.layers.Flatten()(value))
    inputs['numeric'] = tf.keras.Input((6,), dtype='float32', name='numeric')
    vectors.append(inputs['numeric'])
    x = tf.keras.layers.Concatenate()(vectors)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(.1)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(.1)(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x))
    model.compile(optimizer=tf.keras.optimizers.Adam(.003, clipnorm=5.), loss=loss,
                  metrics=[tf.keras.metrics.MeanSquaredError(name='mse')])
    return model


def dataset(x, y=None, batch_size=2048, shuffle=False):
    values = x if y is None else (x, np.asarray(y, dtype=np.float32))
    data = tf.data.Dataset.from_tensor_slices(values)
    if shuffle:
        data = data.shuffle(min(len(next(iter(x.values()))), 50000), seed=42)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return data.batch(batch_size).with_options(options).prefetch(1)


def train(model, x, y, validation_x, validation_y, epochs=8, batch_size=2048):
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=2, restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=.5, patience=1, min_lr=.0001)]
    history = model.fit(dataset(x, y, batch_size, True),
                        validation_data=dataset(validation_x, validation_y, batch_size),
                        epochs=epochs, callbacks=callbacks, verbose=2, shuffle=False)
    return history.history


def predict(model, x, batch_size=2048):
    return model.predict(dataset(x, batch_size=batch_size), verbose=0).reshape(-1)
