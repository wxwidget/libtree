"""Train-only Mercari lexical features; no listing labels or test statistics.

Character n-grams complement word TF-IDF (Salton & Buckley, 1988,
doi:10.1016/0306-4573(88)90021-0) for misspellings and product model numbers.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from benchmarks.mercari import COLUMNS, DenseFeatures, prepare

UNIT = re.compile(r'(\d)\s+(gb|tb|oz|ml|mm|cm|inch|inches|lbs?)\b')
PUNCTUATION = re.compile(r'[^\w.]+', re.UNICODE)
SPACE = re.compile(r'\s+')
ALPHANUMERIC = re.compile(r'([a-z])(\d)')
TOKEN_PATTERN = r'(?u)\b\w[\w.]*\b'


def normalize(value, split_alphanumeric=True):
    if pd.isna(value):
        return '<missing>'
    value = UNIT.sub(r'\1\2', str(value).lower())
    if split_alphanumeric:
        value = ALPHANUMERIC.sub(r'\1 \2', value)
    return SPACE.sub(' ', PUNCTUATION.sub(' ', value)).strip() or '<missing>'


class ListingFeatures:
    """Recover missing brands only from unambiguous known title mentions.

    The dictionary is fitted on training brand metadata, never prices. Inference
    does not grow it, so a prediction is independent of other prediction rows.
    """

    def __init__(self, version=2):
        self.version = version

    def fit(self, frame):
        self.normalization_version_ = self.version
        self.brands_ = {normalize(v, self.version >= 2) for v in frame.brand_name.dropna().unique()}
        self.brands_.discard('<missing>')
        self.brands_.discard('missing')
        return self

    def transform(self, frame):
        p = prepare(frame)[list(COLUMNS)].copy()
        for column in ('name', 'item_description', 'brand_name', 'category_name'):
            p[column] = frame[column].map(lambda v: normalize(v, getattr(self, 'normalization_version_', 1) >= 2))
        missing = p.brand_name.eq('<missing>')

        def infer(title):
            tokens = title.split()
            matches = {candidate for width in (1, 2, 3) for start in range(len(tokens) - width + 1)
                       if (candidate := ' '.join(tokens[start:start + width])) in self.brands_}
            return next(iter(matches)) if len(matches) == 1 else '<missing>'

        p.loc[missing, 'brand_name'] = p.loc[missing, 'name'].map(infer)
        # Preserve the original slash hierarchy rather than the normalized text.
        categories = frame.category_name.fillna('<missing>').str.lower().str.split('/')
        for level in range(3):
            p[f'category_{level}'] = categories.str[level].fillna('<missing>')
        p['category_condition'] = p.category_name + '|' + p.item_condition_id
        p['brand_category'] = p.brand_name + '|' + p.category_1
        p['brand_condition'] = p.brand_name + '|' + p.item_condition_id
        return p


def make_ridge(max_features=250000, alpha=3., char_features=150000):
    transforms = [
        ('name', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                sublinear_tf=True, dtype=np.float32, token_pattern=TOKEN_PATTERN), 'name'),
        ('description', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                       sublinear_tf=True, dtype=np.float32, token_pattern=TOKEN_PATTERN), 'item_description'),
        ('categories', OneHotEncoder(handle_unknown='ignore', dtype=np.float32),
         ['category_name', 'brand_name', 'item_condition_id', 'shipping',
          'category_0', 'category_1', 'category_2', 'category_condition', 'brand_category', 'brand_condition'])]
    weights = dict(name=2., description=1., categories=2., characters=1.)
    if char_features:
        transforms.append(('characters', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                           min_df=3, max_features=char_features, sublinear_tf=True, dtype=np.float32), 'name'))
    extractor = ColumnTransformer(transforms, transformer_weights=weights, sparse_threshold=1.)
    return Pipeline([('features', extractor), ('ridge', Ridge(alpha=alpha, solver='lsqr', tol=1e-4, max_iter=200))])


class ResidualFeatures:
    """Compact nonlinear tree inputs; all target statistics require OOF use.

    Smoothed category and lexical means extend Micci-Barreca (2001),
    doi:10.1145/507533.507538. A listing's own fold must not fit this object.
    """

    def fit(self, frame, y):
        self.base_ = DenseFeatures().fit(frame, y)
        self.prior_ = self.base_.prior_
        self.listings_ = ListingFeatures().fit(frame)
        p = self.listings_.transform(frame)
        self.means_, self.counts_ = {}, {}
        for name in ('category_condition', 'brand_category', 'brand_condition'):
            stats = pd.DataFrame({'key': p[name].to_numpy(), 'target': np.asarray(y)}).groupby('key').target.agg(['sum', 'count'])
            self.means_[name] = ((stats['sum'] + 20 * self.prior_) / (stats['count'] + 20)).to_dict()
            self.counts_[name] = (stats['count'] / len(frame)).to_dict()
        self.words_ = CountVectorizer(min_df=2, max_features=75000, ngram_range=(1, 2), binary=True, token_pattern=TOKEN_PATTERN)
        x = self.words_.fit_transform(p.name)
        count = np.asarray(x.sum(axis=0)).ravel()
        self.word_means_ = (np.asarray(x.T @ np.asarray(y)).ravel() + 10 * self.prior_) / (count + 10)
        self.feature_names_ = self.base_.feature_names_ + [f'{name}_{suffix}' for name in self.means_ for suffix in ('mean', 'frequency')]
        self.feature_names_ += ['word_price_mean', 'word_price_min', 'word_price_max', 'word_count', 'word_price_spread']
        return self

    def transform(self, frame, prepared=None):
        p = self.listings_.transform(frame) if prepared is None else prepared
        columns = [self.base_.transform(frame)]
        for name in self.means_:
            columns.extend([p[name].map(self.means_[name]).fillna(self.prior_).to_numpy()[:, None],
                            p[name].map(self.counts_[name]).fillna(0.).to_numpy()[:, None]])
        x = self.words_.transform(p.name)
        count = np.diff(x.indptr)
        nonempty = np.flatnonzero(count)
        minimum = np.full(len(frame), self.prior_)
        maximum = minimum.copy()
        values = self.word_means_[x.indices]
        if len(nonempty):
            minimum[nonempty] = np.minimum.reduceat(values, x.indptr[nonempty])
            maximum[nonempty] = np.maximum.reduceat(values, x.indptr[nonempty])
        mean = np.asarray(x @ self.word_means_).ravel() / np.maximum(count, 1)
        mean[count == 0] = self.prior_
        columns.append(np.column_stack([mean, minimum, maximum, np.log1p(count), maximum - minimum]))
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)
