"""Complementary text views and selection-only log-price blending.

LSA: Deerwester et al. (1990), doi:10.1002/asi.4630410605; randomized SVD:
Halko et al. (2011), doi:10.1137/090771806. Sentence embeddings: Reimers &
Gurevych (2019), https://aclanthology.org/D19-1410/. Stacking: Wolpert (1992),
doi:10.1016/S0893-6080(05)80023-1. These are independent task adaptations.
"""
import re

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from benchmarks.mercari import prepare
from benchmarks.mercari_features import ListingFeatures, TOKEN_PATTERN


class TextAttributes:
    """Inspectable product quantities and condition phrases, without labels.

    Capacity and numbers use log1p with a fixed cap; unknown/missing quantities
    are zero. These flags are approximate lexical evidence, not factual product
    attributes. In particular a phone case and a phone can share semantic words.
    """

    flags = dict(unlocked=r'\bunlocked\b', accessory=r'\b(case|cover|protector)\b',
        damaged=r'\b(broken|cracked|damaged|for parts|not working)\b',
        box_only=r'\b(box only|empty box)\b', bundle=r'\b(bundle|lot of|set of|pair)\b',
        new=r'\b(new|nwt|nwot|unused)\b')
    attributes = ('storage_gb', 'max_number', 'redactions', *flags)

    def __init__(self):
        self.feature_names_ = [f'{field}_{name}' for field in ('name', 'description') for name in self.attributes]

    def transform(self, frame):
        p, blocks = prepare(frame), []
        for field in ('name', 'item_description'):
            rows = []
            for text in p[field].str.lower():
                capacity = [float(value) * (1024 if unit == 'tb' else 1)
                            for value, unit in re.findall(r'(\d+(?:\.\d+)?)\s*(gb|tb)\b', text)]
                numbers = [float(value) for value in re.findall(r'\d+(?:\.\d+)?', text)]
                rows.append([np.log1p(min(max(capacity, default=0), 1e6)),
                    np.log1p(min(max(numbers, default=0), 1e6)), text.count('[rm]'),
                    *[float(bool(re.search(pattern, text))) for pattern in self.flags.values()]])
            blocks.append(np.asarray(rows, dtype=np.float32))
        return np.ascontiguousarray(np.column_stack(blocks), dtype=np.float32)


class LatentText:
    """Separate title/description LSA plus frozen unsupervised metadata.

    Fitting takes no labels. Category IDs and frequency maps never grow during
    inference. Tree callers must declare categorical_indices_ as categorical;
    linear callers should use only the latent/numeric prefix or one-hot IDs.
    """

    categories = ('brand_name', 'category_name', 'category_0', 'category_1',
                  'category_2', 'category_condition', 'brand_category', 'brand_condition')

    def __init__(self, components=64, max_features=75000, min_df=2, seed=42):
        self.components, self.max_features = components, max_features
        self.min_df, self.seed = min_df, seed

    def fit(self, frame):
        if min(self.components, self.max_features, self.min_df) < 1:
            raise ValueError('positive latent-text settings required')
        if len(frame) < 2:
            raise ValueError('at least two training listings required')
        self.listing_ = ListingFeatures().fit(frame)
        prepared = self.listing_.transform(frame)
        self.fields_, self.feature_names_ = {}, []
        for name in ('name', 'item_description'):
            vectorizer = TfidfVectorizer(max_features=self.max_features, min_df=self.min_df,
                ngram_range=(1, 2), token_pattern=TOKEN_PATTERN, sublinear_tf=True, dtype=np.float32)
            matrix = vectorizer.fit_transform(prepared[name])
            rank = min(self.components, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if rank < 1:
                raise ValueError('at least two distinct text features required')
            svd = TruncatedSVD(n_components=rank, n_iter=3, random_state=self.seed)
            svd.fit(matrix)
            self.fields_[name] = vectorizer, svd
            self.feature_names_ += [f'{name}_lsa_{i}' for i in range(rank)]
        self.feature_names_ += ['condition', 'shipping', 'title_length', 'description_length',
                                'title_words', 'description_words']
        self.codes_, self.frequencies_ = {}, {}
        for name in self.categories:
            counts = prepared[name].value_counts()
            self.codes_[name] = {value: i + 1 for i, value in enumerate(sorted(counts.index))}
            self.frequencies_[name] = (counts / len(frame)).to_dict()
            self.feature_names_.append(name + '_frequency')
        self.categorical_indices_ = list(range(len(self.feature_names_), len(self.feature_names_) + len(self.categories)))
        self.feature_names_ += [name + '_id' for name in self.categories]
        return self

    def transform(self, frame):
        p = self.listing_.transform(frame)
        columns = [svd.transform(vectorizer.transform(p[name]))
                   for name, (vectorizer, svd) in self.fields_.items()]
        columns.append(np.column_stack([p.item_condition_id.astype(float), p.shipping.astype(float),
            np.log1p(p.name.str.len()), np.log1p(p.item_description.str.len()),
            np.log1p(p.name.str.split().str.len()), np.log1p(p.item_description.str.split().str.len())]))
        columns += [p[name].map(self.frequencies_[name]).fillna(0.).to_numpy()[:, None] for name in self.categories]
        columns += [p[name].map(self.codes_[name]).fillna(0).to_numpy()[:, None] for name in self.categories]
        return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)

    def topics(self, terms=8):
        """Largest positive loadings help inspect what each latent axis encodes."""
        return {name: [vectorizer.get_feature_names_out()[np.argsort(-component)[:terms]].tolist()
                       for component in svd.components_]
                for name, (vectorizer, svd) in self.fields_.items()}


class SentenceVectors:
    """Explicit external-model adapter, independent of price, ID or other rows.

    Pass a pinned SentenceTransformer instance; loading/downloading a model is
    deliberately outside this adapter. Both fields retain their own embedding.
    """

    def __init__(self, encoder, batch_size=256):
        if batch_size < 1:
            raise ValueError('positive embedding batch size required')
        self.encoder, self.batch_size = encoder, batch_size

    def transform(self, frame):
        prepared = prepare(frame)
        blocks = [self.encoder.encode(prepared[name].tolist(), batch_size=self.batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            for name in ('name', 'item_description')]
        return np.ascontiguousarray(np.column_stack(blocks), dtype=np.float32)


def blend_weights(prediction, log_y, anchor=0, penalty=1e-4):
    """Fit a nonnegative sum-one blend on selection rows only.

    The quadratic penalty shrinks toward a predeclared baseline rather than
    rewarding a weak extra expert just because it was tried. Never fit on the
    confirmation window or public/private competition scores.
    """
    p, y = np.asarray(prediction, dtype=float), np.asarray(log_y, dtype=float)
    if (p.ndim != 2 or y.ndim != 1 or not y.size or len(p) != len(y)
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not 0 <= anchor < p.shape[1] or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('finite matching prediction matrix, labels, anchor and penalty required')
    prior = np.eye(p.shape[1])[anchor]
    def objective(w):
        return np.mean((np.maximum(p @ w, 0.) - y)**2) + penalty * np.sum((w - prior)**2)
    result = minimize(objective, prior, method='SLSQP', bounds=[(0., 1.)] * p.shape[1],
        constraints=[dict(type='eq', fun=lambda w: w.sum() - 1.)],
        options=dict(ftol=1e-12, maxiter=300))
    if not result.success:
        raise RuntimeError('ensemble optimization did not converge')
    weights = np.clip(result.x, 0., 1.)
    return weights / weights.sum()


def paired_comparison(log_y, baseline, candidate, repeats=500, seed=42):
    """Paired row bootstrap of the RMSLE difference (negative means better).

    This estimates sampling variability conditional on these fitted models; it
    does not remove model-selection bias or measure training-seed variability.
    """
    y, a, b = [np.asarray(v, dtype=float) for v in (log_y, baseline, candidate)]
    if (y.ndim != 1 or not len(y) or y.shape != a.shape or y.shape != b.shape
            or not all(np.isfinite(v).all() for v in (y, a, b)) or repeats < 1):
        raise ValueError('finite matching nonempty vectors and positive repeats required')
    errors = np.column_stack([(np.maximum(a, 0.) - y)**2, (np.maximum(b, 0.) - y)**2])
    scores = np.sqrt(errors.mean(axis=0))
    rng, samples = np.random.default_rng(seed), []
    for _ in range(repeats):
        means = errors[rng.integers(0, len(y), size=len(y))].mean(axis=0)
        samples.append(float(np.sqrt(means[1]) - np.sqrt(means[0])))
    return dict(baseline_rmsle=float(scores[0]), candidate_rmsle=float(scores[1]),
                rmsle_delta=float(scores[1] - scores[0]), ci95=np.quantile(samples, [.025, .975]).tolist(),
                bootstrap_repeats=repeats, seed=seed, unit='row', training_repeats=1)
