"""Bounded comparable-item lookup, with leave-query-group-out target features.

TF-IDF cosine: Salton & Buckley (1988), doi:10.1016/0306-4573(88)90021-0.
Target smoothing: Micci-Barreca (2001), doi:10.1145/507533.507538.
This is approximate, category-blocked lexical retrieval, not semantic ANN search.
Every label-derived prior also excludes all rows sharing the query's features.
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

from benchmarks.mercari import listing_groups
from benchmarks.mercari_features import normalize, TOKEN_PATTERN


@njit(cache=True, parallel=True)
def lookup(
    query, index, vocabulary, neighbors, max_candidates, max_scan, min_similarity
):
    qp, qc, qv, category, brand, condition, own = query
    (
        rp,
        rc,
        rv,
        rbrand,
        rcondition,
        means,
        seconds,
        totals,
        counts,
        cat_total,
        cat_count,
        keys,
        starts,
        postings,
    ) = index
    output = np.zeros((len(category), 8), np.float32)
    total = totals.sum()
    count = counts.sum()
    for row in prange(len(category)):
        group = own[row]
        own_total = totals[group] if group >= 0 else 0.0
        own_count = counts[group] if group >= 0 else 0.0
        prior_global = (
            (total - own_total) / (count - own_count) if count > own_count else 0.0
        )
        code = category[row]
        prior = (cat_total[code] - own_total + 20 * prior_global) / (
            cat_count[code] - own_count + 20
        )
        anchors = np.full(2, -1, np.int64)
        sizes = np.full(2, 9223372036854775807, np.int64)
        for t in range(qp[row], qp[row + 1]):
            key = code * vocabulary + qc[t]
            position = np.searchsorted(keys, key)
            if position == len(keys) or keys[position] != key:
                continue
            size = starts[position + 1] - starts[position]
            if size == 1 and postings[starts[position]] == group:
                continue
            for j in range(2):
                if size < sizes[j]:
                    for k in range(1, j, -1):
                        sizes[k] = sizes[k - 1]
                        anchors[k] = anchors[k - 1]
                    sizes[j] = size
                    anchors[j] = position
                    break
        pool = np.empty(2 * max_scan, np.int64)
        n_pool = 0
        for anchor in anchors:
            if anchor < 0:
                continue
            for j in range(
                starts[anchor], min(starts[anchor + 1], starts[anchor] + max_scan)
            ):
                candidate = postings[j]
                if candidate != group and (
                    brand[row] == 0 or brand[row] == rbrand[candidate]
                ):
                    pool[n_pool] = candidate
                    n_pool += 1
        candidates = np.sort(pool[:n_pool])
        best_scores = np.zeros(neighbors)
        best_ids = np.full(neighbors, -1, np.int64)
        visited = 0
        for j in range(n_pool):
            candidate = candidates[j]
            if j and candidate == candidates[j - 1]:
                continue
            if visited >= max_candidates:
                break
            visited += 1
            a = qp[row]
            b = rp[candidate]
            similarity = 0.0
            while a < qp[row + 1] and b < rp[candidate + 1]:
                if qc[a] == rc[b]:
                    similarity += qv[a] * rv[b]
                    a += 1
                    b += 1
                elif qc[a] < rc[b]:
                    a += 1
                else:
                    b += 1
            similarity *= np.exp(-0.15 * abs(condition[row] - rcondition[candidate]))
            if similarity < min_similarity:
                continue
            for k in range(neighbors):
                if similarity > best_scores[k]:
                    for h in range(neighbors - 1, k, -1):
                        best_scores[h] = best_scores[h - 1]
                        best_ids[h] = best_ids[h - 1]
                    best_scores[k] = similarity
                    best_ids[k] = candidate
                    break
        n = 0
        weight = 0.0
        first = 0.0
        second = 0.0
        sim = 0.0
        for k in range(neighbors):
            if best_ids[k] < 0:
                continue
            n += 1
            s = best_scores[k]
            w = s**4
            weight += w
            first += w * means[best_ids[k]]
            second += w * seconds[best_ids[k]]
            sim += s
        mean = first / weight if weight else prior
        std = np.sqrt(max(second / weight - mean * mean, 0.0)) if weight else 0.0
        shrink = n / (n + 5.0)
        output[row, 0] = prior
        output[row, 1] = mean
        output[row, 2] = std
        output[row, 3] = np.log1p(n)
        output[row, 4] = best_scores[0]
        output[row, 5] = sim / n if n else 0.0
        output[row, 6] = shrink * mean + (1 - shrink) * prior
        output[row, 7] = mean - prior
    return output


class ComparableRetrieval(BaseEstimator):
    """Group-distinct, deterministic top-k from capped rare-token postings.

    Exact category and known-brand filters precede cosine/condition scoring.
    Reference groups sort by their feature hashes, never target prices. The
    two rarest category-token postings bound candidate search; this is not exact
    global nearest-neighbor search. Unknown categories back off to a global prior.
    """

    feature_names_ = (
        "category_prior",
        "neighbor_mean",
        "neighbor_std",
        "log_support",
        "max_similarity",
        "mean_similarity",
        "shrunken_mean",
        "mean_minus_prior",
    )

    def __init__(
        self,
        neighbors=8,
        max_features=75000,
        max_candidates=64,
        max_scan=128,
        min_similarity=0.35,
    ):
        self.neighbors, self.max_features = neighbors, max_features
        self.max_candidates, self.max_scan, self.min_similarity = (
            max_candidates,
            max_scan,
            min_similarity,
        )

    @staticmethod
    def metadata(frame):
        category = (
            frame.category_name.fillna("<missing>").astype(str).str.lower().str.strip()
        )
        brand = frame.brand_name.map(normalize)
        hashes = pd.util.hash_pandas_object(brand, index=False).to_numpy()
        hashes[brand.eq("<missing>").to_numpy()] = 0
        condition = (
            pd.to_numeric(frame.item_condition_id, errors="coerce")
            .fillna(3)
            .clip(1, 5)
            .to_numpy(dtype=np.float32)
        )
        return category, hashes, condition

    def fit(self, frame, log_y):
        y = np.asarray(log_y, dtype=float)
        if (
            y.shape != (len(frame),)
            or not len(y)
            or not np.isfinite(y).all()
            or (y < 0).any()
        ):
            raise ValueError("matching finite nonnegative log prices required")
        if (
            min(self.neighbors, self.max_features, self.max_candidates, self.max_scan)
            < 1
            or not np.isfinite(self.min_similarity)
            or not 0 <= self.min_similarity <= 1
        ):
            raise ValueError(
                "positive retrieval budgets and similarity in [0,1] required"
            )
        self.groups_, first, inverse = np.unique(
            listing_groups(frame), return_index=True, return_inverse=True
        )
        reference = frame.iloc[first]
        count = np.bincount(inverse).astype(float)
        total = np.bincount(inverse, weights=y)
        second = np.bincount(inverse, weights=y * y) / count
        category, brand, condition = self.metadata(reference)
        self.categories_ = {k: i + 1 for i, k in enumerate(sorted(category.unique()))}
        codes = category.map(self.categories_).to_numpy(dtype=np.int64)
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            min_df=1,
            ngram_range=(1, 2),
            token_pattern=TOKEN_PATTERN,
            dtype=np.float32,
        )
        x = self.vectorizer_.fit_transform(reference.name.map(normalize)).tocsr()
        x.sort_indices()
        self.vocabulary_ = x.shape[1]
        rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
        compound = codes[rows] * self.vocabulary_ + x.indices
        order = np.argsort(compound, kind="stable")
        keys, starts = np.unique(compound[order], return_index=True)
        starts = np.r_[starts, len(order)].astype(np.int64)
        cat_total = np.bincount(
            codes, weights=total, minlength=len(self.categories_) + 1
        )
        cat_count = np.bincount(
            codes, weights=count, minlength=len(self.categories_) + 1
        )
        self.index_ = (
            x.indptr,
            x.indices,
            x.data,
            brand,
            condition,
            total / count,
            second,
            total,
            count,
            cat_total,
            cat_count,
            keys,
            starts,
            rows[order],
        )
        return self

    def transform(self, frame):
        check_is_fitted(self, "index_")
        if not len(frame):
            return np.zeros((0, 8), np.float32)
        category, brand, condition = self.metadata(frame)
        x = self.vectorizer_.transform(frame.name.map(normalize)).tocsr()
        x.sort_indices()
        groups = listing_groups(frame)
        positions = np.searchsorted(self.groups_, groups)
        valid = positions < len(self.groups_)
        valid[valid] = self.groups_[positions[valid]] == groups[valid]
        own = np.where(valid, positions, -1).astype(np.int64)
        query = (
            x.indptr,
            x.indices,
            x.data,
            category.map(self.categories_).fillna(0).to_numpy(dtype=np.int64),
            brand,
            condition,
            own,
        )
        return lookup(
            query,
            self.index_,
            self.vocabulary_,
            self.neighbors,
            self.max_candidates,
            self.max_scan,
            self.min_similarity,
        )
