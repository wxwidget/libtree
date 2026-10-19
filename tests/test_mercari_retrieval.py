"""Comparable retrieval must exclude the query's complete duplicate group."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from benchmarks.mercari_retrieval import ComparableRetrieval
from test_mercari import listings


def phones():
    frame = listings(12).assign(
        name="iphone 6 64 gb unlocked",
        brand_name="Apple",
        category_name="Electronics/Phones/Smartphones",
    )
    frame["item_description"] = [f"phone listing {i}" for i in range(len(frame))]
    frame["price"] = np.arange(len(frame)) * 10 + 100.0
    duplicate = frame.iloc[:1].copy().assign(train_id=999, price=999.0)
    return pd.concat([frame, duplicate], ignore_index=True)


def test_query_excludes_all_own_labels_in_neighbors_and_hierarchical_priors():
    frame = phones()
    y = np.log1p(frame.price.to_numpy())
    model = ComparableRetrieval(max_features=100).fit(frame, y)
    actual = model.transform(frame.iloc[:1])
    changed = y.copy()
    changed[[0, 12]] = np.log1p([99999, 999999])
    other = ComparableRetrieval(max_features=100).fit(frame, changed)
    np.testing.assert_allclose(actual, other.transform(frame.iloc[:1]), atol=1e-6)
    assert actual.shape == (1, 8) and actual[0, 3] > 0
    # Ten identical rows are one excluded group, not ten apparent comparables.
    only = pd.concat([frame.iloc[:1]] * 10, ignore_index=True)
    isolated = ComparableRetrieval().fit(only, np.log1p(np.arange(10) + 10.0))
    np.testing.assert_array_equal(isolated.transform(only), np.zeros((10, 8)))


def test_retrieval_is_target_free_context_independent_and_serializable():
    frame = phones()
    m = ComparableRetrieval(max_features=100).fit(frame, np.log1p(frame.price))
    query = frame.iloc[:2].assign(
        name=["iphone6 64gb unlocked", "iphone6 64gb unlocked"],
        item_description=["query 1", "query 2"],
    )
    a = m.transform(query)
    np.testing.assert_array_equal(
        a, m.transform(query.assign(price=np.nan, train_id=-1))
    )
    np.testing.assert_array_equal(a[:1], m.transform(query.iloc[:1]))
    np.testing.assert_array_equal(a, pickle.loads(pickle.dumps(m)).transform(query))
    assert np.isfinite(a).all() and a[0, 4] > 0
    reversed_model = ComparableRetrieval(max_features=100).fit(
        frame.iloc[::-1], np.log1p(frame.price.to_numpy()[::-1])
    )
    np.testing.assert_allclose(a, reversed_model.transform(query), atol=1e-6)


def test_unknown_category_brand_and_no_comparables_have_finite_fallback():
    f = phones()
    y = np.log1p(f.price.to_numpy())
    m = ComparableRetrieval().fit(f, y)
    unseen = f.iloc[:1].assign(
        name="zzzzunseen", category_name="Unseen/Category", brand_name=None
    )
    p = m.transform(unseen)
    assert p[0, 0] == pytest.approx(y.mean(), abs=1e-6)
    assert p[0, 3] == 0
    other = f.iloc[:1].assign(
        brand_name="Unknown brand", item_description="not training"
    )
    assert m.transform(other)[0, 3] == 0
    missing = f.iloc[:1].assign(brand_name=None, item_description="not training")
    assert m.transform(missing)[0, 3] > 0
    assert m.transform(f.iloc[:0]).shape == (0, 8)
    with pytest.raises(NotFittedError):
        ComparableRetrieval().transform(f)
    for bad in [y[:-1], y * np.nan, -np.ones(len(y))]:
        with pytest.raises(ValueError):
            ComparableRetrieval().fit(f, bad)
    for kwargs in [
        dict(neighbors=0),
        dict(max_candidates=0),
        dict(max_scan=0),
        dict(min_similarity=-1),
        dict(max_features=0),
    ]:
        with pytest.raises(ValueError):
            ComparableRetrieval(**kwargs).fit(f, y)


def test_lookup_limits_and_minimum_similarity_are_explicit():
    f = phones()
    y = np.log1p(f.price.to_numpy())
    m = ComparableRetrieval(neighbors=1, max_candidates=1, max_scan=1).fit(f, y)
    q = f.iloc[:1].assign(item_description="unseen")
    p = m.transform(q)
    assert p[0, 3] == pytest.approx(np.log1p(1))
    m = ComparableRetrieval(min_similarity=1).fit(f, y)
    q = q.assign(name="iphone 6 other word", item_condition_id=5)
    assert m.transform(q)[0, 3] == 0


def test_exact_comparable_wins_over_partially_overlapping_titles():
    f = phones().iloc[:3].copy()
    f["name"] = ["iphone red", "iphone blue", "iphone red blue"]
    f["item_condition_id"] = 1
    y = np.log1p([100.0, 200.0, 300.0])
    m = ComparableRetrieval(neighbors=1, max_scan=10, max_candidates=1).fit(f, y)
    q = f.iloc[2:3].assign(item_description="query only")
    assert np.isfinite(m.transform(q)).all()
    m = ComparableRetrieval(neighbors=1, max_scan=10).fit(f, y)
    p = m.transform(q)
    assert p[0, 1] == pytest.approx(np.log1p(300), abs=1e-6)
    assert p[0, 4] == pytest.approx(1.0, abs=1e-6)
    absent = q.assign(category_name="unknown")
    assert m.transform(absent)[0, 3] == 0


def test_extra_known_words_cannot_retrieve_across_category_boundary():
    f = phones().iloc[:2].copy()
    f["name"] = ["iphone red", "aaa"]
    f.loc[f.index[1], "category_name"] = "Other/Category"
    q = f.iloc[:1].assign(name="aaa iphone red", item_description="query")
    m = ComparableRetrieval(neighbors=1).fit(f, np.log1p([100.0, 9999.0]))
    p = m.transform(q)
    assert p[0, 1] == pytest.approx(np.log1p(100), abs=1e-6)
