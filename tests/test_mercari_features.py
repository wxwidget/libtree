"""Feature engineering contracts: frozen dictionaries and no price/ID leakage."""
import numpy as np
import pandas as pd
import pytest

from benchmarks import mercari_features as features


def rows():
    return pd.DataFrame({
        'name': ['Apple iPhone 64 GB', 'NIKE shoes', 'plain shirt', 'Apple case'],
        'brand_name': ['Apple', 'Nike', None, 'Apple'],
        'category_name': ['Electronics/Phones/Smartphones'] * 4,
        'item_description': ['Almost new!', 'size 8', None, 'great'],
        'item_condition_id': [1, 2, 3, 4], 'shipping': [0, 1, 0, 1],
        'price': [50., 20., 5., 4.], 'train_id': range(4)})


def test_brand_dictionary_is_frozen_and_ambiguous_mentions_stay_missing():
    frame = rows()
    encoder = features.ListingFeatures().fit(frame)
    query = frame.copy()
    query['brand_name'] = None
    query['name'] = ['APPLE iphone', 'Nike Apple bundle', 'unknown NEWBRAND', '64 GB Apple case']
    result = encoder.transform(query)
    assert list(result.brand_name) == ['apple', '<missing>', '<missing>', 'apple']
    changed = query.assign(price=1e6, train_id=-10)
    pd.testing.assert_frame_equal(result, encoder.transform(changed))
    singleton = encoder.transform(query.iloc[[0]])
    pd.testing.assert_frame_equal(singleton, result.iloc[[0]])
    assert '64gb' in result.name.iloc[3]
    assert result.category_condition.iloc[0].endswith('|1')
    assert not {'price', 'train_id'} & set(result)


def test_text_features_handle_empty_and_unseen_fields_and_pickle_round_trip():
    import pickle
    frame = pd.concat([rows()] * 3, ignore_index=True)
    encoder = features.ListingFeatures().fit(frame)
    prepared = encoder.transform(frame)
    text = features.make_ridge(max_features=100, alpha=2., char_features=100)
    text.fit(prepared, np.log1p(frame.price))
    unknown = frame.iloc[:2].copy()
    unknown['category_name'] = 'never/seen/category'
    unknown['brand_name'] = 'unseen'
    unknown['name'] = None
    prediction = text.predict(encoder.transform(unknown))
    restored = pickle.loads(pickle.dumps((encoder, text)))
    np.testing.assert_array_equal(prediction, restored[1].predict(restored[0].transform(unknown)))
    assert np.isfinite(prediction).all()
    no_char = features.make_ridge(max_features=100, alpha=2., char_features=0)
    no_char.fit(prepared, np.log1p(frame.price))
    assert len(no_char.predict(prepared)) == len(frame)


def test_words_keep_alphanumeric_model_and_unit_tokens():
    assert features.normalize('iPhone-7! 64 GB / 2.5 oz') == 'iphone 7 64gb 2.5oz'
    assert features.normalize(None) == '<missing>'
    assert features.normalize('iPhone6S 64 GB') == 'iphone 6s 64gb'
    extractor = features.make_ridge(max_features=100, char_features=0)['features']
    tokenizer = extractor.transformers[0][1].build_analyzer()
    assert '6' in tokenizer('iphone 6') and 'x' in tokenizer('iphone x')
    frame = rows(); frame.loc[0, 'name'] = 'iphone6'
    legacy = features.ListingFeatures().fit(frame)
    del legacy.normalization_version_
    assert legacy.transform(frame).name.iloc[0] == 'iphone6'
    refitted = features.ListingFeatures(version=1).fit(frame)
    assert refitted.transform(frame).name.iloc[0] == 'iphone6'


def test_residual_features_are_frozen_finite_and_supply_word_price_summary():
    frame = pd.concat([rows()] * 3, ignore_index=True)
    encoder = features.ResidualFeatures().fit(frame, np.log1p(frame.price))
    x = encoder.transform(frame)
    prepared = encoder.listings_.transform(frame)
    np.testing.assert_array_equal(x, encoder.transform(frame, prepared=prepared))
    assert x.shape == (len(frame), len(encoder.feature_names_))
    np.testing.assert_array_equal(x, encoder.transform(frame.assign(price=10000., train_id=-1)))
    query = frame.iloc[:2].copy()
    query['name'] = ['unknownzzz', 'apple iphone']
    query['brand_name'] = 'unseen'
    z = encoder.transform(query)
    assert np.isfinite(z).all() and z.dtype == np.float32
    np.testing.assert_allclose(z[0, -5:-2], encoder.prior_)
    assert z[0, -2] == 0
    assert z[1, -2] > 0
    with pytest.raises(ValueError):
        features.ResidualFeatures().fit(frame, [1.])

