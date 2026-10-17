"""TDD guards for a leakage-safe Mercari LibTree competition entry."""
from argparse import Namespace
import json
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks import mercari


def listings(n=120):
    i = np.arange(n)
    return pd.DataFrame({'train_id': i, 'name': [f'item model {v} shirt' for v in i],
        'item_condition_id': i % 5 + 1, 'category_name': np.where(i % 2, 'Men/Tops/Shirts', 'Women/Tops/Shirts'),
        'brand_name': np.where(i % 3, 'brand-a', 'brand-b'), 'shipping': i % 2,
        'item_description': [f'A good item in condition number {v}' for v in i],
        'price': 10. + i % 20})


def test_rmsle_matches_official_log_definition_and_rejects_invalid_prices():
    assert mercari.rmsle([0, 3], [0, 8]) == pytest.approx(np.log(9 / 4) / np.sqrt(2))
    assert mercari.rmsle([0, 5], [0, 5]) == 0
    for y, p in (([-1, 1], [0, 1]), ([0, 1], [np.inf, 1]), ([0, 1], [0]), ([], [])):
        with pytest.raises(ValueError):
            mercari.rmsle(y, p)


def test_duplicate_listings_stay_together_and_labels_ids_do_not_define_groups():
    frame = listings()
    duplicates = frame.iloc[:10].copy().assign(train_id=np.arange(1000, 1010), price=999.)
    frame = pd.concat([frame, duplicates], ignore_index=True)
    groups = mercari.listing_groups(frame)
    assert list(groups[:10]) == list(groups[-10:])
    a, b, c = mercari.split(frame, 42)
    assert len(a) + len(b) + len(c) == len(frame)
    sets = [set(mercari.listing_groups(part)) for part in (a, b, c)]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])


def test_frozen_dense_features_and_missing_categories():
    frame = listings()
    y = np.log1p(frame.price.to_numpy())
    encoder = mercari.DenseFeatures().fit(frame, y)
    a = encoder.transform(frame)
    poisoned = frame.assign(price=999., train_id=-1)
    np.testing.assert_array_equal(a, encoder.transform(poisoned))
    unknown = frame.iloc[:2].copy()
    unknown['brand_name'] = None
    unknown['category_name'] = 'unseen/new/category'
    before = encoder.transform(unknown)
    unknown.iloc[1, unknown.columns.get_loc('brand_name')] = 'changed'
    np.testing.assert_array_equal(before[0], encoder.transform(unknown)[0])
    assert np.isfinite(before).all() and before.dtype == np.float32
    assert a.shape[1] == len(encoder.feature_names_)
    for invalid in ([1], np.full(len(frame), np.nan)):
        with pytest.raises(ValueError, match='labels'):
            mercari.DenseFeatures().fit(frame, invalid)


def test_cross_fitting_excludes_a_rows_own_fold_labels(monkeypatch):
    frame = listings(60)
    y = np.log1p(frame.price.to_numpy())

    class MeanText:
        def fit(self, x, labels):
            self.mean = np.mean(labels)
            return self

        def predict(self, x):
            return np.full(len(x), self.mean)

    monkeypatch.setattr(mercari, 'make_text', lambda *args: MeanText())
    a, dense, folds = mercari.cross_fit(frame, y, 3, 100)
    first = folds[0]['validation_positions']
    changed = y.copy()
    changed[first] += 100
    b, other, _ = mercari.cross_fit(frame, changed, 3, 100)
    np.testing.assert_array_equal(a[first], b[first])
    np.testing.assert_array_equal(dense[first], other[first])
    assert np.isfinite(a).all() and len(folds) == 3


def test_schema_preparation_does_not_treat_missing_text_as_nan_tokens():
    frame = listings(10)
    frame.loc[0, ['name', 'item_description', 'brand_name']] = None
    result = mercari.prepare(frame)
    assert not result[list(mercari.COLUMNS)].isna().any().any()
    with pytest.raises(ValueError, match='columns'):
        mercari.prepare(frame.drop(columns='shipping'))


def test_ensemble_adds_scaled_residual_in_log_space_and_passes_base_feature():
    class Encoder:
        def transform(self, frame):
            return np.zeros((len(frame), 2), dtype=np.float32)

    class Text:
        def predict(self, frame):
            return np.full(len(frame), 3.)

    class Native:
        def predict(self, x):
            self.features = x.copy()
            return np.full(len(x), 2.)

        def close(self):
            self.closed = True

    native = Native()
    entry = mercari.Entry(Text(), Encoder(), native, 'ridge_plus_libtree', .25)
    np.testing.assert_array_equal(entry.predict(listings(2)), [3.5, 3.5])
    np.testing.assert_array_equal(native.features[:, -1], [3., 3.])
    plain = mercari.Entry(Text(), Encoder(), native, 'libtree')
    np.testing.assert_array_equal(plain.predict(listings(2)), [2., 2.])
    assert native.features.shape == (2, 2)
    entry.close()
    assert native.closed


def test_submission_preserves_ids_and_refuses_sample_mismatch(tmp_path):
    future = listings(10).drop(columns=['price', 'train_id']).assign(test_id=[f'{i:04d}' for i in range(10)])
    path = tmp_path / 'test.tsv'
    sample = tmp_path / 'sample.csv'
    future.to_csv(path, sep='\t', index=False)
    pd.DataFrame({'test_id': future.test_id, 'price': 0.}).to_csv(sample, index=False)

    class Predictor:
        def predict(self, x):
            return np.arange(len(x), dtype=float) - 1

    info = mercari.write_submission(Predictor(), path, sample, tmp_path / 'submission.csv', chunk_size=4)
    got = pd.read_csv(tmp_path / 'submission.csv', dtype={'test_id': str})
    assert got.test_id.tolist() == future.test_id.tolist()
    assert got.columns.tolist() == ['test_id', 'price'] and (got.price >= 0).all()
    assert info['rows'] == 10
    pd.DataFrame({'test_id': future.test_id[::-1], 'price': 0.}).to_csv(sample, index=False)
    with pytest.raises(ValueError, match='order'):
        mercari.write_submission(Predictor(), path, sample, tmp_path / 'bad.csv', chunk_size=4)


@pytest.mark.parametrize('issue', ['schema', 'duplicate', 'test_price', 'nonfinite', 'short_test'])
def test_bad_submission_inputs_are_rejected(tmp_path, issue):
    future = listings(10).drop(columns=['price', 'train_id']).assign(test_id=np.arange(10))
    sample = pd.DataFrame({'test_id': np.arange(10), 'price': 0.})
    if issue == 'schema':
        sample = sample.rename(columns={'test_id': 'wrong'})
    if issue == 'duplicate':
        sample['test_id'] = 1
    if issue == 'test_price':
        future['price'] = 999.
    if issue == 'short_test':
        future = future.iloc[:5]
    path, sample_path = tmp_path / 'test.tsv', tmp_path / 'sample.csv'
    future.to_csv(path, sep='\t', index=False)
    sample.to_csv(sample_path, index=False)

    class Predictor:
        def predict(self, x):
            return np.full(len(x), np.nan if issue == 'nonfinite' else 1.)

    with pytest.raises(ValueError):
        mercari.write_submission(Predictor(), path, sample_path, tmp_path / 'bad.csv')


def test_end_to_end_real_ridge_native_libtree_and_model_reload(tmp_path):
    import pickle
    data = tmp_path / 'data'
    data.mkdir()
    listings(120).to_csv(data / 'train.tsv', sep='\t', index=False)
    future = listings(12).drop(columns=['price', 'train_id']).assign(test_id=np.arange(12))
    future.to_csv(data / 'test.tsv', sep='\t', index=False)
    pd.DataFrame({'test_id': future.test_id, 'price': 0.}).to_csv(data / 'sample_submission.csv', index=False)
    args = Namespace(data_dir=data, output_dir=tmp_path / 'run', test_file='test.tsv',
        sample_file='sample_submission.csv', seed=42, folds=3, max_features=100,
        trees=3, depth=2, threads=2, repeats=2, train_rows=120)
    report = mercari.run(args)
    assert json.loads((args.output_dir / 'report.json').read_text()) == report
    assert report['selected']['engine'] in ('libtree', 'ridge_plus_libtree')
    assert set(report['holdout']) == {'ridge', 'libtree', 'selected'}
    assert report['submission']['status'] == 'generated_not_submitted'
    assert report['submission']['rows'] == 12
    for row in report['native_timings']:
        assert row['fit_seconds']['mean'] == pytest.approx(np.mean(row['fit_seconds']['samples']))
    with (args.output_dir / 'model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    pred = np.expm1(np.maximum(model.predict(future), 0))
    got = pd.read_csv(args.output_dir / 'submission.csv')
    np.testing.assert_allclose(got.price, pred, rtol=1e-6)
    model.close()
    assert json.loads((args.output_dir / 'tree.json').read_text())
    assert json.loads((args.output_dir / 'explanation.json').read_text())
    with pytest.raises(ValueError, match='empty'):
        mercari.run(args)
    original = listings(120)
    _, _, heldout = mercari.split(original.sample(n=120, random_state=42), 42)
    original.loc[heldout.index, 'price'] += 100
    original.to_csv(data / 'train.tsv', sep='\t', index=False)
    args.output_dir = tmp_path / 'poisoned-holdout'
    other = mercari.run(args)
    assert other['selected'] == report['selected']
    assert other['tuning'] == report['tuning']
    assert other['holdout']['selected'] != report['holdout']['selected']


def test_cli_can_explain_arguments_without_data_or_kaggle_credentials(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['mercari', '--help'])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(mercari.__file__, run_name='__main__')
    assert caught.value.code == 0
    assert '--data-dir' in capsys.readouterr().out


def test_cli_routes_to_runner_with_declared_resource_budget(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(mercari, 'run', lambda args: seen.append(args))
    mercari.main(['--data-dir', str(tmp_path), '--output-dir', str(tmp_path / 'run'), '--threads', '1'])
    assert seen[0].threads == 1 and seen[0].folds == 3


def test_invalid_experiment_inputs_fail_before_training(tmp_path):
    args = Namespace(data_dir=tmp_path, output_dir=tmp_path / 'run', trees=3,
                     depth=2, threads=1, repeats=1, max_features=100, folds=3)
    args.repeats = 0
    with pytest.raises(ValueError, match='positive'):
        mercari.run(args)
    args.repeats = 1
    for frame in (listings().drop(columns='price'), listings().assign(train_id=1)):
        frame.to_csv(tmp_path / 'train.tsv', sep='\t', index=False)
        with pytest.raises(ValueError, match='training data'):
            mercari.run(args)
