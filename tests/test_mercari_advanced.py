"""Honest held-out calibration, simplex blending and complete entry replay."""
import json
import pickle
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('tensorflow')
pytest.importorskip('numba')
from benchmarks import mercari_advanced as advanced
from benchmarks.mercari import listing_groups
from test_mercari import listings


def tiny_config():
    config = advanced.DEFAULT.copy()
    config.update(max_features=100, char_features=100, neural_width=8, max_words=100,
                  neural_rates=[.003], neural_batches=[16], fm_workers=1, fm_rank=2, fm_epochs=1, threads=1,
                  native=dict(n_estimators=2, max_depth=2, max_bins=16, min_samples_leaf=2,
                              learning_rate=.1, reg_lambda=10., n_jobs=1))
    return config


def test_configuration_and_convex_blend_are_explicit_and_validated():
    config = tiny_config()
    advanced.validate(config)
    for changes in [dict(weights=[0, 0, 0]), dict(weights=[-1, 1, 1]),
                    dict(fm_epochs=0), dict(neural_loss='unknown'), dict(unknown=1)]:
        with pytest.raises(ValueError):
            advanced.validate(dict(config, **changes))
    predictions = np.array([[1., 3., 2.], [2., 0., 1.]])
    weights, score = advanced.select_weights(predictions, [2., 1.])
    assert np.sum(weights) == pytest.approx(1.) and score == pytest.approx(0.)
    with pytest.raises(ValueError):
        advanced.select_weights(predictions, [1.])


def test_base_and_calibration_groups_do_not_overlap():
    frame = listings(120)
    frame = pd.concat([frame, frame.iloc[:2].assign(train_id=[1001, 1002])], ignore_index=True)
    base, calibration = advanced.calibration_split(frame, .2, 42)
    assert len(base) + len(calibration) == len(frame)
    assert set(listing_groups(base)).isdisjoint(listing_groups(calibration))


def test_tiny_entry_trains_replays_and_explains_log_contributions(tmp_path):
    frame = listings(100)
    config = tiny_config()
    base, calibration = advanced.calibration_split(frame, .2, 42)
    entry, stages = advanced.fit_entry(base, calibration, config)
    try:
        query = frame.iloc[:3].copy()
        before = entry.predict(query)
        np.testing.assert_array_equal(entry.parts(query)[0], entry.experts.predict(query))
        np.testing.assert_array_equal(before, entry.predict(query.assign(price=1e6, train_id=-1)))
        config['weights'] = [1., 0., 0.]
        np.testing.assert_array_equal(before, entry.predict(query))
        after = pickle.loads(pickle.dumps(entry))
        try:
            np.testing.assert_allclose(before, after.predict(query), atol=1e-7)
        finally:
            after.close()
        explanation = entry.explain(query.iloc[:1])
        assert sum(explanation['weighted_expert_log_contributions']) + explanation['weighted_native_correction'] == pytest.approx(before[0])
        assert explanation['price'] == pytest.approx(np.expm1(max(before[0], 0.)))
        assert set(stages) == {'experts', 'residual_features_seconds', 'native_fit_seconds'}
        with pytest.raises(ValueError):
            entry.explain(query)
    finally:
        entry.close()


def test_cli_study_never_evaluates_holdout_without_explicit_audit(tmp_path, monkeypatch):
    data = tmp_path / 'data'; data.mkdir()
    listings(120).to_csv(data / 'train.tsv', sep='\t', index=False)
    path = tmp_path / 'config.json'; path.write_text(json.dumps(tiny_config()))
    args = ['--data-dir', str(data), '--config', str(path), '--output-dir', str(tmp_path / 'study'), '--mode', 'study']
    report = advanced.main(args)
    assert 'holdout_rmsle' not in report and report['mode'] == 'study'
    assert report['selection_rmsle'] >= 0
    with pytest.raises(ValueError):
        advanced.main(args)
    monkeypatch.setattr(sys, 'argv', ['mercari_advanced', '--help'])
    with pytest.raises(SystemExit) as caught:
        advanced.main()
    assert caught.value.code == 0


def test_cli_final_audit_error_boundaries_and_module_entrypoint(tmp_path, monkeypatch):
    data = tmp_path / 'data'; data.mkdir()
    frame = listings(120)
    frame.to_csv(data / 'train.tsv', sep='\t', index=False)
    query = frame.iloc[:3].drop(columns='price').rename(columns={'train_id': 'test_id'})
    query.to_csv(data / 'test_stg2.tsv', sep='\t', index=False)
    pd.DataFrame({'test_id': query.test_id, 'price': 0}).to_csv(data / 'sample_submission_stg2.csv', index=False)
    config = tiny_config(); config['neural_loss'] = 'huber'; config['neural_architecture'] = 'cnn'
    path = tmp_path / 'config.json'; path.write_text(json.dumps(config))
    base, calibration = advanced.calibration_split(frame, .2, 42)
    entry, stages = advanced.fit_entry(base, calibration, config)
    payload = pickle.dumps(entry); entry.close()
    monkeypatch.setattr(advanced, 'fit_entry', lambda *args: (pickle.loads(payload), stages))
    args = ['--data-dir', str(data), '--config', str(path)]
    audited = advanced.main(args + ['--mode', 'study', '--audit-holdout', '--output-dir', str(tmp_path / 'audit')])
    assert audited['holdout_rmsle'] >= 0
    final = advanced.main(args + ['--output-dir', str(tmp_path / 'final'), '--omit-model-artifact'])
    assert final['submission']['rows'] == 3
    assert not (tmp_path / 'final/model.pkl').exists()
    assert pd.read_csv(tmp_path / 'final/submission.csv').price.ge(0).all()
    with pytest.raises(ValueError, match='only available'):
        advanced.main(args + ['--audit-holdout', '--output-dir', str(tmp_path / 'invalid')])
    frame.assign(train_id=1).to_csv(data / 'train.tsv', sep='\t', index=False)
    with pytest.raises(ValueError, match='unique train_id'):
        advanced.main(args + ['--output-dir', str(tmp_path / 'invalid')])
    monkeypatch.setattr(sys, 'argv', ['mercari_advanced', '--help'])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module('benchmarks.mercari_advanced', run_name='__main__')
    assert caught.value.code == 0


def test_sparse_text_adapter_uses_same_feature_space_and_has_no_sequence_work():
    config = tiny_config()
    config.update(neural_architecture='sparse', neural_loss='huber', sparse_word_limit=100,
                  sparse_char_limit=100, sparse_min_df=2)
    advanced.validate(config)
    frame = listings(100)
    base, calibration = advanced.calibration_split(frame, .2, 42)
    entry, _ = advanced.fit_entry(base, calibration, config)
    try:
        result = entry.predict(frame.iloc[:4])
        assert np.isfinite(result).all()
        clone = pickle.loads(pickle.dumps(entry))
        try:
            np.testing.assert_allclose(result, clone.predict(frame.iloc[:4]), atol=1e-7)
        finally:
            clone.close()
    finally:
        entry.close()


def test_mixed_neural_members_refit_legacy_text_and_preserve_weighted_predictions():
    config = tiny_config()
    config['neural_members'] = [
        dict(weight=.4, neural_architecture='bag', neural_text_version=1),
        dict(weight=.6, neural_architecture='sparse', neural_loss='bins', sparse_optimizer='row_adam',
             sparse_word_limit=100, sparse_char_limit=100, sparse_min_df=2, sparse_bins=8)]
    advanced.validate(config)
    for bad in [dict(config, neural_members=[dict(weight=.4)]),
                dict(config, neural_members=[dict(weight=1., unknown=1)])]:
        with pytest.raises(ValueError):
            advanced.validate(bad)
    frame = listings(100)
    base, calibration = advanced.calibration_split(frame, .2, 42)
    entry, _ = advanced.fit_entry(base, calibration, config)
    try:
        query = frame.iloc[:3]
        prepared = entry.experts.listing.transform(query)
        matrix = entry.experts.extractor.transform(prepared)
        mixture = entry.experts.neural
        individual = [m.predict(query, prepared, matrix) for m in mixture.members]
        expected = .4 * individual[0] + .6 * individual[1]
        np.testing.assert_allclose(mixture.predict(query, prepared, matrix), expected)
        result = entry.predict(query)
        clone = pickle.loads(pickle.dumps(entry))
        try:
            np.testing.assert_allclose(result, clone.predict(query), atol=1e-7)
        finally:
            clone.close()
    finally:
        entry.close()
