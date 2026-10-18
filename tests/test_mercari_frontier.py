"""End-to-end experiment protocol checks with tiny real text/tree/neural fits."""
from argparse import Namespace
import importlib
import json
import pickle
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks import mercari_frontier as frontier
from test_mercari import listings


class ConstantEntry:
    config = dict(weights=[.2, .3, .5])

    def predict(self, frame):
        return np.full(len(frame), 3.)

    def parts(self, frame):
        return np.full((len(frame), 3), 3.), np.ones((len(frame), 4)), np.zeros(len(frame))

    def close(self):
        pass


@pytest.fixture
def windows(tmp_path):
    root = tmp_path / 'data'
    root.mkdir()
    frame = listings(100)
    frame.to_csv(root / 'train.tsv.gz', sep='\t', index=False)
    for i, window in enumerate(frontier.WINDOWS):
        part = listings(20).assign(train_id=np.arange(20) + (i + 1) * 100)
        if window == 'confirmation':
            part = part.drop(columns='price')
        part.to_csv(root / (window + '.tsv.gz'), sep='\t', index=False)
    return root


def settings(data, output, **changes):
    return Namespace(data_dir=data, output_dir=output, components=4, max_features=100,
                     threads=1, allow_cpu=True, revision=None, epochs=1, tree_rounds=5,
                     neural_only='all', sparse_view='separate', sparse_width=256,
                     embedding_dtype='float32', **changes)


def test_prepare_hides_confirmation_labels_and_records_baseline_and_groups(tmp_path):
    data = tmp_path / 'input'; data.mkdir()
    frame = listings(300)
    frame.to_csv(data / 'train.tsv', sep='\t', index=False)
    baseline = tmp_path / 'baseline.pkl'
    baseline.write_bytes(pickle.dumps(ConstantEntry()))
    out = tmp_path / 'prepared'
    frontier.prepare_data(settings(data, out, baseline=baseline))
    parts = frontier.read_windows(out)
    assert 'price' not in parts['confirmation']
    assert len(set().union(*(set(p.train_id) for p in parts.values()))) == 300
    report = json.loads((out / 'protocol.json').read_text())
    assert report['confirmation_labels_transferred'] is False
    assert sum(v['rows'] for v in report['splits'].values()) == 300
    with np.load(out / 'baseline.npz') as p, np.load(out / 'evaluation.npz') as y:
        for window in frontier.WINDOWS:
            np.testing.assert_array_equal(p[window + '_ids'], y[window + '_ids'])
            np.testing.assert_array_equal(p[window], 3.)


def test_tree_experiment_trains_real_models_with_ordered_predictions(windows, tmp_path):
    pytest.importorskip('lightgbm')
    out = tmp_path / 'trees'
    frontier.tree_study(settings(windows, out))
    report = json.loads((out / 'report.json').read_text())
    assert len(report['models']) == 4
    for name in report['models']:
        with np.load(out / (name + '.npz')) as p:
            assert len(p['selection']) == 20 and np.isfinite(p['selection']).all()
            np.testing.assert_array_equal(p['selection_ids'], np.arange(20) + 200)


def test_calibration_uses_disjoint_labels_and_replays_all_tree_families(windows, tmp_path):
    pytest.importorskip('lightgbm')
    pytest.importorskip('xgboost')
    baseline = tmp_path / 'baseline.pkl'; baseline.write_bytes(pickle.dumps(ConstantEntry()))
    parts = frontier.read_windows(windows)
    encoder = frontier.LatentText(components=4, max_features=100, min_df=1).fit(parts['train'])
    encoder_path = tmp_path / 'encoder.pkl'; encoder_path.write_bytes(pickle.dumps(encoder))
    experts = tmp_path / 'experts'; experts.mkdir()
    values = {k: np.full(len(parts[k]), 3.) for k in frontier.WINDOWS}
    frontier.save_predictions(experts, {'new_expert': values}, parts)
    np.savez(experts / 'not_predictions.npz', unrelated=np.zeros(1))
    out = tmp_path / 'calibrated'
    args = settings(windows, out, baseline=baseline, lsa_encoder=encoder_path, experts=[experts])
    frontier.calibration_study(args)
    report = json.loads((out / 'report.json').read_text())
    assert set(report['models']) == {'cal_libtree3', 'cal_libtree5', 'cal_lightgbm', 'cal_xgboost',
                                     'cal_meta_ridge10', 'cal_meta_ridge100'}
    assert report['calibration_rows'] == 20 and not (out / 'confirmation_score.json').exists()
    assert all(0 <= value['strength'] <= 1 for value in report['models'].values() if 'strength' in value)
    duplicate = tmp_path / 'duplicate'; duplicate.mkdir()
    frontier.save_predictions(duplicate, {'new_expert': values}, parts)
    with pytest.raises(ValueError, match='unique'):
        frontier.calibration_study(settings(windows, tmp_path / 'duplicatecal', baseline=baseline,
            lsa_encoder=encoder_path, experts=[experts, duplicate]))
    values['selection_ids'] = np.arange(20)[::-1]
    np.savez(experts / 'bad.npz', **values)
    with pytest.raises(ValueError, match='IDs'):
        frontier.calibration_study(settings(windows, tmp_path / 'badcal', baseline=baseline,
            lsa_encoder=encoder_path, experts=[experts]))


def test_blend_freezes_before_confirmation_and_rejects_misaligned_ids(tmp_path, monkeypatch):
    data, experts = tmp_path / 'data', tmp_path / 'experts'
    data.mkdir(); experts.mkdir()
    y = np.linspace(1., 4., 100)
    labels, baseline, extra = {}, {}, {}
    for window in frontier.WINDOWS:
        labels[window] = y
        baseline[window], extra[window] = y + .2, y - .1
        for container in (labels, baseline, extra):
            container[window + '_ids'] = np.arange(100)
    np.savez(data / 'evaluation.npz', **labels)
    np.savez(data / 'baseline.npz', **baseline)
    np.savez(experts / 'good.npz', **extra)
    worse = {key: value.copy() for key, value in baseline.items()}
    for window in frontier.WINDOWS:
        worse[window] += 1.
    np.savez(experts / 'worse.npz', **worse)
    np.savez(experts / 'nonprediction.npz', values=np.zeros(1))
    out = tmp_path / 'ensemble'
    original = frontier.paired_comparison
    def check_frozen(*args, **kwargs):
        assert (out / 'frozen.json').is_file()
        return original(*args, **kwargs)
    monkeypatch.setattr(frontier, 'paired_comparison', check_frozen)
    frontier.ensemble_study(settings(data, out, experts=[experts], external_pretrained_data=True))
    report = json.loads((out / 'report.json').read_text())
    assert report['frozen']['external_pretrained_data'] is True
    assert report['comparison']['candidate_rmsle'] < .001
    assert 'good' in report['leave_one_expert_out']
    duplicate = tmp_path / 'duplicate'; duplicate.mkdir()
    np.savez(duplicate / 'good.npz', **extra)
    with pytest.raises(ValueError, match='unique'):
        frontier.ensemble_study(settings(data, tmp_path / 'duplicate-blend', experts=[experts, duplicate]))
    extra['selection_ids'] = np.arange(100)[::-1]
    np.savez(experts / 'good.npz', **extra)
    with pytest.raises(ValueError, match='IDs'):
        frontier.ensemble_study(settings(data, tmp_path / 'bad', experts=[experts]))


def test_gpu_memory_growth_is_set_before_loading_training_data(windows, tmp_path, monkeypatch):
    tf = pytest.importorskip('tensorflow')
    device, called = SimpleNamespace(name='test-gpu'), []
    monkeypatch.setattr(tf.config, 'list_physical_devices', lambda name: [device])
    monkeypatch.setattr(tf.config.experimental, 'set_memory_growth', lambda d, enabled: called.append((d, enabled)))
    def stop(_):
        raise RuntimeError('stop before loading data')
    monkeypatch.setattr(frontier, 'read_windows', stop)
    with pytest.raises(RuntimeError, match='stop before'):
        frontier.gpu_study(settings(windows, tmp_path / 'memory'))
    assert called == [(device, True)]


@pytest.mark.parametrize('neural_only', ['all', 'sequence', 'sparse', 'stemmed'])
def test_real_gpu_expert_pipeline_has_an_explicit_cpu_functional_mode(windows, tmp_path, monkeypatch, neural_only):
    tf = pytest.importorskip('tensorflow')
    pytest.importorskip('numba')
    monkeypatch.setattr(tf.config, 'list_physical_devices', lambda name=None: [])
    args = settings(windows, tmp_path / 'gpu')
    args.neural_only = 'sparse' if neural_only == 'stemmed' else neural_only
    args.sparse_view = 'stemmed' if neural_only == 'stemmed' else 'joint' if neural_only == 'sparse' else 'separate'
    args.allow_cpu = False
    with pytest.raises(RuntimeError, match='real TensorFlow GPU'):
        frontier.gpu_study(args)
    args.allow_cpu = True
    frontier.gpu_study(args)
    report = json.loads((args.output_dir / 'report.json').read_text())
    assert report['gpu_devices'] == []
    expected = []
    if args.neural_only != 'sparse':
        expected += ['cnn128_mse', 'gru96_huber']
    if args.neural_only != 'sequence':
        prefix = 'stemmed_' if neural_only == 'stemmed' else ''
        expected += [prefix + name for name in ['sparse256_tfidf_mse', 'sparse256_binary_huber', 'sparse256_binary_bins']]
    assert set(report['models']) - {'sparse_feature_seconds'} == set(expected)
    for name in expected:
        assert np.isfinite(np.load(args.output_dir / (name + '.npz'))['confirmation']).all()


def test_vector_experiment_pins_external_model_and_keeps_gpu_requirement(windows, tmp_path, monkeypatch):
    pytest.importorskip('lightgbm')
    cuda = SimpleNamespace(is_available=lambda: False, get_device_name=lambda: 'test device')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(model_info=lambda _: SimpleNamespace(sha='resolved-test-revision')))
    class Encoder:
        def __init__(self, model, revision, device):
            assert revision in ('resolved-test-revision', 'pinned-test-revision')
        def encode(self, texts, **kwargs):
            return np.asarray([[len(t), len(t.split())] for t in texts], dtype=np.float32)
        def half(self):
            return self
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=Encoder))
    original = importlib.metadata.version
    monkeypatch.setattr(frontier.importlib.metadata, 'version', lambda name: 'test' if name == 'sentence-transformers' else original(name))
    args = settings(windows, tmp_path / 'vectors')
    args.allow_cpu = False
    with pytest.raises(RuntimeError, match='real Torch GPU'):
        frontier.vector_study(args)
    args.allow_cpu = True
    frontier.vector_study(args)
    report = json.loads((args.output_dir / 'report.json').read_text())
    assert report['revision'] == 'resolved-test-revision' and report['external_pretrained_data']
    assert report['gpu'] is None
    args.output_dir = tmp_path / 'vectors-pinned'
    args.revision = 'pinned-test-revision'
    args.embedding_dtype = 'float16'
    monkeypatch.setattr(cuda, 'is_available', lambda: True)
    frontier.vector_study(args)
    report = json.loads((args.output_dir / 'report.json').read_text())
    assert report['gpu'] == 'test device'
    assert report['embedding_dtype'] == 'float16' and report['precision_audit']['max_absolute_error'] == 0.
    class InvalidHalf(Encoder):
        bad = False
        def half(self):
            self.bad = True
            return self
        def encode(self, texts, **kwargs):
            return super().encode(texts, **kwargs) * (np.nan if self.bad else 1.)
    monkeypatch.setattr(sys.modules['sentence_transformers'], 'SentenceTransformer', InvalidHalf)
    args.output_dir = tmp_path / 'invalid-half'
    with pytest.raises(RuntimeError, match='precision audit'):
        frontier.vector_study(args)


def test_main_validates_resources_and_dispatches(monkeypatch, tmp_path):
    basic = ['trees', '--data-dir', str(tmp_path), '--output-dir', str(tmp_path / 'out')]
    for extra in (['--threads', '0'], ['--epochs', '6']):
        with pytest.raises(SystemExit) as error:
            frontier.main(basic + extra)
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        frontier.main(['prepare', *basic[1:]])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        frontier.main(['calibrate', *basic[1:]])
    assert error.value.code == 2
    called = []
    monkeypatch.setattr(frontier, 'tree_study', lambda args: called.append(args.mode))
    frontier.main(basic + ['--neural-only', 'sparse', '--sparse-view', 'joint', '--sparse-width', '128'])
    assert called == ['trees']
    monkeypatch.setattr(sys, 'argv', ['mercari_frontier', '--help'])
    with pytest.raises(SystemExit) as error:
        runpy.run_module('benchmarks.mercari_frontier', run_name='__main__')
    assert error.value.code == 0


@pytest.mark.parametrize('inactive_weight', [0., 1e-18])
def test_ablation_distinguishes_exact_zero_from_solver_boundary_noise(tmp_path, monkeypatch, inactive_weight):
    """Workflow branches are explicit; real SLSQP is tested independently.

    Depending on BLAS/optimizer rounding, the existing integration fixture can
    return an exact zero or a tiny positive bound value for its weak expert.
    Neither case should decide whether CI reaches the zero-weight branch.
    """
    data, experts = tmp_path / 'data', tmp_path / 'experts'
    data.mkdir(); experts.mkdir()
    y = np.linspace(1., 4., 100)
    error = .2 + np.arange(len(y)) / 1000.
    identifiers = {window + '_ids': np.arange(len(y)) for window in frontier.WINDOWS}
    np.savez(data / 'evaluation.npz', **identifiers, **{window: y for window in frontier.WINDOWS})
    np.savez(data / 'baseline.npz', **identifiers, **{window: y + error for window in frontier.WINDOWS})
    np.savez(experts / 'good.npz', **identifiers, **{window: y - error / 2 for window in frontier.WINDOWS})
    np.savez(experts / 'inactive.npz', **identifiers, **{window: y + 5. + error for window in frontier.WINDOWS})
    original = frontier.blend_weights
    calls = []
    def controlled_boundary(predictions, labels, **kwargs):
        calls.append(predictions.shape[1])
        if predictions.shape[1] == 3:
            return np.array([1 / 3, 2 / 3 - inactive_weight, inactive_weight])
        return original(predictions, labels, **kwargs)
    monkeypatch.setattr(frontier, 'blend_weights', controlled_boundary)
    out = tmp_path / 'blend'
    frontier.ensemble_study(settings(data, out, experts=[experts]))
    report = json.loads((out / 'report.json').read_text())
    assert report['frozen']['names'] == ['baseline', 'good', 'inactive']
    assert report['frozen']['weights'][-1] == inactive_weight
    assert set(report['leave_one_expert_out']) == ({'good'} if inactive_weight == 0 else {'good', 'inactive'})
    assert calls == ([3, 2] if inactive_weight == 0 else [3, 2, 2])
    with np.load(out / 'ensemble.npz') as prediction:
        np.testing.assert_allclose(prediction['selection'], y, atol=1e-12)


def test_product_calibration_statistics_fit_base_only_and_reject_confirmation_prices(windows,tmp_path):
    baseline=tmp_path/'baseline.pkl';baseline.write_bytes(pickle.dumps(ConstantEntry()))
    parts=frontier.read_windows(windows)
    encoder=frontier.LatentText(components=4,max_features=100,min_df=1).fit(parts['train'])
    encoder_path=tmp_path/'encoder.pkl';encoder_path.write_bytes(pickle.dumps(encoder))
    out=tmp_path/'product-calibration'
    frontier.calibration_study(settings(windows,out,baseline=baseline,lsa_encoder=encoder_path,experts=[],product_views=True))
    report=json.loads((out/'report.json').read_text())
    assert len(report['product_feature_names'])==90
    assert report['confirmation_labels_loaded'] is False
    with (out/'product-views.pkl').open('rb') as stream:views=pickle.load(stream)
    assert len(views)==6
    for view in views.values():
        assert view.prior_==pytest.approx(np.log1p(parts['train'].price).mean())
        assert sum(view.sufficient_['title']['count'])==len(parts['train'])
    parts['confirmation'].assign(price=1).to_csv(windows/'confirmation.tsv.gz',sep='\t',index=False)
    with pytest.raises(ValueError,match='confirmation labels'):
        frontier.calibration_study(settings(windows,tmp_path/'leaky-product',baseline=baseline,
                                           lsa_encoder=encoder_path,experts=[],product_views=True))
