"""Interleaved serving measurements preserve means, predictions and resources."""
import json
import pickle
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('numba')
from numba import get_num_threads
from benchmarks import mercari_inference as serving


class Estimator:
    def __init__(self, offset=0):
        self.offset, self.closed = offset, False

    def predict(self, frame):
        return frame.test_id.to_numpy(dtype=float) + self.offset

    def close(self):
        self.closed = True


def test_interleaving_means_warmup_and_thread_restoration(monkeypatch):
    calls = []
    class Traced(Estimator):
        def predict(self, frame):
            calls.append(self.offset)
            return super().predict(frame)
    clock = iter([0., 1., 1., 3., 3., 6., 6., 10.])
    monkeypatch.setattr(serving.time, 'perf_counter', lambda: next(clock))
    before = get_num_threads()
    report = serving.measure({'baseline': Traced(0), 'optimized': Traced(1)},
                             pd.DataFrame({'test_id': [0, 1]}), repeats=2,
                             threads=1, fm_threads=1)
    assert calls == [0, 1, 0, 1, 1, 0]
    assert report['timings']['baseline']['samples'] == [1., 4.]
    assert report['timings']['optimized']['samples'] == [2., 3.]
    assert report['timings']['baseline']['mean'] == 2.5
    assert report['optimized_to_baseline_mean_cost_ratio'] == 1.
    assert report['warmup_runs_per_model'] == 1 and get_num_threads() == before


def test_invalid_measurements_and_prediction_drift_restore_threads():
    frame = pd.DataFrame({'test_id': [0, 1]})
    for changes in [dict(repeats=0), dict(threads=0), dict(fm_threads=0)]:
        with pytest.raises(ValueError):
            serving.measure({'baseline': Estimator(), 'optimized': Estimator()}, frame, **changes)
    with pytest.raises(ValueError):
        serving.measure({}, frame)
    with pytest.raises(ValueError):
        serving.measure({'baseline': Estimator(), 'optimized': Estimator()}, frame.assign(price=0))
    with pytest.raises(ValueError):
        serving.measure({'baseline': Estimator(), 'optimized': Estimator()}, frame.iloc[:0])
    class Invalid(Estimator):
        def predict(self, frame):
            return np.full((len(frame), 1), np.nan)
    before = get_num_threads()
    with pytest.raises(ValueError):
        serving.measure({'baseline': Invalid(), 'optimized': Estimator()}, frame, threads=1, fm_threads=1)
    class Drifting(Estimator):
        def predict(self, frame):
            self.offset += 1
            return super().predict(frame)
    with pytest.raises(AssertionError):
        serving.measure({'baseline': Drifting(), 'optimized': Estimator()}, frame, threads=1, fm_threads=1)
    class ShapeDrifting(Estimator):
        def predict(self, frame):
            self.offset += 1
            return np.zeros(len(frame)) if self.offset == 1 else 0.
    with pytest.raises(ValueError):
        serving.measure({'baseline': ShapeDrifting(), 'optimized': Estimator()}, frame, threads=1, fm_threads=1)
    assert get_num_threads() == before


def test_cli_replays_trusted_models_and_writes_aggregate_evidence(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f'model-{index}.pkl'
        path.write_bytes(pickle.dumps(Estimator(index)))
        paths.append(path)
    data = tmp_path / 'test.tsv'
    pd.DataFrame({'test_id': [0, 1, 2]}).to_csv(data, sep='\t', index=False)
    output = tmp_path / 'timings.json'
    loaded, original = [], pickle.load
    def observe(stream):
        model = original(stream); loaded.append(model); return model
    monkeypatch.setattr(serving.pickle, 'load', observe)
    args = ['--baseline-model', str(paths[0]), '--optimized-model', str(paths[1]),
            '--test-file', str(data), '--output', str(output), '--rows', '2',
            '--threads', '1', '--fm-threads', '1', '--repeats', '2']
    report = serving.main(args)
    assert report['rows'] == 2 and all(model.closed for model in loaded)
    assert json.loads(output.read_text()) == report
    assert set(report['model_sha256']) == {'baseline', 'optimized'}
    with pytest.raises(ValueError):
        serving.main(args)
    args[args.index('--rows') + 1] = '0'
    with pytest.raises(ValueError):
        serving.main(args)
    monkeypatch.setattr(sys, 'argv', ['mercari_inference', '--help'])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module('benchmarks.mercari_inference', run_name='__main__')
    assert caught.value.code == 0
