"""Compare complete Mercari pipelines using trusted, locally saved models.

Times include feature preparation and prediction, exclude loading and I/O,
and use one warm-up followed by interleaved repetitions on unlabeled listings.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import time

from numba import get_num_threads, set_num_threads
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmarks.mercari import sha256, timing


class _BatchedSparseExpert:
    def __init__(self, model, batch_size):
        self.model, self.batch_size = model, batch_size

    def predict(self, x):
        return self.model.predict(x, batch_size=self.batch_size)


def configure_sparse_batches(entry, batch_size):
    """Configure sparse adapters in place using their existing trained weights."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('sparse batch size must be a positive integer')
    from benchmarks.mercari_advanced import SparseText
    neural = getattr(getattr(entry, 'experts', None), 'neural', None)
    targets = [member for member in getattr(neural, 'members', [neural]) if isinstance(member, SparseText)]
    if not targets:
        raise ValueError('model has no sparse text adapters')
    for member in targets:
        model = member.model.model if isinstance(member.model, _BatchedSparseExpert) else member.model
        member.model = _BatchedSparseExpert(model, batch_size)
    return len(targets)


def _prediction(value, rows):
    result = np.asarray(value)
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ValueError('each model must return one finite log-price per listing')
    return result


def measure(models, frame, repeats=3, threads=4, fm_threads=2):
    if (set(models) != {'baseline', 'optimized'} or not len(frame)
            or 'price' in frame or min(repeats, threads, fm_threads) < 1):
        raise ValueError('two named models, unlabeled rows and positive counts required')
    names = ['baseline', 'optimized']
    samples = {name: [] for name in names}
    previous = get_num_threads()
    try:
        set_num_threads(fm_threads)
        with threadpool_limits(threads):
            references = {name: _prediction(models[name].predict(frame), len(frame)) for name in names}
            for repeat in range(repeats):
                for name in names if repeat % 2 == 0 else reversed(names):
                    stamp = time.perf_counter()
                    prediction = models[name].predict(frame)
                    seconds = time.perf_counter() - stamp
                    prediction = _prediction(prediction, len(frame))
                    np.testing.assert_allclose(prediction, references[name], atol=3e-6, rtol=1e-6)
                    samples[name].append(seconds)
    finally:
        set_num_threads(previous)
    timings = {name: timing(values) for name, values in samples.items()}
    return dict(rows=len(frame), warmup_runs_per_model=1, threads=threads,
                fm_predict_threads=fm_threads, interleaved_order=True, timings=timings,
                optimized_to_baseline_mean_cost_ratio=timings['optimized']['mean'] / timings['baseline']['mean'],
                timing_scope='Feature preparation and prediction; loading, input I/O and output validation excluded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-model', type=Path, required=True)
    parser.add_argument('--optimized-model', type=Path, required=True)
    parser.add_argument('--test-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--fm-threads', type=int, default=2)
    parser.add_argument('--sparse-batch-size', type=int, help='override optimized model sparse inference batches')
    args = parser.parse_args(argv)
    if (min(args.rows, args.repeats, args.threads, args.fm_threads) < 1
            or (args.sparse_batch_size is not None and args.sparse_batch_size < 1)):
        raise ValueError('row, repeat and thread counts must be positive')
    if args.output.exists():
        raise ValueError('output must be a new file')
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(args.threads)
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    paths = dict(baseline=args.baseline_model, optimized=args.optimized_model)
    with ExitStack() as resources:
        models = {}
        for name, path in paths.items():
            with path.open('rb') as stream:
                models[name] = pickle.load(stream)
            resources.callback(models[name].close)
        overrides = {}
        if args.sparse_batch_size is not None:
            overrides = dict(sparse_batch_size=args.sparse_batch_size,
                             sparse_heads=configure_sparse_batches(models['optimized'], args.sparse_batch_size))
        frame = pd.read_csv(args.test_file, sep='\t', nrows=args.rows)
        report = measure(models, frame, args.repeats, args.threads, args.fm_threads)
        report['inference_overrides'] = overrides
    report.update(created_utc=datetime.now(timezone.utc).isoformat(),
                  model_sha256={name: sha256(path) for name, path in paths.items()},
                  input_sha256=sha256(args.test_file))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    main()
