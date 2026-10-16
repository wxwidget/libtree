"""Paired C5 before/after study, with XGBoost/LightGBM and exact-output guards.

Run the same Release benchmark driver against both C5 static libraries. Python
workers import each checkout's own wrapper and native library in isolation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = ('fit_seconds', 'predict_seconds', 'predict_label_seconds', 'accuracy',
          'balanced_accuracy', 'roc_auc', 'ks', 'ks_directional', 'average_precision',
          'train_accuracy', 'rules', 'dump_seconds', 'explain_seconds', 'dump_bytes')
FINGERPRINTS = ('probability_sha256', 'train_probability_sha256', 'rules_sha256')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def validate(records):
    pairs = defaultdict(dict)
    stable = {}
    for row in records:
        if row['variant'] not in ('before', 'after'):
            continue
        key = tuple(row[k] for k in ('dataset', 'seed', 'threads', 'interface', 'repeat'))
        if row['variant'] in pairs[key]:
            raise ValueError('duplicate before/after measurement')
        pairs[key][row['variant']] = row
        signature = tuple(row[k] for k in FINGERPRINTS)
        shared = (row['dataset'], row['seed'], row['interface'])
        if stable.setdefault(shared, signature) != signature:
            raise ValueError('C5 output changed across versions/repeats/thread counts')
    if not pairs or any(set(pair) != {'before', 'after'} for pair in pairs.values()):
        raise ValueError('incomplete before/after pairs')
    for pair in pairs.values():
        if any(pair['before'][key] != pair['after'][key] for key in FINGERPRINTS):
            raise ValueError('C5 probability or rule regression')


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in ('dataset', 'variant', 'interface', 'threads'))].append(row)
    output = []
    for key, rows in groups.items():
        group = dict(zip(('dataset', 'variant', 'interface', 'threads'), key))
        group['measurements'] = len(rows)
        for field in FIELDS:
            values = [row[field] for row in rows if field in row]
            if values:
                group[field] = dict(mean=float(np.mean(values)), std=float(np.std(values)),
                                    min=min(values), max=max(values))
        output.append(group)
    return output


def worker(args):
    # Set before importing libtree; a newer wrapper cannot load an older ABI.
    sys.path.insert(0, str(args.checkout / 'python'))
    os.environ['LIBTREE_LIBRARY'] = str(args.checkout / 'build/libtree.so')
    from benchmarks import c5, run
    from sklearn.metrics import average_precision_score

    arrays = np.load(args.data)
    train, test, labels, truth = (arrays[k] for k in ('train', 'test', 'labels', 'truth'))
    variant = 'c5' if args.engine in ('before', 'after') else args.engine
    binary, prediction = args.data.with_suffix('.bin'), args.data.with_suffix('.prediction')
    records = []
    for repeat in range(args.repeats):
        model = c5.build(variant, args.jobs, args)
        start = time.perf_counter()
        model.fit(train, labels)
        fit = time.perf_counter() - start
        start = time.perf_counter()
        probability = model.predict_proba(test)
        proba_time = time.perf_counter() - start
        start = time.perf_counter()
        prediction_labels = model.predict(test)
        label_time = time.perf_counter() - start
        np.testing.assert_array_equal(prediction_labels, probability.argmax(axis=1))
        trained = model.predict_proba(train)
        details = dict(repeat=repeat, fit_seconds=fit, predict_seconds=proba_time,
                       predict_label_seconds=label_time, **c5.metrics(truth, probability),
                       train_accuracy=float(np.mean(trained.argmax(axis=1) == labels)),
                       **c5.complexity(model, variant))
        if probability.shape[1] == 2:
            details['average_precision'] = float(average_precision_score(truth, probability[:, 1]))
        if variant == 'c5':
            rules = json.loads(model.dump_rules('json'))
            rules.pop('class_labels')
            details.update(probability_sha256=digest(probability.tobytes()),
                           train_probability_sha256=digest(trained.tobytes()),
                           rules_sha256=digest(json.dumps(rules, sort_keys=True).encode()))
            if args.engine == 'after':
                start = time.perf_counter()
                tree_dump = model.dump_model('json')
                details['dump_seconds'] = time.perf_counter() - start
                details['dump_bytes'] = len(tree_dump.encode())
                start = time.perf_counter()
                explanation = model.explain(test[0])
                details['explain_seconds'] = time.perf_counter() - start
                assert explanation['prediction'] == int(prediction_labels[0])
        records.append(details | {'interface': 'python'})
        if variant == 'c5' or probability.shape[1] == 2:
            if variant == 'c5':
                command = [args.native, binary, prediction, args.rounds, args.depth,
                           args.bins, 5, 1, .001, 1e-9, 0, args.jobs]
                timings = json.loads(subprocess.check_output([str(v) for v in command], text=True))
                cpp = np.fromfile(prediction, dtype='<f8').reshape(probability.shape)
                native_rules = json.loads(Path(str(prediction) + '.rules.json').read_text())
                timings.update(probability_sha256=digest(cpp.tobytes()),
                               train_probability_sha256=digest(Path(str(prediction) + '.train').read_bytes()),
                               rules_sha256=digest(json.dumps(native_rules, sort_keys=True).encode()))
            else:
                timings = c5.native(binary, prediction, variant, args.jobs, args)
                values = np.fromfile(prediction, dtype='<f4')
                cpp = np.column_stack((1 - values, values))
            np.testing.assert_allclose(cpp, probability, rtol=1e-5, atol=2e-6)
            native_details = {k: v for k, v in details.items()
                              if k not in ('predict_label_seconds', 'dump_seconds', 'explain_seconds', 'dump_bytes')}
            native_metrics = c5.metrics(truth, cpp)
            if cpp.shape[1] == 2:
                native_metrics['average_precision'] = float(average_precision_score(truth, cpp[:, 1]))
            records.append(native_details | timings | native_metrics | {'interface': 'cpp'})
        if variant == 'c5':
            model.close()
    print(json.dumps(records, allow_nan=False))


def main(args):
    from benchmarks import c5, risk, run
    baseline = args.baseline.resolve()
    for checkout in (baseline, ROOT):
        if not (checkout / 'build/libtree.a').exists():
            raise ValueError(f'Build Release libraries first: cmake -S {checkout} -B {checkout}/build '
                             '-DCMAKE_BUILD_TYPE=Release && cmake --build ' + str(checkout / 'build'))
    if args.repeats < 1 or not args.seeds or set(args.threads) - {1, 4}:
        raise ValueError('positive repeats, seeds and thread budgets 1/4 required')
    run.RUNS.mkdir(parents=True, exist_ok=True)
    baseline_driver = run.RUNS / 'c5-before-driver'
    subprocess.run(['c++', '-O3', '-DNDEBUG', '-std=c++17', '-I', str(baseline / 'include'),
                    str(ROOT / 'benchmarks/c5_native.cc'), str(baseline / 'build/libtree.a'),
                    '-pthread', '-o', str(baseline_driver)], check=True)
    results, sources = [], {}
    files = sorted([*ROOT.glob('src/*.cc'), *ROOT.glob('src/*.h'),
                    *ROOT.glob('include/libtree/*.h'), *ROOT.glob('python/libtree/*.py'),
                    Path(__file__), ROOT / 'benchmarks/c5.py', ROOT / 'benchmarks/risk.py',
                    ROOT / 'benchmarks/run.py', ROOT / 'benchmarks/metrics.py',
                    ROOT / 'benchmarks/c5_native.cc', ROOT / 'benchmarks/native.cc'])
    metadata = dict(baseline_revision=subprocess.check_output(['git', '-C', str(baseline), 'rev-parse', 'HEAD'], text=True).strip(),
                    code_sha256=digest(b''.join(str(p.relative_to(ROOT)).encode() + b'\0' + p.read_bytes() for p in files)),
                    parameters={k: v for k, v in vars(args).items() if k not in ('baseline', 'output', 'data', 'native', 'checkout')},
                    cpu_quota=run.cpu_quota(), affinity_cpus=len(os.sched_getaffinity(0)),
                    python=sys.version.split()[0], numpy=np.__version__, sklearn=c5.sklearn.__version__,
                    xgboost=c5.xgb.__version__, lightgbm=c5.lgb.__version__,
                    compiler=subprocess.check_output(['c++', '--version'], text=True).splitlines()[0],
                    cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                   if line.startswith('model name')),
                    driver_sha256={'before': digest(baseline_driver.read_bytes()),
                                   'after': digest((ROOT / 'build/c5_benchmark').read_bytes())},
                    aggregation='arithmetic mean; population std; min/max; raw repeats retained',
                    protocol='stratified 75/25; train-only encoding; randomized engine blocks; fresh models; '
                             'fit, probability, then label timing; no discarded warmup; startup/I/O excluded')
    for name in args.datasets:
        if name in risk.SOURCES:
            x, y, source = risk.dataset(name)
        else:
            x, y, source = c5.dataset('synthetic_20k' if name == 'missing_20k' else name)
            if name == 'missing_20k':
                x = x.copy()
                x[np.random.default_rng(19).random(x.shape) < .1] = np.nan
                source = source | {'missing_fraction': .1, 'missing_seed': 19, 'synthetic': True}
        sources[name] = source
        for seed in args.seeds:
            train, test, labels, truth = c5.prepare(x, y, seed)
            data_path = run.RUNS / 'c5-performance-input.npz'
            np.savez(data_path, train=train, test=test, labels=labels, truth=truth)
            with data_path.with_suffix('.bin').open('wb') as stream:
                stream.write(struct.pack('<QQQ', len(train), len(test), train.shape[1]))
                for array in (train, labels, test):
                    stream.write(array.astype('<f4').tobytes())
            split_digest = digest(b''.join(a.tobytes() for a in (train, test, labels, truth)))
            for threads in args.threads:
                engines = ['before', 'after', 'xgboost', 'lightgbm']
                random.Random(seed * 1009 + threads * 17).shuffle(engines)
                for engine in engines:
                    checkout = baseline if engine == 'before' else ROOT
                    native = baseline_driver if engine == 'before' else ROOT / 'build/c5_benchmark'
                    command = [sys.executable, Path(__file__), '--worker', '--engine', engine,
                               '--checkout', checkout, '--native', native, '--data', data_path,
                               '--jobs', threads, '--rounds', args.rounds, '--depth', args.depth,
                               '--bins', args.bins, '--repeats', args.repeats]
                    rows = json.loads(subprocess.check_output([str(v) for v in command], text=True))
                    results.extend(row | dict(dataset=name, seed=seed, threads=threads, variant=engine,
                                              train_rows=len(train), test_rows=len(test), features=train.shape[1],
                                              split_sha256=split_digest) for row in rows)
                    print(f'{name} seed={seed} threads={threads} {engine}: '
                          f'{len(rows)} measurements', flush=True)
        validate(results)
        payload = dict(metadata=metadata, sources=sources, results=results, summary=summarize(results))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--datasets', nargs='+', default=['wine', 'german_credit', 'credit_status',
                        'taiwan_late_payment', 'synthetic_20k', 'synthetic_100k', 'missing_20k'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2024, 7])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/c5-performance-results.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--engine', choices=('before', 'after', 'xgboost', 'lightgbm'), help=argparse.SUPPRESS)
    parser.add_argument('--checkout', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--native', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--data', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--jobs', type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if not arguments.worker and arguments.baseline is None:
        parser.error('--baseline must name a built baseline checkout')
    worker(arguments) if arguments.worker else main(arguments)
