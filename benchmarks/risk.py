"""Credit-risk study with train-only model selection and tie-safe KS metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import struct
import subprocess
import time
import urllib.request

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from libtree import C5Classifier
from benchmarks import c5, run
from benchmarks.metrics import ks_statistic

SOURCES = {
    'german_credit': ('https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv',
                      'ac003d1158e44018e3a1416381cebde213f96df542b2191910baec3b2afa7616'),
    'credit_status': ('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/credit_data.csv',
                      'acd2b41f7923a4b7e2264e6b9ae2eff8f3e8f9ba3080fb6967df1cfec9568cac'),
    'taiwan_late_payment': ('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/causaldata/credit_cards.csv',
                            '204c8a9e6eccd99964074adc52065033d303cd1ace0df633c760dac176a81327'),
}


def metrics(y, probability):
    result = c5.metrics(y, probability)
    score = np.asarray(probability)[:, 1]
    result.update(average_precision=average_precision_score(y, score),
                  bad_recall=recall_score(y, np.asarray(probability).argmax(axis=1), zero_division=0))
    return result


def dataset(name):
    if name == 'rare_risk_20k':
        x, y = make_classification(n_samples=20000, n_features=12, n_informative=8,
                                   n_redundant=2, weights=[.97, .03], flip_y=.01, random_state=19)
        return pd.DataFrame(x), y, {'source': 'sklearn.make_classification', 'synthetic': True,
                                    'generation_seed': 19, 'class_weights': [.97, .03], 'flip_y': .01}
    url, checksum = SOURCES[name]
    path = run.CACHE / (name + '.csv')
    if not path.exists():
        raw = urllib.request.urlopen(url, timeout=30).read()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError('risk source checksum mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError('risk cache checksum mismatch')
    frame = pd.read_csv(path)
    if name == 'german_credit':
        if set(frame.credit_risk.unique()) != {0, 1}:
            raise ValueError('unexpected German credit labels')
        y = 1 - frame.pop('credit_risk').to_numpy()  # Source 1=good; target 1=bad.
        definition = 'bad credit risk; source credit_risk=0 (300/1000)'
    elif name == 'credit_status':
        frame = frame.drop(columns='rownames')
        if set(frame.Status.unique()) != {'good', 'bad'}:
            raise ValueError('unexpected credit Status labels')
        y = (frame.pop('Status') == 'bad').to_numpy(dtype='i4')
        definition = 'Status=bad; missing predictors retained; identifier removed'
    else:
        frame = frame.drop(columns='rownames')
        y = frame.pop('LateSept').to_numpy(dtype='i4')
        definition = 'late September 2005 payment; predictors April late flag/bill and age; not the original default target'
    return frame, y, {'url': url, 'sha256': checksum, 'target': definition,
                       'synthetic': False, 'rows': len(frame), 'positive_fraction': float(np.mean(y))}


def encode(train, test):
    """Fit category vocabulary only on the supplied training fold."""
    categories = train.select_dtypes(include=['object', 'string', 'category']).columns
    numeric = train.columns.difference(categories)
    if len(categories):
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        a = encoder.fit_transform(train[categories].fillna('MISSING'))
        b = encoder.transform(test[categories].fillna('MISSING'))
    else:
        a, b = np.empty((len(train), 0)), np.empty((len(test), 0))
    return (np.ascontiguousarray(np.column_stack((train[numeric], a)), dtype='f4'),
            np.ascontiguousarray(np.column_stack((test[numeric], b)), dtype='f4'))


def configurations(rounds=100, depth=4, bins=64):
    base = dict(n_estimators=30, max_depth=depth, max_bins=bins, min_samples_leaf=5,
                learning_rate=1., pruning_alpha=.001, criterion='gain_ratio', boosting='samme')
    long = base | {'n_estimators': rounds}
    shrink = long | {'learning_rate': .3}
    return {
        'single': base | {'boosting': 'none'}, 'samme_30': base, 'samme_long': long,
        'shrinkage': shrink, 'balanced': shrink | {'class_weight': 'balanced'},
        'hellinger': shrink | {'criterion': 'hellinger'},
        'hellinger_balanced': shrink | {'criterion': 'hellinger', 'class_weight': 'balanced'},
        'gini': shrink | {'criterion': 'gini'}, 'entropy': shrink | {'criterion': 'entropy'},
        'less_pruning': shrink | {'pruning_alpha': .0002},
    }


def select_configuration(frame, y, configs, seed, folds=3, margin=.01):
    """Train-only CV; require a KS improvement before changing the baseline.

    All preprocessing is refit per fold. This implements the model-selection
    separation discussed by Cawley & Talbot (2010), JMLR 11:2079-2107.
    """
    start = time.perf_counter()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prepared = [(encode(frame.iloc[a], frame.iloc[b]), y[a], y[b])
                for a, b in splitter.split(frame, y)]
    candidates = []
    for name, parameters in configs.items():
        measurements = []
        for (train, validation), labels, truth in prepared:
            with C5Classifier(**parameters) as model:
                model.fit(train, labels)
                probability = model.predict_proba(validation)
                measurements.append(metrics(truth, probability) | c5.complexity(model, 'c5'))
        candidates.append({'name': name, 'parameters': parameters, 'folds': measurements,
                           'ks_directional': float(np.mean([m['ks_directional'] for m in measurements])),
                           'roc_auc': float(np.mean([m['roc_auc'] for m in measurements])),
                           'rules': float(np.mean([m['rules'] for m in measurements]))})
    best = max(candidates, key=lambda m: (m['ks_directional'], m['roc_auc'], -m['rules']))
    baseline = next(m for m in candidates if m['name'] == 'samme_30')
    chosen = best if best['ks_directional'] >= baseline['ks_directional'] + margin else baseline
    return chosen['name'], {'candidates': candidates, 'chosen': chosen['name'],
                             'search_seconds': time.perf_counter() - start,
                             'minimum_ks_improvement': margin, 'folds': folds}


def native(train, test, labels, parameters, threads, name, args):
    binary, output = run.RUNS / 'risk.bin', run.RUNS / 'risk.pred'
    with binary.open('wb') as stream:
        stream.write(struct.pack('<QQQ', len(train), len(test), train.shape[1]))
        for a in (train, labels, test):
            stream.write(a.astype('<f4').tobytes())
    if parameters is None:
        timings = c5.native(binary, output, name, threads, args)
        values = np.fromfile(output, dtype='<f4')
        return timings, np.column_stack((1 - values, values))
    p = parameters
    command = [c5.ROOT / 'build/c5_benchmark', binary, output,
               p['n_estimators'] if p['boosting'] == 'samme' else 1,
               p['max_depth'], p['max_bins'], p['min_samples_leaf'], p['learning_rate'],
               p['pruning_alpha'], 1e-9, ('gain_ratio', 'entropy', 'gini', 'hellinger').index(p['criterion']),
               threads, int(p.get('class_weight') == 'balanced')]
    timings = json.loads(subprocess.check_output([str(v) for v in command], text=True))
    return timings, np.fromfile(output, dtype='<f8').reshape(-1, 2)


def summarize(records):
    # Reuse the established mean/std/min/max implementation for all metrics.
    summaries = c5.summarize(records)
    for summary in summaries:
        rows = [r for r in records if all(r[k] == summary[k] for k in ('dataset', 'variant', 'interface', 'threads'))]
        for field in ('ks', 'ks_directional', 'average_precision', 'bad_recall'):
            values = [r[field] for r in rows]
            summary[field] = dict(mean=float(np.mean(values)), std=float(np.std(values)), min=min(values), max=max(values))
    return summaries


def main(args):
    configs = configurations(args.rounds, args.depth, args.bins)
    records, sources, selections = [], {}, []
    run.RUNS.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        frame, y, provenance = dataset(name)
        sources[name] = provenance
        for seed in args.seeds:
            a, b = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=seed)
            raw_train, raw_test, labels, truth = frame.iloc[a], frame.iloc[b], y[a], y[b]
            train, test = encode(raw_train, raw_test)
            chosen, selection = select_configuration(raw_train, labels, configs, seed, args.folds)
            selections.append(selection | {'dataset': name, 'seed': seed})
            split_hash = hashlib.sha256(b''.join(v.tobytes() for v in (train, test, labels, truth))).hexdigest()
            jobs = [(v, t) for v in [*configs, 'selected', 'adaboost', 'xgboost', 'lightgbm'] for t in args.threads
                    if t == 1 or v in ('samme_30', 'selected', 'xgboost', 'lightgbm')]
            for repeat in range(args.repeats):
                random.Random(seed + repeat * 101).shuffle(jobs)
                for variant, threads in jobs:
                    params = configs[chosen] if variant == 'selected' else configs.get(variant)
                    model = C5Classifier(**params, n_jobs=threads) if params is not None else c5.build(variant, threads, args)
                    fit_x, predict_x = train, test
                    start = time.perf_counter()
                    imputer = None
                    if variant == 'adaboost':
                        imputer = SimpleImputer(add_indicator=True, keep_empty_features=True)
                        fit_x = imputer.fit_transform(train)
                    model.fit(fit_x, labels)
                    fit_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    if imputer is not None:
                        predict_x = imputer.transform(test)
                    probability = model.predict_proba(predict_x)
                    predict_seconds = time.perf_counter() - start
                    details = dict(dataset=name, seed=seed, repeat=repeat, variant=variant,
                                   threads=threads, interface='python', fit_seconds=fit_seconds,
                                   predict_seconds=predict_seconds, train_rows=len(train), test_rows=len(test),
                                   features=train.shape[1], positive_fraction=float(np.mean(truth)),
                                   split_sha256=split_hash, parameters=params,
                                   train_accuracy=float(np.mean(model.predict(fit_x) == labels)),
                                   **metrics(truth, probability), **c5.complexity(model, 'c5' if params else variant))
                    records.append(details)
                    if variant != 'adaboost':
                        timing, prediction = native(train, test, labels, params, threads, variant, args)
                        if not np.allclose(prediction, probability, atol=2e-6, rtol=1e-5):
                            raise AssertionError(f'{name}/{variant}: native/Python mismatch')
                        records.append(details | timing | metrics(truth, prediction) | {'interface': 'cpp'})
                    if params is not None:
                        model.close()
                    print(f'{name} seed={seed} {variant} t={threads} KS={details["ks"]:.4f} AUC={details["roc_auc"]:.4f}', flush=True)
        files = sorted([*c5.ROOT.glob('src/*.cc'), *c5.ROOT.glob('src/*.h'),
                        *c5.ROOT.glob('include/libtree/*.h'), *c5.ROOT.glob('python/libtree/*.py'),
                        Path(__file__), c5.ROOT / 'benchmarks/metrics.py', c5.ROOT / 'benchmarks/c5.py', c5.ROOT / 'benchmarks/c5_native.cc', c5.ROOT / 'benchmarks/native.cc'])
        payload = {'metadata': {'parameters': vars(args) | {'output': str(args.output)},
                                'code_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest(),
                                'python': platform.python_version(), 'cpu_quota': run.cpu_quota(),
                                'xgboost': c5.xgb.__version__, 'lightgbm': c5.lgb.__version__,
                                'sklearn': c5.sklearn.__version__,
                                'selection': 'outer stratified 75/25; train-only stratified CV; directional KS; margin .01',
                                'aggregation': 'arithmetic mean, population std, min/max; retain raw repeats'},
                   'sources': sources, 'selections': selections, 'results': records, 'summary': summarize(records)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=[*SOURCES, 'rare_risk_20k'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2024, 7])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--output', type=Path, default=c5.ROOT / 'benchmarks/risk-results.json')
    main(parser.parse_args())
