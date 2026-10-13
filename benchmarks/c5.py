"""C5 quality/speed/complexity study. All reported summaries use arithmetic means."""
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

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

from libtree import C5Classifier
from benchmarks import run
from benchmarks.metrics import ks_statistic

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('c5', 'tree', 'adaboost', 'xgboost', 'lightgbm', 'c5_single',
            'c5_unpruned', 'c5_entropy', 'c5_gini')


def dataset(name):
    if name in ('breast_cancer', 'wine'):
        loader = load_breast_cancer if name == 'breast_cancer' else load_wine
        x, y = loader(return_X_y=True)
        return x, y, {'source': f'sklearn.datasets.{loader.__name__}'}
    if name.startswith('synthetic_'):
        size = int(name.split('_')[1].removesuffix('k')) * 1000
        x, y = make_classification(n_samples=size, n_features=20, n_informative=12,
                                   n_redundant=4, flip_y=.03, random_state=13)
        return x, y, {'source': 'sklearn.make_classification', 'generation_seed': 13,
                      'label_noise': .03}
    for entry in run.kaggle_datasets():
        if entry[0] == name:
            return entry[1], entry[2], entry[4]
    raise ValueError(f'unknown dataset: {name}')


def prepare(x, y, seed):
    # Stratify all classification tasks, including multiclass.
    y = np.unique(y, return_inverse=True)[1]
    train, test, labels, truth = run.prepare(x, y, True, seed)
    return train, test, labels.astype('i4'), truth.astype('i4')


def metrics(y, probability):
    probability = np.asarray(probability, dtype='f8')
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-6):
        raise ValueError('invalid probabilities')
    probability = probability / probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1)
    result = {'accuracy': accuracy_score(y, prediction),
            'balanced_accuracy': balanced_accuracy_score(y, prediction),
            'macro_f1': f1_score(y, prediction, average='macro', zero_division=0),
            'log_loss': log_loss(y, probability, labels=np.arange(probability.shape[1])),
            'roc_auc': roc_auc_score(y, probability[:, 1]) if probability.shape[1] == 2
                       else roc_auc_score(y, probability, multi_class='ovr', average='macro')}

    if probability.shape[1] == 2:
        result.update(ks=ks_statistic(y, probability[:, 1]),
                      ks_directional=ks_statistic(y, probability[:, 1], directional=True))
    return result


def build(variant, threads, args):
    if variant.startswith('c5'):
        criterion = variant[3:] if variant in ('c5_entropy', 'c5_gini') else 'gain_ratio'
        return C5Classifier(n_estimators=1 if variant == 'c5_single' else args.rounds,
                             max_depth=args.depth, max_bins=args.bins,
                             min_samples_leaf=5, learning_rate=1,
                             pruning_alpha=0 if variant == 'c5_unpruned' else .001,
                             criterion=criterion, n_jobs=threads)
    if variant in ('tree', 'adaboost'):
        base = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=5,
                                       criterion='entropy', ccp_alpha=.001, random_state=42)
        return base if variant == 'tree' else AdaBoostClassifier(
            estimator=base, n_estimators=args.rounds, learning_rate=1, random_state=42)
    if variant == 'xgboost':
        return xgb.XGBClassifier(n_estimators=args.rounds, max_depth=args.depth,
                                 max_bin=args.bins, learning_rate=.1, reg_lambda=1,
                                 tree_method='hist', n_jobs=threads, random_state=42)
    return lgb.LGBMClassifier(n_estimators=args.rounds, max_depth=args.depth,
                              num_leaves=1 << args.depth, max_bin=args.bins,
                              learning_rate=.1, reg_lambda=1, min_child_samples=5,
                              min_data_in_bin=1, feature_pre_filter=False,
                              force_col_wise=True, deterministic=True, n_jobs=threads,
                              random_state=42, data_random_seed=42, verbosity=-1)


def complexity(model, variant):
    if variant.startswith('c5'):
        rules = json.loads(model.dump_rules('json'))
        return {'rules': len(rules['rules']), 'conditions': sum(len(r['conditions']) for r in rules['rules']),
                'estimators': rules['estimators']}
    if variant in ('tree', 'adaboost'):
        trees = [model] if variant == 'tree' else model.estimators_
        return {'rules': int(sum(t.get_n_leaves() for t in trees)),
                'estimators': len(trees)}
    if variant == 'xgboost':
        frame = model.get_booster().trees_to_dataframe()
        return {'rules': int((frame['Feature'] == 'Leaf').sum()),
                'estimators': int(frame['Tree'].nunique())}
    trees = model.booster_.dump_model()['tree_info']
    return {'rules': sum(t['num_leaves'] for t in trees), 'estimators': len(trees)}


def native(binary, output, variant, threads, args):
    if variant == 'c5':
        command = [ROOT / 'build/c5_benchmark', binary, output, args.rounds, args.depth,
                   args.bins, 5, 1, .001, 1e-9, 0, threads]
    else:
        library = Path(xgb.__file__).parent / 'lib/libxgboost.so' if variant == 'xgboost' else (
            Path(lgb.__file__).parent / 'lib/lib_lightgbm.so')
        command = [ROOT / 'build/native_benchmark', binary, output, 1, variant,
                   library, args.rounds, args.depth, args.bins, 5, .1, 1, 0, threads]
    return json.loads(subprocess.check_output([str(v) for v in command], text=True))


def summarize(records):
    groups = {}
    for r in records:
        key = (r['dataset'], r['variant'], r['interface'], r['threads'])
        groups.setdefault(key, []).append(r)
    summaries = []
    for (name, variant, interface, threads), rows in groups.items():
        item = dict(dataset=name, variant=variant, interface=interface, threads=threads,
                    measurements=len(rows))
        for field in ('fit_seconds', 'predict_seconds', 'accuracy', 'balanced_accuracy',
                      'macro_f1', 'roc_auc', 'log_loss', 'ks', 'ks_directional', 'train_accuracy', 'rules', 'conditions'):
            values = [float(r[field]) for r in rows if field in r]
            if values:
                item[field] = {'mean': float(np.mean(values)), 'std': float(np.std(values)),
                               'min': min(values), 'max': max(values)}
        summaries.append(item)
    return summaries


def main(args):
    results, sources = [], {}
    run.RUNS.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        x, y, provenance = dataset(name)
        sources[name] = provenance
        for seed in args.seeds:
            train, test, labels, truth = prepare(x, y, seed)
            digest = hashlib.sha256(b''.join(a.tobytes() for a in (train, test, labels, truth))).hexdigest()
            binary, output = run.RUNS / 'c5.bin', run.RUNS / 'c5.pred'
            with binary.open('wb') as stream:
                stream.write(struct.pack('<QQQ', len(train), len(test), train.shape[1]))
                for array in (train, labels, test):
                    stream.write(array.astype('<f4').tobytes())
            # sklearn AdaBoost rejects NaN. Fit medians on TRAIN only, append
            # missingness indicators; its preprocessing time is included in fit.
            for threads in args.threads:
                for repeat in range(args.repeats):
                    order = list(args.variants)
                    random.Random(seed * 1009 + threads * 17 + repeat).shuffle(order)
                    for variant in order:
                        if threads != 1 and variant not in ('c5', 'xgboost', 'lightgbm'):
                            continue  # sklearn SAMME is serial; ablations use one thread.
                        model = build(variant, threads, args)
                        start = time.perf_counter()
                        fit_x, predict_x = train, test
                        imputer = None
                        if variant in ('tree', 'adaboost'):
                            imputer = SimpleImputer(add_indicator=True, keep_empty_features=True)
                            fit_x = imputer.fit_transform(train)
                        model.fit(fit_x, labels)
                        fit_seconds = time.perf_counter() - start
                        start = time.perf_counter()
                        if imputer is not None:
                            predict_x = imputer.transform(test)
                        probability = model.predict_proba(predict_x)
                        predict_seconds = time.perf_counter() - start
                        trained = model.predict_proba(fit_x)
                        details = dict(dataset=name, seed=seed, repeat=repeat, threads=threads,
                                       variant=variant, interface='python', train_rows=len(train),
                                       test_rows=len(test), features=train.shape[1], classes=probability.shape[1],
                                       split_sha256=digest, fit_seconds=fit_seconds,
                                       predict_seconds=predict_seconds, **metrics(truth, probability),
                                       train_accuracy=accuracy_score(labels, trained.argmax(axis=1)),
                                       **complexity(model, variant))
                        results.append(details)
                        if variant == 'c5' or (variant in ('xgboost', 'lightgbm') and probability.shape[1] == 2):
                            timings = native(binary, output, variant, threads, args)
                            values = np.fromfile(output, dtype='<f8' if variant == 'c5' else '<f4')
                            cpp = values.reshape(-1, probability.shape[1]) if variant == 'c5' else np.column_stack((1 - values, values))
                            if not np.allclose(cpp, probability, atol=2e-6, rtol=1e-5):
                                raise AssertionError(f'{name}/{variant}: C++/Python predictions differ')
                            results.append({**details, **timings, **metrics(truth, cpp), 'interface': 'cpp'})
                        if variant.startswith('c5'):
                            model.close()
                        print(f'{name} seed={seed} threads={threads} repeat={repeat} {variant}: '
                              f'fit={fit_seconds:.4f}s acc={details["accuracy"]:.4f} rules={details["rules"]}', flush=True)
        # Save after each dataset so a completed dataset remains reviewable.
        payload = {'metadata': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                                'numpy': np.__version__, 'xgboost': xgb.__version__,
                                'lightgbm': lgb.__version__, 'cpu_quota': run.cpu_quota(),
                                'affinity_cpus': len(os.sched_getaffinity(0)),
                                'parameters': vars(args) | {'output': str(args.output)},
                                'code_sha256': hashlib.sha256(b''.join(
                                    str(p.relative_to(ROOT)).encode() + b'\0' + p.read_bytes()
                                    for p in sorted([*ROOT.glob('src/*.cc'), *ROOT.glob('src/*.h'),
                                                     *ROOT.glob('include/libtree/*.h'),
                                                     *ROOT.glob('python/libtree/*.py'), Path(__file__),
                                                     ROOT / 'benchmarks/c5_native.cc',
                                                     ROOT / 'benchmarks/native.cc']))).hexdigest(),
                                'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                                'cpu_model': next((line.split(':', 1)[1].strip() for line in
                                    Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown'),
                                'aggregation': 'arithmetic mean; population std; min/max; raw repeats retained',
                                'split': 'stratified 75/25; fixed seeds; no test-set tuning'},
                   'sources': sources, 'results': results, 'summary': summarize(results)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['breast_cancer', 'wine', 'pima',
                        'bank_marketing', 'synthetic_20k', 'synthetic_100k'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2024, 7])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/c5-results.json')
    main(parser.parse_args())
