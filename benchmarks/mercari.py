"""Leakage-safe Mercari price prediction with native LibTree and sparse text.

Run ``python -m benchmarks.mercari --help`` from the repository root.
The runner generates predictions; it never submits to Kaggle automatically.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

import libtree
from libtree import GBDTRegressor

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = 'mercari-price-suggestion-challenge'
COLUMNS = ('name', 'item_condition_id', 'category_name', 'brand_name', 'shipping', 'item_description')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def rmsle(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if (y.ndim != 1 or not len(y) or y.shape != prediction.shape
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()
            or (y < 0).any() or (prediction < 0).any()):
        raise ValueError('RMSLE requires matching nonempty vectors of finite nonnegative prices')
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(prediction)) ** 2)))


def prices(log_prediction):
    with np.errstate(over='raise', invalid='raise'):
        return np.expm1(np.maximum(log_prediction, 0.))


def prepare(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'missing listing columns: {sorted(missing)}')
    result = frame.copy()
    for name in COLUMNS:
        result[name] = result[name].fillna('<missing>').astype(str)
    return result


def listing_groups(frame):
    """Identical available listing features share a split, regardless of price/ID."""
    return pd.util.hash_pandas_object(prepare(frame)[list(COLUMNS)], index=False).to_numpy()


def split(frame, seed):
    groups = listing_groups(frame)
    rest, h = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(frame, groups=groups))
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed + 1).split(frame.iloc[rest], groups=groups[rest]))
    return frame.iloc[rest[a]], frame.iloc[rest[b]], frame.iloc[h]


def keys(frame):
    p = prepare(frame)
    levels = p.category_name.str.split('/', expand=True).reindex(columns=range(3), fill_value='<missing>')
    return pd.DataFrame({'brand': p.brand_name, 'category': p.category_name,
                         'category_0': levels[0].fillna('<missing>'),
                         'category_1': levels[1].fillna('<missing>'),
                         'category_2': levels[2].fillna('<missing>')}, index=frame.index)


class DenseFeatures:
    """Smoothed categorical log-price means; use cross_fit for training rows.

    Target-mean smoothing follows Micci-Barreca (2001), SIGKDD Explorations 3(1),
    doi:10.1145/507533.507538. Inner folds exclude their own labels and prior.
    """

    def fit(self, frame, y):
        y = np.asarray(y, dtype=float)
        if y.shape != (len(frame),) or not np.isfinite(y).all() or not len(y):
            raise ValueError('finite log-price labels must match fitting rows')
        self.prior_ = float(np.mean(y))
        self.means_, self.frequencies_ = {}, {}
        for name, column in keys(frame).items():
            work = pd.DataFrame({'key': column.to_numpy(), 'target': y})
            stats = work.groupby('key').target.agg(['sum', 'count'])
            self.means_[name] = ((stats['sum'] + 20 * self.prior_) / (stats['count'] + 20)).to_dict()
            self.frequencies_[name] = (stats['count'] / len(frame)).to_dict()
        self.feature_names_ = ['condition', 'shipping', 'name_length', 'description_length',
                               'name_words', 'description_words', 'missing_brand', 'missing_description']
        self.feature_names_ += [f'{name}_{suffix}' for name in self.means_ for suffix in ('mean', 'frequency')]
        return self

    def transform(self, frame):
        p = prepare(frame)
        numeric = pd.DataFrame({'condition': pd.to_numeric(p.item_condition_id, errors='coerce').fillna(0),
            'shipping': pd.to_numeric(p.shipping, errors='coerce').fillna(0),
            'name_length': np.log1p(p.name.str.len()),
            'description_length': np.log1p(p.item_description.str.len()),
            'name_words': np.log1p(p.name.str.split().str.len()),
            'description_words': np.log1p(p.item_description.str.split().str.len()),
            'missing_brand': p.brand_name.eq('<missing>'),
            'missing_description': p.item_description.eq('<missing>')}, index=frame.index)
        for name, column in keys(frame).items():
            numeric[f'{name}_mean'] = column.map(self.means_[name]).fillna(self.prior_)
            numeric[f'{name}_frequency'] = column.map(self.frequencies_[name]).fillna(0.)
        return np.ascontiguousarray(numeric.to_numpy(), dtype=np.float32)


def make_text(max_features):
    # Train-only vocabularies; L2 Ridge on log1p(price), not raw-price error.
    features = ColumnTransformer([
        ('name', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                sublinear_tf=True, dtype=np.float32), 'name'),
        ('description', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features,
                                       sublinear_tf=True, dtype=np.float32), 'item_description'),
        ('categories', OneHotEncoder(handle_unknown='ignore', dtype=np.float32),
         ['category_name', 'brand_name', 'item_condition_id', 'shipping'])],
        transformer_weights={'name': 2., 'description': 1., 'categories': 1.}, sparse_threshold=1.)
    return Pipeline([('features', features),
                     ('ridge', Ridge(alpha=3., solver='lsqr', tol=1e-3, max_iter=100))])


def cross_fit(frame, y, folds, max_features):
    """Out-of-fold Ridge predictions and target encodings for honest stacking.

    Wolpert (1992), Stacked Generalization, doi:10.1016/S0893-6080(05)80023-1.
    Every inner fold refits its text vocabulary, label means and global prior.
    """
    groups = listing_groups(frame)
    prediction = np.empty(len(frame), dtype=np.float64)
    dense = None
    audit = []
    for number, (a, b) in enumerate(GroupKFold(n_splits=folds).split(frame, groups=groups)):
        started = time.perf_counter()
        model = make_text(max_features).fit(prepare(frame.iloc[a]), y[a])
        prediction[b] = model.predict(prepare(frame.iloc[b]))
        encoder = DenseFeatures().fit(frame.iloc[a], y[a])
        encoded = encoder.transform(frame.iloc[b])
        if dense is None:
            dense = np.empty((len(frame), encoded.shape[1]), dtype=np.float32)
        dense[b] = encoded
        audit.append(dict(fold=number, train_rows=len(a), validation_rows=len(b),
                          validation_positions=b.tolist(), seconds=time.perf_counter() - started))
        print(f'OOF fold {number + 1}/{folds}: {len(a)} fit, {len(b)} predict', flush=True)
    return prediction, dense, audit


class Entry:
    """Log-price predictor: structured LibTree, or Ridge + LibTree correction."""

    def __init__(self, text, encoder, native, engine, strength=1.):
        self.text, self.encoder, self.native = text, encoder, native
        self.engine, self.strength = engine, strength

    def predict(self, frame):
        dense = self.encoder.transform(frame)
        if self.engine == 'libtree':
            return self.native.predict(dense)
        base = self.text.predict(prepare(frame))
        x = np.ascontiguousarray(np.column_stack([dense, base]), dtype=np.float32)
        return base + self.strength * self.native.predict(x)

    def close(self):
        self.native.close()


def timing(samples):
    return dict(samples=samples, mean=float(np.mean(samples)), min=float(min(samples)),
                max=float(max(samples)), std=float(np.std(samples)), repeats=len(samples))


def train_native(x, y, args):
    return GBDTRegressor(n_estimators=args.trees, max_depth=args.depth, max_bins=128,
        min_samples_leaf=20, learning_rate=.06, reg_lambda=10., n_jobs=args.threads).fit(x, y)


def components(frame, y, args):
    oof, dense, audit = cross_fit(frame, y, args.folds, args.max_features)
    started = time.perf_counter()
    text = make_text(args.max_features).fit(prepare(frame), y)
    encoder = DenseFeatures().fit(frame, y)
    return text, encoder, oof, dense, audit, time.perf_counter() - started


def write_submission(entry, test_path, sample_path, output, chunk_size=100000):
    sample = pd.read_csv(sample_path, dtype={'test_id': str})
    if list(sample.columns) != ['test_id', 'price'] or sample.test_id.isna().any() or sample.test_id.duplicated().any():
        raise ValueError('sample submission requires unique test_id values and test_id,price columns')
    seen = 0
    for frame in pd.read_csv(test_path, sep='\t', dtype={'test_id': str}, chunksize=chunk_size):
        if 'price' in frame.columns:
            raise ValueError('test prices must not be supplied')
        expected = sample.test_id.iloc[seen:seen + len(frame)].to_numpy()
        if not np.array_equal(expected, frame.test_id.to_numpy()):
            raise ValueError('test/sample ID order mismatch')
        prediction = prices(entry.predict(frame))
        if not np.isfinite(prediction).all():
            raise ValueError('nonfinite submission predictions')
        pd.DataFrame({'test_id': frame.test_id, 'price': prediction}).to_csv(
            output, index=False, mode='w' if seen == 0 else 'a', header=seen == 0)
        seen += len(frame)
    if seen != len(sample) or seen == 0:
        raise ValueError('test/sample row count mismatch')
    return dict(file=Path(output).name, rows=seen, sha256=sha256(output), status='generated_not_submitted')


def audit_split(frame):
    return dict(rows=len(frame), zero_prices=int((frame.price == 0).sum()),
                id_sha256=hashlib.sha256(frame.train_id.to_numpy().tobytes()).hexdigest(),
                groups=int(len(np.unique(listing_groups(frame)))))


def run(args):
    if min(args.trees, args.depth, args.threads, args.repeats, args.max_features) < 1 or args.folds < 2:
        raise ValueError('positive model settings and at least two folds required')
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty to preserve the experiment journal')
    started = time.perf_counter()
    data = Path(args.data_dir)
    frame = pd.read_csv(data / 'train.tsv', sep='\t')
    if 'price' not in frame or 'train_id' not in frame or frame.train_id.duplicated().any():
        raise ValueError('training data require price and unique train_id')
    rmsle(frame.price, frame.price)
    source_rows = len(frame)
    if args.train_rows is not None:
        frame = frame.sample(n=min(args.train_rows, len(frame)), random_state=args.seed)
    fit, tune, holdout = split(frame, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    report = dict(competition=COMPETITION, created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=commit.stdout.strip(), script_sha256=sha256(__file__),
        python=platform.python_version(), libtree=libtree.__version__,
        native_library_sha256=sha256(libtree._library()._name), source_train_rows=source_rows,
        used_train_rows=len(frame), parameters=vars(args).copy(),
        inputs={name: sha256(data / name) for name in ['train.tsv', args.test_file, args.sample_file]},
        splits={name: audit_split(part) for name, part in [('fit', fit), ('tune', tune), ('holdout', holdout)]},
        native_timings=[], tuning=[])
    report['parameters'] = {k: str(v) if isinstance(v, Path) else v for k, v in report['parameters'].items()}
    y = np.log1p(fit.price.to_numpy())
    text, encoder, oof, x, fold_audit, text_fit_seconds = components(fit, y, args)
    # Keep compact reproducible ID hashes in the report, rather than huge index lists.
    for row in fold_audit:
        positions = np.asarray(row.pop('validation_positions'), dtype=np.int64)
        row['validation_id_sha256'] = audit_split(fit.iloc[positions])['id_sha256']
    report.update(inner_folds=fold_audit, text_fit_seconds=text_fit_seconds)
    v = encoder.transform(tune)
    base = text.predict(prepare(tune))
    report['ridge_tune_rmsle'] = rmsle(tune.price, prices(base))
    models = {}
    for engine in ('libtree', 'ridge_plus_libtree'):
        train_x = x if engine == 'libtree' else np.ascontiguousarray(np.column_stack([x, oof]), dtype=np.float32)
        target = y if engine == 'libtree' else y - oof
        valid_x = v if engine == 'libtree' else np.ascontiguousarray(np.column_stack([v, base]), dtype=np.float32)
        fits, predicts = [], []
        for _ in range(args.repeats):
            stamp = time.perf_counter()
            model = train_native(train_x, target, args)
            fits.append(time.perf_counter() - stamp)
            stamp = time.perf_counter()
            prediction = model.predict(valid_x)
            predicts.append(time.perf_counter() - stamp)
            model.close()
        models[engine] = train_native(train_x, target, args)
        report['native_timings'].append(dict(engine=engine, fit_seconds=timing(fits), predict_seconds=timing(predicts)))
        for strength in ([1.] if engine == 'libtree' else [.25, .5, 1.]):
            result = prediction if engine == 'libtree' else base + strength * prediction
            report['tuning'].append(dict(engine=engine, strength=strength, rmsle=rmsle(tune.price, prices(result))))
    selected = min(report['tuning'], key=lambda row: row['rmsle'])
    report['selected'] = selected.copy()
    chosen = Entry(text, encoder, models[selected['engine']], selected['engine'], selected['strength'])
    structured = Entry(text, encoder, models['libtree'], 'libtree')
    report['holdout'] = dict(ridge=rmsle(holdout.price, prices(text.predict(prepare(holdout)))),
                            libtree=rmsle(holdout.price, prices(structured.predict(holdout))),
                            selected=rmsle(holdout.price, prices(chosen.predict(holdout))))
    for model in models.values():
        model.close()
    print(json.dumps(dict(selected=selected, holdout=report['holdout'])), flush=True)
    del text, encoder, oof, x, v, train_x, valid_x, models, chosen, structured
    # Freeze every decision before incorporating the holdout in the final refit.
    y = np.log1p(frame.price.to_numpy())
    text, encoder, oof, x, final_audit, text_fit_seconds = components(frame, y, args)
    train_x = x if selected['engine'] == 'libtree' else np.ascontiguousarray(np.column_stack([x, oof]), dtype=np.float32)
    target = y if selected['engine'] == 'libtree' else y - oof
    stamp = time.perf_counter()
    model = train_native(train_x, target, args)
    entry = Entry(text, encoder, model, selected['engine'], selected['strength'])
    report.update(final_text_fit_seconds=text_fit_seconds, final_native_fit_seconds=time.perf_counter() - stamp,
                  final_cross_fit_seconds=sum(row['seconds'] for row in final_audit))
    stamp = time.perf_counter()
    try:
        report['submission'] = write_submission(entry, data / args.test_file, data / args.sample_file, output / 'submission.csv')
        report['final_prediction_seconds_including_features_and_io'] = time.perf_counter() - stamp
        with (output / 'model.pkl').open('wb') as stream:
            pickle.dump(entry, stream)
        names = encoder.feature_names_ + (['ridge_log_prediction'] if selected['engine'] != 'libtree' else [])
        for format, filename in [('text', 'tree.txt'), ('json', 'tree.json')]:
            (output / filename).write_text(model.dump_model(format=format, tree_index=0, feature_names=names))
        explanation = model.explain(train_x[0], feature_names=names, format='json')
        (output / 'explanation.json').write_text(json.dumps(explanation, indent=2))
    finally:
        entry.close()
    report['total_seconds'] = time.perf_counter() - started
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), submission=report['submission'])), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-file', default='test_stg2.tsv')
    parser.add_argument('--sample-file', default='sample_submission_stg2.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--max-features', type=int, default=100000)
    parser.add_argument('--trees', type=int, default=300)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--train-rows', type=int, help='optional deterministic pilot sample; omit for full training')
    args = parser.parse_args(argv)
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == '__main__':
    # Keep pickle class names stable when invoked with python -m.
    from benchmarks.mercari import main as entrypoint
    entrypoint()
