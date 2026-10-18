"""Mercari text experts with a LibTree correction fitted on separate listings.

Final training uses every label, with disjoint base-expert and calibration
roles. See benchmarks/mercari/README.md for selection and audit methodology.
"""
import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from libtree import GBDTRegressor
from benchmarks import mercari
from benchmarks.mercari_features import ListingFeatures, ResidualFeatures, make_ridge
from benchmarks.mercari_fm import FactorizationMachine
from benchmarks.mercari_solver import SparseRidge

from benchmarks.mercari_config import DEFAULT, validate

EXPERT_NAMES = ['ridge', 'field_factorization', 'neural']


def calibration_split(frame, fraction, seed):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
        frame, groups=mercari.listing_groups(frame)))
    return frame.iloc[a], frame.iloc[b]


def select_weights(prediction, log_y):
    prediction, log_y = np.asarray(prediction), np.asarray(log_y)
    if (prediction.ndim != 2 or not len(log_y) or prediction.shape != (len(log_y), 3)
            or not np.isfinite(prediction).all() or not np.isfinite(log_y).all()):
        raise ValueError('three finite expert predictions must match validation labels')
    candidates = np.asarray([parts for parts in itertools.product(range(11), repeat=3) if sum(parts) == 10]) / 10.
    scores = [np.mean((np.maximum(prediction @ w, 0.) - log_y)**2) for w in candidates]
    best = int(np.argmin(scores))
    return candidates[best].tolist(), float(np.sqrt(scores[best]))


def fm_columns(extractor):
    size = extractor.output_indices_['categories'].stop
    fields, scale = np.zeros(size, np.int32), np.zeros(size, np.float32)
    for name, field, weight in [('name', 0, .75), ('description', 1, 1.)]:
        block = extractor.output_indices_[name]
        fields[block], scale[block] = field, weight
    position = extractor.output_indices_['categories'].start
    for i, values in enumerate(extractor.named_transformers_['categories'].categories_):
        width = len(values)
        fields[position:position + width] = [3, 2, 4, 4, 3, 3, 3, 3, 2, 2][i]
        if i < 7:
            scale[position:position + width] = .5
        position += width
    keep = np.flatnonzero(scale)
    return keep, scale[keep], fields[keep]


class Experts:
    def __init__(self, listing, extractor, ridge, fm, keep, scales, neural, mean):
        self.listing, self.extractor, self.ridge = listing, extractor, ridge
        self.fm, self.keep, self.scales = fm, keep, scales
        self.neural, self.mean = neural, mean

    def predict(self, frame):
        return self.predict_prepared(frame, self.listing.transform(frame))

    def predict_prepared(self, frame, p):
        x = self.extractor.transform(p)
        ridge = np.full(len(frame), self.mean) if self.ridge is None else self.ridge.predict(x)
        fm = self.mean + self.fm.predict(x[:, self.keep].multiply(self.scales).tocsr())
        neural = self.neural.predict(frame, p, x)
        return np.column_stack([ridge, fm, neural])


class BagText:
    def __init__(self, tokens, model, mean, listing=None):
        self.tokens, self.model, self.mean, self.listing = tokens, model, mean, listing

    def predict(self, frame, prepared, sparse_x):
        from benchmarks.mercari_neural import predict
        if self.listing is not None:
            prepared = self.listing.transform(frame)
        return self.mean + predict(self.model, self.tokens.transform(prepared))


class SparseText:
    def __init__(self, columns, model):
        self.columns, self.model = columns, model

    def predict(self, frame, prepared, sparse_x):
        return self.model.predict(sparse_x[:, self.columns])


def fit_experts(frame, config):
    stamp = time.perf_counter()
    listing = ListingFeatures().fit(frame)
    prepared = listing.transform(frame)
    y = np.log1p(frame.price.to_numpy())
    mean = float(np.mean(y))
    extractor = make_ridge(config['max_features'], config['ridge_alpha'], config['char_features'])['features']
    x = extractor.fit_transform(prepared)
    ridge = None
    if config.get('ridge_reference', True):
        ridge = SparseRidge(alpha=config['ridge_alpha'], tol=1e-4, max_iter=500).fit(x, y)
    times = dict(ridge_and_features_seconds=time.perf_counter() - stamp,
                 ridge_iterations=0 if ridge is None else ridge.n_iter_, ridge_reference_enabled=ridge is not None)
    keep, scales, fields = fm_columns(extractor)
    fm = FactorizationMachine(len(keep), rank=config['fm_rank'], workers=config['fm_workers'],
        learning_rate=config['fm_rate'], l2=config['fm_l2'], chunk_size=config['fm_chunk'],
        fields=fields, error_clip=config['fm_error_clip'], seed=config['seed'])
    fm_x = x[:, keep].multiply(scales).tocsr()
    samples = []
    for epoch in range(config['fm_epochs']):
        stamp = time.perf_counter()
        fm.fit_epoch(fm_x, y - mean, seed=config['seed'] + epoch)
        samples.append(time.perf_counter() - stamp)
    frozen = fm.freeze()
    del fm, fm_x
    times['fm_epoch_seconds'] = mercari.timing(samples)
    stamp = time.perf_counter()
    neural = fit_neural(frame, prepared, x, y, mean, extractor, config)
    times['neural_features_and_fit_seconds'] = time.perf_counter() - stamp
    return Experts(listing, extractor, ridge, frozen, keep, scales, neural, mean), times


class MixtureText:
    """Weighted adapters share the Ridge feature matrix and listing preparation."""

    def __init__(self, members, weights):
        self.members, self.weights = members, np.asarray(weights, dtype=float)

    def predict(self, frame, prepared, sparse_x):
        return self.weights @ np.vstack([m.predict(frame, prepared, sparse_x) for m in self.members])


def fit_neural(frame, prepared, x, y, mean, extractor, config):
    import tensorflow as tf
    from benchmarks.mercari_neural import Tokens, build_model, dataset
    if config['neural_members']:
        members, weights = [], []
        for member in config['neural_members']:
            local = dict(config, **{k: v for k, v in member.items() if k != 'weight'})
            local['neural_members'] = []
            members.append(fit_neural(frame, prepared, x, y, mean, extractor, local))
            weights.append(member['weight'])
        return MixtureText(members, weights)
    if config['neural_architecture'] == 'sparse':
        from benchmarks.mercari_sparse import SparseExpert, select_columns
        blocks = {k: v for k, v in extractor.output_indices_.items() if k != 'remainder'}
        columns = select_columns(x, blocks, config['sparse_word_limit'], config['sparse_char_limit'], config['sparse_min_df'])
        nonlinear = SparseExpert(config['neural_width'], config['seed'], config['neural_loss'],
                                  config['sparse_binary'], config['sparse_optimizer'], config['sparse_bins'])
        nonlinear.fit(x[:, columns], y, config['neural_rates'], config['neural_batches'])
        neural = SparseText(columns, nonlinear)
    else:
        legacy = ListingFeatures(version=1).fit(frame) if config['neural_text_version'] == 1 else None
        local = prepared if legacy is None else legacy.transform(frame)
        tokens = Tokens(max_words=config['max_words'], version=config['neural_text_version'],
                        unknown_tokens=config['neural_architecture'] == 'gru').fit(local)
        encoded = tokens.transform(local)
        loss = tf.keras.losses.Huber(delta=.5) if config['neural_loss'] == 'huber' else 'mse'
        model = build_model(tokens, seed=config['seed'], width=config['neural_width'], loss=loss,
                            architecture=config['neural_architecture'])
        schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch, rate: config['neural_rates'][epoch])
        model.fit(dataset(encoded, y - mean, batch_size=config['neural_batches'][0], shuffle=True),
                  epochs=len(config['neural_rates']), callbacks=[schedule], verbose=2, shuffle=False)
        neural = BagText(tokens, tf.keras.Model(model.inputs, model.outputs), mean, legacy)
    return neural


class Entry:
    def __init__(self, experts, encoder, native, config):
        self.experts, self.encoder, self.native, self.config = experts, encoder, native, copy.deepcopy(config)

    def parts(self, frame):
        prepared = self.experts.listing.transform(frame)
        prediction = self.experts.predict_prepared(frame, prepared)
        x = np.ascontiguousarray(np.column_stack([self.encoder.transform(frame, prepared=prepared), prediction]), dtype=np.float32)
        return prediction, x, self.native.predict(x)

    def predict(self, frame):
        prediction, _, correction = self.parts(frame)
        return prediction @ self.config['weights'] + self.config['correction_strength'] * correction

    def explain(self, frame):
        if len(frame) != 1:
            raise ValueError('explain requires exactly one listing')
        prediction, x, correction = self.parts(frame)
        contributions = prediction[0] * self.config['weights']
        native = float(self.config['correction_strength'] * correction[0])
        return dict(experts=self.expert_names, weighted_expert_log_contributions=contributions.tolist(),
                    weighted_native_correction=native, price=float(mercari.prices(contributions.sum() + native)),
                    native_residual_path=self.native.explain(x[0], feature_names=self.feature_names, format='json'))

    @property
    def expert_names(self):
        names = list(EXPERT_NAMES)
        if self.experts.ridge is None:
            names[0] = 'mean_prior'
        return names

    @property
    def feature_names(self):
        return self.encoder.feature_names_ + [name + '_log_price' for name in self.expert_names]

    def close(self):
        self.native.close()


def fit_entry(base, calibration, config):
    experts, times = fit_experts(base, config)
    stamp = time.perf_counter()
    encoder = ResidualFeatures().fit(base, np.log1p(base.price.to_numpy()))
    prepared = experts.listing.transform(calibration)
    prediction = experts.predict_prepared(calibration, prepared)
    x = np.ascontiguousarray(np.column_stack([encoder.transform(calibration, prepared=prepared), prediction]), dtype=np.float32)
    feature_seconds = time.perf_counter() - stamp
    target = np.log1p(calibration.price.to_numpy()) - prediction @ config['weights']
    stamp = time.perf_counter()
    native = GBDTRegressor(**config['native']).fit(x, target)
    return Entry(experts, encoder, native, config), dict(experts=times,
        residual_features_seconds=feature_seconds, native_fit_seconds=time.perf_counter() - stamp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['study', 'final'], default='final')
    parser.add_argument('--audit-holdout', action='store_true')
    parser.add_argument('--omit-model-artifact', action='store_true', help='retain predictions and explanations without a large pickle')
    parser.add_argument('--test-file', default='test_stg2.tsv')
    parser.add_argument('--sample-file', default='sample_submission_stg2.csv')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output directory must be empty')
    if args.audit_holdout and args.mode != 'study':
        raise ValueError('holdout audit is only available in study mode')
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(config['threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    start = time.perf_counter()
    frame = pd.read_csv(args.data_dir / 'train.tsv', sep='\t')
    if 'train_id' not in frame or frame.train_id.duplicated().any():
        raise ValueError('training rows require unique train_id')
    mercari.rmsle(frame.price, frame.price)
    if args.mode == 'study':
        base, tune, holdout = mercari.split(frame, config['seed'])
        calibration, selection = calibration_split(tune, .5, config['seed'] + 2)
    else:
        base, calibration = calibration_split(frame, config['calibration_fraction'], config['seed'] + 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(mode=args.mode, created_utc=datetime.now(timezone.utc).isoformat(), config=config,
        model_artifact_saved=not args.omit_model_artifact,
        input_files=dict(train='train.tsv', test=args.test_file, sample=args.sample_file),
        input_train_sha256=mercari.sha256(args.data_dir / 'train.tsv'),
        native_library_sha256=mercari.sha256(mercari.libtree._library()._name),
        source_sha256={p.name: mercari.sha256(p) for p in Path(__file__).parent.glob('mercari*.py')},
        splits=dict(base=mercari.audit_split(base), calibration=mercari.audit_split(calibration)))
    with threadpool_limits(config['threads']):
        entry, stages = fit_entry(base, calibration, config)
        try:
            report['stages'] = stages
            if args.mode == 'study':
                report['selection_rmsle'] = mercari.rmsle(selection.price, mercari.prices(entry.predict(selection)))
                if args.audit_holdout:
                    report['holdout_rmsle'] = mercari.rmsle(holdout.price, mercari.prices(entry.predict(holdout)))
                example = selection.iloc[:1]
            else:
                stamp = time.perf_counter()
                report['submission'] = mercari.write_submission(entry, args.data_dir / args.test_file,
                    args.data_dir / args.sample_file, args.output_dir / 'submission.csv')
                report['prediction_seconds_including_features_and_io'] = time.perf_counter() - stamp
                example = pd.read_csv(args.data_dir / args.test_file, sep='\t', nrows=1)
                report['input_test_sha256'] = mercari.sha256(args.data_dir / args.test_file)
                report['input_sample_sha256'] = mercari.sha256(args.data_dir / args.sample_file)
            if not args.omit_model_artifact:
                with (args.output_dir / 'model.pkl').open('wb') as stream:
                    pickle.dump(entry, stream)
            for format in ('text', 'json'):
                filename = 'tree.txt' if format == 'text' else 'tree.json'
                (args.output_dir / filename).write_text(entry.native.dump_model(format=format, tree_index=0, feature_names=entry.feature_names))
            (args.output_dir / 'explanation.json').write_text(json.dumps(entry.explain(example), indent=2))
        finally:
            entry.close()
    report['total_seconds'] = time.perf_counter() - start
    report['versions'] = {name: __import__(name).__version__ for name in
                          ('numpy', 'pandas', 'scipy', 'sklearn', 'tensorflow', 'numba', 'libtree')}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    from benchmarks.mercari_advanced import main as entrypoint
    entrypoint()
