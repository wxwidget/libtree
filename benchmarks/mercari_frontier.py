"""Reproducible modern-resource Mercari experiments, never automatic submission.

The held-out labels stay local until the blend is frozen. This extends the
existing grouped seed-42 protocol; its confirmation window has been inspected
in the preceding study and is therefore not a new untouched test set.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_semantics import LatentText, SentenceVectors, TextAttributes, blend_weights, paired_comparison

WINDOWS = ('calibration', 'selection', 'confirmation')


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def score(y, prediction):
    return float(np.sqrt(np.mean((np.maximum(prediction, 0.) - y)**2)))


def read_windows(data):
    return {name: pd.read_csv(data / (name + '.tsv.gz'), sep='\t') for name in ('train', *WINDOWS)}


def prepare_data(args):
    from benchmarks.mercari_advanced import calibration_split
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(args.data_dir / 'train.tsv', sep='\t')
    base, tune, holdout = mercari.split(frame, 42)
    calibration, selection = calibration_split(tune, .5, 44)
    parts = dict(train=base, calibration=calibration, selection=selection, confirmation=holdout)
    for name, part in parts.items():
        # Transfer no confirmation labels to a remotely tuned model.
        public = part.drop(columns='price') if name == 'confirmation' else part
        public.to_csv(out / (name + '.tsv.gz'), sep='\t', index=False, compression='gzip')
    np.savez_compressed(out / 'evaluation.npz', **{name: np.log1p(parts[name].price.to_numpy()) for name in WINDOWS},
        **{name + '_ids': parts[name].train_id.to_numpy() for name in WINDOWS})
    stamp = time.perf_counter()
    with args.baseline.open('rb') as stream:
        entry = pickle.load(stream)
    predictions = {}
    try:
        for name in WINDOWS:
            predictions[name] = entry.predict(parts[name])
    finally:
        entry.close()
    np.savez_compressed(out / 'baseline.npz', **predictions,
        **{name + '_ids': parts[name].train_id.to_numpy() for name in WINDOWS})
    dump(out / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        train_sha256=mercari.sha256(args.data_dir / 'train.tsv'), baseline_sha256=mercari.sha256(args.baseline),
        splits={name: mercari.audit_split(part) for name, part in parts.items()},
        baseline_prediction_seconds=time.perf_counter() - stamp,
        confirmation_labels_transferred=False, confirmation_previously_inspected=True,
        baseline_selection_rmsle=score(np.log1p(selection.price), predictions['selection']),
        sampling='All official training rows, grouped seed-42 80/5/5/10 split'))


def tree_study(args):
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    frame = read_windows(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    y = np.log1p(frame['train'].price.to_numpy())
    selection_y = np.log1p(frame['selection'].price.to_numpy())
    stamp = time.perf_counter()
    encoder = LatentText(components=args.components, max_features=args.max_features, seed=42).fit(frame['train'])
    x = {name: encoder.transform(part) for name, part in frame.items()}
    preparation_seconds = time.perf_counter() - stamp
    with (args.output_dir / 'encoder.pkl').open('wb') as stream:
        pickle.dump(encoder, stream)
    predictions, metrics = {}, {}
    linear_columns = encoder.categorical_indices_[0]
    for alpha in (10., 100.):
        stamp = time.perf_counter()
        model = Ridge(alpha=alpha).fit(x['train'][:, :linear_columns], y)
        name = f'lsa_ridge_{alpha:g}'
        predictions[name] = {k: model.predict(x[k][:, :linear_columns]) for k in WINDOWS}
        metrics[name] = dict(selection_rmsle=score(selection_y, predictions[name]['selection']),
                             fit_seconds=time.perf_counter() - stamp)
        with (args.output_dir / (name + '.pkl')).open('wb') as stream:
            pickle.dump(model, stream)
    for leaves, rounds in ((31, 900), (63, 900)):
        stamp = time.perf_counter()
        model = lgb.LGBMRegressor(n_estimators=rounds, num_leaves=leaves, learning_rate=.04,
            min_child_samples=80, reg_lambda=10., colsample_bytree=.85,
            n_jobs=args.threads, random_state=42, verbosity=-1)
        model.fit(x['train'], y, categorical_feature=encoder.categorical_indices_,
            eval_set=[(x['selection'], selection_y)], callbacks=[lgb.early_stopping(75, verbose=False)])
        name = f'lsa_lgbm_{leaves}'
        predictions[name] = {k: model.predict(x[k]) for k in WINDOWS}
        metrics[name] = dict(selection_rmsle=score(selection_y, predictions[name]['selection']),
            fit_seconds=time.perf_counter() - stamp, best_iteration=model.best_iteration_)
        model.booster_.save_model(str(args.output_dir / (name + '.txt')))
        print(name, metrics[name], flush=True)
    dump(args.output_dir / 'report.json', dict(models=metrics, preparation_seconds=preparation_seconds,
        components=args.components, max_features=args.max_features, threads=args.threads,
        topics=encoder.topics(), lightgbm=importlib.metadata.version('lightgbm')))
    save_predictions(args.output_dir, predictions, frame)


def save_predictions(out, predictions, frame):
    for name, values in predictions.items():
        np.savez_compressed(out / (name + '.npz'), **values,
            **{window + '_ids': frame[window].train_id.to_numpy() for window in WINDOWS})


def calibration_study(args):
    """Replace the old correction with text-aware native/competitor trees.

    Base expert predictions precede their old fitted correction. Consequently
    every calibration feature comes from a model that did not fit that row's
    label. Confirmation labels are never loaded by this command.
    """
    from libtree import GBDTRegressor
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    args.output_dir.mkdir(parents=True, exist_ok=False)
    parts = read_windows(args.data_dir)
    frame = {k: parts[k] for k in WINDOWS}
    if 'price' in frame['confirmation']:
        raise ValueError('confirmation labels must stay local')
    product_views = {}
    if getattr(args, 'product_views', False):
        from benchmarks.mercari_multiview import ProductStatistics
        # Base-only smoothing: Micci-Barreca (2001), doi:10.1145/507533.507538.
        statistics = ProductStatistics(20.).fit(parts['train'], np.log1p(parts['train'].price.to_numpy()))
        product_views = {f'{prior}{alpha}': statistics.reparameterize(alpha, hierarchical=prior == 'parent')
                         for prior in ('global', 'parent') for alpha in (5, 20, 100)}
        with (args.output_dir / 'product-views.pkl').open('wb') as stream:
            pickle.dump(product_views, stream)
    del parts
    with args.baseline.open('rb') as stream:
        entry = pickle.load(stream)
    with args.lsa_encoder.open('rb') as stream:
        encoder = pickle.load(stream)
    names, prediction, features = ['old_uncorrected'], {}, {}
    try:
        for window, part in frame.items():
            experts, metadata, _ = entry.parts(part)
            prediction[window] = (experts @ entry.config['weights'])[:, None]
            latent = encoder.transform(part)[:, :encoder.categorical_indices_[0]]
            features[window] = np.column_stack([metadata, latent, TextAttributes().transform(part)])
            if product_views:
                features[window] = np.column_stack([features[window], *[view.transform(part) for view in product_views.values()]])
    finally:
        entry.close()
    for root in args.experts:
        for path in sorted(root.glob('*.npz')):
            archive = np.load(path)
            if not all(window in archive for window in WINDOWS):
                continue
            if path.stem in names or path.stem == 'evaluation':
                raise ValueError('expert names must be unique; label archives are not predictions')
            for window in WINDOWS:
                if (window + '_ids' not in archive
                        or not np.array_equal(archive[window + '_ids'], frame[window].train_id.to_numpy())):
                    raise ValueError('expert prediction IDs must match calibration windows')
                prediction[window] = np.column_stack([prediction[window], archive[window]])
            names.append(path.stem)
    y = {k: np.log1p(frame[k].price.to_numpy()) for k in ('calibration', 'selection')}
    weights = blend_weights(prediction['selection'], y['selection'], penalty=1e-4)
    base = {k: prediction[k] @ weights for k in WINDOWS}
    x = {k: np.ascontiguousarray(np.column_stack([features[k], prediction[k]]), dtype=np.float32) for k in WINDOWS}
    target = y['calibration'] - base['calibration']
    common = dict(n_estimators=args.tree_rounds, learning_rate=.04)
    models = {f'cal_libtree{depth}': GBDTRegressor(**common, max_depth=depth, max_bins=128,
        min_samples_leaf=100, reg_lambda=20., n_jobs=args.threads) for depth in (3, 5)}
    models['cal_lightgbm'] = LGBMRegressor(**common, num_leaves=15, min_child_samples=200,
        reg_lambda=20., n_jobs=args.threads, random_state=42, verbosity=-1)
    models['cal_xgboost'] = XGBRegressor(**common, max_depth=4, min_child_weight=100,
        reg_lambda=20., n_jobs=args.threads, tree_method='hist', random_state=42)
    predictions, metrics = {}, {}
    for alpha in (10., 100.):
        stamp = time.perf_counter()
        model = Ridge(alpha=alpha, positive=True).fit(prediction['calibration'], y['calibration'])
        name = f'cal_meta_ridge{alpha:g}'
        predictions[name] = {k: model.predict(prediction[k]) for k in WINDOWS}
        metrics[name] = dict(selection_rmsle=score(y['selection'], predictions[name]['selection']),
            coefficients=model.coef_.tolist(), intercept=float(model.intercept_),
            seconds=time.perf_counter() - stamp)
        with (args.output_dir / (name + '.pkl')).open('wb') as stream:
            pickle.dump(model, stream)
    try:
        for name, model in models.items():
            stamp = time.perf_counter()
            model.fit(x['calibration'], target)
            correction = {k: model.predict(x[k]) for k in WINDOWS}
            strengths = (0., .25, .5, 1.)
            scores = [score(y['selection'], base['selection'] + value * correction['selection']) for value in strengths]
            best = int(np.argmin(scores))
            strength = strengths[best]
            predictions[name] = {k: base[k] + strength * correction[k] for k in WINDOWS}
            metrics[name] = dict(selection_rmsle=scores[best], strength=strength,
                strength_scores=dict(zip(map(str, strengths), scores)), seconds=time.perf_counter() - stamp)
            with (args.output_dir / (name + '.pkl')).open('wb') as stream:
                pickle.dump(model, stream)
            print(name, metrics[name], flush=True)
    finally:
        for model in models.values():
            if hasattr(model, 'close'):
                model.close()
    save_predictions(args.output_dir, predictions, frame)
    dump(args.output_dir / 'report.json', dict(models=metrics, expert_names=names, weights=weights.tolist(),
        calibration_rows=len(target), selection_rows=len(y['selection']), tree_rounds=args.tree_rounds,
        confirmation_labels_loaded=False, feature_count=x['calibration'].shape[1],
        text_attribute_names=TextAttributes().feature_names_,
        product_feature_names=[f'{name}_{feature}' for name, view in product_views.items() for feature in view.feature_names_]))


def gpu_study(args):
    import tensorflow as tf
    from benchmarks.mercari_features import (JointListingFeatures, ListingFeatures,
        RICH_TOKEN_PATTERN, StemmedListingFeatures, make_ridge)
    from benchmarks.mercari_neural import Tokens, build_model, dataset, predict
    from benchmarks.mercari_sparse import SparseExpert, select_columns
    devices = tf.config.list_physical_devices('GPU')
    if not devices and not args.allow_cpu:
        raise RuntimeError('GPU experiment requires a real TensorFlow GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    preparation_start = time.perf_counter()
    print('Loading grouped training and prediction windows', flush=True)
    frame = read_windows(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    listing_type = {'joint': JointListingFeatures, 'stemmed': StemmedListingFeatures}.get(args.sparse_view, ListingFeatures)
    listing = listing_type().fit(frame['train'])
    print('Preparing train-only text dictionaries', flush=True)
    prepared = {k: listing.transform(v) for k, v in frame.items()}
    y = np.log1p(frame['train'].price.to_numpy()).astype(np.float32)
    val_y = np.log1p(frame['selection'].price.to_numpy())
    mean, predictions, metrics = float(np.mean(y)), {}, {}
    hardware = dict(tensorflow=tf.__version__, gpu_devices=[d.name for d in devices],
        gpu_details=[tf.config.experimental.get_device_details(d) for d in devices],
        train_rows=len(y), threads=args.threads, logical_cpu_count=os.cpu_count(),
        preparation_seconds=time.perf_counter() - preparation_start)
    if args.neural_only != 'sparse':
        tokens = Tokens(max_words=args.max_features, title_length=24, description_length=96,
                        unknown_tokens=True).fit(prepared['train'])
        print('Encoding fixed-width sequences', flush=True)
        encoded = {k: tokens.transform(v) for k, v in prepared.items()}
        with (args.output_dir / 'tokens.pkl').open('wb') as stream:
            pickle.dump((listing, tokens, mean), stream)
        for name, architecture, width, loss in [('cnn128_mse', 'cnn', 128, 'mse'),
                                                ('gru96_huber', 'gru', 96, 'huber')]:
            stamp = time.perf_counter()
            criterion = tf.keras.losses.Huber(delta=.5) if loss == 'huber' else loss
            model = build_model(tokens, seed=42, width=width, loss=criterion, architecture=architecture)
            best, history = float('inf'), []
            for epoch, rate in enumerate([.003, .002, .001, .0005, .00025][:args.epochs]):
                model.optimizer.learning_rate.assign(rate)
                start = time.perf_counter()
                fit = model.fit(dataset(encoded['train'], y - mean, batch_size=1024, shuffle=True),
                                epochs=1, verbose=0, shuffle=False)
                selection = mean + predict(model, encoded['selection'], batch_size=2048)
                row = dict(epoch=epoch + 1, rate=rate, seconds=time.perf_counter() - start,
                           loss=float(fit.history['loss'][0]), selection_rmsle=score(val_y, selection))
                history.append(row)
                print(name, row, flush=True)
                if row['selection_rmsle'] < best:
                    best = row['selection_rmsle']
                    tf.keras.Model(model.inputs, model.outputs).save(args.output_dir / (name + '.keras'))
            model = tf.keras.models.load_model(args.output_dir / (name + '.keras'), compile=False)
            predictions[name] = {k: mean + predict(model, encoded[k], batch_size=2048) for k in WINDOWS}
            metrics[name] = dict(history=history, selection_rmsle=best, seconds=time.perf_counter() - stamp)
            save_predictions(args.output_dir, {name: predictions[name]}, frame)
            dump(args.output_dir / 'report.json', dict(models=metrics, **hardware))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        del encoded
        gc.collect()
    if args.neural_only == 'sequence':
        return
    stamp = time.perf_counter()
    extractor = make_ridge(max_features=args.max_features, char_features=75000)['features']
    if args.sparse_view == 'stemmed':
        extractor.transformers = [(name, transformer, 'raw_name' if name == 'characters' else columns)
                                  for name, transformer, columns in extractor.transformers]
        for name, transformer, _ in extractor.transformers:
            if name in ('name', 'description'):
                transformer.set_params(token_pattern=RICH_TOKEN_PATTERN)
    x = {'train': extractor.fit_transform(prepared['train'])}
    x.update({k: extractor.transform(prepared[k]) for k in WINDOWS})
    blocks = {k: v for k, v in extractor.output_indices_.items() if k != 'remainder'}
    keep = select_columns(x['train'], blocks, word_limit=args.max_features, char_limit=75000, min_df=5)
    x = {k: v[:, keep].tocsr() for k, v in x.items()}
    hardware.update(sparse_features=int(x['train'].shape[1]), sparse_training_nonzeros=int(x['train'].nnz))
    with (args.output_dir / 'sparse_encoder.pkl').open('wb') as stream:
        pickle.dump((listing, extractor, keep), stream)
    metrics['sparse_feature_seconds'] = time.perf_counter() - stamp
    prefix = ('stemmed_' if args.sparse_view == 'stemmed' else '') + 'sparse' + str(args.sparse_width)
    for name, binary, loss in [(prefix + '_tfidf_mse', False, 'mse'),
                               (prefix + '_binary_huber', True, 'huber'),
                               (prefix + '_binary_bins', True, 'bins')]:
        model = SparseExpert(width=args.sparse_width, seed=117 if loss == 'bins' else 77 if binary else 43, loss=loss, binary=binary,
                             optimizer='row_adam')
        start = time.perf_counter()
        model.fit(x['train'], y, rates=[.003, .003, .002, .001][:args.epochs],
                  batches=[1024, 2048, 4096, 8192][:args.epochs],
                  validation=(x['selection'], val_y))
        predictions[name] = {k: model.predict(x[k]) for k in WINDOWS}
        with (args.output_dir / (name + '.pkl')).open('wb') as stream:
            pickle.dump(model, stream)
        metrics[name] = dict(history=model.history_, seconds=time.perf_counter() - start,
                             selection_rmsle=score(val_y, predictions[name]['selection']))
        save_predictions(args.output_dir, {name: predictions[name]}, frame)
        dump(args.output_dir / 'report.json', dict(models=metrics, **hardware))
        del model
        tf.keras.backend.clear_session()
        gc.collect()


def vector_study(args):
    import torch
    from huggingface_hub import model_info
    from sentence_transformers import SentenceTransformer
    from lightgbm import LGBMRegressor, early_stopping
    from sklearn.linear_model import Ridge
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('semantic encoding requires a real Torch GPU')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame = read_windows(args.data_dir)
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'
    revision = args.revision or model_info(model_id).sha
    encoder = SentenceTransformer(model_id, revision=revision, device='cuda' if torch.cuda.is_available() else 'cpu')
    encoder.max_seq_length = 128
    adapter = SentenceVectors(encoder, batch_size=256)
    precision_audit, dtype = None, 'float32'
    if torch.cuda.is_available() and args.embedding_dtype == 'float16':
        sample = frame['train'].iloc[:32]
        reference = adapter.transform(sample)
        encoder.half()
        reduced = adapter.transform(sample)
        cosine = np.sum(reference * reduced, axis=1) / np.maximum(
            np.linalg.norm(reference, axis=1) * np.linalg.norm(reduced, axis=1), 1e-12)
        precision_audit = dict(max_absolute_error=float(np.max(np.abs(reference - reduced))),
                               minimum_cosine=float(np.min(cosine)), rows=len(sample))
        if not np.isfinite(reduced).all() or np.min(cosine) < .995 or precision_audit['max_absolute_error'] > .01:
            raise RuntimeError('half-precision audit failed; use float32')
        dtype = 'float16'
    stamp = time.perf_counter()
    x = {}
    for window, part in frame.items():
        x[window] = adapter.transform(part)
        np.save(args.output_dir / (window + '_vectors.npy'), x[window])
        print('vectors', window, x[window].shape, flush=True)
    encoding_seconds = time.perf_counter() - stamp
    y, val_y = [np.log1p(frame[k].price.to_numpy()) for k in ('train', 'selection')]
    predictions, metrics = {}, {}
    for alpha in (10., 100.):
        start = time.perf_counter()
        model = Ridge(alpha=alpha).fit(x['train'], y)
        name = f'minilm_ridge_{alpha:g}'
        predictions[name] = {k: model.predict(x[k]) for k in WINDOWS}
        metrics[name] = dict(selection_rmsle=score(val_y, predictions[name]['selection']),
                             seconds=time.perf_counter() - start)
        with (args.output_dir / (name + '.pkl')).open('wb') as stream:
            pickle.dump(model, stream)
    # Category-aware latent features are fitted exclusively on the same base rows.
    metadata = LatentText(components=16, max_features=10000).fit(frame['train'])
    meta = {k: metadata.transform(v) for k, v in frame.items()}
    offset = x['train'].shape[1]
    combined = {k: np.column_stack([x[k], meta[k]]) for k in frame}
    start = time.perf_counter()
    tree = LGBMRegressor(n_estimators=700, num_leaves=63, learning_rate=.05,
        min_child_samples=80, reg_lambda=10., colsample_bytree=.8, n_jobs=args.threads, random_state=42, verbosity=-1)
    tree.fit(combined['train'], y, categorical_feature=[offset + i for i in metadata.categorical_indices_],
        eval_set=[(combined['selection'], val_y)], callbacks=[early_stopping(60, verbose=False)])
    name = 'minilm_lgbm63'
    predictions[name] = {k: tree.predict(combined[k]) for k in WINDOWS}
    metrics[name] = dict(selection_rmsle=score(val_y, predictions[name]['selection']),
                        seconds=time.perf_counter() - start, best_iteration=tree.best_iteration_)
    tree.booster_.save_model(str(args.output_dir / (name + '.txt')))
    with (args.output_dir / 'metadata.pkl').open('wb') as stream:
        pickle.dump(metadata, stream)
    save_predictions(args.output_dir, predictions, frame)
    dump(args.output_dir / 'report.json', dict(models=metrics, model_id=model_id, revision=revision,
        external_pretrained_data=True, max_sequence_length=128, encoding_seconds=encoding_seconds,
        embedding_dtype=dtype, precision_audit=precision_audit,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        sentence_transformers=importlib.metadata.version('sentence-transformers')))


def ensemble_study(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    evaluation = np.load(args.data_dir / 'evaluation.npz')
    paths = [args.data_dir / 'baseline.npz', *sorted(p for root in args.experts for p in root.glob('*.npz'))]
    names, values = [], []
    for path in paths:
        archive = np.load(path)
        if not all(window in archive for window in WINDOWS):
            continue
        if path.stem in names or path.stem == 'evaluation':
            raise ValueError('expert names must be unique; label archives are not predictions')
        for window in WINDOWS:
            if (window + '_ids' not in archive
                    or not np.array_equal(archive[window + '_ids'], evaluation[window + '_ids'])):
                raise ValueError('prediction IDs must match every evaluation window exactly')
        names.append(path.stem)
        values.append({window: archive[window] for window in WINDOWS})
    prediction = {window: np.column_stack([v[window] for v in values]) for window in WINDOWS}
    weights = blend_weights(prediction['selection'], evaluation['selection'], penalty=1e-4)
    # Persist the frozen choice before inspecting any confirmation labels.
    frozen = dict(created_utc=datetime.now(timezone.utc).isoformat(), names=names, weights=weights.tolist(),
        selection_rmsle=score(evaluation['selection'], prediction['selection'] @ weights),
        candidate_sha256={str(p): mercari.sha256(p) for p in paths},
        confirmation_previously_inspected=True, external_pretrained_data=(getattr(args, 'external_pretrained_data', False)
            or any('minilm' in n for n in names)))
    dump(args.output_dir / 'frozen.json', frozen)
    individual = {n: dict(selection_rmsle=score(evaluation['selection'], prediction['selection'][:, i]),
                           confirmation_rmsle=score(evaluation['confirmation'], prediction['confirmation'][:, i]))
                  for i, n in enumerate(names)}
    comparison = paired_comparison(evaluation['confirmation'], prediction['confirmation'][:, 0],
        prediction['confirmation'] @ weights, repeats=300)
    ablation = {}
    for i, name in enumerate(names[1:], 1):
        if weights[i] == 0:
            continue
        mask = np.arange(len(names)) != i
        local = blend_weights(prediction['selection'][:, mask], evaluation['selection'], penalty=1e-4)
        ablation[name] = dict(selection_rmsle=score(evaluation['selection'], prediction['selection'][:, mask] @ local),
            confirmation_rmsle=score(evaluation['confirmation'], prediction['confirmation'][:, mask] @ local))
    dump(args.output_dir / 'report.json', dict(frozen=frozen, individual=individual, comparison=comparison,
        leave_one_expert_out=ablation, residual_correlation=np.corrcoef(
            (prediction['selection'] - evaluation['selection'][:, None]).T).tolist(),
        status='local confirmation only; not a new Kaggle score or rank'))
    np.savez_compressed(args.output_dir / 'ensemble.npz',
        **{window: prediction[window] @ weights for window in WINDOWS},
        **{window + '_ids': evaluation[window + '_ids'] for window in WINDOWS})
    print(json.dumps(dict(weights=dict(zip(names, weights.tolist())), comparison=comparison)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'trees', 'gpu', 'vectors', 'calibrate', 'ensemble'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--lsa-encoder', type=Path)
    parser.add_argument('--experts', type=Path, nargs='*', default=[])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--components', type=int, default=64)
    parser.add_argument('--max-features', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=5, help='Bounded neural schedule: 1 to 5 epochs')
    parser.add_argument('--tree-rounds', type=int, default=400)
    parser.add_argument('--neural-only', choices=['all', 'sequence', 'sparse'], default='all')
    parser.add_argument('--sparse-view', choices=['separate', 'joint', 'stemmed'], default='separate')
    parser.add_argument('--sparse-width', type=int, default=256)
    parser.add_argument('--revision', help='Pinned pretrained model revision; recorded if resolved automatically')
    parser.add_argument('--embedding-dtype', choices=['float32', 'float16'], default='float32')
    parser.add_argument('--product-views', action='store_true',
                        help='Append base-only multiscale product statistics to calibration tree features')
    parser.add_argument('--external-pretrained-data', action='store_true',
                        help='Declare pretrained data in any expert or upstream baseline')
    parser.add_argument('--allow-cpu', action='store_true', help='Only for functional tests, never GPU evidence')
    args = parser.parse_args(argv)
    if (min(args.threads, args.components, args.max_features, args.tree_rounds, args.sparse_width) < 1
            or not 1 <= args.epochs <= 5):
        parser.error('positive resource and feature settings required')
    if args.mode == 'prepare' and args.baseline is None:
        parser.error('prepare requires --baseline from the documented fitting window')
    if args.mode == 'calibrate' and (args.baseline is None or args.lsa_encoder is None):
        parser.error('calibrate requires --baseline and --lsa-encoder')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(args.threads))
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    with threadpool_limits(args.threads):
        dict(prepare=prepare_data, trees=tree_study, gpu=gpu_study,
             vectors=vector_study, calibrate=calibration_study, ensemble=ensemble_study)[args.mode](args)


if __name__ == '__main__':
    main()
