"""Validated frozen Mercari settings, independent of optional training engines."""
import numpy as np

DEFAULT = dict(seed=42, threads=4, max_features=300000, char_features=150000,
    ridge_alpha=10., max_words=100000, neural_width=64, neural_loss='mse', neural_architecture='bag',
    neural_rates=[.003, .003, .003], neural_members=[], neural_text_version=2,
    fm_workers=2, fm_rank=12, fm_epochs=2, fm_rate=.03,
    neural_batches=[2048, 2048, 2048], sparse_word_limit=100000, sparse_char_limit=50000,
    sparse_min_df=5, sparse_binary=True, sparse_optimizer='adam', sparse_bins=64,
    fm_l2=.001, fm_error_clip=1e30, fm_chunk=32768, weights=[.3, .3, .4],
    correction_strength=.5, calibration_fraction=1/17,
    native=dict(n_estimators=600, max_depth=4, max_bins=128, min_samples_leaf=100,
                learning_rate=.04, reg_lambda=20., n_jobs=4))


def validate(config):
    if set(config) - {'ridge_reference'} != set(DEFAULT):
        raise ValueError('configuration fields must match the documented schema')
    weights = np.asarray(config['weights'], dtype=float)
    positive = ['threads', 'max_features', 'ridge_alpha', 'max_words', 'neural_width',
                'fm_workers', 'fm_rank', 'fm_epochs', 'fm_rate', 'fm_chunk', 'fm_error_clip',
                'sparse_word_limit', 'sparse_min_df', 'sparse_bins']
    if (any(not np.isfinite(config[k]) or config[k] <= 0 for k in positive)
            or weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1.) or not 0 < config['calibration_fraction'] < 1
            or config['neural_loss'] not in ('mse', 'huber', 'bins') or not config['neural_rates']
            or (config['neural_loss'] == 'bins' and config['neural_architecture'] != 'sparse')
            or config['neural_architecture'] not in ('bag', 'cnn', 'gru', 'sparse')
            or config['neural_text_version'] not in (1, 2)
            or not all(np.isfinite(v) and v > 0 for v in config['neural_rates'])
            or len(config['neural_batches']) != len(config['neural_rates']) or min(config['neural_batches']) < 1
            or config['sparse_char_limit'] < 0 or not isinstance(config['sparse_binary'], bool)
            or config['sparse_optimizer'] not in ('adam', 'row_adam')
            or config['char_features'] < 0 or config['fm_l2'] < 0
            or not isinstance(config.get('ridge_reference', True), bool)
            or (not config.get('ridge_reference', True) and weights[0] != 0)
            or config['fm_workers'] > config['threads'] or not 0 <= config['correction_strength'] <= 1):
        raise ValueError('invalid or unknown ensemble configuration')
    if config['neural_members']:
        allowed = {'weight', 'neural_architecture', 'neural_width', 'neural_loss', 'neural_text_version',
                   'neural_rates', 'neural_batches', 'seed', 'sparse_optimizer', 'sparse_binary',
                   'sparse_word_limit', 'sparse_char_limit', 'sparse_min_df', 'sparse_bins'}
        member_weights = [member['weight'] for member in config['neural_members']]
        if not np.isfinite(member_weights).all() or min(member_weights) < 0 or not np.isclose(sum(member_weights), 1.):
            raise ValueError('neural member weights must be nonnegative and sum to one')
        for member in config['neural_members']:
            if not set(member) <= allowed:
                raise ValueError('unknown neural member override')
            local = dict(config, **{k: v for k, v in member.items() if k != 'weight'})
            local['neural_members'] = []
            validate(local)


