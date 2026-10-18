"""Small named blends, selected without confirmation or leaderboard labels.

Nonnegative stacking follows Wolpert (1992), doi:10.1016/S0893-6080(05)80023-1.
Model counts here count prediction columns, not upstream fitted dependencies.
"""

import itertools
from pathlib import Path

import numpy as np
from benchmarks.mercari_frontier import WINDOWS, score
from benchmarks.mercari_semantics import blend_weights


def combine(predictions, specification):
    names, weights = (
        specification["names"],
        np.asarray(specification["weights"], dtype=float),
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in predictions for n in names)
        or weights.shape != (len(names),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError(
            "unique known predictions and normalized nonnegative weights required"
        )
    values = [np.asarray(predictions[n]) for n in names]
    if any(
        v.ndim != 1 or v.shape != values[0].shape or not np.isfinite(v).all()
        for v in values
    ):
        raise ValueError("finite matching one-dimensional predictions required")
    return np.column_stack(values) @ weights


def compare_counts(predictions, selection_y, max_models=4):
    y = np.asarray(selection_y)
    if (
        not predictions
        or max_models < 1
        or y.ndim != 1
        or not len(y)
        or not np.isfinite(y).all()
        or any(
            np.asarray(v).shape != y.shape or not np.isfinite(v).all()
            for v in predictions.values()
        )
    ):
        raise ValueError(
            "finite matching selection predictions and positive model counts required"
        )
    results = []
    for count in range(1, min(max_models, len(predictions)) + 1):
        best = None
        for names in itertools.combinations(predictions, count):
            p = np.column_stack([predictions[n] for n in names])
            weights = blend_weights(p, y, penalty=1e-4)
            row = dict(
                count=count,
                names=list(names),
                weights=weights.tolist(),
                selection_rmsle=score(y, p @ weights),
            )
            if best is None or row["selection_rmsle"] < best["selection_rmsle"]:
                best = row
        results.append(best)
    return results


def load_experts(root, names):
    if not names or len(set(names)) != len(names) or "evaluation" in names:
        raise ValueError("unique expert names required; labels are never predictions")
    predictions, identifiers = {}, {}
    for name in names:
        with np.load(Path(root) / (name + ".npz")) as archive:
            predictions[name] = {}
            for window in WINDOWS:
                ids, values = archive[window + "_ids"], archive[window]
                if (
                    values.ndim != 1
                    or values.shape != ids.shape
                    or not np.isfinite(values).all()
                    or (
                        window in identifiers
                        and not np.array_equal(ids, identifiers[window])
                    )
                ):
                    raise ValueError(
                        "finite predictions and identical ordered IDs required"
                    )
                identifiers[window] = ids.copy()
                predictions[name][window] = values.copy()
    return predictions, identifiers


def validate_hosted(report,config,source_archive_sha256):
    """Fail closed before submitting this specific frozen Mercari full refit."""
    import hashlib
    import json
    import re
    expected_hash=hashlib.sha256(json.dumps(config,indent=2).encode()).hexdigest()
    required={n+'-report' for n in ('prepare','baseline','lsa','sequence','separate','joint','stemmed','vectors','transformer','finish')}
    submission,protocol=report['submission'],report['protocol']
    if (report['smoke'] or report['source_archive_sha256']!=source_archive_sha256
            or report['pretrained_weight_sha256']!=config['pretrained_weight_sha256']
            or not protocol['no_selection_or_confirmation_labels']
            or protocol['train_sha256']!='94b24fbecca842140b6276e9da43ce66be3b56b4b991375228802a74050fe471'
            or protocol['splits']['base']['rows']!=1395326 or protocol['splits']['calibration']['rows']!=87209
            or not required.issubset(report['stages']) or submission['rows']!=3460725
            or not submission['no_final_parameter_selection'] or submission['outer']!=config['outer']
            or submission['frozen_config_sha256']!=expected_hash
            or re.fullmatch('[a-f0-9]{64}',submission['sha256']) is None):
        raise ValueError('complete source-matched frozen full refit required; never submit a smoke or inferior variant')
    return submission
