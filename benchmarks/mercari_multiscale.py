"""Budgeted continuation: global/parent product priors at several strengths.

Only stored base-window statistics are reparameterized. No price expert is
retrained, and confirmation never chooses promotion or mixture weights.
"""

import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from benchmarks import mercari
from benchmarks.mercari_cpu_study import WINDOWS, dump, score
from benchmarks.mercari_multiview import disagreement, fit_strategy, predict, snapshot
from benchmarks.mercari_semantics import TextAttributes, paired_comparison


def run(args):
    start = time.perf_counter()
    out = args.output_dir
    previous = args.refine_from
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((previous / "frozen.json").read_text())
    if mercari.sha256(args.baseline) != old["baseline_sha256"]:
        raise ValueError("baseline must match prior study")
    pool = np.load(previous / "pool.npz")
    frames = {
        k: pd.read_csv(args.data_dir / (k + ".tsv.gz"), sep="\t") for k in WINDOWS
    }
    matrices = {k: pool[k] for k in WINDOWS}
    for k in WINDOWS:
        if (
            not np.array_equal(pool[k + "_ids"], frames[k].train_id.to_numpy())
            or mercari.sha256(args.data_dir / (k + ".tsv.gz")) != old["data_sha256"][k]
        ):
            raise ValueError("feature rows must match prior pool and input hashes")
    if "price" in frames["confirmation"]:
        raise ValueError("confirmation prices must remain separate")
    with (previous / "product-statistics.pkl").open("rb") as stream:
        original = pickle.load(stream)
    shared = original.reparameterize(20.0)
    variants = {
        f"{prior}{alpha}": shared.reparameterize(alpha, hierarchical=prior == "parent")
        for prior, alphas in [("global", (5, 100)), ("parent", (5, 20, 100))]
        for alpha in alphas
    }
    with (out / "statistics-views.pkl").open("wb") as stream:
        pickle.dump(variants, stream)
    with args.baseline.open("rb") as stream:
        entry = pickle.load(stream)
    cols = [
        i
        for i, name in enumerate(old["names"])
        if name in ("ridge", "mean_prior", "ffm") or name.startswith("neural_")
    ]
    features = {name: {} for name in ("global", "hierarchical")}
    try:
        for k in WINDOWS:
            components = matrices[k][:, cols]
            basic = np.column_stack(
                [
                    entry.encoder.transform(frames[k]),
                    components,
                    disagreement(components),
                    TextAttributes().transform(frames[k]),
                    original.transform(frames[k]),
                ]
            )
            extra = {
                name: variant.transform(frames[k]) for name, variant in variants.items()
            }
            global_x = np.column_stack(
                [basic, *[extra[name] for name in extra if name.startswith("global")]]
            )
            features["global"][k] = np.ascontiguousarray(global_x, dtype=np.float32)
            features["hierarchical"][k] = np.ascontiguousarray(
                np.column_stack(
                    [
                        global_x,
                        *[extra[name] for name in extra if name.startswith("parent")],
                    ]
                ),
                dtype=np.float32,
            )
    finally:
        entry.close()
    preparation = time.perf_counter() - start
    print("CPU_MULTISCALE_FEATURES", preparation, flush=True)
    y = {k: np.log1p(frames[k].price.to_numpy()) for k in ("calibration", "selection")}
    base = {k: matrices[k][:, old["names"].index("uncorrected")] for k in WINDOWS}
    from libtree import GBDTRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    common = dict(n_estimators=args.tree_rounds, learning_rate=0.04)
    models = {
        f"hier_libtree{d}": (
            "hierarchical",
            GBDTRegressor(
                **common,
                max_depth=d,
                max_bins=128,
                min_samples_leaf=100,
                reg_lambda=20.0,
                n_jobs=args.threads,
            ),
        )
        for d in (3, 5)
    }
    models.update(
        {
            f"hier_lgb{leaves}": (
                "hierarchical",
                LGBMRegressor(
                    **common,
                    num_leaves=leaves,
                    min_child_samples=100,
                    reg_lambda=20.0,
                    n_jobs=args.threads,
                    random_state=42,
                    verbosity=-1,
                ),
            )
            for leaves in (31, 63)
        }
    )
    models["hier_huber"] = (
        "hierarchical",
        LGBMRegressor(
            **common,
            num_leaves=31,
            objective="huber",
            min_child_samples=100,
            reg_lambda=20.0,
            n_jobs=args.threads,
            random_state=42,
            verbosity=-1,
        ),
    )
    models["hier_xgb4"] = (
        "hierarchical",
        XGBRegressor(
            **common,
            max_depth=4,
            min_child_weight=100,
            reg_lambda=20.0,
            n_jobs=args.threads,
            tree_method="hist",
            device="cpu",
            random_state=42,
        ),
    )
    models["global_lgb31"] = (
        "global",
        LGBMRegressor(
            **common,
            num_leaves=31,
            min_child_samples=100,
            reg_lambda=20.0,
            n_jobs=args.threads,
            random_state=42,
            verbosity=-1,
        ),
    )
    values = {}
    metrics = {}
    for name, (view, model) in models.items():
        stamp = time.perf_counter()
        try:
            model.fit(
                features[view]["calibration"], y["calibration"] - base["calibration"]
            )
            fit_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            residual = {k: model.predict(features[view][k]) for k in WINDOWS}
            for strength in (0.5, 1.0):
                values[name + "_" + str(strength)] = {
                    k: base[k] + strength * residual[k] for k in WINDOWS
                }
            metrics[name] = dict(
                view=view,
                features=features[view]["selection"].shape[1],
                fit_seconds=fit_seconds,
                predict_seconds=time.perf_counter() - stamp,
                selection_rmsle=score(
                    y["selection"], values[name + "_1.0"]["selection"]
                ),
            )
            with (out / (name + ".pkl")).open("wb") as stream:
                pickle.dump(model, stream)
        finally:
            if hasattr(model, "close"):
                model.close()
        print("CPU_MULTISCALE_MODEL", name, metrics[name], flush=True)
    names = old["names"] + list(values)
    full = {
        k: np.column_stack([matrices[k], *[values[n][k] for n in values]])
        for k in WINDOWS
    }
    kind = old["strategy"]["chosen"]
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=193).split(
            frames["selection"], groups=mercari.listing_groups(frames["selection"])
        )
    )
    baseline_cv = fit_strategy(
        kind, matrices["selection"][a], y["selection"][a], frames["selection"].iloc[a]
    )
    proposal_cv = fit_strategy(
        kind, full["selection"][a], y["selection"][a], frames["selection"].iloc[a]
    )
    old_score = score(
        y["selection"][b],
        predict(baseline_cv, matrices["selection"][b], frames["selection"].iloc[b]),
    )
    new_score = score(
        y["selection"][b],
        predict(proposal_cv, full["selection"][b], frames["selection"].iloc[b]),
    )
    accepted = bool(new_score < old_score - 1e-5)
    proposal = fit_strategy(
        kind, full["selection"], y["selection"], frames["selection"]
    )
    baseline = fit_strategy(
        kind, matrices["selection"], y["selection"], frames["selection"]
    )
    mask = np.array([not name.startswith("hier_") for name in names])
    no_parent = fit_strategy(
        kind, full["selection"][:, mask], y["selection"], frames["selection"]
    )
    model = proposal if accepted else baseline
    chosen = full if accepted else matrices
    frozen = dict(
        names=names if accepted else old["names"],
        model=snapshot(model),
        proposal=snapshot(proposal),
        proposed_names=names,
        kind=kind,
        accepted=accepted,
        validation=dict(previous=old_score, proposal=new_score, threshold=1e-5),
        promotion_decided_before_confirmation=True,
        proposal_selection_rmsle=score(
            y["selection"], predict(proposal, full["selection"], frames["selection"])
        ),
        source_sha256={
            p.name: mercari.sha256(p)
            for p in [Path(__file__), Path(__file__).with_name("mercari_multiview.py")]
        },
        prior_frozen_sha256=mercari.sha256(previous / "frozen.json"),
        prior_pool_sha256=mercari.sha256(previous / "pool.npz"),
        variants=list(variants),
        parent_smoothing=20,
        tree_rounds=args.tree_rounds,
        ablation=dict(names=list(np.asarray(names)[mask]), model=snapshot(no_parent)),
    )
    dump(out / "frozen.json", frozen)
    audit = np.load(args.data_dir / "evaluation.npz")
    if not np.array_equal(
        audit["confirmation_ids"], frames["confirmation"].train_id.to_numpy()
    ):
        raise ValueError("audit IDs must match")
    truth = audit["confirmation"]
    prediction = {k: predict(model, chosen[k], frames[k]) for k in WINDOWS}
    report = dict(
        frozen=frozen,
        models=metrics,
        preparation_seconds=preparation,
        seconds=time.perf_counter() - start,
        confirmation_rmsle=score(truth, prediction["confirmation"]),
        proposal_confirmation_rmsle=score(
            truth, predict(proposal, full["confirmation"], frames["confirmation"])
        ),
        ablations=dict(
            without_hierarchical_views=dict(
                selection=score(
                    y["selection"],
                    predict(no_parent, full["selection"][:, mask], frames["selection"]),
                ),
                confirmation=score(
                    truth,
                    predict(
                        no_parent, full["confirmation"][:, mask], frames["confirmation"]
                    ),
                ),
            )
        ),
        paired_comparison=paired_comparison(
            truth,
            matrices["confirmation"][:, 0],
            prediction["confirmation"],
            repeats=300,
        ),
        individual={
            n: dict(
                selection=score(y["selection"], values[n]["selection"]),
                confirmation=score(truth, values[n]["confirmation"]),
            )
            for n in values
        },
        parent_statistics_reused=True,
        gpu_used=False,
        confirmation_previously_inspected=True,
        submission_status="not_submitted",
    )
    np.savez_compressed(
        out / "ensemble.npz",
        **prediction,
        **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
    )
    dump(out / "report.json", report)
    print(
        "CPU_MULTISCALE_RESULT",
        report["confirmation_rmsle"],
        "accepted",
        accepted,
        flush=True,
    )
    return report
