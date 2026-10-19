"""Ablate market-aware heads of ONE base expert; final fusion remains frozen."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_frontier import dump, score, WINDOWS
from benchmarks.mercari_market import MarketSignals, RegimePriceHead
from benchmarks.mercari_multiview import ProductStatistics
from benchmarks.mercari_semantics import TextAttributes, paired_comparison

ROUNDS = 450
REGIME_ROUNDS = 180
BUCKETS = np.array([0, 5, 10, 20, 50, 100, 200, 500, 1000, np.inf])


def bucket_metrics(y, p):
    bins = np.searchsorted(np.log1p(BUCKETS), y, side="right") - 1
    actual = np.expm1(y)
    predicted = np.expm1(np.maximum(p, 0))
    total = np.sum((np.maximum(p, 0) - y) ** 2)
    result = []
    for i, (low, high) in enumerate(zip(BUCKETS[:-1], BUCKETS[1:])):
        use = bins == i
        if use.any():
            result.append(
                dict(
                    lower=float(low),
                    upper=None if np.isinf(high) else float(high),
                    n=int(use.sum()),
                    rmsle=score(y[use], p[use]),
                    mae_dollars=float(np.mean(np.abs(predicted[use] - actual[use]))),
                    bias_dollars=float(np.mean(predicted[use] - actual[use])),
                    underprediction_percent=float(
                        np.mean(predicted[use] < actual[use]) * 100
                    ),
                    log_squared_error_share=float(
                        np.sum((np.maximum(p[use], 0) - y[use]) ** 2) / total
                    )
                    if total
                    else 0.0,
                )
            )
    return result


def run(args):
    start = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    hashes = {
        k: mercari.sha256(args.data_dir / (k + ".tsv.gz")) for k in ("train", *WINDOWS)
    }
    if manifest["data_sha256"] != hashes:
        raise ValueError("market heads must match the complete original data protocol")
    matching = [
        e
        for e in manifest["experts"]
        if Path(e["path"]).resolve() == args.base_predictions.resolve()
    ]
    if (
        len(matching) != 1
        or matching[0]["fit_window"] != "base"
        or mercari.sha256(args.base_predictions) != matching[0]["sha256"]
    ):
        raise ValueError("exactly one hashed base-only prediction archive required")
    frame = {k: pd.read_csv(args.data_dir / (k + ".tsv.gz"), sep="\t") for k in WINDOWS}
    if "price" in frame["confirmation"]:
        raise ValueError("confirmation prices must remain sealed")
    base = {}
    with np.load(args.base_predictions) as archive:
        for k in WINDOWS:
            if not np.array_equal(archive[k + "_ids"], frame[k].train_id.to_numpy()):
                raise ValueError("base expert IDs must match every ordered window")
            base[k] = archive[k].copy()
            if base[k].shape != (len(frame[k]),) or not np.isfinite(base[k]).all():
                raise ValueError("one finite base prediction per listing required")
    fit, tune = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=809).split(
            frame["calibration"], groups=mercari.listing_groups(frame["calibration"])
        )
    )
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        base_expert=matching[0],
        data_sha256=hashes,
        encoder_sha256=mercari.sha256(args.encoder),
        source_sha256={
            n: mercari.sha256(Path(__file__).with_name(n))
            for n in ("mercari_market.py", "mercari_market_study.py")
        },
        fit_rows=len(fit),
        tune_rows=len(tune),
        fit_ids=frame["calibration"].train_id.iloc[fit].tolist(),
        tune_ids=frame["calibration"].train_id.iloc[tune].tolist(),
        rounds=ROUNDS,
        regime_rounds=REGIME_ROUNDS,
        threads=2,
        final_fusion_modified=False,
        base_trunk_retrained=False,
        base_experts_per_input=1,
        target_reweighting=False,
        drop_zero_prices=False,
        early_stop_window="grouped calibration tune",
        selection_role="choose head only; no fitting",
        historical_selection_and_audit_reused=True,
        soft_regimes=["zero", "positive below 10", "10 to below 50", "at least 50"],
        inference_routing="observed listing features and one base prediction only",
        comparable_statistics_fit_window="base",
        comparable_smoothing=20.0,
        comparable_prior="brand/category parent",
    )
    dump(out / "plan.json", plan)
    with args.encoder.open("rb") as stream:
        encoder = pickle.load(stream)
    preparation_start = time.perf_counter()
    train = pd.read_csv(args.data_dir / "train.tsv.gz", sep="\t")
    comparables = (
        ProductStatistics(20.0)
        .fit(train, np.log1p(train.price.to_numpy()))
        .reparameterize(20.0, hierarchical=True)
    )
    del train
    with (out / "comparables.pkl").open("wb") as stream:
        pickle.dump(comparables, stream)
    features = {k: {} for k in ("plain", "legacy", "market", "comparables")}
    for window, part in frame.items():
        features["plain"][window] = np.ascontiguousarray(
            np.column_stack([encoder.transform(part), base[window]]), dtype=np.float32
        )
        features["legacy"][window] = np.ascontiguousarray(
            np.column_stack(
                [features["plain"][window], TextAttributes().transform(part)]
            ),
            dtype=np.float32,
        )
        features["market"][window] = np.ascontiguousarray(
            np.column_stack(
                [features["legacy"][window], MarketSignals().transform(part)]
            ),
            dtype=np.float32,
        )
        features["comparables"][window] = np.ascontiguousarray(
            np.column_stack([features["market"][window], comparables.transform(part)]),
            dtype=np.float32,
        )
        print(
            "MARKET_FEATURES", window, features["comparables"][window].shape, flush=True
        )
    preparation_seconds = time.perf_counter() - preparation_start
    y = {k: np.log1p(frame[k].price.to_numpy()) for k in ("calibration", "selection")}
    metrics = {
        "raw_text": dict(
            selection_rmsle=score(y["selection"], base["selection"]),
            fit_seconds=0.0,
            refit_seconds=0.0,
            cached_trunk_cost_excluded=True,
        )
    }
    prediction = {"raw_text": base}
    models = {}
    common = dict(
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=80,
        reg_lambda=10.0,
        colsample_bytree=0.85,
        n_jobs=2,
        random_state=42,
        verbosity=-1,
        metric="None",
    )
    for name, view, objective in [
        ("plain_l2", "plain", "regression"),
        ("legacy_l2", "legacy", "regression"),
        ("market_l2", "market", "regression"),
        ("market_huber", "market", "huber"),
        ("market_regime", "market", "regime"),
        ("comparables_l2", "comparables", "regression"),
        ("comparables_regime", "comparables", "regime"),
    ]:
        x = features[view]
        stamp = time.perf_counter()
        if objective == "regime":
            meta = RegimePriceHead(
                rounds=REGIME_ROUNDS,
                categorical_indices=tuple(encoder.categorical_indices_),
            ).fit(
                x["calibration"][fit],
                y["calibration"][fit],
                validation=(x["calibration"][tune], y["calibration"][tune]),
            )
            budget = meta.budgets_
        else:
            meta = lgb.LGBMRegressor(n_estimators=ROUNDS, objective=objective, **common)

            def metric(truth, pred):
                return (
                    "final_rmsle",
                    score(y["calibration"][tune], base["calibration"][tune] + pred),
                    False,
                )

            meta.fit(
                x["calibration"][fit],
                y["calibration"][fit] - base["calibration"][fit],
                categorical_feature=encoder.categorical_indices_,
                eval_set=[
                    (
                        x["calibration"][tune],
                        y["calibration"][tune] - base["calibration"][tune],
                    )
                ],
                eval_metric=metric,
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            budget = meta.best_iteration_ or ROUNDS
        fit_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        if objective == "regime":
            final = RegimePriceHead(
                rounds=REGIME_ROUNDS,
                categorical_indices=tuple(encoder.categorical_indices_),
            ).fit(x["calibration"], y["calibration"], budgets=budget)
        else:
            final = lgb.LGBMRegressor(
                n_estimators=budget, objective=objective, **common
            ).fit(
                x["calibration"],
                y["calibration"] - base["calibration"],
                categorical_feature=encoder.categorical_indices_,
            )
        refit_seconds = time.perf_counter() - stamp
        samples = []
        for _ in range(3):
            stamp = time.perf_counter()
            prediction[name] = {
                k: final.predict(x[k]) + (0 if objective == "regime" else base[k])
                for k in WINDOWS
            }
            samples.append(time.perf_counter() - stamp)
        metrics[name] = dict(
            selection_rmsle=score(y["selection"], prediction[name]["selection"]),
            fit_seconds=fit_seconds,
            refit_seconds=refit_seconds,
            rounds=budget,
            features=x["selection"].shape[1],
            view=view,
            objective=objective,
            predict_seconds=float(np.mean(samples)),
            predict_seconds_samples=samples,
            predict_seconds_max=max(samples),
            predict_seconds_std=float(np.std(samples)),
            prediction_rows=sum(len(part) for part in frame.values()),
            cached_trunk_cost_excluded=True,
        )
        models[name] = dict(model=final, view=view, residual=objective != "regime")
        with (out / (name + ".pkl")).open("wb") as stream:
            pickle.dump(models[name], stream)
        print("MARKET_MODEL", name, metrics[name], flush=True)
    best = min(row["selection_rmsle"] for row in metrics.values())
    chosen = next(
        name for name, row in metrics.items() if row["selection_rmsle"] <= best + 1e-5
    )
    for name, values in prediction.items():
        np.savez_compressed(
            out / (name + ".npz"),
            **values,
            **{k + "_ids": frame[k].train_id.to_numpy() for k in WINDOWS},
        )
    frozen = dict(
        plan=plan,
        models=metrics,
        chosen=chosen,
        selection_rule="minimum selection RMSLE; 1e-5 tie prefers earlier/simpler",
        all_choices_frozen_before_audit=True,
    )
    dump(out / "frozen.json", frozen)
    with np.load(args.data_dir / "evaluation.npz") as audit:
        truth = audit["confirmation"]
        if (
            not np.array_equal(
                audit["confirmation_ids"], frame["confirmation"].train_id.to_numpy()
            )
            or truth.shape != (len(frame["confirmation"]),)
            or not np.isfinite(truth).all()
            or (truth < 0).any()
        ):
            raise ValueError(
                "audit log prices and ordered IDs must match frozen predictions"
            )
    result = {
        name: {
            **row,
            "confirmation_rmsle": score(truth, prediction[name]["confirmation"]),
            "confirmation_buckets": bucket_metrics(
                truth, prediction[name]["confirmation"]
            ),
        }
        for name, row in metrics.items()
    }
    comparisons = {
        name: paired_comparison(
            truth, base["confirmation"], values["confirmation"], repeats=300
        )
        for name, values in prediction.items()
        if name != "raw_text"
    }
    report = dict(
        frozen=frozen,
        models=result,
        comparisons=comparisons,
        chosen=chosen,
        preparation_seconds=preparation_seconds,
        total_seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        gaussian_price_assumption_required=False,
        final_fusion_modified=False,
        submission_status="not_submitted",
    )
    dump(out / "report.json", report)
    print("MARKET_RESULT", chosen, result[chosen]["confirmation_rmsle"], flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/mercari/fusion-candidates.json"),
    )
    p.add_argument("--base-predictions", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    with threadpool_limits(limits=2):
        return run(p.parse_args(argv))


if __name__ == "__main__":
    main()
