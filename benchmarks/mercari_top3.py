"""Independent top-three-inspired sparse views; no automatic submission.

Primary sources (2018), NOT reproduced leaderboard results:
1. Jankiewicz/Lopukhin, numerical word vectorization and representation diversity:
   https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50256
2. Chahhou/Shahbazi, unordered title/description word pairs:
   https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50499
3. bird, semantic text normalization and category-dependent blending:
   https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50272

This bounded adaptation hashes pairs, uses log1p(quantity), and reuses the
regularized category blender instead of copying hand-tuned category weights.
No original competition code or fitted parameters are included.
"""

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import pickle
import re
import resource
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from benchmarks import mercari
from benchmarks.mercari_cpu_study import WINDOWS, dump, read_data, score
from benchmarks.mercari_multiview import predict, select_strategy, snapshot
from benchmarks.mercari_semantics import paired_comparison
from benchmarks.mercari_penalties import SparseElasticNet, SparseLasso

TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+|[+%❤$]", re.UNICODE)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
SYMBOLS = {"+": "plus", "%": "percent", "❤": "heart", "$": "dollar"}
FIELDS = (
    "name",
    "item_description",
    "brand_name",
    "category_name",
    "item_condition_id",
    "shipping",
)
VIEWS = dict(
    lexical=("lexical",),
    pairs=("lexical", "pairs"),
    quantities=("lexical", "quantities"),
    both=("lexical", "pairs", "quantities"),
)


def sparse_penalty_models(rows):
    """Built-in L1/L1+L2 experts on the strongest prior lexical view.

    No user tuning: sum-loss penalties are fixed before selection. sklearn's
    average-loss alpha is 20/n for Lasso and 40/n with ratio .5 for Elastic Net.
    Their sum-loss objectives are .5*SSE+20*L1 and .5*SSE+20*L1+10*L2².
    Ridge's default SSE+20*L2² shares the latter L2 term after rescaling.
    This aligns objective normalization, not the statistical effect of L1/L2.

    Tibshirani (1996), doi:10.1111/j.2517-6161.1996.tb02080.x.
    Zou & Hastie (2005), doi:10.1111/j.1467-9868.2005.00503.x.
    Sparse coordinate descent: Friedman et al. (2010), doi:10.18637/jss.v033.i01.
    """
    if rows < 1:
        raise ValueError("positive training row count required")
    common = dict(max_iter=200, tol=1e-4, selection="cyclic")
    return dict(
        lasso=SparseLasso(alpha=20.0 / rows, **common),
        elasticnet=SparseElasticNet(alpha=40.0 / rows, l1_ratio=0.5, **common),
    )


def tokens(value, limit=60):
    if pd.isna(value):
        return []
    text = unicodedata.normalize("NFKD", str(value).lower().replace("w/", " with "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [SYMBOLS.get(w, w) for w in TOKEN.findall(text)[:limit]]


def pairs(name, description, prefix, limit):
    words = sorted(set(name + description[:prefix]))[:limit]
    return [(a + "\x1f" + b, 1.0) for a, b in combinations(words, 2)]


def quantities(words):
    return [
        (word, float(np.log1p(float(amount))))
        for amount, word in zip(words, words[1:])
        if NUMBER.fullmatch(amount)
        and not NUMBER.fullmatch(word)
        and 0 < float(amount) <= 200
    ]


class InteractionFeatures(BaseEstimator):
    """Three CSR blocks; IDF learns only from fit rows, hashing never grows.

    A pair uses sorted unique tokens, so title word order cannot change it.
    At most pair_tokens*(pair_tokens-1)/2 pair entries are emitted per listing.
    Amounts attach only to the immediately following token. Unlike the original
    NumericalVectorizer, repeated mentions add and values use log1p scaling.
    """

    def __init__(
        self, n_features=262144, max_tokens=60, pair_tokens=20, description_prefix=5
    ):
        self.n_features, self.max_tokens = n_features, max_tokens
        self.pair_tokens, self.description_prefix = pair_tokens, description_prefix

    def _raw(self, frame):
        def documents(kind):
            for row in frame[list(FIELDS)].itertuples(index=False, name=None):
                name, desc = [tokens(v, self.max_tokens) for v in row[:2]]
                if kind == "pairs":
                    yield pairs(name, desc, self.description_prefix, self.pair_tokens)
                elif kind == "quantities":
                    yield [
                        (field + "=" + w, value)
                        for field, words in (("n", name), ("d", desc))
                        for w, value in quantities(words)
                    ]
                else:
                    yield (
                        [
                            (field + "=" + w, 1.0)
                            for field, words in (("n", name), ("d", desc))
                            for w in set(words)
                        ]
                        + [
                            (
                                str(i)
                                + "="
                                + ("<missing>" if pd.isna(v) else str(v).lower()),
                                1.0,
                            )
                            for i, v in enumerate(row[2:])
                        ]
                    )

        hasher = FeatureHasher(
            n_features=self.n_features,
            input_type="pair",
            alternate_sign=False,
            dtype=np.float32,
        )
        return {
            kind: hasher.transform(documents(kind)).tocsr()
            for kind in ("lexical", "pairs", "quantities")
        }

    def fit_transform(self, frame, y=None):
        if (
            not all(
                isinstance(v, int)
                for v in (
                    self.n_features,
                    self.max_tokens,
                    self.pair_tokens,
                    self.description_prefix,
                )
            )
            or min(self.n_features, self.max_tokens) < 1
            or self.pair_tokens < 2
            or self.description_prefix < 0
        ):
            raise ValueError(
                "positive integer feature/token budgets and nonnegative description prefix required"
            )
        blocks = self._raw(frame)
        self.idf_ = {
            k: TfidfTransformer(sublinear_tf=True).fit(blocks[k])
            for k in ("lexical", "pairs")
        }
        return {
            k: self.idf_[k].transform(v).tocsr() if k in self.idf_ else v
            for k, v in blocks.items()
        }

    def fit(self, frame, y=None):
        self.fit_transform(frame)
        return self

    def transform(self, frame):
        check_is_fitted(self, "idf_")
        return {
            k: self.idf_[k].transform(v).tocsr() if k in self.idf_ else v
            for k, v in self._raw(frame).items()
        }


def run(args):
    start = time.perf_counter()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    frames = read_data(args)
    anchor = np.load(args.anchor)
    for window in WINDOWS:
        if (
            not np.array_equal(
                anchor[window + "_ids"], frames[window].train_id.to_numpy()
            )
            or anchor[window].shape != (len(frames[window]),)
            or not np.isfinite(anchor[window]).all()
        ):
            raise ValueError(
                "finite anchor predictions and identical ordered IDs required"
            )
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        views={k: list(v) for k, v in VIEWS.items()},
        minimum_selection_gain=args.minimum_gain,
        anchor_sha256=mercari.sha256(args.anchor),
        data_sha256={
            k: mercari.sha256(args.data_dir / (k + ".tsv.gz")) for k in frames
        },
        source_sha256=mercari.sha256(Path(__file__)),
        solver_sha256=mercari.sha256(Path(__file__).with_name("mercari_penalties.py")),
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        confirmation_previously_inspected=True,
        external_pretrained_data=args.external_pretrained_data,
        built_in_penalty_models={
            name: model.get_params()
            for name, model in sparse_penalty_models(len(frames["train"])).items()
        },
        penalty_view="pairs",
        inference_repeats=3,
    )
    dump(out / "plan.json", plan)
    stamp = time.perf_counter()
    encoder = InteractionFeatures(
        args.n_features, args.max_tokens, args.pair_tokens, args.description_prefix
    )
    blocks = {"train": encoder.fit_transform(frames["train"])}
    blocks.update({k: encoder.transform(frames[k]) for k in WINDOWS})
    feature_seconds = time.perf_counter() - stamp
    print("TOP3_FEATURES", feature_seconds, flush=True)
    with (out / "encoder.pkl").open("wb") as stream:
        pickle.dump(encoder, stream)
    y = {k: np.log1p(frames[k].price.to_numpy()) for k in ("train", "selection")}
    values = {"anchor": {k: anchor[k] for k in WINDOWS}}
    metrics, matrix_seconds = {}, {}
    for view, keys in VIEWS.items():
        stamp = time.perf_counter()
        x = sparse.hstack([blocks["train"][k] for k in keys], format="csr")
        queries = {
            k: sparse.hstack([blocks[k][v] for v in keys], format="csr")
            for k in WINDOWS
        }
        coordinate_x = x.tocsc() if view == "pairs" else None
        matrix_seconds[view] = time.perf_counter() - stamp
        models = dict(
            ridge=Ridge(
                alpha=args.alpha, solver="lsqr", tol=args.tol, max_iter=args.max_iter
            )
        )
        if view == "pairs":
            models.update(sparse_penalty_models(len(frames["train"])))
        for family, model in models.items():
            name = view if family == "ridge" else family + "_" + view
            stamp = time.perf_counter()
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x if family == "ridge" else coordinate_x, y["train"])
            fit_seconds = time.perf_counter() - stamp
            timings = []
            for _ in range(3):
                stamp = time.perf_counter()
                values[name] = {k: model.predict(queries[k]) for k in WINDOWS}
                timings.append(time.perf_counter() - stamp)
            metrics[name] = dict(
                family=family,
                view=view,
                parameters=model.get_params(),
                selection_rmsle=score(y["selection"], values[name]["selection"]),
                fit_seconds=fit_seconds,
                predict_seconds=float(np.mean(timings)),
                predict_seconds_samples=timings,
                predict_seconds_max=max(timings),
                predict_seconds_std=float(np.std(timings)),
                features=x.shape[1],
                training_nonzeros=x.nnz,
                iterations=int(np.atleast_1d(model.n_iter_)[0]),
                active_coefficients=int(np.count_nonzero(model.coef_)),
                dual_gap=float(model.dual_gap_)
                if hasattr(model, "dual_gap_")
                else None,
                converged=(
                    not any(
                        issubclass(w.category, ConvergenceWarning) for w in observed
                    )
                )
                if family != "ridge"
                else None,
                iteration_limit_reached=int(np.atleast_1d(model.n_iter_)[0])
                >= model.max_iter,
                fit_warnings=[str(w.message) for w in observed],
            )
            with (out / (name + ".pkl")).open("wb") as stream:
                pickle.dump(model, stream)
            np.savez_compressed(
                out / (name + ".npz"),
                **values[name],
                **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
            )
            print("TOP3_MODEL", name, metrics[name], flush=True)
        del x, queries, coordinate_x, models, model
    # A separate grouped half of selection checks the choice of blending method.
    _, check = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=193).split(
            frames["selection"], groups=mercari.listing_groups(frames["selection"])
        )
    )
    pools = dict(
        anchor=["anchor"],
        without_pairs=[
            "anchor",
            *[n for n, m in metrics.items() if m["view"] not in ("pairs", "both")],
        ],
        without_quantities=[
            "anchor",
            *[n for n, m in metrics.items() if m["view"] not in ("quantities", "both")],
        ],
        ridge_only=[
            "anchor",
            *[n for n, m in metrics.items() if m["family"] == "ridge"],
        ],
        without_lasso=[
            "anchor",
            *[n for n, m in metrics.items() if m["family"] != "lasso"],
        ],
        without_elasticnet=[
            "anchor",
            *[n for n, m in metrics.items() if m["family"] != "elasticnet"],
        ],
        full=list(values),
    )
    fitted, predictions, frozen_pools = {}, {}, {}
    for name, names in pools.items():
        p = {k: np.column_stack([values[n][k] for n in names]) for k in WINDOWS}
        model, protocol = select_strategy(
            p["selection"], y["selection"], frames["selection"]
        )
        fitted[name] = model
        predictions[name] = {k: predict(model, p[k], frames[k]) for k in WINDOWS}
        frozen_pools[name] = dict(
            names=names,
            model=snapshot(model),
            chosen=protocol["chosen"],
            validation=protocol["validation"],
            selection_rmsle=score(y["selection"], predictions[name]["selection"]),
        )
    full = frozen_pools["full"]
    accepted = (
        full["validation"][full["chosen"]]
        < score(y["selection"][check], values["anchor"]["selection"][check])
        - args.minimum_gain
    )
    chosen = predictions["full"] if accepted else values["anchor"]
    frozen = dict(
        plan=plan,
        pools=frozen_pools,
        accepted=bool(accepted),
        promotion_decided_before_confirmation=True,
        selection_rmsle=score(y["selection"], chosen["selection"]),
    )
    with (out / "blenders.pkl").open("wb") as stream:
        pickle.dump(fitted, stream)
    dump(out / "frozen.json", frozen)
    # No audit labels are opened until feature, model, method and promotion freeze.
    evaluation = np.load(args.data_dir / "evaluation.npz")
    if (
        not np.array_equal(
            evaluation["confirmation_ids"], frames["confirmation"].train_id.to_numpy()
        )
        or evaluation["confirmation"].shape != (len(frames["confirmation"]),)
        or not np.isfinite(evaluation["confirmation"]).all()
        or (evaluation["confirmation"] < 0).any()
    ):
        raise ValueError("audit IDs must match the frozen feature rows")
    audit_y = evaluation["confirmation"]
    for name in metrics:
        metrics[name]["confirmation_rmsle"] = score(
            audit_y, values[name]["confirmation"]
        )
    report = dict(
        frozen=frozen,
        training_rows=len(frames["train"]),
        models=metrics,
        feature_seconds=feature_seconds,
        matrix_seconds=matrix_seconds,
        seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        gpu_used=False,
        confirmation_previously_inspected=True,
        submission_status="not_submitted",
        ablations={
            name: dict(
                selection_rmsle=row["selection_rmsle"],
                confirmation_rmsle=score(audit_y, predictions[name]["confirmation"]),
            )
            for name, row in frozen_pools.items()
        },
        baseline_confirmation_rmsle=score(audit_y, values["anchor"]["confirmation"]),
        confirmation_rmsle=score(audit_y, chosen["confirmation"]),
        comparison=paired_comparison(
            audit_y,
            values["anchor"]["confirmation"],
            chosen["confirmation"],
            repeats=300,
        ),
    )
    np.savez_compressed(
        out / "ensemble.npz",
        **chosen,
        **{k + "_ids": frames[k].train_id.to_numpy() for k in WINDOWS},
    )
    dump(out / "report.json", report)
    print("TOP3_REPORT", report["confirmation_rmsle"], "accepted", accepted, flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in (
        ("n-features", 262144),
        ("max-tokens", 60),
        ("pair-tokens", 20),
        ("description-prefix", 5),
        ("max-iter", 80),
        ("threads", 4),
        ("rows", 0),
        ("seed", 42),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("alpha", 20.0), ("tol", 0.002), ("minimum-gain", 1e-5)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--external-pretrained-data", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.alpha,
            args.tol,
            args.max_iter,
            args.threads,
            args.n_features,
            args.max_tokens,
        )
        <= 0
        or args.pair_tokens < 2
        or args.description_prefix < 0
        or args.rows < 0
        or not np.isfinite([args.alpha, args.tol, args.minimum_gain]).all()
        or args.minimum_gain < 0
    ):
        parser.error(
            "positive finite fitting settings and valid token/row budgets required"
        )
    with threadpool_limits(args.threads):
        run(args)


if __name__ == "__main__":
    # Persist importable class names when launched with python -m as well.
    from benchmarks.mercari_top3 import main as entry

    entry()
