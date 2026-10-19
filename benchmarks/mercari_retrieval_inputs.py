"""Immutable full-base inputs for retrieval-conditioned text fine tuning.

The existing base-fitted CNN tokenizer/weights are trusted local artifacts.
Every new price-derived input excludes the query's full listing group. Raw
model/quantity strings get a stable hashed channel, independent of target labels.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import time

import numpy as np
import pandas as pd
from numba import set_num_threads

from benchmarks import mercari
from benchmarks.mercari_frontier import dump
from benchmarks.mercari_market import MarketSignals
from benchmarks.mercari_retrieval import ComparableRetrieval

WINDOWS = ("train", "calibration", "selection", "confirmation")
IDENTITY_BUCKETS = 8192


def identity_tokens(frame):
    result = np.zeros((len(frame), 16), np.int32)
    pattern = re.compile(
        r"\b[a-z]+[\s-]*\d+[a-z\d]*\b|\b\d+(?:\.\d+)?\s*(?:gb|tb|oz|ml|mm|cm|inch)\b"
    )
    for i, (title, description) in enumerate(
        zip(frame.name.fillna(""), frame.item_description.fillna(""))
    ):
        tokens = sorted(
            set(
                re.sub(r"[\s-]+", "", v)
                for v in pattern.findall((str(title) + " " + str(description)).lower())
            )
        )[:16]
        result[i, : len(tokens)] = [
            1
            + int.from_bytes(
                hashlib.blake2b(v.encode(), digest_size=8).digest(), "little"
            )
            % IDENTITY_BUCKETS
            for v in tokens
        ]
    return result


def key(args):
    return dict(
        versions={
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "scipy", "scikit-learn", "numba")
        },
        data_sha256={
            w: mercari.sha256(args.data_dir / (w + ".tsv.gz")) for w in WINDOWS
        },
        model_sha256={
            n: mercari.sha256(args.model_dir / n)
            for n in ("tokens.pkl", "cnn128_mse.keras")
        },
        source_sha256={
            n: mercari.sha256(Path(__file__).with_name(n))
            for n in (
                "mercari_retrieval.py",
                "mercari_retrieval_inputs.py",
                "mercari_market.py",
                "mercari.py",
                "mercari_features.py",
                "mercari_neural.py",
            )
        },
    )


def prepare(args):
    start = time.perf_counter()
    out = args.cache_dir
    out.mkdir(parents=True, exist_ok=False)
    identity = key(args)
    manifest = json.loads(args.manifest.read_text())
    if manifest["data_sha256"] != identity["data_sha256"]:
        raise ValueError(
            "retrieval inputs must match the original complete data protocol"
        )
    reference = [e for e in manifest["experts"] if e["name"] == "frontier_cnn128_mse"]
    if (
        len(reference) != 1
        or reference[0]["fit_window"] != "base"
        or mercari.sha256(reference[0]["path"]) != reference[0]["sha256"]
    ):
        raise ValueError("hashed base-only CNN reference required")
    with (args.model_dir / "tokens.pkl").open("rb") as stream:
        listing, tokens, mean = pickle.load(stream)
    set_num_threads(2)
    train = pd.read_csv(args.data_dir / "train.tsv.gz", sep="\t")
    retrieval = ComparableRetrieval().fit(train, np.log1p(train.price.to_numpy()))
    with (out / "retrieval.pkl").open("wb") as stream:
        pickle.dump(retrieval, stream)
    rows = {}
    files = {}
    with np.load(reference[0]["path"]) as archive:
        for window in WINDOWS:
            frame = (
                train
                if window == "train"
                else pd.read_csv(args.data_dir / (window + ".tsv.gz"), sep="\t")
            )
            if window == "confirmation" and "price" in frame:
                raise ValueError("audit prices must stay sealed")
            if window != "train" and not np.array_equal(
                archive[window + "_ids"], frame.train_id.to_numpy()
            ):
                raise ValueError("reference IDs must match every ordered window")
            encoded = tokens.transform(listing.transform(frame))
            encoded["context"] = np.ascontiguousarray(
                np.column_stack(
                    [MarketSignals().transform(frame), retrieval.transform(frame)]
                ),
                dtype=np.float32,
            )
            encoded["identity"] = identity_tokens(frame)
            encoded["ids"] = frame.train_id.to_numpy()
            if window != "confirmation":
                encoded["y"] = np.log1p(frame.price.to_numpy())
            if window == "train":
                np.save(out / "context-mean.npy", encoded["context"].mean(0))
                np.save(
                    out / "context-scale.npy",
                    np.maximum(encoded["context"].std(0), 1e-3),
                )
            files[window] = list(encoded)
            for name, value in encoded.items():
                np.save(out / (window + "-" + name + ".npy"), value, allow_pickle=False)
            rows[window] = len(frame)
            print(
                "RETRIEVAL_INPUTS",
                window,
                len(frame),
                time.perf_counter() - start,
                flush=True,
            )
    payload = dict(
        identity=identity,
        rows=rows,
        arrays=files,
        mean=float(mean),
        reference=reference[0],
        feature_names=[
            *MarketSignals().feature_names_,
            *ComparableRetrieval.feature_names_,
        ],
        identity_buckets=IDENTITY_BUCKETS,
        retrieval_reference_groups=len(retrieval.groups_),
        query_duplicate_group_excluded=True,
        all_priors_leave_group_out=True,
        preparation_seconds=time.perf_counter() - start,
        files={p.name: mercari.sha256(p) for p in out.iterdir()},
    )
    dump(out / "manifest.json", payload)
    return payload


def load(args):
    saved = json.loads((args.cache_dir / "manifest.json").read_text())
    if saved["identity"] != key(args):
        raise ValueError("retrieval cache identity mismatch")
    for name, digest in saved["files"].items():
        if mercari.sha256(args.cache_dir / name) != digest:
            raise ValueError("retrieval cache hash mismatch")
    arrays = {
        w: {
            name: np.load(
                args.cache_dir / (w + "-" + name + ".npy"),
                mmap_mode="c",
                allow_pickle=False,
            )
            for name in saved["arrays"][w]
        }
        for w in WINDOWS
    }
    return saved, arrays
