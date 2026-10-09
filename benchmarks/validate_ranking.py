"""Validate raw evidence and optionally require fastest pooled median timings."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def validate(data, previous=None, check_sources=False):
    splits = data["split_reports"]
    assert splits and len(splits) == len(data["seeds"])
    groups = defaultdict(list)
    keys = set()
    engines = {"libtree", "xgboost", "lightgbm"}
    names = set(splits[0]["sources"])
    assert names
    for split in splits:
        assert set(split["sources"]) == names
        assert set(split["metadata"]["engines"]) == engines
        assert split["metadata"]["parameters"] == splits[0]["metadata"]["parameters"]
        if check_sources:
            for path, expected in split["metadata"]["source_sha256"].items():
                assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
        for row in split["results"]:
            key = (row["dataset"], row["language"], row["engine"], row["seed"])
            assert key not in keys
            keys.add(key)
            assert set(row["checks"]) == {"beats_baseline", "finite_predictions",
                                          "deterministic_repeats", "cpp_python_parity"}
            assert all(row["checks"].values())
            assert all(math.isfinite(value) for value in row["metrics"].values())
            assert len(row["raw_runs"]) == data["timing_repeats_per_split"]
            for run in row["raw_runs"]:
                assert all(math.isfinite(run[field]) and run[field] > 0
                           for field in ("fit_seconds", "predict_seconds", "peak_rss_kib"))
            groups[key[:3]].extend(row["raw_runs"])
    assert len(keys) == len(names) * 2 * len(engines) * len(splits)
    assert all((name, language, engine, seed) in keys for name in names
               for language in ("cpp", "python") for engine in engines for seed in data["seeds"])
    if previous is not None:
        old_splits = {split['metadata']['seed']: split for split in previous['split_reports']}
        for split in splits:
            before_split = old_splits[split['metadata']['seed']]
            assert before_split['metadata']['parameters'] == split['metadata']['parameters']
            for name in names:
                assert (before_split['sources'][name]['prepared_split_sha256'] ==
                        split['sources'][name]['prepared_split_sha256']), name
        old = {(row["dataset"], row["language"], row["seed"]): row for split in previous["split_reports"]
               for row in split["results"] if row["engine"] == "libtree"}
        for split in splits:
            for row in split["results"]:
                if row["engine"] == "libtree":
                    before = old[(row["dataset"], row["language"], row["seed"])]
                    assert before["metrics"] == row["metrics"], row["dataset"]
    ranking = []
    for name in sorted(names):
        for language in ("cpp", "python"):
            for field in ("fit_seconds", "predict_seconds"):
                timings = {engine: median(r[field] for r in groups[(name, language, engine)])
                           for engine in sorted(engines)}
                ranking.append({"dataset": name, "language": language, "measurement": field,
                                "median_seconds": timings,
                                "libtree_rank": 1 + sum(t < timings["libtree"] for t in timings.values())})
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--require-top1", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    previous = json.loads(args.previous.read_text()) if args.previous else None
    ranking = validate(data, previous, args.check_sources)
    wins = sum(row["libtree_rank"] == 1 for row in ranking)
    result = {"scope": "Pooled medians on recorded datasets, parameters, hardware and thread count",
              "wins": wins, "groups": len(ranking), "ranking": ranking,
              "previous_quality_identical": previous is not None,
              "source_hashes_checked": args.check_sources}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"LibTree fastest in {wins}/{len(ranking)} dataset/interface/measurement groups")
    if args.require_top1 and wins != len(ranking):
        raise SystemExit("Not all groups are fastest; see the raw measurements")


if __name__ == "__main__":
    main()
