"""Run several independent Kaggle-data splits without overwriting earlier results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 2026])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=str(ROOT / "benchmarks/kaggle-results.json"))
    parser.add_argument("--engines", nargs="+", choices=["libtree", "xgboost", "lightgbm"],
                        default=["libtree", "xgboost", "lightgbm"])
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--min-leaf", type=int, default=5)
    parser.add_argument("--rate", type=float, default=.1)
    parser.add_argument("--l2", type=float, default=1.)
    parser.add_argument("--min-gain", type=float, default=0.)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive repeats and distinct seeds required")
    directory = ROOT / "benchmarks/runs"
    directory.mkdir(exist_ok=True)
    reports = []
    for seed in args.seeds:
        path = directory / f"kaggle-seed-{seed}.json"
        subprocess.run([sys.executable, str(ROOT / "benchmarks/run.py"),
                        "--suite", "kaggle", "--seed", str(seed),
                        "--repeats", str(args.repeats), "--output", str(path),
                        "--engines", *args.engines, "--trees", str(args.trees),
                        "--depth", str(args.depth), "--bins", str(args.bins),
                        "--min-leaf", str(args.min_leaf), "--rate", str(args.rate),
                        "--l2", str(args.l2), "--min-gain", str(args.min_gain),
                        "--threads", str(args.threads)], check=True)
        reports.append(json.loads(path.read_text()))
    aggregate = {
        "download_note": "Kaggle API download denied by network proxy (HTTP tunnel 403); public mirrors used. Official Kaggle byte identity not independently verified.",
        "seeds": args.seeds, "timing_repeats_per_split": args.repeats,
        "split_reports": reports,
    }
    path = Path(args.output)
    path.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
