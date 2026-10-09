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
                        "--repeats", str(args.repeats), "--output", str(path)], check=True)
        reports.append(json.loads(path.read_text()))
    aggregate = {
        "download_note": "Kaggle API download denied by network proxy (HTTP tunnel 403); public mirrors used. Official Kaggle byte identity not independently verified.",
        "seeds": args.seeds, "timing_repeats_per_split": args.repeats,
        "split_reports": reports,
    }
    path = ROOT / "benchmarks/kaggle-results.json"
    path.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
