"""Run equal-thread native/Python comparisons with independent timing repeats."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['titanic', 'insurance', 'pima', 'telco', 'wine',
            'bank_marketing', 'friedman_20k', 'friedman_100k']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2024, 2026])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/parallel-results.json')
    args = parser.parse_args()
    if args.repeats < 3 or len(set(args.threads)) != len(args.threads) or not all(1 <= t <= 256 for t in args.threads):
        parser.error('at least three repeats and distinct valid thread counts required')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('distinct seeds required')
    jobs = [(seed, threads) for seed in args.seeds for threads in args.threads]
    random.Random(42).shuffle(jobs)
    directory = ROOT / 'benchmarks/runs/parallel'
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'python')}
    for seed, threads in jobs:
        output = directory / f'threads-{threads}-seed-{seed}.json'
        subprocess.run([sys.executable, str(ROOT / 'benchmarks/run.py'),
                        '--suite', 'all', '--datasets', *args.datasets,
                        '--threads', str(threads), '--seed', str(seed),
                        '--repeats', str(args.repeats), '--output', str(output)],
                       check=True, env=env)
        reports.append(json.loads(output.read_text()))
    result = {'threads': args.threads, 'seeds': args.seeds,
              'timing_repeats_per_split': args.repeats,
              'execution_order': jobs, 'split_reports': reports,
              'scope': 'Equal configured CPU thread budgets; serial independent processes. '
                       'Runtime imports excluded, training preprocessing included. '
                       'Small tasks can use fewer workers; no hyperparameter tuning.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
