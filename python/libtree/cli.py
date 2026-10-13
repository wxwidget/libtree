"""Train a C5-style model and print rules or explain one sample."""
import argparse
import json

import numpy as np

from . import C5Classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('rules', 'explain'))
    parser.add_argument('--train', required=True, help='Numeric CSV: label in column 0')
    parser.add_argument('--header', action='store_true')
    parser.add_argument('--sample', help='Comma-separated numeric features; nan is missing')
    parser.add_argument('--feature-names', help='Comma-separated display labels')
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--min-leaf', type=int, default=5)
    parser.add_argument('--rate', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=.001)
    parser.add_argument('--min-gain', type=float, default=1e-9)
    parser.add_argument('--criterion', choices=('gain_ratio', 'entropy', 'gini'), default='gain_ratio')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    try:
        data = np.genfromtxt(args.train, delimiter=',', skip_header=int(args.header), ndmin=2)
        names = None if args.feature_names is None else args.feature_names.split(',')
        with C5Classifier(n_estimators=args.rounds, max_depth=args.depth, max_bins=args.bins,
                          min_samples_leaf=args.min_leaf, learning_rate=args.rate,
                          pruning_alpha=args.alpha, min_gain=args.min_gain,
                          criterion=args.criterion, n_jobs=args.threads) as model:
            model.fit(data[:, 1:], data[:, 0])
            if args.command == 'rules':
                print(model.dump_rules(args.format, names), end='')
            else:
                if args.sample is None:
                    raise ValueError('explain requires --sample')
                sample = np.array([float(v) for v in args.sample.split(',')], dtype='f4')
                explanation = model.explain(sample, names, args.format)
                print(json.dumps(explanation, ensure_ascii=False) if args.format == 'json'
                      else explanation, end='')
    except (ValueError, OSError) as error:
        parser.error(str(error))
