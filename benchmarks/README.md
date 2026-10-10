# Benchmark Guide and Latest Results

This directory compares LibTree's C++ and Python interfaces with XGBoost and
LightGBM. Results below use local held-out measurements; they are not Kaggle
leaderboard scores. All raw repetitions, data fingerprints, package versions,
and source hashes are kept in [`parallel-results.json`](parallel-results.json).

## Latest measured scaling

Measured 2026-10-10 on AMD EPYC 9V74 (80-core host, four-core cgroup quota),
GCC 14.2, Python 3.12, XGBoost 3.4.1, and LightGBM 4.7.0. The test used four
datasets, three seeds, three timing repetitions, and 1/2/4 threads: 648 raw
timing records, each containing fit and prediction times. The entries below are training speedup ratios:
mean 1-thread fit time divided by mean 4-thread fit time.

| Workload | LibTree C++ | XGBoost C++ | LightGBM C++ | LibTree Python | XGBoost Python | LightGBM Python |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Insurance (small) | 0.96× | 1.01× | 0.45× | 0.99× | 1.10× | 0.60× |
| Bank Marketing (medium) | 1.28× | 0.99× | 1.99× | 1.30× | 0.86× | 1.95× |
| Friedman 20k (medium) | 1.17× | 1.03× | 2.07× | 1.21× | 1.04× | 1.69× |
| Friedman 100k (large) | 1.51× | 1.04× | 2.07× | 1.54× | 0.97× | 2.27× |
| Medium/large arithmetic mean | **1.32×** | 1.02× | **2.04×** | **1.35×** | 0.95× | **1.97×** |

**The requested scaling criterion is not met:** LibTree scales better than
XGBoost across these medium/large workloads, but LightGBM has a higher 1→4
training scaling factor in both APIs. The implementation and benchmark suite
remain useful for tracking the gap; the result is not presented as a win.
Small insurance timings are included as a control, but their few-millisecond
fit time is dominated by scheduling noise and is not a useful scaling target.
The full arithmetic means, standard deviations, ranges, quality metrics, and
all repetitions are in the [generated text report](PARALLEL_REPORT.zh-CN.md)
and [raw JSON](parallel-results.json).

## Run the benchmark

From the repository root, install the benchmark dependencies and build the
native targets:

```sh
python -m pip install -r benchmarks/requirements.txt
make all
```

Run the selected financial, medium, and large datasets at equal 1/2/4-thread
budgets. Each of the three seeds gets three independent timing repetitions per
engine and interface. The run order is reproducibly shuffled by dataset, seed,
and thread count.

```sh
PYTHONPATH=python python benchmarks/parallel.py \
  --datasets insurance bank_marketing friedman_20k friedman_100k \
  --threads 1 2 4 \
  --seeds 42 2024 2026 \
  --repeats 3 \
  --output benchmarks/parallel-results.json
```

The runner verifies finite predictions, baseline quality, repeat determinism,
and C++/Python prediction parity. It stores each raw fit and prediction timing;
preprocessing, data downloads, library imports, and process startup are outside
the timed region. LibTree and competitors receive the same main tree controls
and configured thread budget. Their histogram construction, missing-value
routing, and tree growth algorithms are not identical, so the comparison is a
matched workload rather than a claim of algorithmic equivalence.

Generate the text and interactive HTML reports from the raw file:

```sh
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/parallel_report.py --input benchmarks/parallel-results.json
```

The report generator checks raw run completeness, split hashes, source hashes,
and LibTree's quality parity across thread counts. It writes
[`PARALLEL_REPORT.zh-CN.md`](PARALLEL_REPORT.zh-CN.md),
[`PARALLEL_REPORT.html`](PARALLEL_REPORT.html), and
[`parallel-summary.json`](parallel-summary.json). The Markdown report is usable
without a browser; the HTML includes filters, charts, and raw CSV export.

To run one engine or a single dataset for a quick local check, use the lower
level runner:

```sh
PYTHONPATH=python python benchmarks/run.py \
  --suite all --datasets friedman_100k \
  --threads 4 --seed 42 --repeats 3 \
  --engines libtree xgboost lightgbm
```

## Method and workload selection

The checked-in scaling suite covers:

- `insurance`: smaller real-world tabular regression.
- `bank_marketing`: medium-size financial marketing classification.
- `friedman_20k`: medium synthetic regression with known signal.
- `friedman_100k`: larger synthetic regression used to expose scaling behavior.

The suite uses 75/25 train/test splits, stratified for classification. Encoders
are fitted on training rows only, missing values are retained, and no test-set
tuning is performed. Parameters are 100 trees, depth 4, 64 bins, learning rate
0.1, L2 1, and minimum leaf count 5 where supported. XGBoost's minimum Hessian
weight and LightGBM's leaf-wise growth differ from LibTree's constraints.
Small tasks can be slower with more threads because scheduling costs exceed the
parallel work; include those rows rather than hiding them.

Times are arithmetic means over all seed/repetition observations in each cell.
The report also gives standard deviation and the observed range. Scaling is a
ratio of means, not the mean of per-run ratios. All raw observations remain
available so readers can assess outliers and CPU scheduling noise. Runs execute
serially in fresh processes, with engine and interface order shuffled
deterministically. Native C++ XGBoost uses its public C API; Python timings use
the estimator API. Prediction timings include each interface's required input
conversion; native XGBoost includes test-matrix creation.

## Performance work in this snapshot

- Reuse the split search's left/right gradient, Hessian, and row-count totals
  instead of rescanning every node's rows to recompute child statistics.
- Update gradients and loss in fixed-size row chunks. Independent rows can run
  in parallel, while fixed chunk boundaries and ordered reduction preserve
  deterministic results across thread counts.
- Retain depth-layer scheduling for sufficiently large datasets and a bounded
  histogram-memory estimate; smaller workloads keep the recursive path.

The benchmark disproves the hypothesis that these changes close the scaling
gap to LightGBM. The next high-value work is profiling the large workload's
remaining serial fraction and measuring any follow-up against the same raw
suite before claiming an advantage.

### Change versus the previous LibTree snapshot

On the same three medium/large datasets and seeds, the current implementation
also reduced mean 4-thread LibTree fit time relative to the archived
[`adaptive-parallel-results.json`](adaptive-parallel-results.json):

| Dataset | C++ previous → current | C++ change | Python previous → current | Python change |
| --- | ---: | ---: | ---: | ---: |
| Bank Marketing | 243.5 → 205.7 ms | −15.5% | 254.8 → 202.5 ms | −20.5% |
| Friedman 20k | 95.0 → 82.9 ms | −12.8% | 100.8 → 82.6 ms | −18.0% |
| Friedman 100k | 372.8 → 325.6 ms | −12.6% | 393.1 → 336.6 ms | −14.4% |

This is an optimization-bundle comparison, not an isolated kernel ablation.
Inputs and seeds match, but the archived and current runs used different
randomized engine order and can experience different cloud scheduling. Treat
the timing deltas as directional evidence, not a paired significance test.

## Related files

- [Parallel implementation and scaling report](PARALLEL_REPORT.zh-CN.md)
- [Raw equal-thread measurements](parallel-results.json)
- [Adaptive scaling history](ADAPTIVE_REPORT.zh-CN.md)
- [Experiment design and ablations](EXPERIMENTS.md)
- [Benchmark runner](run.py) and [parallel suite runner](parallel.py)
