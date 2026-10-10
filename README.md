# LibTree

**A compact, deterministic C++17 histogram GBDT library** with native C++, C ABI,
Python/NumPy, and command-line interfaces. It supports squared-error regression,
binary logistic classification, dense row-major features, and learned routing for
missing (`NaN`) values.

This README is the standalone English project guide. The [Chinese README](README.zh-CN.md)
is maintained separately. New to machine learning or decision trees? Start with
the interactive, offline [GBDT Learning Guide](docs/LEARNING_GUIDE.html). For a
research-style account of datasets, competitors, methods, results, and ablations,
read the [Experimental Report](benchmarks/EXPERIMENTS.md).

## At a glance

- One C++ implementation powers both the C++ and Python APIs; Python does not
  reimplement the learner.
- C++17, CMake, and the standard library are sufficient to build the native
  library. Python needs NumPy and a C++17 compiler.
- CPU-only and dense-input focused. Defaults to one thread; parallelism is
  deterministic and selected by workload size.
- No claim of universal superiority over XGBoost or LightGBM. Measured outcomes
  vary by dataset, interface, thread budget, and hardware.

## Quick start

### Build and run C++

Requirements: Linux, GCC or Clang with C++17 support, CMake 3.16+, and Make.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/libtree_example
./build/xgbt --help
```

Minimal native example:

```cpp
#include "libtree/gbdt.h"
#include <vector>

std::vector<float> x{0, 1, 2, 3};  // 4 rows × 1 feature, row-major
std::vector<float> y{0, 0, 1, 1};

libtree::Parameters options;
options.min_samples_leaf = 1;
libtree::Gbdt model(options);
model.Fit({x.data(), 4, 1}, y);
const std::vector<float> prediction = model.Predict({x.data(), 4, 1});
```

Link a C++ application with `build/libtree.a`, `-Iinclude`, `-std=c++17`, and
`-pthread`. `MatrixView` borrows the input buffer. Keep that buffer alive for the
duration of each call; the fitted model does not retain it.

### Install and use Python

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

The install builds the native extension and installs NumPy. Example:

```python
import numpy as np
from libtree import GBDTClassifier, GBDTRegressor

x = np.array([[0], [1], [2], [3]], dtype=np.float32)
y = np.array([0, 0, 1, 1], dtype=np.float32)

with GBDTRegressor(min_samples_leaf=1) as model:
    model.fit(x, y)
    print(model.predict(x))

with GBDTClassifier(min_samples_leaf=1) as model:
    model.fit(x, y)
    print(model.predict_proba(x))  # columns: probability of class 0, class 1
```

Use two-dimensional dense features and one-dimensional labels. Classification
labels must be `0` or `1`. Convert categorical features to numeric features;
fit encoders on training data only. `NaN` means missing. Infinity and invalid
shapes are rejected. Contiguous `float32` input avoids a conversion copy.

### Command line

`xgbt` trains and predicts from numeric CSV files, writes versioned model files,
and exposes tree/training parameters. Start with:

```sh
./build/xgbt --help
```

See the [CLI guide](docs/cli.md) for input format, options, missing values, and
model lifecycle.

## Concepts and learning path

Gradient boosted decision trees add many small trees in sequence. Each new tree
corrects part of the current model's error. LibTree bins continuous values into
histograms, scores candidate splits with first- and second-order loss
derivatives, and learns which side receives missing values. The
[interactive beginner guide](docs/LEARNING_GUIDE.html) introduces these ideas
with diagrams, a split slider, runnable-style examples, and short checks. The
[algorithm and design notes](docs/design.md) give equations and implementation
details.

Important parameters:

| Python | C++ | Default | What it controls |
| --- | --- | ---: | --- |
| `n_estimators` | `num_trees` | 100 | Number of sequential boosting rounds |
| `max_depth` | `max_depth` | 4 | Maximum split levels per tree; 0 gives a root-only tree |
| `max_bins` | `max_bins` | 64 | Histogram resolution per feature; 2–65,535 |
| `min_samples_leaf` | `min_samples_leaf` | 5 | Minimum training rows allowed in a child |
| `learning_rate` | `learning_rate` | 0.1 | Shrinkage applied to each tree's prediction |
| `reg_lambda` | `l2` | 1.0 | L2 penalty on leaf values |
| `min_gain` | `min_gain` | 0.0 | Minimum gain required to make a split |
| `n_jobs` | `num_threads` | 1 | Runtime CPU thread budget, including caller |

There is no built-in early stopping. Use a validation set to choose model size;
keep the final test set untouched. Training loss is a diagnostic, not a measure
of generalization. The learning guide covers this distinction and data leakage.

## Optimization changelog

This is a summary of measured and tested implementation milestones. The
[detailed optimization report](benchmarks/OPTIMIZATION_REPORT.zh-CN.md) and
[parallel scaling report](benchmarks/ADAPTIVE_REPORT.zh-CN.md) contain raw data,
platform details, and caveats.

| Area | Change | Reason / evidence |
| --- | --- | --- |
| Training layout | Quantize features once; keep compact column-major `uint16` bins | Reduce repeated comparisons and histogram working-set size |
| Histogram construction | Build the smaller child's histogram and subtract it from its parent | Avoid rebuilding both child histograms |
| Low-cardinality features | Skip dominant bins, use masked sums for two-valued columns, and reuse constant child histograms | Reduce scattered writes and repeated scans |
| Wide matrices | Reuse row-major bin indices and derivative values across features | Improve locality on wide, mostly continuous tables |
| Split search | Serial and layer-parallel builders share the pure `FindBestSplit` calculation | Keep missing routing, minimum-leaf rules, and gain scoring in one place |
| Large-tree construction | Schedule ready nodes by depth when size/depth/histogram-memory heuristics allow | Expose independent nodes to CPU workers while preserving layer dependencies |
| Prediction | Compile shallow trees into contiguous prediction arrays; process rows in batches | Improve traversal locality and parallel row throughput |
| Output allocation | Add `PredictInto` and direct C/Python output paths | Avoid a temporary result allocation/copy when caller owns the buffer |
| Parallel runtime | Reuse a bounded C++ thread pool; keep small workloads serial | Avoid per-round worker creation and excessive nested parallelism |

Model ordering and arithmetic remain deterministic across tested thread counts.
The depth-wise builder restores the established depth-first model ordering so
serialized output remains compatible within this implementation. These
optimizations do not make every workload faster; small tasks can be dominated
by setup, dispatch, and synchronization.

## Competitor comparison: measured snapshot

The table below reports **arithmetic mean training and prediction time in ms**
over raw repetitions of a five-dataset, three-engine comparison. For each
interface, values are averaged over the five datasets; the detailed report also
shows per-dataset results and variability. Parameters were fixed at 100 trees,
depth 4, 64 bins, learning rate 0.1, L2 1, and one thread. The measurements are
a historical snapshot on an AMD EPYC 9V74 cloud container; they are not a
universal ranking.

| Interface | Engine | Mean fit ms ↓ | Mean predict ms ↓ | Relative to LibTree fit / predict |
| --- | --- | ---: | ---: | ---: |
| C++ | LibTree | 14.336 | 0.363 | 1.00× / 1.00× |
| C++ | XGBoost 3.4.1 | 21.995 | 0.811 | 1.53× / 2.23× |
| C++ | LightGBM 4.7.0 | 19.947 | 1.749 | 1.39× / 4.82× |
| Python | LibTree | 14.577 | 0.496 | 1.00× / 1.00× |
| Python | XGBoost 3.4.1 | 29.287 | 1.590 | 2.01× / 3.21× |
| Python | LightGBM 4.7.0 | 33.016 | 2.713 | 2.27× / 5.47× |

On that snapshot, LibTree was faster on average in the measured suite; individual
dataset outcomes differ. For example, XGBoost scored a slightly higher
insurance R², and Telco AUC was nearly tied. Histogram construction, leaf rules,
and growth strategies differ between libraries, so matching high-level
parameters does not make the algorithms identical. See the
[full experimental report](benchmarks/EXPERIMENTS.md) for task selection,
quality metrics, train/test gaps, thread ablation, the paired before/after study,
and open experimental gaps. Do not interpret these local measurements as
leaderboard placement or production performance guarantees.

### Current scale evidence

The latest same-split experiment measures four datasets at 1/2/4 threads, using
three seeds and three timing repeats. On the three medium/large tasks, mean
1→4 training scaling is 1.32× for LibTree C++, 1.02× for XGBoost C++, and 2.04×
for LightGBM C++; through Python it is 1.35×, 0.95×, and 1.97× respectively.
LibTree currently beats XGBoost on this scaling measure, but does not beat
LightGBM. The 100k-row Friedman workload scales 1.51×/1.54× for LibTree,
1.04×/0.97× for XGBoost, and 2.07×/2.27× for LightGBM (C++/Python).

The requested “scale better than both competitors” criterion is therefore not
met. The [benchmark README](benchmarks/README.md) records the full per-workload
table and how to reproduce it; [the raw report](benchmarks/PARALLEL_REPORT.zh-CN.md)
includes means, standard deviations, observed ranges, and every run.

## Experiments and ablations

The benchmark suite deliberately spans small real tables, a medium-sized
customer dataset, a financial marketing classification task, and generated
20k/100k-row workloads. Evaluation uses fixed 75/25 holdouts, three seeds for
the multi-dataset study, train-only categorical encoding, deterministic
repeats, and prediction parity checks across C++ and Python. Bank `duration` is
excluded because it is observed after the marketing call outcome.

There are two different kinds of ablation evidence in the repository:

1. **Thread-budget ablation:** 1 vs 2 vs 4 threads on identical prepared splits,
   including fit and prediction time. The observed gain depends on workload
   size and interface.
2. **Optimization-bundle paired study:** original baseline vs the combined
   optimization set on identical inputs, nine randomized-order repetitions per
   input, with exact float32 predictions. This is an end-to-end bundle
   comparison, not an isolated ablation of each kernel.

Individual kernel ablations (for example, histogram subtraction on/off or
packed prediction layout on/off) are not yet available. The experimental report
states this gap and proposes how to measure each contribution without changing
quality or data splits.

## Learning guide and references

- [Interactive GBDT Learning Guide (offline HTML)](docs/LEARNING_GUIDE.html)
- [Algorithm and architecture](docs/design.md)
- [Command-line guide](docs/cli.md)
- [Experimental report](benchmarks/EXPERIMENTS.md)
- [Optimization benchmark](benchmarks/OPTIMIZATION_REPORT.zh-CN.md)
- [Scale and finance benchmark](benchmarks/ADAPTIVE_REPORT.zh-CN.md)
- [Validation history](docs/validation.md)

## Test and contribute

```sh
make test                 # CTest and Python behavior/performance/ownership tests
make sanitize             # ASan, UBSan, LSan
make thread-sanitize      # ThreadSanitizer
make coverage             # C++ and Python line coverage gates
```

For an algorithm change, add a behavioral test first, establish the failure,
make the smallest implementation change, then rerun the targeted test and full
suite. Benchmark performance changes with a Release build and preserve the
dataset hash, split, parameters, thread count, raw repeats, and quality metrics.
Read [design.md](docs/design.md) before changing ownership, model format, or
parallel scheduling.

## Scope and compatibility

The supported learner is dense CPU regression and binary classification.
Multiclass, ranking, sparse inputs, GPU/distributed training, sample weights,
Python model persistence, and early stopping are not implemented. Infinity is
invalid; `NaN` features are supported. The current project uses C++ exceptions
for C++ input errors; the C ABI catches them and returns status codes. This is
an explicit deviation from Google's internal C++ style guide, described in the
[design notes](docs/design.md).

The rewrite intentionally removed unsafe raw-pointer ownership and incomplete
legacy forest/entropy paths. The old CLI, headers, and model files are not
compatible; retrain using this API. Linux is the validated platform. XGBoost
and LightGBM are optional benchmark dependencies, not runtime dependencies.
