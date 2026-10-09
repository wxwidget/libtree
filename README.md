# LibTree

[中文](README.zh-CN.md) · [Algorithm and design](docs/design.md) · [Benchmarks](benchmarks/README.md)

A small, deterministic **C++17 histogram GBDT** with a NumPy/Python interface.
It implements squared-error regression and binary logistic classification,
including learned routing for NaN features. Python runs the same native model;
it does not reimplement training.

## Start here

On Linux, install GCC (or Clang), Python 3.9+, and Make. A minimal C++ build needs
CMake 3.16+ and no third-party C++ library. From this checkout:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[dev,benchmark]'
make test
./build/libtree_example
```

`.venv/` is local configuration. On the prepared cloud machine, use
`. /workspace/libtree-venv/bin/activate` and `cd /workspace/libtree` instead.
For Python use alone, `python -m pip install .` builds a native wheel and installs
NumPy. It requires a C++17 compiler. Linux is the currently validated platform;
Windows wheels are not provided. XGBoost and LightGBM are needed only for comparisons.

```python
import numpy as np
from libtree import GBDTRegressor, GBDTClassifier

x = np.array([[0], [1], [2], [3]], dtype=np.float32)
y = np.array([0, 0, 1, 1], dtype=np.float32)
with GBDTRegressor(min_samples_leaf=1) as model:
    model.fit(x, y)
    print(model.predict(x))
    print(model.training_loss_[-1])

classifier = GBDTClassifier(min_samples_leaf=1).fit(x, y)
print(classifier.predict_proba(x))  # columns: P(0), P(1)
classifier.close()
```

Inputs are dense `(samples, features)` arrays. Labels are one-dimensional;
classification labels must be 0 or 1. Categorical columns need training-only
encoding (see the benchmark script). Float32 contiguous arrays avoid conversion
copies. Infinity and invalid shapes are rejected; NaN is a supported feature.
`fit` replaces the previous model and leaves inputs unchanged. `close` releases
native memory; automatic finalization also works. `get_params`, `set_params`,
`score` and sklearn `clone` are supported. This is not a full sklearn estimator
protocol implementation; advanced metadata routing is not supported.

## Command-line training and prediction

The native `xgbt` program supports train/predict, CSV inputs, model files and all
GBDT hyperparameters. See [the command guide](docs/cli.md) and run
`./build/xgbt --help`. [Three-way comparison](benchmarks/COMPARISON_REPORT.zh-CN.md)
adds LightGBM alongside XGBoost for both C++ and Python.

The [latest optimization evidence](benchmarks/OPTIMIZATION_REPORT.zh-CN.md)
records 450 measurements: LibTree has the lowest pooled median training and
prediction time in all 10 dataset/interface groups of the five-dataset,
single-thread suite. Same-split quality metrics are unchanged from the previous
implementation. These rankings describe this recorded workload and hardware.
`PredictInto(view, output)` writes directly into a separate caller-owned float
buffer; allocating `Predict` remains available.

## CPU parallelism

Set Python `GBDTRegressor(n_jobs=4)` / `GBDTClassifier(n_jobs=4)`, C++
`Parameters::num_threads = 4`, or CLI `--threads 4` for train and predict.
The default is 1; the allowed range is 1–256, including the calling thread.
The runtime reuses standard C++ worker threads, parallelizes independent feature
histograms and prediction rows, and keeps small tasks serial. Boosting rounds
and tree construction preserve their dependency order. Model bytes, training
losses and predictions are identical across tested thread counts.

`model.set_params(n_jobs=2)` changes Python execution on a fitted model; C++
uses `SetNumThreads(2)`. Thread count is a runtime preference and is not saved
in the model format. Concurrent prediction is supported; do not change threads,
fit or close the same model concurrently. Prediction calls sharing one model
share a bounded pool and serialize their parallel dispatch. Choose thread counts
within your CPU quota, especially when running multiple models or outer CV jobs.

See the [equal-thread benchmark](benchmarks/PARALLEL_REPORT.zh-CN.md) and
[interactive scaling report](benchmarks/PARALLEL_REPORT.html). Reproduce:

```sh
PYTHONPATH=python python benchmarks/parallel.py --threads 1 2 4 --seeds 42 2024 2026 --repeats 3
python benchmarks/parallel_report.py
make thread-sanitize
```

## C++ in five minutes

Read [examples/train.cc](examples/train.cc), or build only the library:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

```cpp
#include "libtree/gbdt.h"
#include <vector>

std::vector<float> x{0, 1, 2, 3};
std::vector<float> y{0, 0, 1, 1};
libtree::Parameters p;
p.min_samples_leaf = 1;
libtree::Gbdt model(p);
model.Fit({x.data(), 4, 1}, y);
auto prediction = model.Predict({x.data(), 4, 1});
```

Link with `build/libtree.a` and add `-Iinclude -std=c++17 -pthread`. `MatrixView` is a
non-owning row-major view: its buffer must hold at least `rows * cols` floats.
Fit copies neither the feature matrix nor the labels into the trained model.
Concurrent prediction is safe on an immutable fitted C++ model. Do not fit or
close the same model concurrently. Invalid input throws standard exceptions;
the C ABI converts exceptions to status codes and a thread-local error string.

## Parameters and how to learn

| Parameter (Python / C++) | Default | Meaning |
| --- | ---: | --- |
| `n_estimators` / `num_trees` | 100 | Number of boosting rounds |
| `max_depth` | 4 | Maximum splits on a path; 0 means root only |
| `max_bins` | 64 | Quantile histogram bins per feature, 2–65535 |
| `min_samples_leaf` | 5 | Minimum samples in each child leaf |
| `learning_rate` | 0.1 | Shrinkage, strictly positive and at most 1 |
| `reg_lambda` / `l2` | 1 | Nonnegative leaf L2 penalty |
| `min_gain` | 0 | Nonnegative required split improvement |

Start with the example, then read [the derivation](docs/design.md). Try changing
learning rate or tree depth and inspect training loss and held-out RMSE/AUC.
Evaluate on held-out data; lower training loss alone does not prove a better
model. The implementation has no early stopping: select rounds with validation
and keep the final test split untouched.

## Verify a change

```sh
make test       # native tests, Python behavior, performance and ownership guards
make sanitize   # ASan + UBSan + LSan, separate Debug build
make coverage   # native and Python line coverage must reach 100%, HTML artifacts
clang-format --dry-run --Werror include/libtree/*.h src/*.cc tests/*.cc examples/*.cc benchmarks/*.cc
PYTHONPATH=python python benchmarks/run.py --repeats 3
```

Use TDD: add a failing behavioral test, run it to establish the failure, implement
the smallest correction, and rerun the relevant suite before refactoring. Tests
include an independent exhaustive SSE oracle, NaNs, extreme values, deterministic
refits, bad shapes/labels/parameters, C ABI errors, Python packaging, ownership
cycles and a representative workload. Coverage and sanitizers complement these
tests; neither proves absence of every bug. CI runs the checks and a one-repeat
benchmark smoke run; checked-in timing reports use three isolated repeats.

## Comparison and scope

[Measured results](benchmarks/README.md) cover Titanic, medical insurance,
diabetes, breast cancer and a 20k-row Friedman dataset, for **both native C++ and
Python**, against XGBoost 3.4.1. Quality is close on these fixed splits; speed
varies by dataset, and XGBoost is faster on the larger synthetic workload. There
is no leaderboard submission or general claim of superiority.

This rewrite intentionally removes unsafe raw-pointer ownership, incomplete
forest/entropy branches, ad-hoc boosting weights and unvalidated binary model
serialization. The previous `train`/`classify` CLI, headers and serialized models
are incompatible; retrain via the new API. Original `data/` assets are retained
as historical data, but benchmarks use source-tracked datasets. Multiclass,
ranking, sparse input, GPU/distributed training, Python model persistence and sample
weights are outside this implementation. See [design choices and style
exceptions](docs/design.md) before contributing.

[Validation evidence / 验证记录](docs/validation.md)

[新增 Kaggle 五数据集、多划分测试报告](benchmarks/KAGGLE_REPORT.zh-CN.md)

[交互 HTML 图表报告](benchmarks/KAGGLE_REPORT.html)（离线查看，切换 C++ / Python，导出原始数据）
