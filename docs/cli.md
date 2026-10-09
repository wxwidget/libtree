# Parameterized command-line GBDT

[中文](cli.zh-CN.md)

`xgbt` runs **LibTree's GBDT**, not the official XGBoost executable.
It is a native C++17 program built with the library; no Python runtime is needed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/xgbt --help
./build/xgbt train --train examples/csv/train.csv --test examples/csv/test.csv \
  --header --objective binary --trees 80 --depth 3 --bins 32 \
  --min-leaf 1 --learning-rate 0.2 --l2 0 --min-gain 0 \
  --model build/example.lt --output build/predictions.csv
./build/xgbt predict --model build/example.lt --data examples/csv/test.csv \
  --header --output build/loaded-predictions.csv
```

These tiny CSV files are synthetic examples, not benchmark datasets. Inputs are
numeric comma-separated CSV, with an optional header. The **first training
column is the label**; test/predict files contain only feature columns in the
same order. `--header` skips the first line of both input files. Empty feature
fields or `nan` mean missing; blank lines are ignored. Quoted fields and automatic
categorical encoding are unsupported. Reject infinities, invalid numbers,
inconsistent columns and non-binary labels when using the binary objective.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--objective` | `regression` | `regression` or `binary` |
| `--trees` / `--num-trees` | 100 | Boosting rounds |
| `--depth` / `--max-depth` | 4 | Path depth, 0–20 |
| `--bins` / `--max-bins` | 64 | Quantile bins, 2–65535 |
| `--min-leaf` / `--min-samples-leaf` | 5 | Minimum child sample count |
| `--learning-rate` / `--rate` | 0.1 | Shrinkage in (0,1] |
| `--l2` | 1 | Nonnegative leaf regularization |
| `--min-gain` | 0 | Nonnegative required split gain |

`train` requires `--train`; `--test`, `--model` and `--output` are optional.
Without `--test`, predictions use the training features. Without `--model`,
training works entirely in memory. Binary output contains P(class=1), not labels.
Stdout is one JSON record with fitted tree count, predicted row count, fit and
predict seconds and final training loss. CSV loading and model/output writes
are outside those timers. Training loss is in-sample, not a test metric.

`predict` requires `--model`, `--data` and `--output`. Hyperparameters and objective
come from the saved model; reject training options rather than silently ignoring
them. Invalid commands return a nonzero status and a message on stderr. Explicit
output paths are replaced atomically after successful validation; input/output
paths must differ. Existing outputs survive malformed data/models.

The model is a versioned, portable text format (`LIBTREE_GBDT 1`) with round-trip
float/double precision. Load checks bounds, finiteness, feature IDs, tree topology,
missing-direction flags, depth and trailing input. Loading failure leaves an
existing fitted C++ model intact. Safety limits are 10k trees and 1M total nodes;
save refuses larger models. Old binary models are incompatible. Training-loss
history is not serialized. C++ exposes `SaveModel(std::ostream&)` and
`LoadModel(std::istream&)`; Python persistence methods are not yet exposed.

Optional installation: `cmake --install build --prefix /your/writable/prefix`,
then add `/your/writable/prefix/bin` to PATH. No system-wide installation is needed.

## Three-way benchmark with custom parameters

```sh
python -m pip install '.[benchmark]'
PYTHONPATH=python python benchmarks/kaggle.py \
  --engines libtree xgboost lightgbm --seeds 42 2024 2026 --repeats 3 \
  --trees 100 --depth 4 --bins 64 --min-leaf 5 --rate 0.1 --l2 1 --min-gain 0 \
  --output benchmarks/comparison-results.json
python benchmarks/kaggle_report.py --input benchmarks/comparison-results.json --prefix comparison
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/html_report.py
```

Both C++ C-API calls and Python estimator calls use the selected parameters.
The benchmark restricts depth to 1–20 because LightGBM does not support a
root-only tree via this interface. It uses one thread and float32 dense data.
LightGBM is leaf-wise with `num_leaves=2^depth`; XGBoost uses default Hessian
weight 1 rather than an exact minimum sample count. See the report for the
resulting comparability limits. The benchmark is CPU-only, with no test-set
hyperparameter tuning.
