# CPU feature engineering and multiple-view recombination

[中文](CPU_MULTIVIEW.zh-CN.md) · [Earlier CPU screening](CPU_STUDY.md) · [Beginner guide](BEGINNER_GUIDE.md)

This experiment reuses the **full-base CPU experts** rather than retraining
another broad pool on only 200k rows. No GPU or external embeddings are used.
The goal is a better accuracy/cost tradeoff through product features and diverse
calibration. It is not a first-place claim or a new Kaggle score. The historical
GPU-assisted local result **0.380907** remains preserved. Actual scored practice
best remains **0.39199 private / 0.39110 public** until a new receipt exists.

## Features and parameters

Base-only target statistics describe five product views: exact normalized title,
sorted informative title words, the first three informative words, brand × full
category × condition, and brand/category × numeric model/capacity tokens. Simple
sales words such as `new` and `free shipping` are removed from identity views;
condition/accessory evidence remains in separate lexical attributes. These
approximate keys are not verified product IDs.

Each view supplies smoothed log-price mean, log support and dispersion. With
base group sum `S`, support `n`, global mean `μ`, and fixed smoothing `α`:

```text
global view: mean = (S + α μ) / (n + α)
parent mean = (S_parent + 20 μ) / (n_parent + 20)
hierarchical view: mean = (S + α parent_mean) / (n + α)
```

Unknown product keys back off to their fitted brand/category parent; unknown
parents use the global base prior. Dispersion follows the analogous second-
moment calculation. No query labels or query-population statistics are fitted.
This adapts [Micci-Barreca's target smoothing][encoding]; it is not a newly
derived Bayesian estimator. Sufficient statistics can be reused for different
strengths without refitting text vocabularies or reading additional labels.

The earlier CPU pipeline is decomposed into Ridge, field-aware FM and four
neural predictions. Listing preparation and CSR extraction run once per window.
The original corrected branch is reconstructed independently and checked against
the saved reference. New correctors **exclude that entire corrected branch**
from their input and residual target because its old native correction already
fitted calibration labels. Existing word/category statistics fit base rows only.
Two previously fitted 200k-row stemmed experts may enter only the outer blend.

| Calibration view | Inputs | Dimensions in this run |
| --- | --- | ---: |
| Basic | Original metadata, six base outputs, five disagreement summaries | 40 |
| Attributes | Basic + lexical product/condition flags | 58 |
| Product | Attributes + five smoothed statistics triples, global α=20 | 73 |
| Global multiscale | Product + global α=5/100 triples | 103 |
| Hierarchical multiscale | Global multiscale + parent α=5/20/100 triples | 148 |

The initial nine correctors compare LibTree depths 3/5, LightGBM 31 leaves,
LightGBM Huber and XGBoost depth 4. The continuation adds seven predefined
correctors, including a matched global-only LightGBM control, hierarchical
LightGBM 31/63 leaves and the same native/XGBoost capacities. Each supplies
correction strengths .5 and 1 without another fit. These are parameter/view
variants, not independent training repetitions or equal-capacity speed tests.

## Fusion and validation

All blends directly minimize clipped log-price squared error, aligned with
RMSLE, plus shrinkage toward the existing reference. Compare three strategies:

1. Nonnegative sum-one weights and bounded intercept, penalty 1e-4.
2. Nonnegative affine weights, sum in [.8, 1.2], intercept ±.25, penalty .001.
3. Large-category affine weights, minimum 2,000 fit rows, shrinking toward global
   weights with penalty .01; small or unseen groups fall back to global.

These coefficient bounds are in **log-price model space**, not ±20% bounds on
predicted dollars. More flexibility is a candidate, not an assumed advantage.

Original grouped windows remain base 1,186,026, calibration 74,129, selection
74,126 and reused confirmation 148,254. A grouped 50/50 split inside selection
(seed 193, 37,062/37,064 rows) compares fusion strategies. Within 1e-5 of the
best diagnostic score, prefer the simpler strategy. Then refit its weights on
all selection rows and freeze before reading confirmation labels.

This inner split is **diagnostic**, not independent of previous research:
upstream expert checkpoints/configurations used selection data in earlier work.
Confirmation has also been inspected before. Bootstrap intervals are conditional
on these fitted models and cannot remove selection bias or estimate seed-to-seed
training variability. Stacking follows [Wolpert][stacking].

The continuation keeps the selection-chosen strategy family. Its richer pool
is promoted only if the same inner diagnostic improves by more than 1e-5. Both
proposal and retained choice are frozen before audit. Ablation of hierarchical
views removes every corrector depending on them and retains a separately fitted
global-only control; it does not merely hide raw feature columns after training.

## Reproduce

First obtain the trusted CPU reference and grouped data using
[the earlier instructions](CPU_STUDY.md). Full expert refitting is a separate
cost; this experiment times reuse, statistics and new calibration only.

```bash
python -m venv .venv
.venv/bin/pip install '.[mercari,benchmark]' pytest pytest-cov

.venv/bin/python -m benchmarks.mercari_multiview_study \
  --data-dir benchmarks/runs/mercari/cpu-data \
  --baseline benchmarks/runs/mercari/optimized-audit-001/model.pkl \
  --small-experts benchmarks/runs/mercari/cpu-budget-study \
  --output-dir benchmarks/runs/mercari/multiview \
  --threads 4 --tree-rounds 300 --max-seconds 900

# Optional continuation. Its actual experiment uses the remaining budget;
# total allowed time for the two workers stays below 900 seconds.
.venv/bin/python -m benchmarks.mercari_multiview_study \
  --data-dir benchmarks/runs/mercari/cpu-data \
  --baseline benchmarks/runs/mercari/optimized-audit-001/model.pkl \
  --refine-from benchmarks/runs/mercari/multiview \
  --output-dir benchmarks/runs/mercari/multiscale \
  --threads 4 --tree-rounds 300 --max-seconds 600
```

Omit `--small-experts` to exclude the two small cached networks. Output folders
must be new. The launcher hides GPUs, caps BLAS/TensorFlow/Numba threads and
terminates the worker at the hard wall-time limit. It never submits automatically.
The continuation verifies the prior baseline hash, ordered IDs and input hashes.
Saved product maps contain base-derived label aggregates; use only trusted
pickle files. Data, models, pools and raw logs stay Git-ignored.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_DISABLE_JIT=1 \
.venv/bin/python -m pytest tests/test_mercari_multiview.py \
  tests/test_mercari_multiview_study.py --cov=benchmarks.mercari_multiview \
  --cov=benchmarks.mercari_multiview_study --cov=benchmarks.mercari_multiscale \
  --cov-report=term-missing --cov-fail-under=100
```

Coverage is a statement gate for these modules, not branch/whole-repository
coverage. Tests include independent pooled-statistic oracles, parent fallback,
numeric category IDs, floating-point mass bounds, actual fitted expert replay,
native/competitor calibration, hash/label isolation, frozen-before-audit errors
and launcher CPU/time-limit behavior.

[encoding]: https://doi.org/10.1145/507533.507538
[stacking]: https://doi.org/10.1016/S0893-6080(05)80023-1
