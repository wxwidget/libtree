# CPU feature engineering and multiple-view recombination

[中文](CPU_MULTIVIEW.zh-CN.md) · [Earlier CPU screening](CPU_STUDY.md) · [Beginner guide](BEGINNER_GUIDE.md)

This experiment reuses the **full-base CPU experts** rather than retraining
another broad pool on only 200k rows. No GPU or external embeddings are used.
The goal is a better accuracy/cost tradeoff through product features and diverse
calibration. It is not a first-place claim or a new Kaggle score. The historical
GPU-assisted local result **0.380907** remains preserved. Actual scored practice
best remains **0.39199 private / 0.39110 public** until a new receipt exists.

## Measured outcome — 18 October 2026

[Results and environment](cpu-multiview-results.json) link the full
[first-stage evidence](cpu-multiview-001.json) and
[multiscale continuation](cpu-multiscale-001.json), including configurations,
weights, learning roles, input/source hashes and raw stage times.

| CPU system | Selection RMSLE | Reused confirmation RMSLE |
| --- | ---: | ---: |
| Earlier full-data reference | 0.395662 | 0.394153 |
| Previous 200k-row extension | 0.395560 | 0.394065 |
| Decomposition + product views | 0.394410 | 0.393006 |
| **Multiscale global/parent views** | **0.394212** | **0.392846** |

Final confirmation improves **0.332%** relative to the CPU reference. The
absolute delta is −0.00130773; a 300-replicate conditional paired row bootstrap
gives [−0.00148431, −0.00112823]. This is incremental local progress, with the
validation limitations below. It remains worse than the historical GPU-assisted
0.380907 result and has not been submitted to Kaggle.

The two CPU workers took **218.09 + 151.76 = 369.85 seconds**. Maximum individual
child RSS was **5,029.60 MiB**, not summed concurrent memory. Original expert
training is excluded, so this is not a 370-second fresh complete model training
claim. Some local tests overlapped stage timers; they are observations, not
controlled speed rankings. Training has one execution per configuration. No
median, whole-system speedup, scaling ratio or serving latency is claimed.

### Which experiments helped or failed?

The internal fusion diagnostic selects **simplex**, not the more flexible
methods: RMSLE 0.392315 simplex, 0.392703 affine, 0.392794 category-context.
The continuation improves the same simplex diagnostic from 0.392315 to
0.392190, exceeding the predeclared 1e-5 threshold. Its promotion was fixed
before confirmation labels were loaded.

The initial pool has 28 outer columns; the final pool has 42, with 12 weights
above 1e-8. These are prediction branches, not independent fitted models.
Largest branches are product-view Huber LightGBM 20.34%, hierarchical LightGBM
63 leaves 19.97%, hierarchical LibTree depth 5 17.96%, global-multiscale
LightGBM 11.61%, hierarchical Huber 9.76%, reference pipeline 4.90%,
hierarchical XGBoost 4.09%, and the cached stemmed Huber 3.71%. JSON preserves
every weight and the shared upstream dependencies.

LightGBM 63 leaves is **weaker alone** than 31 leaves (0.393598 versus 0.393206
confirmation), yet receives substantial ensemble weight. Complementarity is
useful; the strongest individual score is not the only selection criterion.
By contrast, extra affine/group flexibility did not improve the diagnostic.

| Feature/family ablation | Selection RMSLE | Confirmation RMSLE |
| --- | ---: | ---: |
| First stage, all views | 0.394410 | 0.393006 |
| Remove all product-statistic correctors | 0.395122 | 0.393569 |
| Remove added lexical/product correctors | 0.395189 | 0.393598 |
| Remove the two cached small networks | 0.394532 | 0.393116 |
| Final pool, remove hierarchical correctors | 0.394333 | 0.392976 |
| Final pool, all views | **0.394212** | **0.392846** |

Each removal refits blend weights on selection before audit. The lexical/product
removal retains decomposition/disagreement features in the basic correctors;
the raw JSON key `without_all_new_features` refers to the added rich views,
not removal of every new meta feature. Overlapping group gains are not additive.

A matched depth-3 LibTree control improves confirmation from **0.393839 basic →
0.393797 with attributes → 0.393539 with product statistics**, holding the
calibration rows/target and tree parameters fixed. With all 148 features, the
standalone correction pipelines give 0.393282 (LibTree depth 5), 0.393206
(LightGBM 31), 0.393342 (LightGBM Huber), and 0.393307 (XGBoost depth 4).
Capacities differ and margins are small; this does not establish general
superiority of one tree library. No native C++ learner changed in this round.

The original complete CPU branch reconstructs with maximum absolute log error
**1.78e-15** on all 296,509 calibration/selection/confirmation rows. That verifies
reuse of existing fitted artifacts, not a fresh neural training replay.
The first phase launched before a later floating-point-bound guard and the
hierarchical helper were added; its exact launch-source hashes are preserved.
The continuation's reported source hashes match the implementation files.

Local verification passes **9 tests / 433 statements / 3 modules, zero missed**.
All four [implementation CI jobs](https://github.com/wxwidget/libtree/actions/runs/39140908087)
passed, adding these modules to the existing 100% statement gate. Partial files, an unavailable CI result or a local
accuracy improvement never substitute for a complete Kaggle scoring receipt.

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

# Optional continuation: subtract actual elapsed time from your total budget.
# Our first run took 218 s, leaving >600 s of its 900 s budget.
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
