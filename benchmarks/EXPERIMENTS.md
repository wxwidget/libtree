# LibTree Experimental Report

**Purpose.** This report describes what was measured, why these tasks were
chosen, how comparisons were made, which ablations exist, and what remains
unknown. Results are local CPU measurements, not Kaggle leaderboard scores.

Related artifacts: [raw five-dataset results](optimized-results.json),
[paired optimization runs](optimization-ab-results.json), and
[scale/finance raw runs](adaptive-parallel-results.json). The Chinese reports
provide additional per-run tables: [optimization](OPTIMIZATION_REPORT.zh-CN.md)
and [adaptive parallelism](ADAPTIVE_REPORT.zh-CN.md).

## 1. Questions

The experiments address four practical questions:

1. Does the same LibTree model produce the same predictions through C++ and
   Python?
2. How do fit time, prediction time, and held-out quality compare with
   XGBoost and LightGBM under fixed, approximately matched settings?
3. Do the low-level training and inference optimizations preserve outputs while
   reducing time?
4. Does tree-level parallelism help at different sample sizes, and when does
   dispatch overhead outweigh it?

They do not answer whether LibTree wins on every problem, whether a benchmark
result transfers to production, or whether a financial signal is profitable.

## 2. Task and dataset selection

The suite intentionally ranges from small, familiar tables to larger synthetic
workloads. Real data covers regression and binary classification, numeric and
categorical features, missing values, and a finance-related prediction task.
Generated Friedman data provides repeatable scaling tests whose target function
is known.

| Dataset | Rows before split | Task | Why it is included | Important treatment |
| --- | ---: | --- | --- | --- |
| Titanic | 891 | Binary classification | Small mixed-type, missing-value task | Seven selected features; remove IDs, names, ticket and cabin |
| Insurance | 1,338 | Regression | Financial/medical-cost target with mixed features | Six non-target features; charges are the regression target |
| Pima diabetes | 768 | Binary classification | Small numeric task with substantial missingness | Invalid zero measurements become NaN; zero pregnancies is retained |
| Red wine quality | 1,359 unique rows | Regression | Small numeric quality task | Remove 240 exact duplicates before splitting |
| Telco churn | 7,043 | Binary classification | Medium mixed-type table with missing values | Remove customer ID; blank TotalCharges becomes NaN |
| Bank Marketing | 41,188 | Binary classification | Finance-related customer subscription task | Remove `duration` because it is known only after the call outcome; train-only category encoding |
| Friedman 20k | 20,000 | Regression | Controlled scaling and synthetic signal | 20 features, noise 1, seed 42 |
| Friedman 100k | 100,000 | Regression | Larger scaling and parallel-workload test | 20 features, noise 1, seed 42 |

The bank data is the UCI Bank Marketing data distributed through a pinned
public mirror and linked to a Kaggle dataset page. It is not a stock-price or
investment-return dataset. The insurance dataset is cost prediction, not a
financial market benchmark. No trading, risk-adjusted returns, or portfolio
backtest is included.

## 3. Evaluation protocol

### Splits and leakage controls

- Main comparison uses 75% training and 25% held-out test data with seeds 42,
  2024, and 2026. Binary tasks use stratified splits.
- Numeric/categorical preprocessing is prepared after splitting. One-hot
  encoders are fitted on the training fold only; unseen test categories are
  ignored. Missing numeric values remain NaN for the tree learner.
- Wine duplicates are removed before the split to avoid identical rows in both
  folds. Telco identifiers are removed. Bank `duration` is removed because it
  is a post-call variable and can leak the outcome.
- Hyperparameters are fixed in advance. The held-out test fold is not used to
  tune the model. Training metrics are calculated separately to inspect the
  train/test gap.

The bank split is random, not chronological or grouped by customer/campaign.
Correlated customers or campaign structure may cross the split. A time-based or
grouped validation study is required before using this dataset to estimate
deployment performance.

### Models and parameters

Main fixed settings: 100 boosting rounds, maximum depth 4, 64 bins, learning
rate 0.1, L2 1, minimum leaf count 5, minimum gain 0, CPU only. Thread tests
compare equal configured budgets of 1, 2, and 4 threads.

Competitors are XGBoost 3.4.1 and LightGBM 4.7.0. The main controls are matched
where possible, but the algorithms are not identical: XGBoost's default
`min_child_weight` is Hessian based; LibTree and LightGBM use a minimum row
count; LightGBM grows leaf-wise. Binning sketches and missing-value rules also
differ. No hyperparameter search was conducted for any engine.

### Metrics and timing

- Binary classification: ROC-AUC is the primary ranking metric; average
  precision, accuracy at 0.5, and log loss are also recorded.
- Regression: RMSE is the primary metric; MAE and R² are also recorded.
- Fit and prediction are timed separately. Timing excludes download, CSV
  parsing, preprocessing, Python import, and subprocess startup. Fit includes
  LibTree feature binning and competitor matrix/label setup. Prediction includes
  the interface's required validation or conversion; native XGBoost constructs
  a test DMatrix during prediction timing.
- The five-dataset competition suite uses three seeds and five isolated timing
  repeats for each seed/interface/engine. The scale/finance suite uses three
  seeds and three timing repeats for each thread/interface/engine.
- This report emphasizes the arithmetic mean across raw timing observations
  and reports variability or min/max where available. Historical reports may
  use pooled medians; their statistic is stated in each report. Never compare
  a mean in one table to a median in another as if they were the same measure.
- Every run records source/dataset hashes, library versions, parameters, CPU
  metadata, predictions, and raw times. Repeated fits must be deterministic;
  C++/Python outputs are checked for parity within documented float tolerance.

## 4. Results

### Five-dataset competitor comparison

The following are arithmetic means over raw runs and equally weighted datasets
from the five-dataset, single-thread snapshot. Times are milliseconds. These are
historical measurements from October 9, 2026; see the raw file for per-dataset
results and repeats.

| Interface | Engine | Fit mean ↓ | Predict mean ↓ | Held-out quality summary |
| --- | --- | ---: | ---: | --- |
| C++ | LibTree | 14.336 | 0.363 | Best primary metric on Titanic, Pima, Wine; near tie on Telco; insurance R² slightly below XGBoost |
| C++ | XGBoost | 21.995 | 0.811 | Slightly higher insurance R²; Telco AUC near tie |
| C++ | LightGBM | 19.947 | 1.749 | Close quality on the fixed splits |
| Python | LibTree | 14.577 | 0.496 | Same native model quality as C++ |
| Python | XGBoost | 29.287 | 1.590 | Same XGBoost model quality as its C++ route |
| Python | LightGBM | 33.016 | 2.713 | Same LightGBM model quality as its C++ route |

Quality is close, but it is not uniformly identical. For example, average
three-split ROC-AUC was 0.8393 for LibTree, 0.8325 for XGBoost, and 0.8337 for
LightGBM on Titanic. Insurance R² was 0.8566, 0.8574, and 0.8566 respectively.
These three-seed differences are descriptive, not proof of statistical
significance.

### Finance and scale comparison

The table reports mean time across raw observations at one thread. Quality is
the average primary held-out metric across three seeds. RMSE is lower-is-better;
AUC is higher-is-better.

| Dataset | Interface | Engine | Fit ms | Predict ms | Test metric |
| --- | --- | --- | ---: | ---: | ---: |
| Bank Marketing | C++ | LibTree | 273.01 | 6.209 | AUC 0.8055 |
| Bank Marketing | C++ | XGBoost | 280.77 | 13.898 | AUC 0.8056 |
| Bank Marketing | C++ | LightGBM | 255.67 | 23.058 | AUC 0.8052 |
| Bank Marketing | Python | LibTree | 280.93 | 6.625 | AUC 0.8055 |
| Bank Marketing | Python | XGBoost | 319.51 | 11.610 | AUC 0.8056 |
| Bank Marketing | Python | LightGBM | 286.70 | 26.251 | AUC 0.8052 |
| Friedman 20k | C++ | LibTree | 115.26 | 3.048 | RMSE 1.1668 |
| Friedman 20k | C++ | XGBoost | 133.46 | 5.024 | RMSE 1.1556 |
| Friedman 20k | C++ | LightGBM | 132.06 | 17.001 | RMSE 1.1497 |
| Friedman 20k | Python | LibTree | 108.92 | 3.254 | RMSE 1.1668 |
| Friedman 20k | Python | XGBoost | 153.04 | 6.292 | RMSE 1.1556 |
| Friedman 20k | Python | LightGBM | 149.68 | 18.549 | RMSE 1.1497 |
| Friedman 100k | C++ | LibTree | 557.40 | 14.880 | RMSE 1.1163 |
| Friedman 100k | C++ | XGBoost | 510.74 | 22.970 | RMSE 1.1269 |
| Friedman 100k | C++ | LightGBM | 612.61 | 88.954 | RMSE 1.1159 |
| Friedman 100k | Python | LibTree | 538.20 | 15.187 | RMSE 1.1163 |
| Friedman 100k | Python | XGBoost | 538.09 | 20.107 | RMSE 1.1269 |
| Friedman 100k | Python | LightGBM | 618.83 | 89.204 | RMSE 1.1159 |

There is no single winner across both quality and runtime. At 100k rows, for
example, XGBoost's fit time is similar or lower while LibTree inference is
faster in this setup. On Bank Marketing the AUCs are nearly tied. Such tradeoffs
are more useful than an unsupported claim that one engine is always best.

### Train/test gap: a check against overfitting

These examples use the same fitted model for train and test scoring (seed 42,
one thread). They are selected diagnostics, not an estimate across all possible
deployments.

| Task | Training result | Test result | Gap |
| --- | --- | --- | --- |
| Bank Marketing, LibTree | AUC 0.8292; log loss 0.2609 | AUC 0.8150; log loss 0.2650 | AUC falls 0.0142; log loss rises 0.0041 |
| Friedman 20k, LibTree | RMSE 1.0696; R² 0.9542 | RMSE 1.1478; R² 0.9466 | Test RMSE is about 7.3% higher |
| Friedman 100k, LibTree | RMSE 1.0874; R² 0.9525 | RMSE 1.1127; R² 0.9502 | Test RMSE is about 2.3% higher |

## 5. Ablation studies

### A. Parallelism by thread budget

This is a controlled thread-budget ablation: same data splits, fixed model
parameters, and thread budgets 1/2/4. Times below are arithmetic means of all
raw observations, in milliseconds. The machine exposed five logical CPUs but
had a container quota of four CPU cores.

| Dataset / training rows | Interface | 1 thread fit | 2 threads fit | 4 threads fit | 1→4 fit speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Bank Marketing / 30,891 | C++ | 273.01 | 255.20 | 243.49 | 1.12× |
| Bank Marketing / 30,891 | Python | 280.93 | 258.34 | 254.75 | 1.10× |
| Friedman 20k / 15,000 | C++ | 115.26 | 100.28 | 95.03 | 1.21× |
| Friedman 20k / 15,000 | Python | 108.92 | 90.60 | 100.78 | 1.08× |
| Friedman 100k / 75,000 | C++ | 557.40 | 431.72 | 372.78 | 1.50× |
| Friedman 100k / 75,000 | Python | 538.20 | 421.92 | 393.07 | 1.37× |

Larger training jobs benefit more in this sample. The Python Friedman 20k task
is slower at four threads than at two, demonstrating scheduling noise and
parallel overhead. This supports workload-aware fallback, not a blanket
recommendation to maximize thread count.

### B. Before/after optimization bundle

This paired native C++ ablation compares commit `937e6b4` with a bundle of
training/inference optimizations. Each input ran nine times in randomized
baseline/optimized order. The exact float32 predictions matched for every
input. Values below are arithmetic means computed directly from the raw
observations (the Chinese report also contains median summaries).

| Dataset | Baseline fit ms | Optimized fit ms | Fit speedup | Baseline predict ms | Optimized predict ms | Predict speedup | Predictions identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Titanic | 12.67 | 4.11 | 3.08× | 0.416 | 0.169 | 2.46× | Yes |
| Insurance | 15.74 | 4.34 | 3.62× | 0.670 | 0.192 | 3.50× | Yes |
| Pima | 9.16 | 6.71 | 1.36× | 0.525 | 0.169 | 3.11× | Yes |
| Telco | 292.10 | 57.77 | 5.06× | 4.269 | 1.073 | 3.98× | Yes |
| Wine | 12.90 | 7.57 | 1.71× | 0.857 | 0.199 | 4.32× | Yes |
| Friedman 20k | 250.35 | 108.93 | 2.30× | 12.440 | 3.050 | 4.08× | Yes |

**Interpretation:** this measures the combined change, not each individual
optimization. It cannot attribute a specific speedup to histogram subtraction,
quantization lookup, sparse-column handling, row-major histogram indices, or
compiled prediction layout separately.

### C. Isolated kernel ablations still needed

The next experiments should keep prepared inputs, compiler, hardware, output
quality, and raw timing protocol fixed while toggling exactly one feature:

| Component | Treatment | Primary outcome | Guardrail |
| --- | --- | --- | --- |
| Histogram subtraction | Rebuild both child histograms vs. build small child and subtract from parent | Fit time and bytes scanned | Exact model/prediction equality |
| Sparse/dominant-bin path | General scan vs. dominant-bin and two-value kernels | Fit time by cardinality/missingness | Same split decisions and loss |
| Wide row-major indices | Column scan vs. row-major index path | Fit time across widths 8–256 | Memory high-water and output equality |
| Compiled prediction layout | Ordinary node traversal vs. compiled shallow layout | Predict rows/s, model bytes | Exact traversal oracle |
| Layer-level builder | Recursive builder vs. ready-node layers | Fit time across sizes/depths/thread counts | Exact model bytes, loss and predictions |

These switches do not currently exist as public runtime parameters. They should
be implemented as benchmark-only build options or internal test strategies,
not permanent user-facing controls. Report per-case means and distributions,
not only the fastest run.

## 6. Limitations and reproducibility

- A three-seed holdout study is useful for regression checks, but it is not a
  statistical significance test. Confidence intervals and more seeds are
  needed for small quality differences.
- Random splits can leak group or time structure. Bank needs temporal/customer
  grouped evaluation before any deployment claim; no such result is reported.
- Data was fetched from HTTPS public mirrors with pinned SHA256. A Kaggle URL
  identifies the related dataset page; it does not prove byte identity with an
  official Kaggle download. No leaderboard submission occurred.
- Hardware is one shared cloud CPU with a four-core quota. Results do not
  generalize to other CPUs, storage, memory bandwidth, thread quotas, or GPUs.
- The models use fixed high-level parameter matches, not exhaustive tuning.
  Leaf growth, split sketches, minimum-child definitions, and missing-value
  rules differ. These are practical comparisons, not an identical-algorithm
  contest.
- Fit timings exclude preprocessing but include learner-specific input
  construction. Prediction timings include each public interface's required
  input validation and representation conversion. Interpret them as these
  interfaces' measured end-to-end calls.

Reproduce the latest parallel/finance matrix (after building Release binaries):

```sh
. /workspace/libtree-venv/bin/activate
PYTHONPATH=python python benchmarks/parallel.py \
  --threads 1 2 4 --seeds 42 2024 2026 --repeats 3 \
  --datasets bank_marketing friedman_20k friedman_100k \
  --output benchmarks/adaptive-parallel-results.json
```

The [interactive learning guide](../docs/LEARNING_GUIDE.html) explains the
algorithm to beginners. The top-level [README](../README.md) summarizes project
setup, optimization history, and measured strengths/limits.
