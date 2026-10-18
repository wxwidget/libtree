# Mercari: LibTree Kaggle competition journal

[中文记录](README.zh-CN.md) · [Competition](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge) · [Kaggle Kernel](https://www.kaggle.com/code/wxwidget/libtree-mercari-reproducible-entry)

**Reading paths:** [Beginner walkthrough](BEGINNER_GUIDE.md) for concepts and a
small runnable example; [technical retrospective](TECHNICAL_REPORT.md) for
architecture, theory, successful/failed experiments and reproducibility limits.

**Current direction:** the [CPU-budget study](CPU_STUDY.md) adds explicit product
crosses, sparse objective/view diversity and direct metric gradients. The queued
GPU deployment and automatic watcher were stopped under the CPU-only constraint.
The subsequent [multiple-view experiment](CPU_MULTIVIEW.md) reaches **0.392846
local audit RMSLE**, using full-base expert reuse and hierarchical product
statistics; 370 seconds is incremental work, not a fresh complete refit.

The earlier local frozen blend reaches **0.380907 RMSLE**, down **3.36%** from
0.394153 on the reused audit window; no new Kaggle submission is claimed.
The [complete frozen ensemble pipeline](FULL_ENSEMBLE.md) preserves that original
candidate and the historical GPU workflow; it is not currently running.
The [modern text/GPU study](FRONTIER.md) extends this work with LSA, pinned
sentence vectors, T4/A100 neural experts and selection-only tree/regression
ensembles. It records failed attempts and provides a Colab experiment launcher;
local validation remains distinct from the scored entries below.

The [optimization study](OPTIMIZATION.md) achieved an actual **10.46% Kaggle
private RMSLE reduction**, from **0.43776 to 0.39199** (public **0.39110**), with
submission [57318931](kaggle-submission-57318931.json), status **COMPLETE**.
It includes feature engineering, loss and ensemble ablations, a frozen 10.28%
local holdout improvement, CPU costs and reproduction commands. The score gain
comes from the complete ensemble and increases inference cost.

The separate fast entry scored **0.39361 private / 0.39278 public**
(10.09% private reduction), submission [57319422](kaggle-submission-57319422.json),
with a verified **44.04-minute stage-one runtime**. Its local full-run process
was 17.0% shorter than accuracy in one observed run. See the study for inference
costs, stage-two durations and scope.

## Original version-2 baseline and protocol

The following methods and measurements describe the original scored baseline.
The optimized configurations, separate calibration protocol and actual new
scores are in the [independent study](OPTIMIZATION.md).

This entry predicts listing prices with LibTree's native C++ GBDT. It compares a
structured-only LibTree model, a sparse text Ridge reference, and a Ridge +
LibTree residual ensemble. The final model is selected from the LibTree entries
using local tuning data, not the public leaderboard. All results below are
explicitly identified as local validation or actual Kaggle scores.

The historical competition had **2,380 teams** and ended on February 21, 2018.
The original contest required **Kernels submissions**: training and inference in
one Kernel, with the stage-two test data substituted during the final rerun.
Any current submission is a post-competition practice attempt, not participation
in the original prize contest. Upload success and an available submission quota
do not prove that Kaggle will score an entry.

**Scored practice submission, October 17, 2026:** Kernel version **2** completed
successfully. Submission **57307868** has status **COMPLETE**:

| Actual Kaggle metric | RMSLE ↓ |
| --- | ---: |
| Public score | **0.43775** |
| Private score | **0.43776** |

[Authenticated API evidence](kaggle-submission-57307868.json) and the
[submission journal](record.json) preserve the exact returned scores, submission
ID, Kernel version, timestamp and source hash. The API returns `userRank=0`;
no new original-contest placement is claimed. The original contest is over.

## Metric and data

The official metric is RMSLE, minimized:

```text
sqrt(mean((log1p(predicted_price) - log1p(actual_price)) ** 2))
```

We learn `log1p(price)` and convert predictions with `expm1`, clipping negative
log predictions to zero. No test labels or external price data are used. Mercari
scraping/API price lookups are prohibited by the rules and are not part of this
entry. [competition.json](competition.json) captures the official metric,
data fields, Kernel protocol and timeline.

The official downloaded training file has **1,482,535 listings**, 874 zero-priced
items, 632,682 missing brands, 6,327 missing categories and 6 missing descriptions.
Zero-priced rows are retained. Missing values receive a fixed token. Stage one
has 693,359 test listings; the final run uses the stage-two test and matching
sample submission. File hashes, exact split sizes and submission hashes are
stored with each experiment.

## Validation and methods

Hash the six available feature columns, excluding ID and price, so identical
listings remain in one group. GroupShuffleSplit with seed 42 reserves approximately
80/10/10 for fitting, tuning and holdout. There are no transaction timestamps or
seller IDs here; this is a grouped random split, not a temporal deployment study.
One held-out split does not provide a confidence interval over deployment tasks.

All vocabularies, category mappings and target means use only fitting data.
The fitting window itself uses three **grouped out-of-fold** models. Each inner
fold refits its text vocabulary, categorical means and global prior without that
fold's labels. In particular, the residual learner uses held-out Ridge predictions,
not overly optimistic in-sample predictions.

The Ridge branch uses word unigrams/bigrams from titles and descriptions, TF-IDF,
double weight for title features, up to 150,000 features per text field in the
full study, and categorical one-hot indicators. Ridge has
alpha 3, LSQR, tolerance .001, and at most 100 iterations. The native branch uses
condition, shipping, text-length/missingness features, and smoothed brand/category
log-price means and frequencies. Its defaults are 400 trees in the full study,
depth 6, 128 bins, learning rate .06, leaf minimum 20, L2 10, four threads.

The residual model learns `log1p(price) - OOF_Ridge_prediction` and also receives
the OOF prediction as a feature. Tuning chooses correction strength .25/.5/1 or
the structured-only LibTree alternative. Settings are frozen before reporting
the three predeclared holdout comparisons. Final training then uses all available
training labels with a new three-fold cross fit. The final refit is not the same
model that produced the holdout measurements.

This compares complete systems: the structured-only model has no lexical text
features, while the ensemble adds both a native nonlinear learner and engineered
features. An improvement over Ridge does not isolate the tree algorithm's causal
contribution, establish superiority over every alternative, or imply a historical
leaderboard position.

Method references: [Micci-Barreca (2001), categorical target encoding](https://doi.org/10.1145/507533.507538);
[Wolpert (1992), stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1);
[Salton & Buckley (1988), term weighting](https://doi.org/10.1016/0306-4573(88)90021-0);
[Hoerl & Kennard (1970), Ridge regression](https://doi.org/10.1080/00401706.1970.10488634).
For LibTree's histogram GBDT and research references, see the
[design guide](../../docs/design.md).

## Measurements

The [150,000-row pilot](pilot-001.json), with 120,000 fit / 15,000 tune / 15,000
holdout rows, produced:

| Model | Local holdout RMSLE ↓ |
| --- | ---: |
| Sparse text Ridge | 0.496448 |
| Structured-only LibTree | 0.565001 |
| Ridge + LibTree residual correction | **0.484522** |

The ensemble reduced RMSLE by **2.40% relative to Ridge** on this pilot holdout.
That pilot changes both sample size and model capacity relative to the full run;
it is not a controlled learning curve. Its single native timing repetition is
not a reliable performance distribution. Full-run timing uses three repetitions
and preserves arithmetic mean, min, max, population standard deviation and raw
samples. Native fit/predict timings exclude text preparation; end-to-end times
include the whole pipeline and final test prediction.

Full-data validation (1,482,535 source training rows, decisions frozen before
holdout; final refit and stage-two prediction complete):

| Model | Local holdout RMSLE ↓ |
| --- | ---: |
| Sparse text Ridge | 0.450090 |
| Structured-only LibTree | 0.547755 |
| Ridge + LibTree residual correction | **0.439312** |

The ensemble reduced RMSLE by **2.39% relative to Ridge**. The winning tuning
RMSLE was 0.441273 with correction strength 1. These are local measurements,
not Kaggle scores. Actual submission status is recorded separately.

The [full-data report](full-001.json) records 1,186,026 fit / 148,255 tune /
148,254 holdout rows and a final refit on all 1,482,535 labels. The final
stage-two submission has **3,460,725 rows**. Restoring the saved model reproduces
its first 128 saved predictions (relative tolerance 1e-6).

Four-thread native training of 400 depth-six trees, three repetitions:

| Native branch | Mean fit, s | Min, s | Max, s | Population std, s |
| --- | ---: | ---: | ---: | ---: |
| Structured-only LibTree | 31.976 | 30.921 | 33.602 | 1.166 |
| LibTree residual branch | 30.973 | 30.126 | 31.897 | 0.725 |

Native prediction on the 148,255-row tuning set averaged 272.502 ms for the
structured model and 292.230 ms for the residual branch; raw samples and ranges
are preserved in the JSON. These timings exclude the text model and feature
preparation. Final native residual fitting on all labels took 43.500 s once;
final feature preparation, prediction and CSV writing took 236.999 s for the
3,460,725 test rows. The complete experiment, including validation, repeated
native fits, all cross fits and final prediction, took **1482.396 s (24.71 min)**.

The residual ensemble also requires three additional text fits for honest
out-of-fold training. Its full-data cross fit took 352.976 s; the final
text/encoder fit took 134.541 s. The accuracy gain therefore carries training
cost; the 30.973 s native mean is not the ensemble's total training time.
[environment.json](environment.json) records the local dependency versions,
CPU quota and memory limit. Input/code hashes bind the raw measurements to the
exact runner; these runs were produced from a working tree based on `ff4a4e2`
before the competition code was committed.


## Reproduce locally

From the repository root, with a C++17 compiler:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[dev,benchmark]' kaggle py7zr
mkdir -p benchmarks/runs/mercari/data
kaggle competitions download mercari-price-suggestion-challenge \
  -p benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/mercari-price-suggestion-challenge.zip \
  benchmarks/runs/mercari/data
python -m py7zr x benchmarks/runs/mercari/data/train.tsv.7z benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/test_stg2.tsv.zip benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/sample_submission_stg2.csv.zip benchmarks/runs/mercari/data
python -m benchmarks.mercari \
  --data-dir benchmarks/runs/mercari/data \
  --output-dir benchmarks/runs/mercari/full-001 \
  --trees 400 --depth 6 --max-features 150000 --folds 3 --repeats 3 --threads 4
```

Use a new output directory per run. Configure Kaggle authentication outside Git;
never paste a token into code, reports or command arguments. This cloud workspace
uses `/workspace/libtree-venv/bin/python` and `/workspace/libtree-venv/bin/kaggle`.
For a cheaper pilot add `--train-rows 150000 --trees 300 --repeats 1`, optionally
using `--test-file test.tsv --sample-file sample_submission.csv` after extracting
the corresponding `.7z` files. The full pipeline holds sparse text matrices and
dense native features; the tested local environment has a four-CPU quota and
32 GiB memory. Prediction is chunked in 100,000 rows.

Each run writes `report.json`, `submission.csv`, `model.pkl`, `tree.txt`,
`tree.json` and `explanation.json`. The last file traces the native learner on
its first out-of-fold training vector; in the ensemble it explains a **log-price
residual**, not the full text model or the final dollar price. Tree files show the
first tree only. Load only trusted pickle files; they can execute code.

Raw competition data, models and outputs stay under the ignored
`benchmarks/runs/` tree. Commit only source, aggregate measurements and the
submission journal. [target-history.json](target-history.json) explains the
earlier competition links; their data/results are not mixed into Mercari.

## Reproduce as a Kaggle Kernel

The builder embeds only LibTree C++/Python source and this runner. It does not
upload local models, training records or credentials. The Kernel compiles native
code with `g++`, trains inside Kaggle, uses the competition input, and writes
`submission.csv`. Internet access and GPU are disabled; the Kernel starts private.

```sh
python -m benchmarks.mercari_kernel --owner YOUR_KAGGLE_USERNAME \
  --output-dir benchmarks/runs/mercari/kernel-001
kaggle kernels push -p benchmarks/runs/mercari/kernel-001 -t 3600
kaggle kernels status YOUR_KAGGLE_USERNAME/libtree-mercari-reproducible-entry
```

Record the version number printed by `kernels push`. After that version
completes, replace `VERSION_FROM_PUSH` below with that number and submit its
generated file through the code-competition API:

```sh
kaggle competitions submit mercari-price-suggestion-challenge \
  -k YOUR_KAGGLE_USERNAME/libtree-mercari-reproducible-entry -v VERSION_FROM_PUSH \
  -f submission.csv -m 'LibTree: grouped OOF Ridge + native residual trees'
kaggle competitions submissions mercari-price-suggestion-challenge --format json
```

The executed version 2 source was pulled back through the authenticated API
and matched SHA-256 `e616b484d06d29c99d42c8d06f60294634b1c7d4a13eab0a25e1f9812752a236`.
[Output metadata](kaggle-output-files.json) confirms the stage-two input files,
trained model, report and `submission.csv`. The cloud proxy blocks
`www.kaggleusercontent.com`, so the remote report file itself was not downloaded;
validation used the exact approved source, successful execution and authenticated
output metadata. The score evidence comes directly from the submissions API.
Local reports and the remote score are distinct records.

Version 1 failed before model training because it assumed a flat input mount.
Version 2 searches the input root recursively; a synthetic nested-mount regression
test protects this fix. The current builder additionally removes temporary
extracted data after a successful run. This cleanup was added after version 2;
the executed version remains private. It does not change model calculations.

The implementation commit `3f5ff28` passed
[GitHub Actions run 39025253342](https://github.com/wxwidget/libtree/actions/runs/39025253342).

Retain the exact Kernel version, output/report hashes, submission ID, status,
score, any API error, and capture time. Scores and ranks remain null until
provided by Kaggle. As this is a closed competition, a scored practice submission
must not be reported as a new official contest placement.

## Software checks

TDD tests cover the RMSLE definition, duplicate grouping, exclusion of each
training row's fold labels, frozen features, unknown categories, native prediction
and pickle parity, submission IDs/order/schema, and the source-only Kernel bundle.

```sh
PYTHONPATH=. python -m pytest tests/test_mercari.py tests/test_mercari_kernel.py \
  --cov=benchmarks.mercari --cov=benchmarks.mercari_kernel \
  --cov-report=term-missing --cov-fail-under=100
```

Result: **20 passed; 280/280 runner/builder statements covered (100% line
coverage)**. The generated Kernel's embedded bootstrap string is additionally
checked syntactically and run on Kaggle; its runtime is not Python line-coverage
instrumented. Existing core checks also passed: 8/8 CTest targets, 113 Python/CLI
tests, four performance/memory tests, Python core 505/505 statements and native
core 2790/2790 lines. New competition checks are enforced separately in CI.
