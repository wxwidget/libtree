# Zero-price and positive equal-frequency classification

[中文](PRICE_BUCKETS.zh-CN.md) · [Measured results](price-bucket-results.json) · [Previous regression study](RETRIEVAL_TRUNK.md)

## Task and train-only boundaries

The target is now a discrete price range. **Zero is class 0**. Positive prices
are divided into **nine requested equal-frequency buckets**, giving at most ten
classes in total. The quantile cuts use only the **1,186,026 base-training rows**;
zero rows do not influence positive-price quantiles.

`PriceBuckets` uses NumPy's `inverted_cdf` empirical quantile definition. Equal
prices remain in the same bucket. Duplicate cuts and cuts at the maximum collapse
instead of creating empty classes. Exact equal row counts are generally impossible
when many listings have the same price; no IDs or arbitrary tie breaking split
identical prices. Positive intervals are right closed `(lower, upper]`; the
last extends to infinity. The original log1p targets preserve price ordering,
and comparisons use log-space cuts to avoid dollar round-trip boundary errors.

For a beginner: fitting a boundary of $10 means that positive $10 listings all
belong to the lower interval. A $10.01 listing goes to the next interval. The
model never sees its query's price: it predicts a class from the title,
description and observable listing features.

| Class | Price range (USD) | Base training | Audit rows |
| --- | --- | --- | --- |
| 0 | $0 | 699 | 92 |
| 1 | ($0, $8] | 169,726 | 21,233 |
| 2 | ($8, $10] | 130,498 | 16,353 |
| 3 | ($10, $12] | 95,009 | 11,978 |
| 4 | ($12, $15] | 137,515 | 17,111 |
| 5 | ($15, $19] | 133,621 | 16,520 |
| 6 | ($19, $24] | 134,365 | 16,631 |
| 7 | ($24, $31] | 124,398 | 15,764 |
| 8 | ($31, $49] | 132,888 | 16,552 |
| 9 | > $49 | 127,307 | 16,020 |

The nine positive buckets target about 131,703 rows each; preserved price ties prevent exact equality.

## Models and evaluation protocol

All four classifiers use the same full base-training set and numeric inputs:
the retained base-fitted title/description LSA representation, numeric metadata,
category frequencies, 20 market cues and eight training-only comparable features.
Eight unordered category ID columns are excluded from both engines, since the
native C5 classifier provides numeric threshold splits. This measures the shared
numeric representation; it does not establish either engine's accuracy ceiling.

- **LibTree C5**: up to 20 SAMME rounds, depth 6, 64 bins, Gini splits, rate 0.5,
  minimum leaf support 80 and pruning alpha 0.0001.
- **LightGBM multiclass**: up to 150 rounds, 31 leaves, 127 bins, rate 0.05,
  minimum leaf support 80, L2 10 and feature fraction 0.85. There can be one tree
  per class per round. Round budgets and algorithms differ across engines.
- Both engines repeat with inverse-frequency **balanced class weights**, fitted
  only from training counts. This changes the training distribution and the
  interpretation of probability outputs; higher zero recall can reduce overall
  accuracy. C5 normalized-vote probabilities are also uncalibrated.
- All runs use seed 42 where supported and four CPU threads. There is one fit
  per configuration; no seed-stability claim or hyperparameter search is made.

The original grouped windows remain base **1,186,026**, calibration **74,129**,
selection **74,126**, and reused audit **148,254**. Identical available listing
features cannot cross windows. Data/source/context hashes and ordered IDs are
checked. Comparable inputs exclude a query's entire duplicate group and its
labels from priors. The LSA encoder and comparable index are trusted base-fitted
artifacts; the script reuses them without fitting held-out listing targets.

LightGBM early stopping monitors **unweighted calibration classification error**,
with patience 25. Selection chooses the highest-accuracy classifier; a separate
macro-F1 choice is also frozen for the minority-class question. Ties follow the
declared model order. Audit labels are loaded only after all fits, predictions
and choices are saved to `frozen.json`.

Comparators include the training-majority class, the previous retrieval-CNN
regression converted to these bins, and the frozen final regression ensemble.
The latter has historically fitted its meta layer on selection, so its selection
accuracy is not a held-out comparison. Its audit result is reported separately.
Different regression features, model capacities and fitting roles prevent a
controlled causal claim that classification itself beats regression.

Historical selection/audit reuse limits generalization. A bucket classifier has
a different objective from Kaggle RMSLE. Two very different high prices can share
the last bucket; a correct class does not establish an accurate dollar forecast.
No final fusion changes or Kaggle submissions are made in this study.
All four probability outputs are retained as possible future meta features;
their complementary value for RMSLE still needs a separate fusion evaluation.

## Accuracy and class errors

| Model | Selection accuracy | Audit accuracy | Audit balanced accuracy | Audit macro-F1 | Zero recall |
| --- | --- | --- | --- | --- | --- |
| Training majority | 14.40% | 14.32% | 10.00% | 0.0251 | 0.00% |
| LibTree C5 | 31.28% | 31.23% | 26.60% | 0.2577 | 0.00% |
| LibTree C5, balanced | 29.32% | 29.13% | 26.07% | 0.2588 | 6.52% |
| LightGBM | 32.93% | 32.76% | 27.73% | 0.2634 | 0.00% |
| LightGBM, balanced | 31.77% | 31.83% | 28.06% | 0.2738 | 4.35% |
| Previous retrieval CNN → bins | 33.35% | 33.56% | 29.72% | 0.3056 | 0.00% |
| Frozen final regression ensemble → bins | 36.68% * | 36.71% | 32.53% | 0.3346 | 0.00% |

* The final regression meta layer was fitted on selection; that entry is not held out.

Selection froze **LightGBM** for accuracy and **LightGBM, balanced** for macro-F1. The accuracy choice correctly classifies 48,569 of 148,254 audit rows. Of its positive-price errors, 42.12% land in an adjacent positive bucket.

Direct multiclass training does not beat the existing final regression system here: LightGBM reaches **32.76%**, versus **36.71%** after binning the regression ensemble, a 3.95 percentage-point gap. Different inputs and budgets prevent an objective-only ablation claim. C5 reaches 31.23% at lower measured cost and lower accuracy. Balanced LightGBM improves macro-F1 slightly while reducing overall accuracy to 31.83%.

There are 92 true zero-price audit rows. Zero-class false-positive costs:

| Model | True zeros found | Predicted zeros | False positives | Zero precision | Zero recall |
| --- | --- | --- | --- | --- | --- |
| LibTree C5 | 0 | 0 | 0 | — | 0.00% |
| LibTree C5, balanced | 6 | 6373 | 6367 | 0.09% | 6.52% |
| LightGBM | 0 | 0 | 0 | — | 0.00% |
| LightGBM, balanced | 4 | 2030 | 2026 | 0.20% | 4.35% |

Equal frequency balances positive prices, not the zero class. Only 4 of 2,030 balanced-LightGBM zero predictions and 6 of 6,373 balanced-C5 zero predictions are correct; neither is a reliable zero-price detector here.

Overall accuracy is the fraction of exact class matches. Per-true-bucket recall
is its within-bucket hit rate; precision asks what fraction of predictions of
that class were correct. Macro-F1 averages all schema classes equally, with zero
for undefined F1. Balanced accuracy averages recall over supported true classes.
Missing precision or recall is shown as null when its denominator is zero.

Recall within each true bucket:

| Class | Price | LibTree C5 | LibTree C5, balanced | LightGBM | LightGBM, balanced | Frozen final regression ensemble → bins |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | $0 | 0.00% | 6.52% | 0.00% | 4.35% | 0.00% |
| 1 | ($0, $8] | 62.04% | 49.93% | 68.24% | 58.62% | 45.37% |
| 2 | ($8, $10] | 27.39% | 26.04% | 25.01% | 24.55% | 22.92% |
| 3 | ($10, $12] | 5.08% | 17.84% | 2.43% | 17.52% | 24.64% |
| 4 | ($12, $15] | 17.33% | 14.41% | 16.15% | 12.03% | 32.24% |
| 5 | ($15, $19] | 26.73% | 21.54% | 33.29% | 30.07% | 32.09% |
| 6 | ($19, $24] | 20.47% | 18.31% | 17.57% | 17.01% | 30.88% |
| 7 | ($24, $31] | 19.67% | 20.43% | 20.18% | 24.21% | 29.93% |
| 8 | ($31, $49] | 29.88% | 30.98% | 38.36% | 35.32% | 45.23% |
| 9 | > $49 | 57.45% | 54.68% | 56.07% | 56.89% | 62.05% |

Confusion matrix for the accuracy choice (LightGBM), with true classes in rows and predicted classes in columns:

| True / predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 23 | 12 | 2 | 9 | 25 | 6 | 6 | 5 | 4 |
| 1 | 0 | 14489 | 2691 | 48 | 1145 | 1358 | 527 | 426 | 296 | 253 |
| 2 | 0 | 5384 | 4090 | 133 | 1754 | 2953 | 728 | 611 | 440 | 260 |
| 3 | 0 | 2747 | 2479 | 291 | 1631 | 2911 | 696 | 576 | 415 | 232 |
| 4 | 0 | 3302 | 2458 | 120 | 2763 | 4510 | 1459 | 1205 | 919 | 375 |
| 5 | 0 | 1686 | 1973 | 81 | 1857 | 5500 | 1952 | 1618 | 1369 | 484 |
| 6 | 0 | 1625 | 1114 | 47 | 1376 | 4169 | 2922 | 2459 | 2172 | 747 |
| 7 | 0 | 1229 | 642 | 24 | 840 | 2897 | 2211 | 3181 | 3529 | 1211 |
| 8 | 0 | 727 | 467 | 10 | 452 | 2117 | 1301 | 2378 | 6350 | 2750 |
| 9 | 0 | 519 | 208 | 0 | 209 | 1087 | 456 | 863 | 3695 | 8983 |

The clearest weak slice is `(10, 12]`: ordinary LightGBM recall is only **2.43%**, rising to 17.52% with balancing while overall accuracy falls. Its cheapest positive bucket recalls 68.24% and its highest bucket 56.07%. Unequal interval widths mean class matches are not equal dollar errors; `>49` still cannot distinguish $50 from $1,000.

## Costs, validation and reproduction

Costs are wall-clock observations. Three repeated probability-inference timings
use cached numeric calibration inputs, after earlier predictions warm the model;
arithmetic mean, maximum, standard deviation and all raw samples are retained.
These exclude text/context preparation, model loading and upstream pretraining.

| Model | Actual rounds | Fit seconds | Inference mean seconds | Inference max seconds | Inference std seconds |
| --- | --- | --- | --- | --- | --- |
| LibTree C5 | 20 | 57.56 | 0.0374 | 0.0405 | 0.0025 |
| LibTree C5, balanced | 20 | 56.56 | 0.0359 | 0.0379 | 0.0014 |
| LightGBM | 78 | 279.21 | 1.3328 | 1.3389 | 0.0057 |
| LightGBM, balanced | 94 | 302.76 | 1.6441 | 1.6487 | 0.0041 |

Four vCPUs; each inference repeat covers 74,129 rows and 170 columns. Shared feature preparation takes 164.69s, the complete study 887.41s, and process peak RSS is 6460.4 MiB. One fit and three warm inference repeats do not establish thread scaling or a matched-budget speed advantage.

TDD: 18 new tests, covering 208/208 new executable statements. The CI-equivalent CPU regression passes **248 tests, with 100% statement coverage (4,453/4,453)** across 33 Mercari CPU benchmark modules. This is not whole-repository, branch, GPU or memory-leak coverage. Independent checks use integer order statistics for quantiles and sklearn for every audit metric and confusion matrix. Rebuilding 512 rows from raw, target-free listings gives maximum feature error 0; all four model probability replays have zero error. Source, input, prediction, model and validation-log hashes are retained in the results JSON.

Run from the repository root using the existing `.[mercari,benchmark]` environment.
Prepare the LSA encoder through the [frontier workflow](FRONTIER.md) and the
group-excluded context cache through the [retrieval workflow](RETRIEVAL_TRUNK.md).
Use their matching data windows. The output directory must be new.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 \
python -m benchmarks.mercari_price_bucket_study \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --encoder benchmarks/runs/mercari/frontier-001/trees/encoder.pkl \
  --context-dir benchmarks/runs/mercari/retrieval-trunk-001/inputs \
  --regression \
    benchmarks/runs/mercari/retrieval-trunk-001/full/retrieval_seed42.npz \
    benchmarks/runs/mercari/final-objective-001/full/ensemble.npz \
  --output-dir benchmarks/runs/mercari/price-bucket-reproduction
```

`--regression` is optional. It adds existing baselines without fitting them.
The required trusted LSA encoder is numeric-prefix compatible with `LatentText`;
new pretraining produces a new baseline and potentially different results.
All data and large model/cache files remain under the ignored runs directory.

Outputs include `bins.pkl`, four classifier snapshots, full probability/class
archives, data/model hashes, `plan.json`, `frozen.json` and `report.json` with
every confusion matrix and per-class support/precision/recall/F1. Prediction
archives use explicit `_class` and `_probability` keys so class IDs cannot be
mistaken for log-price regressions by the existing fusion loader. Load trusted
model snapshots with pickle, call `predict_proba` and map `argmax` through the
stored `model.classes_`; `bins.describe()` supplies human-readable ranges.

The measurements and their source hashes correspond to
[603db32](https://github.com/wxwidget/libtree/commit/603db3242f2054a853cf43c3cb376a9f5d261957).
A subsequent small-data guard chooses binary loss/error when only two training
classes remain; a single observed class uses the prior dummy model. Its added
binary/multiclass checks pass all 18 targeted tests with 208/208 statements
covered. Ten-class LightGBM parameter sets, boosters and probabilities match the
measured implementation exactly in both weight modes. The JSON records this
check separately; reported full-data timings belong to the measurement commit.

```bash
python -m pytest tests/test_mercari_price_buckets.py \
  tests/test_mercari_price_bucket_study.py \
  --cov=benchmarks.mercari_price_buckets \
  --cov=benchmarks.mercari_price_bucket_study \
  --cov-report=term-missing --cov-fail-under=100
```

## Sources

- Hyndman & Fan (1996), [sample quantile definitions](https://doi.org/10.1080/00031305.1996.10473566).
- NumPy, [quantile and `inverted_cdf`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).
- Zhu et al. (2009), [multiclass AdaBoost/SAMME](https://doi.org/10.4310/SII.2009.v2.n3.a8).
- Ke et al. (2017), [LightGBM](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa0-Abstract.html).
- [LibTree C5 design and limitations](../../docs/c5.md).

