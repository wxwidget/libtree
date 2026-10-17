# Mercari optimization study

[中文](OPTIMIZATION.zh-CN.md) · [Competition journal](README.md) · [Frozen configuration](optimized-config.json)

The primary target is a **10% relative reduction in Kaggle private RMSLE** from
the scored baseline **0.43776**: the target is **0.393984 or lower**. Training
and inference costs are reported separately. This is a closed-competition
practice experiment; it does not establish an original-contest rank.

## Frozen local audit

The original **148,254-row holdout**, including 92 zero-priced listings, was
evaluated after freezing the configuration:

| Model / ablation | Holdout RMSLE ↓ |
| --- | ---: |
| Previously measured Ridge + LibTree baseline | 0.439312 |
| New Ridge | 0.433543 |
| Field-aware factorization | 0.418600 |
| Neural mixture | 0.397625 |
| Full blend without LibTree correction | 0.395064 |
| **Full blend with LibTree correction** | **0.394153** |

The relative holdout reduction is **10.28%**. Native correction contributes a
further 0.00091 RMSLE reduction with the same fitted experts. Actual Kaggle
verification is separate from this local result.

[Audit evidence](optimization-audit.json) · [Exploratory results](optimization-study.json)
· [Inference measurements](inference-optimization.json).

Serving-cost comparison on the same first 100,000 unlabeled stage-two listings,
one warm-up and three interleaved repetitions:

| Pipeline | Mean, s | Min, s | Max, s | Population std, s |
| --- | ---: | ---: | ---: | ---: |
| Previous scored baseline | 7.941 | 7.067 | 9.231 | 0.931 |
| New audited ensemble | 34.819 | 31.699 | 40.973 | 4.351 |

The ensemble costs about **4.38×** as much inference time here. These are
complete pipeline timings, including feature preparation. The baseline model
was fitted on all labels; the new estimator is the local audit model. A tested
sparse-matmul projection preserved predictions (maximum log-price difference
4.10e-7) but averaged 34.558 s, within the observed variability; it was discarded.

## What changed

The entry combines a field-aware factorization model, several complementary
neural text models, and a native C++ LibTree residual corrector. A sparse Ridge
model supplies another calibration feature. The improvement belongs to this
complete system; it does not isolate an improvement in the GBDT algorithm.

* Preserve single-character model and size tokens such as `6`, `7`, and `x`.
  Normalize `iphone6s` to `iphone 6s` while preserving units such as `64gb`.
  Stored preprocessing versions preserve older models' predictions.
* Recover missing brands only from unambiguous matches in a frozen training
  dictionary. Add category levels, category/condition and brand interactions,
  title character n-grams, and train-only smoothed word-price summaries.
* Solve sparse Ridge with matrix-free, diagonally preconditioned conjugate
  gradients. Implicit centering preserves the intercept and Ridge objective.
  Tests compare the coefficients and predictions with a dense Cholesky solver.
* Learn field-aware interactions across title, description, brand, category,
  and condition/shipping features. CPU workers own independent parameters and
  synchronize between chunks; they do not write shared parameters concurrently.
* Compare MSE, Huber regression, and price-bin classification. Sparse MLPs use
  bounded CSR batches. The row-sparse Adam variant aggregates duplicate gradient
  indices and updates only present rows; dense parameters retain standard Adam.
  This variant omits global matrix L2, which would make the gradient dense.
* Share listing preparation and the sparse feature matrix during inference.
  Fit LibTree on separate calibration listings rather than refitting large text
  models for every calibration fold. Stream stage-two ZIP inputs in Kaggle and
  omit large model pickles there; local runs can retain a reloadable model.

## Evaluation protocol

All 1,482,535 official training rows are eligible, including the 874 zero-priced
listings. No external prices, test labels, scraping, or pretrained model inputs
are used. Identical available listing features share one group; ID and price
are excluded from that group key.

The original seed-42 split reserves approximately 80% for fitting, 10% for
tuning, and 10% for a final audit. Exploratory text hyperparameters and stopping
epochs use tuning data. The tuning window is also split by group into **74,129
calibration** and **74,126 selection** listings. Native correction fits the
former; blend weights, correction strength, and tree settings use the latter.
These exploratory tuning measurements are not an independent final test.

Freeze the configuration before the original holdout audit. Final training
then assigns about **16/17** of all labels to the base models and **1/17** to
LibTree calibration, with disjoint listing groups. All labels contribute to
the final system, in different roles. Final training uses fixed epochs and
weights; it performs no leaderboard-based model selection.

```mermaid
flowchart LR
    A[Training listings] --> B[Grouped base window]
    A --> C[Separate calibration window]
    B --> D[Train-only text and category dictionaries]
    D --> E[Text experts]
    E --> F[Held-out calibration predictions]
    C --> F
    F --> G[Native LibTree residual trees]
    E --> H[Weighted log-price prediction]
    G --> H
    H --> I[expm1 and nonnegative price]
```

## Exploratory ablations

These rows use the same **148,255-row tuning window**, before the native
calibrator. Smaller RMSLE is better. They are local results, not Kaggle scores.

| Candidate | Tuning RMSLE | Observation |
| --- | ---: | --- |
| Expanded features, LSQR Ridge, alpha 3, 200 iterations | 0.449036 | Iteration cap was insufficient |
| Same features/alpha, preconditioned Ridge | 0.440556 | Better solution of the same objective |
| Preconditioned Ridge, alpha 10 | 0.437211 | Chosen regularization |
| Above, preserving model-number tokens | 0.435320 | Lexical resolution helps |
| Field-aware model, original tokens | 0.423157 | Early stopping matters |
| Field-aware model, repaired tokens | 0.420152 | Best at epoch 2 |
| Word-pooling network, MSE | 0.432245 | Useful ensemble diversity |
| Word-pooling network, Huber | 0.432122 | Little individual improvement |
| Binary sparse MLP, Adam + global L2, Huber | 0.417217 | Best at epoch 3 |
| Binary sparse MLP, row-sparse Adam, Huber | 0.416794 | Best at epoch 2 |
| Binary sparse MLP, row-sparse Adam, 64 price bins | 0.417463 | Complementary objective |
| Continuous sparse MLP, row-sparse Adam, Huber | 0.420258 | Marginal ensemble benefit; omitted |

The unrounded six-expert blend scored **0.396718** on the 74,126-row selection
window. A LibTree calibrator reduced that to **0.395657**. Adding four explicit
disagreement features changed the best result by only about 0.00008; that extra
feature variant is not enabled. The released configuration uses simple rounded
weights and is audited separately. CNN and repaired-token pooling candidates
were stopped after weak gains relative to their cost; they are not final models.

Huber and row-sparse optimization change both optimization behavior and
regularization. Their differences are not an isolated causal estimate of the
optimizer alone. Timings from concurrent exploratory runs are not used to claim
a controlled speedup. Repeated inference reports preserve arithmetic mean,
minimum, maximum, standard deviation, and every observation.

## Reproduce

Use Linux, Python 3.12, a C++17 compiler, and the official competition files. See the
[data download instructions](README.md#reproduce-locally). Large inputs,
predictions, caches, and models belong under ignored `benchmarks/runs/`.

```sh
python -m pip install '.[mercari]' kaggle
PYTHONPATH=python:. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/optimized-config.json \
  --output-dir benchmarks/runs/mercari/optimized-final --mode final
```

The output directory must be empty. Outputs include the submission CSV, input
and source hashes, versions, timing report, model pickle, a readable tree,
JSON tree structure, and an explanation for the first actual test listing.
Use `--omit-model-artifact` to retain predictions and explanations without
saving the large pickle.

To reproduce the fixed local protocol, use `--mode study`. It reports selection
RMSLE without evaluating the holdout. Add `--audit-holdout` only after freezing
decisions. Retraining and the audit are substantial full-data computations.

Build and review the source-only private Kaggle script:

```sh
python -m benchmarks.mercari_kernel --owner YOUR_USERNAME \
  --optimized-config benchmarks/mercari/optimized-config.json \
  --output-dir benchmarks/runs/mercari/optimized-kernel
kaggle kernels push -p benchmarks/runs/mercari/optimized-kernel -t 3600
```

Record the version returned by the push. After the full Kernel completes,
submit its `submission.csv` using that exact version, as described in the
[journal](README.md). The builder does not submit automatically.

## Understanding the prediction

RMSLE compares `log1p(price)`, so learning that quantity aligns regression with
the official metric. Ridge learns additive word effects. Factorization models
learn effects such as **a word combined with a brand** using small vectors.
Sparse MLPs learn nonlinear combinations of word and category features. Huber
reduces the influence of large errors; price-bin classification predicts a
distribution and converts it to an expected log price.

The weighted experts produce a base log price. LibTree predicts a correction
from metadata and held-out expert predictions. A correction of `0.1` multiplies
`price + 1` by approximately `1.105`. Exported paths explain this residual
component; the full explanation also reports weighted expert contributions.
Those contributions sum on the **log-price scale**, not in dollars, and are
not causal effects or SHAP values.

For tree fundamentals, see the offline [interactive learning guide](../../docs/LEARNING_GUIDE.html)
and [native prediction-path guide](../../docs/explain.md).

## Tests and references

Tests cover train-only preprocessing, legacy refits, scalar/model tokens,
group separation, dense/sparse numerical equivalence, exact factorization
arithmetic, worker synchronization, sparse optimizer updates, stable soft
labels, serialization, contribution sums, final-only execution and ZIP input
handling. Line coverage for Numba functions uses executable Python kernels
with `NUMBA_DISABLE_JIT=1`; compiled parallel kernels are tested separately.
This is statement coverage, not a claim of 100% branch coverage. The C++ core
retains its separate sanitizer, coverage, worker-lifecycle and memory checks.

1. [Jankiewicz & Lopuhin, first-place Mercari technical presentation](https://github.com/pjankiewicz/mercari-solution/blob/374301ad1c32cbc93dcc40313d5d7bb9c5503746/presentation/5_modeling.tex): sparse MLPs, robust/bin losses, increasing batches and ensemble diversity. Independent Keras 3 implementation here; the authors' reported scores are not our measurements.
2. [Chenglong Chen, fourth-place Mercari report](https://github.com/ChenglongChen/tensorflow-XNN/tree/6534a832f5b4461cbdf1ebbdf5620a0cd80f0aa9): multi-field models and sparse optimization.
3. [Hestenes & Stiefel (1952), conjugate gradients](https://doi.org/10.6028/jres.049.044).
4. [Micci-Barreca (2001), smoothed categorical encoding](https://doi.org/10.1145/507533.507538).
5. [Rendle (2010), factorization machines](https://doi.org/10.1109/ICDM.2010.127); [Juan et al. (2016), field-aware factorization](https://doi.org/10.1145/2959100.2959134).
6. [Stich (2019), local SGD](https://openreview.net/forum?id=S1g2JnRcFX).
7. [Huber (1964), robust estimation](https://doi.org/10.1214/aoms/1177703732); [Kingma & Ba (2015), Adam](https://arxiv.org/abs/1412.6980).
