# Market-aware models before fusion

[中文](MARKET_MODELS.zh-CN.md) · [Measured results](market-models-results.json) · [Final-loss fusion study](FINAL_OBJECTIVE.md)

## The statistical target

A non-Gaussian marginal price distribution does **not** invalidate a decision
tree or log-squared regression. Let `Y=log1p(price)`. Under squared log error,

```text
argmin_a E[(Y-a)^2 | X] = E[Y | X]
price_prediction = expm1(max(E[Y | X], 0))
```

This conditional-mean identity requires a finite second moment, not Gaussian
prices or homoscedastic noise. RMSLE is the root of the aggregate log-squared
error, with the same unregularized optimum. Gaussianity of a marginal target
and distributional adequacy of conditional errors are different questions.

A lognormal retransformation correction such as adding `variance/2` targets
an arithmetic-dollar mean and changes the competition target. Raw-price
Gamma/Tweedie objectives, quantile predictions and bucket-balanced weighting
also target different functionals or populations. They may be useful for a
different business decision, but skew alone does not justify replacing RMSLE.

## What the base data says

The profile uses **all 1,186,026 base-training listings**, with no audit labels.

| Quantity | Raw price | log1p(price) |
| --- | ---: | ---: |
| Arithmetic mean | 26.729896 | 2.979128 |
| Standard deviation | 38.613703 | 0.748730 |
| Skewness | 11.499688 | 0.660284 |
| Excess kurtosis | 289.084293 | 1.092553 |

The price 90th/99th/99.9th percentiles are **$51 / $170 / $450**; maximum
**$2009**. There are **699 zero-price rows** and **42.66% missing brands**.
Log transformation greatly reduces marginal skew, without making a normality
assumption or establishing Gaussian conditional errors.

In Electronics, titles containing the simple `case|cover|protector` accessory
regex have mean price **$13.43** (19,748 rows); titles containing neither those
terms nor the damage regex average **$40.40** (78,115 rows). These are approximate
lexical slices, not verified devices, matched comparables or a causal discount.
The damaged-title slice actually averages $47.93, showing why unadjusted word
slices cannot establish a condition effect.

The observed patterns support product heterogeneity and a sparse upper tail.
They do not establish a parametric conditional distribution or a causal effect
of brand, condition or shipping. Feature-conditioned demand, sale speed, images,
calendar time, seller histories and verified retail prices are unavailable here;
we cannot estimate them from the available columns.

Secondhand pricing needs comparable item identity and generation, genuine item
versus accessory/empty packaging, condition evidence, capacity/size and bundle
quantity. A package's target remains its package price: dividing every bundle
label by a guessed quantity would change the task. Title and description cues
are approximate, and contradictory wording or an unverified claim remains noisy.

## Changes to one upstream model

The fixed base expert is `frontier_sparse512_binary_huber`, fitted on the
original base window. The existing base-fitted LSA encoder is reused. Each new
head consumes **one** expert log prediction plus listing-derived features;
there are no other experts or final-ensemble predictions in its inputs.
The text trunk, final fusion model and final fusion weights are unchanged.
This study refits a pre-fusion prediction head, not the original sparse network.

The reference and seven alternatives are all retained:

| Variant | Training target / inputs |
| --- | --- |
| Raw text reference | Existing single base prediction |
| Plain L2 | Log residual versus that prediction; base LSA/metadata + one prediction |
| Legacy L2 | Same target plus existing lexical attributes |
| Market L2 | Same target plus separate title/description market cues |
| Market Huber | Same rich inputs; a robust-loss control |
| Market regime | Soft four-state gate and conditional log means; same rich inputs |
| Comparables L2 | Market inputs plus base-only comparable-item means/support/dispersion |
| Comparables regime | Same comparable-item inputs with the soft four-regime head |

Market cues distinguish damage tokens from nearby negations (`no cracks or
scratches`), retain explicit failure phrases (`does not work`), and encode
accessory/empty-box terms, bundle quantities, GB/TB capacity and redactions.
Quantities/capacities are capped; unknown quantity is one. These are bounded
lexical cues, not a complete negation parser or verified product facts. Existing
attributes stay present, so the market-feature comparison is an additive ablation.

Comparable-item statistics reuse `ProductStatistics`: normalized title,
unordered identity words, title prefix, brand/category/condition and model-number
keys, with a log mean, support and dispersion per key (15 added columns).
Only base-training labels fit those maps, smoothing is 20 and the prior backs
off to brand/category parents. Query prices never enter their features. The
existing component has independent leakage, hierarchy and unknown-key tests.

The distribution head separates zero, positive-below-$10, $10–<$50 and ≥$50:

```text
p[k]    = Pr(regime=k | observable features)
mu[k]   = E[log1p(price) | observable features, regime=k]
pred_y  = sum(p[k] * mu[k])
```

The zero component has mean zero. Probabilities use **unweighted** cross entropy;
class balancing would change the posterior. Positive heads learn log means,
shrink toward their training-only regime means with `n/(n+500)`, and stay within
the regime's log bounds. Fewer than 200 rows triggers a constant-mean fallback.
Unobserved regimes have zero probability; frozen zero-step heads do not grow on
full refit. Predicting needs no actual price. This is a conditional-mixture
experiment, not Gaussian-mixture likelihood estimation or an NGBoost implementation.

## Controlled protocol and costs

- Original grouped base/calibration/selection/audit windows are unchanged.
- Within 74,129 calibration rows, grouped seed 809 makes 59,305 fit and 14,824
  checkpoint-tune rows. All heads use the same split and fixed seed 42.
- L2/Huber heads allow 450 rounds, 31 leaves, learning rate 0.05, minimum leaf
  support 80, L2 penalty 10 and feature fraction 0.85. Tune early stopping uses
  the final clipped RMSLE, patience 40.
- The regime gate/heads allow 180 rounds each, 15 leaves, minimum support 80
  and L2 penalty 20. Gate checkpoints use cross entropy and conditional heads
  use within-regime squared log error, patience 25. These are surrogates for
  the joint expected-log output, not direct end-to-end RMSLE training.
- Full calibration refits use the frozen iteration budgets. The selection
  window chooses a head, with a 1e-5 tie favoring the earlier/simpler candidate.
  All complete refits and predictions freeze before audit labels are read.
- Nothing is oversampled, target-weighted or deleted. Zero prices stay in every
  metric. Actual-price bins are post-fit diagnostics, never inference routing.
- All runs use two CPU threads. Three inference samples report arithmetic mean,
  maximum and standard deviation over all three evaluation windows. They include
  the head only; LSA/market feature construction is timed separately. Cached text
  trunk training/inference and original LSA fitting are excluded from both.
- Gate/conditional models have different work from a single residual tree.
  Equal iteration counts would not be equal compute; report measured costs.

The comparable-item views were added after inspecting the first five-head
pilot; its original source and results are preserved. This is sequential research,
not a fully preregistered blind test. Historical windows were already reused.

The encoder and model pickles are trusted local artifacts. Ordered IDs, complete
data fingerprints, base-only provenance, encoder/source hashes and a new output
directory are required. Models and competition data remain ignored by Git.

## Results and what to do next

The strategy check chooses **Comparables L2**, improving the single-branch
audit from **0.407564973 to 0.405881799 (0.413%)**. Plain L2 already reaches
**0.406066571**; comparable-item information supplies a further **0.0455%**
relative reduction. Its paired difference versus plain is **−0.000184772**, with
conditional 95% interval **[−0.000244272, −0.000121901]**. Against the raw branch,
the difference is **−0.001683174**, interval **[−0.001828920, −0.001499334]**.
These intervals do not include training/selection uncertainty.

| Model | Selection RMSLE | Audit RMSLE | Head fit + refit seconds | Mean head inference seconds |
| --- | ---: | ---: | ---: | ---: |
| Raw text | 0.409622 | 0.407565 | 0.000 | — |
| Plain L2 | 0.408411 | 0.406067 | 4.508 | 0.200 |
| Legacy L2 | 0.408497 | 0.406056 | 4.765 | 0.423 |
| Market L2 | 0.408508 | 0.406205 | 6.116 | 0.273 |
| Market Huber | 0.408504 | 0.406072 | 6.316 | 0.247 |
| Market regime | 0.410245 | 0.407323 | 22.858 | 4.489 |
| Comparables L2 | 0.408225 | 0.405882 | 5.553 | 0.234 |
| Comparables regime | 0.409637 | 0.406818 | 25.249 | 4.259 |

Inference covers **296,509 rows**, repeated three times. The accuracy head needs
**5.553s** incremental fit/refit and **0.234s** mean head inference; shared feature
preparation is **103.89s**. The whole eight-output study takes **225.57s**, peak
RSS **2620.7 MiB**, excluding imports and cached trunk/encoder fitting. The
comparable-item map itself is **319.7 MiB** on disk. Original raw output has no
new head, so its zero fit cost and missing head timing are not whole-model costs.

The four-state comparable head is **worse** (0.406818) and about **18.2× slower**
for head inference (4.259s) than comparable L2. The lexical-only feature addition
and Huber do not beat the plain control on selection. A probabilistic component
or more domain cues do not automatically improve the scored conditional mean.
The first six outputs replay the pilot exactly across all three windows; timing
changes between runs are not evidence of an optimization.

For deployment cost, plain L2 remains a reasonable simpler option. This study
selects the accuracy head by its declared score rule; it does not hide the extra
map memory, preparation time or the small advantage over plain L2.

| Actual price, USD | Rows | Raw text RMSLE | Comparables L2 RMSLE | Improvement |
| --- | ---: | ---: | ---: | ---: |
| 0–<5 | 3,622 | 0.699136 | 0.711421 | -1.757% |
| 5–<10 | 24,081 | 0.394227 | 0.402649 | -2.136% |
| 10–<20 | 55,585 | 0.321469 | 0.315169 | +1.960% |
| 20–<50 | 48,946 | 0.386006 | 0.380993 | +1.299% |
| 50–<100 | 11,565 | 0.531780 | 0.530787 | +0.187% |
| 100–<200 | 3,353 | 0.721238 | 0.721101 | +0.019% |
| 200–<500 | 993 | 0.946318 | 0.946405 | -0.009% |
| 500–<1000 | 94 | 1.159691 | 1.146775 | +1.114% |
| ≥1000 | 15 | 1.748024 | 1.733784 | +0.815% |

The gain is mainly in **$10–<$50**. **Below $10 worsens**, and $100–<$500 is
almost unchanged. The ≥$500 tail has only 109 rows. Zero-price audit RMSLE is
2.860868 versus raw 2.858185, so the hurdle does not establish a solution to zero
labels. This is a modest branch improvement, not a solved tail/zero problem.

Next priorities are group-excluded comparable retrieval and better product
identity/condition features in the text trunk; feature-based local models can
then be checked on observable slices. Verify zero-price semantics before any
training-only data policy. If prediction intervals are required, evaluate
conditional uncertainty separately from point RMSLE. Do not delete audit zeros,
route using actual target prices, or add a lognormal dollar-mean correction.


A favorable change here is a **single-branch** improvement; it is not a new
0.3793 complete-system score. Complementarity, extra serving cost and a fresh
fusion fit would still need separate validation. No Kaggle submission is made.
Historical selection/audit reuse and one training seed limit generalization.
Bootstrap intervals are paired rows conditional on fitted models, omit training
and selection uncertainty, and are not adjusted for trying multiple heads.

True-price-bucket over/underprediction can occur even for an optimal conditional
mean. It does not prove a deployable bucket correction exists. Future routing,
comparable-item retrieval and uncertainty features must use observable inputs,
with cross-fitted targets and group-held-out validation rather than audit fitting.

## Reproduce

Prepare the original cached base expert and encoder through
[the frontier workflow](FRONTIER.md). From the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m benchmarks.mercari_market_study \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --manifest benchmarks/mercari/fusion-candidates.json \
  --base-predictions benchmarks/runs/mercari/frontier-001/a100/joint/sparse512_binary_huber.npz \
  --encoder benchmarks/runs/mercari/frontier-001/trees/encoder.pkl \
  --output-dir benchmarks/runs/mercari/market-models-reproduction
```

No model tuning flags are required. The output directory must be new.
`plan.json` precedes fitting; each model pickle and ID-aligned prediction archive
is saved; `frozen.json` precedes audit access; `report.json` contains all scores,
price buckets, paired intervals, budgets and resource measurements.

The optional [seven-head manifest](market-head-candidates.json) retains every
new candidate, including the weaker standalone regimes, and passes the existing
fusion loader. It records the measured local paths and hashes. After reproducing
to a different folder, generate entries from that run's prediction files and
hashes. To study a larger pool, combine those entries with the original manifest
and follow the [same-input fusion workflow](FUSION_STRATEGIES.md). Do not overwrite
the historical 72-column manifest or treat this as a validated final-ensemble gain.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest \
  tests/test_mercari_market.py --cov=benchmarks.mercari_market \
  --cov=benchmarks.mercari_market_study --cov-report=term-missing --cov-fail-under=100
```

The CI-matched CPU benchmark regression passes **213 tests in 208.18s**.
Statement coverage is **3811/3811 (100%) across 28 CPU benchmark modules**,
including **231/231** in the two new modules, with no excluded lines. This is
not whole-repository, branch or GPU coverage. The new tests were written before
the implementation and comparable-view extension; they cover real tiny training,
negation/quantity cues, log-mean correctness, absent/rare regimes, frozen zero-step
fallbacks, serialization, ordered IDs, source/provenance guards and freezing before
audit access. Ruff formatting/critical checks pass. All eight metrics and 72
price-bucket scores were independently reconciled with `math.log1p`/`math.fsum`
and `pandas.cut`; all seven serialized heads replayed on target-free raw listings.
The optional seven-head manifest passes the existing fusion loader.

## Sources and attribution

- Gneiting (2011), *Making and Evaluating Point Forecasts*,
  [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138):
  match the point functional to the scoring rule; the conditional-mean identity
  above is also derived directly by expanding squared error.
- Jankiewicz & Lopuhin, [Mercari first-place modeling slides](https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/5_modeling.tex):
  regression via classification, Huber and interactions; their unsuccessful
  mixture-of-experts experiment is explicitly preserved rather than ignored.
- Micci-Barreca (2001), [high-cardinality target smoothing](https://doi.org/10.1145/507533.507538):
  comparable-item hierarchy is an independent base-only task adaptation.
- [LightGBM objective definitions](https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst):
  L2, Huber, multiclass log loss, Gamma and Tweedie have distinct targets/roles.
- Duan et al. (2020), [NGBoost](https://github.com/stanfordmlgroup/ngboost):
  probabilistic prediction is a separate extension when intervals/distributions
  are required; it was not installed or benchmarked in this experiment.

The rich cues, single-expert residual controls and bounded four-regime head are
independent task adaptations; no paper or contest score is claimed reproduced.
