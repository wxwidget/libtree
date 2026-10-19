# Global linear, multilayer MLP and conditional linear fusion

[中文](FUSION_STRATEGIES.zh-CN.md) · [Results](fusion-strategies-results.json) · [Frozen inputs](fusion-candidates.json)

## Question and inputs

Does a nonlinear or category-dependent outer fusion improve over a single
global linear layer? All strategies use the same **72 cached prediction columns**:
25 frontier, 13 small-text/product-calibration, 26 CPU-view, six new sparse
experts and two sparse residual branches. These are prediction columns, not 72
independent end-to-end training pipelines. Some share upstream models or differ
only in residual strength; their errors are strongly correlated.

"Single layer" describes the new outer fusion. Its inputs can already contain
neural networks or residual trees. This experiment does not flatten those
underlying algorithms into linear models. No standalone-score filter removes
weak inputs. A second pool removes the six new sparse experts **and both of
their residual descendants**, leaving 64 columns. This is a family ablation,
not a causal attribution to Lasso or Elastic Net individually.

Previously selection-fitted aggregate inputs, including the best anchor, are
excluded from fusion training. The published manifest pins every source archive
by SHA-256, packed column index where applicable, declared fit stage, family,
and all four prepared-data file hashes. These fit-stage declarations describe
trusted local artifacts; hashes verify identity, not the truth of training history.

## Three requested strategies

A global linear layer gives an expert the same weight for every listing. An MLP
learns nonlinear interactions through hidden transformations; ReLU keeps positive
values and replaces negative ones with zero. Conditional linear fusion keeps a
readable linear formula inside each sufficiently large product category. More
parameters offer flexibility, and also more opportunity to fit noise.

| Strategy | Form on log prices | Fixed behavior |
| --- | --- | --- |
| Single global linear layer | `z = sum(w[j] * p[j]) + b` | Nonnegative weights, sum 1; bias in [-.25,.25]; squared clipped-log error plus 1e-4 shrinkage |
| Multilayer MLP | `z = mean_y + std_y * MLP(standardize(p), category_onehot)` | Two ReLU hidden layers, 32 and 16 units, linear output; Adam .001; batch 512; sklearn L2 alpha .1 |
| Conditional linear | `z = sum(w[category,j] * p[j]) + b[category]` | Global affine prior; category-specific weights with at least 2,000 fit rows; local penalty .01; rare/unseen groups use global |

The first strategy uses `RmsleBlend`. Conditional models use the existing
`ContextBlend`; its global model is `RegularizedBlend`, with nonnegative weights,
sum in [.8,1.2], bias in [-.25,.25] and penalty .001. An additional **global affine
control** uses that same global model, so the conditional comparison is not
confounded solely by differing weight constraints. It is a diagnostic control,
outside the three primary strategy choices.

MLP and conditional fusion receive the same label-free top-level category.
The global models use one set of weights for all categories by definition.
All methods receive every prediction column in their pool. The MLP directly
predicts log price; it has no anchor skip connection or residual-tree stage.
Final prices are `expm1(max(z, 0))` and RMSLE uses this same clipping.

This compares operational strategies, not the isolated effect of hidden layers:
MLP versus global linear also adds category information. MLP versus conditional
linear shares that context. A prediction-only MLP would be an additional control
before attributing the full gain specifically to network depth.

## Validation protocol and its limits

The original data windows remain base 1,186,026 / calibration 74,129 /
selection 74,126 / confirmation 148,254. No base expert is retrained here.
Selection is subdivided **by identical-listing feature groups** into approximately
60% meta-fit / 20% checkpoint-tune / 20% strategy-check, with fixed seeds 703/704.

1. All fusion methods fit only meta-fit labels. Input standardization and category
   vocabulary fit that same window. A best-fit expert is moved to the first column
   as the common shrinkage prior; meta-check labels do not determine that prior.
2. MLP checkpoint selection uses checkpoint-tune only: at most 100 epochs,
   patience 12, minimum RMSLE improvement 1e-6. It restores the best checkpoint.
   Seeds 42,43,44 check stability; **42 is the predeclared primary**, not the
   best-performing seed chosen afterward.
3. The three primary strategies are compared on strategy-check. Ties within
   1e-5 prefer global linear, then conditional, then MLP. The 64-column ablation
   does not choose the primary winner. Three-seed MLP averaging is diagnostic.
4. Every strategy is refit on all selection rows; each MLP uses its frozen epoch
   count, with no new early stopping. All models, choices, timings and promotion
   are saved to `frozen.json` **before opening confirmation labels**.
5. The chosen proposal must beat the old anchor by 1e-5 on both strategy-check
   and full selection to pass the final promotion check; otherwise its output
   arrays are preserved exactly. The historical anchor is only a reference.

**This is a reused local audit, not an unbiased new generalization estimate.**
Earlier candidate hyperparameters, checkpoints and some upstream mixture weights
were selected on this selection window; the confirmation window was inspected
repeatedly. Removing final aggregates and separating the new meta-label roles
does not undo that adaptive reuse. In addition, the anchor's old outer weights
already fitted the strategy-check rows; its check score is historically in-sample.
The new models do not train on their own meta-check labels during strategy
selection, but the entire pipeline cannot be called untouched or fully OOF.
No new Kaggle score is inferred from these numbers.

## Measured results

The primary pool has 72 inputs. The new meta-fit/tune/check split contains
**44,477 / 14,825 / 14,824 rows**. All fusion models refit on 74,126 selection rows
before the 148,254-row audit is opened.

| Fusion | Strategy-check RMSLE | Refit audit RMSLE | Meta-fit + refit seconds | Mean inference seconds |
| --- | ---: | ---: | ---: | ---: |
| Single-layer linear | 0.387702 | 0.380609 | 0.674 + 0.908 | 0.038 |
| MLP 32/16, seed 42 | 0.387926 | 0.379886 | 5.202 + 4.879 | 0.486 |
| Conditional linear | 0.388067 | 0.380704 | 1.415 + 1.845 | 0.148 |
| Global affine control | 0.387766 | 0.380725 | 0.612 + 0.717 | 0.020 |

**The MLP is best among the three primary audit results**, improving RMSLE by
**0.190%** relative to single-layer linear. Its conditional paired row-bootstrap
95% interval for the RMSLE difference is **[-0.000985, -0.000502]**. This interval
conditions on this fitted seed and reused data; it excludes seed variance and
adaptive selection. The row bootstrap treats audit rows as independent rather
than resampling identical-listing groups. MLP prediction is **12.9 times** the linear duration across these recorded calls.

**The pre-audit strategy check instead chose global linear.** Its frozen output
is **0.380608699**, versus the historical anchor's **0.380616255**. That difference
is only **-0.000007555**, with interval **[-0.000051356, +0.000045556]** crossing zero.
It passes the predeclared local selection gate, but does not establish a reliable
improvement over the anchor. Original anchor files remain unchanged.
The better MLP audit outcome does not retroactively replace the frozen choice.
No new Kaggle submission or score was produced.

| MLP seed | Selected epochs | Audit RMSLE |
| --- | ---: | ---: |
| 42 | 36 | 0.379886 |
| 43 | 39 | 0.381149 |
| 44 | 34 | 0.380159 |

The **arithmetic mean** of the three single-model RMSLEs is **0.380398**,
standard deviation **0.000543**, worst **0.381149**. The arithmetic mean
of their **log predictions** scores **0.379682**, a **0.243%** reduction versus linear;
this bag is diagnostic, not the frozen selected strategy. Its separately timed
three-model call averages **1.010s**, maximum **1.043s**,
standard deviation **0.025s**. This was a later timing pass;
cache/system effects mean it should not be divided by the earlier single-model
call to infer an exact batching or parallel speedup. Saved predictions replay exactly.

| Fusion | Mean seconds | Max seconds | Std seconds |
| --- | ---: | ---: | ---: |
| Single linear | 0.038 | 0.069 | 0.020 |
| MLP seed 42 | 0.486 | 1.183 | 0.312 |
| Conditional | 0.148 | 0.166 | 0.017 |
| Affine control | 0.020 | 0.023 | 0.002 |

All calls, including slow first observations, are retained. The MLP has a 1.183s
observation while its other five calls are about .334–.374s. Global simplex and
affine prediction use the same weighted-vector kernel; their observed timing
difference should not be interpreted as different algorithmic efficiency.
Cache, memory and scheduling effects are possible explanations, not diagnosed
causes. The ratio above describes these samples, not a stable hardware guarantee.

All inference samples cover **222,380 rows** (selection + audit), with six
recorded calls across two equivalent timing passes, two CPU threads, cached upstream predictions and category strings. MLP
scaling/category encoding is included; raw-text preprocessing, model loading and
all upstream inference are excluded. Individual call samples/maxima/deviations
are in the JSON. Fit/refit times are means of two observations per model.
MLP total meta-fit+refit time over six seed/run observations averages **9.806s**,
standard deviation **1.165s**, maximum **11.791s**.

The full-selection conditional model fits eight local categories. It slightly improves over
its global affine control (**0.380704 vs 0.380725**), but loses to the simpler
simplex blend. The conditional-vs-affine paired interval is
**[-0.000079350, +0.000033579]**.
The constraints and shrinkage are part of the result; category conditioning
alone does not explain every difference.

For inspection, the global simplex model has ten materially nonzero weights;
the conditional model exposes global and eight category-specific weight sets.
The MLP is less directly interpretable. Exact coefficients/category weights and
its parameter count are recorded in the JSON, not inferred from the score.

**Sparse-family ablation (audit):**

| Fusion | All 72 inputs | Without eight new sparse inputs |
| --- | ---: | ---: |
| Single-layer linear | 0.380609 | 0.380609 |
| MLP 32/16, seed 42 | 0.379886 | 0.380026 |
| Conditional linear | 0.380704 | 0.380690 |

The eight sparse inputs receive effectively zero global linear weight. MLP can
use them indirectly, but the small average gains are not uniform across seeds.
Removing columns also changes input dimension and initialization, so this is a
family ablation, not a causal feature-importance measurement.

**All audit categories:**

| Category | Rows | Linear | MLP seed 42 | Conditional |
| --- | ---: | ---: | ---: | ---: |
| <missing> | 633 | 0.471591 | 0.468628 | 0.471808 |
| beauty | 20,706 | 0.364586 | 0.363614 | 0.364532 |
| electronics | 12,223 | 0.374784 | 0.375187 | 0.374872 |
| handmade | 3,045 | 0.393984 | 0.392456 | 0.394124 |
| home | 6,936 | 0.412809 | 0.410428 | 0.413345 |
| kids | 17,243 | 0.395329 | 0.394516 | 0.395481 |
| men | 9,212 | 0.383450 | 0.382771 | 0.383649 |
| other | 4,550 | 0.435832 | 0.435241 | 0.435248 |
| sports & outdoors | 2,475 | 0.391524 | 0.390072 | 0.392062 |
| vintage & collectibles | 4,681 | 0.496333 | 0.494759 | 0.496818 |
| women | 66,550 | 0.362929 | 0.362448 | 0.362986 |

MLP improves ten of eleven categories here; Electronics worsens. The high-error
Vintage/Collectibles and missing-category groups remain visible. Global RMSLE
was independently reconstructed from category counts and squared errors.

The verification pass took **113.72s**, following the initial **108.94s** pass,
for **222.66s** in recorded experiment phases, with **1313.9 MiB** peak
process RSS. It includes meta-training, full refits, repeated inference, saved
artifacts and bootstrap calculations; it excludes imports and cached base work.
The separate bag timing, editing and regression tests are outside those timers.
A final refactor separated reference-file validation from training eligibility,
removing an inaccurate temporary fit-stage tag on the reference reader. All
12 saved model outputs replay exactly across both windows; epochs, choices and scores are unchanged.
The two timing passes do not create new independent accuracy repetitions.

After refitting on more rows, the strategy ranking reverses. The MLP also takes
more Adam updates at the same epoch count: 87 batches per meta-fit epoch versus
145 per full-selection epoch. This is a possible contributor, not a demonstrated
cause. The proper next check is repeated/nested grouped validation or a new
untouched split, not choosing a seed on this audit.

The final CPU optimization regression passes **183 tests in 208.64 seconds**.
Statement coverage is **3,347/3,347 across 25 modules (100%)**, including all
**221 statements** in the new fusion module. TDD checks held-out nonlinear/category
learning, fit-only preprocessing, best-checkpoint restoration, fixed-epoch refit,
exact pickle/prediction replay, provenance/ID/hash/finite guards, reference reading
separated from training eligibility, grouped role isolation, pre-audit freeze and
exact fallback. All 12 saved outputs were independently recomputed with official
RMSLE; category totals reconcile. Both timing passes have identical fitted
coefficients and outputs. The suite reports 492 existing warnings. This is CPU
optimization-suite statement coverage, not whole-repository, branch or GPU coverage.
**Validation: Share with caveats**, because historical selection/audit reuse remains.

## Reproduce and inspect

From the repository root, install the benchmark environment and reconstruct the
cached artifacts from [Frontier](FRONTIER.md), [small-text](SMALL_TEXT_REPORT.md),
[CPU multiple-view](CPU_MULTIVIEW.md), [sparse](SPARSE_PENALTIES.md) and
[residual](RESIDUAL_FUSION.md) studies. Data, large checkpoints and secrets are
ignored by Git; the manifest and reports are committed. Then run:

```bash
python -m pip install '.[mercari,benchmark]' pytest pytest-cov
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m benchmarks.mercari_fusion \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --manifest benchmarks/mercari/fusion-candidates.json \
  --anchor benchmarks/runs/mercari/smalltext-t4-001/primary-blend/ensemble.npz \
  --output-dir benchmarks/runs/mercari/fusion-strategies-reproduction
```

The output directory must be new. The committed manifest replays the exact
cached study, not an arbitrary fresh refit. Rebuilt stochastic base artifacts may
have different hashes; regenerate a reviewed manifest using each new archive's
SHA-256 and its actual fit stage. A manifest expert has `name`, `path`, `sha256`,
`fit_window` (`base` or `calibration`), `family`, and optional packed `column`.
Every archive needs aligned `selection`, `confirmation`, and corresponding
`*_ids`. A directly selection-fitted aggregate is rejected as a training input.
Do not relabel it as base-only to bypass the check.

Each pool saves `models.pkl` (ordered candidate names and all six fusion models),
ID-aligned prediction archives, and the full fit/tune history. The top directory
contains `plan.json`, `frozen.json`, `report.json` and `ensemble.npz`. Raw prediction
columns are ordered by the saved names. `report.json` contains arithmetic-mean
inference samples, three-seed stability, paired comparisons and every audit
category, including small/high-error groups. Global RMSLE is the square root of
the row-weighted mean squared log error, not an average of category RMSLEs.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest \
  tests/test_mercari_fusion.py --cov=benchmarks.mercari_fusion \
  --cov-report=term-missing --cov-fail-under=100
```

## References

- Wolpert (1992), *Stacked Generalization*,
  [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).
- Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization*,
  [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*,
  [JMLR 12:2825–2830](https://jmlr.org/papers/v12/pedregosa11a.html).

These support stacking, optimization and the implementation library. The category
shrinkage rules and grouped protocol are local adaptations, not a reproduced
paper benchmark or a guarantee that deeper fusion improves accuracy.
