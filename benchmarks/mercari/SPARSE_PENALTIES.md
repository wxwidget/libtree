# Ridge, Lasso and Elastic Net without new configuration

[中文](SPARSE_PENALTIES.zh-CN.md) · [Evidence](sparse-penalties-results.json) · [Original top-three review](TOP3.md)

This records the completed direct-fusion study. The subsequent automatic
[residual-fusion stage](RESIDUAL_FUSION.md) keeps the same raw experts and tests
their indirect contribution through signed stacking and residual boosting.

The Mercari experiment now automatically adds **Lasso and Elastic Net** to the
existing Ridge experts. No new CLI switches, configuration files, manual alpha
selection or hyperparameter search are required. The new experts use the
**same unordered-pair view as Ridge**, selected as the strongest new lexical
view in the previous study. The other three Ridge feature ablations remain.
This is a limited extension, not an exhaustive model/view grid.

## Built-in behavior

All models fit base-window log1p prices. Hashing, tokenization, IDF, metadata,
train/selection/audit rows and prediction clipping for RMSLE are shared.
The new coordinate-descent models receive CSC matrices; inference uses CSR.
Neither densifies the document matrix. Each model and its predictions are saved.

The new [solver](../mercari_penalties.py) preserves the Lasso/Elastic Net
objectives and an **unpenalized intercept**, using sklearn-compatible subclasses.
It maintains a raw residual vector and its sum, so each coefficient update
touches only the column's nonzeros. Means are handled algebraically; no dense
sample-by-feature matrix or per-feature full residual shift is needed. A
primal/dual gap checks convergence every five sweeps and at the budget limit.
Numba compiles the loop; Python-mode coverage and compiled reference tests both run.
The new solver uses float64 arithmetic, including its residuals and coefficients;
Ridge retains its previous float32 feature path. Solver, precision and tolerance
differences must be considered when interpreting runtime and accuracy.

This was needed because sklearn 1.8's sparse centering path in
[`_cd_fast.pyx`](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/linear_model/_cd_fast.pyx)
contains full-sample residual loops inside feature processing. The first native
attempt was stopped after **475 seconds total**, before Lasso finished, with no
Lasso/Elastic Net quality result. That failed attempt is retained in the evidence;
an unfinished run does not support an exact speedup ratio. Small-data reference
tests check predictions and coefficients against sklearn with correlated,
nonzero-mean, constant and zero columns, with/without an intercept. The new
solver supports fresh cyclic fits without sample weights, positivity, warm starts
or a precomputed Gram matrix; unsupported exposed options are rejected explicitly.

With `n` base rows, built-in Lasso uses `alpha=20/n`; Elastic Net uses
`alpha=40/n`, `l1_ratio=.5`. Both use cyclic coordinate descent, at most 200
iterations and tolerance 1e-4. This translates fixed **sum-loss** penalties into
sklearn's **average-loss** convention:

| Model | Objective, omitting the unpenalized intercept |
| --- | --- |
| Default Ridge | ½ SSE + 10 ‖β‖₂² |
| Lasso | ½ SSE + 20 ‖β‖₁ |
| Elastic Net | ½ SSE + 20 ‖β‖₁ + 10 ‖β‖₂² |

The constants are declared before selection, not tuned on selection or audit
labels. Normalizing by row count preserves these sum-loss coefficients; it
does **not** guarantee optimal regularization across datasets, equal effective
complexity, or statistically equivalent penalties. Existing Ridge-only CLI
settings remain for backward compatibility and do not alter the two new
models' built-in penalties.

Lasso can set coefficients exactly to zero. Elastic Net adds an L2 component
that can share weight among correlated features. Neither property guarantees
better RMSLE or faster end-to-end execution. sklearn still stores dense
coefficient arrays here; coefficient sparsity is not compressed model storage.

## Selection, ablations and convergence

The default run produces four Ridge experts plus `lasso_pairs` and
`elasticnet_pairs`. The outer pool checks simplex-with-intercept, affine and
category-aware blending on grouped selection halves. It reports seven pools:
anchor, without pairs, without quantities, Ridge-only, without Lasso, without
Elastic Net, and all experts. It freezes every choice and the promotion gate
before opening confirmation labels. A rejected proposal preserves exact anchor
predictions. Existing window reuse still limits generalization conclusions.

For each model, `report.json` includes effective parameters, iteration count,
nonzero coefficient count, dual gap when available, all captured fit warnings,
and whether a coordinate-descent convergence warning occurred. Reaching the
iteration budget is recorded separately. Ridge's LSQR wrapper does not expose
the complete stopping status; its `converged` field is null rather than an
unsupported success claim. Nonconverged experiments are not silently labeled
as solved optimization problems.

Inference runs **three times** on prepared matrices. The primary timing is
their **arithmetic mean**, with all samples, maximum and population standard
deviation retained. Fit time is one training observation per model. Shared text
preparation and CSR/CSC assembly are recorded separately. Timings exclude old
anchor training and cannot establish a whole-library performance ranking.

## Full-base results

The same **1,186,026 training rows / 524,288 pair-view columns** are used for
all three models; selection has 74,126 rows and reused audit 148,254.

| Same pair view | Selection RMSLE | Reused audit RMSLE | Nonzero coefficients | Fit s (one run) | Mean inference s (3 runs) | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.473492 | 0.472611 | 405,988 | 16.22 | 0.0439 | 30 |
| Lasso | 0.515471 | 0.515175 | 3,474 | 128.49 | 0.1665 | 200 |
| Elastic Net | 0.522837 | 0.522774 | 3,916 | 116.96 | 0.0956 | 200 |

Inference covers the same 296,509 calibration/selection/audit rows. Maximums,
standard deviations and all timing samples are in the evidence JSON. Counts
include all hash columns, including unobserved ones; they do not measure semantic
word importance. The four Ridge controls replay the preceding run exactly:
maximum prediction difference **0** across every window.

Coordinate-descent convergence reports: **lasso_pairs: False, dual gap 0.000144145; elasticnet_pairs: False, dual gap 0.000114113**. The built-in Lasso alpha
is 1.68630367e-05, Elastic Net alpha
3.37260735e-05; these follow row-count conversion,
not validation tuning. Different solvers/tolerances and one training observation
prevent a universal solver-speed conclusion.

**The selection gate rejected the extension; exact anchor predictions are retained.** Baseline audit RMSLE is
**0.380616255**, retained output
**0.380616255**. See seven family/feature ablations and
selection-side method checks in the evidence JSON. No audit result chose the
promotion decision and no new Kaggle submission was made.

The recorded incremental study phase took **481.93 s**, including
**162.57 s** of text features and
**3.95 s** of shared matrix assembly;
peak RSS was **3579.34 MiB**. CPU only, with four-thread limits;
coordinate descent is not itself a four-worker training implementation.
The interrupted native attempt adds **475 s**; measured experiment time totals
**about 957 s**, excluding regression tests and the old anchor refit.
The study timer excludes Python imports and final audit bootstrap/output writing.
Both new experts hit 200 iterations without reaching the declared tolerance.
Their reported metrics describe this fixed budget, not fully converged optima
or evidence that Lasso/Elastic Net cannot work well with other defaults.

Validation: **169 tests passed**, **23 selected Mercari CPU modules / 2,992 statements / 100% statement coverage**. The two changed solver/runner modules cover **249/249 statements**. This is not whole-repository or branch coverage. Another **16 tests passed with JIT enabled**, including sklearn reference agreement; existing dependency warnings remain.

## Run

Reuse the prepared windows and anchor from [the preceding study](SMALL_TEXT.md).
The command is unchanged; both new models run automatically:

```bash
NUMBA_DISABLE_JIT=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m benchmarks.mercari_top3 \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --anchor benchmarks/runs/mercari/smalltext-t4-001/primary-blend/ensemble.npz \
  --output-dir benchmarks/runs/mercari/sparse-penalties-reproduction-001 \
  --external-pretrained-data

NUMBA_DISABLE_JIT=1 python -m pytest \
  tests/test_mercari_top3.py tests/test_mercari_penalties.py \
  --cov=benchmarks.mercari_top3 --cov=benchmarks.mercari_penalties \
  --cov-report=term-missing --cov-fail-under=100
NUMBA_DISABLE_JIT=0 python -m pytest tests/test_mercari_penalties.py
```

Only data, anchor and output paths must be supplied. The pretrained flag
declares the old anchor's lineage; the new linear experts use no pretrained
weights. Use a new output directory. Generated data/models remain ignored by
Git. The configured Mercari environment already contains sklearn, SciPy and
Numba; no additional package installation is needed there.

For a beginner: Ridge shrinks coefficients; Lasso can remove them; Elastic Net
combines both penalties. A nonzero coefficient count measures how many hashed
feature columns remain active, not how many semantic words are important:
hash collisions and correlated features limit that interpretation. Better
individual accuracy is useful only if the expert adds information to the
existing ensemble; the promotion gate tests that separate question.

References: [Tibshirani, Lasso (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x),
[Zou & Hastie, Elastic Net (2005)](https://doi.org/10.1111/j.1467-9868.2005.00503.x),
and [Friedman, Hastie & Tibshirani, coordinate descent (2010)](https://doi.org/10.18637/jss.v033.i01).
These motivate the implementation; published research results are not our measurements.
