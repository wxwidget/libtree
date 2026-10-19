# Train fusion against the final competition loss

[中文](FINAL_OBJECTIVE.zh-CN.md) · [Results](final-objective-results.json) · [Prior fusion comparison](FUSION_STRATEGIES.md)

## What changed and what was already correct

Mercari scores RMSLE, with lower values better. Let `y = log1p(actual_price)` and
let `z` be the final ensemble log prediction. We export `expm1(max(z, 0))`, so:

```text
RMSLE = sqrt(mean((max(z, 0) - y)^2))
```

The existing global and conditional linear models already minimize squared
RMSLE plus their declared shrinkage penalties. The old MLP learns standardized
log-price MSE with sklearn L2 regularization. Without clipping and regularization,
that data objective differs from squared RMSLE by a positive constant scale.
**The root and its square have the same unregularized minimizers.** Changing
the name or adding a square root is not, by itself, a better statistical target.
Putting a fixed regularizer outside a root would change its relative strength;
we avoid that confound during refinement.

The three old MLPs have no negative log predictions on selection or audit; their
minimum audit log predictions are about 1.08–1.18. Clipping was therefore inactive
on those saved outputs. The useful experiments are whether extra optimization
helps and whether optimizing the **final average**, rather than each member,
changes the result.

## Controlled objectives

All refinements use the same warm starts and the same full-window L-BFGS-B
optimizer. The network remains two 32/16 ReLU hidden layers, a linear output,
fit-only standardization, target scaling and category one-hot encoding. The
72 cached upstream prediction columns stay frozen.

| Loss | Exact fit objective | What it tests |
| --- | --- | --- |
| `log_mse` | `mean((z - y)^2)` | Control for switching optimizers and adding refinement steps |
| `squared_rmsle` | `mean((max(z,0) - y)^2)` | Exact final clipping, with a monotone metric surrogate |
| `rmsle` | `sqrt(mean((max(z,0) - y)^2))` | Exact global competition metric, including its root |

Each loss is tested in three ways:

1. **Single:** independently refine seeds 42,43,44; seed 42 remains the primary
   single-model comparison and the other two measure stability.
2. **Independent mean:** average the three separately refined log predictions.
3. **Joint mean:** optimize all three members together through their final log
   prediction mean. One fixed ensemble initialization uses seeds 42,43,44.

For a joint mean, the loss is `L(mean(z42,z43,z44), y)`, not
`mean(L(z42,y),L(z43,y),L(z44,y))`. Clip once **after** that mean. Even a member
whose individual error worsens can help the ensemble objective, so individual
scores are not used to discard members.

The analytic final-output derivative for RMSLE is:

```text
dL/dz[i] = (max(z[i],0) - y[i]) * I(z[i] > 0) / (n * RMSLE)
```

The gradient backpropagates through the mean (factor 1/member_count), each
member's target scale, and all dense weights/biases. At zero RMSLE choose the
zero norm subgradient; at the clipping boundary choose derivative zero. Numeric
directional derivatives test every layer for one, two and three members.

The denominator covers the **complete fusion fit window** at every optimizer
evaluation. Averaging minibatch RMSLEs would be a different objective. A caller
never needs to materialize `expm1` during optimization; the log-domain expression
is equivalent and avoids overflow.

## Protocol, regularization and cost accounting

The original base/calibration/selection/audit split and 72-column manifest are
unchanged. Within selection, the grouped meta-fit/tune/check windows contain
44,477 / 14,825 / 14,824 rows. Candidate generation and these windows have been
inspected historically; this is another conditional local study, not a new test.

- The three Adam warm starts fit only meta-fit labels, with the existing grouped
  checkpoint-tune early stopping. All loss variants share exactly those fitted
  starts. Fresh full-selection warm starts use their frozen epoch counts.
- Each refinement has at most **20 full-window L-BFGS-B iterations**, tune patience
  6 and minimum tune RMSLE improvement 1e-6. Iteration zero is eligible: if a
  refinement does not improve tune RMSLE, the warm start is retained.
- Refinement adds **no L2 term**. The shared warm start, iteration cap and tune
  early stopping provide regularization. The global/conditional controls retain
  their existing penalties and are labeled separately.
- Full-selection refinement uses the frozen chosen iteration budget with no new
  early stopping. An optimizer may converge earlier; requested/actual iterations,
  function evaluations and termination status are saved. A budget limit or tune
  stop does not discard a candidate.
- All 21 outputs are retained. Thirteen predeclared strategies participate in
  the final check; individual seeds 43/44 are stability diagnostics, not extra
  opportunities to pick the best seed. Ties within 1e-5 prefer earlier/simpler
  listed strategies. The 13 choices are global/conditional, warm seed 42/mean,
  three seed-42 loss variants, three independent means and three joint means.
- Every full fit and choice freezes before reading audit labels. Promotion needs
  improvement of 1e-5 on both strategy-check and full selection versus the
  previous frozen linear output. That reference historically fitted selection
  weights, so its check score is not an independent test. Its original file is
  never overwritten. This command never submits to Kaggle.

Warm-start fitting is shared and counted once. Refinement timings are incremental;
averaging wrappers have no fitting work of their own but depend on their three
trained children. Their inference timings include those children. Reporting a
zero wrapper fit time does not mean the whole ensemble is free to train.

## Measured results

All 21 frozen outputs were evaluated on the same reused 148,254-row audit.
Global/conditional controls replay the previous **0.380609 / 0.380704** exactly;
all three warm seeds and their mean replay exactly across selection and audit.
No upstream expert was refit.

| Method | Strategy-check RMSLE | Audit RMSLE | Extra refinement seconds | Mean inference seconds |
| --- | ---: | ---: | ---: | ---: |
| Original mean of three MLPs | 0.387656 | 0.379682 | 0.000 | 0.962 |
| Independent log MSE, then mean | 0.386845 | 0.379348 | 3.863 | 0.897 |
| Independent squared RMSLE, then mean | 0.386853 | 0.379345 | 3.536 | 0.950 |
| Independent RMSLE, then mean | 0.386846 | 0.379336 | 3.687 | 0.944 |
| Joint log MSE on final mean | 0.386984 | 0.379354 | 3.651 | 0.933 |
| Joint squared RMSLE on final mean | 0.386906 | 0.379354 | 4.153 | 0.968 |
| Joint RMSLE on final mean | 0.386936 | 0.379333 | 3.597 | 0.931 |

The shared Adam warm phase costs **21.048s** for the three meta-fit and
three full-selection fits, counted once. Extra times include meta-refinement and
full-selection refinement. Inference covers **222,380 rows**, three calls, two CPU
threads, and includes all required network members, scaling and category encoding;
it excludes cached upstream inference and raw text. All samples, maxima and
standard deviations are preserved in JSON. These are single-study observations,
not a demonstrated inference-speed improvement from the new loss.

**Joint RMSLE gives the lowest audit point score: 0.379333068**, versus the old
three-member mean **0.379682351**: **0.092%** lower RMSLE. But the same-optimizer
log-MSE control already reaches **0.379348195** independently and **0.379353638**
jointly. The extra root-specific joint difference is only
**-0.000020570**, with paired interval
**[-0.000052786, +0.000006837]** crossing zero.
The root-specific single/independent intervals also cross zero. Extra optimization
helps in this conditional study; a reliable advantage from the square root alone
is not established.

Joint RMSLE versus independently refined RMSLE differs by just
**-0.000002788**, interval
**[-0.000033160, +0.000026056]**. Joint training is implemented and
works, but its superiority over independent refinement is not established here.

| Loss, single models | Seed-42 audit | Three-seed arithmetic mean | Std | Worst |
| --- | ---: | ---: | ---: | ---: |
| log_mse | 0.379793 | 0.379738 | 0.000089 | 0.379808 |
| squared_rmsle | 0.379793 | 0.379725 | 0.000085 | 0.379793 |
| rmsle | 0.379780 | 0.379714 | 0.000078 | 0.379780 |

For context, the warm single-model arithmetic mean is **0.380398**, standard
deviation **0.000543**, worst **0.381149**. All three refined seeds improve in this
run. The joint candidate uses one fixed three-member initialization, so it does
not supply three independent ensemble trials.

**Frozen selection:** the strategy check selected **independent log-MSE mean**,
with audit **0.379348195**. The three independent-mean losses differ by less than
1e-5 on the check window and follow the predeclared tie order. The lower joint
RMSLE audit score did not retroactively replace this choice. Its output passes
the predeclared selection gate relative to the old frozen linear reference,
but no new Kaggle entry was submitted. All original reference artifacts remain.

All 21 final selection/audit outputs have **zero negative log predictions**.
That does not imply clipping is irrelevant to the optimizer. Post-freeze replay
of a seed-43 fit found a line-search evaluation with **8,409 negative rows**,
minimum log prediction **-4.718**; the joint fit found **5,243**, minimum **-4.188**.
The unclipped and clipped objectives differ at those trial points. They can
therefore take different finite-budget paths despite ending with positive
predictions. These diagnostic replays preserved the selected iteration counts;
they were not used to alter candidates or promotion.

The whole recorded study takes **116.15s**, peak RSS **1200.9 MiB**,
including data checks, shared warm starts, all refinements, repeated inference,
saved artifacts and paired comparisons. Imports, the later gradient-trace/control
analysis and regression tests are outside that timer. No GPU is used.

The CI-matched CPU benchmark regression passes **200 tests in 219.47s**.
All **26 measured CPU modules cover 3580/3580 statements (100%)**, including
**233/233** in the new objective module, with no excluded lines. This is statement
coverage of those modules, not branch coverage, GPU coverage or whole-repository
coverage. Tests include finite-difference joint gradients, clipping after the
member mean, perfect predictions, zero-step fallback, serialization, immutable
warm controls, frozen-before-audit ordering, artifact/ID checks and runner replay.
The new module and tests pass Ruff formatting and critical-error checks. CI now
runs this test and coverage target alongside the existing CPU benchmark suite.

## Reproduce

Use the benchmark environment and artifacts described in
[the prior same-input fusion study](FUSION_STRATEGIES.md), then run from the
repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m benchmarks.mercari_objective \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --manifest benchmarks/mercari/fusion-candidates.json \
  --anchor benchmarks/runs/mercari/fusion-strategies-001/verified/ensemble.npz \
  --output-dir benchmarks/runs/mercari/final-objective-reproduction
```

The output directory must be new. No model/loss tuning flags are required: all
three objectives, seeds and bounded budgets are predefined. Manifest hashes and
ordered IDs must match. `plan.json` records data/source hashes and split IDs;
`meta-models.pkl` and `meta-check.npz` make the check scores independently
inspectable; `models.pkl` and all ID-aligned prediction archives preserve the
full refits. `frozen.json` precedes audit labels; `report.json` adds scores,
seed mean/std/worst, loss-placement ablations, conditional paired intervals and
arithmetic-mean inference timings with raw samples/maxima/deviations.
Prediction artifacts store fitted preprocessing and parameters, not training
feature matrices or labels. Use the wrapper's final objective/history for
refinement diagnostics; member Adam histories describe their shared warm phase.
Large checkpoints, data and credentials remain ignored by Git.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest \
  tests/test_mercari_objective.py --cov=benchmarks.mercari_objective \
  --cov-report=term-missing --cov-fail-under=100
```

## Interpretation limits and references

RMSLE and squared RMSLE have identical unregularized minimizers, including the
clipping operation. Different finite-budget trajectories can still arise from
gradient scaling, line searches and nonconvex networks. Improvement after
refinement must be compared with the same-optimizer log-MSE control, not solely
with the earlier Adam result. Loss placement and network depth are distinct:
joint optimization changes which output is trained, not the frozen upstream
text experts.

Selection and audit reuse, prior candidate choices and repeated inspection limit
generalization claims. Paired row-bootstrap intervals condition on fitted models;
they omit training/selection uncertainty and do not resample duplicate-listing
groups. Three individually refined seeds are three single-model repetitions;
the joint ensemble has only one fixed three-member initialization. No local
score implies a new leaderboard rank.

- [Official Mercari evaluation](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/overview/evaluation).
- Byrd, Lu, Nocedal & Zhu (1995), *A Limited Memory Algorithm for Bound Constrained
  Optimization*, [doi:10.1137/0916069](https://doi.org/10.1137/0916069).
- Wolpert (1992), *Stacked Generalization*,
  [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1).

Analytic joint backpropagation, the bounded protocol and controls here are our
adaptation, not a claimed reproduction of a published benchmark.
