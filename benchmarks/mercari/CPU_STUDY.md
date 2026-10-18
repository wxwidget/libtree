# Mercari on a CPU budget: features, complementary objectives and direct RMSLE

[中文](CPU_STUDY.zh-CN.md) · [Beginner guide](BEGINNER_GUIDE.md) · [Earlier scored CPU entry](TECHNICAL_REPORT.md)

The current direction uses **CPUs only**, with an explicit wall-time cap. The
previous queued GPU refit was removed after source backup; its watcher and
automatic submission are disabled. The historical 0.380907 local GPU-assisted
ensemble remains preserved, but is not a CPU result or a new Kaggle score.
Actual best scored practice entry remains **0.39199 private / 0.39110 public**.
No first-place claim follows from adopting a winning architecture.

## Measured screening outcome — 18 October 2026

[Complete evidence JSON](cpu-budget-study.json) records every expert, weight,
learning curve, ablation, source/input hash, raw timing and CI link. Four CPU
threads completed the bounded run in **806.00 seconds** (801.56 inside the study),
under the 900-second cap. Maximum individual child RSS was **4,843.22 MiB**;
this is not summed concurrent process memory. Cached-reference costs are excluded.

| System | Selection RMSLE | Reused confirmation RMSLE |
| --- | ---: | ---: |
| Existing full-data CPU reference | 0.395662 | 0.394153 |
| New 200k-row base experts only | 0.439363 | 0.437725 |
| Reference + new experts/calibrators, bounded intercept | **0.395560** | **0.394065** |
| Same pool, no intercept | 0.395562 | 0.394065 |

The absolute confirmation gain is **0.0000881 (0.022%)**. A conditional paired
300-replicate row bootstrap gives delta interval [−0.0001383, −0.0000299]. This
small conditional gain does not remove reused-window bias or show training-seed
stability. The intercept itself does not help confirmation: 0.394065330 with it
versus 0.394065154 without. No new Kaggle submission or first place is claimed.
The historical GPU-assisted **0.380907 remains better** and is preserved.

Three branches have substantive outer weights: cached reference **96.03038%**,
stemmed Huber **3.48805%**, stemmed bins **0.48157%**. The intercept is −0.00104437;
other numerical weights are retained in JSON. The new residual tree branches
have effectively zero outer weight. This screening does not justify training
and serving every proposed candidate. A complete fresh refit remains unmeasured.

Matched-size feature controls improve alpha-10 Ridge selection RMSLE from
0.471921 plain → 0.465879 exact → 0.465277 stemmed. The strongest new standalone
MLP is stemmed Huber, 0.453662 selection / 0.452309 confirmation. Six MLPs,
different seeds/views/losses and five trees do not substitute for sufficient
training data. Base-only ablations below exclude the cached reference and all
correctors, and fit their blend weights on selection only:

| New base-family variant | Selection RMSLE | Confirmation RMSLE |
| --- | ---: | ---: |
| All 12 | 0.439363 | 0.437725 |
| Without MSE MLPs | 0.439682 | 0.438089 |
| Without Huber MLPs | 0.442955 | 0.441112 |
| Without bin MLPs | 0.441600 | 0.440128 |
| Only plain Ridge controls | 0.471921 | 0.469665 |

At full correction strength, new calibration pipelines give confirmation
RMSLE 0.438010 (LibTree depth 3), **0.436845** (LibTree depth 5), 0.436928
(LightGBM L2), 0.437145 (LightGBM Huber) and 0.437215 (XGBoost depth 4).
These correct weaker 200k-row experts; they do not beat the existing full-data
CPU reference. Small point differences are not evidence of general superiority.

**Optimizer performance:** on 74,126 selection rows and 13 cached base experts,
three alternating-order repetitions after one warmup per method give arithmetic
means **0.146412 s numerical / 0.068992 s analytic**, a **2.12× ratio**. Raw
samples, min/max/std are in JSON. No-intercept selection RMSLE differs by only
1.3e-13, with maximum log-prediction difference 2.64e-8. This comparison covers
the complete weight optimizers, including their derivative/constraint handling,
not only a derivative kernel. It saves about 0.077 seconds, **not half of the
806-second training run**. It does not show inference acceleration.

The price study launched with the baseline-vertex optimizer before the later
clipped-anchor initialization regression fix. The saved launch-source hashes
and subsequent analytic replay are recorded separately; fresh retraining is
not claimed bitwise identical. The implementation commit's [four CI jobs][ci]
passed, including 17 CPU tests and the 100% statement gate for both new modules.

The next worthwhile experiment is to spend an explicitly larger **CPU** budget
on the useful stemmed Huber/bin configurations and compare full-data learning
curves, rather than adding more weak parameter variants. That run and a
one-model-per-core scheduler were **not executed** in this budget.

## What the champions actually reported

Jankiewicz and Lopuhin's [first-place solution][winner] reports a 12-model
ensemble of different text views, Huber/regression-via-classification and binary
features, combined using Lasso with an intercept. Their validation RMSLE
**0.3733** belongs to their data/split and is not our measured score.
Their [presentation][modeling] emphasizes sparse MLPs, sparse updates, growing
batch sizes and one model per CPU core. It also lists grid search, CNN/RNN,
factorization machines and fitting residuals among their unsuccessful attempts.
Those observations motivate candidates, not universal rules. Our older CPU
system did benefit from factorization and residual trees on its own split.

The implementation here is independent. It does not copy their trained models,
download embeddings or allocate GPUs. It presently executes experts sequentially
with bounded thread counts; it does **not** claim to implement their one-model-
per-core scheduler or demonstrate its scaling benefit.

## Features and experiment design

Three base-trained views share a fixed grouped split:

| View | Representation |
| --- | --- |
| Plain control | Existing normalized title/description TF-IDF, character grams, categorical interactions |
| Exact engineered | Cross-field markers, brand × title token and leaf category × title token, exact raw title characters, model/quantity/condition signals |
| Stemmed engineered | Stemmed words with the same explicit crosses, signals and separate raw character channel |

Examples of signals are `model_iphone6`, `unit_64gb`, `flag_accessory`,
`flag_box_only`, `flag_damaged` and `flag_bundle`. These are approximate text
evidence, not verified product attributes. Unit spacing is normalized; distinct
capacities remain distinct. Crosses use at most the first 24 title tokens per
listing; signals are bounded. Dictionaries, document frequencies and known-brand
recovery fit base rows only. Predictions cannot change when unrelated query
rows, prices or IDs are changed. Stemming follows [Snowball][snowball].

The predeclared pool has two Ridge alphas (3 and 10) per view and, for each
engineered view, three width-128 sparse MLPs: TF-IDF/MSE, binary/Huber and
binary/64-bin classification. Different seeds accompany these different losses;
this is not an isolated loss-only ablation. Increasing batch sizes are
1024/2048/4096 over three epochs. Existing row-sparse Adam is reused. This gives
**12 new base experts** with shared feature extraction per view.

Five residual predictors compare LibTree depths 3/5, LightGBM L2/Huber and
XGBoost depth 4. Each supplies correction strengths .5 and 1 without retraining.
They share calibration inputs/targets, not identical capacity or implementations;
one fit per setting does not establish a speed ranking. An optional earlier
CPU baseline makes **23 outer prediction columns**, not 23 independent fits.
New correctors use training-only target summaries, product attributes and the
12 new expert predictions. The cached baseline is excluded from their targets
and input features because its old corrector already fitted calibration labels.

The original grouped seed-42 windows are retained: base 1,186,026, calibration
74,129, selection 74,126 and confirmation 148,254. The budget run samples only
the base window; all evaluation rows remain unchanged. Plain/engineered models
therefore have matched training rows. A cached baseline trained on the full
base window is an incremental-system reference, not a matched-size standalone
learner. Its earlier fit and inference costs are excluded from this run and
must be added when budgeting a complete refit. No cached GPU expert is used.

## Directly optimize the actual metric

For nonnegative observed price `y`, predict log price `s = P @ w + b`, then:

```text
z = log1p(y)
predicted_price = expm1(max(s, 0))
RMSLE² = mean((max(s, 0) - z)²)
objective = RMSLE² + λ (||w - anchor||² + b²)
w >= 0, sum(w) = 1, -0.25 <= b <= 0.25, λ = 1e-4
```

The square root is monotone, so minimizing the unregularized square has the
same optimum as RMSLE. There is no reason to invent a raw-dollar loss and call
it a better RMSLE objective. Ridge and MSE experts already target log-price
squared error; Huber and bin classification are complementary **surrogates**.
Their inclusion must improve evaluation, not merely their training objective.

With `e = max(Pw+b,0)-z` and `a = 1[Pw+b>0]`, the analytic data gradients are
`2 P.T @ (e*a) / n` and `2 mean(e*a)`. The regularizer adds its usual linear
gradients. At zero the chosen subgradient is zero. Starting from the best
declared single-expert or uniform prediction avoids a completely clipped
anchor with zero gradient; this is not a general global-optimality guarantee
for the clipped objective. Positive predictions give the usual convex quadratic.
Tests compare the value with independently computed official RMSLE and the
gradient with numerical differences, including negative/zero predictions.

This is **not the first use of the correct metric in the project**: the existing
`blend_weights` already optimized clipped log MSE using numerical differences.
The new component adds analytic derivatives, a bounded intercept and the
clipped-start boundary guard. The no-intercept ablation separates any measured
score gain from simply changing the optimizer implementation.

All weights, intercepts and family ablations are frozen to disk **before**
confirmation labels are loaded. Selection is reused for models/epochs/weights;
confirmation was inspected in earlier work and is a reused audit, not a fresh
blind test. Base-family loss/view ablations exclude all residual correctors
and the cached baseline to avoid retaining hidden upstream dependencies.
They are group ablations, not isolated causal effects. No leaderboard labels
or feedback fit these models. See stacking principles in [Wolpert][stacking].

## Reproduce within a hard budget

```bash
python -m venv .venv
.venv/bin/pip install '.[mercari,benchmark]' pytest pytest-cov

# Prepare the original grouped windows if they do not already exist.
# --baseline must be your trusted, previously fitted CPU reference artifact.
.venv/bin/python -m benchmarks.mercari_frontier prepare \
  --data-dir benchmarks/cache/mercari \
  --baseline benchmarks/runs/mercari/optimized-audit-001/model.pkl \
  --output-dir benchmarks/runs/mercari/cpu-data

# A new, empty output directory is required. This is screening, not final refit.
.venv/bin/python -m benchmarks.mercari_cpu_study \
  --data-dir benchmarks/runs/mercari/cpu-data \
  --output-dir benchmarks/runs/mercari/cpu-budget-study \
  --rows 200000 --threads 4 --max-seconds 900 \
  --max-features 50000 --char-features 30000 --cross-features 30000 \
  --width 128 --epochs 3 --tree-rounds 200
```

The existing reference can be reproduced using [the earlier commands][earlier].
Preparation and prior reference training are outside the study's time cap.
Alternatively use the same grouped feature files with `--no-reference`; this
does not read `baseline.npz`. Prepared confirmation features must exclude
`price`; ordered IDs and listing groups may not overlap across windows.

`--rows 0` uses all base rows and requires a larger independently declared
budget. A timed-out worker is terminated, exits 124 and writes
`budget-exhausted.json`; partial artifacts are not a successful model or score.
The runner forces CPU visibility and BLAS/OpenMP/TensorFlow thread limits before
starting its child. It never uploads or submits. Complete price refitting and
Kernels-only CPU execution would need their own measured release step.

Successful outputs include `frozen.json`, `report.json`, ordered predictions,
feature transformers and fitted models. Use only trusted pickle files. Data,
model and raw-prediction artifacts stay ignored; commit small reports only.
Report raw single-run phase times and arithmetic means where repetitions
exist; successive training epochs are not independent timing repetitions.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m pytest tests/test_mercari_cpu.py tests/test_mercari_cpu_study.py \
  --cov=benchmarks.mercari_cpu --cov=benchmarks.mercari_cpu_study \
  --cov-report=term-missing --cov-fail-under=100
```

Tests exercise real Ridge/MLP/native/competitor training, serialization, the
metric derivative, train-only features, leakage guards, audit-order mismatch,
CLI CPU limits and hard-timeout behavior. Coverage refers to these modules'
statements, not branches, GPU paths or the whole repository.

To repeat the optimizer comparison after this study (no additional price fits):

```python
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from benchmarks.mercari_cpu import RmsleBlend
from benchmarks.mercari_semantics import blend_weights
data = Path('benchmarks/runs/mercari/cpu-data')
run = Path('benchmarks/runs/mercari/cpu-budget-study')
evaluation = np.load(data / 'evaluation.npz')
paths = [data / 'baseline.npz', *sorted(p for p in run.glob('*.npz') if p.stem != 'ensemble')]
columns = []
for path in paths:
    archive = np.load(path)
    np.testing.assert_array_equal(archive['selection_ids'], evaluation['selection_ids'])
    columns.append(archive['selection'])
p, y = np.column_stack(columns), evaluation['selection']
samples = {'numeric': [], 'analytic': []}
with threadpool_limits(limits=4):
    for repeat in range(4):
        order = ['numeric', 'analytic'] if repeat % 2 == 0 else ['analytic', 'numeric']
        for method in order:
            start = perf_counter()
            weights = (blend_weights(p, y) if method == 'numeric' else
                       RmsleBlend(bias_bound=0.).fit(p, y).weights_)
            seconds = perf_counter() - start
            if repeat:  # first pair is warmup
                samples[method].append(seconds)
for method, values in samples.items():
    print(method, values, 'mean', np.mean(values), 'min', min(values),
          'max', max(values), 'std', np.std(values))
```

[winner]: https://github.com/pjankiewicz/mercari-solution
[modeling]: https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/5_modeling.tex
[snowball]: https://snowballstem.org/algorithms/english/stemmer.html
[stacking]: https://doi.org/10.1016/S0893-6080(05)80023-1
[earlier]: OPTIMIZATION.md
[ci]: https://github.com/wxwidget/libtree/actions/runs/39137644958
