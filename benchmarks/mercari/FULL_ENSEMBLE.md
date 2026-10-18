# Deploying the frozen full Mercari ensemble

[中文](FULL_ENSEMBLE.zh-CN.md) · [Text/GPU experiments](FRONTIER.md) · [Beginner guide](BEGINNER_GUIDE.md)

The submission target is the **original full blend with local confirmation
RMSLE 0.3809068532**. The smaller candidate at 0.3819341875 is inferior and is
not promoted. This document describes its complete single-Kernel refit and
submission checks. See the submission evidence for actual Kaggle scores;
local confirmation scores are not leaderboard scores.

## How many models are combined?

The frozen outer matrix has **26 columns**. Nine have weights above 1e-15;
the remaining numerical weights are preserved exactly too.

| Outer prediction branch | Weight |
| --- | ---: |
| LightGBM calibrated pipeline | 31.723558% |
| XGBoost calibrated pipeline | 24.906067% |
| LibTree depth-5 calibrated pipeline | 18.262506% |
| LibTree depth-3 calibrated pipeline | 17.407428% |
| Earlier complete baseline | 4.027710% |
| Supervised MiniLM | 1.898970% |
| GRU | 1.439676% |
| Stemmed sparse MLP, MSE | 0.196278% |
| Stemmed sparse MLP, Huber | 0.137807% |

These are **prediction branches, not nine independent fitted models**. Each
calibrated pipeline shares the same first-level expert predictions and features.
Its prediction is the frozen inner blend plus its selected residual correction.

The complete dependency graph contains 32 fitted price predictors: seven in
the earlier baseline (Ridge, field-aware FM, four neural experts, one residual
tree model), 19 new experts (four LSA heads, three frozen-vector heads, two
sequence networks, nine sparse networks and one supervised transformer), and
six new calibration models (four boosted-tree models and two positive Ridge
models). Train-only text dictionaries, SVDs and categorical maps are additional
feature transformers. A tiny outer weight does not remove upstream dependence.

## Which result is the submission based on?

| Study system | Selection RMSLE | Reused confirmation RMSLE | Decision |
| --- | ---: | ---: | --- |
| Earlier frozen baseline | 0.395662 | 0.394153 | Comparison |
| Original new base blend | 0.383569 | 0.382011 | Comparison |
| **Original full blend** | **0.382498** | **0.380907** | **Submission target** |
| Smaller five-group base blend | 0.384351 | 0.382867 | Cost/accuracy comparison only |
| Smaller blend with recalibrated trees | 0.383435 | 0.381934 | Inferior; not submitted |

The original full blend improves local confirmation RMSLE by **3.36%**.
This confirmation window was previously inspected; it is a reused audit,
not a fresh blind test. The [original experiment](frontier-study.json) contains
its conditional bootstrap and family ablations. Selection and confirmation
results must not be compared directly with the historical private leaderboard.

[full-ensemble-config.json](full-ensemble-config.json) preserves the original
outer weights, inner weights, all 212 calibration feature positions and the
best saved epoch for each neural expert. A regression test compares the config
with the original experiment record. On 128 selection rows, the new stack
reconstruction reproduces the saved original full-blend prediction with
**maximum absolute log-price difference 0**. That check reuses saved expert
predictions while freshly computing baseline/LSA/attribute features; it does
not claim replay of every original neural weight file.

## Final refit and streamed inference

The final refit uses every official training label with disjoint roles:
**1,395,326 base rows** and **87,209 calibration rows**, grouped with seed 44.
Base experts, vocabularies, SVD and category maps fit base rows only. Residual
models fit calibration rows. Neural epochs, tree rounds, correction strengths
and both blend layers stay frozen; final calibration labels do not select them.

The earlier baseline's corrected prediction is used only as its outer branch.
The new calibration matrix uses its **uncorrected** expert blend and metadata,
so its fitted old correction cannot leak calibration targets into its new
calibration features. Feature order is asserted against the frozen 212-column
layout. All stage predictions carry exact ordered IDs.

Eight separate worker processes train/predict the baseline, LSA, sequences,
separate sparse view, joint sparse view, stemmed view, frozen vectors and
supervised transformer. Process exit releases TensorFlow/Torch GPU allocations
between views. Inference streams at most 25,000 test rows at a time, sharing a
sparse feature matrix between heads. Memory-mapped prediction/feature files
avoid constructing the complete 3.46-million-row dense calibration matrix.
Workers verify finite outputs, stable schemas and sample-ID order; final output
must contain exactly **3,460,725** predictions.

The public pretrained initialization is pinned to
`srg9000/all-minilm-l6-v2/transformers/default/1`. Its config and tokenizer files
match those used in the study, and the downloaded weight file matches the
mirror snapshot's Git LFS SHA-256:
`53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.
The Kernel verifies that weight hash before training. MiniLM is Apache-2.0
licensed; see the [model](https://www.kaggle.com/models/srg9000/all-minilm-l6-v2)
and [upstream model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
The offline source bundle includes unmodified Snowball 3.0.1 code with its BSD
license. No fitted Mercari model, competition records or credentials are
embedded or uploaded as external model inputs.

Mercari is a closed, Kernels-only competition. Every price model is trained
and its submission generated inside one Kernel. This is modern GPU practice,
not compliance with the original 2018 CPU/runtime restrictions or a new contest
rank. Hardware/library changes and the larger final base split can change the
result; the final private score must be observed.

## Reproduction

From the repository root, with the official Kaggle account already configured:

```bash
python -m venv .venv
.venv/bin/pip install '.[mercari,benchmark]' 'kaggle==2.2.4' pytest pytest-cov

# Build a private source-only GPU smoke Kernel, then the actual full refit.
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from benchmarks.mercari_full_kernel import build
config = json.loads(Path('benchmarks/mercari/full-ensemble-config.json').read_text())
api = KaggleApi(); api.authenticate()
owner = 'YOUR_KAGGLE_USERNAME'
folder = Path('benchmarks/runs/mercari/full-reproduction-smoke')
metadata = build(owner, folder, config, smoke=True)
print(api.kernels_push(str(folder), timeout='900', acc='nvidiaTeslaT4'))
PY
```

The smoke Kernel uses 100 training rows and seven synthetic test listings with
reduced capacities. It exports `SMOKE_ONLY_DO_NOT_SUBMIT.csv`. It checks the
complete dependency and file pipeline, not accuracy. A real Kaggle T4 run
completed all stages in **170.41 seconds**. A separate local CPU run with the
public pretrained model and isolated TensorFlow/Torch interpreters also passed.
GPU runtime/availability depends on account quota; the 18,000-second final limit
is a timeout ceiling, not a measured duration.

After smoke succeeds, build with `smoke=False` into a new directory and call
`api.kernels_push(..., timeout='18000', acc='nvidiaTeslaT4')`. All fitted models
are rebuilt in the Kernel. The public model is attached through `model_sources`;
Internet access stays disabled. Runtime scratch data stays under `/tmp`; only
source, the final report and submission are exported as Kernel outputs.

For a local final refit, download the attached public model version and use:

```bash
python -m benchmarks.mercari_final \
  --data-dir benchmarks/runs/mercari/data \
  --output-dir benchmarks/runs/mercari/full-final-local \
  --config benchmarks/mercari/full-ensemble-config.json \
  --pretrained PATH_TO_ALL_MINILM_L6_V2 \
  --chunk-size 25000 --threads 4
```

`--torch-python` can point to an isolated Torch environment; it requires the
`mercari-transformer` and `mercari-vectors` dependencies. CPU-only execution is
functional but potentially expensive. The model selection/audit protocol is
in [FRONTIER.md](FRONTIER.md); do not retune weights from Kaggle feedback.

Local release validation passes **131 distinct tests and 2,258/2,258 statements across 18 Mercari modules**. This is statement coverage, not branch or whole-repository coverage. [Verification JSON](full-ensemble-verification.json) and [real Kaggle smoke report](full-ensemble-smoke.json) preserve the evidence.

## Submission gates

Before submitting, require completed execution, the exact pulled Kernel source
hash and version, a single non-smoke `FULL_KAGGLE_REPORT`, its archive hash,
public pretrained weight hash, complete stage list, original training hash,
full split counts, unchanged config/outer weights, valid submission hash, and
3,460,725 rows. `benchmarks.mercari_ensemble.validate_hosted` implements the
report checks. The new local/CI tests must pass too.

Then use the exact completed version:

```python
api.competition_submit_code(
    'submission.csv', 'LibTree frozen full ensemble',
    competition='mercari-price-suggestion-challenge',
    kernel='YOUR_KAGGLE_USERNAME/libtree-mercari-full-ensemble', kernel_version=1,
)
```

A submission call is not proof of a score: wait for `COMPLETE` and record both
public and private results. Check for an existing receipt before retrying an
ambiguous submission request. Full source/report hashes and the receipt belong
in the experiment journal; downloaded datasets, models and raw predictions stay
under ignored `benchmarks/runs/`.
