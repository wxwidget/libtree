# Mercari: complementary text representations and real GPU experiments

[中文](FRONTIER.zh-CN.md) · [Earlier scored system](TECHNICAL_REPORT.md) · [Beginner guide](BEGINNER_GUIDE.md)

This is the **modern-resource extension** of the Mercari study. It evaluates
LSA, pretrained sentence vectors, sequence networks, larger sparse networks,
and tree/regression ensembles. It does not replace the earlier frozen scored
entry. The existing best practice submission is **0.39199 private / 0.39110
public RMSLE**. Local validation is not a new Kaggle score or contest rank.


## Measured outcome — 18 October 2026

The frozen full candidate reduces **local confirmation RMSLE from 0.394153 to
0.380907 (3.36%)**. It has not been submitted to Kaggle. It does not establish
first place: the historical private target remains 0.37758, on different data.
A complete new full-data refit and serving/submission pipeline are still needed
before a new leaderboard comparison can be made.

| System | Selection RMSLE | Reused confirmation RMSLE |
| --- | ---: | ---: |
| Earlier frozen baseline | 0.395662 | 0.394153 |
| New base-expert blend | 0.383569 | 0.382011 |
| New full blend, including residual trees | **0.382498** | **0.380907** |

The [pre-audit protocol](frontier-pre-audit-protocol.json) selects the full
candidate by selection RMSLE **before** this round's confirmation evaluation.
The absolute confirmation difference is −0.0132466, with a paired 300-replicate
row-bootstrap 95% interval [−0.0139295, −0.0125207]. This interval is conditional
on these fitted models and the reused window, not evidence about training-seed
variability or a fresh unseen test set.

The [machine-readable experiment report](frontier-study.json) includes every
model configuration result, learning curves, worker exit statuses, prediction
hashes, frozen weights, error correlations and ablations. The
[verification report](frontier-verification.json) records tests, real GPU
execution, replay differences and cleanup.

### Single experts: larger is not automatically better

All rows below use the same selection window; each expert trains on the
1,186,026 base rows. These are best saved checkpoints, not ensemble scores.

| Expert | Device | Selection RMSLE |
| --- | --- | ---: |
| LSA + Ridge, alpha 10 | CPU | 0.653440 |
| LSA + LightGBM, 63 leaves | CPU | 0.486604 |
| Frozen MiniLM + Ridge, alpha 10 | T4 encoding / CPU head | 0.624249 |
| Frozen MiniLM + LightGBM, 63 leaves | T4 encoding / CPU head | 0.469991 |
| CNN, width 128, MSE | T4 | 0.427283 |
| GRU, width 96, Huber | T4 | 0.424239 |
| Separate sparse MLP, width 256, binary/Huber | A100 | 0.413103 |
| Joint sparse MLP, width 512, binary/Huber | A100 | 0.409622 |
| Stemmed sparse MLP, width 512, binary/Huber | A100 | **0.407484** |
| Supervised MiniLM, 4 epochs, Huber | A100 | 0.416930 |

The supervised MiniLM learning curve is 0.439640 → 0.423525 → 0.419540 →
0.416930. Although weaker alone than the old multi-model baseline, it receives
28.06% weight in the new base blend. Its errors complement lexical models.
Frozen sentence vectors and LSA-only predictors receive effectively zero
base-blend weight. Semantic similarity alone does not preserve price-sensitive
brands, numbers, product identity and accessory distinctions well enough.

### Base-family ablation

These variants were specified before the final audit. Remove the named **new
expert outputs**, refit only selection blend weights, retain all other fitted
experts and the old baseline, and exclude new residual calibrators throughout.
Values near 1e-10 are numerical optimizer differences.

| Base blend variant | Selection RMSLE | Confirmation RMSLE | Change from base confirmation |
| --- | ---: | ---: | ---: |
| All new base experts | 0.383569 | 0.382011 | — |
| Without supervised MiniLM | 0.388317 | 0.386233 | +0.004222 |
| Without all new sparse MLPs | 0.388572 | 0.387386 | +0.005374 |
| Without stemmed sparse MLPs | 0.384731 | 0.383319 | +0.001308 |
| Without joint sparse MLPs | 0.383897 | 0.382403 | +0.000392 |
| Without separate sparse MLPs | 0.383658 | 0.382075 | +0.000063 |
| Without CNN/GRU | 0.383755 | 0.382246 | +0.000235 |
| Without frozen-vector predictors | 0.383569 | 0.382011 | approximately 0 |
| Without LSA-only predictors | 0.383569 | 0.382011 | approximately 0 |

The last row does not remove LSA metadata inside frozen-vector tree models.
Family deltas overlap and must not be added together. Full-blend
`leave_one_expert_out` results remove **outer columns only**: already-fitted
calibrators still depend on upstream expert predictions and LSA features.
A zero outer weight therefore does not establish that its upstream inference
cost can be removed. No pruned serving-latency improvement is claimed.

### Tree and regression comparison

Correctors use 74,129 calibration rows, 212 common features, 400 boosting rounds
and the same residual target. Correction strength is selected separately.

| Calibration model | Selection RMSLE | Confirmation RMSLE | Observed stage seconds |
| --- | ---: | ---: | ---: |
| LibTree depth 3 | 0.382686 | 0.381065 | 7.43 |
| LibTree depth 5 | 0.382712 | 0.381064 | 12.53 |
| LightGBM, 15 leaves | 0.382704 | 0.381051 | 15.52 |
| XGBoost, depth 4 | 0.382725 | 0.381162 | 13.50 |
| Positive Ridge, alpha 10 | 0.383773 | 0.382065 | 0.15 |
| Positive Ridge, alpha 100 | 0.383767 | 0.382059 | 0.19 |

LibTree has the lowest selection point estimate; LightGBM has the lowest
confirmation point estimate. Differences are small and do not establish
algorithmic superiority. CPU stage timers include fit plus prediction; they
ran alongside local tests, and model capacities differ. They are observations,
not a controlled speed ranking. The full blend combines native trees,
LightGBM and XGBoost; its gain is not attributable to LibTree alone.

For an inspectable artifact, see the actual native residual model's
[first tree as text](frontier-tree-example.txt) or
[JSON](frontier-tree-example.json). Leaf contributions are log-price residuals
scaled by learning rate. This is one of 400 correction trees, not an explanation
of the entire neural ensemble or a causal product-price explanation.

### Runtime and unsuccessful attempts

| Recorded stage | Hardware | Elapsed seconds, one execution |
| --- | --- | ---: |
| Frozen MiniLM encoding only | T4, float32 | 1,094.69 |
| Frozen vectors + all heads, worker total | T4 + CPU | 2,225.65 |
| Separate sparse profile, all three experts | A100 80 GB | 380.08 |
| Joint sparse profile, all three experts | A100 80 GB | 536.74 |
| Stemmed sparse profile, all three experts | A100 80 GB | 522.77 |
| Supervised MiniLM, four epochs, worker total | A100 80 GB | 1,157.47 |

Worker totals include feature preparation, training, prediction and writing,
not VM allocation/upload/download. The supervised run records 384.99 seconds
preparation and per-epoch times [187.97, 186.69, 186.79, 186.67]. Epochs are
successive optimization steps, not independent runtime repetitions. This study
has one fit per candidate configuration; it reports raw elapsed observations,
not a median or a statistically supported mean speedup. Complete-ensemble
serving latency and production refit cost have not yet been measured.

The initial T4 job successfully saved CNN/GRU but then failed sparse auto-XLA.
It remains marked failed; subsequent A100 sparse jobs succeed after the fix.
A concurrent second A100 allocation was refused, so jobs ran serially. The
account balance changed from 199.27 to 190.81 compute units, an observed account
change rather than controlled per-model billing. All owned sessions are stopped;
final API usage is zero.

Two integration bugs escaped the first line-coverage gate: bundling an absent
`LICENSE` file and pickling `__main__.ListingInputs` when launched with `-m`.
The bundle now includes available `LICENSE*` files; the entrypoint imports the
canonical module. Tests now exercise the actual repository bundle and a real
module entrypoint. A real portable-launcher T4/float16 smoke run then passed
execution, result download and cleanup on synthetic data; it is not accuracy
evidence. Original transformer metadata is preserved, with repaired canonical
copies for the earlier trusted local artifacts; saved weights/predictions did
not change. TensorFlow CPU/T4 replay differs by at most 4.77e-7 on 128 rows;
Torch float32 CPU versus A100 bfloat16 differs by up to 0.00538 log-price, so
cross-version/precision Torch replay is approximate.

Local verification passes **93 distinct tests, 1,835/1,835 statements across
14 Mercari modules**, plus a second compiled execution of four FM tests.
Coverage is statement coverage, not whole-repository or branch coverage.
GPU sparse weights were not all retained in this research cycle; reports and
predictions are retained. Use `--include-models` for model downloads in future
runs. Full-data refit, consistent serving export and a new Kaggle evaluation
remain necessary before promoting this research blend to a scored entry.

## Goal and comparison contract

The goal is to improve prediction quality toward the strongest historical
systems. The competition closed in 2018. Today's downloaded **public**
leaderboard CSV lists the leading score as **0.37665**. The final leaderboard
returns **0.37758**, corroborated by the champions' [published final leaderboard
screenshot][winner-leaderboard], including its change-from-public column.
The private-score reference target is **0.37758**; public/private scores remain
separately named. Our current scored private gap is 0.01441, before this round's
unsubmitted local experiments.

The modern experiment uses Colab T4/A100 and, for MiniLM, external pretrained
weights. It is not a reproduction under the historical CPU/time restrictions.
No private/test labels, scraped prices or leaderboard feedback fit the models.

The [champions' published solution][winner] motivates complementary text views,
Huber/classification objectives and blending. Its reported validation RMSLE
0.3733 is **their validation result**, not our result or a directly comparable
score. Their presentation also reports unsuccessful CNN/RNN experiments:
architecture complexity alone is not evidence of improvement.

## Validation before model selection

Use all 1,482,535 official training rows, including zero prices. Identical
available listing features stay in one group. The original seed-42 split is
retained for comparison:

| Window | Rows | Allowed role |
| --- | ---: | --- |
| Base | 1,186,026 | Fit vocabularies, SVD, metadata and expert parameters |
| Calibration | 74,129 | Fit residual trees on base-expert predictions |
| Selection | 74,126 | Select epochs, correction strengths and blend weights |
| Confirmation | 148,254 | Evaluate a frozen choice; no model fitting |

The old frozen model reproduces **0.3956615033 selection RMSLE**. Prepared
confirmation files sent to Colab contain features and IDs **without prices**.
Every prediction archive carries the ordered IDs; mismatches fail before
blending. Blend weights are written to `frozen.json` before confirmation labels
are read for scoring.

The confirmation window was inspected in the earlier study. It is an explicitly
reused audit window, not a new untouched test set. A paired row bootstrap
describes sampling variability conditional on fitted models; it does not remove
selection bias or measure variability over training seeds.

## What each representation contributes

| View | How it works | Expected complement and limitation |
| --- | --- | --- |
| Sparse lexical | Word/character TF-IDF or binary indicators | Preserves exact brands, capacities and model identifiers |
| Cross-field lexical | Add field markers and combine title, brand, description and category | Lets the same phrase participate in shared interactions; retains field identity |
| Stemmed lexical | Snowball stems, split letters/numbers and selected symbols; separate exact character view | Shares inflections such as shoe/shoes without relying entirely on semantic vectors |
| LSA | Train-only TF-IDF followed by randomized truncated SVD, separately per text field | Compact co-occurrence vectors; rare price-sensitive tokens can disappear |
| MiniLM | Pinned `all-MiniLM-L6-v2`, normalized title/description vectors, length cap 128 | Semantic similarity; pretrained without task prices and can blur important numbers |
| Product attributes | Capacity, numbers, redactions, accessory/damage/bundle/new phrases | Inspectable approximate lexical evidence, not verified product facts |
| CNN / GRU | Supervised embeddings, pooling/sequence interactions and categorical embeddings | Task-specific representations; validation chooses the saved epoch |
| Sparse MLP | Wide first layer over sparse inputs, MSE/Huber/soft-bin objectives | Efficient nonlinear lexical interactions; sparse input does not guarantee fast optimization |
| Supervised MiniLM | Fine-tune the backbone with mean/CLS pooling and train-only categorical embeddings | Learns price-specific representations; assess its extra gain and cost |

`LatentText.topics()` lists the largest positive word loadings for inspecting
latent axes. For example, the observed first title axis includes “bundle” and
“reserved”; these are co-occurrence patterns, not causal price explanations.

Recurrent inputs use nonzero unknown-token IDs and strictly right-padded
sequences, including otherwise empty rows. This permits cuDNN's recurrent
path. Older bag/pooling encoders retain their existing unknown-token behavior.

Keras 3 / TensorFlow 2.21 exposed a real GPU failure: auto-XLA could not compile
the sparse gather/segment operations. Sparse training explicitly sets
`jit_compile=False`; TensorFlow still executes supported operations on the GPU.
The regression test forces an auto-JIT preference and verifies real training.

## Blending and residual trees

For predictions `P` and log prices `z`, select nonnegative sum-one weights on
the selection window:

```text
minimize mean((max(P @ w, 0) - z)^2) + 0.0001 * ||w - baseline_anchor||²
subject to w >= 0 and sum(w) = 1
```

A weak expert is allowed to receive zero weight. The penalty discourages
unnecessary departures from the baseline. Component error correlations and
base-family ablations test whether extra predictions add accuracy. Full outer-column
ablations retain upstream calibrator dependencies and do not measure serving savings.

The optional `calibrate` command uses **uncorrected** old expert predictions,
LSA/numeric features, lexical attributes and new predictions. Only calibration
labels fit the correctors. LibTree depths 3/5, LightGBM and XGBoost are compared
using the same features and residual target. Each correction strength is chosen
from 0/.25/.5/1 on selection rows. Arbitrary category IDs are excluded from this
residual matrix; original training-only target/frequency summaries remain.

## Reproduce the study

Run from the repository root. Raw data, models, transfer chunks, predictions and
logs remain under ignored `benchmarks/runs/`; never commit credentials or data.

```bash
python -m venv .venv
.venv/bin/pip install '.[mercari,benchmark]' pytest pytest-cov

# Fresh clones first refit the documented base/calibration system.
.venv/bin/python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/optimized-config.json --mode study \
  --output-dir benchmarks/runs/mercari/frontier-base

.venv/bin/python -m benchmarks.mercari_frontier prepare \
  --data-dir benchmarks/runs/mercari/data \
  --baseline benchmarks/runs/mercari/frontier-base/model.pkl \
  --output-dir benchmarks/runs/mercari/frontier-data

.venv/bin/python -m benchmarks.mercari_frontier trees \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --output-dir benchmarks/runs/mercari/frontier-trees \
  --threads 4 --components 64 --max-features 75000
```

Install/authenticate the **official** `google-colab-cli==0.7.4` in a separate
Python 3.13 environment. `colab auth` authenticates a VM for GCP; initial CLI
login is `colab --auth=oauth2 sessions`. The configured cloud machine already
has the writable-state launcher `/workspace/libtree-build/bin/colab`.

```bash
# Caller must already have Colab access/quota. Each invocation owns one VM.
.venv/bin/python -m benchmarks.mercari_colab \
  --cli /workspace/libtree-build/bin/colab \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --output-dir benchmarks/runs/mercari/frontier-t4 \
  --gpu T4 --high-mem --profile all

.venv/bin/python -m benchmarks.mercari_colab \
  --cli /workspace/libtree-build/bin/colab \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --output-dir benchmarks/runs/mercari/frontier-a100 \
  --gpu A100 --high-mem --profile sparse
```

On another machine, replace `--cli` with your official `colab` executable.
High-RAM/A100 availability depends on the account. Source and input SHA-256
hashes are checked. Uploads use 12-MiB chunks. Model runs have bounded timeouts;
execution, result download and cleanup have separate logs/status. The launcher
stops only its own uniquely named session in `finally`. If the local process is
force-killed, use the recorded session in `job.json` to stop it manually.

Default downloads include reports/predictions/logs; `--include-models` also
downloads potentially large `.keras`/`.pkl`/`.pt` artifacts. Original study artifacts
were retained separately while vector experiments continued. A failed model
stage must not be reported as successful because other predictions were saved.

Do not install the CPU-pinned `.[mercari]` extra on the GPU VM. The bootstrap
installs the native package without dependencies and uses the VM's GPU-capable
TensorFlow. MiniLM records its resolved immutable revision; pass `--revision`
to the `vectors` command for a pinned replay. Core GBDT/C5 remain CPU models.

Unzip the returned `results.zip` into new directories, then fit the calibration
trees and freeze/evaluate the blend:

```bash
.venv/bin/python -m benchmarks.mercari_frontier calibrate \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --baseline benchmarks/runs/mercari/frontier-base/model.pkl \
  --lsa-encoder benchmarks/runs/mercari/frontier-trees/encoder.pkl \
  --experts benchmarks/runs/mercari/frontier-trees PATH_TO_SEQUENCE_PREDICTIONS \
    PATH_TO_VECTORS PATH_TO_SEPARATE PATH_TO_JOINT PATH_TO_STEMMED PATH_TO_FINETUNED \
  --output-dir benchmarks/runs/mercari/frontier-calibration --threads 4

.venv/bin/python -m benchmarks.mercari_frontier ensemble \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --experts benchmarks/runs/mercari/frontier-trees PATH_TO_SEQUENCE_PREDICTIONS \
    PATH_TO_VECTORS PATH_TO_SEPARATE PATH_TO_JOINT PATH_TO_STEMMED PATH_TO_FINETUNED \
    benchmarks/runs/mercari/frontier-calibration \
  --output-dir benchmarks/runs/mercari/frontier-ensemble
```

Use new output directories for every invocation. The sequence-only and sparse
local/remote worker commands are also directly usable:

```bash
python -m benchmarks.mercari_frontier gpu --data-dir PREPARED --output-dir OUT \
  --neural-only sparse --sparse-view joint --sparse-width 512 --max-features 150000
```

`--allow-cpu` is for tiny functional checks and explicitly supplies no GPU
evidence. The Colab SDK's original 10-second WebSocket readiness wait failed
on a high-RAM cold start; the configured cloud launcher allows 120 seconds
without weakening TLS or authentication. The initial failed VM was released.

## Tests and interpretation

Tests cover train-only features, unknown/missing text, category handling,
pickle/Keras replay, dense/sparse arithmetic, forced auto-JIT, ordered prediction
IDs, confirmation-label isolation, freezing before audit, blend optimization,
real tiny tree/neural fits, and cleanup on allocation/upload/execution/download
failures. Core feature/model/runner modules have a 100% statement-coverage gate
in CI. Mocked lifecycle tests are not evidence of a real GPU run.

Runtime observations include data preparation and full stage logs. Different
architectures, hardware, library versions or sample sizes do not establish a
controlled GPU speedup. Never select favorable median repetitions: any repeated
timings use arithmetic means plus min/max/std and raw samples.

## Supervised and morphological extensions

For example, `iPhone6S 64GB case` becomes `iphone 6s 64gb case` in the original
word view and `iphon 6 s 64 gb case` in the stemmed view. An independent
character channel retains the model/capacity spelling. These are approximate
text features, not verified product identities. A bounded 200,000-word
rule-result cache contains no price or inference-population statistics.

The `all` Colab profile runs sequence experts and sentence vectors; sparse
experts run in the separate A100 profile, avoiding duplicate expert versions.
The `stemmed` and `transformer` profiles can be reproduced as follows:

```bash
.venv/bin/python -m benchmarks.mercari_colab \
  --cli /workspace/libtree-build/bin/colab \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --output-dir benchmarks/runs/mercari/frontier-stemmed \
  --gpu A100 --high-mem --profile stemmed

.venv/bin/python -m benchmarks.mercari_colab \
  --cli /workspace/libtree-build/bin/colab \
  --data-dir benchmarks/runs/mercari/frontier-data \
  --output-dir benchmarks/runs/mercari/frontier-transformer \
  --gpu A100 --high-mem --profile transformer --transformer-epochs 4 --include-models
```

`--include-models` includes Torch `.pt` checkpoints as well as Keras/pickle
models. Add the stemmed and the chosen transformer prediction directories to
both calibration and ensemble commands above. Select one transformer version;
duplicate expert names and label archives are explicitly rejected.

MiniLM's measured immutable revision is
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The direct workers support `--revision`
for a pinned replay. The frozen-vector run uses float32; optional
`--embedding-dtype float16` audits 32 rows against float32 before full encoding.
The optional precision is not the precision of the reported frozen-vector run.

The supervised worker uses maximum length 128, batch 512, AdamW backbone rate
3e-5 and head rate .001, halved each epoch. Mean/CLS pooling joins six
12-dimensional category embeddings and a width-256 head. Only base rows fit
category mappings. A100 training uses bfloat16 autocast, float32 losses and
norm-one gradient clipping. Tiny real BERT tests verify learning, backbone
updates, padding invariance and checkpoint replay without network access.
Install `.[mercari-transformer]` in a separate environment; for CPU-only tests,
use the official PyTorch CPU wheel index. The core LibTree import never requires
Torch. The study's CPU tests use Torch 2.9.0+cpu; the real Colab run records
Torch 2.11.0+cu130 and Transformers 4.57.1.

Configuration comparisons are not isolated loss ablations: seeds, binary vs
TF-IDF values and widths vary along with losses. Frozen MiniLM + LightGBM and
fine-tuned MiniLM also have different heads/metadata. Leave-one-expert-out
refits quantify outer-column removal while keeping the other experts fixed;
base-family ablations above more directly assess new expert outputs.

## References

- [Deerwester et al. (1990), LSA](https://doi.org/10.1002/asi.4630410605).
- [Halko et al. (2011), randomized matrix decompositions](https://doi.org/10.1137/090771806).
- [Reimers & Gurevych (2019), Sentence-BERT](https://aclanthology.org/D19-1410/).
- [Porter / Snowball stemming](https://snowballstem.org/algorithms/english/stemmer.html).
- [Wang et al. (2020), MiniLM](https://arxiv.org/abs/2002.10957).
- [Devlin et al. (2019), BERT fine-tuning](https://aclanthology.org/N19-1423/).
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [Kim (2014), CNN sentence models](https://aclanthology.org/D14-1181/).
- [Cho et al. (2014), GRU](https://aclanthology.org/D14-1179/).
- [Huber (1964), robust estimation](https://doi.org/10.1214/aoms/1177703732).
- [Wolpert (1992), stacked generalization](https://doi.org/10.1016/S0893-6080(05)80023-1).
- [Mercari first-place modeling presentation][winner-models] and [text views][winner-views].

[winner]: https://github.com/pjankiewicz/mercari-solution
[winner-models]: https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/5_modeling.tex
[winner-views]: https://github.com/pjankiewicz/mercari-solution/blob/master/mercari/datasets_tf.py

[winner-leaderboard]: https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/img/leaderboard.png
