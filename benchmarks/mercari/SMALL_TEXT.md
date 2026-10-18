# Budgeted small text models on Colab

[中文](SMALL_TEXT.zh-CN.md) · [Previous text study](FRONTIER.md) · [Beginner guide](BEGINNER_GUIDE.md)

The 18 October 2026 request re-enables Colab CPU/GPU experiments. The former
CPU-only restriction no longer applies to new studies. The deleted historical
Kaggle job and its automatic watcher remain disabled. This study uses one T4,
not an A100 or concurrent GPU instances. The historical local frozen ensemble
**0.380906853 RMSLE** remains the reference; the CPU-only **0.392845724** result
is a different, weaker cost/accuracy tradeoff. Neither is a new leaderboard score.

## Controlled candidates

All candidates fit the same 1,186,026 base listings. The fixed grouped windows
contain 74,129 calibration, 74,126 selection and 148,254 confirmation rows.
Confirmation features are transferred without prices; their labels stay local.
Epoch choice and ensemble weights use selection labels only. Confirmation was
already inspected in earlier studies and is a reused audit, not a blind test.

| Candidate | Representation | Loss | Epoch ceiling | Batch size |
| --- | --- | --- | ---: | ---: |
| Subword bag | 48-dimensional WordPiece embeddings + 32,768 hashed adjacent-pair buckets | MSE | 4 | 2,048 |
| Subword bag | Same architecture and seed | Huber, delta .5 in standardized log space | 4 | 2,048 |
| BERT-tiny | Pretrained 2-layer, 128-hidden-dimensional BERT; mean + CLS pooling | MSE | 3 | 512 |

Each candidate includes train-only category embeddings, condition, shipping,
lengths and lexical product attributes. Maximum token length is **96**, head
width **128**, CPU thread setting **4**, seed **42** and total model parameter
ceiling **8,000,000**, including metadata embeddings and the regression head.
The worker rejects a model exceeding that ceiling before bulk tokenization or
training. Parameters are exposed as CLI options for separate studies.

The bag independently adapts the average-embedding and hashed word-ngram ideas
of [Joulin et al.][fasttext]. It uses **WordPiece subwords**, not fastText's
character-ngram vocabulary. It is a PyTorch price regressor, not the official
fastText classifier or pretrained fastText vectors. Its tokenizer is pretrained;
its regression embeddings initialize randomly. BERT-tiny also starts from
pretrained backbone weights. Comparing them does not isolate pretraining alone.
Small pretrained BERT follows [Turc et al.][tinybert]; the exact model repository
and immutable revision are recorded in each report.

Models optimize `log1p(price)` standardized using the base mean and standard
deviation. MSE directly matches squared log error before nonnegative clipping.
Huber is a robust alternative, not the exact RMSLE objective. Predictions convert
back to log-price scale; metric evaluation clips log predictions below zero.
Bag embedding learning rate is .003, BERT backbone .00005, head .001, with AdamW,
weight decay .01, gradient norm cap 1 and learning-rate multiplier .7 each epoch.
The best selection checkpoint is restored before exporting all window outputs.

## Resource and artifact contract

The launcher bounds the **combined three-candidate subprocess training window
to 3,600 seconds**, including their preprocessing and prediction. Allocation,
upload, package installation, result packaging/download and cleanup are outside
that window. Exceeding a timeout marks failure rather than inventing a result.
Only the uniquely named session allocated by this job is stopped in `finally`.
The launcher never submits to Kaggle. GPU model parameter count is not whole
ensemble size or full serving memory; the original ensemble has shared upstream
models and calibration dependencies.

Colab CPUs handle decoding, metadata and tokenization; T4 executes regression
training and prediction with mixed precision. The initial measured run (`3828d76`) repeats per-candidate preparation,
included in its worker times. The current launcher reuses an input cache for
subsequent candidates on the same VM. Those times are single runs, not controlled
speed rankings, scaling claims or training variability estimates. The average
per-row squared log error defines RMSLE; no median aggregation is used.

With `--include-models`, result ZIPs retain checkpoints, fitted input mappings,
ordered prediction IDs, selection learning curves, parameter counts, timing,
versions, resolved pretrained revision and failure logs. Source/data transfer
hashes and own-session cleanup are recorded in `job.json` and the input manifest.
Raw listings, price-fitted artifacts and prediction arrays remain Git-ignored.
Only trusted locally generated pickle files may be loaded.

## Reproduce

Use the existing [grouped data preparation](FRONTIER.md) and an authenticated
[official Colab CLI](https://github.com/googlecolab/colab-cli). Preparation never
transfers `evaluation.npz`. Do not paste OAuth credentials into scripts or Git.
Install the project in the local launcher environment and supply the CLI path:

```bash
# Local CPU launcher/tests; keep GPU Torch preinstalled on the Colab VM.
python -m pip install 'torch==2.9.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[mercari-transformer]' pytest pytest-cov
python -m benchmarks.mercari_colab \
  --data-dir benchmarks/runs/mercari/frontier/data \
  --output-dir benchmarks/runs/mercari/smalltext-t4 \
  --cli colab --gpu T4 --profile smalltext --include-models
```

The CLI selects GPU PyTorch on the remote machine; the project's TensorFlow CPU
extra is not installed into Colab. Use `--gpu A100` only for a separately justified
experiment; this profile does not require it. Output folders must be new.

For an individual worker on a GPU machine:

```bash
python -m benchmarks.mercari_smalltext \
  --data-dir prepared-windows --output-dir new-study \
  --architecture bert --model-id prajjwal1/bert-tiny \
  --epochs 3 --length 96 --batch-size 512 --width 128 \
  --max-parameters 8000000 --loss mse
```

For CPU validation, add `--allow-cpu`; this explicitly permits CPU execution
when CUDA is absent. It does not hide a detected GPU. Tests use real tiny BERT
backbones initialized offline, actual gradient updates, parameter rejection,
padding invariance, checkpoint restoration, exact IDs and label isolation.

```bash
USE_TF=0 python -m pytest tests/test_mercari_smalltext.py \
  tests/test_mercari_text_cache.py tests/test_mercari_transformer.py \
  tests/test_mercari_final_torch.py tests/test_mercari_colab.py \
  --cov=benchmarks.mercari_smalltext --cov=benchmarks.mercari_text_cache \
  --cov=benchmarks.mercari_transformer --cov=benchmarks.mercari_final_torch \
  --cov=benchmarks.mercari_colab \
  --cov-report=term-missing --cov-fail-under=100
```

Local verification passes **26 tests / 473 statements / zero missed** across
five worker/lifecycle modules; 14 frontier tests cover a further 372 statements. CI includes the new worker and cache in its
existing offline transformer coverage gate. This is statement coverage, not
complete branch or GPU-instruction coverage.

## Reusing encoded inputs

The current small-text profile shares one cache across the three candidates;
individual workers opt in with `--cache-dir trusted-new-cache`. Cache identity
includes every grouped input file hash, immutable tokenizer revision, length,
encoding/window-reader source hashes, Python and library versions. All 16 arrays and the base-only
input mapping are content-hashed and checked before reuse. Mismatched, corrupt
or partial caches fail instead of silently mixing data. Exclusive directory
creation prevents concurrent writers overwriting a cache; choose a new directory
after an interrupted first write. No fitted price regressor is stored in this
cache. Load only caches you created: hashes detect stale/corrupt content, not
hostile pickle files.

Copy-on-write memory maps avoid bulk array copies while protecting stored
arrays. CPU tests verify exact cached/uncached training outputs for both model
architectures and that mutating a mapped array does not change the stored file.
The measured initial GPU profile predates this change; do not label its timings
as cached. For its original uncached launcher, check out commit `3828d76`.

[fasttext]: https://aclanthology.org/E17-2068/
[tinybert]: https://arxiv.org/abs/1908.08962

### Observed cache cost

[Full input-cache measurements](smalltext-input-cache-001.json): all 1,482,535
rows encoded and written on the local 4-CPU-quota Xeon took **261.34 seconds**;
five warm, hash-verified loads averaged **0.86668 seconds**. Every array matched
exactly. Cache size is **593,380,496 bytes**, maximum process RSS **4,121.59 MiB**.
TSV reading (7.15 s) and tokenizer creation are excluded from both cache timers.
The uncached path has one observation; targeted tests overlapped about 14 s of
its preparation. These observations establish useful reuse, not a controlled
full training speed ratio, cold-disk claim or GPU/CPU hardware comparison.
The key was subsequently tightened to include Python and the window reader.

### Precision replay: an observed failure and its cause

[Replay and matched-batch GPU evidence](smalltext-precision-001.json) retain the
failed initial tolerance check. With the same fitted bag checkpoint, CPU FP32
versus T4 FP16 differs by up to **0.0165677 log-price units**; **23/296,509**
outputs exceed the predeclared .01 absolute tolerance. Mean absolute differences
are approximately .00096 in each window. The tolerance has not been relaxed.

A new short kernel on the **same T4 VM** checked the four worst rows from each
window using their original 2,048-row batches. Encoded inputs match exactly;
GPU FP16 reproduces the saved outputs exactly. GPU FP32 (TF32 disabled) versus
CPU FP32 differs by at most **4.77e-7** on those 12 rows. This isolates a mixed-
precision tail effect in this check, rather than changed cached inputs. It does
not bound every FP32 CUDA/CPU output. The auxiliary kernel was deleted afterward.

The current small-text worker exposes **`--precision fp32`** for explicit full-
precision training and inference; default `amp` keeps the original behavior.
The initial training results remain AMP results. CPU inference replay took
7.95 s for all three windows after warmup, versus the original T4 AMP 3.57 s;
these are single model-only observations on different hardware/precision, not
an end-to-end serving benchmark. No confirmation prices were used in this probe.

Fusion commands should pass `--external-pretrained-data` when pretrained
initialization exists anywhere upstream, including a frozen baseline whose
filename hides MiniLM. This explicit declaration prevents misleading provenance
from expert-name inference. It changes report metadata, not fitted predictions.

## Product-feature calibration extension

`mercari_frontier calibrate --product-views` adds 90 numeric features: five
product identity keys, three statistics per key (smoothed log-price mean,
support, dispersion), global/brand-category-parent priors and smoothing 5/20/100.
It fits all six maps on the **base window only**, saves them for replay and
rejects confirmation files containing prices. This reuses the
[previous product-statistics method](CPU_MULTIVIEW.md), following
[Micci-Barreca (2001)](https://doi.org/10.1145/507533.507538). Tree calibration
uses the existing metadata, LSA and out-of-calibration expert predictions; it
never uses a previously calibrated model's in-sample predictions as a fitting
feature. Positive Ridge controls use only expert predictions, so their duplicate
product-view variants are excluded from the outer pool.

This CPU extension was chosen after weak small-model selection curves and
before viewing this round's confirmation scores. It is an adaptive research
extension, not a preregistered blind experiment. The fixed comparison is the
same tree capacities with/without product views; the confirmation audit remains
reused. The full outer pool has 15 branches, with a frozen historical-best anchor,
three small experts, six basic calibrators, four product-aware tree calibrators
and one cached CPU multiscale ensemble. Branches share upstream models.
