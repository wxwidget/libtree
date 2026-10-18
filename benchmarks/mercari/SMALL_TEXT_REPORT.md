# Small text models, product calibration and native T4 precision

[中文](SMALL_TEXT_REPORT.zh-CN.md) · [Methods and commands](SMALL_TEXT.md) · [Machine-readable evidence](smalltext-results.json)

**18 October 2026:** the frozen local candidate improves RMSLE from **0.380907 to
0.380616**, an absolute reduction of **0.00029060 / 0.0763%**. The 300-replicate
paired row bootstrap gives delta **[−0.00036465, −0.00021905]**, conditional on
these fitted models. Confirmation was inspected in earlier work: this is a
reused audit, not a blind test or a first-place claim. Actual Kaggle practice
best remains **0.39199 private / 0.39110 public**; no new submission was made.

The new raw small models are weaker than the historical MiniLM expert. The
useful result is **conditional calibration and product features**, plus fixing
an inefficient GPU precision choice. Native FP16 improves a controlled T4
training-step microbenchmark by **3.31×** and inference batch by **2.70×**.
These are not complete-training speedups. The corrected FP16 policy has not
received a full accuracy rerun; the frozen score above uses the original
**emulated-BF16** run at commit `3828d76`.

## Controlled small-model results

All models train on the same 1,186,026 base rows; calibration/selection/reused
confirmation contain 74,129 / 74,126 / 148,254 rows. Every new neural model is
capped at 8 million parameters. Length is 96, head width 128 and seed 42.
Bag models use four epochs/batch 2,048, BERT-tiny three epochs/batch 512.
Best checkpoints use selection only. The tokenizer/pretrained BERT revision is
`6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`.

| Expert | Total parameters | Selection RMSLE | Confirmation RMSLE | Mean epoch s | Three-window inference s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Subword bag / MSE | 3,196,069 | 0.464184 | 0.462618 | 27.84 | 3.57 |
| Subword bag / Huber | 3,196,069 | 0.464129 | 0.462915 | 28.69 | 3.97 |
| BERT-tiny / MSE | 4,564,501 | 0.448773 | 0.447619 | 319.93 | 22.67 |

Epoch times include training and selection evaluation. They average recorded
epochs, not independent training repetitions. Inference covers 296,509 already
encoded rows. Full worker times are 676.90 / 683.64 / 1,536.34 s; input preparation
alone costs 561.61 / 564.59 / 553.52 s. A brief precision investigation overlapped
BERT training. No hardware-neutral speed ranking is inferred. Historical
MiniLM selection RMSLE is 0.416930, with a larger encoder, four epochs, length
128 and A100 hardware; it is not a matched performance comparison.

The bag is an independent fastText-inspired regression adaptation using
**WordPiece averages and hashed adjacent pairs**. It is not the official
fastText classifier or its character-ngram implementation. Both bags still
improve in their final epoch, and BERT-tiny improves .475148 → .458221 → .448773.
Thus the budgeted results show underfitting risk, not the models' ultimate limit.
MSE directly learns standardized log-price squared error; Huber is robust but
not the exact RMSLE objective. No confirmation-driven epoch/loss choice was made.

## Ensemble and feature ablation

The historical best is the outer anchor. New calibration fits only on the
74,129 calibration rows, using experts that did not fit those labels. The old
calibrated reference is never an in-sample fitting feature. Product statistics
fit only base labels, with global/brand-category-parent priors and smoothing
5/20/100. A nonnegative sum-one outer blend minimizes selection squared log
error plus penalty 1e-4. A gain threshold of 1e-5 was fixed before audit.

| System | Selection RMSLE | Reused confirmation RMSLE |
| --- | ---: | ---: |
| Reference + raw small experts | 0.382498 | 0.380907 |
| Reference + cached CPU system | 0.382477 | 0.380894 |
| Reference + basic small-expert calibration | 0.382440 | 0.380854 |
| Basic calibration + cached CPU system | 0.382397 | 0.380819 |
| Product-aware pool without cached CPU system | 0.382233 | 0.380648 |
| **Full product-aware pool** | **0.382191** | **0.380616** |

The full pool has 15 branches, five with weight above 1e-8: historical system
38.46%, product-aware LibTree depth 5 **26.59%**, LightGBM 20.10%, XGBoost 10.02%,
and cached CPU system 4.84%. These branches share many upstream fitted models;
they do not constitute a five-model fresh training recipe.

Direct small-expert weights are zero. Their predictions still enter calibration:
the product LibTree depth-5 model splits 38 times on bag Huber, 50 on bag MSE
and 129 on tiny BERT, out of 8,616 splits; 1,042 splits use product statistics.
Counts demonstrate use, not causal importance. Removing that LibTree branch
and refitting selection weights worsens audit RMSLE to **0.380675**. Its
[first tree text](smalltext-calibration-tree-0.txt) names the LSA/product/expert
features; the 32 legacy metadata columns retain numbered labels. Leaf values
are residual log-price adjustments multiplied by learning rate .04, not prices.

The original six calibrators replay **exactly** on all 296,509 rows. The basic
and product variants use identical base mixing weights and residual target;
only the tree feature columns differ (215 versus 305). Product features help
the deeper calibrators, but slightly worsen depth 3. The small-model extension
and product extension were adaptive selection-stage research, not preregistered
experiments. Ablations were reported after freezing the primary choice and did
not replace it. The independent gate's column order differs from the generic
runner, producing ~1.8e-6 coefficient rounding differences; the authoritative
`primary.frozen` weights were written before confirmation access and both pass
the same selection threshold.

## LibTree and competitor calibration

All use 400 rounds, learning rate .04 and four CPU threads. Capacities and leaf
constraints differ. Times include fit, three-window prediction and strength
selection; they exclude feature preparation and serialization. Each is one
observation, not a whole-library speed ranking.

| Calibrator | Basic s | Basic audit RMSLE | Product s | Product audit RMSLE |
| --- | ---: | ---: | ---: | ---: |
| LibTree depth 3 | 9.01 | 0.380946 | 10.39 | 0.380954 |
| LibTree depth 5 | 12.12 | 0.381016 | 15.57 | 0.380905 |
| LightGBM 15 leaves | 11.17 | 0.380964 | 10.86 | 0.380855 |
| XGBoost depth 4 | 10.55 | 0.381039 | 11.69 | 0.380847 |

LibTree depth 3 is useful in the basic view. In the product view, LightGBM and
XGBoost are stronger standalone and faster than LibTree depth 5. LibTree's
measured advantage here is complementary ensemble contribution, not universal
accuracy or performance superiority. No native C++ learner changed in this round.

## Resource, precision and limitations

Two sequential T4 allocations (one full study and one short hardware probe) used
roughly 52 minutes; all runtimes and the diagnostic kernel were released.
Observed rounded balance moved 190.81 → 189.92 CU, about .89 CU; this is not a
settled billing receipt. The model checkpoints are retained locally. Older
full-system training costs are excluded. Product maps add **1,046,998,691 bytes**
on disk: the small neural parameter cap is not a small whole-ensemble footprint.

[Input-cache evidence](smalltext-input-cache-001.json) records 261.34 s for one
full local encoding/write, versus **0.86668 s mean of five warm verified loads**,
with every array equal. TSV reading is excluded; local tests overlapped part
of the first preparation. This is encoding reuse, not a complete GPU speedup.

[Native AMP evidence](smalltext-native-amp-001.json) confirms T4 SM75, PyTorch
2.11.0+cu130: BF16 default support is true, native support false. The old helper
therefore selected emulated BF16. Excluding emulation chooses FP16. In the fixed
4,564,501-parameter architecture microbenchmark, mean training steps are
116.69 → 35.28 ms; inference batches 30.53 → 11.31 ms. Random initialization and
synthetic zero targets establish performance only. A100 can retain native BF16.
`--precision fp32` is available for tighter numerical replay.

The [initial replay failure](smalltext-precision-001.json) is retained: 23/296,509
automatic-AMP outputs exceed a predeclared .01 difference from CPU FP32, maximum
.0165677 log units. The earlier FP16 label was corrected to emulated BF16;
values and the failed tolerance remain unchanged. A matched-batch probe of the
12 worst rows reproduces old AMP exactly; native GPU FP32 agrees with CPU FP32
to 4.77e-7 on those rows. This does not bound every FP32 prediction.

Most category groups improve, but missing-category listings and low-priced
listings can worsen; complete category/price-band audit results are in JSON.
Bootstrap intervals cannot remove repeated-selection bias or estimate training
seed variability. Neither a first-place result nor another 10% quality gain is
claimed. The full corrected-FP16 candidate needs a fresh quality run and an
eligible source-trained Kaggle rerun before any new leaderboard claim.

## Reproduce and learn

Use the [beginner guide](BEGINNER_GUIDE.md), [controlled methods/commands](SMALL_TEXT.md)
and [earlier full-study preparation](FRONTIER.md). The current launcher uses
verified input caching and hardware-native AMP; original timing/quality uses
`3828d76`. The historical frozen reference and CPU branches must be refitted or
restored from trusted artifacts; this is an extension/reuse experiment.

```bash
# After preparing the same grouped windows and authenticating the official CLI:
python -m benchmarks.mercari_colab --gpu T4 --profile smalltext \
  --data-dir prepared-windows --output-dir new-colab-study \
  --cli colab --include-models
# Append the resulting expert folders to the existing frontier calibrate command.
# Compare it against the same command with --product-views, then run ensemble
# with the historical best as baseline and --external-pretrained-data.
```

The [JSON evidence](smalltext-results.json) preserves every candidate hash,
configuration, source version, learning curve, model/family ablation and frozen
weight. Raw data, weights, maps and predictions stay Git-ignored. Methods cite
[Joulin et al.](https://aclanthology.org/E17-2068/),
[Turc et al.](https://arxiv.org/abs/1908.08962),
[Micci-Barreca](https://doi.org/10.1145/507533.507538), and the
[PyTorch BF16 API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html).

Local checks pass **41 tests / 845 statements / six modules / 100% statement
coverage**. All four [implementation CI jobs](https://github.com/wxwidget/libtree/actions/runs/39148472156)
pass, including the existing native sanitizer/coverage checks. This is not
whole-repository branch or GPU-instruction coverage. No new Kaggle submission
was triggered: the corrected precision policy still lacks full quality replay.
