# What the original Mercari top three teach us

[中文](TOP3.zh-CN.md) · [Measured evidence](top3-results.json) · [Beginner guide](BEGINNER_GUIDE.md)

This review checks the **original final leaderboard**, not today's practice
submissions. Its top three were **Paweł and Konstantin (0.37758), Mercaring
(Nima & Chahou, 0.38875), and bird (0.39134)**. These private leaderboard scores
come from the champions' [historical screenshot][board]. They are not comparable
to our local audit scores. Chenglong Chen's useful XNN solution was **fourth**,
not second or third.

The first-place [source][first-code] is pinned to commit
`374301ad1c32cbc93dcc40313d5d7bb9c5503746`. We retrieved the first-, second- and
third-place authors' posts and the latter two public kernels on 18 October
2026. Their URLs and content hashes are in [the evidence manifest](top3-results.json).
**Author-reported findings, inspected implementation details, and our
interpretations below have different evidential strength.** We independently
implement selected ideas; we do not reproduce the original winning systems.

## First: represent the same listing differently

The [winning write-up][first] reports 3 text representations × 4 sparse MLPs:
tokenization/stemming and word/character features vary across representations;
Huber versus soft price-bin classification and binary versus weighted features
vary within each. Binarizing an existing sparse matrix provides another view
without rerunning expensive text extraction. Numeric-word features attach a
preceding number to a word: “10 cards, 5 dolls” retains the amounts associated
with **cards** and **dolls**, rather than merely counting the digits separately.
The authors report an ensemble improvement of about .001 from adding that
numeric representation to one dataset; this is their result, not ours.

Less obvious lessons from the source and report:

- **Representations are an ensemble design choice.** The authors found that
  changing the input view helped more than changing model families on the same
  view. Joining title, brand and description also lets features share parameters.
  Interpretation: optimize error complementarity per unit of runtime, rather
  than accumulating correlated models or random seeds.
- **The schedule determines which model is affordable.** Doubling batch size
  and reducing learning rate allowed more capacity within the runtime limit.
  They intentionally used late, individually overfitted models in an ensemble.
  That is an author-reported recipe, not permission to choose checkpoints on
  audit labels; our current sparse expert retains its selection-best checkpoint.
- **Parallelize independent models when sparse operations scale poorly.** They
  restricted each model to one CPU core, trained four simultaneously, and
  processed representations in sequential processes to reclaim memory. Their
  MXNet variant used sparse row updates. Framework-specific score differences
  are not evidence that changing framework alone improves generalization.
- **A small affine correction can matter.** Their published Lasso blend has
  an intercept and need not have weights summing to one. Their report also
  says Lasso rejected unnecessary models. Our regularized affine blender can
  test that freedom while bounding coefficient mass and the intercept.

The authors report unsuccessful spell checking, special “for [name]” features,
category-specific systems/residual modeling before the merge, mixtures of
experts and hybrid skip/FM architectures. Their failures are useful priorities,
not universal impossibility results: our residual product calibrators already
have measured value on a different ensemble.

## Second: interaction geometry without expensive sequence models

The [second-place post][second] describes Ridge, word sparse NN, a shared-
embedding average-pooling NN, and a character sparse NN. Its released kernel
is **almost similar** to the winning submission: the author reports
**.38888 public / .38948 private in 2,800 s**, which must not be substituted
for the historical final score .38875.

The distinctive Ridge feature combines the title with the **first five
description words**, sorts/deduplicates tokens, and generates **every unordered
two-token combination**. This is not an ordinary adjacent bigram. “red leather
shoe” and “shoe red leather” share pairs even though their adjacent bigrams differ.

Code details that the short description does not fully convey:

- `create_bigrams` lemmatizes and removes stop words. If two tokens concatenated
  in either direction form a known training unigram, it substitutes that word
  for the pair. Interpretation: bridge “lip gloss” and “lipgloss” using a
  training-derived spelling vocabulary. We have **not** implemented this
  vocabulary substitution in this round.
- Brand/category IDs are prefixed into title text; they participate in word
  interactions, rather than remaining isolated numeric IDs.
- The embedding weights are shared between title and description, but their
  **separate averages** are concatenated. Parameter sharing does not erase field
  identity. Our newer single pooled subword bag does not reproduce that design.
- Actual loops double batch size **before** each epoch. The word sparse NN
  therefore uses 4,000 / 8,000 / 16,000, not an initial batch of 2,000.
- The kernel's submission branch averages **log-price** predictions and clips
  resulting prices to [3, 1000]. Its debug branch's fitted linear regression is
  not proof that those learned coefficients were used for submission.

The authors explicitly report that a strong LightGBM did not improve their
blend. Interpretation: standalone accuracy is insufficient if the model repeats
existing errors. Their timing and clipping thresholds are historical choices;
we do not copy their thresholds into a different validation protocol.

## Third: preserve useful symbols, specialize the blend

[bird's post][third] reports a CNN around .400 and FM-FTRL around .415, yet a
blend around .3905 public, with substantial shared text normalization work.
This directly motivates testing complementary errors instead of discarding an
expert merely because its individual metric is worse.

The [released kernel][third-kernel] also shows details absent from the brief post:

- Accented characters are normalized, `+` becomes “plus”, and selected symbols
  receive tokens. Some isolated multi-digit numbers become a generic token in
  a separate neural preprocessing stage. Interpretation: alternate invariance
  patterns can complement exact model-number features. We preserve exact
  numeric tokens in the new view instead of globally masking product numbers.
- Short convolution branches and attention branches use different cropped
  lengths. Training changes sample slices and batch sizes; the later refinement
  switches from Adam to Adagrad. These are implementation observations without
  isolated author ablations, not proven improvements to transplant wholesale.
- The output adds linear, ReLU and Maxout branches trained jointly. This is
  another architectural detail, not three independently trained ensemble members
  and not evidence that this head will improve our current networks.
- `catew` assigns a CNN share from .50 to .65 by top-level category, rather
  than always using the default .60. Our existing `ContextBlend` learns category
  weights with support requirements and shrinkage instead of copying that table.
- Test processing is chunked to limit memory. This matters for the much larger
  final-stage input, not just for the local validation run.

## What was incorporated

The new [independent feature/execution module](../mercari_top3.py) adds:

1. A complementary normalized lexical view: Unicode accent folding, separate
   title/description namespaces, `plus`/`heart`/currency/percent tokens, decimal
   and letter–number separation, and structured feature tokens.
2. Bounded unordered token pairs from title plus five description tokens.
   There are at most 20 unique tokens / **190 pairs per listing** by default.
   Feature hashing bounds dimensionality and avoids a corpus-sized pair
   dictionary; collisions and the token cap are explicit compromises.
3. Numeric successor features for amounts in (0, 200], with **log1p(amount)**
   values and separate title/description namespaces. The original used amount
   divided by 200 and a different repeat policy; ours is an independent adaptation.
4. A four-way factorial study: lexical only, +pairs, +quantities, +both. All
   Ridge experts use fixed alpha 20, LSQR tolerance .002 and at most 80 iterations,
   trained on the same base rows. IDF fits base only; quantity values are not
   normalized to unit length, which would discard their magnitude for singleton
   rows. No labels or row IDs enter extraction.
5. Selection-side comparison of existing sum-one, affine and category-aware
   blenders, with a grouped half-selection check. The original anchor, all
   choices and the promotion decision freeze **before** opening audit labels.

Previously implemented pieces remain important: separate/joint/stemmed sparse
views, binary/TF-IDF channels, Huber and soft-bin objectives, growing batches,
row-sparse Adam, product statistics, competitor calibrators and regularized
category blending. Merely citing the top three does not make these new results.

## Full-base experiment: useful feature, rejected ensemble extension

All four models use **1,186,026 base rows**, with 74,129 calibration, 74,126
selection and 148,254 reused confirmation rows. Only base labels fit new Ridge
models. Each hash block has 262,144 columns; no row sampling was used.

| New expert | Selection RMSLE | Reused audit RMSLE | Fit + CSR assembly s | Three-window inference s | LSQR iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized lexical | 0.496025 | 0.495846 | 3.78 | 0.044 | 26 |
| + unordered pairs | 0.473492 | 0.472611 | 15.56 | 0.144 | 30 |
| + quantity words | 0.511224 | 0.512372 | 6.93 | 0.054 | 62 |
| + both | 0.496204 | 0.496618 | 17.74 | 0.132 | 64 |

Unordered pairs improve this new lexical control by **4.54% on selection** and
**4.69% on audit**; paired row-bootstrap audit delta is
**-0.023235 [-0.023861, -0.022735]** (300 replicates,
one fitted run). This does not make the new expert stronger than our existing
ensemble. Numeric successors worsen both raw variants under this hash/Ridge
configuration. Their scale/regularization and lack of nonlinear interactions
are hypotheses to test, not established explanations or a refutation of the
champions' sparse-MLP results.

**The promotion gate rejected the extension.** The full pool selected sum-one
weights with an intercept; every new expert received effectively zero weight.
Affine and category-aware variants also failed to improve the grouped check.
The raw anchor's selection RMSLE remains **0.382191247**, and the exact promoted
predictions retain **0.380616255 audit RMSLE**. The fitted pool's tiny intercept
(-0.00009673) gives audit .380616369, also no improvement; it was not deployed.
Removing either feature family gives the same fitted-pool predictions to
numerical tolerance. Default output `ensemble.npz` is **bitwise identical to
the anchor**, not a rounded-score tie. No new Kaggle submission was made.

The complete incremental run took **211.59 s**, including
**151.16 s** of feature preparation; peak process RSS was
**2609.71 MiB**. Inference times above include CSR assembly for
296,509 already prepared rows and exclude text extraction. These are single
observations, not means of repeated performance trials. They exclude all old
anchor training, run on a 4-CPU-quota Linux machine and allocate no GPU.

Selection and audit have been inspected in previous research. Freezing this
round's choices prevents direct audit-driven promotion, but does not turn those
windows into a new blind test. The unchanged-output bootstrap delta is exactly
zero by construction, not evidence about generalization uncertainty. New code
is covered by a TDD red/green record, real Ridge fitting, artifact reload,
invalid-input/audit-boundary tests and a failed-promotion preservation test.


Validation: **161 tests passed**, including **8 new tests**. The 22 covered
Mercari CPU experiment modules have **2,891/2,891 statements covered (100%)**;
the new module has **148/148 (100%)**. This is statement coverage, not branch or
whole-repository coverage. Existing Keras/NumPy, SVD and runpy warnings remain.
CI now includes these tests and the coverage gate.

## Reproduce and interpret

Use the prepared grouped windows and previous best anchor described in
[the small-text methods](SMALL_TEXT.md) and [previous report](SMALL_TEXT_REPORT.md).
Large data, checkpoints and run directories stay ignored by Git. Reproducing
the complete anchor requires its upstream experts; the incremental cost below
does not include that training. For an isolated new-expert smoke run, `--anchor`
can instead refer to an ID-aligned baseline archive from the preparation step.

```bash
python -m pip install '.[mercari,benchmark]' pytest pytest-cov
python -m pytest tests/test_mercari_top3.py \
  --cov=benchmarks.mercari_top3 --cov-report=term-missing --cov-fail-under=100
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m benchmarks.mercari_top3 \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --anchor benchmarks/runs/mercari/smalltext-t4-001/primary-blend/ensemble.npz \
  --output-dir benchmarks/runs/mercari/top3-reproduction-001 \
  --external-pretrained-data
```

`--rows 10000 --n-features 16384 --max-iter 20` bounds a smoke experiment, which
must be reported as sampled. Use a new output directory. `--external-pretrained-data`
declares the old anchor's pretrained lineage, even though these new experts use
no pretrained weights. Inspect `plan.json`, `frozen.json`, `report.json`, the
four expert NPZs and `ensemble.npz`; trusted local pickles retain the encoder,
Ridge models and blenders for inference. Sparse CPU preprocessing/LSQR does not
automatically benefit from a GPU; this study allocates none.

For a beginner: **RMSLE** is ordinary RMSE after converting prices to log1p.
The model learns log-prices, blends those predictions, then applies expm1 for
prices. A **word pair** detects co-occurrence at any position; a **quantity
feature** retains how much of which word was mentioned. An **ablation** removes
one ingredient to check whether it contributes beyond the rest. **Ensembling**
helps when errors differ: for a sum-one blend, squared error depends on
`wᵀ E[eeᵀ] w`, including cross-model error products, not only each model's RMSE.
This is a second moment, equal to covariance only for zero-mean errors.

The next justified experiments are the second-place train-only concatenation
vocabulary and separate field pooling, and a controlled native-FP16 small-model
accuracy rerun. They remain future work. Bigger ensembles, arbitrary price
clamps, aggressive number masking and late overfitting are not enabled by default.

Background reading: [feature hashing (Weinberger et al., 2009)](https://arxiv.org/abs/0902.2206)
explains the bounded vocabulary tradeoff;
[growing batch sizes (Smith et al., 2018)](https://arxiv.org/abs/1711.00489)
provides context for the authors' training schedules; and
[model-selection bias (Cawley & Talbot, 2010)](https://jmlr.org/papers/v11/cawley10a.html)
explains why repeatedly inspecting the same validation windows limits conclusions.
The competition-specific evidence is the primary posts and source code above.

[first]: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50256
[first-code]: https://github.com/pjankiewicz/mercari-solution/tree/374301ad1c32cbc93dcc40313d5d7bb9c5503746
[board]: https://github.com/pjankiewicz/mercari-solution/blob/374301ad1c32cbc93dcc40313d5d7bb9c5503746/presentation/img/leaderboard.png
[second]: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50499
[third]: https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50272
[third-kernel]: https://www.kaggle.com/code/whitebird/mercari-price-3rd-0-3905-cv-at-pb-in-3300-s
