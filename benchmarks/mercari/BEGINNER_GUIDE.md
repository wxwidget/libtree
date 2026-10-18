# From a product description to a price: a beginner's Mercari guide

[中文](BEGINNER_GUIDE.zh-CN.md) · [Technical report](TECHNICAL_REPORT.md) · [Full experiment](OPTIMIZATION.md)

You need basic Python, a Linux terminal and some patience. Start with a tiny
synthetic example, then decide whether to run the official 1.48-million-row
training set. The tiny example needs **no Kaggle account or token** and does
not reproduce the published accuracy or performance numbers.

## 1. Understand the problem before installing anything

A row describes an item: title, description, brand, category, condition and
shipping. A **feature** is information available when predicting. The **label**
is the price known during training. A model learns a mapping from features to
labels; a prediction should not read the item's actual price.

**Training** changes model parameters. **Inference** uses a fitted model.
**Loss** guides training; an **evaluation metric** compares final predictions.
Our metric is RMSLE, where smaller is better:

```text
RMSLE = sqrt(mean((log1p(predicted_price) - log1p(actual_price))²))
```

`log1p(x)` means `log(1+x)`, so zero prices are valid. Predicting 20 instead of
10 gives error `log(21)-log(11) ≈ .647`; predicting 200 instead of 100 gives
`log(201)-log(101) ≈ .688`. The metric emphasizes proportional rather than
absolute dollar errors, approximately away from zero. It is not a percentage
accuracy. A 10% improvement here means `.43776 * .9 = .393984` or lower.

## 2. What the different models do

| Concept | Plain-language example | Why it helps here |
| --- | --- | --- |
| Token / n-gram | `iphone`, `6`, `64gb`; adjacent words or characters | Preserve model and size information |
| Sparse matrix / CSR | Store only present words, not every possible word | Text has many possible features but few per listing |
| Ridge regression | Add regularized word and category effects | A strong, understandable baseline |
| Field-aware factorization | Learn that a word has different value with different brands | Capture interactions using small vectors |
| MLP / word pooling | Learn nonlinear combinations of text and metadata | Complement the simpler models |
| MSE | Penalize large log-price errors quadratically | Align regression with the main metric |
| Huber | Quadratic for small errors, linear for large ones | Reduce the influence of very large errors |
| Price-bin classification | Predict probabilities across log-price intervals | Supply a different error pattern for blending |
| Ensemble | Combine several model predictions | Helps when their mistakes differ; costs more to run |
| Residual | Actual log price minus the blend's log prediction | What LibTree learns to correct |
| GBDT | Add small trees that correct remaining loss gradients | Learn nonlinear corrections from expert predictions and metadata |

The actual calculation is:

```text
base_log_price = weighted average of expert log predictions
final_log_price = base_log_price + LibTree correction
price = expm1(max(final_log_price, 0))
```

We average on the **log-price scale**, not the dollar scale. Several trees
inside GBDT and several different models in an ensemble are two different
levels of combination. Read the offline [tree learning guide](../../docs/LEARNING_GUIDE.html)
if trees are new to you.

For example, a correction of `.1` multiplies `price + 1` by `exp(.1)`, about
1.105. This gives the log-scale correction a concrete interpretation.

## 3. Install a known source snapshot

Use Linux or WSL2 with **Python 3.12 and a C++17 compiler**. The Mercari extra
has stricter version requirements than the base LibTree package. The full run
was measured with four CPU cores and a 32-GiB memory limit; peak RSS was about
10 GiB. Allow ample disk space for downloaded archives, extracted data, caches,
models and predictions; the small exercise needs far less.

From a parent directory, if you do not already have a checkout:

```sh
git clone https://github.com/wxwidget/libtree.git
cd libtree
git checkout e711973
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[mercari]' pytest pytest-cov kaggle py7zr
python -c 'import libtree; libtree._library(); print("Native library loaded")'
```

Run every remaining command from the repository root with this virtual
environment activated. The measured release is pinned above; documentation
added later can also be read on GitHub. `benchmarks` are repository modules,
so keep `PYTHONPATH=.`. Using `PYTHONPATH=python:.` can accidentally select an
unbuilt native source package instead of the installed wheel.

## 4. First run: small synthetic data, no credentials

This creates 240 artificial training rows and 16 prediction examples, using
the fast pipeline with much smaller dimensions and one epoch. Prices and text
are synthetic. It checks wiring and output interpretation, not generalization.
Use a fresh directory name if `tutorial-smoke` already exists.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("benchmarks/runs/mercari/tutorial-smoke")
root.mkdir(parents=True, exist_ok=False)
data = root / "data"
data.mkdir()
rng = np.random.default_rng(42)
rows = 240
condition = rng.integers(1, 6, rows)
brand_index = np.arange(rows) % 3
frame = pd.DataFrame({
    "train_id": np.arange(rows),
    "name": [f"phone {i % 8} case item {i}" for i in range(rows)],
    "item_description": [f"phone accessory colour {i % 5}" for i in range(rows)],
    "category_name": "Electronics/Phones/Accessories",
    "brand_name": np.array(["Acme", "Bravo", "Comet"])[brand_index],
    "item_condition_id": condition,
    "shipping": np.arange(rows) % 2,
    "price": 8 + 12 * brand_index + 4 * (6 - condition) + rng.uniform(0, 2, rows),
})
frame.to_csv(data / "train.tsv", sep="\t", index=False)
query = frame.iloc[:16].drop(columns="price").rename(columns={"train_id": "test_id"})
query["name"] = [f"phone x case new {i}" for i in range(len(query))]
query.to_csv(data / "test.tsv", sep="\t", index=False)
pd.DataFrame({"test_id": query.test_id, "price": 0}).to_csv(
    data / "sample_submission.csv", index=False)

config = json.loads(Path("benchmarks/mercari/fast-config.json").read_text())
config.update(threads=2, max_features=500, char_features=250, max_words=500,
              neural_width=8, fm_workers=1, fm_rank=2, fm_epochs=1,
              neural_rates=[.003], neural_batches=[32],
              sparse_word_limit=500, sparse_char_limit=250, sparse_min_df=1)
for member in config["neural_members"]:
    member.update(neural_width=8, neural_rates=[.003], neural_batches=[32])
config["native"].update(n_estimators=4, max_depth=2, max_bins=16,
                        min_samples_leaf=2, n_jobs=2)
(root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
```

Save that block as `/tmp/mercari_tutorial_data.py` and run it with `python`.
Then train and predict:

```sh
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/tutorial-smoke/data \
  --config benchmarks/runs/mercari/tutorial-smoke/config.json \
  --output-dir benchmarks/runs/mercari/tutorial-smoke/output \
  --mode final --test-file test.tsv --sample-file sample_submission.csv
```

Expected outputs are `report.json`, `submission.csv`, `model.pkl`, `tree.txt`,
`tree.json` and `explanation.json`. Tree files contain the **first tree**, not
the full ensemble. The report records parameters, splits, hashes and timing.
An output directory must be empty to protect previous runs.

Reload and check the outputs; save this block to `/tmp/mercari_tutorial_check.py`
and run `PYTHONPATH=. python /tmp/mercari_tutorial_check.py`:

```python
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("benchmarks/runs/mercari/tutorial-smoke")
query = pd.read_csv(root / "data/test.tsv", sep="\t")
saved = pd.read_csv(root / "output/submission.csv")
with (root / "output/model.pkl").open("rb") as stream:
    model = pickle.load(stream)
try:
    log_price = model.predict(query)
    price = np.expm1(np.maximum(log_price, 0))
    np.testing.assert_allclose(saved.price, price, rtol=1e-6, atol=1e-6)
    explanation = model.explain(query.iloc[:1])
    total = (sum(explanation["weighted_expert_log_contributions"])
             + explanation["weighted_native_correction"])
    assert np.isclose(total, log_price[0])
    assert np.isclose(explanation["price"], price[0])
    assert saved.test_id.tolist() == query.test_id.tolist()
    assert len(saved) == 16 and np.isfinite(price).all() and (price >= 0).all()
    assert (root / "output/tree.txt").read_text().strip()
    assert json.loads((root / "output/tree.json").read_text())
    print("PASS: 16 predictions, reload, tree export and explanation sum")
finally:
    model.close()
```

Only load your own trusted pickle files: loading a pickle can execute code.
`predict()` here returns **log prices**. The explanation exposes weighted expert
terms and native residual paths. Their sum is a log prediction, not dollar
contributions, causal effects or SHAP values. See [prediction paths](../../docs/explain.md)
and the [SHAP integration guide](../../docs/python-integration.md) for the native
estimators; the Mercari pipeline is not automatically a TreeExplainer model.

## 5. Keep experiments honest

**Overfitting** means a model learns training-specific detail and performs worse
on unseen examples. **Leakage** means information unavailable at prediction time
enters training or evaluation. Duplicate listings crossing windows, vocabulary
fitted on validation text, and target means computed with validation prices are
examples of compromised protocols.

Use separate roles: base fitting → residual calibration → parameter selection
→ frozen holdout audit. Do not repeatedly inspect holdout scores to choose
parameters. An **ablation** removes or changes one component while keeping the
comparison scope fixed; it helps identify what a component adds. It does not
make every observed difference causal when several things change together.

What this project learned:

- Preserving model tokens and learning interactions helped more than blindly
  increasing model complexity.
- Field-aware tuning RMSLE improved to .420152 at epoch 2, then worsened to
  .426392 at epoch 3 even while training error fell.
- Huber was only slightly better than MSE for the pooling model. Ensemble value
  depends on complementary errors, not the name of a loss.
- Larger sparse batches helped one accuracy experiment but slowed fast. Measure
  complete pipelines on your machine, including slow repetitions.
- The fast model keeps the score target while removing expensive components;
  the old baseline remains substantially faster than either new ensemble.
- A completed Kernel, a generated CSV and a `COMPLETE` scored submission are
  different milestones. A score does not establish historical contest rank.

The [technical report](TECHNICAL_REPORT.md#4-failures-and-decisions-worth-preserving)
contains the evidence and unsuccessful attempts.

## 6. Download official data when ready

Use your own Kaggle account. Accept the competition's applicable rules on its
[webpage](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge)
if Kaggle requires it. Configure authentication through Kaggle's supported
settings or your environment's secure secret store. Never put tokens in source,
Git, terminal arguments or screenshots. Do not create another token variable
just because an existing injected binding is not visible as a literal value.

```sh
mkdir -p benchmarks/runs/mercari/data
kaggle competitions files mercari-price-suggestion-challenge
kaggle competitions download mercari-price-suggestion-challenge \
  -p benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/mercari-price-suggestion-challenge.zip \
  benchmarks/runs/mercari/data
python -m py7zr x benchmarks/runs/mercari/data/train.tsv.7z benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/test_stg2.tsv.zip benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/sample_submission_stg2.csv.zip benchmarks/runs/mercari/data
```

The extraction commands match the archived official filenames. Check the actual
download listing: if Kaggle returns files individually or an input is already
extracted, skip that wrapper's extraction. Training expects `train.tsv`.
Final predictions need the matching test/sample pair. A zero-filled sample
submission is an output template, not test labels. A `403` may mean authentication,
rule acceptance or access requirements; read the response rather than trying to
bypass them. Full-file SHA-256 values are in [the baseline report](full-001.json).

## 7. Train, audit and predict with official data

**Study mode** fits the original base/calibration split and reports selection
RMSLE. It does not evaluate the holdout by default. This is a full-data job,
not another tiny exercise:

```sh
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-study --mode study
```

Freeze decisions first. To run a **new** holdout audit, repeat with a new output
directory and add `--audit-holdout`. It refits the pipeline; it does not simply
read the previous model. Our published holdout result belongs to accuracy, not
fast. Repeatedly optimizing against this audit invalidates its independent role.

**Final mode** refits using all labels in base/calibration roles and produces
test predictions without a local validation score:

```sh
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-final --mode final
```

For accuracy, replace `fast-config.json` with `optimized-config.json` and use a
different output directory. Retain `model.pkl` for inference; add
`--omit-model-artifact` only if you do not need to reload it. Published local
processes took 44.63/53.75 minutes for fast/accuracy, with about 10 GiB peak RSS.
These are individual observations, not a runtime promise. New machine/library
versions can change predictions and scores; preserve your own report.

## 8. Measure performance rather than guess

After training both models, compare their saved pipelines:

```sh
PYTHONPATH=. python -m benchmarks.mercari_inference \
  --baseline-model benchmarks/runs/mercari/tutorial-accuracy/model.pkl \
  --optimized-model benchmarks/runs/mercari/tutorial-final/model.pkl \
  --test-file benchmarks/runs/mercari/data/test_stg2.tsv \
  --output benchmarks/runs/mercari/tutorial-serving.json \
  --rows 100000 --repeats 4 --threads 4 --fm-threads 2
```

`tutorial-accuracy` is the output directory you choose when repeating final
training with the accuracy config. The argument names mean comparison A/B here;
they do not force the first model to be the original scored baseline. Reports
include raw samples, arithmetic mean, min, max and population standard deviation.
Timing includes feature preparation and prediction, excluding model loading,
input I/O and prediction checks. Avoid concurrent training. Four repeats are a
starting point, not a statistical guarantee.

For a same-weights batch experiment, pass the **same model** to both arguments,
use a new output file and add `--sparse-batch-size 16384`. Predictions should
remain close; inspect whether the mean actually improves. This CLI compares two
supplied models; it does not recreate the published four-variant, reused-expert
audit automatically. That experiment's scope is in [its raw report](accuracy-fast-serving.json).

## 9. Optional: reproduce a Kaggle execution and score

The source builder needs no credentials. Set your own username and distinct
Kernel names. Stage one is a **runtime check** and must not be submitted:

```sh
python -m benchmarks.mercari_kernel --owner YOUR_USERNAME --stage one \
  --kernel-name libtree-mercari-tutorial-stage-one \
  --optimized-config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-kernel-one
kaggle kernels push -p benchmarks/runs/mercari/tutorial-kernel-one -t 3600
kaggle kernels status YOUR_USERNAME/libtree-mercari-tutorial-stage-one
```

It writes `stage-one-predictions.csv` and a `STAGE_ONE_BUDGET_CHECK_NEVER_SUBMIT.txt`
marker. Fast passed this check in 44.04 minutes; accuracy's stage-one condition
remains unverified. This does not certify all historical memory/scratch rules.

Build a distinct stage-two version, then explicitly upload:

```sh
python -m benchmarks.mercari_kernel --owner YOUR_USERNAME --stage two \
  --kernel-name libtree-mercari-tutorial-entry \
  --optimized-config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-kernel-two
kaggle kernels push -p benchmarks/runs/mercari/tutorial-kernel-two -t 7200
kaggle kernels logs --follow YOUR_USERNAME/libtree-mercari-tutorial-entry
kaggle kernels status YOUR_USERNAME/libtree-mercari-tutorial-entry
```

The generated Kernel starts private, compiles C++, retrains in Kaggle and embeds
source/config only. It uploads no local model or credentials. Record the version
returned by push. Once it completes, pull its source and inspect output metadata
using Kaggle's CLI/API; match the uploaded source and require root `report.json`
and `submission.csv`. If the artifact index is briefly empty, retry metadata;
do not rerun training or blindly submit again. Direct output downloads depend
on network access; the official log stream also prints the aggregate report.

Replace `VERSION_FROM_PUSH` with that recorded number in all commands below.
The source may have a different downloaded filename; its SHA-256 should match
the locally generated script. Inspect the file listing for the two root outputs:

```sh
kaggle kernels pull YOUR_USERNAME/libtree-mercari-tutorial-entry/VERSION_FROM_PUSH \
  -p benchmarks/runs/mercari/tutorial-kernel-pulled -m
sha256sum benchmarks/runs/mercari/tutorial-kernel-two/mercari_libtree.py \
  benchmarks/runs/mercari/tutorial-kernel-pulled/*.py
kaggle kernels files YOUR_USERNAME/libtree-mercari-tutorial-entry/VERSION_FROM_PUSH \
  --format json --page-size 200
```

If the current competition service still permits scoring, submit that **exact
completed version** explicitly:

```sh
kaggle competitions submit mercari-price-suggestion-challenge \
  -k YOUR_USERNAME/libtree-mercari-tutorial-entry -v VERSION_FROM_PUSH \
  -f submission.csv -m 'LibTree tutorial: frozen fast configuration'
kaggle competitions submissions mercari-price-suggestion-challenge --format json
```

`VERSION_FROM_PUSH` is a placeholder, not literal text to keep. Save submission
ID, status, score and any error. Do not claim success while `PENDING` or on an
unscored CSV. Platform availability and an identical score are not guaranteed;
the 2026 evidence proves the archived attempts were scored, not future access.

## 10. Tests, troubleshooting and reading order

Run the existing tests before changing a model. The exact coverage command is
in the [workflow](../../.github/workflows/test.yml); a focused starting point is:

```sh
PYTHONPATH=. NUMBA_DISABLE_JIT=1 python -m pytest \
  tests/test_mercari_features.py tests/test_mercari_solver.py \
  tests/test_mercari_advanced.py tests/test_mercari_inference.py -q
PYTHONPATH=. python -m pytest tests/test_mercari_fm.py -q
```

The first uses executable reference kernels; the second exercises compiled FM.
Check pass/skip counts: a skipped dependency is not a passing model check. TDD
means write a behavior that fails first, implement the smallest change, then
refactor after it passes. Add independent checks, not tests that copy the code's
formula without a separate reference. 100% statement coverage does not prove
all branches, numerical regimes or real-world accuracy.

| Symptom | First check |
| --- | --- |
| Native library cannot load | Activate the venv, install from the repo, use `PYTHONPATH=.` |
| Output directory/file already exists | Choose a new name; retain earlier evidence |
| Wrong or missing inputs | Match stage-one/stage-two test and sample filenames; inspect nested archives |
| Full run is slow or out of memory | Finish the tiny exercise first; check CPU quota, RAM and concurrent work; shrinking config changes the experiment |
| Training loss falls but validation worsens | Overfitting; stop based on selection data, not holdout tuning |
| More threads or larger batches do not help | Measure complete cost; synchronization, sparse work and library kernels can dominate |
| Model replay produces odd prices | Check log-price conversion, model provenance and matching preprocessing |

Recommended order: [NumPy arrays](https://numpy.org/doc/stable/user/absolute_beginners.html)
→ [pandas tables](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
→ [text features](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
→ [Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
→ [grouped validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
→ [interactive tree guide](../../docs/LEARNING_GUIDE.html)
→ [technical theory and papers](TECHNICAL_REPORT.md#7-reading-and-implementation-map).

Keep raw data, pickles, predictions and scratch outputs under ignored
`benchmarks/runs/`. Commit code, configuration and aggregate evidence. Your
minimum experiment record is: question, fixed split, config/source/data hashes,
metric, raw timings, failed attempts, selection decision and an independent
final check.

The [documentation verification record](documentation-verification.json) covers
the synthetic example, replay checks, focused tests and locally built source
bundles on October 18, 2026. Full-data scores above remain the archived October
17 measurements. Local links were checked; several external learning sites
could not be reached through this environment's proxy, so their availability
was not reverified here.

## Small encoders and GPU precision

The [small-text walkthrough](SMALL_TEXT.md) and [measured report](SMALL_TEXT_REPORT.md)
explain the new bounded experiment. A bag averages learned subword vectors and
uses hashed neighboring pairs; a tiny BERT lets each token depend on its context.
Both output a number for log price. A weaker guess can still help a residual
tree decide when another guess is wrong. Caching token IDs reuses preprocessing,
not fitted price-model weights.

FP16 and BF16 both store 16-bit numbers but have different ranges and precision.
Hardware support also differs: T4 has native FP16, while A100 supports native
BF16. Being able to create a BF16 tensor does not prove fast BF16 arithmetic.
The report shows the observed failure, corrected hardware check and measured
fixed-batch costs; it keeps this distinction separate from Kaggle accuracy.
