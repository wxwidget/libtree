# Python explanations and ensembles

[中文](python-integration.zh-CN.md) · [Tree/path inspection](explain.md) · [C5 rules](c5.md)

All three estimators—`GBDTRegressor`, `GBDTClassifier`, `C5Classifier`—support
sklearn cloning, estimator tags, fitted-state checks, Pipeline, grid search,
permutation importance, Voting, Stacking and Bagging. They remain NumPy-only
when these optional integrations are unused.

```sh
python -m pip install '.[sklearn,explain]'
# Published package name: pip install 'libtree-gbdt[sklearn,explain]'
```

## Explain a prediction with SHAP

A tree path says **which comparisons were made**. A SHAP attribution says **how
much each feature changes an output relative to a background population**.
These are different questions. Use `dump_model`, `dump_rules` and `explain` for
exact native tree/rule inspection; use the following interface for attribution.

```python
import numpy as np
from libtree import C5Classifier, GBDTClassifier, GBDTRegressor

rng = np.random.default_rng(42)
x = rng.normal(size=(120, 3)).astype(np.float32)
y = np.where(x[:, 0] + x[:, 1] > 0, "risk", "safe")

with C5Classifier(n_estimators=5, min_samples_leaf=2).fit(x[:100], y[:100]) as model:
    explainer = model.shap_explainer(
        x[:20], algorithm="permutation", seed=42,
        feature_names=["income", "debt", "age"],
    )
    explanation = explainer(x[100:103], max_evals=70)
    print(explanation.values)       # (rows, features, classes)
    print(explanation.output_names) # same order as model.classes_, as strings
    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        model.predict_proba(x[100:103]), atol=2e-7, rtol=2e-7,
    )
    # With matplotlib installed:
    # import shap
    # shap.plots.waterfall(explanation[0, :, 1])
```

Use the same `shap_explainer` method on either GBDT model. Regression returns
`values` with shape `(rows, features)` in target units. Classification returns
`(rows, features, classes)` on the **probability** scale, including both binary
columns. `base_values + values.sum(axis=1)` reconstructs `predict` for regression
or `predict_proba` for classification, up to numerical precision. Class label
predictions themselves are not attributed. C5 probabilities remain uncalibrated
normalized-vote softmax scores; an explanation does not calibrate a model.

`algorithm` accepts `auto`, `exact`, or `permutation`. The adapter returns a
callable `ShapExplainer`; calling it returns a standard `shap.Explanation`, usable
with SHAP plots. Its `explainer` attribute exposes the underlying SHAP explainer.
The wrapper preserves class names across SHAP versions using the public
Explanation API. DataFrame backgrounds supply column names automatically; classifiers also
accept pandas string targets.

The implementation uses public `shap.Explainer` and `shap.maskers.Independent`
APIs. It does **not** provide a native TreeSHAP/`shap.TreeExplainer` adapter.
All fitted LibTree estimators are also callable, so the standard interface works:

```python
import shap
# Regression callable returns values; classifier callable returns all probabilities.
# Standard SHAP may generate generic output names; use model.classes_ to map them.
# explainer = shap.Explainer(model, background, algorithm="permutation", seed=42)
# explanation = explainer(rows, max_evals=70)
```

Choose a small, representative **training-only** background (for example,
20–100 rows sampled with a fixed seed). All supplied rows are retained; the
adapter performs no hidden subsampling. Exact enumeration costs exponentially
in the number of varying features. For wider tables choose `permutation` and
set a budget of at least `2 * features + 1` evaluations; use multiples of that
budget to reduce sampling noise. Cost also grows with explained rows and
background rows. SHAP additivity alone does not establish attribution stability.

Independent masking replaces hidden features with background values. Correlated
features can therefore produce unrealistic combinations. These explanations are
background-dependent model attributions, not causal claims or conditional SHAP.
NaN routing is supported. Native prediction uses float32 feature precision.

## Compose models with sklearn

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from libtree import C5Classifier, GBDTClassifier, shap_explainer

# x/y from the example above. Each classifier uses two string classes.
voting = VotingClassifier(
    estimators=[("c5", C5Classifier(n_estimators=5, n_jobs=1)),
                ("gbdt", GBDTClassifier(n_estimators=20, n_jobs=1))],
    voting="soft", n_jobs=2,
).fit(x[:100], y[:100])

# Explain the actual ensemble, including averaging and any final estimator.
values = shap_explainer(voting, x[:20], algorithm="exact")(x[100:103])
np.testing.assert_allclose(values.base_values + values.values.sum(1),
                           voting.predict_proba(x[100:103]), atol=2e-7)

stacking = StackingClassifier(
    estimators=[("c5", C5Classifier(n_estimators=3)),
                ("gbdt", GBDTClassifier(n_estimators=10))],
    final_estimator=LogisticRegression(), cv=3, n_jobs=2,
).fit(x[:100], y[:100])

bagging = BaggingClassifier(C5Classifier(n_estimators=3),
                            n_estimators=3, random_state=42, n_jobs=2)
pipeline = make_pipeline(StandardScaler(), bagging).fit(x[:100], y[:100])
```

Use `VotingRegressor`, `StackingRegressor` and `BaggingRegressor` with
`GBDTRegressor`. `shap_explainer(pipeline_or_ensemble, background)` explains the
whole composed model on its input feature scale. Fit preprocessing only within
the training folds; Stacking uses out-of-fold predictions. More ensemble stages
do not guarantee better quality or speed. Keep validation/test data separate.

| Contract | GBDTRegressor | GBDTClassifier | C5Classifier |
| --- | --- | --- | --- |
| `clone`, Pipeline, GridSearchCV, permutation importance | yes | yes | yes |
| Voting / Stacking / Bagging, including `n_jobs=2` processes | yes | yes | yes |
| Labels | numeric target | exactly two finite numeric/string classes | 1–256 finite numeric/string classes |
| `fit(sample_weight=...)` | no | no | yes |
| Additional boosting ensemble | AdaBoostRegressor (resampling) | AdaBoostClassifier unsupported: requires sample weights | AdaBoostClassifier; use `boosting="none"` for a single base tree |
| SHAP value/probability additivity | yes | yes | yes, including multiclass |

`GBDTClassifier.classes_` is sorted and determines probability column order.
Labels such as `"risk"/"safe"` or `2/7` are supported; the C++ GBDT/C ABI still
uses encoded 0/1 targets. The native GBDT path JSON explains the encoded positive
class at index 1. Each training fold/bootstrap passed to GBDT classification must
contain both classes. C5 can fit one-class samples. Dense numeric features and
NaN are supported; sparse matrices, categorical strings, multi-output targets,
advanced sklearn metadata routing and automatic sample-weight routing are not.
Do not pass weights to unsupported GBDT fit methods; they are never ignored.

Set outer ensemble `n_jobs` and inner estimator `n_jobs` together: outer 4 and
inner 4 can create up to 16 workers. Start with inner 1 when parallelizing across
estimators. Never refit, close, or change runtime threads while the same fitted
model is being predicted or serialized.

## Model snapshots and process safety

`pickle`, `copy.deepcopy` and joblib process transport store versioned native
model text, Python parameters, class labels and fitted metadata. Native pointers,
finalizers and training matrices are excluded. Restoring creates an independent
native owner; closing the source does not invalidate the restored estimator.
Predictions, rules, tree JSON and training loss/estimator errors survive a round
trip without retraining. Pending `set_params` values remain pending and do not
rewrite the fitted trees. Closed and unfitted models remain unfitted.

The C++ C5 API adds `SaveModel`/`LoadModel`, with load validation and a strong
failure guarantee. GBDT now writes format version 2 to retain training losses;
loading existing version-1 files remains supported. The C ABI adds
`LtSerialize`/`LtDeserialize` and their `LtC5` counterparts with caller-owned
query/write buffers. Runtime threads are chosen at restoration rather than
written into portable model text. Snapshots cap total native nodes at one
million (and GBDT trees at ten thousand). These formats contain model state, not
training data, though any learned model can still disclose information.

Use pickle/joblib only with trusted files, and the same LibTree package version
in each process. Python pickle is not a promised cross-version model format.
A closed model's sklearn fitted check fails even when historical attributes
such as `classes_` remain available.

## Reproduce the compatibility checks

```sh
python -m pip install '.[dev,benchmark]'
make test
make coverage          # 100% Python statements and native source lines required
make sanitize          # Address/UndefinedBehavior/LeakSanitizer
make thread-sanitize   # native data race checks
PYTHONPATH=python python -m pytest tests/test_integration.py -v
```

Tests first demonstrated missing sklearn tags and missing C5 persistence APIs.
They now exercise three models across real process-based ensembles, multiclass
and NaN SHAP additivity, string labels, DataFrame names, optional dependency
errors, invalid backgrounds, failed serialization, failed refits, independent
ownership and restored training histories. These are compatibility tests, not a
new quality/performance benchmark; existing benchmark reports remain snapshots
of their stated source revisions.

## References

- sklearn: [Developing estimators](https://scikit-learn.org/stable/developers/develop.html),
  [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html).
- SHAP: [Explainer](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html),
  [Independent masker](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html),
  [Exact explainer](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html),
  [Permutation explainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*,
  [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
