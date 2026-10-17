# Inspect trees and explain one prediction

[中文](explain.zh-CN.md) · [Interactive beginner GBDT guide](LEARNING_GUIDE.html)

A tree asks questions about one feature at a time. A value less than or equal
to a split threshold goes left; a larger value goes right. A `NaN` follows the
direction learned during training. A leaf stores a correction to the model's
score. The whole ensemble adds corrections from all its trees.

LibTree exports the actual fitted trees as text or JSON, and records the full
path and leaf contribution for one input row. C++, Python and the CLI share
one native implementation. Inspection is read-only; do not refit or close the
same model while inspecting it.

## Start with Python

```python
import json
from pathlib import Path
from libtree import GBDTRegressor

with GBDTRegressor(n_estimators=1, max_depth=1, min_samples_leaf=1,
                   learning_rate=0.5, reg_lambda=0) as model:
    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    print(model.dump_model(feature_names=["age"]))
    Path("trees.json").write_text(
        model.dump_model(format="json", feature_names=["age"]), encoding="utf-8")
    tree = json.loads(model.dump_model(format="json", tree_index=0))
    explanation = model.explain([3], feature_names=["age"])
    print(explanation["base_score"], explanation["margin"], explanation["prediction"])
    print(model.explain([3], feature_names=["age"], format="text"))
```

The base score is `0.5`, the tree contributes `0.25`, and the regression
prediction is `0.75`. This is synthetic teaching data, not an evaluation task.

- `dump_model(format="text"|"json", tree_index=None, feature_names=None)`
  returns a **string**. Use `json.loads` to parse JSON. `None` exports all trees;
  `tree_index=0` selects the first tree.
- `explain(sample, feature_names=None, format="json")` returns a **dictionary**.
  `format="text"` returns a readable string. The sample shape must be
  `(features,)` or `(1, features)`; batch explanations are not supported.
- Feature names are display labels supplied in the original column order.
  Defaults are `f0`, `f1`, etc. Supply exactly one label per feature. They are
  not inferred from CSV headers or stored in saved models. UTF-8, quotes and
  line breaks are escaped; Python rejects names containing NUL.

Input is converted to float32 just as in prediction. Unfitted/closed models,
incorrect shapes, infinities, bad tree indices and unsupported formats raise
errors. An overflowing margin or float32 prediction is rejected rather than
written as invalid JSON.

## Understand the output

```text
tree contribution = learning_rate × leaf_value
margin = base_score + tree 0 contribution + tree 1 contribution + ...
```

For regression, the margin is the prediction before float32 rounding. For
binary classification, the margin is **log-odds**; `sigmoid(margin)` gives the
probability of class 1. The explanation's `prediction` matches the float32
value from `predict_proba([row])[0, 1]` for a one-dimensional row. The classifier's `predict` returns a
class label by applying a `0.5` probability threshold.

A negative contribution lowers the score; it does not mean a negative
probability. A NaN is a routing decision, not a numeric zero. JSON writes a
missing observed `feature_value` as `null`. `direction` is the branch actually
taken; `missing_direction` is the learned fallback, even when a value is present.
Root-only trees have an empty path but still contribute a leaf value.

These are exact paths and contributions **per tree**, rather than SHAP values
or contributions per feature. A visited feature alone does not establish a
causal effect. Node sample counts, split gains and SHAP calculations are not
retained or computed by this interface.

## C++

```cpp
#include <iostream>
#include <vector>
#include "libtree/gbdt.h"

int main() {
  libtree::Parameters p;
  p.num_trees = 1;
  p.max_depth = 1;
  p.min_samples_leaf = 1;
  p.learning_rate = 0.5;
  p.l2 = 0;
  libtree::Gbdt model(p);
  std::vector<float> x{0, 1, 2, 3}, y{0, 0, 1, 1};
  model.Fit({x.data(), 4, 1}, y);
  model.DumpModel(std::cout, libtree::DumpFormat::kText, -1, {"age"});
  float sample[]{3};
  const auto explanation = model.ExplainPrediction({sample, 1, 1});
  for (const auto& tree : explanation.trees) {
    std::cout << tree.tree_index << " leaf=" << tree.leaf_id
              << " contribution=" << tree.contribution << '\n';
  }
  model.ExplainPrediction({sample, 1, 1}, std::cout,
                          libtree::DumpFormat::kJson, {"age"});
}
```

Save this example as `inspect.cc`, then build and run it from the repository root:

```sh
make all
g++ -std=c++17 -Iinclude inspect.cc build/libtree.a -pthread -o build/inspect
./build/inspect
```

`PredictionExplanation` includes objective, base score, learning rate, ordered
tree explanations, margin and float32 prediction. Each `DecisionStep` includes
node/feature IDs, threshold, observed value, missing direction and next node ID.
Stream exports use full numeric precision and the classic locale without
changing the caller's precision or locale. Persistence now writes
`LIBTREE_GBDT 2` to retain training losses; version-1 models remain readable.

The C ABI exposes `LtDumpModel` and `LtExplainPrediction`. Query with a null
output and capacity zero; the required size includes the trailing NUL.
Allocate that many bytes and call again to fill the caller-owned buffer.
Format `0` is text; `1` is JSON. Errors return `-1` and preserve the output
buffer. An insufficient buffer still receives the required size. Do not
refit/close the model between query and write.

## CLI

```sh
./build/xgbt train --train examples/csv/train.csv --header \
  --model build/example.lt --trees 2 --depth 2 --min-leaf 1
./build/xgbt dump --model build/example.lt --format text
./build/xgbt dump --model build/example.lt --tree 0 --format json \
  --output build/tree-0.json
printf '0,1\n' > build/one-row.csv
./build/xgbt explain --model build/example.lt --data build/one-row.csv \
  --format json --output build/explanation.json
```

`explain` requires exactly one numeric feature row. Use `--header` to skip a
header. Both commands accept `--feature-names names.txt`, with one UTF-8 label
per line in feature-column order. `dump` accepts `--tree`; `explain` includes
all trees to reconstruct the full prediction. Omit `--output` to print only the
inspection document to stdout. File outputs are replaced atomically after
validation and cannot alias model, data or names inputs.

## JSON schema version 1

Both documents include `schema_version`, `objective`, `num_features`,
`tree_count`, `base_score`, `learning_rate`, `feature_names` and `trees`.
`tree_count` describes the entire ensemble even in a selected-tree dump. Tree
and node IDs are zero-based and retained when filtering.

| Document | Contents |
| --- | --- |
| Tree dump | Each tree has `tree_index` and `nodes`. Splits have `node_id`, `feature_index`, `feature_name`, `threshold`, `left_id`, `right_id`, `missing_direction`; leaves have `node_id`, `leaf_value`, `contribution`. |
| Explanation | Includes `margin` and `prediction`. Each tree has `tree_index`, `path`, `leaf_id`, `leaf_value`, `contribution`. Path steps have `node_id`, `feature_index`, `feature_name`, `threshold`, `feature_value`, `condition`, `direction`, `missing_direction`, `next_node_id`. |

Inspection documents are not loadable model files; use C++ `SaveModel` and
`LoadModel` for persistence. Tests in `tests/dump_test.cc`, `tests/c_api_test.cc`,
`tests/test_explain.py` and `tests/test_cli.py` independently traverse exported
trees and reconstruct actual predictions, including threshold equality, both
NaN directions, root-only/deep trees, precision, escaping, old models and errors.

## SHAP feature attribution

For background-relative feature contributions on any Python estimator or an
entire sklearn ensemble, use `model.shap_explainer(background)` or
`libtree.shap_explainer(ensemble, background)`. See the
[Python integration guide](python-integration.md) for probability/value semantics,
examples and computation budgets. Native `explain()` continues to report
the actual paths and per-tree contributions.
