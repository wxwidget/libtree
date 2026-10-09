# Algorithm, performance and contribution guide

[中文](design.zh-CN.md)

## What boosting learns

The model predicts a margin `F(x) = base + eta * sum(tree(x))`. For regression,
base is the training-label mean. For binary classification it is the log odds
of the mean label (clamped to `[1e-6, 1-1e-6]`); prediction is `sigmoid(F)`.
Each round fits a tree to first- and second-order loss derivatives:

| Objective | Gradient g | Hessian h | Reported training loss |
| --- | --- | --- | --- |
| Half squared error | F − y | 1 | Mean squared error |
| Logistic cross entropy | sigmoid(F) − y | p(1 − p), floor 1e-6 | Mean cross entropy |

In a leaf, `G = sum(g)`, `H = sum(h)`, and its value is `−G/(H + lambda)`.
A candidate partition has gain
`0.5 * (G_L²/(H_L+lambda) + G_R²/(H_R+lambda) − G²/(H+lambda))`.
Accept only gains greater than `min_gain + 1e-12`, and only when both children
meet the sample minimum. The implementation uses deterministic feature/bin
order for ties. `max_depth` counts edges from the root. Shrinkage is applied
once when updating margins and once per tree during prediction.

A training-only quantile sketch is computed by sorting each column once.
The sketch here is a deterministic quantile sample of the complete sorted
column, not XGBoost's weighted approximate sketch. Duplicate cuts collapse.
Every feature value maps to a uint16 bin; bin zero is reserved for NaN. At a
node, scan row IDs into gradient/Hessian/count histograms. Prefix sums evaluate
all cuts; each cut tries missing rows on either side. The final observed-value
cut allows a missing-only partition, even for constant observed values. When
there are no missing rows, skip the redundant second direction. No split is
possible on an entirely missing feature.

## Performance rules

* Separate one-time preparation from rounds: sort/quantize once, reuse bins.
* Quantized columns are contiguous uint16 arrays (2 bytes/value). Raw input is
  a borrowed row-major float32 matrix, not per-instance heap objects.
* Keep gradient/Hessian accumulation in double. Stable sigmoid and softplus
  prevent exponential overflow and log-of-zero; thresholds use observed cuts
  rather than overflowing float midpoint arithmetic.
* Partition one row-index vector in place. Build the smaller child's histogram;
  subtract it from the parent for the larger child. Reuse a depth-sized workspace.
* For low-cardinality columns, skip their most frequent bin and recover it from
  node totals. Two-valued columns use masked weighted sums instead of scattered
  writes. Four accumulators break serial dependencies. A single populated parent bin
  stays constant in either child and needs no new scan.
* For wide, mostly continuous matrices, scan compact row-major histogram indices
  and reuse each derivative across features. Narrow/categorical matrices retain
  column scans; histograms larger than 65,536 entries use the column fallback.
* Assign bin IDs with a padded, fixed-step lookup; the actual quantile boundaries
  and NaN bin remain unchanged. Update margins at leaves, avoiding a second
  traversal of the new tree. Reuse the loss exponential for the next derivatives.
* Store each tree in one node vector with integer child indices. Recursive
  construction never keeps references across vector growth; vectors own storage.
* Compile shallow trees to an implicit complete-tree prediction layout: 8-byte
  splits, contiguous weighted leaves and packed missing directions. Early leaves
  repeat their value through padded descendants. Expanded array payload is
  bounded by original node storage, with fixed vector metadata per tree; depths over 8 or thin trees use ordinary traversal.
* Predict 64 rows together to expose independent work and keep each tree hot.
  Preserve double accumulation and tree order. `PredictInto` and the C ABI write
  directly into caller-owned outputs, removing the allocation/copy in Python.
* Reserve bounded tree storage; reuse valid build output. No fast-math, disabled
  checks, architecture-specific instruction requirement, or global random state.
* Keep the core independent of Python, XGBoost and OpenMP. This implementation is
  deliberately single-threaded, enabling an interpretable one-thread baseline.

Preparation is `O(d n log n)` for sorting. Each round costs approximately
`O(d n depth + d B nodes)` for histogram construction and scanning, plus margin
updates. Inference is `O(n_test * trees * depth)`. Temporary training memory is
`O(n d + n + d B * depth)` plus the model. The original dense matrix is borrowed;
column bins use two bytes/value, and the wide-matrix path additionally stores
2-byte row-major indices. Sparse exception row IDs and depth histogram buffers
are also owned scratch storage, not copied raw feature matrices. Worst-case node count
per tree is `min(2n−1, 2^(depth+1)−1)`; set sensible depth/rounds for available
memory. This is not an out-of-core learner.

Optimization decisions come from measured workloads. Preserve the independent
stump oracle, serialized-model traversal oracle, exact before/after predictions,
missing-value behavior and both interfaces. Benchmark a Release build and record
quality with runtime. Rankings apply only to the recorded datasets, parameters,
hardware and thread settings; they do not establish universal superiority.

## Google style and explicit deviations

Formatting uses the checked-in `BasedOnStyle: Google` configuration. The code
uses namespace-scoped APIs, named structs and enum classes, self-contained
headers with guards, `UpperCamelCase` C++ types/methods, `snake_case_` members,
const views, explicit ownership, and warnings as errors. Python uses PEP 8 naming.

Google's internal C++ guide forbids exceptions. This public library deliberately
uses standard exceptions for invalid C++ inputs and strong Fit rollback; its
C ABI catches exceptions so none crosses into Python/C. This is an explicit
style deviation, not a claim of complete Google-internal compliance. The
benchmark's `dlopen`/function-pointer casts are a Linux-only interoperability
boundary. Core data structures use RAII and avoid owning raw pointers; the C ABI
is the documented exception with an opaque handle and matching release function.

## TDD and validation

1. Specify behavior in `tests/gbdt_test.cc` or `tests/test_python.py`. Reproduce
   the failure before implementing. The initial native stub failed its first
   not-fitted check; the implemented suite then passed.
2. Implement the smallest change, run the targeted test, then `make test`.
3. Run `make sanitize` for memory/undefined behavior changes. Native tests include
   100 C-ABI allocation/release cycles and randomized dense training. Python
   tests alternate explicit release/finalization across 300 cycles and check an
   RSS plateau (16 MiB tolerance). RSS is a coarse guard, not a substitute for LSan.
4. Run `make coverage`: source-only C++ line coverage has a 90% gate. Python
   behavior is instrumented separately. Generic exception handlers can produce
   duplicated gcov template lines; the report includes them, not just the core.
5. Run comparisons after algorithm or performance changes. Preserve raw repeats,
   dataset hashes, tool versions, CPU, thread settings, quality and baseline.

The exhaustive stump oracle enumerates all legal partitions and directly sums
squared residuals; it does not reuse the learner's gain formula. A 10k-row
performance guard checks fit quality on the generated training fixture
and fit time below 30 seconds; the wide limit detects catastrophic regressions,
not subtle speed differences. Actual benchmarks use unseen test rows.

ABI callers must supply valid handles, capacities and buffer lifetimes; arbitrary
pointers cannot be validated in C++. Prediction output must not overlap features.
Fit and allocating `Predict` can fail allocation; valid `PredictInto` calls use
stack scratch storage. A failed native Fit retains a previous fitted model. C ABI
errors return −1 or a null handle; read `LtLastError()` immediately. Do not use
a freed handle. Python owns handles with `weakref.finalize` and idempotent close.
