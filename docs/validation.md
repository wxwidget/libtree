# Validation evidence / 验证记录

Latest risk extension: [risk validation/report](../benchmarks/RISK_REPORT.md).
Native source lines 2444/2444 and Python statements 371/371 are covered (100%).
Hellinger, explicit boosting, balanced weights, KS ties/direction, CV encoder
isolation and cache-integrity checks pass, along with Release and sanitizers.

Latest C5 implementation validation and its coverage scope are recorded in
[C5 measurements and validation](../benchmarks/C5_REPORT.md). Both native
instrumented source lines (2434/2434) and Python statements (359/359) reach
100%, with Release, ASan/UBSan/LSan and TSan checks passing. The dated sections
below are historical records. C5 adds no coverage exclusions. Installed-wheel
Unicode rules/explanations and both guide code examples were also executed.

Current local validation: 2026-10-09, Linux x86-64, GCC 14.2, Python 3.12.14.
The initial migration evidence below is historical; the latest CLI and three-engine
validation is recorded at the end of this document.
These are local executions; the GitHub Actions workflow is configured but has
not been observed on GitHub during local validation; inspect Actions after pushing.

| Check / 检查 | Result / 结果 |
| --- | --- |
| Native behavioral checks | 1,022 assertions passed; 30 independent exhaustive stump oracle trials |
| C ABI ownership/errors | 100 allocation/free cycles and error-path checks passed |
| C++ example | Passed, verifies predictions numerically |
| Python behavior | 18 pytest cases passed |
| Performance guard | 10k rows / 50 trees, quality + 30-second ceiling passed |
| Python ownership guard | 300 explicit-close/finalization cycles, RSS plateau passed |
| ASan + UBSan + LSan | All 3 native targets passed with halt-on-error and leak detection enabled |
| Native source line coverage | 275/291 = 94.5%; core 195/196 = 99.5%; gate 90% |
| Native source branch coverage | 257/328 = 78.4%; core 190/226 = 84.1% |
| Native source function coverage | 29/29 = 100% |
| Python line coverage | 128/134, rounded report 96% |
| Google clang-format | Dry-run with warnings as errors passed |
| Native wheel | Built and installed outside checkout; prediction smoke check passed |
| XGBoost comparisons | 5 datasets × 2 languages × 2 engines × 3 isolated repeats; quality baselines and prediction parity passed |

Reports: `build-coverage/coverage.html`, `build-coverage/summary.json` and
[committed raw benchmark measurements](../benchmarks/results.json).
Generated HTML/build outputs are ignored, not source files. Coverage filters
`src/`; it does not count tests or benchmark orchestration toward the gate.
Exception-only paths (including unknown exceptions) are not all exercised;
branch coverage is deliberately reported separately from line coverage.

检查针对本地实际执行，不表示 GitHub CI 已运行。覆盖率过滤源码 `src/`，不使用
测试代码充数。尚未穷尽异常路径，分支覆盖率与行覆盖率分开展示。RSS 是粗略
增长检查；精确的原生内存诊断来自 ASan/LSan。这些测试不证明所有输入都正确。

The migration is intentionally breaking: legacy C++ headers, CLI flags and
binary model files are removed. Both language interfaces now use one C++17 core.
The supported workflow is dense regression/binary classification on CPU.
Other platforms, Python model persistence, multiclass, ranking, sparse/GPU/distributed
training, and Kaggle leaderboard submissions were not validated.

重构有意不兼容旧头文件、命令和模型格式，需要重新训练；支持稠密 CPU 回归
和二分类。未验证其他平台、Python 模型持久化、多分类、排序、稀疏、GPU、分布式或
Kaggle 榜单提交。Google 格式/命名已检查，标准异常是明确的风格例外。

## Additional Kaggle-data validation / 新增 Kaggle 数据验证

Five public Kaggle-data mirrors, seeds 42/2024/2026, both native/Python APIs and
both engines, three isolated timing repeats: 180 final successful runs. Four
benchmark preparation/memory-counter tests passed. Primary and additional
metrics, split variation, raw repeats and source hashes are recorded in the
[new report](../benchmarks/KAGGLE_REPORT.zh-CN.md).

The historical benchmark RSS used getrusage, which can include pre-exec parent
high-water usage. The new measurements use Linux /proc/self/status VmHWM for
the current executable's address space; the old memory comparisons are superseded.
Algorithm core code did not change. Kaggle API access returned proxy tunnel 403;
public mirrors were downloaded via HTTPS and verified with pinned SHA256.

新增五数据集多划分基准及四个基准准备测试均通过。旧 RSS 可能继承父进程
高水位，内存比较以新报告的 VmHWM 实测为准；算法核心未修改。数据通过
公开镜像获取，不是 Kaggle 榜单提交。

## HTML report / HTML 图表报告

Self-contained offline report: benchmarks/KAGGLE_REPORT.html. Built only from
verified kaggle-results.json and kaggle-summary.json; rechecked aggregate means,
standard deviations and timing medians against the raw records. Plotly is embedded,
so chart rendering needs no CDN. Browser validation used installed Chromium:
7 charts rendered, C++/Python switching, logarithmic training axis, dataset
filtering, JSON export, 180-row timing CSV export and a 390px mobile layout all
passed without JavaScript errors. Desktop/mobile screenshots are retained under
benchmarks/report-assets/.

## Parameterized CLI and three engines / 参数化命令与三方比较

TDD: CLI contract tests were added first and failed because `xgbt` did not exist.
The C++17 command now supports numeric CSV training, versioned model saving,
loading and standalone prediction. C++ checks verify lossless model round trips
and rollback on failed loads. Eleven CLI tests cover missing values, invalid
parameters/CSV, model cycles, shared children, invalid feature indices, bounds,
trailing data, output preservation and input/output path aliases.

| Latest check | Result |
| --- | --- |
| C++ behavioral checks | 1,031 checks passed |
| CTest | 4/4 targets passed |
| Python behavior + CLI | 29/29 pytest cases passed (18 + 11) |
| Performance and ownership guards | 2/2 passed |
| ASan + UBSan + LSan | 4/4 native targets and all 11 CLI cases passed |
| Native source line coverage | 532/559 = 95.2%; core 99.5%, model 100%, CLI 94.1% |
| Native source function coverage | 41/41 = 100% |
| Native source branch coverage | 726/1,142 = 63.6%; core 84.1% |
| Python line coverage | 128/134 = 95.5% (rounded display 96%) |
| Google clang-format | Warnings-as-errors dry run passed |
| CMake installation | Installed executable and train/save/load/predict example passed |
| Native Python wheel | Rebuilt, installed and prediction-tested outside the checkout |
| Benchmark preparation tests | 4/4 passed |
| Non-default benchmark parameters | 30 runs; 3 engines × both interfaces × 5 datasets; prediction parity passed |
| Formal comparison | 270 runs; 5 datasets × 3 splits × both interfaces × 3 engines × 3 repeats |

The formal run uses XGBoost 3.4.1 and LightGBM 4.7.0. All quality baselines,
finite predictions, repeat determinism and C++/Python parity checks passed.
Every recorded source SHA256 matches the committed implementation. Timing,
process peak memory, quality metrics and comparability limitations are in the
[three-engine report](../benchmarks/COMPARISON_REPORT.zh-CN.md).

The offline [HTML report](../benchmarks/KAGGLE_REPORT.html) now uses
comparison-results.json and comparison-summary.json. Aggregate means, sample
standard deviations and training/prediction/memory medians are rechecked
against the raw evidence before rendering. Plotly receives independent layout
objects to prevent chart updates from mutating shared axis configuration.
Browser checks verify all seven charts, three-engine traces, numeric axes after
interface/log-scale changes, filtering, 270-row CSV and JSON downloads, and a
390px mobile viewport. No JavaScript errors were observed.

Coverage and sanitizer diagnostics are local evidence, not a proof of all
possible inputs or a claim that remote GitHub Actions has run.

## Latest performance optimization / 最新性能优化

The earlier sections describe their original implementation snapshots. Current
source evidence: 4,642 native behavioral checks; 4/4 CTest targets; 18 Python,
11 CLI, 2 performance/ownership and 4 benchmark preparation cases passed.
ASan/UBSan/LSan passed all native targets and all 11 CLI cases. Source line
coverage is 763/791 = 96.5% (core 427/429 = 99.5%, model 100%); functions
51/51 = 100%; branches 931/1,374 = 67.8% (core 86.0%). Python is 128/134.

TDD started with the new direct-output prediction contract failing to compile.
The serialized-model traversal oracle now verifies ordinary predictions against
the optimized representation for regression, classification, NaN, root-only,
uneven batches, shallow padding and deep-tree fallback. Tests cover bins
2/3/17/64/65,535, including 65,535 distinct finite values and the unseen-NaN
route, plus both wide-matrix histogram strategies and their storage limit.

The final five-dataset comparison has 450 runs: three seeds, both interfaces,
three engines, five timing repeats. The raw-data ranking validator passed
20/20 pooled median groups (10 training, 10 inference), checked every recorded
source hash, and verified unchanged parameters, prepared splits and every
same-split LibTree quality metric against the prior 937e6b4 measurements.
An earlier candidate correctly failed the rank gate at 19/20 and was improved;
its intermediate outputs are retained only in the ignored runs directory.

Thirty additional runs on 20k-row Friedman1 passed both-interface parity,
quality and determinism checks; all four timing groups were fastest. This is
a single-seed supplemental test, separate from the main split-variation study.
Randomized-order paired native measurements (9 repeats per version/input,
108 runs on 6 inputs) verify float32 predictions are exactly equal to the
pre-optimization executable. Baseline source SHA256 matches commit 937e6b4.

The [optimization report](../benchmarks/OPTIMIZATION_REPORT.zh-CN.md) preserves
all final results and reproduction commands. The offline HTML includes seven
charts, before/after and supplemental tables, all raw evidence, and separate
JSON downloads. Browser validation covers numeric axes after interface/log
switching, filtering, a 450-row timing CSV and the mobile layout. Rankings are
observations on this hardware, data and parameters; they do not establish a
universal ranking or a statistically significant quality advantage.

## Parallel execution validation / 多线程验证

TDD first failed compilation because `num_threads` and `SetNumThreads` did not
exist. Tests now compare serialized model bytes, every prediction and every
round's loss across 1/2/4 threads for regression and binary classification.
Wide continuous and mostly low-cardinality data, NaNs, empty batches, uneven
64-row batches, old model loading, invalid thread counts and failed refits are
covered. Invalid parallel prediction preserves caller output; the executor is
usable after exceptions from the caller or any worker. Concurrent callers share
one bounded pool. Fifteen Python close/refit cycles return the thread count to
its initial value; native sanitizer tests also exercise joined worker lifetimes.

Latest local results: 4,701 native checks; 5/5 CTest targets; 31 Python, 13 CLI,
2 performance/ownership and 4 benchmark preparation tests. ASan/UBSan/LSan pass
all native targets and all CLI cases; ThreadSanitizer passes all 5 native targets.
Gcovr source line coverage is 1116/1116 = 100% (core, model and thread executor
also 100%); functions are 106/106 and branches are 1116/1660 = 67.2%.
Python coverage is 148/148 = 100%. Coverage counters use atomic updates and fresh
gcda files. The only excluded lines are the arbitrary C++ exception boundary,
system thread-creation exhaustion, a timestamp collision race, and a compiler
mapping continuation line; each is documented beside its `GCOVR_EXCL` marker.
Google clang-format passes.

The installed CLI and wheel were tested outside the checkout with multiple
thread counts. Source distribution includes the new internal executor header;
package builds use `-pthread`, while CMake exposes `Threads::Threads` transitively.
The version-1 portable model omits runtime thread counts and the legacy C create
signature remains supported. CI now includes ThreadSanitizer, but its remote
execution has not been observed locally.

Formal [parallel evidence](../benchmarks/PARALLEL_REPORT.zh-CN.md) compares seven
datasets, three seeds, both interfaces, all engines, three timing repeats and
thread budgets 1/2/4: 1,134 measurements. Three cgroup-constrained CPU budgets are
measured with sequential processes; the available affinity is 5 CPUs and quota
is 4 CPU-seconds/second. Ranking validates complete groups, finite measurements,
equal parameters and prepared input hashes, current source hashes, unchanged
LibTree metrics across threads and unchanged five-dataset metrics from the prior
optimized snapshot. The first histogram strategy was improved after initial
scaling measurements; intermediate incomplete runs remain ignored. The new HTML
keeps negative speedups, timing ranges, quality, memory and all raw measurements.
Shared-host timing noise, serial portions and different engine algorithms limit
scaling and rankings; the old single-thread HTML remains a historical snapshot.

Chromium validation of the new offline report passed: all five charts rendered,
both interfaces and all three thread budgets displayed the raw-data timings,
scaling dataset selection and numeric/log axes remained correct, and the 1,134-row
CSV and complete JSON downloads matched the evidence. The 390px mobile layout
had no horizontal overflow; no JavaScript errors were observed.

## Tree inspection and single-row explanations / 决策树输出与解释

TDD first produced missing-method compilation errors in the C++ inspection
test, Python `AttributeError` failures, and an unsupported-command CLI failure.
A separate red test exposed invalid JSON when an extreme saved-model leaf
overflowed the float32 prediction. The implementation now rejects that case
and preserves existing output files.

Python tests independently traverse exported JSON trees and reconstruct actual
regression predictions and binary probabilities. They check every path step,
threshold equality, learned left/right NaN routing, raw leaf values and scaled
contributions. Root-only trees, multi-level paths, selected-tree IDs, 1/4-thread
export parity, Unicode/control escaping and native query/write failures are
included. Native tests also load existing version-1 models, preserve stream
locale/precision, exercise undersized caller-owned C buffers and inspect one
immutable model from eight concurrent readers under ThreadSanitizer.

Local validation for this change:

| Check | Result |
| --- | --- |
| Native CTest | 6/6 targets passed, including the new `dump` target |
| Python estimator and inspection contracts | 43/43 passed |
| CLI | 16/16 passed |
| Performance and ownership checks | 2/2 passed |
| Benchmark preparation/metadata | 10/10 passed |
| ASan/UBSan/LSan | 6/6 native targets and 16/16 CLI cases passed |
| ThreadSanitizer | 6/6 native targets passed |
| Native source line coverage | 1688/1688 = 100%; no new exclusion markers |
| Python API statement coverage | 195/195 = 100% |
| Installed wheel outside checkout | Both estimator types export UTF-8 labels and explanations |
| English/Chinese Python and C++ guide examples | Executed; base 0.5 + contribution 0.25 = prediction 0.75 |

The public API, C ABI and CLI share one native serializer and path traversal.
Explanation structs and output buffers are allocated on demand; the fitted
node layout and persistence format do not store additional path caches or
training statistics. These outputs are exact paths and per-tree contributions,
not SHAP feature attribution. See [the English guide](explain.md) or
[中文指南](explain.zh-CN.md) for examples and score semantics.

## Python explanation and ensemble integration (2026-10-17)

The compatibility change covers `GBDTRegressor`, `GBDTClassifier` and
`C5Classifier`. TDD first exposed missing sklearn estimator tags and missing
C5 persistence methods. Regression tests then exercised actual process-based
Voting, Stacking and Bagging; snapshots reconstruct independent native models
without storing training matrices or refitting. Failed snapshots/loads and
failed classifier refits preserve the original owner and predictions. A
malformed-snapshot regression also rejects extreme votes before accumulation
can overflow and produce non-finite probabilities.

SHAP Exact and Permutation explain numeric predictions or all class
probabilities, including multiclass C5, NaN routes, final ensemble outputs and
a DataFrame Pipeline with named column selection. Tests check additive
reconstruction, class names, dropped-column zero contributions and explicit
background validation. The adapter uses public SHAP APIs and returns standard
`shap.Explanation` objects. Native TreeSHAP is not implemented. See the
[English integration guide](python-integration.md) or
[中文指南](python-integration.zh-CN.md) for contracts and limitations.

| Check | Local result |
| --- | --- |
| Release CTest | 8/8 passed |
| Python estimator, tree/rule inspection and ecosystem contracts | 97/97 passed |
| CLI cases | 16/16 passed |
| Performance/RSS ownership guards | 4/4 passed; 300 ownership cycles per model, including 100 snapshot round trips |
| Benchmark methodology | 20/20 passed |
| Python production statement coverage | 505/505 = 100% |
| Native source line coverage | 2790/2790 = 100%; no new exclusion markers |
| ASan/UBSan/LSan | 8/8 native targets and 16/16 CLI cases passed |
| ThreadSanitizer | 8/8 native targets passed |
| sklearn 1.9.1 / SHAP 0.54.1 / NumPy 2.5.4 | 20/20 integration cases passed |
| sklearn 1.4.2 / SHAP 0.46.0 / NumPy 1.26.4 | 20/20 integration cases passed; three upstream deprecation warnings |
| Installed platform wheel, outside checkout | 20/20 integration cases passed; three models also fit/predict/pickle with sklearn and SHAP imports blocked |
| English and Chinese guide Python examples | Both executed successfully |
| clang-format and Git whitespace checks | Passed |

The existing Linux CI job now runs integration tests and enforces their
coverage. A second CI job installs the built package with the earlier dependency
combination and exercises the same integration suite. Local success does not
by itself establish remote CI success.

GBDT persistence writes version 2 to retain training-loss history and continues
to read version-1 models. C5 persistence retains trained parameters, pruned
storage, votes, learner errors and node statistics; rules are reconstructed.
The original tree/rule inspection JSON schemas and learning algorithms are
unchanged. This is an interoperability validation, not a new comparative
quality/speed study; benchmark reports retain their original revision labels.
