# Validation evidence / 验证记录

Current local validation: 2026-10-09, Linux x86-64, GCC 14.2, Python 3.12.14.
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
Other platforms, model persistence, multiclass, ranking, sparse/GPU/distributed
training, and Kaggle leaderboard submissions were not validated.

重构有意不兼容旧头文件、命令和模型格式，需要重新训练；支持稠密 CPU 回归
和二分类。未验证其他平台、模型持久化、多分类、排序、稀疏、GPU、分布式或
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
