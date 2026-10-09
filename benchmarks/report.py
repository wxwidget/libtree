"""Render the committed raw measurements as an inspectable Markdown report."""
import json
from pathlib import Path

root = Path(__file__).parent
report = json.loads((root / "results.json").read_text())
meta = report["metadata"]
lines = [
    "# XGBoost comparison / XGBoost 对比", "",
    "[English guide](../README.md) · [中文指南](../README.zh-CN.md)", "",
    "## Reproduce / 复现", "", "```sh",
    "python -m pip install -r benchmarks/requirements.txt",
    "make all",
    "PYTHONPATH=python python benchmarks/run.py --suite original --repeats 3",
    "python benchmarks/report.py", "```", "",
    "Raw repeats, hashes and versions: [results.json](results.json).",
    "原始重复测量、数据指纹和版本见 results.json。", "",
    f"Measured / 测量时间 (UTC): {meta['date_utc']}. {meta['cpu']}.",
    f"{meta['compiler']}; Python {meta['python']}; XGBoost {meta['xgboost']}.", "",
    "## Method / 方法", "",
    "Five datasets, fixed 75/25 train/test split (seed 42); binary splits are stratified.",
    "Categorical encoding is fitted on training rows only; unknown categories are ignored.",
    "NaNs remain missing for both learners. There is no tuning on the test set.",
    "100 trees, depth 4, 64 bins, learning rate 0.1, L2 1; one CPU thread; no GPU.",
    "LibTree uses minimum leaf count 5; XGBoost uses its default minimum Hessian weight 1.",
    "Their sketches, missing routing, base estimates and leaf constraints differ:",
    "this compares matched main controls, not mathematically identical learners.", "",
    "每个数据集固定 75/25 划分、seed=42，二分类分层抽样；编码器只拟合训练集。",
    "两种算法保留 NaN，不用测试集调参。主参数相同，但分位数算法、叶约束等不同。", "",
    "Each timing repeat runs in a fresh process; table shows the median of three repeats.",
    "Native XGBoost calls its public C API from C++; no Python bridge is timed there.",
    "Fit time includes binning/DMatrix creation and label transfer; imports, CSV loading,",
    "preprocessing and subprocess startup are excluded. Prediction includes each interface's",
    "required input validation/conversion: native XGBoost builds a test DMatrix; Python uses",
    "its estimator prediction path. These are end-to-end interface timings, not isolated tree traversal.",
    "RSS is whole-process peak memory in MiB, not incremental model allocation. Python workers",
    "import both engines, so Python RSS includes the same substantial runtime baseline.", "",
    "每次测量启动独立进程，取三次中位数。C++ 通过 XGBoost 的公开 C API 直接调用。",
    "训练计时包含分箱/矩阵构建，不包含导入、文件读取、预处理和进程启动。预测计时",
    "包含接口必要的转换；C++ XGBoost 构造测试 DMatrix，Python 使用 estimator 路径。",
    "内存是整个进程的峰值，不是单个模型的增量；Python 进程均导入两种库。", "",
    "Quality uses held-out RMSE (lower is better) or ROC AUC (higher is better).",
    "All runs must beat a mean-label/AUC=0.5 baseline and emit finite predictions.",
    "C++ and Python predictions of each engine are checked with rtol=2e-5, atol=2e-3.",
    "Repeated predictions are also checked for determinism.", "",
    "效果采用留出集 RMSE（越低越好）或 AUC（越高越好），检查优于简单基线、",
    "预测有限、重复运行确定性及同一引擎的 C++/Python 一致性。", "",
    "## Results / 结果", "",
    "Historical RSS used getrusage and may include inherited pre-exec high-water. Use the [new Kaggle report](KAGGLE_REPORT.zh-CN.md) for corrected Linux VmHWM measurements.",
    "此历史报告的 RSS 可能受启动前继承的内存高水位影响；内存比较以新 Kaggle 报告为准。", "",
    "| Dataset | Interface | Engine | Metric | Value | Fit ms | Predict ms | Peak RSS MiB |",
    "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
]
for r in report["results"]:
    lines.append(f"| {r['dataset']} | {r['language']} | {r['engine']} | {r['metric_name']} | "
                 f"{r['metric']:.4f} | {r['fit_seconds']*1000:.2f} | "
                 f"{r['predict_seconds']*1000:.3f} | {r['peak_rss_kib']/1024:.1f} |")
lines += ["", "## Sources and limits / 数据来源与限制", "",
    "Titanic uses the public Kaggle training data mirrored in Géron's teaching repository.",
    "Medical insurance uses the public insurance dataset mirrored by Machine Learning with R",
    "(also used in Kaggle medical-cost exercises); it is not an official competition score.",
    "Both downloads use HTTPS and pinned SHA256. Diabetes and breast cancer are sklearn",
    "bundled datasets; Friedman is synthetic (20,000 rows, 20 features, noise 1, seed 42).", "",
    "Titanic 来自公开 Kaggle 训练集镜像，保险费用来自公开教学镜像（也用于 Kaggle",
    "练习），均校验 SHA256；其他为 sklearn 内置或合成数据。没有 Kaggle 榜单提交。", "",
    "These are small/medium local workloads and one split, not a universal ranking.",
    "LibTree quality is close to XGBoost here, with a higher Titanic AUC on this split.",
    "XGBoost is faster on the larger Friedman workload. Historical RSS is not suitable",
    "for engine comparisons; see the corrected Kaggle report. Small timings and shared-cloud CPU",
    "scheduling are noisy; examine raw repeats. No hyperparameter search, cross-validation,",
    "GPU, sparse, multiclass or multi-thread comparison was run.", "",
    "效果相近，Titanic 的这个划分上 LibTree AUC 较高；较大合成任务上 XGBoost 更快。",
    "内存比较请查看修正后的新报告。小样本毫秒级计时容易受共享 CPU 调度影响，",
    "只有一个划分，未进行交叉验证、超参搜索、GPU、稀疏、多分类或多线程比较。", "",
]
(root / "README.md").write_text("\n".join(lines))
