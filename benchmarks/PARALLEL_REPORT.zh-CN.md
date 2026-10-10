# LibTree CPU 多线程实测

3 个线程数 × 4 个数据集 × 3 个划分 × C++/Python × 3 个引擎 × 3 次计时，共 **648 次**。

Equal configured thread budgets; independent processes. Each cell uses the arithmetic mean of 9 runs. All LibTree same-split quality metrics are unchanged across configured thread counts.

训练与预测分开测量。每组 9 次原始计时，时间报告算术平均值±样本标准差并保留范围；效果为三个划分的均值及样本标准差。

可用逻辑 CPU：5；cgroup CPU 配额：400000 100000；种子：[42, 2024, 2026]。线程数为预算，小任务可串行。

| 数据集 | 接口 | 引擎 | 线程 | 训练 ms 平均±SD | 推理 ms 平均±SD | 训练加速 | 推理加速 | 指标 | 测试效果 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| bank_marketing | cpp | libtree | 1 | 262.92 ± 12.37 | 6.543 ± 0.635 | 1.00× | 1.00× | AUC ↑ | 0.8055 ± 0.0089 |
| bank_marketing | cpp | libtree | 2 | 209.00 ± 17.38 | 4.174 ± 1.343 | 1.26× | 1.57× | AUC ↑ | 0.8055 ± 0.0089 |
| bank_marketing | cpp | libtree | 4 | 205.65 ± 18.72 | 3.084 ± 0.812 | 1.28× | 2.12× | AUC ↑ | 0.8055 ± 0.0089 |
| bank_marketing | cpp | lightgbm | 1 | 251.86 ± 9.90 | 23.112 ± 1.800 | 1.00× | 1.00× | AUC ↑ | 0.8052 ± 0.0081 |
| bank_marketing | cpp | lightgbm | 2 | 174.21 ± 20.09 | 12.535 ± 0.817 | 1.45× | 1.84× | AUC ↑ | 0.8052 ± 0.0081 |
| bank_marketing | cpp | lightgbm | 4 | 126.80 ± 6.28 | 8.303 ± 1.473 | 1.99× | 2.78× | AUC ↑ | 0.8052 ± 0.0081 |
| bank_marketing | cpp | xgboost | 1 | 279.56 ± 10.46 | 13.895 ± 1.290 | 1.00× | 1.00× | AUC ↑ | 0.8056 ± 0.0087 |
| bank_marketing | cpp | xgboost | 2 | 278.61 ± 13.27 | 13.692 ± 0.691 | 1.00× | 1.01× | AUC ↑ | 0.8056 ± 0.0087 |
| bank_marketing | cpp | xgboost | 4 | 281.49 ± 15.57 | 13.416 ± 0.491 | 0.99× | 1.04× | AUC ↑ | 0.8056 ± 0.0087 |
| bank_marketing | python | libtree | 1 | 262.45 ± 9.42 | 7.241 ± 2.309 | 1.00× | 1.00× | AUC ↑ | 0.8055 ± 0.0089 |
| bank_marketing | python | libtree | 2 | 227.85 ± 30.51 | 4.238 ± 1.146 | 1.15× | 1.71× | AUC ↑ | 0.8055 ± 0.0089 |
| bank_marketing | python | libtree | 4 | 202.48 ± 21.45 | 2.860 ± 1.056 | 1.30× | 2.53× | AUC ↑ | 0.8055 ± 0.0089 |
| bank_marketing | python | lightgbm | 1 | 278.48 ± 16.36 | 25.629 ± 1.954 | 1.00× | 1.00× | AUC ↑ | 0.8052 ± 0.0081 |
| bank_marketing | python | lightgbm | 2 | 187.29 ± 27.98 | 14.201 ± 1.075 | 1.49× | 1.80× | AUC ↑ | 0.8052 ± 0.0081 |
| bank_marketing | python | lightgbm | 4 | 142.78 ± 20.64 | 9.144 ± 1.552 | 1.95× | 2.80× | AUC ↑ | 0.8052 ± 0.0081 |
| bank_marketing | python | xgboost | 1 | 290.64 ± 9.57 | 9.611 ± 0.582 | 1.00× | 1.00× | AUC ↑ | 0.8056 ± 0.0087 |
| bank_marketing | python | xgboost | 2 | 306.10 ± 53.76 | 9.576 ± 0.856 | 0.95× | 1.00× | AUC ↑ | 0.8056 ± 0.0087 |
| bank_marketing | python | xgboost | 4 | 337.41 ± 121.10 | 9.353 ± 0.415 | 0.86× | 1.03× | AUC ↑ | 0.8056 ± 0.0087 |
| friedman_100k | cpp | libtree | 1 | 492.11 ± 16.58 | 14.750 ± 0.981 | 1.00× | 1.00× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | cpp | libtree | 2 | 381.57 ± 36.37 | 10.417 ± 3.509 | 1.29× | 1.42× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | cpp | libtree | 4 | 325.65 ± 14.15 | 6.412 ± 0.985 | 1.51× | 2.30× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | cpp | lightgbm | 1 | 592.79 ± 30.88 | 85.393 ± 4.195 | 1.00× | 1.00× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | cpp | lightgbm | 2 | 362.89 ± 29.13 | 46.856 ± 4.010 | 1.63× | 1.82× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | cpp | lightgbm | 4 | 285.83 ± 21.31 | 24.647 ± 1.475 | 2.07× | 3.46× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | cpp | xgboost | 1 | 514.00 ± 33.39 | 24.189 ± 3.689 | 1.00× | 1.00× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | cpp | xgboost | 2 | 514.01 ± 35.55 | 22.592 ± 0.723 | 1.00× | 1.07× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | cpp | xgboost | 4 | 494.80 ± 25.88 | 22.439 ± 0.718 | 1.04× | 1.08× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | python | libtree | 1 | 517.35 ± 24.94 | 16.495 ± 3.219 | 1.00× | 1.00× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | python | libtree | 2 | 360.38 ± 18.66 | 9.002 ± 2.458 | 1.44× | 1.83× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | python | libtree | 4 | 336.63 ± 38.15 | 6.435 ± 0.976 | 1.54× | 2.56× | RMSE ↓ | 1.1163 ± 0.0049 |
| friedman_100k | python | lightgbm | 1 | 608.06 ± 30.38 | 87.178 ± 4.121 | 1.00× | 1.00× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | python | lightgbm | 2 | 380.98 ± 40.01 | 47.909 ± 2.480 | 1.60× | 1.82× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | python | lightgbm | 4 | 267.93 ± 21.10 | 26.291 ± 2.154 | 2.27× | 3.32× | RMSE ↓ | 1.1159 ± 0.0083 |
| friedman_100k | python | xgboost | 1 | 510.45 ± 25.64 | 19.199 ± 1.187 | 1.00× | 1.00× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | python | xgboost | 2 | 520.63 ± 27.10 | 20.510 ± 3.130 | 0.98× | 0.94× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_100k | python | xgboost | 4 | 529.13 ± 45.88 | 19.012 ± 0.812 | 0.96× | 1.01× | RMSE ↓ | 1.1269 ± 0.0048 |
| friedman_20k | cpp | libtree | 1 | 97.34 ± 5.76 | 2.865 ± 0.117 | 1.00× | 1.00× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | cpp | libtree | 2 | 82.76 ± 7.50 | 1.622 ± 0.393 | 1.18× | 1.77× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | cpp | libtree | 4 | 82.92 ± 9.28 | 1.217 ± 0.271 | 1.17× | 2.35× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | cpp | lightgbm | 1 | 135.97 ± 17.12 | 19.103 ± 5.057 | 1.00× | 1.00× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | cpp | lightgbm | 2 | 85.15 ± 10.08 | 8.785 ± 0.396 | 1.60× | 2.17× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | cpp | lightgbm | 4 | 65.58 ± 11.89 | 5.163 ± 0.486 | 2.07× | 3.70× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | cpp | xgboost | 1 | 129.29 ± 7.91 | 5.275 ± 0.770 | 1.00× | 1.00× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | cpp | xgboost | 2 | 127.49 ± 12.20 | 5.329 ± 1.303 | 1.01× | 0.99× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | cpp | xgboost | 4 | 125.09 ± 7.66 | 4.685 ± 0.237 | 1.03× | 1.13× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | python | libtree | 1 | 99.59 ± 5.71 | 2.985 ± 0.284 | 1.00× | 1.00× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | python | libtree | 2 | 83.99 ± 8.78 | 1.718 ± 0.395 | 1.19× | 1.74× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | python | libtree | 4 | 82.64 ± 5.84 | 1.236 ± 0.333 | 1.21× | 2.41× | RMSE ↓ | 1.1668 ± 0.0165 |
| friedman_20k | python | lightgbm | 1 | 147.84 ± 9.06 | 18.335 ± 1.163 | 1.00× | 1.00× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | python | lightgbm | 2 | 101.61 ± 15.61 | 10.344 ± 0.777 | 1.45× | 1.77× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | python | lightgbm | 4 | 87.68 ± 21.74 | 6.690 ± 0.672 | 1.69× | 2.74× | RMSE ↓ | 1.1497 ± 0.0150 |
| friedman_20k | python | xgboost | 1 | 131.91 ± 4.63 | 4.667 ± 0.350 | 1.00× | 1.00× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | python | xgboost | 2 | 130.77 ± 4.19 | 4.612 ± 0.194 | 1.01× | 1.01× | RMSE ↓ | 1.1556 ± 0.0085 |
| friedman_20k | python | xgboost | 4 | 127.39 ± 3.85 | 4.643 ± 0.231 | 1.04× | 1.01× | RMSE ↓ | 1.1556 ± 0.0085 |
| insurance | cpp | libtree | 1 | 4.04 ± 0.46 | 0.207 ± 0.047 | 1.00× | 1.00× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | cpp | libtree | 2 | 3.90 ± 0.55 | 0.186 ± 0.010 | 1.03× | 1.11× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | cpp | libtree | 4 | 4.20 ± 0.76 | 0.240 ± 0.089 | 0.96× | 0.86× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | cpp | lightgbm | 1 | 7.23 ± 0.35 | 0.924 ± 0.064 | 1.00× | 1.00× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | cpp | lightgbm | 2 | 12.75 ± 6.91 | 0.549 ± 0.046 | 0.57× | 1.68× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | cpp | lightgbm | 4 | 16.12 ± 4.42 | 0.427 ± 0.116 | 0.45× | 2.16× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | cpp | xgboost | 1 | 9.28 ± 0.74 | 0.441 ± 0.066 | 1.00× | 1.00× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | cpp | xgboost | 2 | 9.29 ± 0.81 | 0.421 ± 0.058 | 1.00× | 1.05× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | cpp | xgboost | 4 | 9.24 ± 0.63 | 0.413 ± 0.049 | 1.00× | 1.07× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | python | libtree | 1 | 4.31 ± 0.28 | 0.218 ± 0.019 | 1.00× | 1.00× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | python | libtree | 2 | 4.36 ± 0.28 | 0.216 ± 0.025 | 0.99× | 1.01× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | python | libtree | 4 | 4.38 ± 0.29 | 0.206 ± 0.006 | 0.99× | 1.06× | RMSE ↓ | 4642.0184 ± 148.9018 |
| insurance | python | lightgbm | 1 | 18.02 ± 0.95 | 1.547 ± 0.104 | 1.00× | 1.00× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | python | lightgbm | 2 | 25.24 ± 5.97 | 1.480 ± 0.787 | 0.71× | 1.05× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | python | lightgbm | 4 | 29.90 ± 7.29 | 1.086 ± 0.123 | 0.60× | 1.42× | RMSE ↓ | 4640.7890 ± 191.6937 |
| insurance | python | xgboost | 1 | 16.01 ± 2.40 | 1.186 ± 0.144 | 1.00× | 1.00× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | python | xgboost | 2 | 15.12 ± 1.35 | 1.255 ± 0.207 | 1.06× | 0.95× | RMSE ↓ | 4631.3675 ± 115.5969 |
| insurance | python | xgboost | 4 | 14.58 ± 1.09 | 1.237 ± 0.169 | 1.10× | 0.96× | RMSE ↓ | 4631.3675 ± 115.5969 |

## 排名与限制

1 线程：LibTree 训练 6/8 组最快，推理 8/8 组最快。
2 线程：LibTree 训练 5/8 组最快，推理 8/8 组最快。
4 线程：LibTree 训练 3/8 组最快，推理 8/8 组最快。

加速比为同版本单线程平均耗时除以当前线程平均耗时；原始计时按数据集、种子和线程数固定随机排序，不是逐轮配对实验。共享云 CPU 有波动，不能将差异解释为普遍最优。小任务可能因启动和同步变慢，全部结果均保留。

主参数：100 棵树、深度 4、64 分箱、学习率 0.1、L2=1、最小增益 0。LibTree/LightGBM 最小叶样本 5；XGBoost 默认最小 Hessian 权重 1；LightGBM leaf-wise、16 叶。各引擎分箱、生长方式并不完全相同，无超参数搜索。

训练包含分箱/DMatrix/标签传递及 LibTree 冷线程池准备；排除数据下载、CSV、预处理、导入和进程启动。预测计时包含接口必要校验和转换，C++ XGBoost 包含测试 DMatrix 构建；RSS 为执行进程 VmHWM。

75/25 留出，二分类分层；训练集拟合编码，保留 NaN，红酒先去重。Kaggle 对应数据来自固定 SHA256 的公开镜像；Friedman 为合成压力测试。未覆盖百万行、稀疏、多分类、GPU、跨硬件任务。

训练、质量和重复确定性检查均由 benchmark runner 执行；仓库级单元测试、覆盖率与 sanitizer 命令见根目录 Makefile。

## 复现

```sh
. /workspace/libtree-venv/bin/activate
make all
PYTHONPATH=python python benchmarks/parallel.py --datasets insurance bank_marketing friedman_20k friedman_100k --threads 1 2 4 --seeds 42 2024 2026 --repeats 3
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/parallel_report.py
make test
make sanitize
make thread-sanitize
make coverage
```

[交互 HTML](PARALLEL_REPORT.html) · [全部原始计时](parallel-results.json) · [聚合数据](parallel-summary.json) · [并行原理](../docs/design.zh-CN.md#确定性并行)
