# LibTree 训练与推理优化：三方实测

测试时间（北京时间）：2026-10-09 20:53。

完成 **5 个数据集 × 3 个不同划分 × 2 个接口 × 3 个引擎 × 5 次计时 = 450 次实测**。
libtree、xgboost、lightgbm 使用相同训练/测试数据；分别通过 C++ 原生调用和 Python 接口执行。
每次运行的预测有限、优于简单基线且重复运行确定；同一引擎的 C++/Python 预测一致性检查通过。

Kaggle 官方 API 被本环境网络代理拒绝（连接隧道 HTTP 403），因此从公开镜像下载对应数据。
校验了镜像文件 SHA256；未独立核验其与 Kaggle 官方压缩包逐字节相同。没有登录 Kaggle 或提交榜单。

## 数据集与处理

| 数据集 | 清理后样本 | 编码后特征 | 任务 | 特征缺失值数 | 数据处理 |
| --- | ---: | ---: | --- | ---: | --- |
| titanic | 891 | 11 | 二分类 | 179 | 只用 7 个常用特征；删除 ID、姓名等 |
| insurance | 1338 | 11 | 回归 | 0 | 6 个输入特征；费用原值回归 |
| pima | 768 | 8 | 二分类 | 652 | 5 项生理指标的无效零值转 NaN；怀孕次数的 0 保留 |
| telco | 7043 | 45 | 二分类 | 11 | 删除 customerID；11 个空白 TotalCharges 转 NaN |
| wine | 1359 | 11 | 回归 | 0 | 先删除 240 条完全重复记录，再划分；quality 回归 |

每个划分为 75% 训练、25% 测试，随机种子：42, 2024, 2026。二分类采用分层抽样。
类别编码仅在训练集拟合，忽略测试集新类别；数值缺失保留为 NaN，不学习全数据填充值。
未使用测试集选择参数。红酒去重后没有相同输入特征的重复行。

## 效果：跨三个划分的均值 ± 样本标准差

AUC 越高越好；RMSE 越低越好。C++ 与 Python 的效果一致，因此本表列 C++ 汇总，原始报告保留所有接口。

| 数据集 | 指标 | libtree | xgboost | lightgbm |
| --- | --- | ---: | ---: | ---: |
| titanic | roc_auc | 0.8393 ± 0.0144 | 0.8325 ± 0.0145 | 0.8337 ± 0.0143 |
| insurance | rmse | 4642.0184 ± 148.9018 | 4631.3675 ± 115.5969 | 4640.7890 ± 191.6937 |
| pima | roc_auc | 0.8074 ± 0.0390 | 0.8047 ± 0.0338 | 0.8030 ± 0.0426 |
| telco | roc_auc | 0.8465 ± 0.0015 | 0.8465 ± 0.0019 | 0.8444 ± 0.0002 |
| wine | rmse | 0.6375 ± 0.0116 | 0.6481 ± 0.0236 | 0.6389 ± 0.0115 |

二分类的附加指标：阈值为 0.5 的准确率、PR AUC（average precision）、log loss。

| 数据集 | 引擎 | Accuracy | PR AUC | Log loss |
| --- | --- | ---: | ---: | ---: |
| titanic | libtree | 0.7982 ± 0.0250 | 0.7972 ± 0.0094 | 0.4929 ± 0.0306 |
| titanic | xgboost | 0.8072 ± 0.0135 | 0.7874 ± 0.0214 | 0.4905 ± 0.0213 |
| titanic | lightgbm | 0.8102 ± 0.0187 | 0.7934 ± 0.0217 | 0.4913 ± 0.0221 |
| pima | libtree | 0.7344 ± 0.0463 | 0.6716 ± 0.0452 | 0.5494 ± 0.0712 |
| pima | xgboost | 0.7517 ± 0.0318 | 0.6882 ± 0.0554 | 0.5571 ± 0.0739 |
| pima | lightgbm | 0.7309 ± 0.0398 | 0.6815 ± 0.0567 | 0.5600 ± 0.0755 |
| telco | libtree | 0.7995 ± 0.0046 | 0.6576 ± 0.0091 | 0.4150 ± 0.0031 |
| telco | xgboost | 0.7982 ± 0.0047 | 0.6591 ± 0.0086 | 0.4150 ± 0.0029 |
| telco | lightgbm | 0.7986 ± 0.0046 | 0.6563 ± 0.0080 | 0.4171 ± 0.0012 |

回归附加指标（MAE 越低越好，R² 越高越好）：

| 数据集 | 引擎 | MAE | R² |
| --- | --- | ---: | ---: |
| insurance | libtree | 2542.2562 ± 21.6609 | 0.8566 ± 0.0156 |
| insurance | xgboost | 2548.8425 ± 24.7676 | 0.8574 ± 0.0135 |
| insurance | lightgbm | 2529.6383 ± 18.0922 | 0.8566 ± 0.0183 |
| wine | libtree | 0.5012 ± 0.0139 | 0.3999 ± 0.0378 |
| wine | xgboost | 0.5033 ± 0.0163 | 0.3787 ± 0.0650 |
| wine | lightgbm | 0.4980 ± 0.0120 | 0.3974 ± 0.0319 |

## 性能与内存

每个组合的时间和内存取 3 个划分、每划分 5 次，共 15 次运行的中位数。每次启动独立进程。内存采用执行后地址空间的 Linux VmHWM，避免继承父进程的高水位。

| 数据集 | 接口 | 引擎 | 训练 ms | 预测 ms | 峰值 RSS MiB |
| --- | --- | --- | ---: | ---: | ---: |
| titanic | cpp | libtree | 3.98 | 0.151 | 2.1 |
| titanic | cpp | xgboost | 9.46 | 0.383 | 7.3 |
| titanic | cpp | lightgbm | 7.74 | 0.627 | 7.9 |
| titanic | python | libtree | 4.50 | 0.240 | 127.5 |
| titanic | python | xgboost | 15.62 | 1.166 | 130.8 |
| titanic | python | lightgbm | 20.31 | 1.451 | 133.1 |
| insurance | cpp | libtree | 4.26 | 0.180 | 2.1 |
| insurance | cpp | xgboost | 9.64 | 0.424 | 7.4 |
| insurance | cpp | lightgbm | 7.69 | 0.912 | 9.3 |
| insurance | python | libtree | 4.76 | 0.239 | 127.5 |
| insurance | python | xgboost | 15.31 | 1.215 | 130.8 |
| insurance | python | lightgbm | 18.96 | 1.576 | 134.3 |
| pima | cpp | libtree | 5.36 | 0.163 | 2.1 |
| pima | cpp | xgboost | 15.57 | 0.427 | 7.3 |
| pima | cpp | lightgbm | 10.57 | 0.669 | 9.3 |
| pima | python | libtree | 5.63 | 0.239 | 127.4 |
| pima | python | xgboost | 22.18 | 1.271 | 130.9 |
| pima | python | lightgbm | 22.41 | 1.367 | 134.6 |
| telco | cpp | libtree | 49.66 | 1.054 | 4.5 |
| telco | cpp | xgboost | 53.12 | 2.193 | 12.2 |
| telco | cpp | lightgbm | 60.00 | 5.201 | 10.4 |
| telco | python | libtree | 48.27 | 1.179 | 130.1 |
| telco | python | xgboost | 63.90 | 2.699 | 133.5 |
| telco | python | lightgbm | 74.66 | 6.570 | 134.8 |
| wine | cpp | libtree | 7.33 | 0.182 | 2.1 |
| wine | cpp | xgboost | 20.27 | 0.430 | 7.4 |
| wine | cpp | lightgbm | 12.07 | 1.014 | 9.5 |
| wine | python | libtree | 7.78 | 0.242 | 127.5 |
| wine | python | xgboost | 26.22 | 1.261 | 131.0 |
| wine | python | lightgbm | 24.65 | 1.887 | 134.4 |

## 测试结论与限制

固定参数下，效果优劣应查看逐数据集均值与划分波动，不能据三次划分宣称统计显著。
Telco 是本组较大的数据集，C++ LibTree 训练中位数为 49.7 ms，XGBoost 为 53.1 ms；LibTree 耗时约为 XGBoost 的 0.93 倍。
Telco C++ LightGBM 训练中位数为 60.0 ms；LibTree 耗时约为 LightGBM 的 0.83 倍。
小型数据集的接口开销占比更高，毫秒级差异容易受共享 CPU 调度影响。原始文件保存了每次计时，可检查波动。

**参数与公平性**：100 棵树、深度 4、64 分箱、学习率 0.1、L2=1.0、最小增益 0.0；CPU 单线程，无 GPU。
LibTree 最少叶样本数为 5；XGBoost 默认最小 Hessian 权重为 1。LightGBM 为 leaf-wise，最少叶样本数 5、最大深度 4、num_leaves=16。分箱、树生长和叶约束并不完全相同，没有做超参数搜索。

**计时边界**：排除下载、CSV 读取、预处理、Python 导入和进程启动；训练包含分箱/DMatrix 构建与标签传递。
预测包含各接口必要转换：C++ XGBoost 构造测试 DMatrix，Python 使用 estimator 预测路径。比较的是端到端接口，不能视为纯树遍历吞吐量。

**内存边界**：RSS 为整个进程峰值，不是模型增量。Python 子进程均导入三种库；C++ 进程无 Python 运行时。
RSS 不能替代泄漏检查。本次额外通过四个基准准备测试（划分可复现、测试专有类别不被学习、损坏缓存被拒绝、执行进程内存计数有效）。
本轮实现和 ASan/UBSan/LSan、覆盖率验证见 [验证记录](../docs/validation.md)。

**外推限制**：随机留出集不是官方 Kaggle 测试集；Telco/Titanic 可能存在家庭或客户结构，未做分组或时间划分；不代表真实上线表现。
数据规模 768–7043 行，未包含百万行、稀疏、多分类、GPU或多线程任务。

## 数据来源与复现

环境：model name	: AMD EPYC 9V74 80-Core Processor；g++ (Debian 14.2.0-19) 14.2.0；Python 3.12.14；NumPy 2.3.5；sklearn 1.8.0；XGBoost 3.4.1；LightGBM 4.7.0。

```sh
cd /workspace/libtree
. /workspace/libtree-venv/bin/activate
python -m pip install -r benchmarks/requirements.txt
make all
PYTHONPATH=python:. python -m pytest tests/test_benchmark.py
PYTHONPATH=python python benchmarks/kaggle.py --engines libtree xgboost lightgbm --seeds 42 2024 2026 --repeats 5 --output benchmarks/optimized-results.json
python benchmarks/kaggle_report.py --input benchmarks/optimized-results.json --prefix optimized
```

原始数据保存在被 Git 忽略的 `benchmarks/cache/`。缓存不存在时通过 HTTPS 自动下载，并核验固定 SHA256。
划分后的输入与预测在 `benchmarks/runs/`，同样不纳入 Git。脚本不覆盖以前的 `results.json`。

[全部原始测量](optimized-results.json) · [聚合统计](optimized-summary.json)

- **titanic**：[Kaggle 页面](https://www.kaggle.com/competitions/titanic)；[实际下载镜像](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/train.csv)。SHA256：`14769fb1850e2d26d8e6db0ee49c213878040432827e39b13caaa15603c6598f`。
- **insurance**：[Kaggle 页面](https://www.kaggle.com/datasets/mirichoi0218/insurance)；[实际下载镜像](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)。SHA256：`505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281`。
- **pima**：[Kaggle 页面](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)；[实际下载镜像](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)。SHA256：`6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`。
- **telco**：[Kaggle 页面](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)；[实际下载镜像](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)。SHA256：`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`。
- **wine**：[Kaggle 页面](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)；[实际下载镜像](https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv)。SHA256：`4678927bac9ff54ac7431a10979a3a9b5ba014d20e14196b047073f2ceb75f4c`。

## 本轮性能排名

按每组运行中位数排名：训练 10/10 组第一，推理 10/10 组第一。
排名限定为上述本机、CPU 单线程、固定参数与数据集，不代表所有硬件或任务，也不表示每一次运行或每个划分都获胜。

同一划分的全部 LibTree 效果指标与优化前逐项相同。训练保留全部轮次和训练损失记录。

## 随机交错顺序的优化前后实测

原生 C++，同一输入，9 次独立重复，旧版 937e6b4fffeff31bc72018afa118630616b422b6。
此表为固定输入的配对复测，与上方多划分三方计时独立；每个样本的 float32 预测逐值相同。

| 数据集 | 旧训练 ms | 新训练 ms | 训练加速 | 旧推理 ms | 新推理 ms | 推理加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| titanic | 12.77 | 3.98 | 3.21× | 0.398 | 0.154 | 2.59× |
| insurance | 15.74 | 4.18 | 3.77× | 0.628 | 0.184 | 3.41× |
| pima | 8.45 | 5.75 | 1.47× | 0.506 | 0.159 | 3.18× |
| telco | 281.82 | 52.29 | 5.39× | 3.786 | 1.032 | 3.67× |
| wine | 12.61 | 7.47 | 1.69× | 0.836 | 0.196 | 4.27× |
| friedman_20k | 249.42 | 108.10 | 2.31× | 12.128 | 2.877 | 4.22× |

[全部配对测量](optimization-ab-results.json)


## 补充：20k 行合成数据

Friedman1，20 个特征、100 棵树；随机种子 42，训练 15000 / 测试 5000 行；每组合 5 次独立计时，共 30 次。
单划分补充测试，不与五数据集的多划分效果标准差混合。

| 接口 | 引擎 | 测试 RMSE | 训练 ms | 推理 ms |
| --- | --- | ---: | ---: | ---: |
| cpp | libtree | 1.1478 | 102.65 | 2.873 |
| cpp | xgboost | 1.1470 | 120.71 | 4.472 |
| cpp | lightgbm | 1.1326 | 130.01 | 17.007 |
| python | libtree | 1.1478 | 103.24 | 2.836 |
| python | xgboost | 1.1470 | 127.26 | 4.827 |
| python | lightgbm | 1.1326 | 140.02 | 18.030 |

[合成数据全部原始测量](optimized-synthetic-results.json)

## 复测优化前后及排名

```sh
python benchmarks/validate_ranking.py benchmarks/optimized-results.json --previous benchmarks/comparison-results.json --check-sources --require-top1 --output benchmarks/optimized-ranking.json
PYTHONPATH=python python benchmarks/run.py --suite builtin --datasets friedman_20k --seed 42 --repeats 5 --output benchmarks/optimized-synthetic-results.json
mkdir -p /tmp/libtree-before
git archive 937e6b4 | tar -x -C /tmp/libtree-before
cmake -S /tmp/libtree-before -B /tmp/libtree-before/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/libtree-before/build -j4
python benchmarks/ab.py --baseline /tmp/libtree-before/build/native_benchmark --baseline-ref 937e6b4 --repeats 9
```

配对测试使用上述两类比较生成的 benchmarks/runs/*.bin；重复执行不需要工作树或复制原始 CSV 入 Git。
[排名检查结果](optimized-ranking.json) · [性能设计与原理](../docs/design.zh-CN.md)
