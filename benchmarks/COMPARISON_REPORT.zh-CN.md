# Kaggle 数据集 GBDT 三方比较

测试时间（北京时间）：2026-10-09 17:40。

完成 **5 个数据集 × 3 个不同划分 × 2 个接口 × 3 个引擎 × 3 次计时 = 270 次实测**。
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

每个组合的时间和内存取三个划分、每划分三次，共 9 次运行的中位数。每次启动独立进程。内存采用执行后地址空间的 Linux VmHWM，避免继承父进程的高水位。

| 数据集 | 接口 | 引擎 | 训练 ms | 预测 ms | 峰值 RSS MiB |
| --- | --- | --- | ---: | ---: | ---: |
| titanic | cpp | libtree | 12.47 | 0.499 | 2.0 |
| titanic | cpp | xgboost | 8.85 | 0.384 | 7.3 |
| titanic | cpp | lightgbm | 7.59 | 0.570 | 9.2 |
| titanic | python | libtree | 12.18 | 0.497 | 127.4 |
| titanic | python | xgboost | 14.84 | 1.177 | 130.8 |
| titanic | python | lightgbm | 18.59 | 1.195 | 134.5 |
| insurance | cpp | libtree | 15.57 | 0.638 | 2.0 |
| insurance | cpp | xgboost | 9.31 | 0.406 | 7.4 |
| insurance | cpp | lightgbm | 7.51 | 0.892 | 9.2 |
| insurance | python | libtree | 15.24 | 0.666 | 127.4 |
| insurance | python | xgboost | 14.48 | 1.125 | 130.8 |
| insurance | python | lightgbm | 18.00 | 1.505 | 132.9 |
| pima | cpp | libtree | 7.89 | 0.489 | 2.0 |
| pima | cpp | xgboost | 14.02 | 0.359 | 7.3 |
| pima | cpp | lightgbm | 10.32 | 0.645 | 7.7 |
| pima | python | libtree | 8.41 | 0.595 | 127.3 |
| pima | python | xgboost | 19.54 | 1.170 | 130.8 |
| pima | python | lightgbm | 21.94 | 1.348 | 132.7 |
| telco | cpp | libtree | 283.59 | 3.283 | 3.6 |
| telco | cpp | xgboost | 58.71 | 2.370 | 12.2 |
| telco | cpp | lightgbm | 63.55 | 5.218 | 11.8 |
| telco | python | libtree | 292.98 | 3.502 | 129.2 |
| telco | python | xgboost | 59.99 | 2.630 | 133.5 |
| telco | python | lightgbm | 72.47 | 5.963 | 136.3 |
| wine | cpp | libtree | 12.42 | 0.779 | 2.0 |
| wine | cpp | xgboost | 19.81 | 0.461 | 7.4 |
| wine | cpp | lightgbm | 11.84 | 1.002 | 9.3 |
| wine | python | libtree | 12.50 | 0.865 | 127.4 |
| wine | python | xgboost | 26.66 | 1.381 | 130.9 |
| wine | python | lightgbm | 22.88 | 1.680 | 134.4 |

## 测试结论与限制

固定参数下，效果优劣应查看逐数据集均值与划分波动，不能据三次划分宣称统计显著。
Telco 是本组较大的数据集，C++ LibTree 训练中位数为 283.6 ms，XGBoost 为 58.7 ms；LibTree 耗时约为 XGBoost 的 4.83 倍。
Telco C++ LightGBM 训练中位数为 63.6 ms；LibTree 耗时约为 LightGBM 的 4.46 倍。
小型数据集的接口开销占比更高，毫秒级差异容易受共享 CPU 调度影响。原始文件保存了每次计时，可检查波动。

**参数与公平性**：100 棵树、深度 4、64 分箱、学习率 0.1、L2=1.0、最小增益 0.0；CPU 单线程，无 GPU。
LibTree 最少叶样本数为 5；XGBoost 默认最小 Hessian 权重为 1。LightGBM 为 leaf-wise，最少叶样本数 5、最大深度 4、num_leaves=16。分箱、树生长和叶约束并不完全相同，没有做超参数搜索。

**计时边界**：排除下载、CSV 读取、预处理、Python 导入和进程启动；训练包含分箱/DMatrix 构建与标签传递。
预测包含各接口必要转换：C++ XGBoost 构造测试 DMatrix，Python 使用 estimator 预测路径。比较的是端到端接口，不能视为纯树遍历吞吐量。

**内存边界**：RSS 为整个进程峰值，不是模型增量。Python 子进程均导入三种库；C++ 进程无 Python 运行时。
RSS 不能替代泄漏检查。本次额外通过四个基准准备测试（划分可复现、测试专有类别不被学习、损坏缓存被拒绝、执行进程内存计数有效）。
算法核心未修改，之前的 ASan/UBSan/LSan 和覆盖率结果见 [验证记录](../docs/validation.md)。

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
PYTHONPATH=python python benchmarks/kaggle.py --engines libtree xgboost lightgbm --seeds 42 2024 2026 --repeats 3 --output benchmarks/comparison-results.json
python benchmarks/kaggle_report.py --input benchmarks/comparison-results.json --prefix comparison
```

原始数据保存在被 Git 忽略的 `benchmarks/cache/`。缓存不存在时通过 HTTPS 自动下载，并核验固定 SHA256。
划分后的输入与预测在 `benchmarks/runs/`，同样不纳入 Git。脚本不覆盖以前的 `results.json`。

[全部原始测量](comparison-results.json) · [聚合统计](comparison-summary.json)

- **titanic**：[Kaggle 页面](https://www.kaggle.com/competitions/titanic)；[实际下载镜像](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/titanic/train.csv)。SHA256：`14769fb1850e2d26d8e6db0ee49c213878040432827e39b13caaa15603c6598f`。
- **insurance**：[Kaggle 页面](https://www.kaggle.com/datasets/mirichoi0218/insurance)；[实际下载镜像](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)。SHA256：`505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281`。
- **pima**：[Kaggle 页面](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)；[实际下载镜像](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)。SHA256：`6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`。
- **telco**：[Kaggle 页面](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)；[实际下载镜像](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)。SHA256：`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`。
- **wine**：[Kaggle 页面](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)；[实际下载镜像](https://raw.githubusercontent.com/plotly/datasets/master/winequality-red.csv)。SHA256：`4678927bac9ff54ac7431a10979a3a9b5ba014d20e14196b047073f2ceb75f4c`。
