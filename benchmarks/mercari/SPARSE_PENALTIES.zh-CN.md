# 无需新增配置的 Ridge、Lasso 与 Elastic Net

[English](SPARSE_PENALTIES.md) · [完整证据](sparse-penalties-results.json) · [原榜前三名复盘](TOP3.zh-CN.md)

本页记录已完成的直接融合实验。后续默认接入的[残差融合阶段](RESIDUAL_FUSION.zh-CN.md)
继续保留同一批原始模型，并通过正负系数 stacking 和残差 boosting 检查间接贡献。

Mercari 实验现在会在 Ridge 之外**自动加入 Lasso 和 Elastic Net**，不增加命令行
开关或配置文件，不需要你手动选择 alpha，也不执行参数搜索。新增两者都使用与
Ridge 相同的**无序词对视角**；这是上一轮四种新词法视角中表现最好的一组。
其他三组 Ridge 特征消融继续保留，避免扩成昂贵的模型／特征网格。

## 内置设置与原理

三者都用基础训练集的 log1p 价格训练，文本、IDF、哈希、结构字段、数据划分和
RMSLE 的预测裁剪一致。坐标下降接收 CSC，推理使用 CSR，不把文档矩阵转成稠密矩阵。
模型和预测都保存到运行目录。

新增[求解器](../mercari_penalties.py)保留 Lasso／Elastic Net 的目标函数和
**不受正则惩罚的截距**，沿用 sklearn 的预测接口。它维护原始残差向量及其总和，
每个系数更新只处理该列的非零元素，用代数关系处理均值，不逐列移动全部残差，
也不创建稠密的“样本数×特征数”矩阵。每五轮及预算末尾检查一次原始／对偶间隙。
Numba 编译循环；Python 模式负责覆盖率，编译模式另做参考验证。
新求解器采用 float64 残差和系数；Ridge 保留此前的 float32 特征计算路径。
比较时间和精度时，需要同时考虑求解器、精度和容差差异。

这样做是因为 sklearn 1.8 的稀疏中心化路径在
[`_cd_fast.pyx`](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/linear_model/_cd_fast.pyx)
中包含特征处理内部的全样本残差循环。首轮原生尝试运行 **475 秒**后停止，
Lasso 尚未完成，没有两种新模型的精度结果。失败记录保存在证据中，不能用未完成
运行计算精确加速比。小规模测试已与 sklearn 核对相关特征、非零均值、常数／零列，
以及有／无截距时的预测和系数。新求解器限定为不带样本权重的全新循环坐标拟合，
不支持正系数约束、热启动或预计算 Gram 矩阵；暴露出的不支持选项会明确拒绝。

基础训练集有 `n` 行时，Lasso 内置 `alpha=20/n`；Elastic Net 内置
`alpha=40/n`、`l1_ratio=.5`。两者使用循环坐标下降，最多 200 次迭代、容差 1e-4。
按行数换算，是为了把固定的**求和损失正则系数**转换成 sklearn 的**平均损失约定**：

| 模型 | 目标函数，省略不受惩罚的截距 |
| --- | --- |
| 默认 Ridge | ½ 残差平方和 ＋ 10 ‖β‖₂² |
| Lasso | ½ 残差平方和 ＋ 20 ‖β‖₁ |
| Elastic Net | ½ 残差平方和 ＋ 20 ‖β‖₁ ＋ 10 ‖β‖₂² |

常数在查看本轮 selection 和审计结果之前固定。换算不会保证不同数据规模下的
最优正则化，也不代表 L1 和 L2 的统计效果相同。已有的 Ridge 命令行选项保留兼容，
不会改变两种新模型的内置正则设置。

Lasso 可以把部分系数直接压到零；Elastic Net 加入 L2，有助于相关特征共享权重。
这些性质不能自动推出更好的 RMSLE 或更快的完整训练。此处 sklearn 仍用稠密数组
保存系数，所以“非零系数更少”也不等于已经压缩了模型文件。

## 融合与测试规则

默认输出四个 Ridge 专家，以及 `lasso_pairs`、`elasticnet_pairs`。
外层在分组 selection 半集上比较带截距的和为 1 融合、仿射融合、品类融合，
并记录七组消融：原 anchor、去词对、去数量词、仅 Ridge、去 Lasso、
去 Elastic Net、完整池。所有选择和晋级决定先冻结，再打开审计标签。
晋级失败时保留逐元素完全一致的旧预测。历史窗口复用的局限仍然存在。

报告记录实际参数、迭代次数、非零系数数量、可获得的对偶间隙和全部拟合警告。
坐标下降没有收敛时会明确标记，是否达到迭代上限另行记录。
Ridge 的 LSQR 包装没有暴露完整停止状态，因此其 `converged` 为 null，
不会仅凭“没有警告”就宣称已经求得收敛解。

推理在已准备矩阵上**重复三次**，主指标使用**算术平均数**，同时保留三次原始值、
最大值和总体标准差。拟合时间为每个模型训练一次的观察值；文本处理、CSR／CSC
拼接时间分别记录，不包含旧 anchor 的完整训练成本。

## 完整基础集结果

三者使用相同的 **1,186,026 条训练记录／524,288 列词对视角特征**；
selection 为 74,126 条，复用审计集为 148,254 条。

| 相同词对视角 | Selection RMSLE | 复用审计 RMSLE | 非零系数 | 拟合秒（一次） | 推理平均秒（三次） | 迭代次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 0.473492 | 0.472611 | 405,988 | 16.22 | 0.0439 | 30 |
| Lasso | 0.515471 | 0.515175 | 3,474 | 128.49 | 0.1665 | 200 |
| Elastic Net | 0.522837 | 0.522774 | 3,916 | 116.96 | 0.0956 | 200 |

推理都覆盖相同的 296,509 条校准／selection／审计记录。
最大值、标准差和三次原始时间保存在 JSON。系数数量以所有哈希列为范围，
其中也包括未见过的列，不能当作“重要单词数”。四组 Ridge 对照与上一轮所有窗口
的预测最大差值为 **0**。

坐标下降收敛记录：**lasso_pairs：未收敛，对偶间隙 0.000144145；elasticnet_pairs：未收敛，对偶间隙 0.000114113**。实际 Lasso alpha 为
1.68630367e-05，Elastic Net 为
3.37260735e-05，来自训练行数换算，不是验证集调参。
求解器与容差不同，且每个模型只训练一次，不能据此宣称某求解器普遍更快。

**未通过 selection 晋级门禁，逐元素保留原 anchor 预测。** 基线审计 RMSLE 为 **0.380616255**，
最终保留输出为 **0.380616255**。
七组家族／特征消融与 selection 方法检查全部保存在 JSON；晋级决定没有使用本轮
审计结果，没有新增 Kaggle 提交。

报告计时的增量阶段耗时 **481.93 秒**，其中文本特征
**162.57 秒**、共享矩阵准备 **3.95 秒**，
峰值 RSS **3579.34 MiB**。仅使用 CPU，线程上限为四；
坐标下降本身不是四个 worker 的并行训练。另有原生尝试 **475 秒**，
两次记录的实验阶段合计约 **957 秒**，不含回归测试与旧 anchor 的完整重训成本。
计时不含 Python 导入、末尾审计 bootstrap 和输出写入。
两个新专家都达到 200 轮上限，未达到约定收敛容差；这里是固定预算结果，
不能当作充分收敛解或否定两种算法在其他默认设置下的效果。

验证：**169 项测试通过**，本次检查的 **23 个 Mercari CPU 模块／2,992 条语句覆盖率为 100%**；两个变更的求解器／运行模块为 **249／249**。这不是整个仓库或分支覆盖率。开启 JIT 后另有 **16 项测试通过**，包括与 sklearn 的参考核对；已有依赖警告仍保留。

## 使用方法

复用[前一轮](SMALL_TEXT.zh-CN.md)准备的数据和 anchor。命令保持不变，
无需为 Lasso 或 Elastic Net 填写任何参数：

```bash
NUMBA_DISABLE_JIT=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -m benchmarks.mercari_top3 \
  --data-dir benchmarks/runs/mercari/frontier-001/data \
  --anchor benchmarks/runs/mercari/smalltext-t4-001/primary-blend/ensemble.npz \
  --output-dir benchmarks/runs/mercari/sparse-penalties-reproduction-001 \
  --external-pretrained-data

NUMBA_DISABLE_JIT=1 python -m pytest \
  tests/test_mercari_top3.py tests/test_mercari_penalties.py \
  --cov=benchmarks.mercari_top3 --cov=benchmarks.mercari_penalties \
  --cov-report=term-missing --cov-fail-under=100
NUMBA_DISABLE_JIT=0 python -m pytest tests/test_mercari_penalties.py
```

只需提供数据、anchor 和新输出目录。预训练标记声明旧 anchor 的来源，新线性专家
自身不使用预训练权重。模型、数据和运行产物继续由 Git 忽略，不需要新增依赖。

初学者可以这样理解：Ridge 倾向于把系数变小，Lasso 还能把系数变成零，
Elastic Net 同时使用两种约束。非零系数数目对应活跃的哈希列，不等于有多少个
“重要单词”，因为哈希碰撞和相关性都会影响解释。单模型更好与改善最终融合，
仍然是两个需要分别验证的问题。

引用：[Tibshirani，Lasso（1996）](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)、
[Zou 与 Hastie，Elastic Net（2005）](https://doi.org/10.1111/j.1467-9868.2005.00503.x)、
[Friedman、Hastie 与 Tibshirani，坐标下降（2010）](https://doi.org/10.18637/jss.v033.i01)。
它们是方法来源，论文结果不算作本仓库实测成绩。
