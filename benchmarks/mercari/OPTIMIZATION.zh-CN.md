# Mercari 优化实验与复现指南

[English](OPTIMIZATION.md) · [比赛记录](README.zh-CN.md) · [冻结参数](optimized-config.json)

目标是让 Kaggle private RMSLE 相对原成绩 **0.43776** 再下降 **10%**，即达到
**0.393984** 或更低。这里的 10% 指误差指标；训练与推理时间单独报告。
原比赛已经结束，当前是补赛练习，不宣称获得原赛事名次。

## 本地独立审计结果

所有参数与融合权重冻结后，检查原先保留的 **148,254 条留出样本**，包含
92 条零价格样本。结果如下；这些是本地结果，不能替代实际 Kaggle 分数。

| 模型或组件 | 留出集 RMSLE，越低越好 |
| --- | ---: |
| 原版 Ridge + LibTree 基线 | 0.439312 |
| 新版 Ridge | 0.433543 |
| 字段感知因子分解 | 0.418600 |
| 四个神经网络的融合 | 0.397625 |
| 新的完整融合，去掉 LibTree 校准 | 0.395064 |
| **完整融合 + LibTree 校准** | **0.394153** |

相对原基线下降 **10.28%**。同一组基础模型加上 LibTree 后再下降约 0.00091。
主要收益来自整个特征与融合系统，不能把全部提升归因于 C++ 树算法。

[审计原始数据](optimization-audit.json) · [探索实验汇总](optimization-study.json) ·
[推理计时记录](inference-optimization.json)。同一批 10 万条无标签测试样本，预热一次、
交错重复三次，新系统平均 **34.819 秒**，最小／最大 **31.699／40.973 秒**；旧系统
平均 **7.941 秒**，最小／最大 **7.067／9.231 秒**。新系统推理成本约为原来的
**4.38 倍**，包含特征处理。测试的显式稀疏矩阵乘法方案没有稳定速度收益，已弃用。

## 本地全量重训成本

[最终重训报告](optimized-final-001.json)记录了 **1,395,326 条基础模型训练样本**、
**87,209 条独立校准样本**，以及 **3,460,725 条第二阶段预测**。使用四 CPU 配额，
没有使用 GPU；进程峰值 RSS 为 **10.15 GiB**，环境内存上限为 32 GiB。

| 阶段 | 本次实际墙钟时间，秒 |
| --- | ---: |
| 共享特征与 Ridge | 387.672 |
| 字段感知模型，连续两个训练轮次合计 | 85.308 |
| 四个神经网络，含各自特征准备 | 1,418.208 |
| 校准特征与基础模型预测 | 116.627 |
| LibTree 校准拟合 | 1.452 |
| 第二阶段完整预测、特征处理与 CSV 写入 | 1,184.582 |
| **整个进程，含启动及其他准备** | **3,225.167（53.75 分钟）** |

这些是一次完整运行的实际成本，期间有少量并发工具检查，不是重复训练的性能分布。
原生拟合使用校准窗口与 300 棵深度为 3 的树。[资源记录](optimization-resources.json)
保存了测量范围；重复推理的均值与波动见上表。结束后已核验 CSV 行数，以及首条预测
与解释贡献之和一致。上述全量成本来自本地进程；Kaggle 验证使用已认证的源码、
产物元数据和真实比赛评分。当前出口代理无法下载托管运行的报告文件。

## 为什么这些改动有用

1. **保留型号信息。** 原默认分词忽略单字符 `6`、`7`、`x`；新版保留它们，
   并把 `iphone6s` 规范为 `iphone 6s`，保留 `64gb` 等容量单位。
   预处理版本随模型保存，旧模型也可以按原规则重训、加载。
2. **增加可泛化的商品信息。** 品牌缺失时，只从训练品牌字典里寻找无歧义的
   标题匹配；加入品类层级、品牌／品类／成色交叉、字符 n-gram、平滑价格统计。
   字典和价格统计只用允许的训练窗口，预测时不更新。
3. **改善稀疏求解。** Ridge 用带对角预条件的共轭梯度，隐式处理中心化，
   不生成巨大的稠密商品矩阵。数值测试与独立的稠密 Cholesky 解对照。
4. **学习词语与类别的交互。** 字段感知因子分解能学习“某个词出现在某品牌下”
   的效应。各 CPU worker 使用独立参数，按批同步，避免共享参数并发写入。
5. **利用不同损失的互补性。** 比较 MSE、Huber 回归和 64 档价格分类。
   稀疏网络按批处理 CSR；按行更新的 Adam 合并重复梯度，只更新出现的特征行。
   该变体同时去掉会产生稠密梯度的全矩阵 L2，不能只把差异归因于优化器。
6. **控制部署开销。** 预测时复用预处理和稀疏特征；LibTree 学习独立样本上的
   残差。Kaggle 脚本直接读取 ZIP，不重复解压大型测试文件，也不保存大模型 pickle。

## 评估如何避免自我欺骗

官方训练集共 **1,482,535 条**，包括 874 条零价格数据，均保留。没有使用测试
标签、外部价格查询、Mercari 抓取或预训练模型输入。

按六个可用商品字段分组，排除价格和 ID；完全相同的商品不能跨窗口。
原 seed-42 划分约为训练 80%、调参 10%、审计 10%。基础模型探索和停止轮次使用
调参数据；其中再按组分为 **74,129 条校准样本**与 **74,126 条选择样本**。
LibTree 只拟合校准样本的残差，融合和树参数在选择样本上比较。因此探索期的
调参成绩不是独立测试；最终结论另看未用于选择的原留出集与实际 Kaggle 评分。

最终训练使用固定参数，把全部训练标签分成约 **16/17 的基础模型窗口**和
**1/17 的 LibTree 校准窗口**。同一商品组不跨窗口；所有标签参与系统，但角色不同。
Kaggle 内不再搜索参数或根据榜单选择模型。

在相同的 148,255 条调参样本上，修复分词后的 Ridge 为 **0.435320**，字段感知
模型为 **0.420152**。稀疏 MLP 的 Adam + L2、按行 Adam + Huber、按行 Adam +
价格分箱分别为 **0.417217、0.416794、0.417463**。单模型相近，误差互补才是
融合的价值。连续 TF-IDF 网络只有很小的融合收益；CNN 和另一个词向量候选成本
偏高，均未纳入最终系统。完整消融表见[英文报告](OPTIMIZATION.md#exploratory-ablations)。

探索模型曾并发运行，其耗时不能当作严格的速度对照。性能重复测量保留每次结果、
**算术平均值、最小值、最大值和标准差**。准确度提升也伴随更多模型与计算成本。

## 新手如何运行

准备 Linux、Python 3.12、C++17 编译器，以及[官方数据下载步骤](README.zh-CN.md)中的文件：

```sh
python -m pip install '.[mercari]' kaggle
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/optimized-config.json \
  --output-dir benchmarks/runs/mercari/optimized-final --mode final
```

输出目录必须为空。结果包含 submission CSV、数据与代码哈希、版本、耗时、模型、
文本树、JSON 树及第一条真实测试样本的解释。加 `--omit-model-artifact` 可省去
大模型文件，保留预测、报告和解释。数据、模型、缓存与预测均放在被 Git 忽略的
`benchmarks/runs/` 下；配置、代码和汇总报告可以提交 Git。

要复测推理耗时，保留可信的本地模型文件，在没有并发训练的新进程中运行：

```sh
PYTHONPATH=. python -m benchmarks.mercari_inference \
  --baseline-model benchmarks/runs/mercari/full-001/model.pkl \
  --optimized-model benchmarks/runs/mercari/optimized-audit/model.pkl \
  --test-file benchmarks/runs/mercari/data/test_stg2.tsv \
  --output benchmarks/runs/mercari/serving-comparison.json \
  --rows 100000 --repeats 3 --threads 4 --fm-threads 2
```

按实际输出目录调整模型路径。命令会预热、交错执行、核对重复预测，输出算术平均值、
最小／最大值、标准差和每次耗时，并恢复 Numba 线程数。加载、输入 I/O 和预测校验
不计时。pickle 可执行代码，只加载可信文件。比较对象是所提供的完整系统，
其中训练样本量和模型容量的差别也应明确报告。

把 `--mode final` 改为 `--mode study` 可复现固定的本地划分，默认只报告选择集。
仅在冻结选择后加 `--audit-holdout`。这是全量计算，不能把小样本试跑时间当全量时间。

```sh
python -m benchmarks.mercari_kernel --owner YOUR_USERNAME \
  --optimized-config benchmarks/mercari/optimized-config.json \
  --output-dir benchmarks/runs/mercari/optimized-kernel
kaggle kernels push -p benchmarks/runs/mercari/optimized-kernel -t 3600
```

脚本在私有 Kernel 内编译 C++ 并重新训练、预测；不会上传本地模型或自动提交。
记录 push 返回的版本，完成后按比赛记录中的方法提交该版本的 `submission.csv`。

## 怎样理解预测与解释

RMSLE 比较 `log1p(price)`，所以模型学习这个值。Ridge 学习词语的加性效应；
因子分解学习词语与品牌等组合；稀疏神经网络学习更复杂的词语／类别关系。
Huber 减少极大误差的影响；价格分箱模型先预测各价格区间的概率，再求预期对数价格。

各模型加权得到基础对数价格，LibTree 再给出修正。修正 `0.1` 表示把 `price + 1`
乘以约 `1.105`。解释文件同时给出模型分支贡献和原生树路径；贡献在**对数价格**
上求和，不是美元贡献，也不是因果效应或 SHAP 值。

可先阅读离线[交互学习指南](../../docs/LEARNING_GUIDE.html)，再看
[原生树路径说明](../../docs/explain.md)。

## 测试与引用

[验证记录](optimization-verification.json)保存了 **50 项 Mercari 测试通过、
1,049/1,049 条 Python 语句覆盖**，并另测 4 项真实编译的因子分解内核测试。
其中模型测试 47 项、推理命令测试 3 项，本地分开执行，CI 中合并验证。核心仍为
**Python 505/505 条语句、C++ 2,790/2,790 行覆盖**。
推理对照命令另有 **3 项测试通过、72/72 条语句覆盖**，并用两套真实保存模型
完成加载、预测一致性与输出检查；这次小样本检查不作为性能结论。
[GitHub Actions 39042430151](https://github.com/wxwidget/libtree/actions/runs/39042430151)
全部成功，包含 sanitizer、线程 sanitizer 和新增的 wheel 安装集成测试。
CI 明确验证已安装 wheel 的原生库，避免未编译的源码包覆盖已安装包。
[工作流文件](../../.github/workflows/test.yml)提供完整的依赖安装、覆盖率和编译内核
验证命令，可直接按步骤复现。

测试涵盖无标签泄漏、旧版本重训、分组隔离、稀疏／稠密数值一致性、显式成对
因子分解公式、worker 同步、重复梯度合并、稳定分箱、模型重放、贡献求和、ZIP
读取与完整 CLI。Numba 源码行覆盖使用 `NUMBA_DISABLE_JIT=1` 执行 Python 参考
内核，另外测试真实编译的并行内核；行覆盖率不等同于分支覆盖率或所有情况都正确。
C++ 核心保留独立的 sanitizer、覆盖率、线程生命周期和内存回归检查。

主要借鉴：[Jankiewicz 与 Lopuhin 的第一名技术报告](https://github.com/pjankiewicz/mercari-solution/blob/374301ad1c32cbc93dcc40313d5d7bb9c5503746/presentation/5_modeling.tex)
中的稀疏 MLP、鲁棒／分箱损失与融合，以及
[Chenglong Chen 的第四名报告](https://github.com/ChenglongChen/tensorflow-XNN/tree/6534a832f5b4461cbdf1ebbdf5620a0cd80f0aa9)
中的多字段模型与稀疏优化。这里是独立的 Keras 3 实现，前人的成绩不作为我们的
实测成绩。共轭梯度、目标编码、因子分解、局部 SGD、Huber 与 Adam 的论文链接
见[英文引用列表](OPTIMIZATION.md#tests-and-references)。
