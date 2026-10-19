# LibTree

[同款检索与文本主干联合微调](benchmarks/mercari/RETRIEVAL_TRUNK.zh-CN.md)将一个 CNN
分支的复用审计 RMSLE 从 0.424760 降至 0.423351，相对同预算训练对照改善 0.199%。
第二个种子未通过高价保护，原始输入推理为对照耗时的 1.94 倍；保留失败诊断、
完整成本及复现方法，最终融合保持冻结。

[二手市场与融合前模型优化](benchmarks/mercari/MARKET_MODELS.zh-CN.md)用基础集同款
统计，将一个文本分支的审计 RMSLE 从 0.407565 降到 0.405882（0.413%）。
最终融合保持冻结；报告解释非高斯价格与目标函数的关系，保留分布头失败、低价
仍退步的结果，并给出成本与复现方法。

[最终损失融合实验](benchmarks/mercari/FINAL_OBJECTIVE.zh-CN.md)实现精确 RMSLE 梯度及
三个 MLP 最终平均输出的联合训练。复用审计 RMSLE 从 0.379682 降至 0.379333，
同优化器的对数 MSE 对照为 0.379348。根号本身及联合训练的优势仍不确定；
冻结前的检查选择独立对数 MSE 平均方案。

[Mercari 融合对照](benchmarks/mercari/FUSION_STRATEGIES.zh-CN.md)用相同候选预测，
验证单层线性、多层 MLP、分条件线性，包含分组元模型验证、种子波动与家族消融。

[Mercari 弱候选残差融合](benchmarks/mercari/RESIDUAL_FUSION.zh-CN.md)说明当前线性与
残差融合层次，保留弱模型参与后续校正。该前序实验改善了稀疏分支，保留 0.380616 原方案。

[Mercari Kaggle 打榜记录](benchmarks/mercari/README.zh-CN.md)
提供 LibTree 价格预测、文本／结构化模型消融、重复商品不跨集合的验证和真实提交记录；
[英文运行指南](benchmarks/mercari/README.md)提供复现命令与历史榜单比较的限制。
新增[Mercari 优化实验](benchmarks/mercari/OPTIMIZATION.zh-CN.md)：冻结配置后本地留出集
RMSLE 下降 10.28%；实际 Kaggle private RMSLE 从 0.43776 降到
**0.39199（下降 10.46%）**。报告包含多模型融合、损失消融、CPU 成本与复现方法。这是赛后练习；
精度收益来自完整融合系统，同时增加了推理成本。

[Mercari 技术复盘](benchmarks/mercari/TECHNICAL_REPORT.zh-CN.md)介绍架构、成功与失败的
实验、理论引用及证据边界；独立的[初学者实战指南](benchmarks/mercari/BEGINNER_GUIDE.zh-CN.md)
从无需凭据的小练习开始，逐步解释训练、预测解释、性能测量和 Kaggle 提交的复现方法。

[当前 CPU 预算实验](benchmarks/mercari/CPU_STUDY.zh-CN.md)借鉴冠军的稀疏文本、
多视图与目标互补，使用解析梯度直接优化 RMSLE 平方，并执行硬时间限制。
排队中的 GPU 部署已停止。20 万样本筛选用时 806 秒，相对缓存 CPU 基线的本地复核
从 0.394153 改善到 0.394065（0.022%），收益太小，本轮不晋升，不宣称新榜单成绩或排名。

后续[CPU 多视角实验](benchmarks/mercari/CPU_MULTIVIEW.zh-CN.md)复用完整基础集专家，增加
商品身份统计、父级平滑和多参数校准，本地复核进一步达到 **0.392846**。
两轮增量 CPU 工作合计 **370 秒**，不包含旧专家训练成本；简单非负融合优于更灵活的替代方法。

此前融合在复用的本地审计窗口取得 **0.380907 RMSLE，较 0.394153 下降 3.36%**；
尚未产生新的 Kaggle 成绩。
[原完整融合的最终重训与提交](benchmarks/mercari/FULL_ENSEMBLE.zh-CN.md)保留 0.380907 本地方案，
记录现已停止的 GPU 部署工作流。
[Colab 小尺寸文本实验](benchmarks/mercari/SMALL_TEXT.zh-CN.md)增加词袋回归与
BERT-tiny，限制参数和运行预算、保留检查点，并校验和复用编码缓存。
[本轮实测报告](benchmarks/mercari/SMALL_TEXT_REPORT.zh-CN.md)：本地复核 RMSLE
**0.380616**，较旧 0.380907 改善 0.0763%；T4 原生精度修复在固定批量微基准中
训练步 **3.31 倍**、推理批次 **2.70 倍**，不是端到端加速。这些是 benchmark
专家模型，LibTree 原生学习器仍在 CPU 上运行。

[原榜前三名源码复盘](benchmarks/mercari/TOP3.zh-CN.md)核对数量词、无序词对和品类融合的
一手来源，提供完整基础集消融与复现方法；本轮新专家未通过晋级门禁，保留较强的旧方案。
[Lasso／Elastic Net 扩展](benchmarks/mercari/SPARSE_PENALTIES.zh-CN.md)默认加入两种稀疏
线性专家，无需增加参数配置，并报告收敛、稀疏度和推理平均耗时。

[现代文本与 GPU 实验](benchmarks/mercari/FRONTIER.zh-CN.md)增加 LSA、预训练句向量、
T4/A100 神经网络和树／回归融合，提供复现命令、失败记录与验证边界。

风险建模新增 boosting 开关、类别平衡、Hellinger 二分类准则与 KS 指标：
[优化方法及引用](docs/c5-risk.zh-CN.md) · [风险评测](benchmarks/RISK_REPORT.zh-CN.md)。

新增独立的 **C5 风格多分类器**：SAMME boosting、剪枝、IF–THEN 规则、
gain ratio/entropy/gini、样本权重与单条预测解释。参阅[C5 新手指南](docs/c5.zh-CN.md)、
[交互规则演示](docs/C5_GUIDE.html)和[实测竞品比较与消融](benchmarks/C5_REPORT.zh-CN.md)。

C5 现支持完整剪枝树的文本／JSON 导出、逐节点预测路径及各棵树的票数。
本轮复用直方图与训练叶标签，消除重复分裂计算和原生校验，Python 类别预测直接使用原生投票。
参阅[最新 C5 性能报告](benchmarks/C5_PERFORMANCE_REPORT.zh-CN.md)：七个任务、C++／Python、
单／四线程，包含旧版本和竞品对比、算术平均值、波动及退化结果，并检查模型输出逐位一致。

三个 Python 模型新增 **SHAP 解释与 sklearn 集成兼容性**：预测值／概率归因、
Voting／Stacking／Bagging、Pipeline、网格搜索，以及不保存训练数据的多进程模型快照。
参阅[中文集成指南](docs/python-integration.zh-CN.md)，包含可运行示例和明确的支持范围。

[English](README.md) · [原理与性能设计](docs/design.zh-CN.md) · [实测基准](benchmarks/README.md)

一个简洁、确定性的 **C++17 直方图 GBDT**，提供 NumPy/Python 接口。
支持平方误差回归、二分类 logistic 损失，并自动学习 NaN 特征的分支方向。
Python 直接调用同一个 C++ 模型，没有重复实现算法。

## 新手从这里开始

Linux 上需要 GCC 或 Clang、Python 3.9+ 和 Make。纯 C++ 编译只需要
CMake 3.16+，不依赖其他 C++ 库。在仓库根目录执行：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install '.[dev,benchmark]'
make test
./build/libtree_example
```

当前云环境已经准备好，可直接执行
`. /workspace/libtree-venv/bin/activate`，然后 `cd /workspace/libtree`。
只使用 Python 时，`python -m pip install .` 即可编译并安装原生 wheel 和 NumPy。
必须有 C++17 编译器。目前验证的平台为 Linux，未提供 Windows wheel。
XGBoost 与 LightGBM 仅用于基准比较。

```python
import numpy as np
from libtree import GBDTRegressor, GBDTClassifier

x = np.array([[0], [1], [2], [3]], dtype=np.float32)
y = np.array([0, 0, 1, 1], dtype=np.float32)
with GBDTRegressor(min_samples_leaf=1) as model:
    model.fit(x, y)
    print(model.predict(x))
    print(model.training_loss_[-1])

classifier = GBDTClassifier(min_samples_leaf=1).fit(x, y)
print(classifier.predict_proba(x))  # 两列分别为 P(0)、P(1)
classifier.close()
```

输入为稠密二维矩阵 `(样本数, 特征数)`，标签为一维向量；GBDT 分类器支持
恰好两个有限数值或字符串类别，概率列顺序由排序后的 `classes_` 决定。
类别特征需要编码，编码器只在训练集上拟合，基准脚本提供了示例。
连续的 float32 数组可避免类型转换复制。NaN 表示缺失值；无穷大和错误形状
会被拒绝。`fit` 替换旧模型，不修改输入。`close` 释放原生内存，自动回收也
有效。支持 sklearn 模型类型／拟合状态标签、`get_params`、`set_params`、`score`、
`clone` 和常见模型组合；高级 metadata routing 尚未实现。

## 参数化命令行

原生主程序 `xgbt` 支持 train/predict、CSV 输入、模型文件和全部 GBDT 参数。
先运行 `./build/xgbt --help`，再查看[命令指南](docs/cli.zh-CN.md)。
[三方对比报告](benchmarks/COMPARISON_REPORT.zh-CN.md)增加了 LightGBM，分别对比
C++ 与 Python 的效果、训练/预测时间及内存。

[最新优化证据](benchmarks/OPTIMIZATION_REPORT.zh-CN.md)包含 450 次测量：
五数据集、单线程基准的 10 个数据集/接口组合中，LibTree 训练和推理均为
三者中中位耗时最低，同一划分的效果指标与优化前完全相同。排名限于本次
工作负载和硬件。C++ `PredictInto(view, output)` 直接写入独立的调用者
float 缓冲区；仍可使用返回 vector 的 `Predict`。

LibTree 将 XGBoost 的二阶划分评分和数据局部性思路，与 LightGBM 的紧凑
直方图、直方图差分和独立任务调度结合，并通过本地测试决定优化是否保留。
加权近似 sketch、GOSS、EFB 和 leaf-wise 生长尚未实现；实现差异、论文引用和
实测优势边界见[研究引用与实现对应关系](docs/references.zh-CN.md)。

## 五分钟使用 C++

先看 [examples/train.cc](examples/train.cc)。编译命令：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

```cpp
#include "libtree/gbdt.h"
#include <vector>

std::vector<float> x{0, 1, 2, 3};
std::vector<float> y{0, 0, 1, 1};
libtree::Parameters p;
p.min_samples_leaf = 1;
libtree::Gbdt model(p);
model.Fit({x.data(), 4, 1}, y);
auto prediction = model.Predict({x.data(), 4, 1});
```

编译时添加 `-Iinclude -std=c++17`，链接 `build/libtree.a`。
`MatrixView` 是不拥有数据的行优先视图，调用者必须保证缓冲区至少包含
`rows * cols` 个 float。训练后的模型不持有输入矩阵或标签。
不可变模型支持 C++ 并发预测；同一模型不能同时训练、关闭和预测。
C++ 输入错误抛出标准异常，C ABI 将异常转换为返回码和线程局部错误消息。

## 输出决策树与解释单条预测

训练后的模型支持 `model.dump_model(format="text"|"json")`，可输出全部树或
用 `tree_index=0` 指定第一棵树。`model.explain(sample)` 返回完整路径与各棵树
的贡献，可按列顺序提供 `feature_names`，默认名称是 `f0` 等。C++ 提供
`DumpModel` / `ExplainPrediction`，CLI 提供 `xgbt dump` / `xgbt explain`。
详见 [新手解释指南](docs/explain.zh-CN.md) 的可运行示例、分数语义和 JSON 字段。

## 调参和学习路线

| 参数（Python / C++） | 默认值 | 含义 |
| --- | ---: | --- |
| `n_estimators` / `num_trees` | 100 | 提升轮数 |
| `max_depth` | 4 | 路径上最多的划分次数；0 表示只有根节点 |
| `max_bins` | 64 | 每个特征的分位数分箱数，范围 2–65535 |
| `min_samples_leaf` | 5 | 每个子节点的最少样本数 |
| `learning_rate` | 0.1 | 步长，范围 `(0, 1]` |
| `reg_lambda` / `l2` | 1 | 非负叶节点 L2 正则 |
| `min_gain` | 0 | 非负最小划分增益 |

建议先运行小例子，再阅读[公式推导](docs/design.zh-CN.md)，最后修改学习率、
深度并观察损失和验证集 RMSE/AUC。训练损失下降不意味着泛化更好。
目前没有 early stopping，需要用验证集选择轮数，最终测试集保持不参与调参。

## 修改代码后验证

```sh
make test       # 原生和 Python 行为测试、性能与内存增长保护
make sanitize   # 独立 Debug 构建，ASan + UBSan + LSan
make coverage   # C++ 与 Python 行覆盖率达到 100%，并生成 HTML
clang-format --dry-run --Werror include/libtree/*.h src/*.cc tests/*.cc examples/*.cc benchmarks/*.cc
PYTHONPATH=python python benchmarks/run.py --repeats 3
```

TDD 流程：先写行为测试并确认失败，再实现最小修复，相关测试通过后重构。
已有测试包括独立的穷举 SSE 校验、缺失值、极端值、确定性、重复拟合、错误
参数与形状、C ABI 错误、Python 安装验证、反复释放及代表性工作负载。
覆盖率和内存工具提供额外证据，并不证明所有输入上都没有错误。
CI 运行检查和一次基准冒烟测试；仓库保存的性能结果采用三次独立重复。

## 对比与迁移

[benchmark README](benchmarks/README.md)提供中大型数据的最新 **C++ 和 Python**
对比、可复现命令和完整报告。固定划分上的效果和速度随任务变化；多线程方面，
LibTree 的平均 1→4 训练扩展优于 XGBoost，但仍低于 LightGBM。严格的 scaling
目标尚未达到。上述原生 GBDT 横向对比来自本地留出集；Mercari 的实际 Kaggle
评分单独记录，两者都不代表普遍排名。

本次重构删除了不安全的裸指针所有权、不完整的随机森林和 entropy 分支、
未经推导的 boosting 权重及未经校验的二进制模型读写。旧 `train`/`classify`
命令、头文件和模型格式不兼容，需通过新 API 重新训练。原 `data/` 文件保留为
历史数据。原生 GBDT 仍不支持多分类、排序、稀疏矩阵、GPU/分布式训练或样本权重；
C5 分类器提供多分类与样本权重，三个 Python 模型均支持快照。Mercari 的稀疏文本处理
属于独立的上层融合管线。贡献前请阅读[设计决策和风格例外](docs/design.zh-CN.md)。

[Validation evidence / 验证记录](docs/validation.md)

[新增 Kaggle 五数据集、多划分测试报告](benchmarks/KAGGLE_REPORT.zh-CN.md)

[交互 HTML 图表报告](benchmarks/KAGGLE_REPORT.html)（离线查看，切换 C++ / Python，导出原始数据）

## CPU 多线程

Python 使用 `GBDTRegressor(n_jobs=4)` 或 `GBDTClassifier(n_jobs=4)`；C++ 设置
`Parameters::num_threads = 4`；命令行训练、预测均可加 `--threads 4`。
默认 1，范围 1–256，包含调用线程。线程池复用工作线程，训练按特征并行，
推理按行并行，小任务自动串行。Boosting 各轮和树节点仍按依赖顺序执行，
已测试线程数下的模型文件、训练损失和预测值逐项一致。

已拟合的 Python 模型可用 `set_params(n_jobs=2)` 修改执行线程数，C++ 使用
`SetNumThreads(2)`。线程数是运行配置，不写入模型文件；加载模型可另外选择线程数。
支持同一模型并发预测，共享线程池的任务分发会排队；不能同时拟合、关闭或
修改该模型的线程数。外层交叉验证或多个模型并发时，应按 CPU 配额分配线程。
直接链接静态库时需加 `-pthread`；CMake 和 Python 安装已自动处理。

[同线程三方实测](benchmarks/PARALLEL_REPORT.zh-CN.md) · [交互扩展性图表](benchmarks/PARALLEL_REPORT.html)

```sh
PYTHONPATH=python python benchmarks/parallel.py \
  --datasets insurance bank_marketing friedman_20k friedman_100k \
  --threads 1 2 4 --seeds 42 2024 2026 --repeats 3
python benchmarks/parallel_report.py --input benchmarks/parallel-results.json
make thread-sanitize
```
