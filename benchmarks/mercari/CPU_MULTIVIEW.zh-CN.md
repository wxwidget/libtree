# CPU 特征工程与多参数、多视角融合

[English](CPU_MULTIVIEW.md) · [上轮 CPU 筛选](CPU_STUDY.zh-CN.md) · [初学者指南](BEGINNER_GUIDE.zh-CN.md)

本轮复用已在完整基础集训练好的 CPU 专家，新增商品统计与校准模型，不再训练一批较弱的
20 万样本网络。没有使用 GPU 或外部向量。原 GPU 辅助方案的本地 **0.380907** 继续保留；
实际最佳练习成绩仍为 **Kaggle Private 0.39199／Public 0.39110**。
本地实验不代表新的榜单成绩，也不代表已达到第一名。

## 新特征

五种商品视角：规范化完整标题；去掉简单促销词后排序的身份词；前三个身份词；
品牌 × 完整类别 × 成色；品牌／类别 × 数字型号和容量词。
它们只是近似商品身份，不能当作真实商品编号；配件、损坏等信息仍由独立文本属性保留。
每种视角输出对数价格均值、样本支持度和离散程度。

稀有商品的均值不能仅由一两条报价决定。[目标统计平滑的来源][encoding]：

```text
全局视角均值 = (组内价格对数之和 + α × 全局均值) / (组内数量 + α)
父级均值 = (品牌／类别价格对数之和 + 20 × 全局均值) / (父级数量 + 20)
层次视角均值 = (组内价格对数之和 + α × 父级均值) / (组内数量 + α)
```

原全局 α=20；续轮新增全局 α=5／100，父级 α=5／20／100。
没有见过的商品回退到已拟合的品牌／类别；连父级也未知才使用全局先验。
只复用基础训练集的充分统计，不读取新标签或重新建立文本词表。
离散程度使用相同逻辑平滑二阶矩。这里是固定参数的独立适配，不宣称新的贝叶斯推导。

把旧 CPU 管线拆成 Ridge、FFM 和四个神经网络分支；一次文本预处理和 CSR 矩阵提取供各分支共享。
单独重构旧整体预测，并与缓存逐行校验。旧校准器见过校准标签，因此它修正后的整个分支
不进入新增校准器的特征或残差目标，只能作为外层候选。
上轮两个词干网络也只进入外层，消融时不存在隐藏的校准依赖。

| 特征视角 | 内容 | 本轮维数 |
| --- | --- | ---: |
| 基础 | 原元数据、六个专家输出、五个预测分歧统计 | 40 |
| 文本属性 | 基础 + 商品／状态词属性 | 58 |
| 商品统计 | 文本属性 + 全局 α=20 的五组统计 | 73 |
| 多强度全局 | 商品统计 + 全局 α=5／100 | 103 |
| 层次多强度 | 多强度全局 + 父级 α=5／20／100 | 148 |

首轮九个校准器比较 LibTree 深度 3／5、LightGBM 31 叶、LightGBM Huber、XGBoost 深度 4。
续轮七个校准器增加 LightGBM 63 叶和仅用全局多强度特征的匹配对照。
每个模型复用 0.5／1 两档修正强度，无需重训。这是预先限定的特征／参数组合，
不是穷举搜索，也不是相同容量的速度排名。

## 融合如何防止过度灵活

直接优化裁剪后的对数价格平方误差，并加正则。比较三种预先定义的方法：

1. 非负、和为 1 的融合，截距有边界，正则 1e-4。
2. 非负仿射融合，权重和在 [0.8,1.2]、截距 ±0.25，正则 0.001。
3. 大类别单独融合，至少 2,000 条拟合样本，权重以 0.01 正则收缩到全局；小类别／未知类别回退。

这些限制在**对数预测与系数空间**，不表示美元价格只会变动 20%。
保留原分组窗口：基础 1,186,026，校准 74,129，选型 74,126，复核 148,254。
在选型集内部再按组对半切分，seed=193，37,062／37,064 行比较融合策略。
差异不超过 1e-5 时优先简单方法；确定方法后用完整选型集重拟合权重，再冻结复核。

这是复用选型集内部的**诊断切分**：旧专家的 checkpoint／配置此前使用过选型标签，
因此不能宣称全新的独立验证。复核窗口也曾经看过。配对 bootstrap 仅描述这些固定模型的
条件采样波动，不能消除选型偏差，也不能代表不同训练 seed 的稳定性。[Stacking 来源][stacking]。

续轮保留首轮在选型诊断中选中的融合类型。只有在相同诊断窗口改善超过 1e-5 才晋升，
否则保留旧池。提案、晋升决定和权重都在复核前写盘。
层次特征消融移除所有依赖它的校准器，保留独立拟合的全局对照，不是在训练后简单隐藏几列。

## 复现

先按[上轮指南](CPU_STUDY.zh-CN.md)准备可信 CPU 基线和分组数据。
原专家的完整训练成本单独计算；本轮计时仅包括复用预测、统计、新校准与融合。

```bash
python -m venv .venv
.venv/bin/pip install '.[mercari,benchmark]' pytest pytest-cov

.venv/bin/python -m benchmarks.mercari_multiview_study \
  --data-dir benchmarks/runs/mercari/cpu-data \
  --baseline benchmarks/runs/mercari/optimized-audit-001/model.pkl \
  --small-experts benchmarks/runs/mercari/cpu-budget-study \
  --output-dir benchmarks/runs/mercari/multiview \
  --threads 4 --tree-rounds 300 --max-seconds 900

.venv/bin/python -m benchmarks.mercari_multiview_study \
  --data-dir benchmarks/runs/mercari/cpu-data \
  --baseline benchmarks/runs/mercari/optimized-audit-001/model.pkl \
  --refine-from benchmarks/runs/mercari/multiview \
  --output-dir benchmarks/runs/mercari/multiscale \
  --threads 4 --tree-rounds 300 --max-seconds 600
```

续轮只使用剩余预算；本次两轮的时间上限之和小于 900 秒。
没有小样本网络缓存时可省略 `--small-experts`。输出目录必须为空。
启动器隐藏 GPU、限制线程、超时终止，不自动提交。
续轮校验基线、上一轮池、输入文件的哈希及顺序 ID；只加载可信 pickle。
商品映射包含基础集衍生统计，数据、模型、池和原始日志均保持 Git 忽略。

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_DISABLE_JIT=1 \
.venv/bin/python -m pytest tests/test_mercari_multiview.py \
  tests/test_mercari_multiview_study.py --cov=benchmarks.mercari_multiview \
  --cov=benchmarks.mercari_multiview_study --cov=benchmarks.mercari_multiscale \
  --cov-report=term-missing --cov-fail-under=100
```

覆盖率仅指这三个模块的语句，不等于分支或整个代码库覆盖。
测试包括独立统计公式、父级回退、数字类别、浮点边界、真实专家重放、原生／竞品校准、
哈希／标签隔离、先冻结后复核，以及 CPU／超时行为。

[encoding]: https://doi.org/10.1145/507533.507538
[stacking]: https://doi.org/10.1016/S0893-6080(05)80023-1
