# Mercari：有限 CPU 预算下的特征、模型互补与直接 RMSLE 优化

[English](CPU_STUDY.md) · [初学者指南](BEGINNER_GUIDE.zh-CN.md) · [已有 CPU 打榜结果](TECHNICAL_REPORT.zh-CN.md)

当前约束为 **只用 CPU、明确限制计算时间**。此前排队中的 GPU 重训已在备份源码后移除，
监控与自动提交均已关闭。原 **0.380907** 本地完整融合仍保留作历史依据，不能改称 CPU 成绩。
实际最佳 Kaggle 练习成绩仍是 **Private 0.39199／Public 0.39110**。
采用冠军的方法，不等于已经取得第一名。

## 冠军经验与本轮选择

Jankiewicz 与 Lopuhin 的[冠军代码库][winner]公布了 12 模型融合：
多种文本视图、Huber／分箱分类、二值特征，最后用带截距的 Lasso 融合。
它的 **0.3733 是作者自己的验证成绩**，与我们的切分不同，不能直接横向排名。
[原始演示文稿][modeling]强调稀疏 MLP、稀疏更新、逐轮增大 batch，以及每核训练一个模型；
同时把网格搜索、CNN/RNN、FM 和残差拟合列为他们的失败尝试。
这些是具体实验的经验，不是普遍定律；LibTree 之前的 CPU 实验确实受益于 FM 与残差树。

本实现独立编写，不复制冠军权重，不下载预训练向量，不分配 GPU。
当前逐个训练候选并限制线程数；**尚未实现每核一个模型的调度，也不宣称这种调度的加速比**。
先在小预算下筛选有互补价值的配置，再决定完整重训值得投入多少时间。

## 特征与候选池

| 视图 | 内容 |
| --- | --- |
| 原始对照 | 原有标题／描述 TF-IDF、字符 n-gram、类别交叉 |
| 精确增强 | 跨字段标记、品牌 × 标题词、末级类别 × 标题词、原始标题字符、型号／容量／状态信号 |
| 词干增强 | 词干化单词，同时保留原始字符、交叉与商品信号 |

例如 `model_iphone6`、`unit_64gb`、`flag_accessory`、`flag_box_only`、
`flag_damaged`、`flag_bundle`。这些只是文本证据，不是验证过的商品事实。
`64 GB` 与 `64gb` 统一；不同容量仍然区分。每条商品最多取前 24 个标题词生成交叉，信号数量有限。
字符通道保留词干处理可能抹掉的型号细节。[词干算法来源][snowball]。
词表、词频和已知品牌字典只在基础训练集拟合；改变价格、ID 或旁边的查询样本不应改变特征。

三种视图分别训练 alpha=3／10 的 Ridge；两种增强视图各训练三种 width=128 的稀疏 MLP：
TF-IDF + MSE、二值特征 + Huber、二值特征 + 64 价格分箱，共 **12 个新增基础模型**。
batch 按 1024／2048／4096 增长，最多三轮，复用已有行稀疏 Adam。
不同目标同时使用不同 seed；这不是严格隔离其他变量的 loss 因果实验。

校准阶段比较 LibTree 深度 3／5、LightGBM L2／Huber、XGBoost 深度 4，五个模型分别提供
0.5／1 两档修正强度，不重复训练。加上可选旧 CPU 基线后共 **23 列外层预测**，不是 23 次独立训练。
树模型共享校准输入与残差目标，但容量不同；单次阶段时间不构成公平的性能排名。
新增校准器使用基础训练集拟合的目标统计、商品属性和 12 个新增专家预测。
旧基线的校准器已经见过校准标签，因此**整个旧基线分支不进入新增校准器的输入或残差目标**。

沿用原 seed-42 分组切分：基础训练 1,186,026；校准 74,129；选型 74,126；复核 148,254。
本轮只抽取基础训练中的 20 万条，其他窗口不变，因此新模型之间的特征比较使用相同训练样本。
旧 CPU 基线训练于全部基础训练数据，只是增量系统的参考，不能作为相同训练规模的单模型对照。
它的训练与已有预测成本不计入本轮计时，完整部署预算必须另加。没有使用缓存的 GPU 专家。

## 直接优化指标，避免把新名字当作新收益

对真实非负价格 `y`，先令 `z=log1p(y)`，模型输出 `s=P@w+b`：

```text
预测价格 = expm1(max(s, 0))
RMSLE² = mean((max(s, 0) - z)²)
优化目标 = RMSLE² + λ (||w-anchor||² + b²)
w ≥ 0，sum(w)=1，-0.25 ≤ b ≤ 0.25，λ=1e-4
```

平方根单调，所以不加正则时最小化 RMSLE 的平方与 RMSLE 的最优点相同。
Ridge 和 MSE 模型原来就直接学习对数价格的平方误差；Huber 和分箱分类是互补的**替代目标**，
是否保留看验证结果，不能仅凭训练 loss 更小就宣称改善了 RMSLE。

设 `e=max(Pw+b,0)-z`，`a=1[Pw+b>0]`，数据项对权重和截距的解析梯度分别为
`2*P.T@(e*a)/n`、`2*mean(e*a)`，再加正则项梯度。零点取零次梯度。
从预先定义的单模型／均匀融合中选一个合理初始点，避免全部负预测被裁剪后梯度为零。
预测全正时是通常的凸二次问题；不宣称裁剪目标的一般全局最优保证。
测试用独立官方 RMSLE 计算核对目标值，并用数值差分核对梯度，覆盖负预测和零边界。

**项目原 `blend_weights` 已经优化裁剪后的 log MSE**，并非这轮才发现正确指标。
新增价值是解析梯度、带边界的截距与裁剪起点保护；无截距消融用于区分实际校正收益。
没有必要改成原始美元误差，再把它包装成更直接的 RMSLE 优化。

权重、截距和所有消融配置先写入 `frozen.json`，随后才读取复核标签。
选型集反复用于模型、轮次、权重选择；复核集此前已经查看过，是复用审计，不是新的盲测。
基础模型的目标／视图消融同时排除所有新增校准器和旧基线，避免隐藏的上游依赖仍在工作。
这仍是分组消融，不是单因素因果结论；不同消融收益不能相加。[Stacking 原理][stacking]。

## 复现与预算

安装并准备数据：

```bash
python -m venv .venv
.venv/bin/pip install '.[mercari,benchmark]' pytest pytest-cov

# 先按已有 CPU 实验说明重现可信的基线模型。
.venv/bin/python -m benchmarks.mercari_frontier prepare \
  --data-dir benchmarks/cache/mercari \
  --baseline benchmarks/runs/mercari/optimized-audit-001/model.pkl \
  --output-dir benchmarks/runs/mercari/cpu-data

.venv/bin/python -m benchmarks.mercari_cpu_study \
  --data-dir benchmarks/runs/mercari/cpu-data \
  --output-dir benchmarks/runs/mercari/cpu-budget-study \
  --rows 200000 --threads 4 --max-seconds 900 \
  --max-features 50000 --char-features 30000 --cross-features 30000 \
  --width 128 --epochs 3 --tree-rounds 200
```

基线重现见[此前实验命令](OPTIMIZATION.zh-CN.md)。准备数据和旧基线训练不在本轮时间上限内。
也可在同样的分组文件上加 `--no-reference`，不读取 `baseline.npz`。
复核特征文件必须没有 `price` 列；各窗口的 ID 和相同商品组不允许交叉。
输出目录必须为空。`--rows 0` 使用全部基础训练数据，需要另行声明更大的时间预算。

启动器在子进程导入训练框架前隐藏 GPU，并限制 BLAS／OpenMP／TensorFlow 线程。
超过时限就终止训练子进程，返回 124，写入 `budget-exhausted.json`；
**未完成的模型与部分文件不算有效结果，不得晋升**。该命令从不上传或自动提交。
完整价格重训、单个 CPU Kaggle Kernel 执行和实际新成绩还需要单独验证。

成功输出包含冻结配置、`report.json`、带顺序 ID 的预测、特征转换器和模型。
只加载可信 pickle；训练数据、模型与原始预测保持 Git 忽略，只发布小型证据文件。
单次训练阶段保留原始秒数；有独立重复时使用平均值，而不是中位数。
连续训练轮次不是独立计时重复，不能据此宣称平均推理加速。

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m pytest tests/test_mercari_cpu.py tests/test_mercari_cpu_study.py \
  --cov=benchmarks.mercari_cpu --cov=benchmarks.mercari_cpu_study \
  --cov-report=term-missing --cov-fail-under=100
```

测试实际运行 Ridge、稀疏 MLP、原生树与竞品树，检查序列化、梯度、无泄露特征、复核 ID、
CPU 参数及硬超时。覆盖率仅指这些模块的语句，不代表分支、GPU 路径或整个代码库。

[winner]: https://github.com/pjankiewicz/mercari-solution
[modeling]: https://github.com/pjankiewicz/mercari-solution/blob/master/presentation/5_modeling.tex
[snowball]: https://snowballstem.org/algorithms/english/stemmer.html
[stacking]: https://doi.org/10.1016/S0893-6080(05)80023-1
