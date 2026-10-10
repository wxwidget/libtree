# 研究引用与实现对应关系

[English](references.md)

本页列出 LibTree 设计参考的已发表研究。引用某个算法不代表 LibTree
完整复现了论文方法，也不代表获得了论文中的加速结果。表格说明本仓库实际做法和差异。

## 主要来源

1. Tianqi Chen、Carlos Guestrin， “XGBoost: A Scalable Tree Boosting System”，
   *第 22 届 ACM SIGKDD 知识发现与数据挖掘国际会议论文集*，2016，785–794 页。
   [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)，
   [作者预印本](https://arxiv.org/abs/1603.02754)。
2. Guolin Ke、Qi Meng、Thomas Finley、Taifeng Wang、Wei Chen、Weidong Ma、
   Qiwei Ye、Tie-Yan Liu， “LightGBM: A Highly Efficient Gradient Boosting
   Decision Tree”，*神经信息处理系统进展第 30 卷*，2017，3146–3154 页。
   [NeurIPS 论文](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)，
   [作者预印本](https://arxiv.org/abs/1711.07230)。
3. Google，[Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html)。
   这是代码风格规范，不是性能或算法结论。

## 论文方法与 LibTree 实现

| 已发表的方法 | LibTree 实现 | 明确差异与限制 |
| --- | --- | --- |
| 二阶梯度树评分、正则化叶值、考虑缺失值的划分搜索 [1] | `Stats` 累加梯度、Hessian 和数量；`FindBestSplit` 计算前缀分数并评估缺失值左右分配 | 仅支持稠密 `float32` 输入；没有外存分块、分布式/GPU 训练或加权分位数 sketch |
| 缓存友好的数据结构及稀疏感知划分 [1] | 借用行优先输入、紧凑 `uint16` 分箱、连续列分箱和高频分箱例外路径 | 稀疏优化作用于稠密输入，不是稀疏矩阵 API；没有实现 XGBoost 的缓存块或近似 sketch |
| 直方图构造、父子直方图差分及特征/数据并行策略 [2] | 构造较小子节点直方图，再用父直方图相减得到兄弟节点；多个提升轮复用分箱；大任务并行独立特征/节点工作 | 递归路径上的单棵树划分依赖仍是串行的；直方图差分减少计算不等于保证并行加速 |
| 单边梯度采样（GOSS）和互斥特征捆绑（EFB）[2] | 尚未实现 | GOSS 会改变采样和权重近似；EFB 依赖稀疏且互斥的特征。引入前都需要单独验证效果、内存和速度 |
| 受叶子数量约束的 leaf-wise 生长 [2] | 当前公共学习器限制最大深度，构树为深度优先/按层并行 | 不宣称实现了 LightGBM leaf-wise 模型及其质量/速度权衡。新策略需要独立 API、序列化兼容性和相同叶子数对照实验 |
| 特征/数据并行直方图工作 [2] | 有界 C++ 执行器并行执行特征准备、符合条件的直方图、树层、导数更新和预测批次 | 记录的中大型数据集上，LightGBM 的 1→4 训练扩展仍更好；见[实测基准](../benchmarks/README.md) |

## 工程取舍

最可迁移的组合不是“复制两个库”，而是保留 XGBoost 的二阶划分目标和
明确的数据局部性设计，并结合 LightGBM 的紧凑直方图、较小子节点差分和
并行任务管理。LibTree 通过确定的累加顺序、RAII 临时空间、有界线程池、
小任务串行回退和可移植 CPU 核心实现这些设计。

实测优势范围有限且写明边界：历史五数据集单线程快照记录了 LibTree 更快的推理；
最新的 1→4 训练扩展显示，LibTree 在已测中大型任务上的平均值高于 XGBoost。
LightGBM 的平均训练扩展仍更高。这些结果不能证明某个单独优化是原因，也不是普适排名或
Kaggle 榜单成绩。数据、方法、算术平均数、波动、版本和硬件见
[benchmark README](../benchmarks/README.md) 和
[实验报告](../benchmarks/EXPERIMENTS.md)。

## 可复现研究实践

采纳新方法前，先说明其机制和前提，添加独立行为测试，再用相同数据哈希、划分、
参数和 CPU 预算做单项消融。记录效果、训练/推理时间、峰值内存、源码版本和原始
重复数据。只有权衡有用时才保留；组合优化的收益不能在没有独立消融时归因到某个内核。
