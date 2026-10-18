# Colab 小尺寸文本模型实验

[English](SMALL_TEXT.md) · [原文本实验](FRONTIER.zh-CN.md) · [初学者指南](BEGINNER_GUIDE.zh-CN.md)

2026 年 10 月 18 日重新授权使用 Colab CPU/GPU。这轮只使用一个 T4，
候选训练进程合计上限 3,600 秒，每个完整模型最多 800 万参数。
实例申请、上传、安装、下载和清理时间另计。旧 Kaggle 排队任务与 watcher
继续停用。历史最佳本地融合 **0.380906853 RMSLE** 保留为基准；
CPU 方案 **0.392845724** 是另一种成本取舍，不能当成最佳精度。

## 模型与控制变量

固定三个候选，不无限搜索：

| 候选 | 特征与损失 | 最多轮数 | 批量 |
| --- | --- | ---: | ---: |
| 子词词袋回归 | 48 维 WordPiece 向量、32,768 个哈希相邻词对桶；MSE | 4 | 2,048 |
| 子词词袋回归 | 相同结构；Huber | 4 | 2,048 |
| BERT-tiny | 2 层、128 隐藏维预训练 BERT；MSE | 3 | 512 |

统一序列长度 96、输出头宽度 128、CPU 线程设置 4、随机种子 42。
800 万参数上限包含类别嵌入与回归头；超限在批量编码和训练前报错。
Colab CPU 处理数据与分词，T4 负责混合精度训练和预测。

词袋模型借鉴 [Joulin 等人的 fastText 思路](https://aclanthology.org/E17-2068/)，
使用平均向量与哈希词对。这是独立的 PyTorch 价格回归实现，**不是官方
fastText 分类接口**；使用 WordPiece 子词，也不是 fastText 的字符 n-gram。
分词器预训练，回归词向量随机初始化。小 BERT 参考
[Turc 等人](https://arxiv.org/abs/1908.08962)，使用预训练骨干。
两者差异同时包括预训练和架构，不能据此单独归因于某一因素。

学习目标为标准化的 `log1p(price)`。MSE 对应平方对数误差；Huber 是抗异常值
备选，不等于直接最小化 RMSLE。模型还输入训练集拟合的品牌/类别嵌入、
新旧程度、运费、长度和产品词语标记。详细学习率和优化器设置见英文页。

## 验证与保留

训练/校准/选择/确认分别为 1,186,026 / 74,129 / 74,126 / 148,254 条。
轮数与融合权重只看选择集；确认集价格留在本地，先冻结再审计。
这个确认集过去已经看过，不能冒充全新盲测。参数量只代表一个候选，
不代表历史完整融合的总模型体积或服务内存。

`--include-models` 保留检查点、输入映射、预测 ID、每轮选择集指标、参数量、
耗时、版本和预训练版本号；清理只关闭本次创建的实例。数据、价格拟合权重
和原始预测不提交 Git。训练时间包含各候选重复的预处理，单次测量不能当成
稳定性能排名或扩展性证明。指标使用全体样本的平均平方对数误差，再开平方。

先测后写的本地验证已通过 22 项测试，新增 worker 与生命周期 launcher
共 202/202 条语句覆盖。覆盖率不是全部分支或 GPU 指令覆盖。CI 已加入
离线小模型真实训练测试。实验不会自动提交 Kaggle，只有完整评分回执
才能称为新榜单成绩。

## 复现

先按[原实验](FRONTIER.zh-CN.md)准备分组数据并完成官方 Colab CLI 授权。

```bash
python -m pip install '.[mercari-transformer]' pytest pytest-cov
python -m benchmarks.mercari_colab \
  --data-dir benchmarks/runs/mercari/frontier/data \
  --output-dir benchmarks/runs/mercari/smalltext-t4 \
  --cli colab --gpu T4 --profile smalltext --include-models
```

输出目录须为新目录。逐个候选的执行命令与测试命令见[英文页](SMALL_TEXT.md)。
CPU 机器可加 `--allow-cpu` 执行独立 worker，CUDA 可用时仍优先使用 GPU。
不要把 OAuth 凭证写进命令、代码或 Git。
