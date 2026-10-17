# Python 可解释性与模型组合

[English](python-integration.md) · [树结构与预测路径](explain.zh-CN.md) · [C5 规则](c5.zh-CN.md)

三个模型 `GBDTRegressor`、`GBDTClassifier`、`C5Classifier` 均支持 sklearn
克隆、模型类型标签、拟合状态检查、Pipeline、网格搜索、置换重要性，
以及 Voting、Stacking、Bagging。相关集成都是可选依赖，基础预测仍只需要 NumPy。

```sh
python -m pip install '.[sklearn,explain]'
# 已发布的包名：pip install 'libtree-gbdt[sklearn,explain]'
```

## 用 SHAP 分析一条预测

预测路径回答“模型进行了哪些判断”；SHAP 回答“相对于指定背景人群，
各个特征使模型输出改变了多少”。前者使用 `dump_model`、`dump_rules` 和
`explain`；后者使用下面的统一接口。

```python
import numpy as np
from libtree import C5Classifier, GBDTClassifier, GBDTRegressor

rng = np.random.default_rng(42)
x = rng.normal(size=(120, 3)).astype(np.float32)
y = np.where(x[:, 0] + x[:, 1] > 0, "risk", "safe")

with C5Classifier(n_estimators=5, min_samples_leaf=2).fit(x[:100], y[:100]) as model:
    explainer = model.shap_explainer(
        x[:20], algorithm="permutation", seed=42,
        feature_names=["收入", "负债", "年龄"],
    )
    explanation = explainer(x[100:103], max_evals=70)
    print(explanation.values)       # 形状：样本数 × 特征数 × 类别数
    print(explanation.output_names) # 与 classes_ 顺序一致的类别名称
    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        model.predict_proba(x[100:103]), atol=2e-7, rtol=2e-7,
    )
    # 安装 matplotlib 后，可以画指定类别的单条预测瀑布图：
    # import shap
    # shap.plots.waterfall(explanation[0, :, 1])
```

两个 GBDT 模型使用相同的 `shap_explainer` 方法。回归解释的形状为
`(样本数, 特征数)`，单位与预测目标一致。分类解释的形状为
`(样本数, 特征数, 类别数)`，解释的是**概率**，二分类也保留两列。
“背景基准值 + 特征贡献之和”在浮点误差范围内还原回归 `predict` 或分类
`predict_proba`，不对字符串类别标签进行数值归因。C5 的概率仍是未经校准的
归一化投票 softmax 分数；可解释性不等于概率校准。

`algorithm` 支持 `auto`、`exact`、`permutation`。返回可调用的 `ShapExplainer`
适配器，调用结果是标准 `shap.Explanation`，可以交给 SHAP 绘图函数。
适配器的 `explainer` 属性是原始 SHAP 解释器。适配器通过公开的 Explanation
属性保留类别名称，避免部分 SHAP 版本在 Exact 分派时丢失名称。
使用 DataFrame 背景时，默认采用列名作为特征名。分类训练支持 pandas 字符串标签。

实现使用公开的 `shap.Explainer` 与 `shap.maskers.Independent` API，当前尚无
原生 TreeSHAP／`shap.TreeExplainer` 适配。所有拟合模型也可直接调用，因此支持
标准写法 `shap.Explainer(model, background, algorithm="permutation", seed=42)`：
回归返回预测值，分类返回全部类别概率；标准 SHAP 可能生成通用输出名称，
此时按 `model.classes_` 确认概率列对应的类别。

背景集应从**训练数据**中固定随机抽取，例如 20–100 条有代表性的记录。
适配器保留全部指定背景，不暗中抽样。Exact 枚举的成本随变化特征数指数增长；
特征较多时选择 Permutation，`max_evals` 至少为 `2 * 特征数 + 1`，
提高到该值的若干倍可减少采样波动。成本还随解释样本数、背景样本数增长。
贡献可以相加，还不足以证明归因已经稳定。

Independent 遮蔽会用背景中的特征值替换隐藏特征；存在相关特征时，可能产生
现实中罕见的组合。解释依赖所选背景，属于模型归因，不能直接作因果解释，
也不是条件 SHAP。支持 NaN 的原生路由，原生特征计算采用 float32 精度。

## 用 sklearn 组合模型

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from libtree import C5Classifier, GBDTClassifier, shap_explainer

# 接着使用上例的 x、y；每个分类器处理同一组二分类字符串标签。
voting = VotingClassifier(
    estimators=[("c5", C5Classifier(n_estimators=5, n_jobs=1)),
                ("gbdt", GBDTClassifier(n_estimators=20, n_jobs=1))],
    voting="soft", n_jobs=2,
).fit(x[:100], y[:100])

# 解释最终集成输出，包含投票平均或最后一层模型的实际影响。
values = shap_explainer(voting, x[:20], algorithm="exact")(x[100:103])
np.testing.assert_allclose(values.base_values + values.values.sum(1),
                           voting.predict_proba(x[100:103]), atol=2e-7)

stacking = StackingClassifier(
    estimators=[("c5", C5Classifier(n_estimators=3)),
                ("gbdt", GBDTClassifier(n_estimators=10))],
    final_estimator=LogisticRegression(), cv=3, n_jobs=2,
).fit(x[:100], y[:100])

bagging = BaggingClassifier(C5Classifier(n_estimators=3),
                            n_estimators=3, random_state=42, n_jobs=2)
pipeline = make_pipeline(StandardScaler(), bagging).fit(x[:100], y[:100])
```

回归使用 `GBDTRegressor` 搭配 `VotingRegressor`、`StackingRegressor` 和
`BaggingRegressor`。`shap_explainer(pipeline_or_ensemble, background)`
按原始输入特征解释整个组合模型。预处理应在各训练折内拟合，Stacking 使用
折外预测；增加集成层次并不保证效果或速度改善，应另设验证集和最终测试集。

| 能力 | GBDTRegressor | GBDTClassifier | C5Classifier |
| --- | --- | --- | --- |
| 克隆、Pipeline、网格搜索、置换重要性 | 支持 | 支持 | 支持 |
| Voting／Stacking／Bagging，含 `n_jobs=2` 多进程 | 支持 | 支持 | 支持 |
| 标签 | 数值目标 | 恰好两个有限数值或字符串类别 | 1–256 个有限数值或字符串类别 |
| `fit(sample_weight=...)` | 不支持 | 不支持 | 支持 |
| 外层 boosting | 支持基于重采样的 AdaBoostRegressor | 不支持 AdaBoostClassifier：它要求样本权重 | 支持 AdaBoostClassifier；建议 `boosting="none"`，每个基学习器只训练一棵树 |
| SHAP 预测值／概率的可加性 | 支持 | 支持 | 支持，含多分类 |

GBDT 分类器的 `classes_` 排序决定概率列的顺序，支持 `"risk"/"safe"`、
`2/7` 等标签。C++ GBDT／C ABI 继续使用编码后的 0/1 标签；原生路径 JSON
解释的是索引为 1 的编码正类。GBDT 分类训练的每个交叉验证折或自助采样都应
包含两个类别；C5 支持单类别训练。输入须为稠密数值矩阵，NaN 表示缺失。
目前不支持稀疏矩阵、字符串特征、多输出目标、高级 sklearn metadata routing
和自动样本权重路由。GBDT 不会静默忽略不支持的样本权重。

外层和内层线程数应一起考虑：外层 4 个进程、内层各 4 个线程可能产生最多
16 个计算线程。先让内层使用 `n_jobs=1`，再根据机器资源调整。
同一模型预测或序列化期间，不要并发执行 refit、close 或运行时线程数更新。

## 模型快照与进程安全

`pickle`、`copy.deepcopy` 和 joblib 多进程传输保存版本化原生模型文本、
Python 参数、类别与拟合元数据，排除原生指针、finalizer 和训练矩阵。
恢复时创建独立的原生资源；关闭原模型不影响恢复后的模型。预测、规则、
树 JSON、训练损失／基学习器误差均可往返保存，无需重新训练。
`set_params` 尚未生效的训练参数继续保留，不会改写已经拟合的树。
关闭或未拟合的模型经保存恢复后仍未拟合。

C++ C5 新增 `SaveModel`／`LoadModel`，验证文件并保证加载失败不破坏已有模型。
GBDT 改写为版本 2，保存训练损失；继续读取原有版本 1 文件。
C ABI 新增 `LtSerialize`／`LtDeserialize` 和对应的 `LtC5` 接口，缓冲区由调用方
查询长度后分配。运行时线程数在恢复时指定，不写入可移植模型文本。
快照限制原生节点总数不超过一百万，GBDT 树数不超过一万。
模型快照不含训练数据，但模型本身仍可能透露从数据学到的信息。

只加载可信的 pickle／joblib 文件，多个进程使用相同 LibTree 包版本。
Python pickle 不承诺跨版本兼容。关闭模型后，即使历史 `classes_` 属性仍然
存在，sklearn 的拟合状态检查也会正确失败。

## 复现验证

```sh
python -m pip install '.[dev,benchmark]'
make test
make coverage          # 要求 Python 语句、原生源码行覆盖率均达到 100%
make sanitize          # 地址、未定义行为和内存泄漏检查
make thread-sanitize   # 原生数据竞争检查
PYTHONPATH=python python -m pytest tests/test_integration.py -v
```

TDD 先暴露了 sklearn 标签和 C5 持久化接口缺失。测试覆盖三个模型的实际
多进程组合、多分类与 NaN SHAP 可加性、字符串标签、DataFrame 名称、可选依赖
缺失、非法背景集、序列化失败、refit 失败、独立资源与训练历史恢复。
这些是接口兼容性测试，本次不产生新的竞品效果／性能结论；历史基准报告继续
对应其标明的源码版本。

## 参考资料

- sklearn：[估计器开发规范](https://scikit-learn.org/stable/developers/develop.html)、
  [集成方法](https://scikit-learn.org/stable/modules/ensemble.html)。
- SHAP：[统一解释器](https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html)、
  [独立背景遮蔽](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html)、
  [精确枚举](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html)、
  [排列解释](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html)。
- Lundberg 与 Lee（2017），《A Unified Approach to Interpreting Model Predictions》，
  [NeurIPS 论文](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)。
