# C5 风险建模：优化方法与评估

[English 与完整引用](c5-risk.md) · [风险实测报告](../benchmarks/RISK_REPORT.zh-CN.md)

已有 C5 使用 SAMME boosting。这轮增加显式开关、类别平衡权重、Hellinger
分裂准则与训练集内的交叉验证选参流程；不根据测试结果直接更改默认准则。

| 功能 | 参数／方法 | 研究来源 |
| --- | --- | --- |
| Boosting 参数化 | boosting=none/samme，树数、学习率 | Zhu 等，2009，SAMME |
| 类别平衡 | class_weight='balanced'，逆频率权重 | He & Garcia，2009，不平衡学习 |
| Hellinger | criterion='hellinger'，二分类条件分布距离 | Cieslak 等，2012 |
| 训练集内选参 | 三折 CV，方向性 KS，最小提升门槛 | Cawley & Talbot，2010 |

代码中的 SAMME、Hellinger、类别权重、选参和 KS 计算旁都附有引用。
这是独立实现，不表示与原版 C5.0 全面兼容。

Hellinger 比较两类分别进入左右分支的条件概率：

```text
sqrt((sqrt(L0/T0)-sqrt(L1/T1))² + (sqrt(R0/T0)-sqrt(R1/T1))²)
```

它消除分裂评分中直接的类别先验因素，但整个模型仍受加权叶子决策、剪枝、
分位阈值与 boosting 影响；不保证任何不平衡任务都更好。只支持二分类
（允许退化的单类别拟合），多分类输入拒绝且保留旧模型。该准则下 min_gain
以 Hellinger 距离为单位；计算路径无需对数。

平衡权重为 `样本数/(类别数×该类数量)`，只从拟合标签计算，再与用户给的
sample_weight 相乘，输入不变。增加坏样本召回可能同时增加误报；应一起检查。
这些权重不是业务损失成本，模型概率仍未校准。

```python
import numpy as np
from libtree import C5Classifier
from sklearn.model_selection import train_test_split
from benchmarks.risk import dataset, encode  # 从仓库目录执行。

frame, labels, source = dataset('german_credit')
frame = frame[['duration', 'amount', 'age']]  # 教学数值子集；benchmark 使用全部合法特征。
names = sorted(frame.columns)  # encode 的数值列按名称排序。
a, b = train_test_split(np.arange(len(labels)), test_size=.25,
                        stratify=labels, random_state=42)
X_train, X_test = encode(frame.iloc[a], frame.iloc[b])
y_train = labels[a]
with C5Classifier(boosting='samme', n_estimators=100, learning_rate=.3,
                  criterion='hellinger', class_weight='balanced',
                  max_depth=4, pruning_alpha=.001, n_jobs=4) as model:
    model.fit(X_train, y_train)  # 风险标签 1=坏
    scores = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    print(model.explain(X_test[0], feature_names=names, format='text'))
```

该配置是候选，不是通用推荐。boosting='none' 恰好拟合一棵树；请求的树数仍须
合法。默认保留 SAMME、学习率 1、不平衡加权关闭。get_params/set_params
支持新参数，下一次 fit 生效。原生 C++ 设置 num_estimators=1 关闭 boosting，
设置 kHellinger 使用新准则；逆频率权重传给 Fit。C ABI 原来的准则 ID 0/1/2
保持不变，新增 ID 3。native benchmark 与 Python 对齐权重并检查预测一致性。

```sh
PYTHONPATH=python python -m libtree rules --train credit.csv --header \
  --boosting samme --rounds 100 --rate .3 --criterion hellinger \
  --class-weight balanced --depth 4 --alpha .001 --threads 4
PYTHONPATH=python python -m libtree rules --train credit.csv --header --boosting none
```

## KS 与实验规则

统一 1=坏／风险事件，分数越高风险越大。标准 KS 是坏／好两类经验分布最大
绝对差；另给方向性 KS=max(TPR-FPR)。相同分数作为一组，不能拆开制造虚假
阈值。常量分数 KS=0；完全反向分数绝对 KS=1，却有方向性 KS=0、AUC=0。
因此必须同时看 KS 与 AUC。多分类 KS 未定义，省略该项。

测试集 KS 只是描述排序效果，不是用测试集选择上线阈值。原 C5 benchmark
与新的风险 benchmark 都增加二分类 KS；风险报告还给 PR/AP、坏样本召回、
平衡准确率、准确率、log loss、训练准确率、规则/条件数量与时间。

真实任务：德国信用风险、信用 good/bad 状态、台湾付款逾期。台湾公开子集用
4 月特征预测 9 月逾期，不是完整信用卡违约目标。标识列和目标列移除，下载
源有 SHA-256 固定校验。另加 2 万行稀少风险事件的合成数据，明确标注为合成。
随机分层测试不是时间回测，也不证明能迁移到别的机构。

外层三个随机种子，各自 75/25 训练/测试。十个预设 C5 配置只在外层训练集做
三折分层 CV，每折单独拟合类别编码。按平均方向性 KS、AUC、少规则的顺序
选择；相对 samme_30 的 KS 至少提升 .01 才切换，否则保留基线。选定后用
完整外层训练集重新拟合，最后才评估未参与选择的测试集。

消融分别变化 boosting 开关、轮数、学习率、权重、准则与剪枝。竞品固定
100 轮，不单独调参；selected C5 多用了搜索预算，报告会披露搜索时间和每折
成绩，不能把它解释为同等预算下的普遍第一。验证门槛也不能保证测试集一定提升。

后续可评估：重复／嵌套 CV 的规则稳定性；概率型 boosting（未实现 SAMME.R）；
独立验证集概率校准；按误报／漏报成本选业务阈值；时间切分与外部人群验证。
单调校准通常不提高 KS/AUC，阈值改善也不等于排序改善。保留基线与原始结果，
仅在新的验证证据支持时保留优化。
