# 输出决策树，解释一条预测

[English](explain.md) · [交互式 GBDT 入门指南](LEARNING_GUIDE.html)

决策树每次问一个问题，例如“年龄是否小于等于阈值”。成立走左分支，不成立
走右分支。遇到 NaN 时，走训练时学到的缺失值分支。叶子保存对模型分数的
修正值，一棵树负责一部分修正，整个 GBDT 累加全部树的贡献。

现在可以输出真实模型的文本或 JSON，查看全部树或指定某棵树，也可以解释
单条样本在每棵树上的路径、落到的叶子及其贡献。C++、Python 和 CLI 共用
同一个原生实现，模型文件仍使用原有 `LIBTREE_GBDT 1` 格式。

## 五分钟上手

```python
from libtree import GBDTRegressor

with GBDTRegressor(n_estimators=1, max_depth=1, min_samples_leaf=1,
                   learning_rate=0.5, reg_lambda=0) as model:
    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    print(model.dump_model(feature_names=["年龄"]))
    explanation = model.explain([3], feature_names=["年龄"])
    print(explanation["base_score"], explanation["margin"], explanation["prediction"])
    print(model.explain([3], feature_names=["年龄"], format="text"))
```

此人工例子的基础分数是 `0.5`，样本 `[3]` 对应的树贡献为 `0.25`，预测是
`0.75`。它用于讲解原理，不是效果评估数据。

- `dump_model(format="text"|"json", tree_index=None, feature_names=None)`
  返回字符串。JSON 用 `json.loads` 解析；用 `Path.write_text` 写文件。
  `None` 输出全部树，`tree_index=0` 输出第一棵树。
- `explain(sample, feature_names=None, format="json")` 默认返回字典；
  `format="text"` 返回字符串。形状必须是 `(特征数,)` 或 `(1, 特征数)`，
  第一版只解释一条样本。
- 特征名按输入列顺序提供，数量与特征数相同。默认名称是 `f0`、`f1` 等。
  名字只是显示标签，不会从 CSV 表头自动推断，也不保存到模型文件。
  支持中文、引号和换行转义；Python 名称不能包含 NUL 字符。

输入与预测一样转为 float32。错误形状、无限值、非法树索引、格式或未训练/
已关闭模型会报错；分数或 float32 预测溢出也会报错，避免非法 JSON。
检查是只读操作，不要同时重训或关闭同一模型。

## 怎样理解分数与路径

```text
某棵树的贡献 = learning_rate × leaf_value
margin = base_score + 第 0 棵树的贡献 + 第 1 棵树的贡献 + ……
```

回归的 `margin` 是转换为 float32 前的预测。二分类的 `margin` 是对数几率
（log-odds），经过 sigmoid 才是类别 1 的概率。解释中的 `prediction` 对应
`predict_proba([sample])[0, 1]`（sample 为一维样本），分类器的 `predict` 则按 `0.5` 阈值返回类别。
负贡献表示降低分数，不是“负概率”。

`condition` 表示 `<=`、`>` 或 `missing`；`direction` 是实际方向，
`missing_direction` 是该节点学到的缺失方向。NaN 在 JSON 的 `feature_value`
中写为 `null`，不能当作 0。`next_node_id` 指向下一节点，`leaf_id` 是最终叶子。
没有分裂的树路径为空，但仍有叶子贡献。

这是精确路径与**逐棵树贡献**，不是 SHAP，也不是逐特征贡献。路径出现某个
特征不代表它有因果作用。目前不输出未保存的节点样本数、分裂增益等统计。

## CLI 与 C++

```sh
./build/xgbt train --train examples/csv/train.csv --header \
  --model build/example.lt --trees 2 --depth 2 --min-leaf 1
./build/xgbt dump --model build/example.lt --format text
./build/xgbt dump --model build/example.lt --tree 0 --format json \
  --output build/tree-0.json
printf '0,1\n' > build/one-row.csv
./build/xgbt explain --model build/example.lt --data build/one-row.csv \
  --format json --output build/explanation.json
```

`explain` 的 CSV 必须恰好一条纯特征记录，有表头时用 `--header` 跳过。
两个命令都支持 `--feature-names names.txt`，每行一个 UTF-8 特征名。
`dump` 支持 `--tree`，`explain` 始终包含全部树；未提供 `--output` 时 stdout
只输出检查文档。文件原子替换，失败保留原内容，不能覆盖输入。

C++ 用 `model.DumpModel(output, libtree::DumpFormat::kJson, -1, names)`
导出全部树；把 `-1` 换为 `0` 可选第一棵。用
`model.ExplainPrediction({sample, 1, feature_count})` 得到结构体，包含基础分数、
全部树的路径与贡献、最终分数和预测。还可传输出流与格式，直接输出解释。
完整 C++ 示例和 C ABI 缓冲区约定见 [英文指南](explain.md)。

## JSON 与验证

JSON 使用 `schema_version=1`。公共字段是 `objective`、`num_features`、
`tree_count`、`base_score`、`learning_rate`、`feature_names` 和 `trees`。
`tree_count` 始终是整个模型的树数，即使只输出一棵树。编号从 0 开始，筛选
树不改变编号；解释通过全部树贡献重建完整预测。

树的 `nodes` 保存阈值、左右子节点和缺失方向，或叶子值与贡献。解释另含
`margin`、`prediction`，每棵树有 `path`、`leaf_id`、`leaf_value`、`contribution`。
检查文档不能当作模型加载；持久化请用 C++ `SaveModel` / `LoadModel`。

TDD 先验证接口不存在时测试失败，再实现。测试从导出的树独立遍历样本，
累加贡献，核对真实回归预测和二分类概率，并覆盖阈值等号、左右缺失方向、
特殊字符、无分裂与深层树、旧模型、缓冲区不足和错误时输出保护。
