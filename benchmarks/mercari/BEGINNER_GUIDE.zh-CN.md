# 从商品描述到价格：初学者的 Mercari 实战指南

[English](BEGINNER_GUIDE.md) · [技术报告](TECHNICAL_REPORT.zh-CN.md) · [完整实验](OPTIMIZATION.zh-CN.md)

只需基础 Python 和 Linux 终端知识。先跑一个小练习，再决定是否下载官方百万行数据。
小练习**无需 Kaggle 账号或 token**；它使用人工生成的数据，只验证程序，不复现榜单成绩或性能。

## 1. 先理解任务和指标

一行数据描述商品：标题、描述、品牌、品类、成色与运费。**特征**是预测时能取得的信息；
**标签**是训练时已知的价格。训练让模型学习从特征到价格的关系，预测不能读取实际价格。

**训练**会改变模型参数，**推理**用已拟合模型计算结果；**损失函数**指导训练，
**评估指标**比较最终预测。这里用越小越好的 RMSLE：

```text
RMSLE = sqrt(mean((log1p(predicted_price) - log1p(actual_price))²))
```

`log1p(x)` 就是 `log(1+x)`，因此零价格合法。把 10 预测成 20，误差项约为
`log(21)-log(11)=.647`；把 100 预测成 200，约为 `.688`。远离零时，它更关注比例偏差，
不会让高价商品仅凭绝对金额差主导评价。它不是“准确率百分比”。这里提升 10% 的定义是
误差降至 `.43776 × .9 = .393984` 或更低。

## 2. 用直觉认识模型

| 概念 | 例子或含义 | 在本项目中的作用 |
| --- | --- | --- |
| token／n-gram | `iphone`、`6`、`64gb`，以及相邻词或字符 | 不丢失型号和容量 |
| 稀疏矩阵／CSR | 只保存出现的词 | 商品只含少量词，不保存所有词的零值 |
| Ridge 回归 | 对词语与品类作用做带正则的加法 | 可靠的线性基线 |
| 字段感知因子分解 | 某个词在不同品牌下价值不同 | 用小向量学习交互 |
| MLP／词袋池化 | 学习文本与类别的非线性组合 | 补充线性模型的不足 |
| MSE | 对数价格误差越大，平方惩罚越大 | 与主要指标较直接地对齐 |
| Huber | 小误差用平方，大误差近似线性 | 降低极大误差的影响 |
| 价格分档 | 给每个对数价格区间一个概率 | 提供另一种错误模式 |
| ensemble／融合 | 组合多个模型 | 错误互补时有益，也更耗时 |
| residual／残差 | 实际对数价格减去基础预测 | LibTree 需要修正的部分 |
| GBDT | 逐步添加小树，修正损失梯度 | 用商品信息和专家预测学习非线性修正 |

最终的计算为：

```text
base_log_price = weighted average of expert log predictions
final_log_price = base_log_price + LibTree correction
price = expm1(max(final_log_price, 0))
```

融合发生在**对数价格**上，不是直接平均美元金额。GBDT 内部叠加多棵树，外层融合
不同类型模型，这是两层不同的组合。不了解决策树时，先看可离线使用的
[交互学习指南](../../docs/LEARNING_GUIDE.html)。

例如修正 `.1` 会让 `price + 1` 乘以 `exp(.1)`，约为 1.105，这样就能直观理解
对数尺度的修正幅度。

## 3. 安装已知版本

使用 Linux 或 WSL2、**Python 3.12 和 C++17 编译器**。Mercari 扩展依赖比基础
LibTree 包严格。全量实测使用四 CPU 配额、32 GiB 内存上限，峰值约 10 GiB；
还需为压缩包、解压数据、缓存、模型和预测预留足够磁盘。小练习所需资源少得多。

若没有仓库，从其父目录运行：

```sh
git clone https://github.com/wxwidget/libtree.git
cd libtree
git checkout e711973
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '.[mercari]' pytest pytest-cov kaggle py7zr
python -c 'import libtree; libtree._library(); print("Native library loaded")'
```

后续均在仓库根目录、激活虚拟环境后运行。`e711973` 是已测量的发布快照，后加的这些
文档也可在 GitHub 阅读。`benchmarks` 是仓库内模块，需要 `PYTHONPATH=.`。
不要改成 `PYTHONPATH=python:.`，否则未编译的源码包可能遮蔽安装好的原生库。

## 4. 第一次练习：不用凭据的小数据

下面生成 240 条人工训练数据和 16 条预测样本，并把轻量版维度缩小、每个模型只训练一轮。
这不是独立测试集，不评价泛化能力。若 `tutorial-smoke` 已存在，请换新目录名，保留旧结果。

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("benchmarks/runs/mercari/tutorial-smoke")
root.mkdir(parents=True, exist_ok=False)
data = root / "data"
data.mkdir()
rng = np.random.default_rng(42)
rows = 240
condition = rng.integers(1, 6, rows)
brand_index = np.arange(rows) % 3
frame = pd.DataFrame({
    "train_id": np.arange(rows),
    "name": [f"phone {i % 8} case item {i}" for i in range(rows)],
    "item_description": [f"phone accessory colour {i % 5}" for i in range(rows)],
    "category_name": "Electronics/Phones/Accessories",
    "brand_name": np.array(["Acme", "Bravo", "Comet"])[brand_index],
    "item_condition_id": condition,
    "shipping": np.arange(rows) % 2,
    "price": 8 + 12 * brand_index + 4 * (6 - condition) + rng.uniform(0, 2, rows),
})
frame.to_csv(data / "train.tsv", sep="\t", index=False)
query = frame.iloc[:16].drop(columns="price").rename(columns={"train_id": "test_id"})
query["name"] = [f"phone x case new {i}" for i in range(len(query))]
query.to_csv(data / "test.tsv", sep="\t", index=False)
pd.DataFrame({"test_id": query.test_id, "price": 0}).to_csv(
    data / "sample_submission.csv", index=False)

config = json.loads(Path("benchmarks/mercari/fast-config.json").read_text())
config.update(threads=2, max_features=500, char_features=250, max_words=500,
              neural_width=8, fm_workers=1, fm_rank=2, fm_epochs=1,
              neural_rates=[.003], neural_batches=[32],
              sparse_word_limit=500, sparse_char_limit=250, sparse_min_df=1)
for member in config["neural_members"]:
    member.update(neural_width=8, neural_rates=[.003], neural_batches=[32])
config["native"].update(n_estimators=4, max_depth=2, max_bins=16,
                        min_samples_leaf=2, n_jobs=2)
(root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
```

将上面的 Python 保存为 `/tmp/mercari_tutorial_data.py`，用 `python` 执行。然后训练与预测：

```sh
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/tutorial-smoke/data \
  --config benchmarks/runs/mercari/tutorial-smoke/config.json \
  --output-dir benchmarks/runs/mercari/tutorial-smoke/output \
  --mode final --test-file test.tsv --sample-file sample_submission.csv
```

输出包括 `report.json`、`submission.csv`、`model.pkl`、`tree.txt`、`tree.json` 和
`explanation.json`。树文件仅导出**第一棵树**；报告保存参数、划分、哈希和耗时。
输出目录必须为空，避免覆盖之前的实验。

将以下代码保存为 `/tmp/mercari_tutorial_check.py`，运行
`PYTHONPATH=. python /tmp/mercari_tutorial_check.py`，检查重载和解释：

```python
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("benchmarks/runs/mercari/tutorial-smoke")
query = pd.read_csv(root / "data/test.tsv", sep="\t")
saved = pd.read_csv(root / "output/submission.csv")
with (root / "output/model.pkl").open("rb") as stream:
    model = pickle.load(stream)
try:
    log_price = model.predict(query)
    price = np.expm1(np.maximum(log_price, 0))
    np.testing.assert_allclose(saved.price, price, rtol=1e-6, atol=1e-6)
    explanation = model.explain(query.iloc[:1])
    total = (sum(explanation["weighted_expert_log_contributions"])
             + explanation["weighted_native_correction"])
    assert np.isclose(total, log_price[0])
    assert np.isclose(explanation["price"], price[0])
    assert saved.test_id.tolist() == query.test_id.tolist()
    assert len(saved) == 16 and np.isfinite(price).all() and (price >= 0).all()
    assert (root / "output/tree.txt").read_text().strip()
    assert json.loads((root / "output/tree.json").read_text())
    print("PASS: 16 predictions, reload, tree export and explanation sum")
finally:
    model.close()
```

成功时会输出 `PASS: 16 predictions, reload, tree export and explanation sum`。
只加载可信的 pickle，反序列化可能执行代码。

注意这里的 `predict()` 返回**对数价格**，要用 `expm1` 还原；解释中的专家贡献与树修正
也在对数尺度相加，不是金额贡献、因果效应或 SHAP 值。
[原生预测路径](../../docs/explain.zh-CN.md)与[SHAP 集成指南](../../docs/python-integration.zh-CN.md)
介绍原生估计器的解释；Mercari 整条文本管线不会自动变成 TreeExplainer 支持的模型。

## 5. 怎样避免把实验做错

**过拟合**是学到了训练样本细节，却在未见数据上更差。**泄漏**是让预测时拿不到的信息
进入训练或评估。重复商品跨集合、用验证文本拟合词表、用验证价格计算类别目标均值，
都会破坏本项目规定的验证边界。

区分基础训练、残差校准、参数选择和冻结后的留出审计。校准集参与训练，不能当独立测试。
不要一边看留出成绩一边改参数。**消融实验**是在比较范围固定的前提下删除或改变一个组件，
用来判断它贡献了什么；同时改了多个因素，就不能只归因于其中一个。

本项目值得记住的经验：

- 保留型号信息、学习字段交互，比盲目增加模型复杂度更有价值。
- 字段感知模型调参 RMSLE 第 2 轮为 .420152，第 3 轮却升至 .426392，训练误差仍在下降。
- 池化模型的 Huber 只比 MSE 略好，不能凭损失名称判断收益；融合看的是错误是否互补。
- 大批次在一个精度版试验中变快，却使轻量版变慢。应测量完整管线，并保留慢的重复值。
- 轻量版删掉昂贵组件后仍达到分数目标；旧基线的推理则仍显著更快。
- Kernel 完成、生成 CSV、真实提交评分 `COMPLETE` 是不同阶段。当前分数也不是原赛事名次。

[技术报告](TECHNICAL_REPORT.zh-CN.md)列出对照、失败尝试和取舍的证据。

## 6. 准备好后再下载官方数据

使用自己的 Kaggle 账号；若平台要求，在[比赛网页](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge)
接受适用规则。按 Kaggle 支持的方式或环境的安全密钥设置配置认证，不把 token 放进源码、
Git、命令参数或截图。平台注入的凭据未显示为明文，并不表示需要创建重复变量。

```sh
mkdir -p benchmarks/runs/mercari/data
kaggle competitions files mercari-price-suggestion-challenge
kaggle competitions download mercari-price-suggestion-challenge \
  -p benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/mercari-price-suggestion-challenge.zip \
  benchmarks/runs/mercari/data
python -m py7zr x benchmarks/runs/mercari/data/train.tsv.7z benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/test_stg2.tsv.zip benchmarks/runs/mercari/data
python -m zipfile -e benchmarks/runs/mercari/data/sample_submission_stg2.csv.zip benchmarks/runs/mercari/data
```

解压命令对应已归档的官方文件名。先检查下载清单：若平台直接下载单个文件，或者数据已解压，
跳过相应外层压缩包步骤。训练需要 `train.tsv`；最终预测必须使用配套测试文件与提交模板。
模板中的零不是测试标签。`403` 可能涉及认证、规则或访问要求，应读提示，不绕过限制。
完整文件 SHA-256 可在[原基线报告](full-001.json)核对。

## 7. 官方数据：训练、审计与最终预测

**study** 使用原来的基础／校准划分，报告选择集 RMSLE，默认不读留出成绩。
这已经是全量任务，不是小练习：

```sh
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-study --mode study
```

先冻结选择。要做一次**新的**留出审计，在新输出目录重复命令并加 `--audit-holdout`。
这会重新拟合模型，不是直接读取上一份模型。本项目公布的独立留出成绩属于精度版，
不能写成轻量版成绩；反复对审计结果调参也会破坏独立性。

**final** 将全部标签分配给基础模型或校准器，生成测试预测，但不产生本地验证成绩：

```sh
PYTHONPATH=. python -m benchmarks.mercari_advanced \
  --data-dir benchmarks/runs/mercari/data \
  --config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-final --mode final
```

精度版只需把配置改为 `optimized-config.json`，并换一个输出目录。
想重载模型就保留 `model.pkl`；不需要重载时才加 `--omit-model-artifact`。
此前轻量／精度完整本地进程为 44.63／53.75 分钟，峰值约 10 GiB，都是单次观察。
机器与依赖版本改变可能改变预测和分数，不能把历史耗时当保证。

## 8. 用测量判断性能

两版都训练并保存模型后运行：

```sh
PYTHONPATH=. python -m benchmarks.mercari_inference \
  --baseline-model benchmarks/runs/mercari/tutorial-accuracy/model.pkl \
  --optimized-model benchmarks/runs/mercari/tutorial-final/model.pkl \
  --test-file benchmarks/runs/mercari/data/test_stg2.tsv \
  --output benchmarks/runs/mercari/tutorial-serving.json \
  --rows 100000 --repeats 4 --threads 4 --fm-threads 2
```

`tutorial-accuracy` 是你用精度参数重训时选的输出目录。参数名 `baseline/optimized`
此处只是对照 A/B，不强制第一个模型为历史基线。
报告包含全部重复值、**算术平均值**、最小／最大和总体标准差。计时含特征和预测，
排除模型加载、输入 I/O 和结果校验。不要同时跑训练；四次重复也不是统计保证。

测量同权重批大小时，两个模型参数填**同一文件**，用新输出文件并加
`--sparse-batch-size 16384`。检查预测是否接近以及均值是否真的改善。
此 CLI 比较你给出的两个模型，不会自动重建已经发布的四项复用专家审计实验；
其范围保存在[原始记录](accuracy-fast-serving.json)。

## 9. 可选：在 Kaggle 执行并评分

本地源码打包不需要凭据；上传与评分需要账号。替换自己的用户名，使用独立 Kernel 名称。
第一阶段只是**时间检查，不得提交**：

```sh
python -m benchmarks.mercari_kernel --owner YOUR_USERNAME --stage one \
  --kernel-name libtree-mercari-tutorial-stage-one \
  --optimized-config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-kernel-one
kaggle kernels push -p benchmarks/runs/mercari/tutorial-kernel-one -t 3600
kaggle kernels status YOUR_USERNAME/libtree-mercari-tutorial-stage-one
```

它写入 `stage-one-predictions.csv` 和 `STAGE_ONE_BUDGET_CHECK_NEVER_SUBMIT.txt`。
轻量版已在 44.04 分钟通过，精度版第一阶段条件仍未验证；这也不等于复核全部历史资源要求。

另建第二阶段版本，再显式上传：

```sh
python -m benchmarks.mercari_kernel --owner YOUR_USERNAME --stage two \
  --kernel-name libtree-mercari-tutorial-entry \
  --optimized-config benchmarks/mercari/fast-config.json \
  --output-dir benchmarks/runs/mercari/tutorial-kernel-two
kaggle kernels push -p benchmarks/runs/mercari/tutorial-kernel-two -t 7200
kaggle kernels logs --follow YOUR_USERNAME/libtree-mercari-tutorial-entry
kaggle kernels status YOUR_USERNAME/libtree-mercari-tutorial-entry
```

Kernel 默认私有，内含源码与参数，在 Kaggle 编译 C++、重新训练，不上传本地模型或凭据。
记下 push 返回的版本。完成后用 Kaggle CLI/API 拉回源码、检查产物元数据，核对上传内容，
并确认根目录存在 `report.json` 与 `submission.csv`。索引短暂为空时有界重试读取元数据，
不要为此重新训练或盲目重复提交。产物下载可能受网络限制，官方日志流也会打印汇总报告。

下面所有命令中的 `VERSION_FROM_PUSH` 都要替换成刚才记录的版本号。
拉回的源码文件可能改名，但 SHA-256 应与本地打包脚本一致；检查清单中的两个根目录产物：

```sh
kaggle kernels pull YOUR_USERNAME/libtree-mercari-tutorial-entry/VERSION_FROM_PUSH \
  -p benchmarks/runs/mercari/tutorial-kernel-pulled -m
sha256sum benchmarks/runs/mercari/tutorial-kernel-two/mercari_libtree.py \
  benchmarks/runs/mercari/tutorial-kernel-pulled/*.py
kaggle kernels files YOUR_USERNAME/libtree-mercari-tutorial-entry/VERSION_FROM_PUSH \
  --format json --page-size 200
```

只有平台仍允许评分时，才显式提交这个**完成的确切版本**：

```sh
kaggle competitions submit mercari-price-suggestion-challenge \
  -k YOUR_USERNAME/libtree-mercari-tutorial-entry -v VERSION_FROM_PUSH \
  -f submission.csv -m 'LibTree tutorial: frozen fast configuration'
kaggle competitions submissions mercari-price-suggestion-challenge --format json
```

`VERSION_FROM_PUSH` 是占位符，要替换成实际版本。保存提交 ID、状态、成绩和错误。
`PENDING` 或只有 CSV 不能写成打榜成功。2026 年的记录证明当时这些尝试确实被评分，
不保证未来权限、服务可用性或完全相同分数。当前属于赛后练习，没有原赛新增正式名次。

## 10. 测试、排错和学习顺序

改模型前先运行已有行为测试。完整覆盖命令在[CI 工作流](../../.github/workflows/test.yml)，
可以先跑下面的重点测试：

```sh
PYTHONPATH=. NUMBA_DISABLE_JIT=1 python -m pytest \
  tests/test_mercari_features.py tests/test_mercari_solver.py \
  tests/test_mercari_advanced.py tests/test_mercari_inference.py -q
PYTHONPATH=. python -m pytest tests/test_mercari_fm.py -q
```

第一条用 Python 参考内核，第二条测真实编译的 FM。检查通过和跳过数量，缺依赖而跳过
不是验证成功。TDD 的做法是先写能失败的行为检查，再做最小实现，通过后重构；测试应有
独立参考，不照抄实现。100% 语句覆盖不等于所有分支、数值范围或真实模型效果都正确。

| 现象 | 先检查什么 |
| --- | --- |
| 原生库加载失败 | 激活虚拟环境、安装仓库包、使用 `PYTHONPATH=.` |
| 输出已存在 | 换新名称，保留旧证据 |
| 文件缺失或不匹配 | 测试阶段与模板是否配套，压缩包和嵌套目录是否正确 |
| 全量运行太慢或内存不足 | 先做小练习；查看 CPU 配额、内存和并发任务，缩参数会改变实验 |
| 训练误差下降而验证变差 | 过拟合；依据选择集决定停止，不用留出集调参 |
| 增线程／增批次没有加速 | 测完整成本，稀疏工作、同步及库内核都可能主导 |
| 重载价格异常 | 是否正确还原对数、来源是否可信、预处理版本是否匹配 |

推荐顺序：[NumPy 数组](https://numpy.org/doc/stable/user/absolute_beginners.html)
→ [pandas 表格](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
→ [文本特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
→ [Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
→ [分组验证](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
→ [交互树指南](../../docs/LEARNING_GUIDE.html)
→ [技术报告里的论文与代码映射](TECHNICAL_REPORT.zh-CN.md)。先理解直觉，再读公式和原论文。

原始数据、pickle、预测和临时输出放在被忽略的 `benchmarks/runs/`。
提交代码、参数与汇总证据。最小实验记录应包含：问题、固定划分、源码／数据／参数哈希、
指标、完整计时、失败尝试、选择理由，以及未参与选择的最终检查。

[文档验证记录](documentation-verification.json)保存 2026 年 10 月 18 日的小练习、重载、
重点测试与本地源码打包结果。全量分数仍为 10 月 17 日的归档测量。本地链接已检查；
部分外部学习网站受当前代理限制，未能在本环境重新验证其可访问性。

## 小编码器与 GPU 精度

[小文本方法页](SMALL_TEXT.zh-CN.md)和[实测报告](SMALL_TEXT_REPORT.zh-CN.md)介绍新实验。
词袋把学到的子词向量取平均，哈希相邻词对补充局部顺序；小 BERT 则让词语结合
上下文理解。它们输出对数价格。一个较弱预测，也可能帮助残差树判断另一个模型
何时容易出错。缓存词 ID 只是复用预处理，不是复用已经学好的价格模型权重。

FP16/BF16 都用 16 位，但数值范围、有效精度及硬件支持不同。T4 原生支持 FP16，
A100 还支持原生 BF16。能创建 BF16 张量，不等于能高效执行 BF16 运算。
报告保留了失败、正确的原生支持判断及固定批量成本，并与 Kaggle 精度分开描述。
