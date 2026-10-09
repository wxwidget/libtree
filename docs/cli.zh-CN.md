# 参数化命令行 GBDT

[English](cli.md)

`xgbt` 是本仓库 **LibTree GBDT** 的 C++17 主程序，不是 XGBoost 官方命令。
编译后无需 Python 即可运行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/xgbt --help
./build/xgbt train --train examples/csv/train.csv --test examples/csv/test.csv \
  --header --objective binary --trees 80 --depth 3 --bins 32 \
  --min-leaf 1 --learning-rate 0.2 --l2 0 --min-gain 0 \
  --model build/example.lt --output build/predictions.csv
./build/xgbt predict --model build/example.lt --data examples/csv/test.csv \
  --header --output build/loaded-predictions.csv
```

示例文件是很小的合成数据，用于学习命令，不作为基准。输入为数字 CSV：
**训练集第一列是标签**，测试集和预测输入只包含特征，顺序必须一致。
`--header` 跳过两个输入文件的首行；空字段或 `nan` 表示特征缺失，空行忽略。
不支持引号字段或自动类别编码。不合理数字、无穷值、错误列数会被拒绝；
二分类标签必须为 0/1。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--objective` | `regression` | 回归 `regression` 或二分类 `binary` |
| `--trees` / `--num-trees` | 100 | 提升轮数 |
| `--depth` / `--max-depth` | 4 | 路径深度，0–20 |
| `--bins` / `--max-bins` | 64 | 分位数分箱数，2–65535 |
| `--min-leaf` / `--min-samples-leaf` | 5 | 子节点最少样本数 |
| `--learning-rate` / `--rate` | 0.1 | 学习率，(0,1] |
| `--l2` | 1 | 非负叶节点正则 |
| `--min-gain` | 0 | 非负最小划分增益 |
| `--threads` | 1 | CPU 线程预算，含调用线程，1–256；训练和预测均支持 |

`train` 必须传 `--train`；`--test`、`--model`、`--output` 均可省略。
没有 `--test` 时对训练特征预测；没有 `--model` 时仅在内存中训练。
二分类输出为正类概率，每行一个值，不直接输出类别。stdout 输出 JSON，
包含树数、预测样本数、训练/预测秒数和最后一轮训练损失。计时不包含 CSV
读取和模型/预测文件写入；训练损失不是测试集效果。

`predict` 必须传 `--model`、`--data`、`--output`，使用模型内保存的目标和
参数，拒绝训练参数而不是忽略。`--threads` 是运行配置，加载时可以另设；
小任务使用较少工作线程。标准输出 JSON 包含配置的线程数。错误返回非零状态并写入 stderr。输出路径
不能与输入相同；校验成功后原子替换明确指定的输出，错误输入不会破坏旧结果。

模型为版本化文本格式 `LIBTREE_GBDT 1`，保留 float/double 往返精度。
加载检查数值、特征编号、节点边界、树结构、缺失方向、深度和尾部内容。
加载失败保留旧模型。安全限制为最多 1 万棵树、总计 100 万节点；超限模型
无法保存。旧二进制格式不兼容。训练损失历史不写入模型文件。
C++ 提供 `SaveModel(std::ostream&)`、`LoadModel(std::istream&)`，Python 暂未
暴露持久化方法。可使用 `cmake --install build --prefix /可写目录` 安装到
自选目录，无需系统级安装。

## 三方比较与参数化复现

```sh
python -m pip install '.[benchmark]'
PYTHONPATH=python python benchmarks/kaggle.py \
  --engines libtree xgboost lightgbm --seeds 42 2024 2026 --repeats 3 \
  --trees 100 --depth 4 --bins 64 --min-leaf 5 --rate 0.1 --l2 1 --min-gain 0 \
  --output benchmarks/comparison-results.json
python benchmarks/kaggle_report.py --input benchmarks/comparison-results.json --prefix comparison
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/html_report.py
```

C++ 原生 C API 和 Python estimator 都使用相同主参数。基准深度限制为 1–20，
因为此 LightGBM 接口不支持根节点单独成树。统一 CPU 单线程、float32 稠密
数据。LightGBM 为 leaf-wise，`num_leaves=2^depth`；XGBoost 保留默认最小
Hessian 权重 1，并非严格的最少样本数，具体可比性限制见报告。没有使用测试
集做参数搜索，也不包含 GPU 或多线程比较。
