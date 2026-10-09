# LibTree

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

输入为稠密二维矩阵 `(样本数, 特征数)`，标签为一维向量；二分类标签必须为
0 或 1。类别特征需要编码，编码器只在训练集上拟合，基准脚本提供了示例。
连续的 float32 数组可避免类型转换复制。NaN 表示缺失值；无穷大和错误形状
会被拒绝。`fit` 替换旧模型，不修改输入。`close` 释放原生内存，自动回收也
有效。支持 `get_params`、`set_params`、`score` 和 sklearn `clone`，但尚未
实现完整的 sklearn estimator 协议，例如高级 metadata routing。

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
make coverage   # C++ 行覆盖率至少 90%，Python 报告和 HTML
clang-format --dry-run --Werror include/libtree/*.h src/*.cc tests/*.cc examples/*.cc benchmarks/*.cc
PYTHONPATH=python python benchmarks/run.py --repeats 3
```

TDD 流程：先写行为测试并确认失败，再实现最小修复，相关测试通过后重构。
已有测试包括独立的穷举 SSE 校验、缺失值、极端值、确定性、重复拟合、错误
参数与形状、C ABI 错误、Python 安装验证、反复释放及代表性工作负载。
覆盖率和内存工具提供额外证据，并不证明所有输入上都没有错误。
CI 运行检查和一次基准冒烟测试；仓库保存的性能结果采用三次独立重复。

## 对比与迁移

[实测报告](benchmarks/README.md)涵盖 Titanic、医疗保险费用、糖尿病、乳腺癌
及 2 万行 Friedman 数据，分别对比 **C++ 原生调用和 Python 接口**与
XGBoost 3.4.1 的效果、训练时间、预测时间和进程内存。
固定划分上的效果接近，速度因数据集而异；较大的合成数据上 XGBoost 更快。
这些是本地留出集结果，没有提交 Kaggle 榜单，也不构成普遍优于 XGBoost 的结论。

本次重构删除了不安全的裸指针所有权、不完整的随机森林和 entropy 分支、
未经推导的 boosting 权重及未经校验的二进制模型读写。旧 `train`/`classify`
命令、头文件和模型格式不兼容，需通过新 API 重新训练。原 `data/` 文件保留为
历史数据。当前不支持多分类、排序、稀疏矩阵、GPU/分布式训练、Python 模型持久化或
样本权重。贡献前请阅读[设计决策和风格例外](docs/design.zh-CN.md)。

[Validation evidence / 验证记录](docs/validation.md)

[新增 Kaggle 五数据集、多划分测试报告](benchmarks/KAGGLE_REPORT.zh-CN.md)

[交互 HTML 图表报告](benchmarks/KAGGLE_REPORT.html)（离线查看，切换 C++ / Python，导出原始数据）
