"""Render bilingual C5 performance reports from the paired raw measurements."""
import argparse
import json
from pathlib import Path

from benchmarks.c5_performance import summarize, validate


def render(data, chinese=False):
    validate(data['results'])
    index = {(r['dataset'], r['variant'], r['interface'], r['threads']): r
             for r in summarize(data['results'])}
    meta, names = data['metadata'], list(data['sources'])
    params = meta['parameters']
    labels = {'wine': '葡萄酒多分类（wine）', 'german_credit': '德国信用风险（german_credit）',
              'credit_status': '信用状态（credit_status）',
              'taiwan_late_payment': '台湾逾期付款（taiwan_late_payment）',
              'synthetic_20k': '合成数据（synthetic_20k）',
              'synthetic_100k': '合成数据（synthetic_100k）',
              'missing_20k': '含缺失值合成数据（missing_20k）',
              'before': '优化前 C5', 'after': '优化后 C5',
              'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'python': 'Python', 'cpp': 'C++'}
    def label(value):
        return labels.get(value, value) if chinese else value
    def value(name, variant, field, interface='python', threads=1):
        row = index.get((name, variant, interface, threads), {})
        return row.get(field, {}).get('mean')
    def show(number, scale=1, digits=3):
        return '—' if number is None else f'{number * scale:.{digits}f}'
    def ratio(a, b):
        return '—' if a is None or b is None else f'{a / b:.2f}×'
    out = []
    def line(english, translated):
        out.extend([translated if chinese else english, ''])
    def header(english, translated, separator):
        out.extend([translated if chinese else english, separator])

    line('# C5: faster training, direct class prediction and tree inspection',
         '# C5 性能优化：训练、直接类别预测与树结构解释')
    line('[English](C5_PERFORMANCE_REPORT.md) · [中文](C5_PERFORMANCE_REPORT.zh-CN.md) · '
         '[raw measurements](c5-performance-results.json) · [guide and references](../docs/c5.md)',
         '[English](C5_PERFORMANCE_REPORT.md) · [中文](C5_PERFORMANCE_REPORT.zh-CN.md) · '
         '[原始测量](c5-performance-results.json) · [使用指南与引用](../docs/c5.zh-CN.md)')
    line('## Scope and method', '## 范围与方法')
    line(f"{len(data['results'])} interface measurements on {len(names)} tasks; seeds {params['seeds']}; "
         f"{params['repeats']} timed fresh models per split/configuration; threads {params['threads']}. "
         'Arithmetic means, population standard deviations and extrema are retained. Repeated seeds '
         'add timing evidence, not independent test sets; the three holdouts overlap.',
         f"共 {len(data['results'])} 次接口测量、{len(names)} 个任务；随机种子 {params['seeds']}；"
         f"每个划分、配置创建新模型计时 {params['repeats']} 次；线程数 {params['threads']}。"
         '采用算术平均值，保留总体标准差和极值。同一种子的重复只增加计时证据，不增加独立测试集；三个留出划分存在重叠。')
    line(f"Baseline revision: `{meta['baseline_revision']}`. C5 defaults, split criteria, round budget and "
         'pruning are unchanged. Both static libraries use the same Release benchmark-driver source. '
         'Each Python worker imports its own checkout wrapper/library; engines run in reproducibly '
         'shuffled blocks, never concurrently. Fit, probability prediction, then class prediction are '
         'timed separately. Startup, imports and input I/O are excluded; wrapper validation is included. '
         'Class timing follows probability timing and benefits from warm caches; no dedicated warmup is discarded.',
         f"基线提交：`{meta['baseline_revision']}`。C5 默认参数、分裂准则、轮数预算与剪枝不变。"
         '同一份发布构建的评测主程序分别链接新旧静态库；Python 子进程分别导入对应版本的封装与库。'
         '按可复现随机顺序排列引擎测量块，各引擎依次运行。先训练，再分别计时概率预测和类别预测；'
         '排除启动、导入、输入文件读写，包含封装校验。类别预测在概率预测后进行，受益于已热缓存；不专门丢弃预热运行。')
    line('The first Python fit in each worker also includes lazy native-library loading; subsequent '
         'repetitions reuse that loaded library. This matters most for short tasks.',
         '每个 Python 子进程的首次拟合还计入原生库的延迟加载，后续重复复用已加载的库；这一开销对短任务影响更明显。')
    line(f"C5: up to {params['rounds']} SAMME rounds, depth {params['depth']}, {params['bins']} bins, "
         'minimum leaf rows 5, rate 1, alpha=.001, gain ratio. Competitors use the same round/depth/bin '
         'budgets, gradient rate .1 and L2=1; LightGBM has at most 16 leaves at depth 4. Algorithms and '
         'pruning units differ; this is fixed-budget comparison, not equal tuning. Native competitor '
         'comparisons cover binary tasks; Wine is multiclass, and gradient boosting may build a tree '
         'per class per round. Native competitor label timings are unavailable. Compare engines '
         'within an interface; native XGBoost probability timing includes test DMatrix construction.',
         f"C5 最多使用 {params['rounds']} 轮 SAMME、深度 {params['depth']}、{params['bins']} 个分箱、"
         '叶节点最少 5 行、学习率 1、alpha=.001 和增益率准则。竞品使用相同轮数、深度、分箱预算，'
         '梯度提升学习率 .1、L2=1；深度 4 时 LightGBM 最多 16 叶。算法与剪枝单位不同，属于固定预算比较，'
         '不是同等调优比较。原生竞品只比较二分类任务，葡萄酒任务为多分类；原生竞品未测类别预测时间。')
    line('Round budgets are not equal tree counts in multiclass tasks; Python timings include label '
         'encoding and wrapper checks during fitting, whereas native inputs are already prepared. '
         'C5 can stop on perfect or weak learners; accepted estimator counts are retained per run.',
         '多分类梯度提升每轮可能为每个类别构建一棵树，同轮数不代表同树数。应在同一接口内比较引擎；'
         '原生 XGBoost 概率预测计时包含测试 DMatrix 构建。Python 训练包含标签编码及封装校验，原生输入已预处理。'
         'C5 遇到完美或不合格学习器会提前停止，每次运行保留实际接受的树数。')
    line('Stratified 75/25 holdouts; categorical encoders fit training rows only. Source URLs/checksums '
         'and split hashes are in JSON. German credit and credit status use bad-credit labels; Taiwan '
         'uses September late payment with April features, not the original full default target. Random '
         'splits are not financial time backtests. Synthetic tasks have 3% label noise; missing_20k '
         'adds a fixed 10% NaN mask. No test-set tuning was used.',
         '按类别分层，以 75/25 划分训练集和测试集，类别编码仅拟合训练行；来源网址、校验和及划分指纹见 JSON。'
         '德国信用和信用状态使用坏信用标签；台湾使用 4 月信息预测 9 月逾期，不是原始完整违约目标。'
         '随机划分不是金融时间回测。合成任务有 3% 标签噪声，缺失值任务额外加入固定 10% NaN 掩码。没有使用测试集调参。')
    line(f"CPU {meta['cpu_model']}; {meta['affinity_cpus']} affinity CPUs; quota {meta['cpu_quota']}; "
         f"{meta['compiler']}; Python {meta['python']}; NumPy {meta['numpy']}; sklearn {meta['sklearn']}; "
         f"XGBoost {meta['xgboost']}; LightGBM {meta['lightgbm']}.",
         f"测量环境：CPU {meta['cpu_model']}；可用逻辑 CPU {meta['affinity_cpus']}；配额 {meta['cpu_quota']}；"
         f"编译器 {meta['compiler']}；Python {meta['python']}；NumPy {meta['numpy']}；sklearn {meta['sklearn']}；"
         f"XGBoost {meta['xgboost']}；LightGBM {meta['lightgbm']}。")
    header('| Task | Train rows | Test rows | Features in first split |',
           '| 任务 | 训练行数 | 测试行数 | 首次划分特征数 |', '| --- | ---: | ---: | ---: |')
    for name in names:
        row = next(r for r in data['results'] if r['dataset'] == name)
        out.append(f'| {label(name)} | {row["train_rows"]} | {row["test_rows"]} | {row["features"]} |')
    out.append('')
    line('## Implemented changes', '## 实现的优化')
    line('Feature-local histogram workspaces persist across nodes/rounds; parent impurity is computed '
         'once per node. Without missing rows, duplicate missing-direction candidates are skipped, '
         'preserving left-first ties. Retained leaf labels are reused for SAMME weighted errors. '
         'Prediction shares validation/traversal, avoids duplicate native validation and exp(0), and '
         'Python class prediction calls native class voting directly. Scratch remains call-local. '
         'One padded allocation replaces per-worker vote-buffer allocations; 64-byte gaps avoid '
         'sharing 64-byte cache lines, and static row ranges balance small/deep ensembles. '
         'Histogram workspace adds temporary O(features × bins × classes) storage during fitting. '
         'See the cited data-locality motivation in the guide; C5 does not implement histogram subtraction.',
         '各特征直方图缓冲在节点和提升轮次之间复用；父节点不纯度每个节点只算一次。'
         '没有缺失行时跳过等价的第二种缺失路由，保留优先向左的平局规则。'
         '复用保留叶子上的训练行类别，计算 SAMME 加权误差。推理共用校验与遍历，'
         '消除原生重复校验与 exp(0)，Python 类别预测直接调用原生投票。临时票数缓冲由每次调用独立持有。'
         '一次带间隔的分配替代逐线程分配，64 字节间隔避免共享 64 字节缓存行；连续行区间分工兼顾小批量、深模型。'
         '直方图复用会在拟合期间增加 O(特征数 × 分箱数 × 类别数) 临时工作区。数据局部性动机及引用见指南；C5 没有实现直方图差分。')
    line('## Before/after training', '## 优化前后训练时间')
    line('Speedup = before mean / after mean; below 1 means slower. These are bundle-level A/B '
         'results, not isolated causal attribution to each optimization.',
         '加速比＝优化前平均耗时／优化后平均耗时，小于 1 表示变慢。这是整组优化的 A/B，不能据此给每项改动单独归因。')
    header('| Task | Interface | Threads | Before ms | After mean ± std ms | After min–max ms | Speedup |',
           '| 任务 | 接口 | 线程 | 优化前毫秒 | 优化后均值 ± 标准差（毫秒） | 优化后最小–最大值（毫秒） | 加速比 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |')
    fit_ratios = []
    for name in names:
        for interface in ('python', 'cpp'):
            for threads in params['threads']:
                row = index[name, 'after', interface, threads]['fit_seconds']
                a = value(name, 'before', 'fit_seconds', interface, threads)
                b = row['mean']
                fit_ratios.append(a / b)
                out.append(f'| {label(name)} | {label(interface)} | {threads} | {a*1000:.3f} | '
                           f'{b*1000:.3f} ± {row["std"]*1000:.3f} | '
                           f'{row["min"]*1000:.3f}–{row["max"]*1000:.3f} | {a/b:.2f}× |')
    out.append('')
    line('## Before/after inference', '## 优化前后推理时间')
    header('| Task | Interface | Threads | Before probability ms | After probability ms | Speedup | Before class ms | After class ms | Speedup |',
           '| 任务 | 接口 | 线程 | 优化前概率预测毫秒 | 优化后概率预测毫秒 | 概率加速比 | 优化前类别预测毫秒 | 优化后类别预测毫秒 | 类别加速比 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for name in names:
        for interface in ('python', 'cpp'):
            for threads in params['threads']:
                p, q = [value(name, v, 'predict_seconds', interface, threads) for v in ('before', 'after')]
                a, b = [value(name, v, 'predict_label_seconds', interface, threads) for v in ('before', 'after')]
                out.append(f'| {label(name)} | {label(interface)} | {threads} | {show(p,1000)} | {show(q,1000)} | '
                           f'{ratio(p,q)} | {show(a,1000)} | {show(b,1000)} | {ratio(a,b)} |')
    out.append('')
    line('## Inference variability', '## 推理波动与异常')
    header('| Task | Interface | Threads | After probability mean ± std ms | Probability min–max ms | After class mean ± std ms | Class min–max ms |',
           '| 任务 | 接口 | 线程 | 优化后概率均值 ± 标准差（毫秒） | 概率最小–最大值（毫秒） | 优化后类别均值 ± 标准差（毫秒） | 类别最小–最大值（毫秒） |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |')
    for name in names:
        for interface in ('python', 'cpp'):
            for threads in params['threads']:
                row = index[name, 'after', interface, threads]
                p, q = row['predict_seconds'], row['predict_label_seconds']
                out.append(f'| {label(name)} | {label(interface)} | {threads} | '
                           f'{p["mean"]*1000:.3f} ± {p["std"]*1000:.3f} | '
                           f'{p["min"]*1000:.3f}–{p["max"]*1000:.3f} | '
                           f'{q["mean"]*1000:.3f} ± {q["std"]*1000:.3f} | '
                           f'{q["min"]*1000:.3f}–{q["max"]*1000:.3f} |')
    out.append('')
    line('## Held-out quality and competitor speed', '## 留出集效果与竞品速度')
    line('Python, one thread, arithmetic means. Before/after C5 rule JSON, training probabilities and '
         'test probabilities are byte-identical within each interface, seed and split, including across '
         'thread counts/repeats. Thus measured C5 accuracy/AUC/KS are unchanged. KS is absolute '
         'bad/good score-distribution distance; directional KS and AP are also retained. Wine has no '
         'binary KS/AP. Scores are uncalibrated; these are not production risk probabilities.',
         '下表为 Python 单线程算术平均值。逐接口、种子、划分检查，新旧 C5 的规则 JSON、训练概率和测试概率'
         '逐字节一致，不同线程数与重复运行也保持一致；因此实测准确率、AUC、KS 均未改变。'
         'KS 为坏样本与好样本评分分布的最大绝对差，原始记录同时保留方向性 KS 和平均精确率 AP。'
         '葡萄酒多分类不计算二分类 KS/AP。评分未经校准，不代表生产风险概率。')
    header('| Task | Model | Fit ms | Probability ms | Accuracy % | Balanced acc % | AUC | KS | AP | Rules |',
           '| 任务 | 模型 | 训练毫秒 | 概率预测毫秒 | 准确率 % | 平衡准确率 % | AUC | KS | AP | 规则数 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for name in names:
        for variant in ('after', 'xgboost', 'lightgbm'):
            fields = [('fit_seconds',1000,3), ('predict_seconds',1000,3), ('accuracy',100,2),
                      ('balanced_accuracy',100,2), ('roc_auc',1,4), ('ks',1,4),
                      ('average_precision',1,4), ('rules',1,1)]
            cells = ' | '.join(show(value(name,variant,f), scale,digits) for f,scale,digits in fields)
            out.append(f'| {label(name)} | {label(variant)} | {cells} |')
    out.append('')
    line('## Thread scaling', '## 多线程扩展性')
    line('Each engine uses its own one-thread mean as baseline. Small C5 fits may create no workers; '
         'fluctuations there are not parallel gains. All ratios, including regressions, are retained.',
         '各引擎分别用自身单线程平均耗时作为基线。小型 C5 拟合可能不创建工作线程，'
         '其耗时波动不代表并行收益。所有比值均保留，包括退化结果。')
    header('| Task | Interface | Model | Fit 1→4 speedup | Probability 1→4 speedup | Class 1→4 speedup |',
           '| 任务 | 接口 | 模型 | 训练单线程→四线程加速比 | 概率预测单线程→四线程加速比 | 类别预测单线程→四线程加速比 |',
           '| --- | --- | --- | ---: | ---: | ---: |')
    for name in names:
        for interface in ('python', 'cpp'):
            for variant in ('before', 'after', 'xgboost', 'lightgbm'):
                if (name, variant, interface, 4) not in index:
                    continue
                cells = ' | '.join(ratio(value(name,variant,f,interface,1), value(name,variant,f,interface,4))
                                   for f in ('fit_seconds', 'predict_seconds', 'predict_label_seconds'))
                out.append(f'| {label(name)} | {label(interface)} | {label(variant)} | {cells} |')
    out.append('')
    line('## Tree inspection cost', '## 树结构与单条解释的开销')
    line('On-demand Python one-thread means, outside fit/predict timing. JSON export includes all '
         'retained trees; one-sample explanation includes actual paths and votes. Both include C ABI '
         'size queries, UTF-8 handling and Python JSON work. They are inspection formats, not model persistence.',
         '下表为按需调用的 Python 单线程平均值，不计入训练或预测时间。JSON 导出全部保留树；'
         '单条解释包含真实路径与票数。两者包含 C ABI 缓冲大小查询、UTF-8 处理和 Python JSON 开销。'
         '输出用于检查，不是可重新加载的持久化模型格式。')
    header('| Task | Full JSON export ms | One-sample explanation ms | JSON bytes |',
           '| 任务 | 完整 JSON 导出毫秒 | 单条预测解释毫秒 | JSON 字节数 |', '| --- | ---: | ---: | ---: |')
    for name in names:
        out.append(f'| {label(name)} | {show(value(name,"after","dump_seconds"),1000)} | '
                   f'{show(value(name,"after","explain_seconds"),1000)} | {show(value(name,"after","dump_bytes"),digits=0)} |')
    out.append('')
    line('## Findings and limitations', '## 结论与限制')
    faster = sum(r > 1 for r in fit_ratios)
    line(f'Training improves in {faster}/{len(fit_ratios)} before/after groups; observed training '
         f'speedups range from {min(fit_ratios):.2f}× to {max(fit_ratios):.2f}×. '
         'Inspect each inference row separately: reducing training work does not guarantee faster '
         'inference or better thread scaling on every task. Quality did not improve because model '
         'construction semantics and parameters were preserved.',
         f'训练在 {faster}/{len(fit_ratios)} 个新旧对照组中变快，实测训练加速比范围为 '
         f'{min(fit_ratios):.2f}× 至 {max(fit_ratios):.2f}×。推理应逐行查看；减少训练工作不保证每个任务的'
         '推理或线程扩展都更好。本轮保持建模语义和参数，效果未发生提升或退化。')
    line('Shared cloud CPU scheduling, short-task overhead and overlapping splits limit generalization. '
         'No statistical-significance, universal winner or human-comprehension claim follows from '
         'these timings. Rule counts are leaf-count proxies across different algorithms. C5 probabilities '
         'remain uncalibrated, and competitor learning-rate/regularization semantics differ. '
         'Raw means/std/min/max and negative outcomes remain available.',
         '共享云 CPU 调度、短任务开销和重叠划分限制结论外推。计时不能支持统计显著、普遍最优或用户理解能力提升的宣称。'
         '跨算法规则数只是叶节点数量代理；C5 概率仍未经校准，竞品学习率和正则化语义也不同。'
         '原始均值、标准差、最小值、最大值和负收益结果均保留。')
    line('The [first candidate raw runs](c5-performance-first-pass.json) are retained, including a '
         '4.03 ms Python four-thread class-prediction spike on missing_20k. That candidate used separate '
         'worker buffers and 64-row tiles. The final candidate uses padded shared storage and static '
         'row ranges. [This patch](c5-first-pass.patch), applied to this measured source snapshot in '
         'a separate checkout, reconstructs the first candidate. These are separate measurement phases, '
         'not a randomized isolated ablation; their difference cannot be attributed solely to a kernel. '
         'The final allocation-budget test intentionally rejects the first candidate.',
         '[首轮候选原始测量](c5-performance-first-pass.json)也保留，包含缺失值任务中 Python 四线程'
         '类别预测的一次 4.03 毫秒尖峰。该候选使用独立线程缓冲和 64 行分块；最终版本使用带间隔的'
         '连续缓冲和连续行区间分工。在独立检出中对本次被测源码应用[重建补丁](c5-first-pass.patch)，'
         '可还原首轮候选。这是不同测量阶段，不是随机交错的单因素消融，不能把两轮差异完全归因于某个内核。'
         '首轮候选不满足最终新增的一次临时分配测试契约。')
    if 'validation' in data:
        line('## Validation', '## 验证')
        line('TDD: tree/path export and direct-class tests failed before implementation; the CLI '
             'initially rejected trees, and the one-allocation contract failed with std::bad_alloc. '
             'Independent JSON-tree traversal now reproduces actual paths, rules and votes for all '
             'four split criteria, including multiclass/NaN. Tests cover constant models, 256 classes, '
             'failed refits, output buffers, concurrent inspection/readers and thread-count parity.',
             '测试驱动过程：先写树导出、路径重放与直接类别预测测试，确认失败；命令行最初拒绝 trees，'
             '一次临时分配契约最初因 std::bad_alloc 失败，再通过实现修复。独立 JSON 树遍历现在能复现'
             '四种分裂准则下的路径、规则和票数，包含多分类与缺失值。另覆盖常量模型、256 类边界、'
             '重新拟合失败、输出缓冲、并发解释与读取，以及跨线程数一致性。')
        v = data['validation']
        line(f"Native CTest {v['ctest']}; Python estimator/inspection tests {v['python']}; CLI {v['cli']}; "
             f"performance/RSS {v['performance']}; methodology {v['methodology']}. "
             f"ASan/UBSan/LSan and TSan passed. Native line coverage {v['native_lines']}; Python "
             f"statement coverage {v['python_statements']}, both 100%. No new coverage exclusions. "
             'The benchmark harness and report generator are outside production coverage gates.',
             f"原生 CTest {v['ctest']}；Python 模型与解释测试 {v['python']}；命令行 {v['cli']}；"
             f"性能与常驻内存 {v['performance']}；评测方法 {v['methodology']}。"
             f"ASan/UBSan/LSan 和 TSan 检查通过。原生行覆盖 {v['native_lines']}；"
             f"Python 语句覆盖 {v['python_statements']}，均为 100%，未新增覆盖排除项。评测与报告脚本不在生产覆盖率门槛内。")
    line('## Reproduce', '## 复现')
    commands = """```sh
python -m pip install '.[dev,benchmark]'
make all
git worktree add --detach /tmp/libtree-c5-before 60f003d
cmake -S /tmp/libtree-c5-before -B /tmp/libtree-c5-before/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/libtree-c5-before/build -j4
PYTHONPATH=python:. python benchmarks/c5_performance.py --baseline /tmp/libtree-c5-before \\
  --seeds 42 2024 7 --threads 1 4 --repeats 3 --rounds 30 \\
  --output benchmarks/c5-performance-results.json
PYTHONPATH=python:. python benchmarks/c5_performance_report.py
make test
make coverage
make sanitize
make thread-sanitize
```"""
    line(commands, commands)
    line(f"Measured implementation/harness SHA-256: `{meta['code_sha256']}`. The JSON retains baseline "
         'revision, driver binary hashes, source URLs/checksums, split hashes and raw per-run fingerprints. '
         'Downloaded/prepared data, predictions and local drivers remain ignored by git.',
         f"被测实现与评测脚本 SHA-256：`{meta['code_sha256']}`。JSON 保留基线提交、主程序二进制指纹、"
         '来源网址与校验和、划分指纹及每次原始运行的指纹。下载与预处理数据、预测文件和本地评测程序由 git 忽略。')
    return '\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('benchmarks/c5-performance-results.json'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    for name, chinese in (('C5_PERFORMANCE_REPORT.md', False), ('C5_PERFORMANCE_REPORT.zh-CN.md', True),
                          ('C5_PERFORMANCE_REPORT.txt', True)):
        Path('benchmarks', name).write_text(render(data, chinese), encoding='utf-8')
