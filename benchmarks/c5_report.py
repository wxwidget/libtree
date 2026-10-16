"""Render inspectable C5 comparisons from raw runs, recomputing all summaries."""
import argparse
import json
from pathlib import Path

from benchmarks.c5 import summarize


def render(data, chinese=False):
    summaries = summarize(data['results'])
    index = {(r['dataset'], r['variant'], r['interface'], r['threads']): r for r in summaries}
    names = list(data['sources'])
    meta = data['metadata']
    out = ['# C5 风格分类器：实测、消融与验证' if chinese else
           '# C5-style classifier: measurements, ablations and validation', '']
    def line(english, translated=None):
        out.extend([translated if chinese and translated is not None else english, ''])
    labels = {
        'breast_cancer': '乳腺癌（breast_cancer）',
        'wine': '葡萄酒多分类（wine）',
        'pima': '皮马糖尿病（pima）',
        'bank_marketing': '银行营销（bank_marketing）',
        'synthetic_20k': '合成数据（synthetic_20k）',
        'synthetic_100k': '合成数据（synthetic_100k）',
        'c5': 'C5 默认增益率（c5）',
        'tree': 'sklearn 单棵决策树（tree）',
        'adaboost': 'sklearn 自适应提升（adaboost）',
        'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
        'c5_single': 'C5 单棵树（c5_single）',
        'c5_unpruned': 'C5 不剪枝（c5_unpruned）',
        'c5_entropy': 'C5 信息熵（c5_entropy）',
        'c5_gini': 'C5 基尼准则（c5_gini）',
        'python': 'Python', 'cpp': 'C++',
    }
    def label(name):
        return labels.get(name, name) if chinese else name
    def mean(name, variant, field, interface='python', threads=1):
        return index[name, variant, interface, threads][field]['mean']
    line('新增 C5 风格多分类器已经实现；以下结果来自真实运行。它不是原版 C5.0 兼容实现。'
         if chinese else 'The independent C5-style multiclass classifier is implemented. '
         'The results below are measured; this is not a compatible C5.0 implementation.')
    line('[raw measurements](c5-results.json) · [English](C5_REPORT.md) · [中文](C5_REPORT.zh-CN.md) · '
         '[methods/citations](../docs/c5.md) · [interactive guide](../docs/C5_GUIDE.html)',
         '[原始记录](c5-results.json) · [English](C5_REPORT.md) · [中文](C5_REPORT.zh-CN.md) · '
         '[方法与引用](../docs/c5.zh-CN.md) · [交互指南](../docs/C5_GUIDE.html)')
    line('## Method', '## 方法')
    line(f"{len(data['results'])} measurements; {len(names)} tasks; seeds 42/2024/7; "
         '3 repetitions per split/configuration; stratified 75/25 train/test; 1 and 4 threads. '
         'Arithmetic means are used throughout; raw repeats, population standard deviations and min/max '
         'are retained. The three splits overlap and are not nine independent test sets. No test-set tuning.',
         f"共 {len(data['results'])} 次测量、{len(names)} 个任务；随机种子为 42/2024/7；"
         '每个划分、每种配置重复计时 3 次；按类别分层，以 75/25 划分训练集和测试集；使用 1 和 4 个线程。'
         '全文采用算术平均值，保留原始重复记录、总体标准差、最小值和最大值。三个划分存在重叠，'
         '不是九个独立测试集；没有使用测试集调参。')
    line('Fixed budgets: 30 boosting rounds, max depth 4, 64 histogram bins, min leaf rows 5. '
         'C5: SAMME rate 1, alpha .001, gain ratio. sklearn tree/SAMME: entropy, ccp_alpha .001; '
         'SAMME rate 1. XGBoost/LightGBM: gradient rate .1, L2=1; LightGBM max 16 leaves/tree. '
         'Gradient and discrete boosting rates are not directly interchangeable; these are fixed practical '
         'settings, not equal optimized accuracy. XGBoost retains its Hessian-based minimum child weight. '
         'The sklearn tree has one estimator; multiclass GBDT can build one tree per class per round.',
         '固定预算：提升 30 轮，最大深度 4，直方图分箱数 64，叶节点最少 5 行样本。'
         'C5 使用 SAMME，学习率 1、剪枝参数 alpha=.001、增益率分裂准则。'
         'sklearn 决策树和 SAMME 使用信息熵、ccp_alpha=.001，SAMME 学习率为 1。'
         'XGBoost/LightGBM 梯度提升学习率为 .1、L2=1，LightGBM 每棵树最多 16 个叶节点。'
         '梯度提升与离散提升的学习率不能直接等同；这里比较固定实用配置，未经同等调优。'
         'XGBoost 保留基于 Hessian 的最小子节点权重约束。sklearn 单树只有一个模型；'
         '多分类 GBDT 每轮可能为每个类别分别构建一棵树。')
    line('Numeric NaN is preserved for native learners. sklearn inputs use training-fitted median imputation '
         'plus missingness indicators; transformation costs are timed. Bank categories are one-hot encoded '
         'from training data only, before engine timing. Public bank data excludes call duration (post-outcome '
         'information); Pima invalid physiological zeros become NaN. Real-source URLs/SHA-256 are recorded. '
         'Bank marketing predicts subscriptions, not credit defaults; it is not a temporal backtest. '
         'Synthetic 20k/100k tasks have 20 features and fixed 3% label noise. Wine has three classes.',
         '原生学习器保留数值缺失值 NaN。sklearn 使用训练集拟合的中位数填充，并增加缺失指示特征，'
         '转换开销计入时间。银行类别特征仅用训练集拟合独热编码，在引擎计时前处理。'
         '公开银行数据删除通话时长字段，因为它包含通话结束后才知道的信息；'
         '皮马糖尿病数据中无效的生理指标零值转为 NaN。真实数据的来源网址和 SHA-256 均已记录。'
         '银行营销任务预测是否订阅存款产品，不是信用违约预测，也不是时间回测。'
         '合成任务分别有 2 万和 10 万行、20 个特征，固定 3% 标签噪声；葡萄酒任务有三个类别。')
    line(f"CPU: {meta.get('cpu_model','unknown')}; {meta['affinity_cpus']} affinity CPUs; quota {meta['cpu_quota']}. "
         f"Compiler: {meta.get('compiler','unknown')}. Python {meta['python']}, NumPy {meta['numpy']}, "
         f"sklearn {meta['sklearn']}, XGBoost {meta['xgboost']}, LightGBM {meta['lightgbm']}.",
         f"CPU 型号：{meta.get('cpu_model','未知')}；进程可用逻辑 CPU 数：{meta['affinity_cpus']}；"
         f"CPU 配额：{meta['cpu_quota']}。编译器：{meta.get('compiler','未知')}。"
         f"运行版本：Python {meta['python']}，NumPy {meta['numpy']}，sklearn {meta['sklearn']}，"
         f"XGBoost {meta['xgboost']}，LightGBM {meta['lightgbm']}。")
    line('Timing: fresh models in one imported Python process; native engines run in fresh subprocesses. '
         'Startup/data-file I/O are outside native timing; Python includes wrapper/validation costs. '
         'Engine order is reproducibly shuffled. No concurrent stress/sanitizer job ran during measurements. '
         'No dedicated warm-up is discarded; small timings show noticeable noise. Compare engines within '
         'the same interface. sklearn SAMME is serial; its four-thread timings are omitted. Native competitor '
         'comparisons cover binary tasks; C5 native/Python parity is checked on all tasks, at each run.',
         '计时方法：Python 在已完成导入的同一进程内创建新模型；原生引擎每次在新子进程中运行。'
         '原生计时排除启动和数据文件读写；Python 包含封装及校验开销。引擎执行顺序可复现地随机排列。'
         '测量期间没有并行运行压力测试或内存、线程检查任务；不专门丢弃预热运行，短耗时测量存在明显噪声。'
         '引擎间应在相同接口内比较。sklearn SAMME 为串行实现，省略其四线程计时。'
         '原生竞品比较覆盖二分类任务；所有任务每次运行均检查 C5 原生与 Python 接口输出的一致性。')
    line('## Quality and speed', '## 效果与性能')
    line('Python, one thread; means across 3 seeds × 3 repeats. Fit/predict are milliseconds. '
         'Repeating a seed measures timing, not additional statistical evidence about accuracy. '
         '“Rules” counts retained leaves, including gradient trees; this is a structural proxy, not '
         'equivalent global explanations across algorithms.',
         '下表为 Python 单线程结果，取 3 个种子 × 3 次重复的算术平均值，训练与推理时间单位为毫秒。'
         '重复同一种子只增加计时证据，不增加效果的独立统计证据。“规则数”按保留的叶节点统计，'
         '也包括梯度提升树的叶节点，仅作为结构复杂度代理，不代表不同算法具有等价的全局解释。'
         '平衡准确率为各类别召回率的平均值，适合同时观察不平衡任务。表中括号内标识对应原始 JSON 名称。')
    out += ['| 任务 | 引擎 | 训练毫秒 | 推理毫秒 | 准确率 % | 平衡准确率 % | AUC | 规则数 |' if chinese else
            '| Task | Engine | Fit ms | Predict ms | Accuracy % | Balanced acc % | AUC | Rules |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('c5', 'tree', 'adaboost', 'xgboost', 'lightgbm'):
            out.append(f'| {label(name)} | {label(variant)} | {mean(name,variant,"fit_seconds")*1000:.3f} | '
                       f'{mean(name,variant,"predict_seconds")*1000:.3f} | {mean(name,variant,"accuracy")*100:.2f} | '
                       f'{mean(name,variant,"balanced_accuracy")*100:.2f} | {mean(name,variant,"roc_auc"):.4f} | '
                       f'{mean(name,variant,"rules"):.1f} |')
    out.append('')
    line('## Thread scaling', '## 线程扩展')
    line('Speedup = mean time at one thread / mean time at four threads. Small C5 models deliberately '
         'use no pool; their fluctuations do not establish parallel gains. Show medium/large tasks only.',
         '加速比＝单线程平均耗时／四线程平均耗时。小型 C5 模型按设计不使用线程池，'
         '其计时波动不能作为并行收益。下表只展示中、大规模任务。')
    out += ['| 任务 | 接口 | 引擎 | 训练加速比 | 推理加速比 | 四线程训练毫秒 |' if chinese else
            '| Task | Interface | Engine | Fit speedup | Predict speedup | Fit at 4 threads ms |',
            '| --- | --- | --- | ---: | ---: | ---: |']
    for name in ('bank_marketing', 'synthetic_20k', 'synthetic_100k'):
        if name not in names:
            continue
        for interface in ('python', 'cpp'):
            for variant in ('c5', 'xgboost', 'lightgbm'):
                if (name, variant, interface, 4) not in index:
                    continue
                a = mean(name, variant, 'fit_seconds', interface, 1)
                b = mean(name, variant, 'fit_seconds', interface, 4)
                speed = mean(name, variant, 'predict_seconds', interface, 1) / mean(name, variant, 'predict_seconds', interface, 4)
                out.append(f'| {label(name)} | {label(interface)} | {label(variant)} | {a/b:.2f}× | {speed:.2f}× | {b*1000:.3f} |')
    out.append('')
    line('## Ablations', '## 消融实验')
    line('All are one-thread, preselected configurations. `c5_single`: one tree; `c5_unpruned`: alpha=0; '
         '`c5_entropy`/`c5_gini`: change only the criterion. All other C5 settings stay fixed. '
         'Shorter rules always come from actual pruning/merged bounds, not selective export.',
         '所有配置均预先确定，使用单线程。c5_single 表示单棵树；c5_unpruned 将 alpha 设为 0、关闭剪枝；'
         'c5_entropy 和 c5_gini 仅改变分裂准则，其他 C5 参数不变。'
         '规则缩短来自实际剪枝或条件边界合并，不是选择性隐藏导出的规则。')
    out += ['| 任务 | 配置 | 准确率 % | 训练准确率 % | AUC | 训练毫秒 | 规则数 | 条件数 |' if chinese else
            '| Task | Variant | Accuracy % | Train accuracy % | AUC | Fit ms | Rules | Conditions |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('c5_single', 'c5_unpruned', 'c5', 'c5_entropy', 'c5_gini'):
            out.append(f'| {label(name)} | {label(variant)} | {mean(name,variant,"accuracy")*100:.2f} | '
                       f'{mean(name,variant,"train_accuracy")*100:.2f} | {mean(name,variant,"roc_auc"):.4f} | '
                       f'{mean(name,variant,"fit_seconds")*1000:.3f} | {mean(name,variant,"rules"):.1f} | '
                       f'{mean(name,variant,"conditions"):.1f} |')
    out.append('')
    line('## Variability and outliers', '## 波动与异常')
    out += ['| 任务 | 接口 | 线程数 | C5 训练均值 ± 标准差（毫秒） | 训练最小–最大值（毫秒） | 推理均值 ± 标准差（毫秒） |' if chinese else
            '| Task | Interface | Threads | C5 fit mean ± std ms | Min–max ms | Predict mean ± std ms |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in names:
        for interface in ('python', 'cpp'):
            for threads in (1, 4):
                if (name, 'c5', interface, threads) not in index:
                    continue
                r = index[name, 'c5', interface, threads]
                fit, pred = r['fit_seconds'], r['predict_seconds']
                out.append(f'| {label(name)} | {label(interface)} | {threads} | {fit["mean"]*1000:.3f} ± '
                           f'{fit["std"]*1000:.3f} | {fit["min"]*1000:.3f}–{fit["max"]*1000:.3f} | '
                           f'{pred["mean"]*1000:.3f} ± {pred["std"]*1000:.3f} |')
    out.append('')
    line('## Conclusions and limits', '## 结论与限制')
    line('默认增益率准则更偏向紧凑规则，但不是全面第一，不能宣称所有规模的线程扩展性均为最优。'
         if chinese else 'Default gain ratio favors compact rules but does not lead every task. '
         'No universal scaling/top1 claim is supported.')
    if 'bank_marketing' in names:
        name = 'bank_marketing'
        line(f'Bank pruning: rules {mean(name,"c5_unpruned","rules"):.1f} → '
             f'{mean(name,"c5","rules"):.1f}; accuracy '
             f'{mean(name,"c5_unpruned","accuracy")*100:.2f}% → {mean(name,"c5","accuracy")*100:.2f}%.',
             f'银行营销任务剪枝前后：规则数 {mean(name,"c5_unpruned","rules"):.1f} → '
             f'{mean(name,"c5","rules"):.1f}；准确率 '
             f'{mean(name,"c5_unpruned","accuracy")*100:.2f}% → {mean(name,"c5","accuracy")*100:.2f}%。')
    for name in ('bank_marketing', 'synthetic_100k'):
        if (name, 'c5', 'python', 4) in index:
            speedups = '; '.join(f'{v}: {mean(name,v,"fit_seconds")/mean(name,v,"fit_seconds",threads=4):.2f}× fit, '
                f'{mean(name,v,"predict_seconds")/mean(name,v,"predict_seconds",threads=4):.2f}× predict'
                for v in ('c5','xgboost','lightgbm'))
            translated_speedups = '；'.join(
                f'{label(v)}：训练 {mean(name,v,"fit_seconds")/mean(name,v,"fit_seconds",threads=4):.2f}×，'
                f'推理 {mean(name,v,"predict_seconds")/mean(name,v,"predict_seconds",threads=4):.2f}×'
                for v in ('c5', 'xgboost', 'lightgbm'))
            line(f'Python {name} scaling: {speedups}.',
                 f'Python {label(name)}从单线程到四线程的加速比：{translated_speedups}。')
    for name in ('synthetic_20k', 'synthetic_100k', 'pima'):
        if name in names:
            accuracy = '; '.join(f'{v}: {mean(name,v,"accuracy")*100:.2f}%'
                                for v in ('c5_gini','c5','xgboost','lightgbm'))
            translated_accuracy = '；'.join(f'{label(v)}：{mean(name,v,"accuracy")*100:.2f}%'
                                          for v in ('c5_gini', 'c5', 'xgboost', 'lightgbm'))
            line(f'{name} accuracy: {accuracy}.', f'{label(name)}准确率：{translated_accuracy}。')
    line('预设基尼准则在合成任务效果更好，但规则更多，在皮马糖尿病任务上则更差。不能根据测试结果切换默认准则再把'
         '同一测试集当独立证据。小样本训练准确率接近 100%，需要交叉验证和更强的简化；'
         '银行类别不均衡，应看平衡准确率和 AUC。'
         if chinese else 'Gini improves the synthetic tasks, with more rules, but worsens Pima. '
         'Switching defaults based on these tests needs fresh validation. Small-data training accuracy '
         'near 100% warrants cross-validation and stronger simplification. Bank imbalance makes '
         'balanced accuracy/AUC essential.')
    line('C5 probabilities use a conservative, uncalibrated normalized-vote softmax, unlike competitor '
         'probability transforms. Log loss is retained in raw summaries; calibration is not an advantage '
         'claim. Rule counts/merged condition counts are structural proxies; no human comprehension study, '
         'official C5.0 comparison, Kaggle leaderboard submission or temporal financial backtest was performed.',
         'C5 将归一化投票经 softmax 转换为较保守、未经校准的概率，与竞品的概率变换不同。'
         '原始汇总保留对数损失，不宣称概率校准优势。规则数和合并后的条件数仅作为结构复杂度代理；'
         '未进行用户理解实验、原版 C5.0 对比、Kaggle 榜单提交或金融时间回测。')
    line('## TDD and validation', '## 测试驱动开发与验证')
    line('Tests first failed because C5Classifier did not exist; implementation then made them pass. '
         'Independent rule evaluators reproduce predictions/votes; tests cover multiclass, NaN routing, '
         'sample weights, pruning, early stop, ties, refit/ownership, errors, thread parity and concurrent readers. '
         'Allocation fault injection checks failed refits preserve models. No new coverage exclusion was added.',
         '测试先因 C5Classifier 尚未实现而失败，再由实现使其通过。独立规则求值器能复现预测与投票；'
         '测试覆盖多分类、NaN 路由、样本权重、剪枝、提前停止、平局、重新拟合与资源所有权、'
         '异常、跨线程一致性及并发读取。通过内存分配故障注入，检查重新拟合失败后旧模型仍保留。'
         '没有新增覆盖排除项。')
    line('Release CTest 7/7; Python estimator/inspection/C5 68/68; CLI 16/16; performance/RSS 3/3; '
         'benchmark methodology 12/12. ASan/UBSan/LSan: CTest 7/7 and CLI 16/16; TSan CTest 7/7. '
         'Native instrumented source lines 2434/2434 (100%); Python package statements 359/359 (100%). '
         'CI enforces both 100% gates. Coverage scope: src/*.cc and internal headers, plus python/libtree; '
         'benchmark harnesses and interactive guide JavaScript are outside these gates. '
         'Existing documented platform/exhaustion exclusions remain. Coverage is not proof of absence of bugs.',
         '该次评测记录的验证结果：发布构建 CTest 7/7；Python 模型、结构解释与 C5 测试 68/68；'
         '命令行测试 16/16；性能与常驻内存测试 3/3；评测方法测试 12/12。'
         'ASan/UBSan/LSan 检查：CTest 7/7、命令行 16/16；TSan 检查：CTest 7/7。'
         '原生源码行覆盖 2434/2434（100%）；Python 包语句覆盖 359/359（100%）。'
         '持续集成强制执行这两个 100% 门槛。覆盖范围为 src/*.cc、内部头文件及 python/libtree；'
         '评测框架与交互指南的 JavaScript 不在范围内。保留已有文档说明的平台和资源耗尽排除项。'
         '覆盖率不能证明没有缺陷。')
    line('## Reproduce', '## 复现')
    line("```sh\npython -m pip install '.[dev,benchmark]'\nmake test\nmake sanitize\nmake thread-sanitize\nmake coverage\n"
         "PYTHONPATH=python:. python benchmarks/c5.py --seeds 42 2024 7 --threads 1 4 \\\n"
         "  --repeats 3 --rounds 30 --depth 4 --output benchmarks/c5-results.json\n"
         "PYTHONPATH=python:. python benchmarks/c5_report.py\n```\n")
    line(f"Measured implementation/source SHA-256: `{meta['code_sha256']}`. "
         'Downloaded data/builds/temporary predictions are ignored by git; raw results, report scripts '
         'and these reports are tracked. Complete per-group arithmetic mean/std/min/max values '
         'and per-run metrics remain in the JSON.',
         f"被测实现与源码 SHA-256：`{meta['code_sha256']}`。下载数据、构建产物和临时预测文件由 git 忽略；"
         '原始结果、报告脚本及报告文件纳入版本控制。每组完整的算术平均值、标准差、最小值、最大值'
         '以及每次测量的指标都保留在 JSON 中。')
    return '\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('benchmarks/c5-results.json'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    for path, chinese in (('benchmarks/C5_REPORT.md', False),
                          ('benchmarks/C5_REPORT.zh-CN.md', True), ('benchmarks/C5_REPORT.txt', True)):
        Path(path).write_text(render(data, chinese), encoding='utf-8')
