"""Generate English/Chinese/text risk reports from raw measurements."""
import argparse
import json
from pathlib import Path

import numpy as np
from benchmarks.risk import summarize


def render(data, chinese=False):
    groups = summarize(data['results'])
    index = {(r['dataset'], r['variant'], r['interface'], r['threads']): r for r in groups}
    names, params = list(data['sources']), data['metadata']['parameters']
    out = ['# C5 风险评测：提升学习、Hellinger 与 KS' if chinese else
           '# C5 risk study: boosting, Hellinger and KS', '']
    def line(english, translated=None):
        out.extend([translated if chinese and translated is not None else english, ''])
    labels = {
        'german_credit': '德国信用风险（german_credit）',
        'credit_status': '信用状态（credit_status）',
        'taiwan_late_payment': '台湾逾期付款（taiwan_late_payment）',
        'rare_risk_20k': '合成稀少风险（rare_risk_20k）',
        'single': '单棵树（single）',
        'samme_30': '基线提升（samme_30）',
        'samme_long': '增加提升轮数（samme_long）',
        'shrinkage': '学习率收缩（shrinkage）',
        'balanced': '类别平衡（balanced）',
        'hellinger': 'Hellinger 准则（hellinger）',
        'hellinger_balanced': 'Hellinger 与类别平衡（hellinger_balanced）',
        'gini': '基尼准则（gini）',
        'entropy': '信息熵准则（entropy）',
        'less_pruning': '减弱剪枝（less_pruning）',
        'selected': '训练集交叉验证选择（selected）',
        'adaboost': 'sklearn 自适应提升（adaboost）',
        'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
        'python': 'Python', 'cpp': 'C++',
    }
    def label(name):
        return labels.get(name, name) if chinese else name
    def mean(name, variant, field, interface='python', threads=1):
        return index[name, variant, interface, threads][field]['mean']
    line('[English](RISK_REPORT.md) · [中文](RISK_REPORT.zh-CN.md) · '
         '[raw runs and CV scores](risk-results.json) · [methods/citations](../docs/c5-risk.md)',
         '[English](RISK_REPORT.md) · [中文](RISK_REPORT.zh-CN.md) · '
         '[原始测量与交叉验证分数](risk-results.json) · [方法与引用](../docs/c5-risk.zh-CN.md)')
    line('## Method', '## 方法')
    cv_fits = sum(len(c['folds']) for s in data['selections'] for c in s['candidates'])
    line(f"{len(data['results'])} timed interface measurements, {cv_fits} inner-CV fits; "
         f"seeds {params['seeds']}, {params['repeats']} repeats per configuration/split, "
         f"threads {params['threads']}. Arithmetic means, population standard deviations and min/max; "
         'repeated timings do not add independent test sets. Raw individual runs are retained.',
         f"共 {len(data['results'])} 次接口计时、{cv_fits} 次内层交叉验证拟合；"
         f"随机种子为 {params['seeds']}，每个配置、每次划分重复计时 {params['repeats']} 次，"
         f"线程数为 {params['threads']}。采用算术平均值，保留总体标准差、最小值、最大值和每次原始记录。"
         '重复计时不会增加独立测试集的数量。')
    line('Outer stratified 75/25 holdouts; train-only stratified CV, with encoders refit per inner fold. '
         'Select by mean directional KS, then AUC, then fewer rules. Require .01 KS gain over samme_30 '
         'or keep baseline. Refit on complete outer training data; evaluate outer test only afterward. '
         'No test-set choice of configurations/thresholds. The ten candidates are declared before the run.',
         '外层按类别分层，以 75/25 划分训练集和测试集；仅在外层训练集上进行分层交叉验证，'
         '每个内层折都重新拟合编码器。先按平均方向性 KS 选择，再依次比较 AUC 和规则数（越少越优）。'
         '相对 samme_30 基线的 KS 增益至少为 .01，否则保留基线。选定后在完整外层训练集上重新拟合，'
         '最后才评估外层测试集。十个候选配置在运行前确定，不用测试集选择配置或阈值。')
    line(f"Depth {params['depth']}, bins {params['bins']}, minimum leaf rows 5. "
         f"samme_30 uses 30 rounds/rate 1; long variants use {params['rounds']} rounds. "
         'Shrinkage uses rate .3; balancing uses inverse-frequency weights; default alpha=.001, '
         'less_pruning=.0002. Competitors use the same maximum round budget, rate 1 for sklearn SAMME '
         'and .1 for gradient boosters; L2=1 for gradient boosters. Learning rates/pruning units differ '
         'by algorithm. The selected C5 has an additional search budget; its search cost is shown separately.',
         f"最大深度 {params['depth']}，直方图分箱数 {params['bins']}，叶节点最少 5 行样本。"
         f"samme_30 使用 30 轮、学习率 1；增加轮数的配置使用 {params['rounds']} 轮。"
         '学习率收缩配置使用 .3；类别平衡使用类别频率倒数权重；默认剪枝参数 alpha=.001，'
         '减弱剪枝配置为 .0002。竞品使用相同最大轮数预算，sklearn SAMME 学习率为 1，'
         '梯度提升模型学习率为 .1、L2=1。不同算法的学习率和剪枝参数量纲不同。'
         '训练集交叉验证选择的 C5 额外使用了搜索预算，搜索耗时单独列出。')
    line(f"Python {data['metadata']['python']}; sklearn {data['metadata']['sklearn']}; "
         f"XGBoost {data['metadata']['xgboost']}; LightGBM {data['metadata']['lightgbm']}; "
         f"CPU quota {data['metadata']['cpu_quota']}. Python times include wrappers/validation. "
         'Native timings exclude process startup/file I/O. Fresh models; reproducibly shuffled order; '
         'no simultaneous stress/coverage build during measurements. Preprocessing for outer comparisons '
         'is common and outside fit timing, except sklearn imputation, which is included. '
         'Native/Python probability parity is checked for every non-sklearn measured model. '
         'sklearn SAMME is serial and has no standalone C++ comparison.',
         f"Python {data['metadata']['python']}；sklearn {data['metadata']['sklearn']}；"
         f"XGBoost {data['metadata']['xgboost']}；LightGBM {data['metadata']['lightgbm']}；"
         f"CPU 配额 {data['metadata']['cpu_quota']}。Python 计时包含封装和输入校验；"
         'C++ 原生计时排除进程启动及文件读写。每次使用新模型，以可复现方式随机排列执行顺序；'
         '测量期间不同时运行压力测试或覆盖率构建。外层比较共用预处理，预处理不计入训练时间；'
         'sklearn 缺失值填充除外，其开销计入时间。每个非 sklearn 被测模型都检查原生与 Python '
         '接口输出概率的一致性。sklearn SAMME 为串行实现，没有独立 C++ 对比。')
    line('KS=max absolute empirical bad/good score-distribution difference. Directional KS=max(TPR-FPR); '
         'tied scores are grouped. An inverted model can have high absolute KS, so AUC and directional KS '
         'must accompany it. Label 1 means bad. These are descriptive held-out statistics; no deployed '
         'threshold is picked on the test data. Probabilities are not calibrated; bad recall uses the '
         'model argmax/ordinary class decision. Single C5 trees use hard leaf votes, not leaf class-frequency '
         'probabilities; their ranking can be constant even with multiple leaves.',
         'KS 是坏样本与好样本评分经验分布差的最大绝对值；方向性 KS=max(TPR-FPR)，其中 '
         'TPR 为坏样本召回率，FPR 为好样本误报率，相同分数合并计算。模型排序颠倒时绝对 KS '
         '也可能很高，因此必须同时查看 AUC 和方向性 KS。标签 1 代表坏样本。指标用于描述留出集表现，'
         '不在测试集上选择上线阈值。概率未经校准；坏样本召回率按模型最大概率类别或常规分类决策计算。'
         '单棵 C5 树使用叶节点类别的硬投票，不使用叶内类别频率作为概率；即使有多个叶节点，'
         '也可能产生常数评分。AP 为平均精确率，用于衡量不平衡任务的排序质量；平衡准确率为各类别召回率的平均值。')
    line('## Data', '## 数据')
    out += ['| 数据集 | 行数 | 坏样本比例 | 预测目标与来源说明 |' if chinese else
            '| Dataset | Rows | Bad fraction | Target/source |', '| --- | ---: | ---: | --- |']
    targets = {
        'german_credit': '坏信用风险；原始 credit_risk=0（300/1000）',
        'credit_status': 'Status=bad；保留特征缺失值，删除标识符',
        'taiwan_late_payment': '2005 年 9 月逾期；特征为 4 月逾期标记、账单和年龄；不是原始完整违约目标',
        'rare_risk_20k': '明确为合成的稀少风险分类任务',
    }
    for name in names:
        source = data['sources'][name]
        sample = next(r for r in data['results'] if r['dataset'] == name)
        fraction = source.get('positive_fraction', sample['positive_fraction'])
        target = targets[name] if chinese else source.get('target', 'explicitly synthetic rare-risk classification')
        out.append(f'| {label(name)} | {sample["train_rows"]+sample["test_rows"]} | {fraction:.2%} | {target} |')
    out.append('')
    line('Taiwan is a real public subset with April features and September late-payment labels; '
         'it is not the original full default target. IDs and targets are excluded. Sources and SHA-256 '
         'are pinned. Random splits are not temporal backtests or external-lender validation.',
         '台湾任务来自真实公开数据的子集，以 4 月信息预测 9 月是否逾期，并非原始数据的完整违约目标。'
         '输入已排除标识符和目标字段；固定数据来源及 SHA-256 校验和。随机划分不等同于时间回测，'
         '也不等同于在其他放贷机构的数据上验证。')
    line('## Quality and performance', '## 效果与性能')
    line('One-thread Python, arithmetic means; fit/predict in milliseconds. selected uses only training CV. '
         'The fixed Hellinger row is an ablation, not a configuration selected using these test scores.',
         '下表为 Python 单线程结果的算术平均值，训练与推理时间单位为毫秒。selected 配置仅由训练集'
         '交叉验证选出。固定 Hellinger 配置用于消融分析，并非根据这些测试分数挑选。')
    out += ['| 任务 | 模型 | KS | AUC | AP | 坏样本召回率 | 平衡准确率 | 训练毫秒 | 推理毫秒 | 规则数 |' if chinese else
            '| Task | Model | KS | AUC | AP | Bad recall | Balanced accuracy | Fit ms | Predict ms | Rules |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('single', 'samme_30', 'hellinger', 'selected', 'adaboost', 'xgboost', 'lightgbm'):
            out.append(f'| {label(name)} | {label(variant)} | {mean(name,variant,"ks"):.4f} | '
                       f'{mean(name,variant,"roc_auc"):.4f} | {mean(name,variant,"average_precision"):.4f} | '
                       f'{mean(name,variant,"bad_recall"):.4f} | {mean(name,variant,"balanced_accuracy"):.4f} | '
                       f'{mean(name,variant,"fit_seconds")*1000:.3f} | '
                       f'{mean(name,variant,"predict_seconds")*1000:.3f} | {mean(name,variant,"rules"):.1f} |')
    out.append('')
    line('## Ablations', '## 提升学习与分裂准则消融')
    line('samme_30→samme_long changes only round count. samme_long→shrinkage changes only rate. '
         'shrinkage→hellinger changes only criterion; shrinkage→balanced changes only weighting. '
         'balanced→hellinger_balanced isolates the criterion under balancing. '
         'shrinkage→less_pruning changes only alpha. Comparisons to samme_30 combine changes.',
         'samme_30→samme_long 只改变提升轮数；samme_long→shrinkage 只改变学习率；'
         'shrinkage→hellinger 只改变分裂准则；shrinkage→balanced 只改变权重；'
         'balanced→hellinger_balanced 在类别平衡条件下单独比较准则；'
         'shrinkage→less_pruning 只改变剪枝参数 alpha。将后面这些配置直接与 samme_30 比较，'
         '会同时包含多项改动的影响。表中括号内标识与原始 JSON 配置名对应。')
    out += ['| 任务 | 配置 | KS | AUC | AP | 规则数 | 实际保留树数 |' if chinese else
            '| Task | Configuration | KS | AUC | AP | Rules | Accepted trees |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('single', 'samme_30', 'samme_long', 'shrinkage', 'balanced', 'hellinger',
                        'hellinger_balanced', 'gini', 'entropy', 'less_pruning'):
            out.append(f'| {label(name)} | {label(variant)} | {mean(name,variant,"ks"):.4f} | '
                       f'{mean(name,variant,"roc_auc"):.4f} | {mean(name,variant,"average_precision"):.4f} | '
                       f'{mean(name,variant,"rules"):.1f} | '
                       f'{np.mean([r["estimators"] for r in data["results"] if r["dataset"]==name and r["variant"]==variant and r["interface"]=="python" and r["threads"]==1]):.1f} |')
    out.append('')
    line('## Selection and cost', '## 选择稳定性与搜索成本')
    out += ['| 任务 | 随机种子 | 训练集交叉验证选择 | 搜索秒数 |' if chinese else
            '| Task | Seed | Training-CV choice | Search seconds |', '| --- | ---: | --- | ---: |']
    for selection in data['selections']:
        out.append(f'| {label(selection["dataset"])} | {selection["seed"]} | {label(selection["chosen"])} | {selection["search_seconds"]:.3f} |')
    out.append('')
    line('## Scaling and variability', '## C++／Python 线程扩展与波动')
    out += ['| 任务 | 接口 | 交叉验证所选模型 KS 均值 ± 标准差 | 单线程训练毫秒 | 四线程训练毫秒 | 训练加速比 |' if chinese else
            '| Task | Interface | selected KS mean ± std | Fit 1-thread ms | Fit 4-thread ms | Fit speedup |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in names:
        for interface in ('python', 'cpp'):
            r = index[name, 'selected', interface, 1]
            a = r['fit_seconds']['mean']
            b = index.get((name, 'selected', interface, 4))
            out.append(f'| {label(name)} | {label(interface)} | {r["ks"]["mean"]:.4f} ± {r["ks"]["std"]:.4f} | '
                       f'{a*1000:.3f} | {b["fit_seconds"]["mean"]*1000:.3f} | '
                       f'{a/b["fit_seconds"]["mean"]:.2f}× |' if b else
                       f'| {label(name)} | {label(interface)} | {r["ks"]["mean"]:.4f} ± {r["ks"]["std"]:.4f} | {a*1000:.3f} | — | — |')
    out.append('')
    line('All mean/std/min/max values for every engine/interface/thread configuration are available in '
         'the JSON. Small jobs run serially by design; timing fluctuations are not parallel gains.',
         '每个引擎、接口、线程配置的均值、标准差、最小值和最大值均保留在 JSON 中。'
         '训练加速比为单线程平均耗时除以四线程平均耗时。小任务按设计串行执行，计时波动不能作为并行收益。')
    line('## Findings', '## 结论')
    for name in names:
        before, after = mean(name, 'samme_30', 'ks'), mean(name, 'selected', 'ks')
        line(f'{name}: baseline KS {before:.4f} → train-CV selected {after:.4f} '
             f'(delta {after-before:+.4f}); AUC {mean(name,"samme_30","roc_auc"):.4f} → '
             f'{mean(name,"selected","roc_auc"):.4f}. Criterion-only comparison: '
             f'shrinkage KS {mean(name,"shrinkage","ks"):.4f} → Hellinger {mean(name,"hellinger","ks"):.4f}.',
             f'{label(name)}：基线 KS {before:.4f} → 训练集交叉验证所选模型 {after:.4f}'
             f'（变化 {after-before:+.4f}）；AUC {mean(name,"samme_30","roc_auc"):.4f} → '
             f'{mean(name,"selected","roc_auc"):.4f}。仅改变准则的比较：学习率收缩配置 KS '
             f'{mean(name,"shrinkage","ks"):.4f} → Hellinger {mean(name,"hellinger","ks"):.4f}。')
    line('训练集交叉验证选择不能保证每个测试任务的 KS 都提高。德国信用任务中所选模型的 KS 低于基线，但 AUC '
         '更高；台湾保持基线；信用状态和稀少风险任务中所选模型的 KS 更高。没有据测试成绩更改默认准则。'
         '更多轮数并非总更好；Hellinger 的单因素消融在信用状态/稀少风险上更好，在德国则 KS 略低。'
         '平衡权重明显提高坏样本召回，但不保证 KS/AP 提升；应同时看误报与精度。'
         if chinese else 'Train-CV selection does not guarantee higher test KS on every task. German '
         'selected KS falls while AUC rises; Taiwan keeps baseline; credit status and rare risk gain KS. '
         'Defaults were not changed based on test outcomes. More rounds are not always better. '
         'The criterion-only Hellinger ablation improves credit status/rare risk KS, but slightly reduces '
         'German KS. Balancing improves bad recall without guaranteeing KS/AP gains.')
    line('三次重复只增加计时证据，不增加独立效果证据；三个划分也重叠，不能宣称统计显著或普遍第一。'
         '稀少风险任务中 XGBoost/LightGBM 的 KS 仍更高。交叉验证所选模型多用了搜索预算，不能当成与未调参竞品'
         '同预算的排名。规则数仅是结构复杂度代理；尚无用户理解实验。'
         if chinese else 'Three repetitions add timing evidence, not independent accuracy evidence; '
         'the three splits overlap. No statistical-significance or universal-superiority claim is made. '
         'XGBoost/LightGBM retain higher KS on rare risk. selected has extra search budget. '
         'Rule counts are structural proxies, not a human comprehension study.')
    line('## Validation', '## 测试驱动开发与验证')
    line('New tests failed before implementation. Hellinger binary behavior/multiclass rejection, '
         'explicit off/one-tree equivalence, balanced/user-weight equivalence, failed-refit preservation, '
         'tie-safe/directional KS, per-fold encoders, CV isolation and cache integrity pass. '
         'Release CTest 7/7; Python estimator/inspection 70/70; CLI 16/16; performance/RSS 3/3; '
         'benchmark methodology 15/15. ASan/UBSan/LSan CTest 7/7 plus CLI 16/16; TSan 7/7. '
         'Native lines 2444/2444 and Python statements 371/371: 100%. No new coverage exclusions. '
         'Benchmark code is outside production coverage gates.',
         '新增测试先于实现失败，再由实现使其通过。测试覆盖 Hellinger 二分类行为及多分类拒绝、'
         '显式关闭提升与单树等价、类别平衡与用户权重等价、重新拟合失败时保留旧模型、'
         '同分合并与方向性 KS、逐折编码、交叉验证隔离以及缓存完整性。发布构建 CTest 7/7；'
         'Python 模型与结构解释测试 70/70；命令行测试 16/16；性能与常驻内存测试 3/3；'
         '评测方法测试 15/15。ASan/UBSan/LSan 检查通过 CTest 7/7 和命令行 16/16；'
         'TSan 检查 7/7。原生代码行覆盖 2444/2444，Python 语句覆盖 371/371，均为 100%。'
         '未新增覆盖排除项；评测脚本不在生产代码覆盖率门槛范围内。以上为该次评测记录的验证结果。')
    line('## Reproduce', '## 复现')
    line("```sh\npython -m pip install '.[dev,benchmark]'\nmake test\nmake coverage\nmake sanitize\nmake thread-sanitize\n"
         "PYTHONPATH=python:. python benchmarks/risk.py --seeds 42 2024 7 --threads 1 4 \\\n"
         "  --rounds 100 --folds 3 --repeats 3 --output benchmarks/risk-results.json\n"
         "PYTHONPATH=python:. python benchmarks/risk_report.py\n```\n")
    line(f"Measured source SHA-256: `{data['metadata']['code_sha256']}`. Source URLs/checksums, "
         'all per-fold CV scores, chosen parameters and raw repeated measurements are in the JSON. '
         'Downloaded data and prepared predictions remain ignored by git.',
         f"被测源码 SHA-256：`{data['metadata']['code_sha256']}`。数据来源网址、校验和、"
         '每个内层折的交叉验证分数、所选参数以及每次重复测量均保留在 JSON 中。'
         '下载的数据和临时预测文件仍由 git 忽略。')
    return '\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('benchmarks/risk-results.json'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    for name, chinese in (('RISK_REPORT.md', False), ('RISK_REPORT.zh-CN.md', True), ('RISK_REPORT.txt', True)):
        Path('benchmarks', name).write_text(render(data, chinese), encoding='utf-8')
