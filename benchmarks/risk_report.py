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
    out = ['# C5 风险评测：Boosting、Hellinger 与 KS' if chinese else
           '# C5 risk study: boosting, Hellinger and KS', '']
    def line(s):
        out.extend([s, ''])
    def mean(name, variant, field, interface='python', threads=1):
        return index[name, variant, interface, threads][field]['mean']
    line('[English](RISK_REPORT.md) · [中文](RISK_REPORT.zh-CN.md) · '
         '[raw runs and CV scores](risk-results.json) · [methods/citations](../docs/c5-risk.md)')
    line('## 方法 / Method')
    cv_fits = sum(len(c['folds']) for s in data['selections'] for c in s['candidates'])
    line(f"{len(data['results'])} timed interface measurements, {cv_fits} inner-CV fits; "
         f"seeds {params['seeds']}, {params['repeats']} repeats per configuration/split, "
         f"threads {params['threads']}. Arithmetic means, population standard deviations and min/max; "
         'repeated timings do not add independent test sets. Raw individual runs are retained.')
    line('Outer stratified 75/25 holdouts; train-only stratified CV, with encoders refit per inner fold. '
         'Select by mean directional KS, then AUC, then fewer rules. Require .01 KS gain over samme_30 '
         'or keep baseline. Refit on complete outer training data; evaluate outer test only afterward. '
         'No test-set choice of configurations/thresholds. The ten candidates are declared before the run.')
    line(f"Depth {params['depth']}, bins {params['bins']}, minimum leaf rows 5. "
         f"samme_30 uses 30 rounds/rate 1; long variants use {params['rounds']} rounds. "
         'Shrinkage uses rate .3; balancing uses inverse-frequency weights; default alpha=.001, '
         'less_pruning=.0002. Competitors use the same maximum round budget, rate 1 for sklearn SAMME '
         'and .1 for gradient boosters; L2=1 for gradient boosters. Learning rates/pruning units differ '
         'by algorithm. The selected C5 has an additional search budget; its search cost is shown separately.')
    line(f"Python {data['metadata']['python']}; sklearn {data['metadata']['sklearn']}; "
         f"XGBoost {data['metadata']['xgboost']}; LightGBM {data['metadata']['lightgbm']}; "
         f"CPU quota {data['metadata']['cpu_quota']}. Python times include wrappers/validation. "
         'Native timings exclude process startup/file I/O. Fresh models; reproducibly shuffled order; '
         'no simultaneous stress/coverage build during measurements. Preprocessing for outer comparisons '
         'is common and outside fit timing, except sklearn imputation, which is included. '
         'Native/Python probability parity is checked for every non-sklearn measured model. '
         'sklearn SAMME is serial and has no standalone C++ comparison.')
    line('KS=max absolute empirical bad/good score-distribution difference. Directional KS=max(TPR-FPR); '
         'tied scores are grouped. An inverted model can have high absolute KS, so AUC and directional KS '
         'must accompany it. Label 1 means bad. These are descriptive held-out statistics; no deployed '
         'threshold is picked on the test data. Probabilities are not calibrated; bad recall uses the '
         'model argmax/ordinary class decision. Single C5 trees use hard leaf votes, not leaf class-frequency '
         'probabilities; their ranking can be constant even with multiple leaves.')
    line('## 数据 / Data')
    out += ['| Dataset | Rows | Bad fraction | Target/source |', '| --- | ---: | ---: | --- |']
    for name in names:
        source = data['sources'][name]
        sample = next(r for r in data['results'] if r['dataset'] == name)
        fraction = source.get('positive_fraction', sample['positive_fraction'])
        out.append(f'| {name} | {sample["train_rows"]+sample["test_rows"]} | {fraction:.2%} | '
                   f'{source.get("target", "explicitly synthetic rare-risk classification")} |')
    out.append('')
    line('Taiwan is a real public subset with April features and September late-payment labels; '
         'it is not the original full default target. IDs and targets are excluded. Sources and SHA-256 '
         'are pinned. Random splits are not temporal backtests or external-lender validation.')
    line('## 效果与性能 / Quality and performance')
    line('One-thread Python, arithmetic means; fit/predict in milliseconds. selected uses only training CV. '
         'The fixed Hellinger row is an ablation, not a configuration selected using these test scores.')
    out += ['| Task | Model | KS | AUC | AP | Bad recall | Balanced accuracy | Fit ms | Predict ms | Rules |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('single', 'samme_30', 'hellinger', 'selected', 'adaboost', 'xgboost', 'lightgbm'):
            out.append(f'| {name} | {variant} | {mean(name,variant,"ks"):.4f} | '
                       f'{mean(name,variant,"roc_auc"):.4f} | {mean(name,variant,"average_precision"):.4f} | '
                       f'{mean(name,variant,"bad_recall"):.4f} | {mean(name,variant,"balanced_accuracy"):.4f} | '
                       f'{mean(name,variant,"fit_seconds")*1000:.3f} | '
                       f'{mean(name,variant,"predict_seconds")*1000:.3f} | {mean(name,variant,"rules"):.1f} |')
    out.append('')
    line('## Boosting 与准则消融 / Ablations')
    line('samme_30→samme_long changes only round count. samme_long→shrinkage changes only rate. '
         'shrinkage→hellinger changes only criterion; shrinkage→balanced changes only weighting. '
         'balanced→hellinger_balanced isolates the criterion under balancing. '
         'shrinkage→less_pruning changes only alpha. Comparisons to samme_30 combine changes.')
    out += ['| Task | Configuration | KS | AUC | AP | Rules | Accepted trees |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('single', 'samme_30', 'samme_long', 'shrinkage', 'balanced', 'hellinger',
                        'hellinger_balanced', 'gini', 'entropy', 'less_pruning'):
            out.append(f'| {name} | {variant} | {mean(name,variant,"ks"):.4f} | '
                       f'{mean(name,variant,"roc_auc"):.4f} | {mean(name,variant,"average_precision"):.4f} | '
                       f'{mean(name,variant,"rules"):.1f} | '
                       f'{np.mean([r["estimators"] for r in data["results"] if r["dataset"]==name and r["variant"]==variant and r["interface"]=="python" and r["threads"]==1]):.1f} |')
    out.append('')
    line('## 选择稳定性与搜索成本 / Selection and cost')
    out += ['| Task | Seed | Training-CV choice | Search seconds |', '| --- | ---: | --- | ---: |']
    for selection in data['selections']:
        out.append(f'| {selection["dataset"]} | {selection["seed"]} | {selection["chosen"]} | {selection["search_seconds"]:.3f} |')
    out.append('')
    line('## C++／Python 扩展与波动 / Scaling and variability')
    out += ['| Task | Interface | selected KS mean ± std | Fit 1-thread ms | Fit 4-thread ms | Fit speedup |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in names:
        for interface in ('python', 'cpp'):
            r = index[name, 'selected', interface, 1]
            a = r['fit_seconds']['mean']
            b = index.get((name, 'selected', interface, 4))
            out.append(f'| {name} | {interface} | {r["ks"]["mean"]:.4f} ± {r["ks"]["std"]:.4f} | '
                       f'{a*1000:.3f} | {b["fit_seconds"]["mean"]*1000:.3f} | '
                       f'{a/b["fit_seconds"]["mean"]:.2f}× |' if b else
                       f'| {name} | {interface} | {r["ks"]["mean"]:.4f} ± {r["ks"]["std"]:.4f} | {a*1000:.3f} | — | — |')
    out.append('')
    line('All mean/std/min/max values for every engine/interface/thread configuration are available in '
         'the JSON. Small jobs run serially by design; timing fluctuations are not parallel gains.')
    line('## 结论 / Findings')
    for name in names:
        before, after = mean(name, 'samme_30', 'ks'), mean(name, 'selected', 'ks')
        line(f'{name}: baseline KS {before:.4f} → train-CV selected {after:.4f} '
             f'(delta {after-before:+.4f}); AUC {mean(name,"samme_30","roc_auc"):.4f} → '
             f'{mean(name,"selected","roc_auc"):.4f}. Criterion-only comparison: '
             f'shrinkage KS {mean(name,"shrinkage","ks"):.4f} → Hellinger {mean(name,"hellinger","ks"):.4f}.')
    line('训练集 CV 选择没有保证每个测试任务的 KS 都提高。德国信用的 selected KS 低于基线，但 AUC '
         '更高；台湾保持基线；信用状态和稀少风险任务的 selected KS 更高。没有据测试成绩更改默认准则。'
         '更多轮数并非总更好；Hellinger 的单因素消融在信用状态/稀少风险上更好，在德国则 KS 略低。'
         '平衡权重明显提高坏样本召回，但不保证 KS/AP 提升；应同时看误报与精度。'
         if chinese else 'Train-CV selection does not guarantee higher test KS on every task. German '
         'selected KS falls while AUC rises; Taiwan keeps baseline; credit status and rare risk gain KS. '
         'Defaults were not changed based on test outcomes. More rounds are not always better. '
         'The criterion-only Hellinger ablation improves credit status/rare risk KS, but slightly reduces '
         'German KS. Balancing improves bad recall without guaranteeing KS/AP gains.')
    line('三次重复只增加计时证据，不增加独立效果证据；三个划分也重叠，不能宣称统计显著或普遍第一。'
         '稀少风险任务中 XGBoost/LightGBM 的 KS 仍更高。selected 多用了搜索预算，不能当成与未调参竞品'
         '同预算的排名。规则数仅是结构复杂度代理；尚无用户理解实验。'
         if chinese else 'Three repetitions add timing evidence, not independent accuracy evidence; '
         'the three splits overlap. No statistical-significance or universal-superiority claim is made. '
         'XGBoost/LightGBM retain higher KS on rare risk. selected has extra search budget. '
         'Rule counts are structural proxies, not a human comprehension study.')
    line('## TDD 与验证 / Validation')
    line('New tests failed before implementation. Hellinger binary behavior/multiclass rejection, '
         'explicit off/one-tree equivalence, balanced/user-weight equivalence, failed-refit preservation, '
         'tie-safe/directional KS, per-fold encoders, CV isolation and cache integrity pass. '
         'Release CTest 7/7; Python estimator/inspection 70/70; CLI 16/16; performance/RSS 3/3; '
         'benchmark methodology 15/15. ASan/UBSan/LSan CTest 7/7 plus CLI 16/16; TSan 7/7. '
         'Native lines 2444/2444 and Python statements 371/371: 100%. No new coverage exclusions. '
         'Benchmark code is outside production coverage gates.')
    line('## 复现 / Reproduce')
    line("```sh\npython -m pip install '.[dev,benchmark]'\nmake test\nmake coverage\nmake sanitize\nmake thread-sanitize\n"
         "PYTHONPATH=python:. python benchmarks/risk.py --seeds 42 2024 7 --threads 1 4 \\\n"
         "  --rounds 100 --folds 3 --repeats 3 --output benchmarks/risk-results.json\n"
         "PYTHONPATH=python:. python benchmarks/risk_report.py\n```\n")
    line(f"Measured source SHA-256: `{data['metadata']['code_sha256']}`. Source URLs/checksums, "
         'all per-fold CV scores, chosen parameters and raw repeated measurements are in the JSON. '
         'Downloaded data and prepared predictions remain ignored by git.')
    return '\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('benchmarks/risk-results.json'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    for name, chinese in (('RISK_REPORT.md', False), ('RISK_REPORT.zh-CN.md', True), ('RISK_REPORT.txt', True)):
        Path('benchmarks', name).write_text(render(data, chinese))
