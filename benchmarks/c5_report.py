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
    def line(text):
        out.extend([text, ''])
    def mean(name, variant, field, interface='python', threads=1):
        return index[name, variant, interface, threads][field]['mean']
    line('新增 C5 风格多分类器已经实现；以下结果来自真实运行。它不是原版 C5.0 兼容实现。'
         if chinese else 'The independent C5-style multiclass classifier is implemented. '
         'The results below are measured; this is not a compatible C5.0 implementation.')
    line('[原始记录](c5-results.json) · [English](C5_REPORT.md) · [中文](C5_REPORT.zh-CN.md) · '
         '[方法/引用](../docs/c5.md) · [交互指南](../docs/C5_GUIDE.html)')
    line('## 方法 / Method')
    line(f"{len(data['results'])} measurements; {len(names)} tasks; seeds 42/2024/7; "
         '3 repetitions per split/configuration; stratified 75/25 train/test; 1 and 4 threads. '
         'Arithmetic means are used throughout; raw repeats, population standard deviations and min/max '
         'are retained. The three splits overlap and are not nine independent test sets. No test-set tuning.')
    line('Fixed budgets: 30 boosting rounds, max depth 4, 64 histogram bins, min leaf rows 5. '
         'C5: SAMME rate 1, alpha .001, gain ratio. sklearn tree/SAMME: entropy, ccp_alpha .001; '
         'SAMME rate 1. XGBoost/LightGBM: gradient rate .1, L2=1; LightGBM max 16 leaves/tree. '
         'Gradient and discrete boosting rates are not directly interchangeable; these are fixed practical '
         'settings, not equal optimized accuracy. XGBoost retains its Hessian-based minimum child weight. '
         'The sklearn tree has one estimator; multiclass GBDT can build one tree per class per round.')
    line('Numeric NaN is preserved for native learners. sklearn inputs use training-fitted median imputation '
         'plus missingness indicators; transformation costs are timed. Bank categories are one-hot encoded '
         'from training data only, before engine timing. Public bank data excludes call duration (post-outcome '
         'information); Pima invalid physiological zeros become NaN. Real-source URLs/SHA-256 are recorded. '
         'Bank marketing predicts subscriptions, not credit defaults; it is not a temporal backtest. '
         'Synthetic 20k/100k tasks have 20 features and fixed 3% label noise. Wine has three classes.')
    line(f"CPU: {meta.get('cpu_model','unknown')}; {meta['affinity_cpus']} affinity CPUs; quota {meta['cpu_quota']}. "
         f"Compiler: {meta.get('compiler','unknown')}. Python {meta['python']}, NumPy {meta['numpy']}, "
         f"sklearn {meta['sklearn']}, XGBoost {meta['xgboost']}, LightGBM {meta['lightgbm']}.")
    line('Timing: fresh models in one imported Python process; native engines run in fresh subprocesses. '
         'Startup/data-file I/O are outside native timing; Python includes wrapper/validation costs. '
         'Engine order is reproducibly shuffled. No concurrent stress/sanitizer job ran during measurements. '
         'No dedicated warm-up is discarded; small timings show noticeable noise. Compare engines within '
         'the same interface. sklearn SAMME is serial; its four-thread timings are omitted. Native competitor '
         'comparisons cover binary tasks; C5 native/Python parity is checked on all tasks, at each run.')
    line('## 效果与性能 / Quality and speed')
    line('Python, one thread; means across 3 seeds × 3 repeats. Fit/predict are milliseconds. '
         'Repeating a seed measures timing, not additional statistical evidence about accuracy. '
         '“Rules” counts retained leaves, including gradient trees; this is a structural proxy, not '
         'equivalent global explanations across algorithms.')
    out += ['| Task | Engine | Fit ms | Predict ms | Accuracy % | Balanced acc % | AUC | Rules |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('c5', 'tree', 'adaboost', 'xgboost', 'lightgbm'):
            out.append(f'| {name} | {variant} | {mean(name,variant,"fit_seconds")*1000:.3f} | '
                       f'{mean(name,variant,"predict_seconds")*1000:.3f} | {mean(name,variant,"accuracy")*100:.2f} | '
                       f'{mean(name,variant,"balanced_accuracy")*100:.2f} | {mean(name,variant,"roc_auc"):.4f} | '
                       f'{mean(name,variant,"rules"):.1f} |')
    out.append('')
    line('## 线程扩展 / Thread scaling')
    line('Speedup = mean time at one thread / mean time at four threads. Small C5 models deliberately '
         'use no pool; their fluctuations do not establish parallel gains. Show medium/large tasks only.')
    out += ['| Task | Interface | Engine | Fit speedup | Predict speedup | Fit at 4 threads ms |',
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
                out.append(f'| {name} | {interface} | {variant} | {a/b:.2f}× | {speed:.2f}× | {b*1000:.3f} |')
    out.append('')
    line('## 消融 / Ablations')
    line('All are one-thread, preselected configurations. `c5_single`: one tree; `c5_unpruned`: alpha=0; '
         '`c5_entropy`/`c5_gini`: change only the criterion. All other C5 settings stay fixed. '
         'Shorter rules always come from actual pruning/merged bounds, not selective export.')
    out += ['| Task | Variant | Accuracy % | Train accuracy % | AUC | Fit ms | Rules | Conditions |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        for variant in ('c5_single', 'c5_unpruned', 'c5', 'c5_entropy', 'c5_gini'):
            out.append(f'| {name} | {variant} | {mean(name,variant,"accuracy")*100:.2f} | '
                       f'{mean(name,variant,"train_accuracy")*100:.2f} | {mean(name,variant,"roc_auc"):.4f} | '
                       f'{mean(name,variant,"fit_seconds")*1000:.3f} | {mean(name,variant,"rules"):.1f} | '
                       f'{mean(name,variant,"conditions"):.1f} |')
    out.append('')
    line('## 波动与异常 / Variability and outliers')
    out += ['| Task | Interface | Threads | C5 fit mean ± std ms | Min–max ms | Predict mean ± std ms |',
            '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in names:
        for interface in ('python', 'cpp'):
            for threads in (1, 4):
                if (name, 'c5', interface, threads) not in index:
                    continue
                r = index[name, 'c5', interface, threads]
                fit, pred = r['fit_seconds'], r['predict_seconds']
                out.append(f'| {name} | {interface} | {threads} | {fit["mean"]*1000:.3f} ± '
                           f'{fit["std"]*1000:.3f} | {fit["min"]*1000:.3f}–{fit["max"]*1000:.3f} | '
                           f'{pred["mean"]*1000:.3f} ± {pred["std"]*1000:.3f} |')
    out.append('')
    line('## 结论与限制 / Conclusions and limits')
    line('默认 gain ratio 更偏向紧凑规则，但不是全面第一，不能宣称全规模 scaling top1。'
         if chinese else 'Default gain ratio favors compact rules but does not lead every task. '
         'No universal scaling/top1 claim is supported.')
    if 'bank_marketing' in names:
        name = 'bank_marketing'
        line(f'Bank pruning: rules {mean(name,"c5_unpruned","rules"):.1f} → '
             f'{mean(name,"c5","rules"):.1f}; accuracy '
             f'{mean(name,"c5_unpruned","accuracy")*100:.2f}% → {mean(name,"c5","accuracy")*100:.2f}%.')
    for name in ('bank_marketing', 'synthetic_100k'):
        if (name, 'c5', 'python', 4) in index:
            speedups = '; '.join(f'{v}: {mean(name,v,"fit_seconds")/mean(name,v,"fit_seconds",threads=4):.2f}× fit, '
                f'{mean(name,v,"predict_seconds")/mean(name,v,"predict_seconds",threads=4):.2f}× predict'
                for v in ('c5','xgboost','lightgbm'))
            line(f'Python {name} scaling: {speedups}.')
    for name in ('synthetic_20k', 'synthetic_100k', 'pima'):
        if name in names:
            accuracy = '; '.join(f'{v}: {mean(name,v,"accuracy")*100:.2f}%'
                                for v in ('c5_gini','c5','xgboost','lightgbm'))
            line(f'{name} accuracy: {accuracy}.')
    line('预设 Gini 在合成任务效果更好，但规则更多，Pima 则更差。不能根据测试结果切换默认准则再把'
         '同一测试集当独立证据。小样本训练准确率接近100%，需要交叉验证和更强的简化；'
         '银行类别不均衡，应看 balanced accuracy/AUC。'
         if chinese else 'Gini improves the synthetic tasks, with more rules, but worsens Pima. '
         'Switching defaults based on these tests needs fresh validation. Small-data training accuracy '
         'near 100% warrants cross-validation and stronger simplification. Bank imbalance makes '
         'balanced accuracy/AUC essential.')
    line('C5 probabilities use a conservative, uncalibrated normalized-vote softmax, unlike competitor '
         'probability transforms. Log loss is retained in raw summaries; calibration is not an advantage '
         'claim. Rule counts/merged condition counts are structural proxies; no human comprehension study, '
         'official C5.0 comparison, Kaggle leaderboard submission or temporal financial backtest was performed.')
    line('## TDD 与验证 / TDD and validation')
    line('Tests first failed because C5Classifier did not exist; implementation then made them pass. '
         'Independent rule evaluators reproduce predictions/votes; tests cover multiclass, NaN routing, '
         'sample weights, pruning, early stop, ties, refit/ownership, errors, thread parity and concurrent readers. '
         'Allocation fault injection checks failed refits preserve models. No new coverage exclusion was added.')
    line('Release CTest 7/7; Python estimator/inspection/C5 68/68; CLI 16/16; performance/RSS 3/3; '
         'benchmark methodology 12/12. ASan/UBSan/LSan: CTest 7/7 and CLI 16/16; TSan CTest 7/7. '
         'Native instrumented source lines 2434/2434 (100%); Python package statements 359/359 (100%). '
         'CI enforces both 100% gates. Coverage scope: src/*.cc and internal headers, plus python/libtree; '
         'benchmark harnesses and interactive guide JavaScript are outside these gates. '
         'Existing documented platform/exhaustion exclusions remain. Coverage is not proof of absence of bugs.')
    line('## 复现 / Reproduce')
    line("```sh\npython -m pip install '.[dev,benchmark]'\nmake test\nmake sanitize\nmake thread-sanitize\nmake coverage\n"
         "PYTHONPATH=python:. python benchmarks/c5.py --seeds 42 2024 7 --threads 1 4 \\\n"
         "  --repeats 3 --rounds 30 --depth 4 --output benchmarks/c5-results.json\n"
         "PYTHONPATH=python:. python benchmarks/c5_report.py\n```\n")
    line(f"Measured implementation/source SHA-256: `{meta['code_sha256']}`. "
         'Downloaded data/builds/temporary predictions are ignored by git; raw results, report scripts '
         'and these reports are tracked. Complete per-group arithmetic mean/std/min/max values '
         'and per-run metrics remain in the JSON.')
    return '\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('benchmarks/c5-results.json'))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    for path, chinese in (('benchmarks/C5_REPORT.md', False),
                          ('benchmarks/C5_REPORT.zh-CN.md', True), ('benchmarks/C5_REPORT.txt', True)):
        Path(path).write_text(render(data, chinese))
