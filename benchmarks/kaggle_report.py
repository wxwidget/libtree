"""Generate a Chinese report and machine-readable summaries from actual runs."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import argparse
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "kaggle-results.json"))
    parser.add_argument("--prefix", default="kaggle", choices=["kaggle", "comparison", "optimized"])
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    engines = data["split_reports"][0]["metadata"].get("engines", ["libtree", "xgboost"])
    markdown_name = {"kaggle": "KAGGLE_REPORT.zh-CN.md", "comparison": "COMPARISON_REPORT.zh-CN.md",
                     "optimized": "OPTIMIZATION_REPORT.zh-CN.md"}[args.prefix]
    groups = defaultdict(list)
    all_records = []
    for split in data['split_reports']:
        for row in split['results']:
            groups[(row['dataset'], row['language'], row['engine'])].append(row)
            all_records.append(row)
    summaries = []
    for (dataset, language, engine), rows in groups.items():
        metrics = {}
        for name in rows[0]['metrics']:
            values = [row['metrics'][name] for row in rows]
            metrics[name] = {'mean': mean(values), 'std': stdev(values) if len(values) > 1 else 0}
        timings = [run for row in rows for run in row['raw_runs']]
        summaries.append({
            'dataset': dataset, 'language': language, 'engine': engine,
            'metrics': metrics, 'fit_seconds_median': median(r['fit_seconds'] for r in timings),
            'predict_seconds_median': median(r['predict_seconds'] for r in timings),
            'peak_rss_mib_median': median(r['peak_rss_kib'] / 1024 for r in timings),
            'fit_seconds_min': min(r['fit_seconds'] for r in timings),
            'fit_seconds_max': max(r['fit_seconds'] for r in timings),
            'all_checks_passed': all(all(row['checks'].values()) for row in rows),
        })
    (ROOT / f'{args.prefix}-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    lookup = {(r['dataset'], r['language'], r['engine']): r for r in summaries}
    first = data['split_reports'][0]
    metadata = first['metadata']
    parameters = metadata['parameters']
    names = list(first['sources'])
    runs = sum(len(row['raw_runs']) for row in all_records)
    def fmt(metric):
        return f"{metric['mean']:.4f} ± {metric['std']:.4f}"
    lines = [
        '# LibTree 训练与推理优化：三方实测' if args.prefix == 'optimized' else ('# Kaggle 数据集 GBDT 三方比较' if len(engines) == 3 else '# Kaggle 数据集 GBDT 测试报告'), '',
        f"测试时间（北京时间）：{datetime.fromisoformat(data['split_reports'][-1]['metadata']['date_utc'].replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')}。", '',
        f"完成 **{len(names)} 个数据集 × {len(data['seeds'])} 个不同划分 × 2 个接口 × {len(engines)} 个引擎 × {data['timing_repeats_per_split']} 次计时 = {runs} 次实测**。",
        '、'.join(engines) + ' 使用相同训练/测试数据；分别通过 C++ 原生调用和 Python 接口执行。',
        '每次运行的预测有限、优于简单基线且重复运行确定；同一引擎的 C++/Python 预测一致性检查通过。', '',
        'Kaggle 官方 API 被本环境网络代理拒绝（连接隧道 HTTP 403），因此从公开镜像下载对应数据。',
        '校验了镜像文件 SHA256；未独立核验其与 Kaggle 官方压缩包逐字节相同。没有登录 Kaggle 或提交榜单。', '',
        '## 数据集与处理', '',
        '| 数据集 | 清理后样本 | 编码后特征 | 任务 | 特征缺失值数 | 数据处理 |',
        '| --- | ---: | ---: | --- | ---: | --- |',
    ]
    descriptions = {
        'titanic': '只用 7 个常用特征；删除 ID、姓名等',
        'insurance': '6 个输入特征；费用原值回归',
        'pima': '5 项生理指标的无效零值转 NaN；怀孕次数的 0 保留',
        'telco': '删除 customerID；11 个空白 TotalCharges 转 NaN',
        'wine': '先删除 240 条完全重复记录，再划分；quality 回归',
    }
    for name in names:
        source = first['sources'][name]
        row = next(r for r in first['results'] if r['dataset'] == name)
        binary = row['metric_name'] == 'roc_auc'
        lines.append(f"| {name} | {source['rows_after_cleaning']} | {row['features']} | {'二分类' if binary else '回归'} | {source['feature_missing_values']} | {descriptions[name]} |")
    lines += ['',
        '每个划分为 75% 训练、25% 测试，随机种子：' + ', '.join(map(str, data['seeds'])) + '。二分类采用分层抽样。',
        '类别编码仅在训练集拟合，忽略测试集新类别；数值缺失保留为 NaN，不学习全数据填充值。',
        '未使用测试集选择参数。红酒去重后没有相同输入特征的重复行。', '',
        '## 效果：跨三个划分的均值 ± 样本标准差', '',
        'AUC 越高越好；RMSE 越低越好。C++ 与 Python 的效果一致，因此本表列 C++ 汇总，原始报告保留所有接口。', '',
        '| 数据集 | 指标 | ' + ' | '.join(engines) + ' |',
        '| --- | --- | ' + ' | '.join(['---:'] * len(engines)) + ' |',
    ]
    for name in names:
        a, b = lookup[(name, 'cpp', 'libtree')], lookup[(name, 'cpp', 'xgboost')]
        metric = 'roc_auc' if 'roc_auc' in a['metrics'] else 'rmse'
        lines.append(f"| {name} | {metric} | " + " | ".join(fmt(lookup[(name, 'cpp', e)]['metrics'][metric]) for e in engines) + " |")
    lines += ['', '二分类的附加指标：阈值为 0.5 的准确率、PR AUC（average precision）、log loss。', '',
        '| 数据集 | 引擎 | Accuracy | PR AUC | Log loss |',
        '| --- | --- | ---: | ---: | ---: |']
    for name in names:
        for engine in engines:
            r = lookup[(name, 'cpp', engine)]
            if 'roc_auc' in r['metrics']:
                lines.append(f"| {name} | {engine} | {fmt(r['metrics']['accuracy'])} | {fmt(r['metrics']['average_precision'])} | {fmt(r['metrics']['log_loss'])} |")
    lines += ['', '回归附加指标（MAE 越低越好，R² 越高越好）：', '',
        '| 数据集 | 引擎 | MAE | R² |', '| --- | --- | ---: | ---: |']
    for name in names:
        for engine in engines:
            r = lookup[(name, 'cpp', engine)]
            if 'rmse' in r['metrics']:
                lines.append(f"| {name} | {engine} | {fmt(r['metrics']['mae'])} | {fmt(r['metrics']['r2'])} |")
    lines += ['', '## 性能与内存', '',
        f"每个组合的时间和内存取 {len(data['seeds'])} 个划分、每划分 {data['timing_repeats_per_split']} 次，共 {len(data['seeds'])*data['timing_repeats_per_split']} 次运行的中位数。每次启动独立进程。内存采用执行后地址空间的 Linux VmHWM，避免继承父进程的高水位。", '',
        '| 数据集 | 接口 | 引擎 | 训练 ms | 预测 ms | 峰值 RSS MiB |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for name in names:
        for language in ('cpp', 'python'):
            for engine in engines:
                r = lookup[(name, language, engine)]
                lines.append(f"| {name} | {language} | {engine} | {r['fit_seconds_median']*1000:.2f} | {r['predict_seconds_median']*1000:.3f} | {r['peak_rss_mib_median']:.1f} |")
    telco_lib = lookup[('telco', 'cpp', 'libtree')]['fit_seconds_median']
    telco_xgb = lookup[('telco', 'cpp', 'xgboost')]['fit_seconds_median']
    lines += ['', '## 测试结论与限制', '',
        '固定参数下，效果优劣应查看逐数据集均值与划分波动，不能据三次划分宣称统计显著。',
        f"Telco 是本组较大的数据集，C++ LibTree 训练中位数为 {telco_lib*1000:.1f} ms，XGBoost 为 {telco_xgb*1000:.1f} ms；LibTree 耗时约为 XGBoost 的 {telco_lib/telco_xgb:.2f} 倍。",
        '小型数据集的接口开销占比更高，毫秒级差异容易受共享 CPU 调度影响。原始文件保存了每次计时，可检查波动。', '',
        f"**参数与公平性**：{parameters['trees']} 棵树、深度 {parameters['depth']}、{parameters['bins']} 分箱、学习率 {parameters['rate']}、L2={parameters['l2']}、最小增益 {parameters['min_gain']}；CPU 单线程，无 GPU。",
        f"LibTree 最少叶样本数为 {parameters['min_leaf']}；XGBoost 默认最小 Hessian 权重为 1。" + (f"LightGBM 为 leaf-wise，最少叶样本数 {parameters['min_leaf']}、最大深度 {parameters['depth']}、num_leaves={2**parameters['depth']}。" if 'lightgbm' in engines else '') + '分箱、树生长和叶约束并不完全相同，没有做超参数搜索。', '',
        '**计时边界**：排除下载、CSV 读取、预处理、Python 导入和进程启动；训练包含分箱/DMatrix 构建与标签传递。',
        '预测包含各接口必要转换：C++ XGBoost 构造测试 DMatrix，Python 使用 estimator 预测路径。比较的是端到端接口，不能视为纯树遍历吞吐量。', '',
        '**内存边界**：RSS 为整个进程峰值，不是模型增量。Python 子进程均导入三种库；C++ 进程无 Python 运行时。',
        'RSS 不能替代泄漏检查。本次额外通过四个基准准备测试（划分可复现、测试专有类别不被学习、损坏缓存被拒绝、执行进程内存计数有效）。',
        '本轮实现和 ASan/UBSan/LSan、覆盖率验证见 [验证记录](../docs/validation.md)。', '',
        '**外推限制**：随机留出集不是官方 Kaggle 测试集；Telco/Titanic 可能存在家庭或客户结构，未做分组或时间划分；不代表真实上线表现。',
        '数据规模 768–7043 行，未包含百万行、稀疏、多分类、GPU或多线程任务。', '',
        '## 数据来源与复现', '',
        f"环境：{metadata['cpu']}；{metadata['compiler']}；Python {metadata['python']}；NumPy {metadata['numpy']}；sklearn {metadata['sklearn']}；XGBoost {metadata['xgboost']}；LightGBM {metadata.get('lightgbm', '未参与历史比较')}。", '',
        '```sh', 'cd /workspace/libtree', '. /workspace/libtree-venv/bin/activate',
        'python -m pip install -r benchmarks/requirements.txt', 'make all',
        'PYTHONPATH=python:. python -m pytest tests/test_benchmark.py',
        f'PYTHONPATH=python python benchmarks/kaggle.py --engines {" ".join(engines)} --seeds {" ".join(map(str, data["seeds"]))} --repeats {data["timing_repeats_per_split"]} --output benchmarks/{args.prefix}-results.json',
        f'python benchmarks/kaggle_report.py --input benchmarks/{args.prefix}-results.json --prefix {args.prefix}',  '```', '',
        '原始数据保存在被 Git 忽略的 `benchmarks/cache/`。缓存不存在时通过 HTTPS 自动下载，并核验固定 SHA256。',
        '划分后的输入与预测在 `benchmarks/runs/`，同样不纳入 Git。脚本不覆盖以前的 `results.json`。', '',
        f'[全部原始测量]({args.prefix}-results.json) · [聚合统计]({args.prefix}-summary.json)', '',
    ]
    if "lightgbm" in engines:
        lgb_time = lookup[('telco', 'cpp', 'lightgbm')]['fit_seconds_median']
        position = lines.index('小型数据集的接口开销占比更高，毫秒级差异容易受共享 CPU 调度影响。原始文件保存了每次计时，可检查波动。')
        lines.insert(position, f"Telco C++ LightGBM 训练中位数为 {lgb_time*1000:.1f} ms；LibTree 耗时约为 LightGBM 的 {telco_lib/lgb_time:.2f} 倍。")
    for name in names:
        source = first['sources'][name]
        lines += [f"- **{name}**：[Kaggle 页面]({source['kaggle_url']})；[实际下载镜像]({source['url']})。SHA256：`{source['sha256']}`。"]
    if args.prefix == "optimized":
        wins = {field: sum(
            lookup[(name, language, 'libtree')][field] ==
            min(lookup[(name, language, engine)][field] for engine in engines)
            for name in names for language in ('cpp', 'python'))
            for field in ('fit_seconds_median', 'predict_seconds_median')}
        lines += ['', '## 本轮性能排名', '',
                  f"按每组运行中位数排名：训练 {wins['fit_seconds_median']}/{len(names)*2} 组第一，推理 {wins['predict_seconds_median']}/{len(names)*2} 组第一。",
                  '排名限定为上述本机、CPU 单线程、固定参数与数据集，不代表所有硬件或任务，也不表示每一次运行或每个划分都获胜。', '']
        previous = json.loads((ROOT / 'comparison-results.json').read_text())
        old = {(r['dataset'], r['language'], r['seed']): r for split in previous['split_reports']
               for r in split['results'] if r['engine'] == 'libtree'}
        unchanged = all(row['metrics'] == old[(row['dataset'], row['language'], row['seed'])]['metrics']
                        for row in all_records if row['engine'] == 'libtree')
        lines += [f"同一划分的全部 LibTree 效果指标与优化前{'逐项相同' if unchanged else '存在差异，请对照原始文件'}。训练保留全部轮次和训练损失记录。", '']
        ab_path = ROOT / 'optimization-ab-results.json'
        if ab_path.exists():
            ab = json.loads(ab_path.read_text())
            lines += ['## 随机交错顺序的优化前后实测', '',
                      f"原生 C++，同一输入，{ab['metadata']['repeats']} 次独立重复，旧版 {ab['metadata']['baseline_ref']}。",
                      '此表为固定输入的配对复测，与上方多划分三方计时独立；每个样本的 float32 预测逐值相同。', '',
                      '| 数据集 | 旧训练 ms | 新训练 ms | 训练加速 | 旧推理 ms | 新推理 ms | 推理加速 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
            for row in ab['results']:
                assert row['predictions_identical']
                lines += [f"| {row['dataset']} | {row['baseline_fit_seconds']*1000:.2f} | {row['optimized_fit_seconds']*1000:.2f} | {row['fit_speedup']:.2f}× | {row['baseline_predict_seconds']*1000:.3f} | {row['optimized_predict_seconds']*1000:.3f} | {row['predict_speedup']:.2f}× |"]
            lines += ['', '[全部配对测量](optimization-ab-results.json)', '']
    if args.prefix == 'optimized':
        synthetic_path = ROOT / 'optimized-synthetic-results.json'
        if synthetic_path.exists():
            synthetic = json.loads(synthetic_path.read_text())
            lines += ['', '## 补充：20k 行合成数据', '',
                      'Friedman1，20 个特征、100 棵树；随机种子 42，训练 15000 / 测试 5000 行；每组合 5 次独立计时，共 30 次。',
                      '单划分补充测试，不与五数据集的多划分效果标准差混合。', '',
                      '| 接口 | 引擎 | 测试 RMSE | 训练 ms | 推理 ms |',
                      '| --- | --- | ---: | ---: | ---: |']
            for row in synthetic['results']:
                assert all(row['checks'].values())
                lines += [f"| {row['language']} | {row['engine']} | {row['metric']:.4f} | {row['fit_seconds']*1000:.2f} | {row['predict_seconds']*1000:.3f} |"]
            lines += ['', '[合成数据全部原始测量](optimized-synthetic-results.json)', '']
        lines += ['## 复测优化前后及排名', '',
                  '```sh',
                  'python benchmarks/validate_ranking.py benchmarks/optimized-results.json --previous benchmarks/comparison-results.json --check-sources --require-top1 --output benchmarks/optimized-ranking.json',
                  'PYTHONPATH=python python benchmarks/run.py --suite builtin --datasets friedman_20k --seed 42 --repeats 5 --output benchmarks/optimized-synthetic-results.json',
                  'mkdir -p /tmp/libtree-before',
                  'git archive 937e6b4 | tar -x -C /tmp/libtree-before',
                  'cmake -S /tmp/libtree-before -B /tmp/libtree-before/build -DCMAKE_BUILD_TYPE=Release',
                  'cmake --build /tmp/libtree-before/build -j4',
                  'python benchmarks/ab.py --baseline /tmp/libtree-before/build/native_benchmark --baseline-ref 937e6b4 --repeats 9',
                  '```', '',
                  '配对测试使用上述两类比较生成的 benchmarks/runs/*.bin；重复执行不需要工作树或复制原始 CSV 入 Git。',
                  '[排名检查结果](optimized-ranking.json) · [性能设计与原理](../docs/design.zh-CN.md)', '']
    lines.append('')
    (ROOT / markdown_name).write_text('\n'.join(lines).rstrip() + '\n')
    print(f"Report: {ROOT / markdown_name}; {runs} runs")


if __name__ == '__main__':
    main()
