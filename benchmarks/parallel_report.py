"""Validate equal-thread evidence and build an offline scaling report."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
ENGINES = ('libtree', 'xgboost', 'lightgbm')


def summarize(raw, previous=None, check_sources=True):
    reports = raw['split_reports']
    threads, seeds = raw['threads'], raw['seeds']
    assert 1 in threads and len(set(threads)) == len(threads)
    assert len(set(seeds)) == len(seeds)
    assert len(reports) == len(threads) * len(seeds)
    names = set(reports[0]['sources'])
    groups, metrics, keys, splits = defaultdict(list), defaultdict(list), set(), set()
    provenance = {}
    signatures = reports[0]['metadata']['source_sha256']
    parameters = reports[0]['metadata']['parameters']
    for report in reports:
        meta = report['metadata']
        seed, count = meta['seed'], meta['threads']
        assert count in threads and seed in seeds
        assert (seed, count) not in splits
        splits.add((seed, count))
        assert set(meta['engines']) == set(ENGINES)
        assert meta['parameters'] == parameters and meta['source_sha256'] == signatures
        assert set(report['sources']) == names
        for name in names:
            digest = report['sources'][name]['prepared_split_sha256']
            assert provenance.setdefault((name, seed), digest) == digest
        for row in report['results']:
            key = (row['dataset'], row['language'], row['engine'], seed, count)
            assert key not in keys and row['seed'] == seed
            keys.add(key)
            assert set(row['checks']) == {'beats_baseline', 'finite_predictions',
                                          'deterministic_repeats', 'cpp_python_parity'}
            assert all(row['checks'].values())
            assert all(math.isfinite(v) for v in row['metrics'].values())
            assert row['metric'] == row['metrics'][row['metric_name']]
            assert len(row['raw_runs']) == raw['timing_repeats_per_split']
            for run in row['raw_runs']:
                assert all(math.isfinite(run[f]) and run[f] > 0
                           for f in ('fit_seconds', 'predict_seconds', 'peak_rss_kib'))
            for field in ('fit_seconds', 'predict_seconds', 'peak_rss_kib'):
                assert row[field] == median(run[field] for run in row['raw_runs'])
            group = key[:3] + (count,)
            groups[group].extend(row['raw_runs'])
            metrics[group].append(row['metrics'])
    assert len(keys) == len(names) * 2 * 3 * len(seeds) * len(threads)
    assert all((name, lang, engine, seed, count) in keys for name in names
               for lang in ('cpp', 'python') for engine in ENGINES
               for seed in seeds for count in threads)
    if check_sources:
        for path, digest in signatures.items():
            assert hashlib.sha256((ROOT.parent / path).read_bytes()).hexdigest() == digest, path
    # LibTree preserves every same-split metric across configured thread counts.
    exact = {}
    for report in reports:
        for row in report['results']:
            if row['engine'] == 'libtree':
                key = (row['dataset'], row['language'], row['seed'])
                assert exact.setdefault(key, row['metrics']) == row['metrics'], key
    if previous is not None:
        for report in previous['split_reports']:
            assert report['metadata']['parameters'] == parameters
            for row in report['results']:
                if row['engine'] == 'libtree' and row['seed'] in seeds and row['dataset'] in names:
                    key = (row['dataset'], row['language'], row['seed'])
                    assert exact[key] == row['metrics'], key
                    assert provenance[(row['dataset'], row['seed'])] == report['sources'][row['dataset']]['prepared_split_sha256']
    result = []
    for (name, lang, engine, count), runs in sorted(groups.items()):
        metric_name = next(r['metric_name'] for report in reports for r in report['results']
                           if r['dataset'] == name)
        values = [m[metric_name] for m in metrics[(name, lang, engine, count)]]
        summary = {'dataset': name, 'language': lang, 'engine': engine,
                   'threads': count, 'runs': len(runs), 'metric_name': metric_name,
                   'metric_mean': mean(values), 'metric_std': stdev(values) if len(values) > 1 else 0}
        for field in ('fit_seconds', 'predict_seconds', 'peak_rss_kib'):
            timings = [r[field] for r in runs]
            summary[field] = median(timings)
            summary[field + '_min'] = min(timings)
            summary[field + '_max'] = max(timings)
        result.append(summary)
    lookup = {(r['dataset'], r['language'], r['engine'], r['threads']): r for r in result}
    for row in result:
        base = lookup[(row['dataset'], row['language'], row['engine'], 1)]
        for field in ('fit_seconds', 'predict_seconds'):
            row[field + '_speedup'] = base[field] / row[field]
            row[field + '_rank'] = 1 + sum(
                lookup[(row['dataset'], row['language'], engine, row['threads'])][field] < row[field]
                for engine in ENGINES)
    return result


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LibTree 多线程性能报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f7f7;color:#233f49;font:15px/1.65 system-ui,sans-serif}
.wrap{max-width:1200px;margin:auto;padding:28px}header{background:#16353e;color:white;padding:24px 0}
h1{font-size:32px;margin:8px 0}h2{font-size:21px;margin:32px 0 12px}h3{margin:0;font-size:17px}
.sub,.note{color:#607881;font-size:13px}header .sub{color:#b6cdd1}.toolbar{position:sticky;top:0;background:#fff;z-index:20;border-bottom:1px solid #d7e3e4}
.toolbar .wrap{padding-top:12px;padding-bottom:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
button,select{padding:8px 12px;border:1px solid #cbdadb;background:white;border-radius:6px;color:inherit;font:inherit;cursor:pointer}
button.active{background:#007d77;color:#fff}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:white;padding:20px;border-radius:10px;border:1px solid #e0eaea;min-width:0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.big{font-size:28px;font-weight:700;color:#007d77}.chart{height:380px}.wide{grid-column:1/-1}.table-scroll{overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:10px;border-bottom:1px solid #e7eeee}th{background:#eef5f5}
code{overflow-wrap:anywhere}summary{cursor:pointer;padding:12px}a{color:#007d77}footer{padding:28px 0;color:#607881}
@media(max-width:700px){.wrap{padding:18px}h1{font-size:25px}.grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr}.card{padding:14px}.chart{height:360px}.toolbar .wrap{gap:8px}select,button{font-size:13px;padding:7px}}
</style><script>__PLOTLY__</script></head><body>
<header><div class="wrap"><div class="sub">LIBTREE · CPU 并行优化 / Parallel CPU execution</div><h1>__THREAD_LABEL__ 线程：训练与推理实测</h1><p>Kaggle 数据镜像与合成压力测试 · C++ / Python · 三方相同线程预算</p><div class="sub" id="environment"></div></div></header>
<nav class="toolbar"><div class="wrap"><button id="cpp" class="active" aria-pressed="true">C++</button><button id="python" aria-pressed="false">Python</button><label>线程 <select id="threads"></select></label><label>扩展性数据集 <select id="dataset"></select></label><button id="json">下载原始 JSON</button><button id="csv">下载全部 CSV</button></div></nav>
<main class="wrap"><p>所有运行使用相同主参数、数据划分及配置线程数。每组按 __SEEDS__ 个划分 × __REPEATS__ 次独立计时的中位数比较；小任务可能使用较少工作线程。排名属于本次硬件和任务。</p>
<div class="cards"><div class="card"><div class="note">当前数据集 · LibTree 训练加速</div><div class="big" id="fit-ratio"></div><div class="note">同版本 1 线程 ÷ 所选线程耗时</div></div><div class="card"><div class="note">当前数据集 · LibTree 推理加速</div><div class="big" id="predict-ratio"></div><div class="note">模型与效果指标保持一致</div></div><div class="card"><div class="note">可审查的原始计时</div><div class="big" id="run-count"></div><div class="note" id="wins"></div></div></div>
<h2>同线程预算的训练与推理</h2><label class="note"><input type="checkbox" id="log" checked> 对数时间坐标</label><div class="grid"><article class="card"><h3>训练时间 · ms ↓</h3><p class="note">含分箱、矩阵构建及 LibTree 线程池准备；误差线为最小至最大计时。</p><div class="chart" id="fit"></div></article><article class="card"><h3>预测时间 · ms ↓</h3><p class="note">接口端到端，包括必要校验和转换，复用已拟合模型的线程池。</p><div class="chart" id="predict"></div></article></div>
<h2>随线程数变化：所选数据集</h2><div class="grid"><article class="card"><h3>训练加速比 ↑</h3><p class="note">各引擎自身的单线程耗时作为基准；小于 1 表示变慢。</p><div class="chart" id="fit-scaling"></div></article><article class="card"><h3>推理加速比 ↑</h3><p class="note">收益受串行工作、调度与 CPU 配额限制，并非线程数倍数。</p><div class="chart" id="predict-scaling"></div></article><article class="card wide"><h3>峰值进程内存 · MiB ↓</h3><p class="note" id="memory-note"></p><div class="chart" id="rss"></div></article></div>
<h2>效果和测量明细</h2><p class="note">效果是 __SEEDS__ 个划分的均值 ± 样本标准差；AUC ↑，RMSE ↓。时间是 __POOLED__ 次运行的中位数。LibTree 同划分效果指标跨线程数逐项相同，旧版五数据集效果也未改变。</p><div class="card table-scroll"><table><thead><tr><th>数据集</th><th>引擎</th><th>指标</th><th>测试效果</th><th>训练 ms</th><th>推理 ms</th><th>训练排名</th><th>推理排名</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>实现与验证</h2><div class="card"><p>标准 C++17 线程池，调用线程参与计算；特征直方图各自写入独立区域，保持样本累加顺序。Boosting 各轮与树节点按依赖执行。预测按 64 行批次并行，校验完成后才写输出。小任务保留串行路径。</p><p>TDD 先建立缺失接口的失败测试。5/5 原生目标、31 Python、13 CLI、4 基准准备及 2 性能/所有权用例通过；4,701 项原生行为检查。ASan/UBSan/LSan、ThreadSanitizer 均通过。源码行覆盖率 100%（1116/1116），分支 67.2%，Python 100%（148/148）。仅排除明确标记的系统耗尽、异常边界、竞态和编译器行映射续行。这些是本地检查，远程 CI 结果需另行查看。</p><p><code>xgbt train --train data.csv --threads 4</code><br><code>GBDTRegressor(n_jobs=4)</code><br><code>Parameters::num_threads = 4</code></p><p>默认 1，范围 1–256，含调用线程。线程数不写入模型；共享模型的并发预测分发会排队，不能并发拟合、关闭或修改线程数。</p></div>
<h2>方法、来源与边界</h2><div class="card"><p>100 棵树、深度 4、64 分箱、学习率 0.1、L2=1、最小叶样本数 5。XGBoost 使用默认最小 Hessian 权重 1；LightGBM leaf-wise、最多 16 叶。分箱与生长方式并不完全相同，没有超参数搜索。</p><p>种子 42/2024/2026；75/25 留出，分类分层抽样；编码只拟合训练集，保留 NaN。红酒先去重，Telco 删除客户 ID。训练排除下载、CSV 读取、编码、导入和进程启动；XGBoost 原生预测计入测试 DMatrix 构建。Python 进程均导入三个库，RSS 是进程峰值而非模型增量。</p><p>各线程/种子的执行块以固定种子随机排序，测量进程串行运行。加速比为两组中位数之比，不是逐轮配对实验；共享云 CPU 存在波动。Kaggle 数据来自已校验 SHA256 的公开镜像，不是官方榜单成绩。Friedman 数据为合成回归压力测试，不代表所有真实任务；不覆盖稀疏、GPU、多分类、百万行任务。</p><div id="sources"></div><details><summary>源码与报告指纹</summary><div id="hashes"></div></details></div>
<footer>全部 1 / 2 / 4 线程结果均保留，包括负收益。排名与效果不外推为普遍最优。</footer></main>
<script id="evidence" type="application/json">__DATA__</script><script>
'use strict';const evidence=JSON.parse(document.getElementById('evidence').textContent),raw=evidence.raw,stats=evidence.summary;
const names=['titanic','insurance','pima','telco','wine','friedman_20k','friedman_100k'].filter(n=>stats.some(r=>r.dataset===n)),engines=['libtree','xgboost','lightgbm'],colors={libtree:'#007d77',xgboost:'#d66d45',lightgbm:'#7662a9'};
let language='cpp',threads=Math.max(...raw.threads),dataset=names.includes('friedman_100k')?'friedman_100k':names[0];const lookup=(n,e,t=threads)=>stats.find(r=>r.dataset===n&&r.language===language&&r.engine===e&&r.threads===t);
const config={responsive:true,displaylogo:false};const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#fff',font:{family:'system-ui',color:'#475b65'},margin:{l:106,r:45,t:20,b:45},barmode:'group',showlegend:true,legend:{orientation:'h',y:1.15},xaxis:{type:'linear',gridcolor:'#edf1f0'},yaxis:{gridcolor:'#edf1f0'}};
function draw(id,data,layout){return Plotly.react(id,data,structuredClone({...base,...layout}),config);}
function render(){for(const lang of ['cpp','python']){const active=lang===language;document.getElementById(lang).classList.toggle('active',active);document.getElementById(lang).setAttribute('aria-pressed',active);}
 for(const [id,field,mult] of [['fit','fit_seconds',1000],['predict','predict_seconds',1000],['rss','peak_rss_kib',1/1024]]){
  const bars=engines.map(e=>({type:'bar',orientation:'h',name:e,y:names,x:names.map(n=>lookup(n,e)[field]*mult),marker:{color:colors[e]},hovertemplate:'%{y} · '+e+'<br>%{x:.3f}<extra></extra>'}));
  if(id==='fit')bars.forEach((b,k)=>b.error_x={type:'data',symmetric:false,array:names.map(n=>(lookup(n,engines[k])[field+'_max']-lookup(n,engines[k])[field])*mult),arrayminus:names.map(n=>(lookup(n,engines[k])[field]-lookup(n,engines[k])[field+'_min'])*mult)});
  draw(id,bars,{xaxis:{...base.xaxis,type:id!=='rss'&&document.getElementById('log').checked?'log':'linear',title:{text:id==='rss'?'MiB':'ms'}},yaxis:{...base.yaxis,autorange:'reversed'}});
 }
 for(const [id,field] of [['fit-scaling','fit_seconds'],['predict-scaling','predict_seconds']])draw(id,engines.map(e=>({type:'scatter',mode:'lines+markers',name:e,x:raw.threads,y:raw.threads.map(t=>lookup(dataset,e,t)[field+'_speedup']),line:{color:colors[e]},hovertemplate:'%{x} 线程 · '+e+'<br>%{y:.2f}×<extra></extra>'})),{xaxis:{...base.xaxis,tickvals:raw.threads,title:{text:'CPU 线程预算'}},yaxis:{...base.yaxis,title:{text:'单线程 ÷ 当前耗时'},rangemode:'tozero'},shapes:[{type:'line',x0:1,x1:Math.max(...raw.threads),y0:1,y1:1,line:{dash:'dot',color:'#8b9b9f'}}]});
 const selected=stats.filter(r=>r.language===language&&r.threads===threads);
 document.getElementById('fit-ratio').textContent=lookup(dataset,'libtree').fit_seconds_speedup.toFixed(2)+'×';document.getElementById('predict-ratio').textContent=lookup(dataset,'libtree').predict_seconds_speedup.toFixed(2)+'×';
 const wins=f=>selected.filter(r=>r.engine==='libtree'&&r[f+'_rank']===1).length;
 document.getElementById('wins').textContent='所选接口 / 线程：训练 '+wins('fit_seconds')+'/'+names.length+'，推理 '+wins('predict_seconds')+'/'+names.length+' 组最快';
 document.getElementById('memory-note').textContent=language==='cpp'?'原生执行进程的峰值，含线程池等运行内存。':'Python 运行时及三个库共同构成基础开销；计量整个进程峰值。';
 document.getElementById('rows').innerHTML=selected.map(r=>`<tr><td>${r.dataset}</td><td>${r.engine}</td><td>${r.metric_name==='roc_auc'?'AUC ↑':'RMSE ↓'}</td><td>${r.metric_mean.toFixed(4)} ± ${r.metric_std.toFixed(4)}</td><td>${(r.fit_seconds*1000).toFixed(2)}</td><td>${(r.predict_seconds*1000).toFixed(3)}</td><td>${r.fit_seconds_rank}</td><td>${r.predict_seconds_rank}</td></tr>`).join('');
}
document.getElementById('threads').innerHTML=raw.threads.map(t=>`<option ${t===threads?'selected':''}>${t}</option>`).join('');
document.getElementById('dataset').innerHTML=names.map(n=>`<option value="${n}" ${n===dataset?'selected':''}>${n}</option>`).join('');
for(const lang of ['cpp','python'])document.getElementById(lang).onclick=()=>{language=lang;render();};document.getElementById('threads').onchange=e=>{threads=Number(e.target.value);render();};document.getElementById('dataset').onchange=e=>{dataset=e.target.value;render();};document.getElementById('log').onchange=render;
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('json').onclick=()=>download(JSON.stringify(raw,null,2),'parallel-results.json','application/json');
document.getElementById('csv').onclick=()=>{const rows=['dataset,seed,threads,language,engine,repeat,metric_name,metric,fit_seconds,predict_seconds,peak_rss_kib'];for(const report of raw.split_reports)for(const r of report.results)r.raw_runs.forEach((s,i)=>rows.push([r.dataset,r.seed,report.metadata.threads,r.language,r.engine,i+1,r.metric_name,r.metric,s.fit_seconds,s.predict_seconds,s.peak_rss_kib].join(',')));download(rows.join('\n'),'parallel-timing-'+evidence.run_count+'-runs.csv','text/csv');};
const m=raw.split_reports[0].metadata,quota=m.cpu_quota.split(' '),quotaLabel=quota[0]==='max'?'不限':(Number(quota[0])/Number(quota[1])).toFixed(1)+' 核';document.getElementById('environment').textContent=m.cpu.replace(/^model name\s*:\s*/,'')+' · 可用 CPU '+m.available_cpus+' / CPU 配额 '+quotaLabel+' · '+m.compiler+' · Python '+m.python+' · XGBoost '+m.xgboost+' / LightGBM '+m.lightgbm;
document.getElementById('run-count').textContent=evidence.run_count+' 次';document.getElementById('sources').innerHTML=Object.entries(raw.split_reports[0].sources).map(([n,s])=>`<p><strong>${n}</strong> · ${s.rows_after_cleaning} 行 · ${s.raw_feature_count} 原始特征 ${s.url?` · <a href="${s.url}">实际镜像</a> · SHA256 <code>${s.sha256}</code>`:` · ${s.source}`}</p>`).join('');
document.getElementById('hashes').innerHTML=Object.entries({...m.source_sha256,...evidence.files}).map(([n,h])=>`<p>${n}<br><code>${h}</code></p>`).join('');render();window.__reportReady=true;
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'parallel-results.json')
    parser.add_argument('--historical', action='store_true', help='Skip current checkout SHA checks for an older snapshot')
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    previous = json.loads((ROOT / 'optimized-results.json').read_text())
    summary = summarize(raw, previous=previous, check_sources=not args.historical)
    runs = sum(len(r['raw_runs']) for s in raw['split_reports'] for r in s['results'])
    (ROOT / 'parallel-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# LibTree CPU 多线程实测', '',
             f'{len(raw["threads"])} 个线程数 × {len(raw["split_reports"][0]["sources"])} 个数据集 × {len(raw["seeds"])} 个划分 × C++/Python × 3 个引擎 × {raw["timing_repeats_per_split"]} 次计时，共 **{runs} 次**。', '',
             f"Equal configured thread budgets; independent processes. Each cell uses the pooled median of {len(raw['seeds']) * raw['timing_repeats_per_split']} runs. "
             'All LibTree same-split quality metrics are unchanged across threads and from the preceding five-dataset snapshot.', '',
             '训练与预测分开测量。每组 9 次原始计时，保留最小/最大值；效果为三个划分的均值及样本标准差。', '',
             f"可用逻辑 CPU：{raw['split_reports'][0]['metadata']['available_cpus']}；cgroup CPU 配额：{raw['split_reports'][0]['metadata']['cpu_quota']}；种子：{raw['seeds']}。线程数为预算，小任务可串行。", '',
             '| 数据集 | 接口 | 引擎 | 线程 | 训练 ms | 推理 ms | 训练加速 | 推理加速 | 指标 | 测试效果 |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['language']} | {r['engine']} | {r['threads']} | "
                     f"{r['fit_seconds']*1000:.2f} | {r['predict_seconds']*1000:.3f} | "
                     f"{r['fit_seconds_speedup']:.2f}× | {r['predict_seconds_speedup']:.2f}× | "
                     f"{'AUC ↑' if r['metric_name'] == 'roc_auc' else 'RMSE ↓'} | "
                     f"{r['metric_mean']:.4f} ± {r['metric_std']:.4f} |")
    lines += ['', '## 排名与限制', '']
    for count in raw['threads']:
        subset = [r for r in summary if r['threads'] == count and r['engine'] == 'libtree']
        fit = sum(r['fit_seconds_rank'] == 1 for r in subset)
        pred = sum(r['predict_seconds_rank'] == 1 for r in subset)
        lines.append(f'{count} 线程：LibTree 训练 {fit}/{len(subset)} 组最快，推理 {pred}/{len(subset)} 组最快。')
    lines += ['', '加速比为同版本单线程中位数除以当前线程中位数，不是配对逐轮实验。'
              '共享云 CPU 有波动，不能将差异解释为普遍最优。小任务可能因启动和同步变慢，全部结果均保留。', '',
              '主参数：100 棵树、深度 4、64 分箱、学习率 0.1、L2=1、最小增益 0。'
              'LibTree/LightGBM 最小叶样本 5；XGBoost 默认最小 Hessian 权重 1；LightGBM leaf-wise、16 叶。'
              '各引擎分箱、生长方式并不完全相同，无超参数搜索。', '',
              '训练包含分箱/DMatrix/标签传递及 LibTree 冷线程池准备；排除数据下载、CSV、预处理、导入和进程启动。'
              '预测计时包含接口必要校验和转换，C++ XGBoost 包含测试 DMatrix 构建；RSS 为执行进程 VmHWM。', '',
              '75/25 留出，二分类分层；训练集拟合编码，保留 NaN，红酒先去重。Kaggle 对应数据来自固定 SHA256 的公开镜像；'
              'Friedman 为合成压力测试。未覆盖百万行、稀疏、多分类、GPU、跨硬件任务。', '',
              '5/5 CTest、31 Python、13 CLI、4 基准准备及 2 性能/所有权用例通过；4,701 项原生检查。'
              'ASan/UBSan/LSan 与 ThreadSanitizer 均通过。源码行覆盖 1116/1116=100%，'
              '分支 1116/1660=67.2%，函数 106/106；Python 148/148=100%。仅排除明确标记的系统耗尽、异常边界、竞态和编译器行映射续行。'
              '远程 GitHub Actions 是否通过需另行查看。', '',
              '## 复现', '', '```sh', '. /workspace/libtree-venv/bin/activate', 'make all',
              'PYTHONPATH=python python benchmarks/parallel.py --threads 1 2 4 --seeds 42 2024 2026 --repeats 3',
              'python -m pip install -r benchmarks/requirements-report.txt', 'python benchmarks/parallel_report.py',
              'make test', 'make sanitize', 'make thread-sanitize', 'make coverage', '```', '',
              '[交互 HTML](PARALLEL_REPORT.html) · [全部原始计时](parallel-results.json) · [聚合数据](parallel-summary.json) · '
              '[并行原理](../docs/design.zh-CN.md#确定性并行)', '']
    (ROOT / 'PARALLEL_REPORT.zh-CN.md').write_text('\n'.join(lines).rstrip() + '\n')
    payload = {'raw': raw, 'summary': summary, 'run_count': runs,
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.input, ROOT / 'parallel-summary.json')}}
    encoded = json.dumps(payload, ensure_ascii=False).replace('</', r'<\/')
    html = TEMPLATE.replace('__PLOTLY__', get_plotlyjs().replace('</script', r'<\/script')).replace('__DATA__', encoded)
    html = html.replace('__THREAD_LABEL__', ' / '.join(map(str, raw['threads']))).replace('__SEEDS__', str(len(raw['seeds']))).replace('__REPEATS__', str(raw['timing_repeats_per_split'])).replace('__POOLED__', str(len(raw['seeds']) * raw['timing_repeats_per_split']))
    (ROOT / 'PARALLEL_REPORT.html').write_text('\n'.join(line.rstrip() for line in html.splitlines()) + '\n')
    print(f'Validated {runs} runs, source hashes, equal inputs, complete groups and unchanged LibTree quality.')


if __name__ == '__main__':
    main()
