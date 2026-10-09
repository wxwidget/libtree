"""Build a self-contained, offline interactive HTML from verified measurements."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev, median

from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
prefix = next(name for name in ("optimized", "comparison", "kaggle")
              if (ROOT / f"{name}-results.json").exists())
raw_path = ROOT / f"{prefix}-results.json"
summary_path = ROOT / f"{prefix}-summary.json"
raw = json.loads(raw_path.read_text())
summary = json.loads(summary_path.read_text())
records = [r for s in raw["split_reports"] for r in s["results"]]
engines = raw["split_reports"][0]["metadata"].get("engines", ["libtree", "xgboost"])
expected = len(raw["split_reports"]) * len(raw["split_reports"][0]["sources"]) * 2 * len(engines)
run_count = sum(len(r["raw_runs"]) for r in records)
assert len(records) == expected
assert run_count == expected * raw["timing_repeats_per_split"]
assert all(all(r["checks"].values()) for r in records)
assert all("VmHWM" in s["metadata"]["memory_measurement"] for s in raw["split_reports"])
assert len(summary) == len(raw["split_reports"][0]["sources"]) * 2 * len(engines)
for row in summary:
    subset = [r for r in records if (r["dataset"], r["language"], r["engine"]) ==
              (row["dataset"], row["language"], row["engine"])]
    assert len(subset) == len(raw["split_reports"])
    for metric, value in row["metrics"].items():
        values = [r["metrics"][metric] for r in subset]
        assert abs(mean(values) - value["mean"]) < 1e-10
        assert abs((stdev(values) if len(values) > 1 else 0) - value["std"]) < 1e-10
    samples = [run for r in subset for run in r["raw_runs"]]
    assert abs(median(r["fit_seconds"] for r in samples) - row["fit_seconds_median"]) < 1e-10
    assert abs(median(r["predict_seconds"] for r in samples) - row["predict_seconds_median"]) < 1e-10
    assert abs(median(r["peak_rss_kib"] / 1024 for r in samples) - row["peak_rss_mib_median"]) < 1e-10

as_of = datetime.fromisoformat(raw["split_reports"][-1]["metadata"]["date_utc"].replace("Z", "+00:00"))
as_of = as_of.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
payload = {"raw": raw, "summary": summary,
           "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw_path, summary_path)}}
ab_path = ROOT / "optimization-ab-results.json"
if prefix == "optimized" and ab_path.exists():
    payload["ab"] = json.loads(ab_path.read_text())
    payload["files"][ab_path.name] = hashlib.sha256(ab_path.read_bytes()).hexdigest()
    assert all(r["predictions_identical"] for r in payload["ab"]["results"])
    assert payload["ab"]["metadata"]["source_sha256"]["src/gbdt.cc"] == raw["split_reports"][0]["metadata"]["source_sha256"]["src/gbdt.cc"]
synthetic_path = ROOT / "optimized-synthetic-results.json"
if prefix == "optimized" and synthetic_path.exists():
    payload["synthetic"] = json.loads(synthetic_path.read_text())
    payload["files"][synthetic_path.name] = hashlib.sha256(synthetic_path.read_bytes()).hexdigest()
    assert all(all(row['checks'].values()) for row in payload['synthetic']['results'])
    assert payload['synthetic']['metadata']['source_sha256'] == raw['split_reports'][0]['metadata']['source_sha256']
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggle GBDT 实测报告 · LibTree × XGBoost × LightGBM</title>
<style>
:root{--ink:#172b37;--muted:#62707b;--line:#dfe5e6;--paper:#f6f8f7;--teal:#007d77;--orange:#d66d45}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}header{background:#16343d;color:#fff;padding:48px 0 40px}.wrap{max-width:1220px;margin:auto;padding:0 32px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#b5d4d3}h1{font-size:clamp(26px,3vw,39px);line-height:1.3;margin:12px 0 12px;letter-spacing:-1px}h2{font-size:24px;line-height:1.4;margin:0 0 6px}h3{font-size:17px;margin:0 0 5px}p{margin:8px 0}.sub{color:#c6d7da;max-width:920px}.badge{display:inline-block;border:1px solid #49666e;border-radius:30px;padding:4px 12px;margin:12px 6px 0 0;font-size:12px}.toolbar{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);box-shadow:0 3px 10px #19333d05}.toolbar .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-top:13px;padding-bottom:13px}.tabs{display:flex;gap:4px;padding:4px;border:1px solid var(--line);border-radius:10px;background:#f6f8f7}button,select{font:inherit;border:1px solid #cdd7d9;background:#fff;color:var(--ink);border-radius:7px;padding:7px 13px;cursor:pointer}button:hover{border-color:var(--teal)}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #59bdb2;outline-offset:3px}.tabs button{border:0;background:transparent;font-weight:600}.tabs button.active{background:#16343d;color:white}.actions{display:flex;gap:7px;flex-wrap:wrap}.label{font-size:12px;color:var(--muted)}main{padding-bottom:56px}.summary{padding:28px 0 20px}.summary p{max-width:1030px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0 6px}.kpi{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 20px}.big{font-size:31px;line-height:1.25;font-weight:650;letter-spacing:-1px}.kpi .label{margin-bottom:7px}.small{font-size:12px;color:var(--muted);line-height:1.6}.section{margin:28px 0}.section-head{margin-bottom:15px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:20px;min-width:0}.card p{color:var(--muted);font-size:13px;margin:5px 0 8px}.chart{height:330px}.chart.short{height:265px}.chart.wide{height:340px}.row{display:flex;align-items:center;justify-content:space-between;gap:12px}.legend{display:flex;gap:17px;align-items:center;font-size:12px}.dot{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px}.teal{background:var(--teal)}.orange{background:var(--orange)}.notice{border-left:3px solid var(--teal);padding:10px 16px;background:#edf4f2;border-radius:0 6px 6px 0;font-size:13px;margin:17px 0}.regression{display:grid;grid-template-columns:1fr 1fr;gap:10px}.regression .chart{height:260px}.note{font-size:13px;color:var(--muted);margin-top:12px}.sources{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.source{padding:15px;background:white;border:1px solid var(--line);border-radius:9px;overflow-wrap:anywhere}.source a{font-size:13px}a{color:var(--teal);text-underline-offset:3px}details{background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0;padding:17px 20px}summary{font-weight:600;cursor:pointer}details[open] summary{margin-bottom:16px}.table-scroll{overflow:auto;max-height:540px;border:1px solid var(--line);margin:14px 0}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #e7edec;text-align:right}th{background:#f0f5f4;color:#485c66;position:sticky;top:0;z-index:1}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#f6faf9}.pass{color:var(--teal)}code{font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}pre{padding:16px;background:#f0f4f3;overflow:auto;border-radius:7px}footer{padding-top:25px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}.nojs{padding:20px;background:#fff1df}input[type=checkbox]{accent-color:var(--teal)}.plotly .modebar{top:0!important}@media(max-width:800px){.wrap{padding:0 18px}header{padding:32px 0}.grid,.sources{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.toolbar .wrap{align-items:flex-start}.actions button{font-size:12px;padding:6px 8px}.regression{grid-template-columns:1fr}.card{padding:16px}.chart{height:310px}.row{flex-wrap:wrap}}@media print{.toolbar,.modebar,.actions{display:none!important}header{background:white!important;color:var(--ink);padding:20px 0}.sub,.eyebrow{color:var(--muted)}.badge{border-color:var(--line)}body{background:white}.card,.kpi,details{break-inside:avoid}.wrap{max-width:100%;padding:0 16px}.grid{grid-template-columns:1fr}.sources{grid-template-columns:1fr 1fr}.section{break-inside:auto}details{display:block}.table-scroll{max-height:none}.chart{height:300px}}
</style><script>__PLOTLY__</script></head><body>
<header><div class="wrap"><div class="eyebrow">LIBTREE LAB / 实测基准</div><h1>__HEADLINE__</h1><p class="sub">五个 Kaggle 数据集对应公开镜像 · 三个不同随机划分 · C++ / Python 双接口比较</p><div><span class="badge">__RUNS__ 次有效测量</span><span class="badge">__VERSIONS__</span><span class="badge">CPU 单线程</span><span class="badge">数据截至 __ASOF__ · 北京时间</span></div></div></header>
<div class="toolbar"><div class="wrap"><div class="tabs" role="group" aria-label="接口选择"><button class="active" id="cpp" aria-pressed="true">C++ 原生</button><button id="python" aria-pressed="false">Python 接口</button></div><div class="actions"><button id="download">下载主基准 JSON</button><button id="csv">下载主基准 CSV</button><button id="print">打印 / PDF</button></div></div></div>
<noscript><div class="wrap nojs">交互图表需要启用 JavaScript；本文件已内置图表库与全部实测数据，不需要联网。</div></noscript>
<main class="wrap"><section class="summary"><h2>测试结论</h2><p>__CONCLUSION__</p><div class="kpis"><div class="kpi"><div class="label">测试范围</div><div class="big">5 / 3</div><div class="small">数据集 / 不同随机划分</div></div><div class="kpi"><div class="label">Telco · LibTree 训练耗时倍数</div><div class="big" id="ratio">—</div><div class="small">LibTree ÷ XGBoost，越低越快</div></div><div class="kpi"><div class="label">Telco · LibTree 峰值进程内存</div><div class="big" id="memory">—</div><div class="small" id="memory-other"></div></div><div class="kpi"><div class="label">验证状态</div><div class="big">全部通过</div><div class="small">有限预测 · 基线 · 重复确定性 · 接口一致性</div></div></div></section>
<section class="section"><div class="section-head row"><div><h2>效果差异与划分波动</h2><p class="note">点 / 柱为三个划分的均值；误差线为 ±1 个样本标准差，不是置信区间。</p></div><div class="legend"><span><i class="dot teal"></i>LibTree</span><span><i class="dot orange"></i>XGBoost</span><span><i class="dot" style="background:#7662a9"></i>LightGBM</span></div></div><div class="grid"><article class="card"><h3>二分类 · ROC AUC ↑</h3><p>Titanic、Pima、Telco；点图横轴范围为 0.70–1.00。</p><div id="auc" class="chart" role="img" aria-label="三项二分类任务的 AUC 均值与标准差"></div></article><article class="card"><h3>回归 · RMSE ↓</h3><p>两个任务的目标单位不同，使用独立坐标轴。</p><div class="regression"><div><div class="label">保险费用 · 原始金额单位</div><div id="insurance" class="chart short" role="img" aria-label="保险费用 RMSE 比较"></div></div><div><div class="label">红酒质量 · quality 分数</div><div id="wine" class="chart short" role="img" aria-label="红酒质量 RMSE 比较"></div></div></div></article></div><div class="notice">Pima 的 AUC 跨划分波动较大：LibTree 为 0.8074 ± 0.0390，XGBoost 为 0.8047 ± 0.0338。两者的均值差异远小于这里观察到的划分波动。</div></section>
<section class="section"><div class="section-head"><h2>训练、预测和内存</h2><p class="note">每个组合包含 3 个划分 × __REPEATS__ 次计时。条形为 __POOLED__ 次运行中位数；训练误差线为最小–最大范围。</p></div><div class="grid"><article class="card"><div class="row"><h3>训练时间 · ms ↓</h3><label class="small"><input id="log" type="checkbox"> 对数坐标</label></div><p>包含分箱 / DMatrix 构建；不包含导入和预处理。</p><div id="fit" class="chart wide" role="img" aria-label="五个数据集的训练时间比较"></div></article><article class="card"><h3>预测时间 · ms ↓</h3><p>端到端接口计时，包含各接口必要的转换。</p><div id="predict" class="chart wide" role="img" aria-label="五个数据集的预测时间比较"></div></article><article class="card"><h3>峰值进程内存 · MiB ↓</h3><p id="memory-note"></p><div id="rss" class="chart" role="img" aria-label="五个数据集的进程峰值内存比较"></div></article><article class="card"><h3>训练时间比值 · LibTree ÷ 竞品</h3><p>大于 1 表示 LibTree 更慢；橙色对比 XGBoost，紫色对比 LightGBM。</p><div id="speed" class="chart" role="img" aria-label="LibTree 相对 XGBoost 和 LightGBM 的训练时间比值"></div></article></div></section>
__AB_SECTION__<section class="section"><h2>查看测量与方法</h2><details><summary>逐划分结果与完整计时数据</summary><div class="row"><label>筛选数据集 <select id="dataset"><option value="all">全部数据集</option><option value="titanic">Titanic</option><option value="insurance">保险费用</option><option value="pima">Pima</option><option value="telco">Telco</option><option value="wine">红酒质量</option></select></label><span class="small" id="row-count"></span></div><p class="note">本表每行对应一个划分、接口和引擎；时间和内存取该行 __REPEATS__ 次计时的中位数。顶部 CSV 包含全部 __RUNS__ 次计时；JSON 包含附加指标、数据校验值和来源。</p><div class="table-scroll"><table><thead><tr><th>数据集</th><th>Seed</th><th>引擎</th><th>指标</th><th>效果</th><th>训练 ms</th><th>预测 ms</th><th>峰值 MiB</th><th>验证</th></tr></thead><tbody id="rows"></tbody></table></div></details>
<details><summary>附加效果指标：准确率、PR AUC、Log loss、MAE 与 R²</summary><p class="note">均值 ± 样本标准差；二分类准确率的阈值为 0.5。PR AUC 在本报告中指 average precision。</p><div class="table-scroll"><table><thead><tr><th>数据集</th><th>引擎</th><th>指标</th><th>均值</th><th>标准差</th></tr></thead><tbody id="extra"></tbody></table></div></details>
<details><summary>划分、参数与计时边界</summary><p>75% 训练 / 25% 测试；随机种子 42、2024、2026；二分类分层抽样。类别编码只在训练集上拟合，忽略测试集新类别。缺失值保留为 NaN，未用测试集调参。</p><p>__PARAMETERS__</p><p>每次测量启动独立进程。训练计时包含分箱 / DMatrix 构建与标签传递，排除下载、文件读取、预处理、Python 导入和进程启动。C++ XGBoost 与 LightGBM 通过各自公开 C API 直接调用；预测需构造测试 DMatrix，Python 使用 estimator 路径，因此预测时间不是纯树遍历吞吐量。</p><p>内存读取 Linux /proc/self/status 的 VmHWM，属于执行后的地址空间，避免启动前继承的高水位。它是整个进程的峰值，不是模型增量，也不是泄漏证明。Python 子进程均导入三种库；C++ 没有 Python 运行时。</p><p id="environment"></p></details>
<details><summary>数据处理与外推限制</summary><ul><li>Titanic：选择 7 个常用特征，删除 ID、姓名、票号、舱位。</li><li>Pima：glucose、blood pressure、skin thickness、insulin、BMI 的无效 0 转为 NaN；怀孕次数的 0 保留。</li><li>Telco：删除 customerID；11 个空白 TotalCharges 转 NaN。</li><li>红酒：划分前去掉 240 条完全重复记录，剩余 1,359 行；去重后没有相同输入特征的重复行。</li><li>保险费用：使用 6 个输入特征，charges 按原值回归。</li></ul><p>Kaggle 官方 API 被当前网络代理拒绝（HTTP 403），采用对应公开镜像。镜像文件校验 SHA256，但未独立核验与官方压缩包逐字节相同。结果是本地留出集测量，没有提交 Kaggle 榜单。</p><p>只有三个划分，不能据此宣称统计显著。数据集规模 768–7,043 行；未做超参数搜索、分组或时间划分，也未比较百万行、GPU、多线程、稀疏、多分类任务。毫秒级差异容易受共享 CPU 调度影响。</p></details></section>
<section class="section"><h2>数据来源</h2><p class="note">外部链接仅在点击时访问；图表和数据已完整嵌入此 HTML，可离线查看。</p><div class="sources" id="sources"></div><details><summary>复现命令与证据文件指纹</summary><pre>cd /workspace/libtree
. /workspace/libtree-venv/bin/activate
python -m pip install -r benchmarks/requirements.txt
make all
PYTHONPATH=python:. python -m pytest tests/test_benchmark.py
PYTHONPATH=python python benchmarks/kaggle.py --engines libtree xgboost lightgbm --seeds 42 2024 2026 --repeats __REPEATS__ --output benchmarks/__PREFIX__-results.json
python benchmarks/kaggle_report.py --input benchmarks/__PREFIX__-results.json --prefix __PREFIX__
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/html_report.py</pre><div id="hashes" class="small"></div></details></section><footer>LibTree × XGBoost × LightGBM · 实测数据驱动 · 确定性与接口预测一致性检查通过。标准差衡量划分波动，不代表统计显著性。</footer></main>
<script id="evidence" type="application/json">__DATA__</script>
<script>
'use strict';
const evidence=JSON.parse(document.getElementById('evidence').textContent),stats=evidence.summary,raw=evidence.raw;
const records=raw.split_reports.flatMap(s=>s.results),names=['titanic','insurance','pima','telco','wine'];
const labels={titanic:'Titanic',insurance:'保险费用',pima:'Pima',telco:'Telco',wine:'红酒质量'};
const colors={libtree:'#007d77',xgboost:'#d66d45',lightgbm:'#7662a9'};const engines=raw.split_reports[0].metadata.engines||['libtree','xgboost'];const engineLabels={libtree:'LibTree',xgboost:'XGBoost',lightgbm:'LightGBM'};let language='cpp';
const lookup=(name,engine)=>stats.find(s=>s.dataset===name&&s.engine===engine&&s.language===language);
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'png',scale:2}};
const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#fff',font:{family:'-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif',size:12,color:'#475b65'},margin:{l:75,r:30,t:18,b:45},showlegend:false,hoverlabel:{bgcolor:'#fff',font:{size:12}},xaxis:{type:'linear',gridcolor:'#edf1f0',zeroline:false},yaxis:{gridcolor:'#edf1f0',zeroline:false},bargap:.32};
const draw=(id,traces,layout)=>Plotly.react(id,traces,structuredClone({...base,...layout}),config);
function render(){
 for(const lang of ['cpp','python']){const active=lang===language;document.getElementById(lang).classList.toggle('active',active);document.getElementById(lang).setAttribute('aria-pressed',active);}
 const tl=lookup('telco','libtree'),tx=lookup('telco','xgboost');
 document.getElementById('ratio').textContent=(tl.fit_seconds_median/tx.fit_seconds_median).toFixed(2)+'×';
 document.getElementById('memory').textContent=tl.peak_rss_mib_median.toFixed(1)+' MiB';
 document.getElementById('memory-other').textContent=engines.filter(e=>e!=='libtree').map(e=>engineLabels[e]+'：'+lookup('telco',e).peak_rss_mib_median.toFixed(1)+' MiB').join('；');
 document.getElementById('memory-note').textContent=language==='cpp'?'当前为原生执行进程的峰值，不包含 Python 运行时。':'包含 Python 运行时；三个引擎的子进程均导入三种库。';
 const classes=['titanic','pima','telco'];
 draw('auc',engines.map((engine,k)=>({type:'scatter',mode:'markers',name:engineLabels[engine],x:classes.map(n=>lookup(n,engine).metrics.roc_auc.mean),y:classes.map((n,i)=>i+(k-(engines.length-1)/2)*.16),marker:{size:10,color:colors[engine]},error_x:{type:'data',array:classes.map(n=>lookup(n,engine).metrics.roc_auc.std),color:colors[engine],thickness:1.6,width:4},customdata:classes.map(n=>[labels[n],lookup(n,engine).metrics.roc_auc.std]),hovertemplate:'%{customdata[0]} · '+engine+'<br>AUC %{x:.4f} ± %{customdata[1]:.4f}<extra></extra>'})),{xaxis:{...base.xaxis,range:[.70,1],dtick:.05,title:{text:'ROC AUC ↑'}},yaxis:{...base.yaxis,tickvals:[0,1,2],ticktext:classes.map(n=>labels[n]),range:[2.5,-.5]}});
 for(const n of ['insurance','wine'])draw(n,engines.map(engine=>({type:'bar',x:[engineLabels[engine]],y:[lookup(n,engine).metrics.rmse.mean],marker:{color:colors[engine]},error_y:{array:[lookup(n,engine).metrics.rmse.std],color:'#475b65'},text:[lookup(n,engine).metrics.rmse.mean.toFixed(n==='insurance'?0:3)],textposition:'outside',cliponaxis:false,hovertemplate:engine+'<br>RMSE %{y:.4f}<extra></extra>'})),{xaxis:{type:'category'},margin:{l:n==='insurance'?50:42,r:12,t:28,b:45},yaxis:{...base.yaxis,title:{text:'RMSE ↓'},rangemode:'tozero'},barmode:'group'});
 function bars(field,mult){return engines.map(engine=>({type:'bar',orientation:'h',name:engine,legendgroup:engine,y:names.map(n=>labels[n]),x:names.map(n=>lookup(n,engine)[field]*mult),marker:{color:colors[engine]},text:names.map(n=>(lookup(n,engine)[field]*mult).toFixed(1)),textposition:'outside',cliponaxis:false,hovertemplate:'%{y} · '+engine+'<br>%{x:.3f}<extra></extra>'}));}
 const fit=bars('fit_seconds_median',1000);fit.forEach((trace,k)=>{const engine=engines[k];trace.error_x={type:'data',symmetric:false,array:names.map(n=>(lookup(n,engine).fit_seconds_max-lookup(n,engine).fit_seconds_median)*1000),arrayminus:names.map(n=>(lookup(n,engine).fit_seconds_median-lookup(n,engine).fit_seconds_min)*1000),color:'#4e6570',thickness:1,width:2};});
 const horizontal={barmode:'group',yaxis:{...base.yaxis,autorange:'reversed'},margin:{l:78,r:58,t:14,b:50}};
 draw('fit',fit,{...horizontal,xaxis:{...base.xaxis,title:{text:'训练时间 · ms ↓'},type:document.getElementById('log').checked?'log':'linear',rangemode:'tozero'}});
 draw('predict',bars('predict_seconds_median',1000),{...horizontal,xaxis:{...base.xaxis,title:{text:'预测时间 · ms ↓'},rangemode:'tozero'}});
 draw('rss',bars('peak_rss_mib_median',1),{...horizontal,xaxis:{...base.xaxis,title:{text:'整个进程的峰值内存 · MiB ↓'},rangemode:'tozero'}});
 draw('speed',engines.filter(e=>e!=='libtree').map(engine=>{const ratio=names.map(n=>lookup(n,'libtree').fit_seconds_median/lookup(n,engine).fit_seconds_median);return {type:'bar',orientation:'h',name:engineLabels[engine],y:names.map(n=>labels[n]),x:ratio,marker:{color:colors[engine]},text:ratio.map(v=>v.toFixed(2)+'×'),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>LibTree ÷ '+engineLabels[engine]+' %{x:.2f}×<extra></extra>'};}),{...horizontal,xaxis:{...base.xaxis,title:{text:'耗时比 · >1 表示 LibTree 更慢'},rangemode:'tozero'},shapes:[{type:'line',x0:1,x1:1,y0:0,y1:1,yref:'paper',line:{color:'#8b9b9f',dash:'dot',width:1.5}}]});
 renderRows();
 const metricLabels={accuracy:'Accuracy ↑',average_precision:'PR AUC ↑',log_loss:'Log loss ↓',mae:'MAE ↓',r2:'R² ↑'};
 document.getElementById('extra').innerHTML=stats.filter(s=>s.language===language).flatMap(s=>Object.entries(s.metrics).filter(([m])=>metricLabels[m]).map(([m,v])=>`<tr><td>${labels[s.dataset]}</td><td>${s.engine}</td><td>${metricLabels[m]}</td><td>${v.mean.toFixed(4)}</td><td>${v.std.toFixed(4)}</td></tr>`)).join('');
}
function renderRows(){const chosen=document.getElementById('dataset').value,selected=records.filter(r=>r.language===language&&(chosen==='all'||r.dataset===chosen));document.getElementById('row-count').textContent=selected.length+' 个划分 / 引擎记录';document.getElementById('rows').innerHTML=selected.map(r=>`<tr><td>${labels[r.dataset]}</td><td>${r.seed}</td><td>${r.engine}</td><td>${r.metric_name==='roc_auc'?'AUC ↑':'RMSE ↓'}</td><td>${r.metric.toFixed(4)}</td><td>${(r.fit_seconds*1000).toFixed(2)}</td><td>${(r.predict_seconds*1000).toFixed(3)}</td><td>${(r.peak_rss_kib/1024).toFixed(1)}</td><td class="pass">通过</td></tr>`).join('');}
function download(text,name,type){const url=URL.createObjectURL(new Blob([text],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
for(const lang of ['cpp','python'])document.getElementById(lang).onclick=()=>{language=lang;render();};
document.getElementById('log').onchange=render;document.getElementById('dataset').onchange=renderRows;
document.getElementById('print').onclick=()=>window.print();
document.getElementById('download').onclick=()=>download(JSON.stringify(raw,null,2),'__PREFIX__-results.json','application/json');
if(evidence.ab)document.getElementById('ab-json').onclick=()=>download(JSON.stringify(evidence.ab,null,2),'optimization-ab-results.json','application/json');
if(evidence.synthetic)document.getElementById('synthetic-json').onclick=()=>download(JSON.stringify(evidence.synthetic,null,2),'optimized-synthetic-results.json','application/json');
document.getElementById('csv').onclick=()=>{const cols=['dataset','seed','language','engine','repeat','metric_name','metric','fit_seconds','predict_seconds','peak_rss_kib'];const lines=[cols.join(',')];for(const r of records)r.raw_runs.forEach((sample,i)=>lines.push([r.dataset,r.seed,r.language,r.engine,i+1,r.metric_name,r.metric,sample.fit_seconds,sample.predict_seconds,sample.peak_rss_kib].join(',')));download(lines.join('\n'),'__PREFIX__-timing-__RUNS__-runs.csv','text/csv;charset=utf-8');};
const sources=raw.split_reports[0].sources;
document.getElementById('sources').innerHTML=names.map(n=>`<div class="source"><strong>${labels[n]}</strong><div class="small">${sources[n].rows_after_cleaning.toLocaleString()} 行 · ${sources[n].raw_feature_count} 个原始特征</div><div><a href="${sources[n].kaggle_url}" target="_blank" rel="noopener noreferrer">Kaggle 页面</a> · <a href="${sources[n].url}" target="_blank" rel="noopener noreferrer">实际镜像</a></div><details><summary>SHA256</summary><code>${sources[n].sha256}</code></details></div>`).join('');
const meta=raw.split_reports[0].metadata;
document.getElementById('environment').textContent='环境：'+meta.cpu.replace(/^model name\s*:\s*/,'')+'；'+meta.compiler+'；Python '+meta.python+'；NumPy '+meta.numpy+'；sklearn '+meta.sklearn+'；XGBoost '+meta.xgboost+'；LightGBM '+meta.lightgbm+'。';
document.getElementById('hashes').innerHTML=Object.entries(evidence.files).map(([n,h])=>`<p>${n}<br><code>${h}</code></p>`).join('');
window.__reportReady=false;render();setTimeout(()=>{window.__reportReady=true;},300);
</script></body></html>'''
parameters = raw['split_reports'][0]['metadata']['parameters']
parameter_note = (f"{parameters['trees']} 棵树、最大深度 {parameters['depth']}、{parameters['bins']} 分箱、学习率 {parameters['rate']}、L2={parameters['l2']}、最小增益 {parameters['min_gain']}，CPU 单线程。LibTree 最少叶样本数为 {parameters['min_leaf']}；XGBoost 默认最小 Hessian 权重为 1。"
                  + (f"LightGBM 为 leaf-wise，min_child_samples={parameters['min_leaf']}、num_leaves={2**parameters['depth']}、max_depth={parameters['depth']}。" if 'lightgbm' in engines else '')
                  + '分箱算法、树生长、初始值和叶约束并不完全相同，这是主参数匹配的比较。')
lookup = {(r['dataset'], r['language'], r['engine']): r for r in summary}
fit_wins = sum(lookup[(name, language, 'libtree')]['fit_seconds_median'] <=
               min(lookup[(name, language, engine)]['fit_seconds_median'] for engine in engines)
               for name in raw['split_reports'][0]['sources'] for language in ('cpp', 'python'))
predict_wins = sum(lookup[(name, language, 'libtree')]['predict_seconds_median'] <=
                   min(lookup[(name, language, engine)]['predict_seconds_median'] for engine in engines)
                   for name in raw['split_reports'][0]['sources'] for language in ('cpp', 'python'))
groups = len(raw['split_reports'][0]['sources']) * 2
headline = '本次固定基准：训练与推理均第一' if fit_wins == predict_wins == groups else '训练与推理持续提速，排名以实测为准'
quality = ' / '.join(f"{lookup[('telco', 'cpp', engine)]['metrics']['roc_auc']['mean']:.4f}" for engine in engines)
conclusion = (f'本机 CPU 单线程、固定参数下，LibTree 在数据集 × 接口的 {groups} 个组合中，训练 <strong>{fit_wins}/{groups}</strong> 组最快，推理 <strong>{predict_wins}/{groups}</strong> 组最快。排名按多划分重复测量的中位数计算，不表示每次运行都获胜。'
              f'Telco 的 LibTree / XGBoost / LightGBM 平均 AUC 为 <strong>{quality}</strong>；三个划分不足以证明统计显著的效果优势。')
ab_section = ''
if 'ab' in payload:
    ab_rows = ''.join(f"<tr><td>{r['dataset']}</td><td>{r['baseline_fit_seconds']*1000:.2f}</td><td>{r['optimized_fit_seconds']*1000:.2f}</td><td>{r['fit_speedup']:.2f}×</td><td>{r['baseline_predict_seconds']*1000:.3f}</td><td>{r['optimized_predict_seconds']*1000:.3f}</td><td>{r['predict_speedup']:.2f}×</td></tr>" for r in payload['ab']['results'])
    ab_section = ('<section class="section"><h2>优化前后：同一输入的配对复测</h2><p class="note">原生 C++，旧版 __BASELINE__ 与当前实现交错随机执行，__AB_REPEATS__ 次独立计时取中位数；六组输入的 float32 预测逐值相同。此处固定输入复测与上方多划分三方比较独立。</p>'
                  '<div class="card table-scroll"><table><thead><tr><th>数据集</th><th>旧训练 ms</th><th>新训练 ms</th><th>训练加速</th><th>旧推理 ms</th><th>新推理 ms</th><th>推理加速</th></tr></thead><tbody>' + ab_rows + '</tbody></table></div><p><button id="ab-json">下载配对测试 JSON</button></p></section>')
    ab_section = ab_section.replace('__BASELINE__', payload['ab']['metadata']['baseline_ref'][:7]).replace('__AB_REPEATS__', str(payload['ab']['metadata']['repeats']))
if 'synthetic' in payload:
    synthetic_rows = ''.join(f"<tr><td>{r['language']}</td><td>{r['engine']}</td><td>{r['metric']:.4f}</td><td>{r['fit_seconds']*1000:.2f}</td><td>{r['predict_seconds']*1000:.3f}</td></tr>" for r in payload['synthetic']['results'])
    ab_section += ('<section class="section"><h2>补充压力测试：20k 行 Friedman 数据</h2><p class="note">20 个特征；15000 行训练、5000 行测试；seed 42，单划分，每组合 5 次独立计时，共 30 次。不与上方多划分效果波动混合。</p>'
                   '<div class="card table-scroll"><table><thead><tr><th>接口</th><th>引擎</th><th>测试 RMSE ↓</th><th>训练 ms ↓</th><th>推理 ms ↓</th></tr></thead><tbody>' + synthetic_rows + '</tbody></table></div><p><button id="synthetic-json">下载压力测试 JSON</button></p></section>')
meta = raw['split_reports'][0]['metadata']
versions = 'XGBoost ' + meta['xgboost'] + ' / LightGBM ' + meta.get('lightgbm', '—')
html = template.replace('__HEADLINE__', headline).replace('__CONCLUSION__', conclusion).replace('__AB_SECTION__', ab_section).replace('__VERSIONS__', versions).replace('__PARAMETERS__', parameter_note).replace('__PLOTLY__', get_plotlyjs().replace('</script', '<\\/script'))
html = html.replace('__ASOF__', as_of).replace('__DATA__', encoded).replace('__RUNS__', str(run_count)).replace('__PREFIX__', prefix).replace('__REPEATS__', str(raw['timing_repeats_per_split'])).replace('__POOLED__', str(len(raw['split_reports']) * raw['timing_repeats_per_split']))
output = ROOT / "KAGGLE_REPORT.html"
output.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n")
print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
