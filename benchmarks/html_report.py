"""Build a self-contained, offline interactive HTML from verified measurements."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
raw_path = ROOT / "kaggle-results.json"
summary_path = ROOT / "kaggle-summary.json"
raw = json.loads(raw_path.read_text())
summary = json.loads(summary_path.read_text())
records = [r for s in raw["split_reports"] for r in s["results"]]
assert len(records) == 60
assert sum(len(r["raw_runs"]) for r in records) == 180
assert all(all(r["checks"].values()) for r in records)
assert all("VmHWM" in s["metadata"]["memory_measurement"] for s in raw["split_reports"])
assert len(summary) == 20
for row in summary:
    subset = [r for r in records if (r["dataset"], r["language"], r["engine"]) ==
              (row["dataset"], row["language"], row["engine"])]
    assert len(subset) == 3
    from statistics import mean, stdev, median
    for metric, value in row["metrics"].items():
        values = [r["metrics"][metric] for r in subset]
        assert abs(mean(values) - value["mean"]) < 1e-10
        assert abs(stdev(values) - value["std"]) < 1e-10
    samples = [run for r in subset for run in r["raw_runs"]]
    assert abs(median(r["fit_seconds"] for r in samples) - row["fit_seconds_median"]) < 1e-10

as_of = datetime.fromisoformat(raw["split_reports"][-1]["metadata"]["date_utc"].replace("Z", "+00:00"))
as_of = as_of.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
payload = {"raw": raw, "summary": summary,
           "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw_path, summary_path)}}
encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
template = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggle GBDT 实测报告 · LibTree × XGBoost</title>
<style>
:root{--ink:#172b37;--muted:#62707b;--line:#dfe5e6;--paper:#f6f8f7;--teal:#007d77;--orange:#d66d45}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}header{background:#16343d;color:#fff;padding:48px 0 40px}.wrap{max-width:1220px;margin:auto;padding:0 32px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#b5d4d3}h1{font-size:clamp(26px,3vw,39px);line-height:1.3;margin:12px 0 12px;letter-spacing:-1px}h2{font-size:24px;line-height:1.4;margin:0 0 6px}h3{font-size:17px;margin:0 0 5px}p{margin:8px 0}.sub{color:#c6d7da;max-width:920px}.badge{display:inline-block;border:1px solid #49666e;border-radius:30px;padding:4px 12px;margin:12px 6px 0 0;font-size:12px}.toolbar{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);box-shadow:0 3px 10px #19333d05}.toolbar .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-top:13px;padding-bottom:13px}.tabs{display:flex;gap:4px;padding:4px;border:1px solid var(--line);border-radius:10px;background:#f6f8f7}button,select{font:inherit;border:1px solid #cdd7d9;background:#fff;color:var(--ink);border-radius:7px;padding:7px 13px;cursor:pointer}button:hover{border-color:var(--teal)}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #59bdb2;outline-offset:3px}.tabs button{border:0;background:transparent;font-weight:600}.tabs button.active{background:#16343d;color:white}.actions{display:flex;gap:7px;flex-wrap:wrap}.label{font-size:12px;color:var(--muted)}main{padding-bottom:56px}.summary{padding:28px 0 20px}.summary p{max-width:1030px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0 6px}.kpi{background:white;border:1px solid var(--line);border-radius:12px;padding:16px 20px}.big{font-size:31px;line-height:1.25;font-weight:650;letter-spacing:-1px}.kpi .label{margin-bottom:7px}.small{font-size:12px;color:var(--muted);line-height:1.6}.section{margin:28px 0}.section-head{margin-bottom:15px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:20px;min-width:0}.card p{color:var(--muted);font-size:13px;margin:5px 0 8px}.chart{height:330px}.chart.short{height:265px}.chart.wide{height:340px}.row{display:flex;align-items:center;justify-content:space-between;gap:12px}.legend{display:flex;gap:17px;align-items:center;font-size:12px}.dot{width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:6px}.teal{background:var(--teal)}.orange{background:var(--orange)}.notice{border-left:3px solid var(--teal);padding:10px 16px;background:#edf4f2;border-radius:0 6px 6px 0;font-size:13px;margin:17px 0}.regression{display:grid;grid-template-columns:1fr 1fr;gap:10px}.regression .chart{height:260px}.note{font-size:13px;color:var(--muted);margin-top:12px}.sources{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.source{padding:15px;background:white;border:1px solid var(--line);border-radius:9px;overflow-wrap:anywhere}.source a{font-size:13px}a{color:var(--teal);text-underline-offset:3px}details{background:white;border:1px solid var(--line);border-radius:12px;margin:16px 0;padding:17px 20px}summary{font-weight:600;cursor:pointer}details[open] summary{margin-bottom:16px}.table-scroll{overflow:auto;max-height:540px;border:1px solid var(--line);margin:14px 0}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #e7edec;text-align:right}th{background:#f0f5f4;color:#485c66;position:sticky;top:0;z-index:1}th:first-child,td:first-child{text-align:left}tbody tr:hover{background:#f6faf9}.pass{color:var(--teal)}code{font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}pre{padding:16px;background:#f0f4f3;overflow:auto;border-radius:7px}footer{padding-top:25px;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}.nojs{padding:20px;background:#fff1df}input[type=checkbox]{accent-color:var(--teal)}.plotly .modebar{top:0!important}@media(max-width:800px){.wrap{padding:0 18px}header{padding:32px 0}.grid,.sources{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.toolbar .wrap{align-items:flex-start}.actions button{font-size:12px;padding:6px 8px}.regression{grid-template-columns:1fr}.card{padding:16px}.chart{height:310px}.row{flex-wrap:wrap}}@media print{.toolbar,.modebar,.actions{display:none!important}header{background:white!important;color:var(--ink);padding:20px 0}.sub,.eyebrow{color:var(--muted)}.badge{border-color:var(--line)}body{background:white}.card,.kpi,details{break-inside:avoid}.wrap{max-width:100%;padding:0 16px}.grid{grid-template-columns:1fr}.sources{grid-template-columns:1fr 1fr}.section{break-inside:auto}details{display:block}.table-scroll{max-height:none}.chart{height:300px}}
</style><script>__PLOTLY__</script></head><body>
<header><div class="wrap"><div class="eyebrow">LIBTREE LAB / 实测基准</div><h1>效果接近，Telco 上的训练速度差距最明显</h1><p class="sub">五个 Kaggle 数据集对应公开镜像 · 三个不同随机划分 · C++ / Python 双接口比较</p><div><span class="badge">180 次有效测量</span><span class="badge">XGBoost 3.4.1</span><span class="badge">CPU 单线程</span><span class="badge">数据截至 __ASOF__ · 北京时间</span></div></div></header>
<div class="toolbar"><div class="wrap"><div class="tabs" role="group" aria-label="接口选择"><button class="active" id="cpp" aria-pressed="true">C++ 原生</button><button id="python" aria-pressed="false">Python 接口</button></div><div class="actions"><button id="download">下载原始 JSON</button><button id="csv">下载计时 CSV</button><button id="print">打印 / PDF</button></div></div></div>
<noscript><div class="wrap nojs">交互图表需要启用 JavaScript；本文件已内置图表库与全部实测数据，不需要联网。</div></noscript>
<main class="wrap"><section class="summary"><h2>测试结论</h2><p>固定主参数下，两种模型在这些数据上的效果接近，具体差异随划分而变化。Telco 的平均 AUC 均为 <strong>0.8465</strong>，XGBoost 训练更快；LibTree 的原生进程内存占用较小。三个划分不足以证明统计显著的效果优势。</p><div class="kpis"><div class="kpi"><div class="label">测试范围</div><div class="big">5 / 3</div><div class="small">数据集 / 不同随机划分</div></div><div class="kpi"><div class="label">Telco · LibTree 训练耗时倍数</div><div class="big" id="ratio">—</div><div class="small">LibTree ÷ XGBoost，越低越快</div></div><div class="kpi"><div class="label">Telco · LibTree 峰值进程内存</div><div class="big" id="memory">—</div><div class="small" id="memory-other"></div></div><div class="kpi"><div class="label">验证状态</div><div class="big">全部通过</div><div class="small">有限预测 · 基线 · 重复确定性 · 接口一致性</div></div></div></section>
<section class="section"><div class="section-head row"><div><h2>效果差异与划分波动</h2><p class="note">点 / 柱为三个划分的均值；误差线为 ±1 个样本标准差，不是置信区间。</p></div><div class="legend"><span><i class="dot teal"></i>LibTree</span><span><i class="dot orange"></i>XGBoost</span></div></div><div class="grid"><article class="card"><h3>二分类 · ROC AUC ↑</h3><p>Titanic、Pima、Telco；点图横轴范围为 0.70–1.00。</p><div id="auc" class="chart" role="img" aria-label="三项二分类任务的 AUC 均值与标准差"></div></article><article class="card"><h3>回归 · RMSE ↓</h3><p>两个任务的目标单位不同，使用独立坐标轴。</p><div class="regression"><div><div class="label">保险费用 · 原始金额单位</div><div id="insurance" class="chart short" role="img" aria-label="保险费用 RMSE 比较"></div></div><div><div class="label">红酒质量 · quality 分数</div><div id="wine" class="chart short" role="img" aria-label="红酒质量 RMSE 比较"></div></div></div></article></div><div class="notice">Pima 的 AUC 跨划分波动较大：LibTree 为 0.8074 ± 0.0390，XGBoost 为 0.8047 ± 0.0338。两者的均值差异远小于这里观察到的划分波动。</div></section>
<section class="section"><div class="section-head"><h2>训练、预测和内存</h2><p class="note">每个组合包含 3 个划分 × 3 次计时。条形为 9 次运行中位数；训练误差线为最小–最大范围。</p></div><div class="grid"><article class="card"><div class="row"><h3>训练时间 · ms ↓</h3><label class="small"><input id="log" type="checkbox"> 对数坐标</label></div><p>包含分箱 / DMatrix 构建；不包含导入和预处理。</p><div id="fit" class="chart wide" role="img" aria-label="五个数据集的训练时间比较"></div></article><article class="card"><h3>预测时间 · ms ↓</h3><p>端到端接口计时，包含各接口必要的转换。</p><div id="predict" class="chart wide" role="img" aria-label="五个数据集的预测时间比较"></div></article><article class="card"><h3>峰值进程内存 · MiB ↓</h3><p id="memory-note"></p><div id="rss" class="chart" role="img" aria-label="五个数据集的进程峰值内存比较"></div></article><article class="card"><h3>训练时间比值 · LibTree ÷ XGBoost</h3><p>大于 1 表示 LibTree 更慢；虚线为等速参考线。</p><div id="speed" class="chart" role="img" aria-label="LibTree 相对 XGBoost 的训练时间比值"></div></article></div></section>
<section class="section"><h2>查看测量与方法</h2><details><summary>逐划分结果与完整计时数据</summary><div class="row"><label>筛选数据集 <select id="dataset"><option value="all">全部数据集</option><option value="titanic">Titanic</option><option value="insurance">保险费用</option><option value="pima">Pima</option><option value="telco">Telco</option><option value="wine">红酒质量</option></select></label><span class="small" id="row-count"></span></div><p class="note">本表每行对应一个划分、接口和引擎；时间和内存取该行三次计时的中位数。顶部 CSV 包含全部 180 次计时；JSON 包含附加指标、数据校验值和来源。</p><div class="table-scroll"><table><thead><tr><th>数据集</th><th>Seed</th><th>引擎</th><th>指标</th><th>效果</th><th>训练 ms</th><th>预测 ms</th><th>峰值 MiB</th><th>验证</th></tr></thead><tbody id="rows"></tbody></table></div></details>
<details><summary>附加效果指标：准确率、PR AUC、Log loss、MAE 与 R²</summary><p class="note">均值 ± 样本标准差；二分类准确率的阈值为 0.5。PR AUC 在本报告中指 average precision。</p><div class="table-scroll"><table><thead><tr><th>数据集</th><th>引擎</th><th>指标</th><th>均值</th><th>标准差</th></tr></thead><tbody id="extra"></tbody></table></div></details>
<details><summary>划分、参数与计时边界</summary><p>75% 训练 / 25% 测试；随机种子 42、2024、2026；二分类分层抽样。类别编码只在训练集上拟合，忽略测试集新类别。缺失值保留为 NaN，未用测试集调参。</p><p>100 棵树、最大深度 4、64 分箱、学习率 0.1、L2=1，CPU 单线程。LibTree 最少叶样本数为 5；XGBoost 默认最小 Hessian 权重为 1。分箱算法、初始值和叶约束并不完全相同，这是主参数匹配的比较。</p><p>每次测量启动独立进程。训练计时包含分箱 / DMatrix 构建与标签传递，排除下载、文件读取、预处理、Python 导入和进程启动。C++ XGBoost 通过公开 C API 直接调用；预测需构造测试 DMatrix，Python 使用 estimator 路径，因此预测时间不是纯树遍历吞吐量。</p><p>内存读取 Linux /proc/self/status 的 VmHWM，属于执行后的地址空间，避免启动前继承的高水位。它是整个进程的峰值，不是模型增量，也不是泄漏证明。Python 子进程均导入两种库；C++ 没有 Python 运行时。</p><p id="environment"></p></details>
<details><summary>数据处理与外推限制</summary><ul><li>Titanic：选择 7 个常用特征，删除 ID、姓名、票号、舱位。</li><li>Pima：glucose、blood pressure、skin thickness、insulin、BMI 的无效 0 转为 NaN；怀孕次数的 0 保留。</li><li>Telco：删除 customerID；11 个空白 TotalCharges 转 NaN。</li><li>红酒：划分前去掉 240 条完全重复记录，剩余 1,359 行；去重后没有相同输入特征的重复行。</li><li>保险费用：使用 6 个输入特征，charges 按原值回归。</li></ul><p>Kaggle 官方 API 被当前网络代理拒绝（HTTP 403），采用对应公开镜像。镜像文件校验 SHA256，但未独立核验与官方压缩包逐字节相同。结果是本地留出集测量，没有提交 Kaggle 榜单。</p><p>只有三个划分，不能据此宣称统计显著。数据集规模 768–7,043 行；未做超参数搜索、分组或时间划分，也未比较百万行、GPU、多线程、稀疏、多分类任务。毫秒级差异容易受共享 CPU 调度影响。</p></details></section>
<section class="section"><h2>数据来源</h2><p class="note">外部链接仅在点击时访问；图表和数据已完整嵌入此 HTML，可离线查看。</p><div class="sources" id="sources"></div><details><summary>复现命令与证据文件指纹</summary><pre>cd /workspace/libtree
. /workspace/libtree-venv/bin/activate
python -m pip install -r benchmarks/requirements.txt
make all
PYTHONPATH=python:. python -m pytest tests/test_benchmark.py
PYTHONPATH=python python benchmarks/kaggle.py --seeds 42 2024 2026 --repeats 3
python benchmarks/kaggle_report.py
python -m pip install -r benchmarks/requirements-report.txt
python benchmarks/html_report.py</pre><div id="hashes" class="small"></div></details></section><footer>LibTree × XGBoost · 实测数据驱动 · 确定性与接口预测一致性检查通过。标准差衡量划分波动，不代表统计显著性。</footer></main>
<script id="evidence" type="application/json">__DATA__</script>
<script>
'use strict';
const evidence=JSON.parse(document.getElementById('evidence').textContent),stats=evidence.summary,raw=evidence.raw;
const records=raw.split_reports.flatMap(s=>s.results),names=['titanic','insurance','pima','telco','wine'];
const labels={titanic:'Titanic',insurance:'保险费用',pima:'Pima',telco:'Telco',wine:'红酒质量'};
const colors={libtree:'#007d77',xgboost:'#d66d45'};let language='cpp';
const lookup=(name,engine)=>stats.find(s=>s.dataset===name&&s.engine===engine&&s.language===language);
const config={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'png',scale:2}};
const base={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'#fff',font:{family:'-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif',size:12,color:'#475b65'},margin:{l:75,r:30,t:18,b:45},showlegend:false,hoverlabel:{bgcolor:'#fff',font:{size:12}},xaxis:{gridcolor:'#edf1f0',zeroline:false},yaxis:{gridcolor:'#edf1f0',zeroline:false},bargap:.32};
const draw=(id,traces,layout)=>Plotly.react(id,traces,{...base,...layout},config);
function render(){
 for(const lang of ['cpp','python']){const active=lang===language;document.getElementById(lang).classList.toggle('active',active);document.getElementById(lang).setAttribute('aria-pressed',active);}
 const tl=lookup('telco','libtree'),tx=lookup('telco','xgboost');
 document.getElementById('ratio').textContent=(tl.fit_seconds_median/tx.fit_seconds_median).toFixed(2)+'×';
 document.getElementById('memory').textContent=tl.peak_rss_mib_median.toFixed(1)+' MiB';
 document.getElementById('memory-other').textContent='XGBoost：'+tx.peak_rss_mib_median.toFixed(1)+' MiB';
 document.getElementById('memory-note').textContent=language==='cpp'?'当前为原生执行进程的峰值，不包含 Python 运行时。':'包含 Python 运行时；两个引擎的子进程均导入两种库。';
 const classes=['titanic','pima','telco'];
 draw('auc',['libtree','xgboost'].map((engine,k)=>({type:'scatter',mode:'markers',name:engine==='libtree'?'LibTree':'XGBoost',x:classes.map(n=>lookup(n,engine).metrics.roc_auc.mean),y:classes.map((n,i)=>i+(k===0?-.11:.11)),marker:{size:10,color:colors[engine]},error_x:{type:'data',array:classes.map(n=>lookup(n,engine).metrics.roc_auc.std),color:colors[engine],thickness:1.6,width:4},customdata:classes.map(n=>[labels[n],lookup(n,engine).metrics.roc_auc.std]),hovertemplate:'%{customdata[0]} · '+engine+'<br>AUC %{x:.4f} ± %{customdata[1]:.4f}<extra></extra>'})),{xaxis:{...base.xaxis,range:[.70,1],dtick:.05,title:{text:'ROC AUC ↑'}},yaxis:{...base.yaxis,tickvals:[0,1,2],ticktext:classes.map(n=>labels[n]),range:[2.5,-.5]}});
 for(const n of ['insurance','wine'])draw(n,['libtree','xgboost'].map(engine=>({type:'bar',x:[engine==='libtree'?'LibTree':'XGBoost'],y:[lookup(n,engine).metrics.rmse.mean],marker:{color:colors[engine]},error_y:{array:[lookup(n,engine).metrics.rmse.std],color:'#475b65'},text:[lookup(n,engine).metrics.rmse.mean.toFixed(n==='insurance'?0:3)],textposition:'outside',cliponaxis:false,hovertemplate:engine+'<br>RMSE %{y:.4f}<extra></extra>'})),{margin:{l:n==='insurance'?50:42,r:12,t:28,b:45},yaxis:{...base.yaxis,title:{text:'RMSE ↓'},rangemode:'tozero'},barmode:'group'});
 function bars(field,mult){return ['libtree','xgboost'].map(engine=>({type:'bar',orientation:'h',name:engine,legendgroup:engine,y:names.map(n=>labels[n]),x:names.map(n=>lookup(n,engine)[field]*mult),marker:{color:colors[engine]},text:names.map(n=>(lookup(n,engine)[field]*mult).toFixed(1)),textposition:'outside',cliponaxis:false,hovertemplate:'%{y} · '+engine+'<br>%{x:.3f}<extra></extra>'}));}
 const fit=bars('fit_seconds_median',1000);fit.forEach((trace,k)=>{const engine=k===0?'libtree':'xgboost';trace.error_x={type:'data',symmetric:false,array:names.map(n=>(lookup(n,engine).fit_seconds_max-lookup(n,engine).fit_seconds_median)*1000),arrayminus:names.map(n=>(lookup(n,engine).fit_seconds_median-lookup(n,engine).fit_seconds_min)*1000),color:'#4e6570',thickness:1,width:2};});
 const horizontal={barmode:'group',yaxis:{...base.yaxis,autorange:'reversed'},margin:{l:78,r:58,t:14,b:50}};
 draw('fit',fit,{...horizontal,xaxis:{...base.xaxis,title:{text:'训练时间 · ms ↓'},type:document.getElementById('log').checked?'log':'linear',rangemode:'tozero'}});
 draw('predict',bars('predict_seconds_median',1000),{...horizontal,xaxis:{...base.xaxis,title:{text:'预测时间 · ms ↓'},rangemode:'tozero'}});
 draw('rss',bars('peak_rss_mib_median',1),{...horizontal,xaxis:{...base.xaxis,title:{text:'整个进程的峰值内存 · MiB ↓'},rangemode:'tozero'}});
 const ratios=names.map(n=>lookup(n,'libtree').fit_seconds_median/lookup(n,'xgboost').fit_seconds_median);
 draw('speed',[{type:'bar',orientation:'h',y:names.map(n=>labels[n]),x:ratios,marker:{color:ratios.map(v=>v>1?'#d66d45':'#007d77')},text:ratios.map(v=>v.toFixed(2)+'×'),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>耗时比 %{x:.2f}×<extra></extra>'}],{...horizontal,xaxis:{...base.xaxis,title:{text:'耗时比 · >1 表示 LibTree 更慢'},rangemode:'tozero'},shapes:[{type:'line',x0:1,x1:1,y0:0,y1:1,yref:'paper',line:{color:'#8b9b9f',dash:'dot',width:1.5}}]});
 renderRows();
 const metricLabels={accuracy:'Accuracy ↑',average_precision:'PR AUC ↑',log_loss:'Log loss ↓',mae:'MAE ↓',r2:'R² ↑'};
 document.getElementById('extra').innerHTML=stats.filter(s=>s.language===language).flatMap(s=>Object.entries(s.metrics).filter(([m])=>metricLabels[m]).map(([m,v])=>`<tr><td>${labels[s.dataset]}</td><td>${s.engine}</td><td>${metricLabels[m]}</td><td>${v.mean.toFixed(4)}</td><td>${v.std.toFixed(4)}</td></tr>`)).join('');
}
function renderRows(){const chosen=document.getElementById('dataset').value,selected=records.filter(r=>r.language===language&&(chosen==='all'||r.dataset===chosen));document.getElementById('row-count').textContent=selected.length+' 个划分 / 引擎记录';document.getElementById('rows').innerHTML=selected.map(r=>`<tr><td>${labels[r.dataset]}</td><td>${r.seed}</td><td>${r.engine}</td><td>${r.metric_name==='roc_auc'?'AUC ↑':'RMSE ↓'}</td><td>${r.metric.toFixed(4)}</td><td>${(r.fit_seconds*1000).toFixed(2)}</td><td>${(r.predict_seconds*1000).toFixed(3)}</td><td>${(r.peak_rss_kib/1024).toFixed(1)}</td><td class="pass">通过</td></tr>`).join('');}
function download(text,name,type){const url=URL.createObjectURL(new Blob([text],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
for(const lang of ['cpp','python'])document.getElementById(lang).onclick=()=>{language=lang;render();};
document.getElementById('log').onchange=render;document.getElementById('dataset').onchange=renderRows;
document.getElementById('print').onclick=()=>window.print();
document.getElementById('download').onclick=()=>download(JSON.stringify(raw,null,2),'kaggle-results.json','application/json');
document.getElementById('csv').onclick=()=>{const cols=['dataset','seed','language','engine','repeat','metric_name','metric','fit_seconds','predict_seconds','peak_rss_kib'];const lines=[cols.join(',')];for(const r of records)r.raw_runs.forEach((sample,i)=>lines.push([r.dataset,r.seed,r.language,r.engine,i+1,r.metric_name,r.metric,sample.fit_seconds,sample.predict_seconds,sample.peak_rss_kib].join(',')));download(lines.join('\n'),'kaggle-timing-180-runs.csv','text/csv;charset=utf-8');};
const sources=raw.split_reports[0].sources;
document.getElementById('sources').innerHTML=names.map(n=>`<div class="source"><strong>${labels[n]}</strong><div class="small">${sources[n].rows_after_cleaning.toLocaleString()} 行 · ${sources[n].raw_feature_count} 个原始特征</div><div><a href="${sources[n].kaggle_url}" target="_blank" rel="noopener noreferrer">Kaggle 页面</a> · <a href="${sources[n].url}" target="_blank" rel="noopener noreferrer">实际镜像</a></div><details><summary>SHA256</summary><code>${sources[n].sha256}</code></details></div>`).join('');
const meta=raw.split_reports[0].metadata;
document.getElementById('environment').textContent='环境：'+meta.cpu.replace(/^model name\s*:\s*/,'')+'；'+meta.compiler+'；Python '+meta.python+'；NumPy '+meta.numpy+'；sklearn '+meta.sklearn+'；XGBoost '+meta.xgboost+'。';
document.getElementById('hashes').innerHTML=Object.entries(evidence.files).map(([n,h])=>`<p>${n}<br><code>${h}</code></p>`).join('');
window.__reportReady=false;render();setTimeout(()=>{window.__reportReady=true;},300);
</script></body></html>'''
html = template.replace('__PLOTLY__', get_plotlyjs().replace('</script', '<\\/script'))
html = html.replace('__ASOF__', as_of).replace('__DATA__', encoded)
output = ROOT / "KAGGLE_REPORT.html"
output.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n")
print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")
