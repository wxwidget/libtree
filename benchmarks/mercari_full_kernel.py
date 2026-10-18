"""Build a private source-only Kernel for the frozen complete Mercari blend.

Public MiniLM initialization is external, untrained on Mercari prices. Every
price expert and residual model is refitted inside the same hosted Kernel.
"""

import base64
import importlib.metadata
import io
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = r"""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile,io,base64
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMBA_NUM_THREADS='2',
    TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',USE_TF='0',
    CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='true',RAYON_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
work=Path('/kaggle/working');project=work/'libtree-source';project.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:archive.extractall(project)
os.environ['PYTHONPATH']=os.pathsep.join(str(project/p) for p in ['','python','vendor'])
sources=sorted(str(p) for p in (project/'src').glob('*.cc') if p.name!='cli.cc')
subprocess.run(['g++','-std=c++17','-O3','-DNDEBUG','-shared','-fPIC','-pthread','-I'+str(project/'include'),*sources,'-o',str(project/'python/libtree/_native.so')],check=True)
subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],check=True)
config_path=project/'benchmarks/mercari/full-ensemble-config.json'
config=json.loads(config_path.read_text())
inputs=Path('/kaggle/input')
pretrained=next(p.parent for p in inputs.rglob('model.safetensors') if (p.parent/'1_Pooling/config.json').exists())
weight_hash=hashlib.sha256((pretrained/'model.safetensors').read_bytes()).hexdigest()
assert weight_hash==config['pretrained_weight_sha256'],'pretrained weight hash mismatch'
data=Path('/tmp/libtree-full-data');data.mkdir(exist_ok=False)
for name in ('train.tsv','test_stg2.tsv','sample_submission_stg2.csv'):
    raw=list(inputs.rglob(name))
    if raw:(data/name).symlink_to(raw[0]);continue
    zipped=list(inputs.rglob(name+'.zip'))
    if zipped:(data/(name+'.zip')).symlink_to(zipped[0]);continue
    seven=list(inputs.rglob(name+'.7z'))
    if not seven:raise FileNotFoundError(name)
    subprocess.run(['7z','x',str(seven[0]),'-o'+str(data),'-y'],check=True)
test_name='test_stg2.tsv' if (data/'test_stg2.tsv').exists() else 'test_stg2.tsv.zip'
sample_name='sample_submission_stg2.csv' if (data/'sample_submission_stg2.csv').exists() else 'sample_submission_stg2.csv.zip'
if SMOKE:
    import pandas as pd
    small=data/'smoke';small.mkdir()
    train=pd.read_csv(data/'train.tsv',sep='\t',nrows=100)
    train.to_csv(small/'train.tsv',sep='\t',index=False)
    test=train.iloc[:7].drop(columns=['train_id','price']).assign(test_id=list(range(7)))
    test.to_csv(small/'test.tsv',sep='\t',index=False)
    pd.DataFrame({'test_id':list(range(7)),'price':0.}).to_csv(small/'sample.csv',index=False)
    data=small;test_name='test.tsv';sample_name='sample.csv'
    config.update(tiny_width=8,max_features=50,char_features=30,tree_rounds=3,threads=1,lsa_rounds=3,vector_rounds=3,vector_components=2)
    config['lsa'].update(components=2,max_features=50)
    config['epochs']={n:1 for n in config['epochs']}
    config['transformer']=dict(epochs=1,batch_size=10,length=16,width=16)
    config['feature_count']=88
    config['baseline'].update(max_features=50,char_features=30,max_words=50,neural_members=[],neural_architecture='bag',neural_width=8,neural_loss='mse',neural_rates=[.003],neural_batches=[16],fm_rank=2,fm_epochs=1)
    config['baseline']['native'].update(n_estimators=3,min_samples_leaf=2,n_jobs=1)
    config_path=project/'smoke-config.json';config_path.write_text(json.dumps(config,indent=2))
run=Path('/tmp/libtree-full-run')
start=time.perf_counter()
subprocess.run([sys.executable,'-u','-m','benchmarks.mercari_final','--data-dir',str(data),'--output-dir',str(run),
    '--config',str(config_path),'--pretrained',str(pretrained),'--test-file',test_name,'--sample-file',sample_name,
    '--threads',str(config['threads']),'--chunk-size','3' if SMOKE else '25000'],check=True,env=os.environ)
output='SMOKE_ONLY_DO_NOT_SUBMIT.csv' if SMOKE else 'submission.csv'
shutil.copy2(run/'submission.csv',work/output)
report=dict(smoke=SMOKE,source_archive_sha256=hashlib.sha256(base64.b64decode(SOURCE_ARCHIVE)).hexdigest(),
    pretrained_weight_sha256=weight_hash,seconds=time.perf_counter()-start,
    protocol=json.loads((run/'protocol.json').read_text()),
    stages={p.stem:json.loads(p.read_text()) for p in run.glob('*-report.json')},
    submission=json.loads((run/'submission-report.json').read_text()))
(work/'full-report.json').write_text(json.dumps(report,indent=2))
print('FULL_KAGGLE_REPORT',json.dumps(report),flush=True)
"""


def build(owner, output_dir, config, smoke=False):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("valid Kaggle owner required")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("kernel output must be new or empty")
    payload = io.BytesIO()
    files = sorted(
        [
            *ROOT.glob("LICENSE*"),
            *ROOT.glob("src/*.cc"),
            *ROOT.glob("src/*.h"),
            *ROOT.glob("include/libtree/*.h"),
            *ROOT.glob("python/libtree/*.py"),
            *ROOT.glob("benchmarks/mercari*.py"),
        ]
    )
    # Vendored unmodified BSD-licensed dependency permits an offline Kernel.
    dependency = importlib.metadata.distribution("snowballstemmer")
    stemmer_root = Path(dependency.locate_file("snowballstemmer"))
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:

        def put(name, data):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)

        for path in files:
            put(str(path.relative_to(ROOT)), path.read_bytes())
        for path in sorted(stemmer_root.glob("*.py")):
            put("vendor/snowballstemmer/" + path.name, path.read_bytes())
        put(
            "vendor/snowballstemmer-COPYING",
            Path(
                dependency.locate_file(
                    "snowballstemmer-3.0.1.dist-info/licenses/COPYING"
                )
            ).read_bytes(),
        )
        put("benchmarks/__init__.py", "")
        put(
            "benchmarks/mercari/full-ensemble-config.json", json.dumps(config, indent=2)
        )
    code = (
        "SOURCE_ARCHIVE = "
        + repr(base64.b64encode(payload.getvalue()).decode())
        + "\nSMOKE = "
        + repr(smoke)
        + "\n"
        + BOOTSTRAP
    )
    suffix = "-smoke" if smoke else ""
    metadata = dict(
        id=f"{owner}/libtree-mercari-full-ensemble" + suffix,
        title="LibTree Mercari full ensemble" + suffix,
        code_file="full_ensemble.py",
        language="python",
        kernel_type="script",
        is_private=True,
        enable_gpu=True,
        enable_internet=False,
        dataset_sources=[],
        kernel_sources=[],
        model_sources=[config["pretrained_model"]],
        competition_sources=["mercari-price-suggestion-challenge"],
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "full_ensemble.py").write_text(code)
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
