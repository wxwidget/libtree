"""Run the prepared Mercari study on one Colab VM, then release that VM.

Uses the official google-colab-cli, already authenticated by the caller. It
never transfers credentials or confirmation labels. Large inputs use bounded
upload chunks; all results have a manifest and diagnostic logs.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tarfile
import uuid

import pandas as pd

from benchmarks.mercari import ROOT, sha256


def remote_script(include_models, profile='all', transformer_epochs=2):
    suffixes = ['.json', '.npz', '.log'] + (['.pkl', '.keras', '.pt'] if include_models else [])
    packages = {'transformer': ['transformers==4.57.1'], 'all': ['sentence-transformers==5.2.0', 'lightgbm==4.7.0'],
                'stemmed': ['snowballstemmer==3.0.1'], 'sparse': []}[profile]
    jobs = {
        'all': [('gpu', ['--neural-only', 'sequence']), ('vectors', [])],
        'sparse': [
            ('separate', ['--neural-only', 'sparse', '--sparse-view', 'separate', '--sparse-width', '256']),
            ('joint', ['--neural-only', 'sparse', '--sparse-view', 'joint', '--sparse-width', '512'])],
        'transformer': [('finetuned', [])],
        'stemmed': [('stemmed', ['--neural-only', 'sparse', '--sparse-view', 'stemmed', '--sparse-width', '512'])],
    }[profile]
    return '''import hashlib,json,os,pathlib,subprocess,sys,tarfile,time,zipfile
root=pathlib.Path('/content/libtree-frontier');root.mkdir(exist_ok=True)
with tarfile.open('/content/frontier-source.tar.gz') as archive:archive.extractall(root,filter='data')
os.chdir(root)
manifest=json.loads(pathlib.Path('/content/frontier-transfer.json').read_text())
data=root/'data';data.mkdir(exist_ok=True)
for name,item in manifest.items():
    destination=data/name
    with destination.open('wb') as target:
        for part in item['parts']:target.write(pathlib.Path('/content',part).read_bytes())
    digest=hashlib.sha256()
    with destination.open('rb') as source:
        while chunk:=source.read(1048576):digest.update(chunk)
    assert digest.hexdigest()==item['sha256'], 'transferred input hash mismatch'
subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','.'],check=True)
if PACKAGES:subprocess.run([sys.executable,'-m','pip','install','--quiet',*PACKAGES],check=True)
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',HF_HUB_DISABLE_XET='1',USE_TF='0',TOKENIZERS_PARALLELISM='TOKENIZER_MODE',RAYON_NUM_THREADS='4')
results=root/'results';results.mkdir(exist_ok=True)
status={}
try:
    for mode,extra in JOBS:
        start=time.perf_counter()
        with (results/(mode+'.log')).open('w') as log:
            try:
                task=([sys.executable,'-u','-m','benchmarks.mercari_transformer'] if mode=='finetuned' else
                      [sys.executable,'-u','-m','benchmarks.mercari_frontier','vectors' if mode=='vectors' else 'gpu'])
                options=(['--epochs',TRANSFORMER_EPOCHS,'--batch-size','512'] if mode=='finetuned' else ['--max-features','150000'])
                run=subprocess.run([*task,'--data-dir',str(data),'--output-dir',str(results/mode),'--threads','4',*options,*extra],
                    stdout=log,stderr=subprocess.STDOUT,timeout=5400)
                code=run.returncode
            except subprocess.TimeoutExpired:code=124
        status[mode]={'exit_code':code,'seconds':time.perf_counter()-start}
        (results/'status.json').write_text(json.dumps(status,indent=2))
        print('EXPERIMENT_STATUS',mode,status[mode],flush=True)
finally:
    with zipfile.ZipFile('/content/libtree-frontier-results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in results.rglob('*'):
            if path.is_file() and path.suffix in SUFFIXES:
                archive.write(path,path.relative_to(results))
    print('RESULTS_READY',json.dumps(status),flush=True)
if any(value['exit_code'] for value in status.values()):raise SystemExit(1)
'''.replace('SUFFIXES', repr(suffixes)).replace('JOBS', repr(jobs)).replace('PACKAGES', repr(packages)).replace('TRANSFORMER_EPOCHS', repr(str(transformer_epochs))).replace('TOKENIZER_MODE', 'true' if profile == 'transformer' else 'false')


def run(args):
    if args.output_dir.exists():
        raise ValueError('use a new output directory')
    if args.transformer_epochs < 1:
        raise ValueError('positive transformer epochs required')
    names = [name + '.tsv.gz' for name in ('train', 'calibration', 'selection', 'confirmation')]
    for name in names:
        if not (args.data_dir / name).is_file():
            raise ValueError('prepare every grouped window before allocating a VM')
    if 'price' in pd.read_csv(args.data_dir / names[-1], sep='\t', nrows=0).columns:
        raise ValueError('confirmation labels must stay local')
    args.output_dir.mkdir(parents=True)
    transfer = args.output_dir / 'transfer'; transfer.mkdir()
    session = 'libtree-mercari-' + uuid.uuid4().hex[:12]
    manifest = dict(session=session, created_utc=datetime.now(timezone.utc).isoformat(),
                    gpu=args.gpu, high_mem=args.high_mem, profile=args.profile,
                    transformer_epochs=args.transformer_epochs, allocation_confirmed=False, cleanup_verified=False)
    def save():
        (args.output_dir / 'job.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def call(arguments, name, timeout=300):
        with (args.output_dir / (name + '.log')).open('w') as log:
            completed = subprocess.run([args.cli, *arguments], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        return completed.returncode
    def checked(arguments, name, timeout=300):
        code = call(arguments, name, timeout)
        if code:
            raise RuntimeError(name + ' failed; inspect its log')
    save()
    script = transfer / 'remote.py'; script.write_text(remote_script(args.include_models, args.profile, args.transformer_epochs))
    bundle = transfer / 'source.tar.gz'
    files = [ROOT / name for name in ('setup.py', 'pyproject.toml', 'README.md')]
    files += list(ROOT.glob('LICENSE*'))
    for folder in ('src', 'include', 'python/libtree'):
        files += [p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.cc', '.h')]
    files += list((ROOT / 'benchmarks').glob('mercari*.py'))
    with tarfile.open(bundle, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    manifest['source_sha256'] = sha256(bundle)
    save()
    try:
        new = ['new', '-s', session, '--gpu', args.gpu] + (['--high-mem'] if args.high_mem else [])
        checked(new, 'allocation')
        manifest['allocation_confirmed'] = True
        save()
        checked(['upload', '-s', session, str(bundle), '/content/frontier-source.tar.gz'], 'upload-source')
        inputs = {}
        for name in names:
            path, parts = args.data_dir / name, []
            with path.open('rb') as source:
                while chunk := source.read(12 * 1024 * 1024):
                    part = transfer / (name + f'.part{len(parts):03d}')
                    part.write_bytes(chunk); parts.append(part.name)
                    checked(['upload', '-s', session, str(part), '/content/' + part.name], 'upload-' + part.name)
            inputs[name] = dict(parts=parts, sha256=sha256(path))
        path = transfer / 'inputs.json'; path.write_text(json.dumps(inputs))
        checked(['upload', '-s', session, str(path), '/content/frontier-transfer.json'], 'upload-manifest')
        manifest['execution_exit_code'] = call(['exec', '-s', session, '--timeout', '11100', '-f', str(script)],
                                               'execution', timeout=11400)
        # A failed model can still produce valuable prediction/log artifacts.
        checked(['download', '-s', session, '/content/libtree-frontier-results.zip',
                 str(args.output_dir / 'results.zip')], 'download', timeout=600)
        manifest['artifact_downloaded'] = True
    except BaseException as error:
        manifest['error_type'] = type(error).__name__
        raise
    finally:
        try:
            manifest['cleanup_exit_code'] = call(['stop', '-s', session], 'cleanup', timeout=180)
            manifest['cleanup_verified'] = manifest['cleanup_exit_code'] == 0
        finally:
            save()
    if manifest['execution_exit_code'] or not manifest['cleanup_verified']:
        raise RuntimeError('experiment or cleanup failed; inspect job.json and logs')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Prepared frontier windows')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cli', default='colab', help='Official CLI executable or writable-state launcher')
    parser.add_argument('--gpu', choices=['T4', 'A100'], default='T4')
    parser.add_argument('--high-mem', action='store_true', help='Requires account entitlement')
    parser.add_argument('--profile', choices=['all', 'sparse', 'transformer', 'stemmed'], default='all')
    parser.add_argument('--transformer-epochs', type=int, default=2)
    parser.add_argument('--include-models', action='store_true', help='Download large model artifacts too')
    print(json.dumps(run(parser.parse_args(argv))), flush=True)


if __name__ == '__main__':
    main()
