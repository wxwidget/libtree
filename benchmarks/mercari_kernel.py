"""Build a private, source-only Kaggle Kernel for the Mercari LibTree entry.

This creates reviewable local files. Upload explicitly with kaggle kernels push.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile
from benchmarks.mercari_config import validate

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = '''
import base64
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
work = Path('/kaggle/working')
project = work / 'libtree-source'
project.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
    archive.extractall(project)
sources = sorted(str(p) for p in (project / 'src').glob('*.cc') if p.name != 'cli.cc')
subprocess.run(['g++', '-std=c++17', '-O3', '-DNDEBUG', '-shared', '-fPIC', '-pthread',
                '-I' + str(project / 'include'), *sources,
                '-o', str(project / 'python/libtree/_native.so')], check=True)
sys.path.insert(0, str(project / 'python'))
sys.path.insert(0, str(project))

inputs = Path('/kaggle/input')
print('Competition files:', [str(p.relative_to(inputs)) for p in inputs.rglob('*') if p.is_file()], flush=True)
data = work / 'mercari-data'
data.mkdir(exist_ok=True)
stage2 = any(inputs.rglob('test_stg2.tsv*'))
test_name = 'test_stg2.tsv' if stage2 else 'test.tsv'
sample_name = 'sample_submission_stg2.csv' if stage2 else 'sample_submission.csv'
for name in ('train.tsv', test_name, sample_name):
    raw = list(inputs.rglob(name))
    if raw:
        (data / name).symlink_to(raw[0])
        continue
    zipped = list(inputs.rglob(name + '.zip'))
    if zipped:
        if OPTIMIZED:
            (data / (name + '.zip')).symlink_to(zipped[0])
            if name == test_name:
                test_name += '.zip'
            if name == sample_name:
                sample_name += '.zip'
            continue
        with zipfile.ZipFile(zipped[0]) as archive:
            archive.extract(name, data)
        continue
    seven = list(inputs.rglob(name + '.7z'))
    if not seven:
        raise FileNotFoundError('Competition input not found: ' + name)
    subprocess.run(['7z', 'x', str(seven[0]), '-o' + str(data), '-y'], check=True)

if OPTIMIZED:
    from benchmarks.mercari_advanced import main
    main(['--data-dir', str(data), '--output-dir', str(work / 'entry-run'),
          '--test-file', test_name, '--sample-file', sample_name, '--mode', 'final',
          '--omit-model-artifact',
          '--config', str(project / 'benchmarks/mercari/optimized-config.json')])
else:
    from benchmarks.mercari import main
    main(['--data-dir', str(data), '--output-dir', str(work / 'entry-run'),
          '--test-file', test_name, '--sample-file', sample_name,
          '--trees', '400', '--depth', '6', '--max-features', '150000',
          '--folds', '3', '--repeats', '1', '--threads', '4'])
shutil.rmtree(data)
shutil.copy2(work / 'entry-run/submission.csv', work / 'submission.csv')
shutil.copy2(work / 'entry-run/report.json', work / 'report.json')
'''


def build(owner, output_dir, optimized_config=None, stage='auto'):
    if stage not in ('auto', 'one', 'two'):
        raise ValueError('stage must be auto, one or two')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('kernel output directory must be empty')
    if optimized_config is not None:
        validate(optimized_config)
    payload = io.BytesIO()
    paths = sorted([*ROOT.glob('LICENSE*'), *ROOT.glob('src/*.cc'), *ROOT.glob('src/*.h'),
                    *ROOT.glob('include/libtree/*.h'), *ROOT.glob('python/libtree/*.py'),
                    *[ROOT / f'benchmarks/{name}.py' for name in
                      ('mercari', 'mercari_config', 'mercari_advanced', 'mercari_features',
                       'mercari_solver', 'mercari_fm', 'mercari_neural', 'mercari_sparse')]])
    with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        def store(name, data):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
        for path in paths:
            store(str(path.relative_to(ROOT)), path.read_bytes())
        store('benchmarks/__init__.py', '')
        if optimized_config is not None:
            store('benchmarks/mercari/optimized-config.json', json.dumps(optimized_config, indent=2))
    code = '# Source-only LibTree build; no models or competition records embedded.\n'
    code += 'SOURCE_ARCHIVE = ' + repr(base64.b64encode(payload.getvalue()).decode()) + '\n'
    code += 'OPTIMIZED = ' + repr(optimized_config is not None) + '\n'
    code += BOOTSTRAP
    if stage != 'auto':
        code = code.replace("stage2 = any(inputs.rglob('test_stg2.tsv*'))", f"stage2 = {stage == 'two'}")
    metadata = dict(id=f'{owner}/libtree-mercari-reproducible-entry',
        title='LibTree Mercari reproducible entry', code_file='mercari_libtree.py',
        language='python', kernel_type='script', is_private=True,
        enable_gpu=False, enable_internet=False,
        dataset_sources=[], kernel_sources=[],
        competition_sources=['mercari-price-suggestion-challenge'])
    if stage == 'one':
        metadata.update(id=f'{owner}/libtree-mercari-stage-one-budget', title='LibTree Mercari stage one budget')
        code = code.replace("work / 'submission.csv'", "work / 'stage-one-predictions.csv'")
        code += "\n(work / 'STAGE_ONE_BUDGET_CHECK_NEVER_SUBMIT.txt').write_text('Stage-one budget check completed; never submit')\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / metadata['code_file']).write_text(code)
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True, help='your Kaggle username')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--optimized-config', type=Path, help='frozen ensemble settings; uses final-only training')
    parser.add_argument('--stage', choices=('auto', 'one', 'two'), default='auto',
                        help='one creates a separate timing-check Kernel with a non-submission output')
    args = parser.parse_args(argv)
    config = None if args.optimized_config is None else json.loads(args.optimized_config.read_text())
    metadata = build(args.owner, args.output_dir, optimized_config=config, stage=args.stage)
    print(json.dumps(dict(kernel=metadata['id'], output=str(args.output_dir))))


if __name__ == '__main__':
    main()
