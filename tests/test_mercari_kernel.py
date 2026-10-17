"""The Kaggle entry must carry source code, never local models or credentials."""
import ast
import base64
import io
import json
from pathlib import Path
import runpy
import sys
import zipfile

import pytest

from benchmarks import mercari_kernel


def test_kernel_bundle_is_private_self_contained_and_data_free(tmp_path):
    metadata = mercari_kernel.build('wxwidget', tmp_path / 'kernel')
    assert metadata['is_private'] and not metadata['enable_internet']
    assert metadata['competition_sources'] == ['mercari-price-suggestion-challenge']
    script = (tmp_path / 'kernel' / metadata['code_file']).read_text()
    ast.parse(script)
    encoded = script.split('SOURCE_ARCHIVE = ', 1)[1].splitlines()[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(encoded)))) as archive:
        names = archive.namelist()
        assert 'benchmarks/mercari.py' in names
        assert 'python/libtree/__init__.py' in names
        assert 'src/gbdt.cc' in names
        assert all('..' not in Path(name).parts and not Path(name).is_absolute() for name in names)
        assert not any(name.endswith(('.csv', '.tsv', '.pkl', '.env', '.so')) for name in names)
        assert archive.read('benchmarks/mercari.py') == Path('benchmarks/mercari.py').read_bytes()
    assert 'KAGGLE_API_TOKEN' not in script and 'submission.csv' in script
    assert json.loads((tmp_path / 'kernel' / 'kernel-metadata.json').read_text()) == metadata
    with pytest.raises(ValueError):
        mercari_kernel.build('wxwidget', tmp_path / 'kernel')


def test_kernel_cli_can_show_help_without_credentials(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['mercari_kernel', '--help'])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(mercari_kernel.__file__, run_name='__main__')
    assert caught.value.code == 0 and '--owner' in capsys.readouterr().out


def test_kernel_cli_builds_reviewable_files(tmp_path):
    mercari_kernel.main(['--owner', 'wxwidget', '--output-dir', str(tmp_path / 'kernel')])
    assert (tmp_path / 'kernel' / 'mercari_libtree.py').is_file()


def test_optimized_kernel_embeds_only_validated_frozen_parameters(tmp_path):
    from benchmarks.mercari_config import DEFAULT
    config = json.loads(json.dumps(DEFAULT))
    metadata = mercari_kernel.build('wxwidget', tmp_path / 'optimized', optimized_config=config)
    script = (tmp_path / 'optimized' / metadata['code_file']).read_text()
    ast.parse(script)
    encoded = script.split('SOURCE_ARCHIVE = ', 1)[1].splitlines()[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(encoded)))) as archive:
        assert json.loads(archive.read('benchmarks/mercari/optimized-config.json')) == config
        for name in ['mercari_advanced', 'mercari_config', 'mercari_features', 'mercari_solver', 'mercari_fm', 'mercari_neural']:
            assert f'benchmarks/{name}.py' in archive.namelist()
    assert 'OPTIMIZED = True' in script
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    mercari_kernel.main(['--owner', 'wxwidget', '--output-dir', str(tmp_path / 'cli-optimized'),
                         '--optimized-config', str(config_path)])
    for invalid in [dict(config, weights=[0, 0, 0]), {}]:
        with pytest.raises(ValueError):
            mercari_kernel.build('wxwidget', tmp_path / 'invalid', optimized_config=invalid)


def test_bootstrap_finds_competition_in_nested_input_mount(tmp_path, monkeypatch):
    from benchmarks import mercari
    metadata = mercari_kernel.build('wxwidget', tmp_path / 'kernel')
    script = (tmp_path / 'kernel' / metadata['code_file']).read_text()
    work, inputs = tmp_path / 'working', tmp_path / 'input'
    work.mkdir()
    nested = inputs / 'competitions' / 'mercari-price-suggestion-challenge'
    nested.mkdir(parents=True)
    for name in ('train.tsv', 'test.tsv', 'sample_submission.csv'):
        (nested / name).write_text('synthetic test fixture')
    script = script.replace("Path('/kaggle/working')", f'Path({str(work)!r})')
    script = script.replace("Path('/kaggle/input')", f'Path({str(inputs)!r})')
    calls = []
    monkeypatch.setattr('subprocess.run', lambda *args, **kwargs: calls.append(args))

    def fake_main(argv):
        directory = Path(argv[argv.index('--data-dir') + 1])
        assert (directory / 'train.tsv').read_text() == 'synthetic test fixture'
        output = Path(argv[argv.index('--output-dir') + 1])
        output.mkdir()
        (output / 'submission.csv').write_text('test_id,price\n0,1\n')
        (output / 'report.json').write_text('{}')

    monkeypatch.setattr(mercari, 'main', fake_main)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    previous_path = sys.path.copy()
    try:
        exec(compile(script, 'generated_mercari_kernel.py', 'exec'), {})
    finally:
        sys.path[:] = previous_path
    assert calls[0][0][0] == 'g++'
    assert (work / 'submission.csv').read_text() == 'test_id,price\n0,1\n'
    assert not (work / 'mercari-data').exists()
    assert (nested / 'train.tsv').read_text() == 'synthetic test fixture'


def test_optimized_bootstrap_streams_zip_inputs_and_preserves_readonly_archives(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.mercari_config import DEFAULT
    metadata = mercari_kernel.build('wxwidget', tmp_path / 'kernel', optimized_config=DEFAULT)
    script = (tmp_path / 'kernel' / metadata['code_file']).read_text()
    work, inputs = tmp_path / 'work', tmp_path / 'input'
    work.mkdir(); inputs.mkdir()
    (inputs / 'train.tsv').write_text('synthetic training fixture')
    for name, content in [('test_stg2.tsv', 'test_id\tname\n0\titem\n'),
                           ('sample_submission_stg2.csv', 'test_id,price\n0,0\n')]:
        with zipfile.ZipFile(inputs / (name + '.zip'), 'w') as archive:
            archive.writestr(name, content)
    script = script.replace("Path('/kaggle/working')", f'Path({str(work)!r})')
    script = script.replace("Path('/kaggle/input')", f'Path({str(inputs)!r})')
    monkeypatch.setattr('subprocess.run', lambda *args, **kwargs: None)

    def fake_main(argv):
        data = Path(argv[argv.index('--data-dir') + 1])
        assert argv[argv.index('--test-file') + 1] == 'test_stg2.tsv.zip'
        assert argv[argv.index('--sample-file') + 1] == 'sample_submission_stg2.csv.zip'
        assert (data / 'test_stg2.tsv.zip').is_symlink()
        assert not (data / 'test_stg2.tsv').exists()
        assert argv[argv.index('--mode') + 1] == 'final'
        output = Path(argv[argv.index('--output-dir') + 1]); output.mkdir()
        (output / 'submission.csv').write_text('test_id,price\n0,1\n')
        (output / 'report.json').write_text('{}')

    monkeypatch.setitem(sys.modules, 'benchmarks.mercari_advanced', SimpleNamespace(main=fake_main))
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(variable, '1')
    previous = sys.path.copy()
    try:
        exec(compile(script, 'optimized_kernel.py', 'exec'), {})
    finally:
        sys.path[:] = previous
    assert not (work / 'mercari-data').exists()
    assert (inputs / 'test_stg2.tsv.zip').is_file()


def test_stage_one_budget_check_cannot_replace_the_competition_entry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks.mercari_config import DEFAULT
    metadata = mercari_kernel.build('wxwidget', tmp_path / 'kernel', optimized_config=DEFAULT, stage='one')
    assert metadata['id'] == 'wxwidget/libtree-mercari-stage-one-budget'
    script = (tmp_path / 'kernel' / metadata['code_file']).read_text()
    work, inputs = tmp_path / 'work', tmp_path / 'input'
    work.mkdir(); inputs.mkdir()
    for name in ('train.tsv', 'test.tsv', 'sample_submission.csv', 'test_stg2.tsv', 'sample_submission_stg2.csv'):
        (inputs / name).write_text('synthetic input')
    script = script.replace("Path('/kaggle/working')", f'Path({str(work)!r})')
    script = script.replace("Path('/kaggle/input')", f'Path({str(inputs)!r})')
    monkeypatch.setattr('subprocess.run', lambda *args, **kwargs: None)
    def fake_main(argv):
        assert argv[argv.index('--test-file') + 1] == 'test.tsv'
        assert argv[argv.index('--sample-file') + 1] == 'sample_submission.csv'
        assert '--omit-model-artifact' in argv
        output = Path(argv[argv.index('--output-dir') + 1]); output.mkdir()
        (output / 'submission.csv').write_text('test_id,price\n0,1\n')
        (output / 'report.json').write_text('{}')
    monkeypatch.setitem(sys.modules, 'benchmarks.mercari_advanced', SimpleNamespace(main=fake_main))
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(variable, '1')
    previous = sys.path.copy()
    try:
        exec(compile(script, 'stage_one_kernel.py', 'exec'), {})
    finally:
        sys.path[:] = previous
    assert (work / 'stage-one-predictions.csv').is_file()
    assert (work / 'STAGE_ONE_BUDGET_CHECK_NEVER_SUBMIT.txt').is_file()
    assert not (work / 'submission.csv').exists()
    assert (inputs / 'test_stg2.tsv').is_file()
    mercari_kernel.main(['--owner', 'wxwidget', '--output-dir', str(tmp_path / 'cli-one'), '--stage', 'one'])
    assert json.loads((tmp_path / 'cli-one/kernel-metadata.json').read_text())['id'] == metadata['id']


def test_explicit_stage_two_and_deterministic_source_archive(tmp_path):
    from benchmarks.mercari_config import DEFAULT
    with pytest.raises(ValueError):
        mercari_kernel.build('wxwidget', tmp_path / 'invalid', stage='three')
    scripts = []
    for name in ('first', 'second'):
        folder = tmp_path / name
        metadata = mercari_kernel.build('wxwidget', folder, optimized_config=DEFAULT, stage='two')
        scripts.append((folder / metadata['code_file']).read_text())
    assert scripts[0] == scripts[1] and 'stage2 = True' in scripts[0]
    encoded = scripts[0].split('SOURCE_ARCHIVE = ', 1)[1].splitlines()[0]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(ast.literal_eval(encoded)))) as archive:
        assert all(info.date_time == (2026, 1, 1, 0, 0, 0) for info in archive.infolist())
