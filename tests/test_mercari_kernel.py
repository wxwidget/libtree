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
