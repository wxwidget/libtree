"""Colab lifecycle boundary tests: transfers, diagnostics and own-VM cleanup."""
from argparse import Namespace
import ast
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks import mercari_colab as remote
from test_mercari import listings


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data = tmp_path / 'data'; data.mkdir()
    for name in ('train', 'calibration', 'selection', 'confirmation'):
        frame = listings(4)
        if name == 'confirmation':
            frame = frame.drop(columns='price')
        frame.to_csv(data / (name + '.tsv.gz'), sep='\t', index=False)
    repo = tmp_path / 'repo'; repo.mkdir()
    for name in ('setup.py', 'pyproject.toml', 'README.md', 'LICENSE'):
        (repo / name).write_text('test source')
    for folder in ('src', 'include', 'python/libtree', 'benchmarks'):
        (repo / folder).mkdir(parents=True, exist_ok=True)
        (repo / folder / 'mercari_test.py').write_text('pass')
    monkeypatch.setattr(remote, 'ROOT', repo)
    return Namespace(data_dir=data, output_dir=tmp_path / 'job', cli='official-test-colab',
                     gpu='T4', high_mem=True, include_models=False, profile='all', transformer_epochs=2)


def fake_cli(monkeypatch, failure=None, exception=None):
    calls = []
    def execute(args, **kwargs):
        calls.append(args)
        assert kwargs['timeout'] > 0
        if exception and args[1] == exception:
            raise subprocess.TimeoutExpired(args, kwargs['timeout'])
        return SimpleNamespace(returncode=int(args[1] == failure))
    monkeypatch.setattr(remote.subprocess, 'run', execute)
    return calls


def test_job_uploads_only_feature_windows_and_stops_its_own_session(prepared, monkeypatch):
    calls = fake_cli(monkeypatch)
    report = remote.run(prepared)
    assert report['cleanup_verified'] and report['artifact_downloaded']
    assert calls[-1][1:] == ['stop', '-s', report['session']]
    assert all('evaluation.npz' not in ' '.join(call) for call in calls)
    assert '--high-mem' in calls[0]
    transfers = json.loads((prepared.output_dir / 'transfer/inputs.json').read_text())
    assert len(transfers) == 4 and all(value['sha256'] for value in transfers.values())
    assert json.loads((prepared.output_dir / 'job.json').read_text()) == report
    with pytest.raises(ValueError, match='new output'):
        remote.run(prepared)


def test_real_repository_bundle_contains_source_and_excludes_private_artifacts(prepared, monkeypatch):
    monkeypatch.setattr(remote, 'ROOT', Path(__file__).resolve().parents[1])
    fake_cli(monkeypatch)
    remote.run(prepared)
    with tarfile.open(prepared.output_dir / 'transfer/source.tar.gz') as archive:
        names = archive.getnames()
    assert 'src/gbdt.cc' in names and 'benchmarks/mercari_transformer.py' in names
    assert not any(name.startswith('benchmarks/runs/') or 'token.json' in name for name in names)


def test_required_files_and_confirmation_labels_fail_before_allocation(prepared, monkeypatch):
    calls = fake_cli(monkeypatch)
    prepared.transformer_epochs = 0
    with pytest.raises(ValueError, match='positive transformer'):
        remote.run(prepared)
    prepared.transformer_epochs = 2
    path = prepared.data_dir / 'confirmation.tsv.gz'
    path.unlink()
    with pytest.raises(ValueError, match='every grouped window'):
        remote.run(prepared)
    listings(4).to_csv(path, sep='\t', index=False)
    with pytest.raises(ValueError, match='labels must stay local'):
        remote.run(prepared)
    assert not calls and not prepared.output_dir.exists()


@pytest.mark.parametrize('failure', ['new', 'upload', 'exec', 'download', 'stop'])
def test_every_stage_failure_preserves_cleanup_and_manifest(prepared, monkeypatch, failure):
    prepared.high_mem = False
    calls = fake_cli(monkeypatch, failure=failure)
    with pytest.raises(RuntimeError):
        remote.run(prepared)
    report = json.loads((prepared.output_dir / 'job.json').read_text())
    assert calls[-1][1] == 'stop'
    assert report['cleanup_verified'] == (failure != 'stop')
    if failure == 'exec':
        assert report['artifact_downloaded']


def test_transport_timeout_still_releases_session(prepared, monkeypatch):
    calls = fake_cli(monkeypatch, exception='upload')
    with pytest.raises(subprocess.TimeoutExpired):
        remote.run(prepared)
    report = json.loads((prepared.output_dir / 'job.json').read_text())
    assert report['error_type'] == 'TimeoutExpired' and report['cleanup_verified']
    assert calls[-1][1] == 'stop'


def test_script_is_valid_and_model_transfer_is_explicit():
    for include in (False, True):
        for profile in ('all', 'sparse', 'transformer', 'stemmed'):
            source = remote.remote_script(include, profile, transformer_epochs=4)
            compile(source, 'remote.py', 'exec')
            assert ("'.keras'" in source) == include
            assert 'evaluation.npz' not in source
            loops = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Tuple)
                     and [value.id for value in node.target.elts] == ['mode', 'extra']]
            jobs = ast.literal_eval(loops[0].iter)
            assert [name for name, _ in jobs] == {'all': ['gpu', 'vectors'],
                'sparse': ['separate', 'joint'], 'transformer': ['finetuned'], 'stemmed': ['stemmed']}[profile]
            options = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'options' for target in node.targets))
            assert ast.literal_eval(options.value.body)[:2] == ['--epochs', '4']
            if profile == 'sparse':
                assert jobs[1][1][-2:] == ['--sparse-width', '512']


def test_main_and_module_help(prepared, monkeypatch, capsys):
    monkeypatch.setattr(remote, 'run', lambda args: dict(gpu=args.gpu))
    remote.main(['--data-dir', str(prepared.data_dir), '--output-dir', str(prepared.output_dir)])
    assert json.loads(capsys.readouterr().out)['gpu'] == 'T4'
    monkeypatch.setattr(sys, 'argv', ['mercari_colab', '--help'])
    with pytest.raises(SystemExit) as error:
        runpy.run_module('benchmarks.mercari_colab', run_name='__main__')
    assert error.value.code == 0


def test_smalltext_profile_is_bounded_and_preserves_gpu_artifacts():
    source = remote.remote_script(True, 'smalltext')
    compile(source, 'smalltext-remote.py', 'exec')
    assert 'benchmarks.mercari_smalltext' in source
    assert '3600' in source and '8000000' in source
    assert 'bag_mse' in source and 'bag_huber' in source and 'bert_tiny' in source
    assert "'.pt'" in source and 'evaluation.npz' not in source
