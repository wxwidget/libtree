"""Behavioral guards for leakage-safe benchmark preparation and cache integrity."""
from argparse import Namespace
import importlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest
from benchmarks import run
from benchmarks import c5


def test_memory_counter_uses_current_address_space():
    assert run.peak_rss_kib() > 0


@pytest.mark.parametrize('quota,error,expected', [
    ('400000 100000\n', None, '400000 100000'),
    ('max 100000\n', None, 'max 100000'),
    ('', None, None),
    (None, FileNotFoundError, None),
    (None, PermissionError, None),
])
def test_report_survives_unavailable_cpu_quota(tmp_path, monkeypatch,
                                             quota, error, expected):
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path('/sys/fs/cgroup/cpu.max'):
            if error:
                raise error('cgroup CPU quota unavailable')
            return quota
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read_text)
    monkeypatch.setattr(run, 'datasets', lambda suite: iter(()))
    monkeypatch.setattr(run, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(run.subprocess, 'check_output',
                        lambda *args, **kwargs: 'test compiler\n')
    output = tmp_path / 'report.json'
    args = Namespace(engines=['libtree'], trees=2, depth=2, bins=8,
                     min_leaf=1, rate=.1, l2=1., min_gain=0., threads=1,
                     suite='builtin', datasets=None, seed=42, repeats=1,
                     output=output)
    run.main(args)
    report = json.loads(output.read_text())
    assert report['metadata']['cpu_quota'] == expected
    assert report['results'] == []


def test_numeric_only_preparation_and_seed():
    x = pd.DataFrame({"value": np.arange(80), "missing": [np.nan] * 80})
    y = np.arange(80, dtype=np.float32)
    first = run.prepare(x, y, False, seed=42)
    second = run.prepare(x, y, False, seed=42)
    other = run.prepare(x, y, False, seed=2024)
    assert first[0].shape == (60, 2)
    assert first[1].shape == (20, 2)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[2], other[2])
    assert np.isnan(first[0]).sum() == 60


def test_test_only_category_is_not_learned():
    x = pd.DataFrame({"category": ["common"] * 40})
    y = np.arange(40)
    _, _, _, test_y = run.prepare(x, y, False, seed=42)
    x.loc[test_y[0], "category"] = "test-only"
    train, test, _, _ = run.prepare(x, y, False, seed=42)
    assert train.shape[1] == test.shape[1] == 1
    assert (test == 0).any()


def test_corrupt_cached_dataset_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "CACHE", tmp_path)
    (tmp_path / "pima.csv").write_text("corrupted data")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        run.source("pima")


def test_measurement_order_is_reproducible_but_varies_by_thread_and_seed():
    engines = ("libtree", "xgboost", "lightgbm")
    first = run.measurement_order("friedman_100k", 42, 1, engines)
    assert first == run.measurement_order("friedman_100k", 42, 1, engines)
    assert set(first) == {(language, engine)
                          for language in ("cpp", "python")
                          for engine in engines}
    assert first != run.measurement_order("friedman_100k", 42, 4, engines)
    assert first != run.measurement_order("friedman_100k", 2024, 1, engines)


def test_c5_summary_uses_means_and_preserves_outliers():
    rows = [dict(dataset='tiny', variant='c5', interface='python', threads=1,
                 fit_seconds=v, predict_seconds=v / 2, accuracy=.8)
            for v in (1, 1, 10)]
    summary = c5.summarize(rows)[0]
    assert summary['fit_seconds']['mean'] == 4
    assert summary['fit_seconds']['max'] == 10
    assert summary['fit_seconds']['min'] == 1
    assert summary['fit_seconds']['std'] > 4


def test_c5_stratifies_multiclass_and_rejects_invalid_probabilities():
    x = np.arange(180).reshape(90, 2)
    y = np.repeat([4, 8, 12], 30)
    train, test, labels, truth = c5.prepare(x, y, 42)
    assert train.shape[0] + test.shape[0] == 90
    assert set(labels) == set(truth) == {0, 1, 2}
    assert np.bincount(truth).max() - np.bincount(truth).min() <= 1
    for probability in (np.array([[np.nan, 0]]), np.array([[.1, .1]])):
        with pytest.raises(ValueError):
            c5.metrics([0], probability)
    assert c5.metrics([0, 1], [[.8, .2], [.2, .8]])['accuracy'] == 1


def test_risk_ks_is_tie_safe_and_has_clear_orientation():
    from benchmarks.risk import ks_statistic
    assert ks_statistic([0, 0, 1, 1], [.1, .2, .8, .9]) == 1
    assert ks_statistic([0, 1, 0, 1], [.5, .5, .5, .5]) == 0
    assert ks_statistic([0, 0, 1, 1], [.9, .8, .2, .1]) == 1
    assert ks_statistic([0, 0, 1, 1], [.9, .8, .2, .1], directional=True) == 0
    assert ks_statistic([0, 1, 0, 1], [.1, .8, .2, .2]) == .5
    for y, score in (([1, 1], [.1, .2]), ([0, 1], [np.nan, .2]),
                     ([0, 1], [.1]), ([0, 2], [.1, .2])):
        with pytest.raises(ValueError):
            ks_statistic(y, score)


def test_risk_encoder_and_selector_are_train_only(monkeypatch):
    from benchmarks import risk
    train = pd.DataFrame({'category': ['known'] * 30, 'value': np.arange(30)})
    test = pd.DataFrame({'category': ['test-only'], 'value': [100]})
    a, b = risk.encode(train, test)
    assert a.shape == (30, 2) and b.shape == (1, 2)
    assert b[0, 1] == 0
    frame = pd.DataFrame({'value': np.arange(60)}, index=np.arange(100, 160))
    y = (np.arange(60) % 3 == 0).astype(int)
    seen = []
    original = risk.encode

    def spy(a, b):
        assert set(a.index).isdisjoint(b.index)
        assert set(a.index) | set(b.index) == set(frame.index)
        seen.append((len(a), len(b)))
        return original(a, b)

    monkeypatch.setattr(risk, 'encode', spy)
    parameters = {'n_estimators': 2, 'min_samples_leaf': 1}
    selected, report = risk.select_configuration(frame, y,
        {'samme_30': parameters, 'identical_candidate': parameters}, seed=42, folds=3)
    assert selected == 'samme_30'
    assert len(seen) == 3 and len(report['candidates']) == 2
    assert risk.metrics([0, 1], [[.9, .1], [.1, .9]])['ks'] == 1


def test_risk_data_corruption_is_rejected(tmp_path, monkeypatch):
    from benchmarks import risk
    monkeypatch.setattr(run, 'CACHE', tmp_path)
    (tmp_path / 'german_credit.csv').write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        risk.dataset('german_credit')


@pytest.mark.parametrize('study', ['c5', 'risk', 'c5_performance'])
def test_report_languages_preserve_measurements_and_commands(study):
    root = Path(__file__).resolve().parents[1] / 'benchmarks'
    data = json.loads((root / f'{study.replace("_", "-")}-results.json').read_text(encoding='utf-8'))
    render = importlib.import_module(f'benchmarks.{study}_report').render
    english, chinese = render(data), render(data, chinese=True)
    in_code = False
    for line in chinese.splitlines():
        if line.startswith('```'):
            in_code = not in_code
        elif line.strip() and not in_code and not re.fullmatch(r'[| :\-]+', line):
            assert re.search(r'[\u4e00-\u9fff]', line), f'Untranslated Chinese report: {line}'
    for line in english.splitlines():
        if line.startswith('#'):
            assert not re.search(r'[\u4e00-\u9fff]', line), line

    def numeric_cells(report):
        return [cell.strip() for line in report.splitlines() if line.startswith('|')
                for cell in line.strip('|').split('|')
                if re.fullmatch(r'\s*[\d.,%±×+–—/\s]+\s*', cell)]

    assert numeric_cells(chinese) == numeric_cells(english)
    assert re.findall(r'```sh\n(.*?)```', chinese, re.S) == re.findall(
        r'```sh\n(.*?)```', english, re.S)
    assert data['metadata']['code_sha256'] in chinese
    assert f'../docs/{"c5-risk" if study == "risk" else "c5"}.zh-CN.md' in chinese
    assert (root / f'{study.upper()}_REPORT.zh-CN.md').read_text(encoding='utf-8') == chinese
    assert (root / f'{study.upper()}_REPORT.txt').read_text(encoding='utf-8') == chinese
    assert (root / f'{study.upper()}_REPORT.md').read_text(encoding='utf-8') == english


def test_chinese_report_prose_is_chinese():
    root = Path(__file__).resolve().parents[1] / 'benchmarks'
    for path in root.glob('*.zh-CN.md'):
        in_code = False
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('```'):
                in_code = not in_code
            elif line.strip() and not in_code and not line.startswith('|'):
                assert re.search(r'[\u4e00-\u9fff]', line), f'{path.name}: {line}'


def test_c5_performance_requires_exact_complete_before_after_pairs():
    from benchmarks.c5_performance import validate, summarize
    rows = [dict(dataset='tiny', seed=42, threads=1, interface='python', repeat=i,
                 variant=variant, probability_sha256='probability',
                 train_probability_sha256='training', rules_sha256='rules', fit_seconds=timing)
            for i, timing in enumerate((1., 1., 10.)) for variant in ('before', 'after')]
    validate(rows)
    stats = summarize(rows)[0]['fit_seconds']
    assert stats['mean'] == 4 and stats['min'] == 1 and stats['max'] == 10
    with pytest.raises(ValueError, match='incomplete'):
        validate(rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        validate(rows + rows[:1])
    for field in ('probability_sha256', 'train_probability_sha256', 'rules_sha256'):
        changed = rows[:-1] + [rows[-1] | {field: 'regression'}]
        with pytest.raises(ValueError, match='output changed'):
            validate(changed)
