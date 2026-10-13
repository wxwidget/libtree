"""Behavioral contracts: exported rules must independently reproduce predictions."""
import json

import numpy as np
import pytest
from libtree import C5Classifier


def matches(rule, row):
    for c in rule['conditions']:
        v = row[c['feature_index']]
        if np.isnan(v):
            if not c['accepts_missing']:
                return False
        elif ((c['lower'] is not None and v <= c['lower']) or
              (c['upper'] is not None and v > c['upper'])):
            return False
    return True


@pytest.mark.parametrize('criterion', ['gain_ratio', 'entropy', 'gini'])
def test_multiclass_rules_are_faithful(criterion):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(450, 3)).astype('f4')
    y = np.where(x[:, 0] < -.5, 'low', np.where(x[:, 1] > .2, 'high', 'middle'))
    x[::11, 0] = np.nan
    with C5Classifier(n_estimators=8, max_depth=2, min_samples_leaf=2,
                      criterion=criterion, pruning_alpha=0) as m:
        m.fit(x, y)
        doc = json.loads(m.dump_rules('json', feature_names=['年龄', 'income', 'risk']))
        assert doc['schema_version'] == 1 and len(m.classes_) == 3
        assert len(m.estimator_errors_) > 1
        assert m.score(x, y) > .8
        votes = np.zeros((len(x), 3))
        for i, row in enumerate(x):
            active = [r for r in doc['rules'] if matches(r, row)]
            assert len(active) == doc['estimators']
            for r in active:
                votes[i, r['class_index']] += r['vote']
                assert len({c['feature_index'] for c in r['conditions']}) == len(r['conditions'])
            explanation = m.explain(row)
            np.testing.assert_allclose(explanation['votes'], votes[i], rtol=1e-13)
            assert explanation['prediction'] == int(votes[i].argmax())
            assert [r['rule_id'] for r in active] == [r['rule_id'] for r in explanation['matched_rules']]
        np.testing.assert_array_equal(m.predict(x), m.classes_[votes.argmax(axis=1)])
        np.testing.assert_allclose(m.predict_proba(x).sum(axis=1), 1)
        assert 'IF' in m.dump_rules() and 'THEN' in m.dump_rules()
        assert 'votes' in m.explain(x[:1], format='text')
        assert m.explain(x[0])['predicted_label'] == m.predict(x[:1])[0]


def test_pruning_and_weights_and_failed_refit():
    x = np.arange(100, dtype='f4').reshape(-1, 1)
    y = (x[:, 0].astype(int) % 7 < 3).astype(int)
    with C5Classifier(n_estimators=1, max_depth=8, min_samples_leaf=1,
                      pruning_alpha=0) as full, C5Classifier(
                          n_estimators=1, max_depth=8, min_samples_leaf=1,
                          pruning_alpha=.5) as small:
        full.fit(x, y)
        small.fit(x, y)
        assert len(json.loads(full.dump_rules('json'))['rules']) > 1
        assert len(json.loads(small.dump_rules('json'))['rules']) == 1
        before = small.predict(x)
        with pytest.raises(ValueError):
            small.fit(x, y, sample_weight=np.full(100, np.nan))
        np.testing.assert_array_equal(small.predict(x), before)
        w = np.where(y == 1, 10., 1.)
        small.fit(x, y, sample_weight=w)
        assert np.all(small.predict(x) == 1)
        assert small.get_params()['pruning_alpha'] == .5
        assert small.set_params(max_depth=2) is small
        with pytest.raises(ValueError):
            small.set_params(unknown=1)


def test_parallel_determinism_and_missing():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 20)).astype('f4')
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    x[::4, 0] = np.nan
    docs = []
    for jobs in (1, 4):
        with C5Classifier(n_estimators=4, max_depth=3, n_jobs=jobs) as m:
            m.fit(x, y)
            docs.append((m.dump_rules('json'), m.predict(x), m.predict_proba(x)))
            assert m.predict(np.empty((0, 20))).shape == (0,)
            assert m.predict_proba(np.empty((0, 20))).shape == (0, 2)
    assert docs[0][0] == docs[1][0]
    np.testing.assert_array_equal(docs[0][1], docs[1][1])
    np.testing.assert_array_equal(docs[0][2], docs[1][2])


@pytest.mark.parametrize('params', [
    {'n_estimators': 0}, {'n_estimators': 1.5}, {'n_jobs': 2**40},
    {'criterion': 'invalid'}, {'learning_rate': float('nan')},
    {'pruning_alpha': -1}, {'max_bins': 300},
])
def test_invalid_parameters(params):
    with C5Classifier(**params) as m:
        with pytest.raises(ValueError):
            m.fit([[0], [1]], [0, 1])


@pytest.mark.parametrize('x,y,w', [
    ([[0], [1]], [0], None), ([[0], [1]], [[0], [1]], None),
    (np.empty((0, 1)), [], None), ([[0], [1]], [0, np.nan], None),
    ([[0], [1]], [None, None], None), ([[0], [1]], [0, 1], [1]),
    ([[0], [1]], [0, 1], [[1], [1]]), ([[0], [1]], [0, 1], [0, 0]),
    ([[0], [np.inf]], [0, 1], None),
])
def test_invalid_fit(x, y, w):
    with C5Classifier() as m:
        with pytest.raises(ValueError):
            m.fit(x, y, sample_weight=w)


def test_lifecycle_validation_and_native_error(monkeypatch):
    from libtree import _library
    m = C5Classifier(n_estimators=1, min_samples_leaf=1)
    for operation in (lambda: m.predict([[0]]), lambda: m.estimator_errors_,
                      lambda: m.dump_rules(), lambda: m.explain([0])):
        with pytest.raises(RuntimeError):
            operation()
    m.fit([[0], [1]], ['no', 'yes'])
    for sample in (np.zeros((2, 1)), np.zeros((1, 1, 1))):
        with pytest.raises(ValueError):
            m.explain(sample)
    for x in ([[0, 1]], [[np.inf]]):
        with pytest.raises(ValueError):
            m.predict(x)
    for y in ([], [[0]], [0, 1]):
        with pytest.raises(ValueError):
            m.score([[0]], y)
    for labels in ('age', [1], ['a\0b'], ['a', 'b']):
        with pytest.raises(ValueError):
            m.dump_rules(feature_names=labels)
    with pytest.raises(ValueError):
        m.dump_rules('yaml')
    lib = _library()
    with monkeypatch.context() as patch:
        patch.setattr(lib, 'LtC5Errors', lambda *a: -1)
        with pytest.raises(RuntimeError):
            _ = m.estimator_errors_
    m.close()
    m.close()


def test_constant_and_missing_only_features():
    x = np.array([[3], [3], [np.nan], [np.nan]], dtype='f4')
    with C5Classifier(min_samples_leaf=1) as m:
        m.fit(x, ['no', 'no', 'yes', 'yes'])
        np.testing.assert_array_equal(m.predict(x), ['no', 'no', 'yes', 'yes'])
        assert m.explain(x[2])['observations'][0]['value'] is None
        assert 'MISSING' in m.explain(x[2], format='text')
        m.fit(x, ['one'] * 4)
        np.testing.assert_array_equal(m.predict_proba(x), np.ones((4, 1)))
        assert 'IF TRUE' in m.dump_rules()
        assert m.explain(x[0])['matched_rules'][0]['conditions'] == []
    with C5Classifier(learning_rate=np.nextafter(0., 1.), min_samples_leaf=1) as m:
        m.fit(x, [0, 0, 1, 0])
        assert np.isfinite(m.predict_proba(x)).all()


def test_workers_released_after_refit_and_close():
    from pathlib import Path
    import time

    def workers():
        result = set()
        for task in Path('/proc/self/task').iterdir():
            try:
                if (task / 'comm').read_text().strip() == 'libtree-worker':
                    result.add(task.name)
            except FileNotFoundError:
                pass
        return result

    baseline = workers()
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4000, 20)).astype('f4')
    y = (x[:, 0] > 0).astype(int)
    for _ in range(15):
        with C5Classifier(n_estimators=2, n_jobs=4) as m:
            m.fit(x, y)
            m.fit(x, y)
            before = m.predict(x)
            with pytest.raises(ValueError):
                m.fit(x, y, sample_weight=np.zeros(len(x)))
            np.testing.assert_array_equal(before, m.predict(x))
        deadline = time.monotonic() + 2
        while workers() != baseline and time.monotonic() < deadline:
            time.sleep(.01)
        assert workers() == baseline


def test_command_line(tmp_path, monkeypatch, capsys):
    import runpy
    import sys
    from libtree.cli import main

    train = tmp_path / 'train.csv'
    train.write_text('label,age\n0,1\n0,2\n1,3\n1,4\n')
    common = ['libtree', '--train', str(train), '--header', '--min-leaf', '1',
              '--feature-names', 'age']
    monkeypatch.setattr(sys, 'argv', [*common, 'rules'])
    runpy.run_module('libtree', run_name='__main__')
    assert 'IF' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', [*common, 'rules', '--format', 'json'])
    main()
    assert json.loads(capsys.readouterr().out)['rules']
    for format in ('json', 'text'):
        monkeypatch.setattr(sys, 'argv', [*common, 'explain', '--sample', '3', '--format', format])
        main()
        output = capsys.readouterr().out
        assert ('"predicted_label": 1.0' if format == 'json' else 'Predicted class: 1') in output
    for extra in (['explain'], ['explain', '--sample', 'bad'], ['rules', '--rounds', '0']):
        monkeypatch.setattr(sys, 'argv', [*common, *extra])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
    monkeypatch.setattr(sys, 'argv', ['libtree', 'rules', '--train', str(tmp_path / 'missing')])
    with pytest.raises(SystemExit):
        main()
