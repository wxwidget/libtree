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


@pytest.mark.parametrize('criterion', ['gain_ratio', 'entropy', 'gini', 'hellinger'])
def test_exported_trees_and_paths_reproduce_votes(criterion):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4100, 20)).astype('f4')
    y = np.where(x[:, 0] < -.4, 0, np.where(x[:, 1] > .3, 1, 2))
    if criterion == 'hellinger':
        y = (y == 0).astype(int)
    x[::7, 0] = np.nan
    names = ['年龄"\\\n'] + [f'feature_{i}' for i in range(1, 20)]
    with C5Classifier(n_estimators=5, max_depth=3, pruning_alpha=0,
                      criterion=criterion, n_jobs=4) as model:
        model.fit(x, y)
        before = model.predict_proba(x[:33])
        doc = json.loads(model.dump_model('json', feature_names=names))
        assert doc['schema_version'] == 1
        assert doc['algorithm'] == 'c5_style_samme'
        assert doc['feature_names'] == names
        assert doc['class_labels'] == model.classes_.tolist()
        assert doc['estimators'] == len(doc['trees']) == len(model.estimator_errors_)
        assert json.loads(model.dump_model('json', tree_index=0))['trees'][0]['tree_index'] == 0
        assert len(json.loads(model.dump_model('json', tree_index=0))['trees']) == 1
        text = model.dump_model(feature_names=names)
        assert 'Tree 0' in text and 'node 0' in text and 'missing ->' in text
        for row_index, row in enumerate(x[:33]):
            explanation = model.explain(row, feature_names=names)
            votes = np.zeros(len(model.classes_))
            assert len(explanation['paths']) == len(doc['trees'])
            for tree, path, rule in zip(doc['trees'], explanation['paths'], explanation['matched_rules']):
                nodes = {node['node_id']: node for node in tree['nodes']}
                assert nodes[0]['support'] == len(x)
                node_id, steps = 0, []
                while not nodes[node_id]['is_leaf']:
                    node = nodes[node_id]
                    assert nodes[node['left_child']]['support'] + nodes[node['right_child']]['support'] == node['support']
                    value = row[node['feature_index']]
                    left = node['missing_direction'] == 'left' if np.isnan(value) else value <= node['threshold']
                    direction = 'left' if left else 'right'
                    next_id = node[f'{direction}_child']
                    steps.append((node_id, direction, next_id))
                    node_id = next_id
                leaf = nodes[node_id]
                assert [(s['node_id'], s['direction'], s['next_node_id']) for s in path['steps']] == steps
                for step in path['steps']:
                    assert step['feature_name'] == names[step['feature_index']]
                    value = row[step['feature_index']]
                    assert step['value'] is None if np.isnan(value) else step['value'] == float(value)
                assert path['estimator_index'] == tree['tree_index']
                assert path['leaf_id'] == node_id and path['rule_id'] == leaf['rule_id'] == rule['rule_id']
                assert path['class_index'] == leaf['class_index'] == rule['class_index']
                assert path['vote'] == tree['vote'] == rule['vote']
                votes[leaf['class_index']] += tree['vote']
            np.testing.assert_array_equal(votes, explanation['votes'])
            np.testing.assert_allclose(explanation['probabilities'], before[row_index], rtol=1e-15)
        np.testing.assert_array_equal(model.predict_proba(x[:33]), before)
        assert 'Path for tree 0' in model.explain(x[0], format='text')


def test_tree_export_validation_and_leaf_only_model():
    with C5Classifier(max_depth=0) as model:
        with pytest.raises(RuntimeError):
            model.dump_model()
        model.fit([[np.nan], [0]], ['only', 'only'])
        for index in (-1, 1, 1.5, 2**40, '0'):
            with pytest.raises(ValueError):
                model.dump_model(tree_index=index)
        with pytest.raises(ValueError):
            model.dump_model(format='yaml')
        with pytest.raises(ValueError):
            model.dump_model(feature_names=['a', 'b'])
        tree = json.loads(model.dump_model('json'))['trees'][0]
        assert len(tree['nodes']) == 1 and tree['nodes'][0]['is_leaf']
        path = model.explain([np.nan])['paths'][0]
        assert path['steps'] == [] and path['leaf_id'] == 0
        assert 'THEN class 0' in model.dump_model()


def test_predict_uses_native_class_votes_without_probability_transform(monkeypatch):
    from libtree import _library
    with C5Classifier(min_samples_leaf=1) as model:
        model.fit([[0], [1]], ['no', 'yes'])
        monkeypatch.setattr(_library(), 'LtC5PredictProba', lambda *args: -1)
        np.testing.assert_array_equal(model.predict([[0], [1]]), ['no', 'yes'])


def test_vote_workspace_handles_maximum_classes_and_failed_refit():
    x = np.tile(np.arange(256, dtype='f4')[:, None], (1, 256))
    with C5Classifier(max_depth=0, n_estimators=1, n_jobs=4) as model:
        model.fit(x, np.arange(256))
        probability = model.predict_proba(x[:3])
        assert probability.shape == (3, 256)
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(x[:3]), probability.argmax(axis=1))
        np.testing.assert_allclose(model.explain(x[0])['probabilities'], probability[0])
        with pytest.raises(ValueError, match='class index'):
            model.fit(np.arange(257, dtype='f4')[:, None], np.arange(257))
        np.testing.assert_array_equal(model.predict_proba(x[:3]), probability)


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
            docs.append((m.dump_rules('json'), m.predict(x), m.predict_proba(x), m.dump_model('json')))
            assert m.predict(np.empty((0, 20))).shape == (0,)
            assert m.predict_proba(np.empty((0, 20))).shape == (0, 2)
    assert docs[0][0] == docs[1][0]
    np.testing.assert_array_equal(docs[0][1], docs[1][1])
    np.testing.assert_array_equal(docs[0][2], docs[1][2])
    assert docs[0][3] == docs[1][3]


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
    for operation in (lambda: m.predict([[0]]), lambda: m.predict_proba([[0]]), lambda: m.estimator_errors_,
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
        with pytest.raises(ValueError):
            m.predict_proba(x)
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
        monkeypatch.setattr(sys, 'argv', [*common, 'trees', '--tree', '0', '--format', format])
        main()
        output = capsys.readouterr().out
        assert json.loads(output)['trees'][0]['tree_index'] == 0 if format == 'json' else 'Tree 0' in output
    for format in ('json', 'text'):
        monkeypatch.setattr(sys, 'argv', [*common, 'explain', '--sample', '3', '--format', format])
        main()
        output = capsys.readouterr().out
        assert ('"predicted_label": 1.0' if format == 'json' else 'Predicted class: 1') in output
    for extra in (['explain'], ['explain', '--sample', 'bad'], ['rules', '--rounds', '0'],
                  ['trees', '--tree', '999'], ['rules', '--tree', '0']):
        monkeypatch.setattr(sys, 'argv', [*common, *extra])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
    monkeypatch.setattr(sys, 'argv', ['libtree', 'rules', '--train', str(tmp_path / 'missing')])
    with pytest.raises(SystemExit):
        main()


def test_hellinger_and_explicit_boosting():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(600, 4)).astype('f4')
    y = (x[:, 0] + x[:, 1] > 1.8).astype(int)
    with C5Classifier(criterion='hellinger', n_estimators=12, max_depth=2,
                      pruning_alpha=0, min_samples_leaf=2) as model:
        model.fit(x, y)
        assert model.score(x, y) > .9
        assert len(model.estimator_errors_) > 1
        assert model.explain(x[0])['predicted_label'] == model.predict(x[:1])[0]
        before = model.predict(x)
        with pytest.raises(ValueError, match='binary'):
            model.fit(x, np.arange(len(x)) % 3)
        np.testing.assert_array_equal(before, model.predict(x))
    with C5Classifier(boosting='none', n_estimators=50) as off, C5Classifier(n_estimators=1) as single:
        off.fit(x, y)
        single.fit(x, y)
        assert len(off.estimator_errors_) == 1
        np.testing.assert_array_equal(off.predict_proba(x), single.predict_proba(x))


def test_balanced_class_weights_equal_explicit_weights():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 3)).astype('f4')
    y = (x[:, 0] > 1).astype(int)
    supplied = rng.uniform(.5, 2., len(x))
    weights = supplied * len(y) / (2 * np.bincount(y)[y])
    with C5Classifier(class_weight='balanced') as balanced, C5Classifier() as explicit:
        balanced.fit(x, y, sample_weight=supplied)
        explicit.fit(x, y, sample_weight=weights)
        np.testing.assert_allclose(balanced.predict_proba(x), explicit.predict_proba(x))
        balanced.fit(x, y)
        explicit.fit(x, y, sample_weight=len(y) / (2 * np.bincount(y)[y]))
        np.testing.assert_allclose(balanced.predict_proba(x), explicit.predict_proba(x))
        assert balanced.get_params()['class_weight'] == 'balanced'
    for params in ({'boosting': 'unknown'}, {'class_weight': 'unknown'},
                   {'boosting': 'none', 'n_estimators': 0}):
        with C5Classifier(**params) as model:
            with pytest.raises(ValueError):
                model.fit(x, y)
