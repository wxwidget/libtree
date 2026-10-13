"""Tree exports and paths must independently reconstruct actual predictions."""
import json

import numpy as np
import pytest

from libtree import GBDTClassifier, GBDTRegressor


def traverse(dump, row):
    margin = dump['base_score']
    for tree in dump['trees']:
        nodes = tree['nodes']
        node = nodes[0]
        while 'feature_index' in node:
            value = row[node['feature_index']]
            left = (node['missing_direction'] == 'left' if np.isnan(value)
                    else value <= node['threshold'])
            node = nodes[node['left_id'] if left else node['right_id']]
        margin += dump['learning_rate'] * node['leaf_value']
    if dump['objective'] == 'binary_logistic':
        margin = 1 / (1 + np.exp(-margin))
    return np.float32(margin)


@pytest.mark.parametrize('cls', [GBDTRegressor, GBDTClassifier])
@pytest.mark.parametrize('missing_high', [False, True])
def test_dump_and_paths_reconstruct_predictions(cls, missing_high):
    x = np.array([[0, 1], [0, 2], [1, 1], [1, 2],
                  [np.nan, 1], [np.nan, 2]], dtype=np.float32)
    y = [0, 0, 1, 1, int(missing_high), int(missing_high)]
    names = ['年龄"\\\t', '收入\n€']
    with cls(n_estimators=4, max_depth=2, min_samples_leaf=1,
             reg_lambda=0, learning_rate=.3) as model:
        model.fit(x, y)
        exported = model.dump_model(format='json', feature_names=names)
        dump = json.loads(exported)
        assert dump['schema_version'] == 1
        assert dump['num_features'] == 2
        assert dump['tree_count'] == 4
        assert dump['feature_names'] == names
        assert len(dump['trees']) == 4
        expected = (model.predict_proba(x)[:, 1] if cls is GBDTClassifier
                    else model.predict(x))
        for row, prediction in zip(x, expected):
            assert traverse(dump, row) == prediction
            explanation = model.explain(row, feature_names=names)
            assert explanation['prediction'] == float(prediction)
            margin = explanation['base_score']
            for tree in explanation['trees']:
                margin += tree['contribution']
            assert explanation['margin'] == margin
            assert len(explanation['trees']) == 4
            for tree in explanation['trees']:
                nodes = dump['trees'][tree['tree_index']]['nodes']
                node_id = 0
                for step in tree['path']:
                    assert step['node_id'] == node_id
                    node = nodes[node_id]
                    value = row[node['feature_index']]
                    assert step['feature_name'] == names[node['feature_index']]
                    assert step['threshold'] == node['threshold']
                    if np.isnan(value):
                        assert step['feature_value'] is None
                        assert step['condition'] == 'missing'
                        left = node['missing_direction'] == 'left'
                    else:
                        assert step['feature_value'] == float(value)
                        left = value <= node['threshold']
                        assert step['condition'] == ('<=' if left else '>')
                    node_id = node['left_id'] if left else node['right_id']
                    assert step['next_node_id'] == node_id
                    assert step['direction'] == ('left' if left else 'right')
                    assert step['missing_direction'] == node['missing_direction']
                assert tree['leaf_id'] == node_id
                assert tree['leaf_value'] == nodes[node_id]['leaf_value']
                assert tree['contribution'] == dump['learning_rate'] * tree['leaf_value']
        assert model.dump_model(format='json', feature_names=names) == exported
        assert 'tree 0' in model.dump_model()
        assert 'contribution' in model.explain(x[:1], format='text')
        assert 'missing' in model.explain(x[-1], format='text')
        selected = json.loads(model.dump_model(format='json', tree_index=np.int64(2)))
        assert selected['feature_names'] == ['f0', 'f1']
        assert selected['tree_count'] == 4
        assert [tree['tree_index'] for tree in selected['trees']] == [2]


def test_root_leaf_and_parallel_dump():
    x = np.arange(40000, dtype=np.float32).reshape(2000, 20)
    exports = []
    for threads in (1, 4):
        with GBDTRegressor(n_estimators=2, max_depth=0, n_jobs=threads) as model:
            model.fit(x, np.full(len(x), 7))
            exports.append(model.dump_model(format='json'))
            explanation = model.explain(x[0])
            assert explanation['prediction'] == 7
            assert all(not tree['path'] for tree in explanation['trees'])
            assert all(tree['leaf_value'] == 0 for tree in explanation['trees'])
            assert 'node 0: leaf_value=' in model.dump_model()
    assert exports[0] == exports[1]


def test_inspection_validation_and_closed_models():
    model = GBDTRegressor(n_estimators=2, min_samples_leaf=1)
    for operation in (lambda: model.dump_model(), lambda: model.explain([0])):
        with pytest.raises(RuntimeError):
            operation()
    model.fit([[0], [1]], [0, 1])
    for index in (-1, 2, 1.5, 2**31):
        with pytest.raises(ValueError):
            model.dump_model(tree_index=index)
    for names in (['a', 'b'], [1], 'name', ['nul\0name']):
        with pytest.raises(ValueError):
            model.dump_model(feature_names=names)
    for format in ('dot', '', None):
        with pytest.raises(ValueError):
            model.dump_model(format=format)
        with pytest.raises(ValueError):
            model.explain([0], format=format)
    for sample in (0, [], [[0], [1]], np.empty((0, 1)), [[[0]]], [0, 1], [np.inf]):
        with pytest.raises(ValueError):
            model.explain(sample)
    model.close()
    with pytest.raises(RuntimeError):
        model.dump_model()
    with pytest.raises(RuntimeError):
        model.explain([0])


def test_deep_path_reconstructs_prediction():
    with GBDTRegressor(n_estimators=1, max_depth=2, min_samples_leaf=1,
                       reg_lambda=0, learning_rate=1) as model:
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        model.fit(x, [0, 2, 1, 3])
        assert len(model.explain(x[0])['trees'][0]['path']) == 2
        dump = json.loads(model.dump_model(format='json'))
        for row, prediction in zip(np.asarray(x), model.predict(x)):
            assert traverse(dump, row) == prediction


def test_parallel_exports_and_explanations_match():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4000, 20)).astype(np.float32)
    y = x[:, 0] + 2 * x[:, 1]
    x[7, 3] = np.nan
    exports, explanations = [], []
    for threads in (1, 4):
        with GBDTRegressor(n_estimators=3, max_depth=2, min_samples_leaf=1,
                           n_jobs=threads) as model:
            model.fit(x, y)
            exports.append(model.dump_model(format='json'))
            explanations.append(model.explain(x[7]))
    assert exports[0] == exports[1]
    assert explanations[0] == explanations[1]


@pytest.mark.parametrize('operation', ['dump_model', 'explain'])
@pytest.mark.parametrize('failure_stage', ['query', 'write'])
def test_native_inspection_errors_preserve_model(monkeypatch, operation, failure_stage):
    import ctypes as ct
    import libtree as libtree_module

    with GBDTRegressor(n_estimators=1).fit([[0], [1]], [0, 1]) as model:
        before = model.predict([[0]])

        class FailedInspection:
            calls = 0

            def inspect(self, *args):
                self.calls += 1
                if failure_stage == 'write' and self.calls == 1:
                    ct.cast(args[-1], ct.POINTER(ct.c_size_t)).contents.value = 32
                    return 0
                return -1

            LtDumpModel = LtExplainPrediction = inspect

            def LtLastError(self):
                return b'inspection failed'

        failed_inspection = FailedInspection()
        with monkeypatch.context() as patch:
            patch.setattr(libtree_module, '_library', lambda: failed_inspection)
            exception = ValueError if failure_stage == 'query' else RuntimeError
            with pytest.raises(exception, match='inspection failed'):
                getattr(model, operation)(*([[0]] if operation == 'explain' else []))
        np.testing.assert_array_equal(model.predict([[0]]), before)
