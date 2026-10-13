"""CLI contracts first: parameters, standalone model round trip, safe failures."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

CLI = Path(os.environ.get("LIBTREE_CLI", "build/xgbt")).resolve()


def invoke(*args):
    return subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)


def test_train_predict_round_trip(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    model = tmp_path / "model.lt"
    predictions = tmp_path / "prediction.csv"
    train.write_text("label,value\n0,0\n0,1\n1,2\n1,3\n")
    test.write_text("value\n0\n3\n\n")
    result = invoke("train", "--train", train, "--test", test, "--header",
                    "--model", model, "--output", predictions, "--objective", "binary",
                    "--trees", 40, "--depth", 2, "--bins", 8, "--min-leaf", 1,
                    "--rate", .2, "--l2", 0, "--min-gain", 0)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["trees"] == 40
    first = np.loadtxt(predictions)
    assert first[0] < .01 and first[1] > .99
    second = tmp_path / "second.csv"
    result = invoke("predict", "--model", model, "--data", test, "--header", "--output", second)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(first, np.loadtxt(second))


def test_missing_regression_and_no_model_needed(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    output = tmp_path / "pred.csv"
    train.write_text("0,nan\n0,\n10,1\n10,1\n")
    test.write_text("nan\n1\n")
    result = invoke("train", "--train", train, "--test", test, "--output", output,
                    "--min-leaf", 1, "--trees", 60)
    assert result.returncode == 0, result.stderr
    prediction = np.loadtxt(output)
    assert prediction[0] < .1 and prediction[1] > 9.9


@pytest.mark.parametrize("extra", [["--unknown", "1"], ["--trees", "2junk"],
    ["--rate", "nan"], ["--bins", "1"], ["--objective", "multiclass"], ["--trees"]])
def test_invalid_options(tmp_path, extra):
    train = tmp_path / "train.csv"
    train.write_text("0,1\n1,2\n")
    result = invoke("train", "--train", train, *extra)
    assert result.returncode != 0
    assert result.stderr


def test_invalid_csv_and_model_preserves_output(tmp_path):
    train = tmp_path / "train.csv"
    output = tmp_path / "out.csv"
    output.write_text("existing result\n")
    for text in ("0,1\n1,2,3\n", "0,1oops\n", "0,inf\n", "", "0,\"quoted\"\n"):
        train.write_text(text)
        result = invoke("train", "--train", train, "--output", output)
        assert result.returncode != 0
        assert output.read_text() == "existing result\n"
    model = tmp_path / "bad.lt"
    for text in ("garbage", "LIBTREE_GBDT 99", "LIBTREE_GBDT 1\n1000000000"):
        model.write_text(text)
        assert invoke("predict", "--model", model, "--data", train, "--output", output).returncode != 0


def test_help_and_missing_file():
    assert invoke("--help").returncode == 0
    assert "--learning-rate" in invoke("--help").stdout
    assert invoke("train", "--train", "/nonexistent/libtree.csv").returncode != 0
    assert invoke().returncode == 2
    assert invoke("unknown").returncode != 0


def test_model_graph_validation_and_output_alias(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("0\n1\n")
    model = tmp_path / "model.lt"
    out = tmp_path / "out.csv"
    valid = ("LIBTREE_GBDT 1\n0 1 1 8 1 .1 0 0 1 0 1\n3\n"
             "0 1 2 .5 0 1\n-1 -1 -1 0 -1 1\n-1 -1 -1 0 1 1\n")
    model.write_text(valid)
    assert invoke("predict", "--model", model, "--data", data, "--output", out).returncode == 0
    for invalid in (
        valid.replace("0 1 2 .5", "0 0 2 .5"),  # cycle
        valid.replace("0 1 2 .5", "0 1 1 .5"),  # shared child
        valid.replace("0 1 2 .5", "9 1 2 .5"),  # wrong feature
        valid.replace(".5 0 1", ".5 0 2"),      # invalid missing flag
        valid.replace("0 1 2 .5", "0 1 9 .5"),  # out of bounds
        valid + "trailing content",
        valid.replace("\n3\n", "\n1000000000\n"),
    ):
        model.write_text(invalid)
        out.write_text("existing output\n")
        assert invoke("predict", "--model", model, "--data", data, "--output", out).returncode != 0
        assert out.read_text() == "existing output\n"
    model.write_text(valid)
    assert invoke("predict", "--model", model, "--data", data, "--output", data).returncode != 0


def test_parallel_thread_option(tmp_path):
    model = tmp_path / 'model.lt'
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    train = Path('examples/csv/train.csv').resolve()
    test = Path('examples/csv/test.csv').resolve()
    result = invoke('train', '--train', train, '--test', test, '--header',
                    '--model', model, '--output', a, '--threads', 4)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['threads'] == 4
    result = invoke('predict', '--model', model, '--data', test, '--header',
                    '--output', b, '--threads', 2)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['threads'] == 2
    assert a.read_bytes() == b.read_bytes()
    for value in ('0', '-1', '257', '2oops'):
        assert invoke('train', '--train', train, '--header', '--threads', value).returncode != 0


def test_cli_parse_and_atomic_error_paths(tmp_path):
    train = tmp_path / 'train.csv'
    train.write_text('label,value\n0,0\n1,1\n')
    # Parse validation branches.
    assert invoke('train', '--train', train, '--data', train).returncode != 0
    assert invoke('predict', '--model', train, '--data', train, '--output', train,
                  '--trees', 2).returncode != 0
    assert invoke('predict').returncode != 0
    assert invoke('train', '--train', train, '--train', train).returncode != 0
    # WriteAtomic's temporary-file cleanup runs when the output parent is absent.
    missing_parent = tmp_path / 'missing' / 'out.csv'
    result = invoke('train', '--train', train, '--header', '--output', missing_parent)
    assert result.returncode != 0
    assert not missing_parent.exists()


def test_dump_and_explain_saved_model(tmp_path):
    model = tmp_path / 'model.lt'
    model.write_text('LIBTREE_GBDT 1\n0 2 1 8 1 .5 0 0 1 1.25 2\n3\n'
                     '0 1 2 .5 0 0\n-1 -1 -1 0 -2 1\n-1 -1 -1 0 2 1\n'
                     '1\n-1 -1 -1 0 1 1\n')
    names = tmp_path / 'names.txt'
    names.write_text('年龄"\\\t\n')
    result = invoke('dump', '--model', model, '--format', 'json', '--feature-names', names)
    assert result.returncode == 0, result.stderr
    dump = json.loads(result.stdout)
    assert dump['feature_names'] == ['年龄"\\\t']
    assert len(dump['trees']) == 2
    assert dump['trees'][0]['nodes'][0]['missing_direction'] == 'right'
    assert dump['trees'][0]['nodes'][1]['contribution'] == -1
    out = tmp_path / 'dump.txt'
    result = invoke('dump', '--model', model, '--tree', 1, '--output', out)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ''
    assert 'tree 1' in out.read_text() and 'tree 0' not in out.read_text()
    for value, expected in (('0', .75), ('.5', .75), ('1', 2.75), ('nan', 2.75)):
        data = tmp_path / 'sample.csv'
        data.write_text('value\n' + value + '\n')
        result = invoke('explain', '--model', model, '--data', data, '--header', '--format', 'json')
        assert result.returncode == 0, result.stderr
        explanation = json.loads(result.stdout)
        assert explanation['prediction'] == expected
        assert len(explanation['trees']) == 2
        assert explanation['trees'][0]['path'][0]['condition'] == (
            'missing' if value == 'nan' else '>' if value == '1' else '<=')
        assert invoke('explain', '--model', model, '--data', data, '--header',
                      '--feature-names', names, '--output', out).returncode == 0
        assert 'contribution' in out.read_text()


def test_inspection_cli_rejects_bad_requests_without_overwriting(tmp_path):
    model = tmp_path / 'model.lt'
    model.write_text('LIBTREE_GBDT 1\n0 1 0 8 1 .1 0 0 1 3 1\n1\n-1 -1 -1 0 0 1\n')
    data = tmp_path / 'data.csv'
    data.write_text('0\n1\n')
    names = tmp_path / 'names.txt'
    out = tmp_path / 'out.txt'
    out.write_text('preserved')
    for extra in (['--format', 'dot'], ['--tree', '-1'], ['--tree', '1'],
                  ['--trees', '1'], ['--data', data], ['--header'],
                  ['--feature-names', tmp_path / 'missing'], ['--threads', '2']):
        assert invoke('dump', '--model', model, '--output', out, *extra).returncode != 0
        assert out.read_text() == 'preserved'
    assert invoke('dump').returncode != 0
    assert invoke('explain', '--model', model).returncode != 0
    assert invoke('explain', '--model', model, '--data', data, '--output', out).returncode != 0
    assert out.read_text() == 'preserved'
    for text in ('', 'a\nb\n'):
        names.write_text(text)
        assert invoke('dump', '--model', model, '--feature-names', names).returncode != 0
    assert invoke('dump', '--model', model, '--output', model).returncode != 0
    names.write_text('age\r\n')
    assert invoke('dump', '--model', model, '--feature-names', names).returncode == 0
    assert invoke('train', '--train', data, '--feature-names', names).returncode != 0


def test_explanation_overflow_does_not_write_invalid_json(tmp_path):
    model = tmp_path / 'model.lt'
    model.write_text('LIBTREE_GBDT 1\n0 1 0 8 1 1 0 0 1 0 1\n1\n'
                     '-1 -1 -1 0 1e308 1\n')
    sample = tmp_path / 'sample.csv'
    sample.write_text('0\n')
    out = tmp_path / 'out.json'
    out.write_text('preserved')
    result = invoke('explain', '--model', model, '--data', sample,
                    '--format', 'json', '--output', out)
    assert result.returncode != 0
    assert 'finite' in result.stderr
    assert out.read_text() == 'preserved'
