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
