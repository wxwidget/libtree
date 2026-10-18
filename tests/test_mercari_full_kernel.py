"""A source-only single-Kernel refit; smoke output is never submittable."""

import ast
import base64
import io
import json
import zipfile
from pathlib import Path
import pytest
from benchmarks.mercari_full_kernel import build


def test_source_only_full_kernel_preserves_frozen_weights_and_pretrained_version(
    tmp_path,
):
    config = json.loads(
        Path("benchmarks/mercari/full-ensemble-config.json").read_text()
    )
    meta = build("tester", tmp_path / "kernel", config, smoke=False)
    assert meta["enable_gpu"] and not meta["enable_internet"] and meta["is_private"]
    assert meta["model_sources"] == [config["pretrained_model"]]
    code = (tmp_path / "kernel" / meta["code_file"]).read_text()
    compile(code, "kernel", "exec")
    tree = ast.parse(code)
    literal = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ["SOURCE_ARCHIVE", "SMOKE"]
    }
    assert not literal["SMOKE"]
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(literal["SOURCE_ARCHIVE"]))) as z:
        saved = json.loads(z.read("benchmarks/mercari/full-ensemble-config.json"))
        assert saved["outer"] == config["outer"] and saved["inner"] == config["inner"]
        assert "benchmarks/mercari_final_torch.py" in z.namelist()
        assert "vendor/snowballstemmer/english_stemmer.py" in z.namelist()
        assert not any(
            n.endswith((".tsv", ".npy", ".npz", ".pkl", ".safetensors", ".pt", ".so"))
            for n in z.namelist()
        )
    with pytest.raises(ValueError):
        build("tester", tmp_path / "kernel", config)
    with pytest.raises(ValueError):
        build("../bad", tmp_path / "bad", config)


def test_smoke_has_distinct_kernel_name_and_never_exports_submission_csv(tmp_path):
    config = json.loads(
        Path("benchmarks/mercari/full-ensemble-config.json").read_text()
    )
    meta = build("tester", tmp_path / "smoke", config, smoke=True)
    assert meta["id"].endswith("-smoke")
    code = (tmp_path / "smoke" / meta["code_file"]).read_text()
    assert "SMOKE_ONLY_DO_NOT_SUBMIT.csv" in code
    assert "SMOKE = True" in code
