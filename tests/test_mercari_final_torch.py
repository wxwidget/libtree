"""Real tiny backbone refit and text-vector heads with streamed final exports."""

from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from benchmarks import mercari_final as final
from benchmarks import mercari_final_torch as workers
from test_mercari_final import inputs, tiny_config
from test_mercari_transformer import TinyTokenizer, backbone


class Vectors:
    def encode(self, texts, **kwargs):
        values = np.asarray(
            [[len(s), len(s.split()), len(set(s)), 1.0] for s in texts],
            dtype=np.float32,
        )
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_vector_worker_fits_real_linear_and_tree_heads_and_streams_predictions(
    tmp_path, monkeypatch
):
    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", lambda *a, **kw: Vectors()
    )
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    args.pretrained = tmp_path
    (tmp_path / "model.safetensors").write_bytes(b"test-only")
    final.prepare(args)
    report = workers.vectors_stage(args, tiny_config())
    assert report["dtype"] == "float32"
    block = final.read_stage(args.output_dir, "vectors", "test")
    assert set(block) == {"ids", "minilm_lgbm63", "minilm_ridge_10", "minilm_ridge_100"}
    assert all(np.isfinite(v).all() for k, v in block.items() if k != "ids")


def test_transformer_worker_updates_real_tiny_backbone_for_frozen_epochs(
    tmp_path, monkeypatch
):
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer()
    )
    model = backbone()
    original = model.embeddings.word_embeddings.weight.detach().clone()
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **kw: model
    )
    args = inputs(tmp_path)
    args.output_dir.mkdir()
    args.pretrained = tmp_path
    (tmp_path / "model.safetensors").write_bytes(b"test-only")
    final.prepare(args)
    report = workers.transformer_stage(args, tiny_config())
    assert len(report["losses"]) == 1 and report["gpu"] is None
    assert not torch.equal(original, model.embeddings.word_embeddings.weight)
    block = final.read_stage(args.output_dir, "transformer", "test")
    assert (
        block["finetuned_minilm_huber"].shape == (7,)
        and np.isfinite(block["finetuned_minilm_huber"]).all()
    )
