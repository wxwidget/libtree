"""Tiny real transformer training; no network or pretrained weights in tests."""
import json
import pickle
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
from transformers import BertConfig, BertModel
from benchmarks import mercari_transformer as neural
from test_mercari import listings


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        length = kwargs['max_length']
        ids = [[1] + [2 + len(word) % 20 for word in value.split()[:length - 2]] + [2] for value in text]
        return {'input_ids': [row + [0] * (length - len(row)) for row in ids],
                'attention_mask': [[1] * len(row) + [0] * (length - len(row)) for row in ids]}


def backbone():
    return BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, hidden_dropout_prob=0., attention_probs_dropout_prob=0.))


def test_metadata_and_token_encoding_are_train_only_finite_and_persistent():
    frame = listings(20)
    encoder = neural.ListingInputs().fit(frame)
    p = encoder.transform(frame.iloc[:3], TinyTokenizer(), length=16)
    changed = encoder.transform(frame.iloc[:3].assign(price=1e8, train_id=-1), TinyTokenizer(), length=16)
    for key in p:
        np.testing.assert_array_equal(p[key], changed[key])
        np.testing.assert_array_equal(p[key], pickle.loads(pickle.dumps(encoder)).transform(
            frame.iloc[:3], TinyTokenizer(), length=16)[key])
        assert np.isfinite(p[key]).all()
    assert p['input_ids'].dtype == np.uint16 and p['attention_mask'].dtype == np.uint8
    unknown = frame.iloc[:1].assign(brand_name='unknown', category_name='unseen/path/category')
    assert np.all(encoder.transform(unknown, TinyTokenizer(), length=8)['categories'] == 0)


@pytest.mark.parametrize('loss', ['mse', 'huber'])
def test_tiny_backbone_learns_replays_and_keeps_masked_padding_irrelevant(loss):
    torch.set_num_threads(1);torch.manual_seed(42)
    frame = listings(20)
    inputs = neural.ListingInputs().fit(frame)
    x = inputs.transform(frame, TinyTokenizer(), length=16)
    model = neural.PriceTransformer(backbone(), inputs.cardinalities_, x['numeric'].shape[1], width=16)
    original_backbone = model.backbone.embeddings.word_embeddings.weight.detach().clone()
    loader = neural.loader(x, np.linspace(-.2, .2, 20), batch_size=10, shuffle=True)
    optimizer = neural.optimizer(model, backbone_rate=.003, head_rate=.01)
    device = torch.device('cpu')
    first = neural.train_epoch(model, loader, optimizer, device, loss=loss)
    last = first
    for _ in range(3):
        last = neural.train_epoch(model, loader, optimizer, device, loss=loss)
    assert last < first
    assert not torch.allclose(original_backbone, model.backbone.embeddings.word_embeddings.weight)
    prediction = neural.predict(model, x, device, batch_size=10)
    altered = {key: value.copy() for key, value in x.items()}
    altered['input_ids'][altered['attention_mask'] == 0] = 31
    np.testing.assert_allclose(prediction, neural.predict(model, altered, device), atol=1e-6)
    clone = neural.PriceTransformer(backbone(), inputs.cardinalities_, x['numeric'].shape[1], width=16)
    clone.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(prediction, neural.predict(clone, x, device, batch_size=10))
    with pytest.raises(ValueError):
        neural.loader(x, np.zeros(1))
    with pytest.raises(ValueError):
        neural.loader(x, batch_size=0)
    with pytest.raises(ValueError):
        neural.train_epoch(model, loader, optimizer, device, loss='unknown')


def test_end_to_end_worker_saves_frozen_checkpoint_predictions_and_metadata(tmp_path, monkeypatch):
    data = tmp_path / 'data';data.mkdir()
    for i, window in enumerate(('train', 'calibration', 'selection', 'confirmation')):
        frame = listings(30).assign(train_id=np.arange(30) + i * 100)
        if window == 'confirmation':frame = frame.drop(columns='price')
        frame.to_csv(data / (window + '.tsv.gz'), sep='\t', index=False)
    monkeypatch.setattr(neural, 'load_pretrained', lambda model, revision: (TinyTokenizer(), backbone(), 'offline-test'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    args = ['--data-dir', str(data), '--output-dir', str(tmp_path / 'study'), '--epochs', '2',
            '--batch-size', '10', '--length', '16', '--width', '16', '--threads', '1']
    with pytest.raises(RuntimeError, match='real GPU'):
        neural.main(args)
    neural.main(args + ['--allow-cpu'])
    report = json.loads((tmp_path / 'study/report.json').read_text())
    assert report['revision'] == 'offline-test' and report['gpu'] is None
    assert report['confirmation_labels_loaded'] is False
    assert report['models']['finetuned_minilm_huber']['selection_rmsle'] >= 0
    assert (tmp_path / 'study/checkpoint.pt').is_file()
    with np.load(tmp_path / 'study/finetuned_minilm_huber.npz') as p:
        np.testing.assert_array_equal(p['confirmation_ids'], np.arange(30) + 300)
        assert np.isfinite(p['confirmation']).all()
    with pytest.raises(ValueError, match='new output'):
        neural.main(args + ['--allow-cpu'])
    with pytest.raises(SystemExit) as error:
        neural.main(args + ['--epochs', '0'])
    assert error.value.code == 2
    monkeypatch.setattr(sys, 'argv', ['mercari_transformer', '--help'])
    with pytest.raises(SystemExit) as error:
        runpy.run_module('benchmarks.mercari_transformer', run_name='__main__')
    assert error.value.code == 0


def test_pretrained_loader_uses_an_immutable_revision(monkeypatch):
    import huggingface_hub
    import transformers
    monkeypatch.setattr(huggingface_hub, 'model_info', lambda _: SimpleNamespace(sha='resolved-revision'))
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda model, **kwargs: calls.append(kwargs) or TinyTokenizer())
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', lambda model, **kwargs: calls.append(kwargs) or backbone())
    assert neural.load_pretrained('test-model', None)[2] == 'resolved-revision'
    assert neural.load_pretrained('test-model', 'pinned-revision')[2] == 'pinned-revision'
    assert [value['revision'] for value in calls] == ['resolved-revision'] * 2 + ['pinned-revision'] * 2


def test_module_execution_saves_importable_metadata(tmp_path, monkeypatch):
    import huggingface_hub
    import transformers
    data = tmp_path / 'data';data.mkdir()
    for i, name in enumerate(('train', 'calibration', 'selection', 'confirmation')):
        part = listings(20).assign(train_id=np.arange(20) + 100 * i)
        if name == 'confirmation':part = part.drop(columns='price')
        part.to_csv(data / (name + '.tsv.gz'), sep='\t', index=False)
    monkeypatch.setattr(huggingface_hub, 'model_info', lambda _: SimpleNamespace(sha='offline-test'))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: TinyTokenizer())
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', lambda *a, **kw: backbone())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    out = tmp_path / 'module-worker'
    monkeypatch.setattr(sys, 'argv', ['mercari_transformer', '--data-dir', str(data),
        '--output-dir', str(out), '--epochs', '1', '--length', '16', '--batch-size', '10',
        '--width', '16', '--threads', '1', '--allow-cpu'])
    runpy.run_module('benchmarks.mercari_transformer', run_name='__main__')
    with (out / 'inputs.pkl').open('rb') as stream:
        inputs = pickle.load(stream)
    assert inputs.__class__.__module__ == 'benchmarks.mercari_transformer'
