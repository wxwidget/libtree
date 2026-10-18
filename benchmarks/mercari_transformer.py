"""Supervised MiniLM price regression with train-only categorical embeddings.

SBERT/MiniLM initialization: https://aclanthology.org/D19-1410/ and
https://arxiv.org/abs/2002.10957. Task-specific fine tuning follows the transfer
learning principle of Devlin et al., https://aclanthology.org/N19-1423/.
This independently implemented head predicts standardized log prices.
"""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.mercari_features import ListingFeatures
from benchmarks.mercari_semantics import TextAttributes


class ListingInputs:
    categories = ('brand_name', 'category_name', 'category_0', 'category_1', 'category_2', 'category_condition')

    def fit(self, frame):
        self.listing_ = ListingFeatures().fit(frame)
        p = self.listing_.transform(frame)
        self.codes_ = {name: {value: i + 1 for i, value in enumerate(sorted(p[name].unique()))}
                       for name in self.categories}
        self.cardinalities_ = [len(self.codes_[name]) + 1 for name in self.categories]
        return self

    def transform(self, frame, tokenizer, length=128):
        p = self.listing_.transform(frame)
        text = p.name + ' [SEP] ' + p.brand_name + ' [SEP] ' + p.category_name + ' [SEP] ' + p.item_description
        ids, masks = [], []
        for begin in range(0, len(frame), 8192):
            encoded = tokenizer(text.iloc[begin:begin + 8192].tolist(), padding='max_length',
                truncation=True, max_length=length, return_token_type_ids=False)
            values = np.asarray(encoded['input_ids'], dtype=np.int32)
            ids.append(values.astype(np.uint16) if np.max(values) < 65536 else values)
            masks.append(np.asarray(encoded['attention_mask'], dtype=np.uint8))
        categories = np.column_stack([p[name].map(self.codes_[name]).fillna(0) for name in self.categories]).astype(np.int32)
        numeric = np.column_stack([p.item_condition_id.astype(float) / 5., p.shipping.astype(float),
            np.log1p(p.name.str.len()) / 5., np.log1p(p.item_description.str.len()) / 7.,
            TextAttributes().transform(frame)]).astype(np.float32)
        return dict(input_ids=np.concatenate(ids), attention_mask=np.concatenate(masks), categories=categories, numeric=numeric)


class PriceTransformer(nn.Module):
    def __init__(self, backbone, cardinalities, numeric_features, width=256):
        super().__init__()
        self.backbone = backbone
        self.categories = nn.ModuleList([nn.Embedding(size, 12, padding_idx=0) for size in cardinalities])
        self.head = nn.Sequential(nn.Linear(2 * backbone.config.hidden_size + 12 * len(cardinalities) + numeric_features, width),
            nn.ReLU(), nn.Dropout(.1), nn.Linear(width, 1))

    def forward(self, ids, mask, categories, numeric):
        hidden = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        valid = mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.)
        cls = hidden[:, 0] * valid[:, 0]
        categorical = [layer(categories[:, i]) for i, layer in enumerate(self.categories)]
        return self.head(torch.cat([mean, cls, *categorical, numeric], dim=1)).squeeze(-1)


def loader(encoded, target=None, batch_size=512, shuffle=False):
    n = len(encoded['input_ids'])
    labels = np.zeros(n, dtype=np.float32) if target is None else np.asarray(target, dtype=np.float32)
    if batch_size < 1 or labels.shape != (n,) or not np.isfinite(labels).all():
        raise ValueError('finite matching labels and positive batch size required')
    tensors = [torch.from_numpy(np.asarray(encoded[name])) for name in ('input_ids', 'attention_mask', 'categories', 'numeric')]
    return DataLoader(TensorDataset(*tensors, torch.from_numpy(labels)), batch_size=batch_size,
        shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(42))


def optimizer(model, backbone_rate=3e-5, head_rate=1e-3):
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    heads = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=backbone_rate),
                             dict(params=heads, lr=head_rate)], weight_decay=.01)


def move(batch, device):
    return [value.to(device, dtype=torch.long if i < 3 else torch.float32, non_blocking=True)
            for i, value in enumerate(batch)]


def autocast(device, enabled=True):
    if device.type != 'cuda' or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return torch.autocast('cuda', dtype=dtype)


def train_epoch(model, batches, algorithm, device, loss='huber', mixed_precision=True):
    if loss not in ('mse', 'huber'):
        raise ValueError('unknown regression loss')
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision and device.type == 'cuda' and not torch.cuda.is_bf16_supported(including_emulation=False))
    total, count = 0., 0
    for step, batch in enumerate(batches, 1):
        ids, mask, categories, numeric, y = move(batch, device)
        algorithm.zero_grad(set_to_none=True)
        with autocast(device, enabled=mixed_precision):
            output = model(ids, mask, categories, numeric).float()
            value = nn.functional.huber_loss(output, y, delta=.5) if loss == 'huber' else nn.functional.mse_loss(output, y)
        scaler.scale(value).backward()
        scaler.unscale_(algorithm)
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(algorithm); scaler.update()
        total += float(value.detach()) * len(y)
        count += len(y)
        if step == 1 or step % 500 == 0:
            print('Transformer train batches:', step, '/', len(batches), 'mean loss:', total / count, flush=True)
    return total / count


def predict(model, encoded, device, batch_size=512, mixed_precision=True):
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader(encoded, batch_size=batch_size):
            ids, mask, categories, numeric, _ = move(batch, device)
            with autocast(device, enabled=mixed_precision):
                output = model(ids, mask, categories, numeric)
            values.append(output.float().cpu().numpy())
    return np.concatenate(values)


def load_pretrained(model_id, revision):
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer
    revision = revision or model_info(model_id).sha
    return (AutoTokenizer.from_pretrained(model_id, revision=revision),
            AutoModel.from_pretrained(model_id, revision=revision), revision)


def main(argv=None):
    from benchmarks.mercari_frontier import WINDOWS, dump, read_windows, save_predictions, score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-id', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--revision')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--loss', choices=['mse', 'huber'], default='huber')
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.length, args.width, args.threads) < 1:
        parser.error('positive model and resource settings required')
    if args.output_dir.exists():
        raise ValueError('use a new output directory')
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('supervised transformer experiment requires a real GPU')
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.perf_counter()
    tokenizer, backbone, revision = load_pretrained(args.model_id, args.revision)
    frame = read_windows(args.data_dir)
    inputs = ListingInputs().fit(frame['train'])
    encoded = {}
    for name, part in frame.items():
        encoded[name] = inputs.transform(part, tokenizer, args.length)
        print('Transformer inputs ready:', name, len(part), flush=True)
    y, val_y = [np.log1p(frame[name].price.to_numpy()) for name in ('train', 'selection')]
    mean, scale = float(np.mean(y)), max(float(np.std(y)), 1e-6)
    model = PriceTransformer(backbone, inputs.cardinalities_, encoded['train']['numeric'].shape[1], args.width).to(device)
    algorithm = optimizer(model)
    batches = loader(encoded['train'], (y - mean) / scale, args.batch_size, shuffle=True)
    preparation_seconds = time.perf_counter() - start
    best, history = float('inf'), []
    for epoch in range(args.epochs):
        stamp = time.perf_counter()
        train_loss = train_epoch(model, batches, algorithm, device, args.loss)
        selection = mean + scale * predict(model, encoded['selection'], device, args.batch_size)
        metric = score(val_y, selection)
        row = dict(epoch=epoch + 1, train_loss=train_loss, selection_rmsle=metric,
                   seconds=time.perf_counter() - stamp)
        history.append(row)
        print('Fine-tuned MiniLM:', row, flush=True)
        if metric < best:
            best = metric
            torch.save(model.state_dict(), args.output_dir / 'checkpoint.pt')
        for group in algorithm.param_groups:
            group['lr'] *= .5
    model.load_state_dict(torch.load(args.output_dir / 'checkpoint.pt', map_location=device, weights_only=True))
    name = 'finetuned_minilm_' + args.loss
    prediction = {k: mean + scale * predict(model, encoded[k], device, args.batch_size) for k in WINDOWS}
    save_predictions(args.output_dir, {name: prediction}, frame)
    with (args.output_dir / 'inputs.pkl').open('wb') as stream:
        pickle.dump(inputs, stream)
    dump(args.output_dir / 'report.json', dict(model_id=args.model_id, revision=revision, external_pretrained_data=True,
        mean=mean, scale=scale, width=args.width, max_length=args.length, batch_size=args.batch_size,
        preparation_seconds=preparation_seconds, total_seconds=time.perf_counter() - start,
        train_rows=len(y), gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None,
        amp='bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else 'float16' if device.type == 'cuda' else None,
        models={name: dict(selection_rmsle=best, history=history)}, confirmation_labels_loaded=False,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'pandas')}))


if __name__ == '__main__':
    from benchmarks.mercari_transformer import main as entrypoint
    entrypoint()
