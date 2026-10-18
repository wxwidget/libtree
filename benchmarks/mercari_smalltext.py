"""Bounded subword-bag and BERT-tiny price regressors.

The bag is an independent regression adaptation of Joulin et al. (2017),
https://aclanthology.org/E17-2068/: average embeddings plus hashed bigrams.
WordPiece subwords replace fastText's character n-grams; this is NOT the
fastText package. Small pretrained BERT follows Turc et al. (2019),
https://arxiv.org/abs/1908.08962. Both optimize standardized log-price loss.
"""

import argparse
import importlib.metadata
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch import nn

from benchmarks.mercari_frontier import (
    WINDOWS,
    dump,
    read_windows,
    save_predictions,
    score,
)
from benchmarks.mercari_text_cache import TextCache, identity
from benchmarks.mercari_transformer import (
    ListingInputs,
    PriceTransformer,
    load_pretrained,
    loader,
    optimizer,
    predict,
    train_epoch,
)


class SubwordBag(nn.Module):
    """Order-free unigram view plus order-sensitive hashed adjacent pairs."""

    def __init__(
        self,
        vocabulary,
        cardinalities,
        numeric_features,
        embedding_dim=48,
        width=128,
        buckets=32768,
    ):
        super().__init__()
        self.buckets = buckets
        self.backbone = nn.ModuleDict(
            dict(
                words=nn.Embedding(vocabulary, embedding_dim, padding_idx=0),
                bigrams=nn.Embedding(buckets + 1, embedding_dim, padding_idx=0),
            )
        )
        self.categories = nn.ModuleList(
            [nn.Embedding(size, 12, padding_idx=0) for size in cardinalities]
        )
        self.head = nn.Sequential(
            nn.Linear(
                2 * embedding_dim + 12 * len(cardinalities) + numeric_features, width
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(width, 1),
        )

    def forward(self, ids, mask, categories, numeric):
        valid = mask.bool()
        pairs_valid = valid[:, 1:] & valid[:, :-1]
        pairs = (ids[:, :-1] * 1000003 + ids[:, 1:]) % self.buckets + 1
        pairs = pairs.masked_fill(~pairs_valid, 0)
        words = self.backbone["words"](ids) * valid.unsqueeze(-1)
        bigrams = self.backbone["bigrams"](pairs) * pairs_valid.unsqueeze(-1)
        pooled = [
            words.sum(1) / valid.sum(1, keepdim=True).clamp_min(1),
            bigrams.sum(1) / pairs_valid.sum(1, keepdim=True).clamp_min(1),
        ]
        cat = [layer(categories[:, i]) for i, layer in enumerate(self.categories)]
        return self.head(torch.cat([*pooled, *cat, numeric], dim=1)).squeeze(-1)


def parameter_counts(model, maximum):
    counts = dict(
        total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    if counts["total"] > maximum:
        raise ValueError(f"parameter budget exceeded: {counts['total']} > {maximum}")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["bag", "bert"], default="bag")
    parser.add_argument("--model-id", default="prajjwal1/bert-tiny")
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--buckets", type=int, default=32768)
    parser.add_argument("--max-parameters", type=int, default=8000000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cache-dir", type=Path, help="Trusted immutable input cache")
    args = parser.parse_args(argv)
    if (
        min(
            args.epochs,
            args.batch_size,
            args.length,
            args.width,
            args.embedding_dim,
            args.buckets,
            args.max_parameters,
            args.threads,
        )
        < 1
    ):
        parser.error("positive model and resource settings required")
    if args.output_dir.exists():
        raise ValueError("use a new output directory")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("GPU required unless --allow-cpu is explicit")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    start, cpu_start = time.perf_counter(), time.process_time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, backbone, revision = load_pretrained(args.model_id, args.revision)
    frame = read_windows(args.data_dir)
    if "price" in frame["confirmation"]:
        raise ValueError("confirmation labels must stay local")
    cache = (
        TextCache(
            args.cache_dir,
            identity(args.data_dir, args.model_id, revision, args.length),
        )
        if args.cache_dir
        else None
    )
    cached = cache.load() if cache else None
    inputs = cached[0] if cached else ListingInputs().fit(frame["train"])
    numeric_features = inputs.transform(
        frame["train"].iloc[:1], tokenizer, args.length
    )["numeric"].shape[1]
    if args.architecture == "bag":
        model = SubwordBag(
            backbone.config.vocab_size,
            inputs.cardinalities_,
            numeric_features,
            args.embedding_dim,
            args.width,
            args.buckets,
        )
        del backbone
    else:
        model = PriceTransformer(
            backbone, inputs.cardinalities_, numeric_features, args.width
        )
    counts = parameter_counts(model, args.max_parameters)
    encoded = cached[1] if cached else {}
    if not cached:
        for name, part in frame.items():
            encoded[name] = inputs.transform(part, tokenizer, args.length)
            print("Inputs ready:", name, len(part), flush=True)
        if cache:
            cache.save(inputs, encoded)
    model = model.to(device)
    y, selection_y = [
        np.log1p(frame[name].price.to_numpy()) for name in ("train", "selection")
    ]
    mean, scale = float(np.mean(y)), max(float(np.std(y)), 1e-6)
    algorithm = optimizer(
        model,
        backbone_rate=0.003 if args.architecture == "bag" else 5e-5,
        head_rate=0.001,
    )
    batches = loader(
        encoded["train"], (y - mean) / scale, args.batch_size, shuffle=True
    )
    preparation_seconds = time.perf_counter() - start
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        stamp = time.perf_counter()
        train_loss = train_epoch(model, batches, algorithm, device, args.loss)
        selection = mean + scale * predict(
            model, encoded["selection"], device, args.batch_size
        )
        metric = score(selection_y, selection)
        history.append(
            dict(
                epoch=epoch + 1,
                train_loss=train_loss,
                selection_rmsle=metric,
                seconds=time.perf_counter() - stamp,
            )
        )
        print("SMALLTEXT_EPOCH", history[-1], flush=True)
        if metric < best:
            best = metric
            torch.save(model.state_dict(), args.output_dir / "checkpoint.pt")
        for group in algorithm.param_groups:
            group["lr"] *= 0.7
    model.load_state_dict(
        torch.load(
            args.output_dir / "checkpoint.pt", map_location=device, weights_only=True
        )
    )
    stamp = time.perf_counter()
    prediction = {
        k: mean + scale * predict(model, encoded[k], device, args.batch_size)
        for k in WINDOWS
    }
    prediction_seconds = time.perf_counter() - stamp
    name = args.architecture + "_" + args.loss
    save_predictions(args.output_dir, {name: prediction}, frame)
    with (args.output_dir / "inputs.pkl").open("wb") as stream:
        pickle.dump(inputs, stream)
    dump(
        args.output_dir / "report.json",
        dict(
            model_id=args.model_id,
            revision=revision,
            architecture=args.architecture,
            input_cache_hit=cached is not None,
            parameters=counts,
            max_parameters=args.max_parameters,
            tokenizer_pretrained=True,
            backbone_pretrained=args.architecture == "bert",
            mean=mean,
            scale=scale,
            width=args.width,
            embedding_dim=args.embedding_dim,
            buckets=args.buckets,
            max_length=args.length,
            batch_size=args.batch_size,
            threads=args.threads,
            loss=args.loss,
            train_rows=len(y),
            history=history,
            best_selection_rmsle=best,
            preparation_seconds=preparation_seconds,
            prediction_seconds=prediction_seconds,
            total_seconds=time.perf_counter() - start,
            process_cpu_seconds=time.process_time() - cpu_start,
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
            confirmation_labels_loaded=False,
            versions={
                k: importlib.metadata.version(k)
                for k in ("torch", "transformers", "numpy", "pandas")
            },
        ),
    )


if __name__ == "__main__":
    from benchmarks.mercari_smalltext import main as entrypoint

    entrypoint()
