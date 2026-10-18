"""GPU text workers for the frozen full Mercari refit; optional Torch dependency."""

import gc
import numpy as np
from benchmarks import mercari
from benchmarks.mercari_final import training, export


def vectors_stage(args, config):
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor
    from benchmarks.mercari_semantics import LatentText, SentenceVectors

    base, calibration = training(args)
    y = np.log1p(base.price.to_numpy())
    model = SentenceTransformer(
        str(args.pretrained), device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.max_seq_length = 128
    vectors = SentenceVectors(model, batch_size=256)
    x = vectors.transform(base)
    models = {f"minilm_ridge_{a:g}": Ridge(alpha=a).fit(x, y) for a in (10.0, 100.0)}
    metadata = LatentText(
        components=config.get("vector_components", 16), max_features=10000
    ).fit(base)
    combined = np.column_stack([x, metadata.transform(base)])
    offset = x.shape[1]
    tree = LGBMRegressor(
        n_estimators=config["vector_rounds"],
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=80,
        reg_lambda=10.0,
        colsample_bytree=0.8,
        n_jobs=args.threads,
        random_state=42,
        verbosity=-1,
    )
    tree.fit(
        combined,
        y,
        categorical_feature=[offset + i for i in metadata.categorical_indices_],
    )
    del x, combined, base
    gc.collect()

    def prediction(frame):
        z = vectors.transform(frame)
        return dict(
            minilm_lgbm63=tree.predict(np.column_stack([z, metadata.transform(frame)])),
            **{n: m.predict(z) for n, m in models.items()},
        )

    export(args, "vectors", calibration, prediction)
    return dict(
        model_sha256=mercari.sha256(args.pretrained / "model.safetensors"),
        dtype="float32",
        rounds=config["vector_rounds"],
    )


def transformer_stage(args, config):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from benchmarks.mercari_transformer import (
        ListingInputs,
        PriceTransformer,
        loader,
        optimizer,
        train_epoch,
        predict,
    )

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = config["transformer"]
    base, calibration = training(args)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained), local_files_only=True
    )
    backbone = AutoModel.from_pretrained(str(args.pretrained), local_files_only=True)
    inputs = ListingInputs().fit(base)
    encoded = inputs.transform(base, tokenizer, settings["length"])
    y = np.log1p(base.price.to_numpy())
    mean = float(np.mean(y))
    scale = max(float(np.std(y)), 1e-6)
    model = PriceTransformer(
        backbone, inputs.cardinalities_, encoded["numeric"].shape[1], settings["width"]
    ).to(device)
    algorithm = optimizer(model)
    batches = loader(encoded, (y - mean) / scale, settings["batch_size"], shuffle=True)
    history = []
    for epoch in range(settings["epochs"]):
        history.append(train_epoch(model, batches, algorithm, device, "huber"))
        for group in algorithm.param_groups:
            group["lr"] *= 0.5
    del encoded, base, batches, algorithm
    gc.collect()

    def prediction(frame):
        z = inputs.transform(frame, tokenizer, settings["length"])
        return dict(
            finetuned_minilm_huber=mean
            + scale * predict(model, z, device, settings["batch_size"])
        )

    export(args, "transformer", calibration, prediction)
    return dict(
        losses=history,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        model_sha256=mercari.sha256(args.pretrained / "model.safetensors"),
    )
