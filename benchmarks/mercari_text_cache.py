"""Verified reuse of train-only text encoding across bounded model variants.

Cache files are trusted local artifacts, not downloaded model inputs. Hashes
prevent accidental stale/corrupt reuse; they do not authenticate hostile pickle.
Memory maps use copy-on-write so Torch never receives a read-only NumPy buffer.
"""

import importlib.metadata
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from benchmarks.mercari import sha256

WINDOWS = ("train", "calibration", "selection", "confirmation")
ARRAYS = ("input_ids", "attention_mask", "categories", "numeric")


def identity(data, model_id, revision, length):
    source = Path(__file__).parent
    modules = (
        "mercari.py",
        "mercari_frontier.py",
        "mercari_features.py",
        "mercari_semantics.py",
        "mercari_transformer.py",
        "mercari_text_cache.py",
    )
    return dict(
        python=sys.version.split()[0],
        model_id=model_id,
        revision=revision,
        length=length,
        inputs={w: sha256(data / (w + ".tsv.gz")) for w in WINDOWS},
        source={name: sha256(source / name) for name in modules},
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "transformers",
                "tokenizers",
            )
        },
    )


class TextCache:
    def __init__(self, directory, key):
        self.directory, self.key = Path(directory), key

    def load(self):
        if not self.directory.exists():
            return None
        manifest = self.directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError("incomplete input cache; use a new cache directory")
        saved = json.loads(manifest.read_text())
        if saved["identity"] != self.key:
            raise ValueError("input cache identity mismatch; use a new directory")
        for name, digest in saved["files"].items():
            if sha256(self.directory / name) != digest:
                raise ValueError("input cache hash mismatch")
        with (self.directory / "inputs.pkl").open("rb") as stream:
            inputs = pickle.load(stream)
        arrays = {
            w: {
                n: np.load(
                    self.directory / f"{w}-{n}.npy", mmap_mode="c", allow_pickle=False
                )
                for n in ARRAYS
            }
            for w in WINDOWS
        }
        return inputs, arrays

    def save(self, inputs, encoded):
        # Exclusive creation rejects concurrent writers and accidental overwrite.
        # The manifest is written last; interrupted writes cannot look complete.
        self.directory.mkdir(parents=True, exist_ok=False)
        with (self.directory / "inputs.pkl").open("wb") as stream:
            pickle.dump(inputs, stream)
        for window in WINDOWS:
            for name in ARRAYS:
                np.save(
                    self.directory / f"{window}-{name}.npy",
                    encoded[window][name],
                    allow_pickle=False,
                )
        files = {p.name: sha256(p) for p in self.directory.iterdir()}
        (self.directory / "manifest.json").write_text(
            json.dumps(dict(identity=self.key, files=files), indent=2) + "\n"
        )
