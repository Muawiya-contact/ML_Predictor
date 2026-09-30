"""Fail closed when a resumed study no longer has its original inputs.

Existing experiments without a manifest must be run in a fresh directory. We
never attach a new identity to old scores, or silently invalidate final results.
"""

import hashlib
import json
from pathlib import Path
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_study(root):
    """Freeze/verify CSVs, split assignments and every encoder artifact before reuse."""
    root = Path(root)
    names = ["trimmed_dataset.csv", "dataset_with_splits.csv", "split_manifest.csv"]
    for name in ("sapbert_concept", "mpnet_concept", "minilm_complaint"):
        array = root / f"emb_{name}.npy"
        metadata = root / f"emb_{name}.json"
        if array.exists() or metadata.exists():
            if not array.exists() or not metadata.exists():
                raise ValueError(
                    f"Incomplete embeddings for {name}; use a fresh study directory."
                )
            saved = json.loads(metadata.read_text())
            actual = np.load(array, mmap_mode="r", allow_pickle=False)
            if list(actual.shape) != saved.get("shape") or saved.get(
                "array_sha256"
            ) != file_sha256(array):
                raise ValueError(
                    f"Unverified embeddings for {name}; use a fresh study directory."
                )
            names.extend([array.name, metadata.name])
    identity = {
        "version": 1,
        "files": {name: file_sha256(root / name) for name in names},
    }
    manifest = root / "study_inputs.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Study inputs changed after freezing; use a fresh study directory."
            )
    else:
        prior = [
            "matrix_cache",
            "cv3",
            "cv5",
            "frozen_selection.json",
            "final_results.json",
            "models",
        ]
        if any((root / name).exists() for name in prior):
            raise ValueError(
                "Existing results have no input manifest; use a fresh study directory."
            )
        # Exclusive creation prevents two initializers from replacing each other's identity.
        try:
            with manifest.open("x") as out:
                json.dump(identity, out, indent=2)
        except FileExistsError:
            if json.loads(manifest.read_text()) != identity:
                raise ValueError(
                    "Concurrent study inputs differ; use a fresh study directory."
                )
    return identity
