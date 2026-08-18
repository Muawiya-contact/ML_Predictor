"""
src/encoders.py
=======================================================================
One embedding interface, two sources: precomputed vectors and live text.
=======================================================================

The whole point of the dual pipeline is that offline evaluation and live
inference must produce the SAME representation. If they drift, a model
validated at cross-validation time is not the model answering at the
keyboard, and every number in the benchmark stops meaning anything.

So both encoders below commit to the settings the precomputed vectors
were built with, read out of the baseline's own scripts rather than
guessed:

    model            intfloat/multilingual-e5-small   (384 dims)
    prefix           "passage: "  on every complaint
    normalization    L2 (normalize_embeddings=True)

The e5 prefix is not decoration. e5 models are trained with "query: " and
"passage: " markers, and dropping the prefix at inference while the
stored vectors have it shifts the whole embedding space - the classifier
would receive vectors from a different distribution than it was fitted
on, and would degrade quietly rather than fail.

StaticEncoder is offline and instant. DynamicEncoder downloads the model
once (a few hundred MB) and is offline afterwards.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

#: Settings the precomputed .npy vectors were produced with. Changing any
#: of these invalidates every stored embedding in professor_baseline/.
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
E5_PREFIX = "passage: "


class BaseEncoder:
    """Common shape contract, so callers can swap encoders blindly."""

    dim = EMBEDDING_DIM

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def _check(self, mat: np.ndarray, n: int) -> np.ndarray:
        if mat.ndim != 2 or mat.shape != (n, self.dim):
            raise ValueError(
                f"{type(self).__name__} produced {mat.shape}, expected "
                f"({n}, {self.dim}). Static and dynamic vectors must match "
                f"or the classifier is being fed a different space than it "
                f"was trained on.")
        return mat.astype(np.float32, copy=False)


class StaticEncoder(BaseEncoder):
    """Serve precomputed vectors from a .npy file.

    Used for cross-validation, where re-encoding 185 complaints on every
    fold would be slow and would also let a model change slip in
    unnoticed between runs. Row order must match the dataset row order -
    that is the entire contract, and it is checked rather than assumed.
    """

    def __init__(self, npy_path: str, expected_rows: int | None = None):
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"No embedding file at {npy_path}")
        self.path = npy_path
        self.matrix = np.load(npy_path, allow_pickle=False)
        if self.matrix.size == 0:
            raise ValueError(
                f"{npy_path} is empty (shape {self.matrix.shape}). Two files "
                f"in the supplied baseline are 128-byte headers with no data; "
                f"this is one of them.")
        if self.matrix.ndim != 2 or self.matrix.shape[1] != self.dim:
            raise ValueError(
                f"{npy_path} has shape {self.matrix.shape}, expected "
                f"(n, {self.dim}).")
        if expected_rows is not None and len(self.matrix) != expected_rows:
            raise ValueError(
                f"{npy_path} has {len(self.matrix)} rows but the dataset has "
                f"{expected_rows}. Row i of the matrix must be row i of the "
                f"CSV; a mismatch means the labels belong to other vectors.")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Ignores `texts` by design - returns the stored matrix.

        Kept on the same interface so baseline.py and predict.py can hold
        an encoder without caring which kind it is.
        """
        return self._check(self.matrix, len(self.matrix))

    def __repr__(self):
        return f"StaticEncoder({os.path.basename(self.path)}, {self.matrix.shape})"


class DynamicEncoder(BaseEncoder):
    """Encode raw text live, reproducing the static vectors' settings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None          # loaded on first use, not at import

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is required for live encoding. "
                    "Install it, or use StaticEncoder for offline work."
                ) from e
            try:
                # Prefer the local cache so a machine with no network still
                # works once the model has been fetched once.
                self._model = SentenceTransformer(
                    self.model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        texts = ["" if t is None else str(t) for t in texts]
        # Empty text would embed to a meaningless point that happens to sit
        # somewhere definite, and the classifier would answer confidently
        # from it. Give it a marker instead so the row is at least honest.
        prepared = [E5_PREFIX + (t.strip() or "empty complaint") for t in texts]
        mat = self.model.encode(
            prepared, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False)
        return self._check(np.asarray(mat), len(texts))

    def __repr__(self):
        return f"DynamicEncoder({self.model_name}, dim={self.dim})"


def get_encoder(kind: str = "static", **kwargs) -> BaseEncoder:
    """Factory: 'static' needs npy_path, 'dynamic' takes model_name."""
    kind = kind.lower()
    if kind == "static":
        return StaticEncoder(**kwargs)
    if kind == "dynamic":
        return DynamicEncoder(**kwargs)
    raise ValueError(f"Unknown encoder kind {kind!r}; use 'static' or 'dynamic'.")
