"""
src/embedding_pipeline.py
=======================================================================
Roman Urdu -> local translation -> cleaned English -> 384-dim vector.
=======================================================================

    raw Roman Urdu
        |  translate    Ollama on localhost, temperature 0
        v
    clinical English
        |  normalize    lowercase, strip punctuation, drop stop words
        v
    cleaned English
        |  encode       MiniLM-L12-v2 from the local HF cache
        v
    384-dim L2-normalised vector

Every stage is local: Ollama is a service on 127.0.0.1 and the encoder
loads from the on-disk cache. No stage reaches the public internet.

READ THIS BEFORE FEEDING THESE VECTORS TO A CLASSIFIER
------------------------------------------------------
This module uses paraphrase-multilingual-MiniLM-L12-v2, which is what the
cluster analysis is specified against. The RandomForest heads in
models_src/ were fitted on intfloat/multilingual-e5-small with a
"passage: " prefix. BOTH MODELS EMIT 384 DIMENSIONS, so a vector from
here will load into those classifiers, predict, and be wrong, with
nothing to signal it.

These vectors are for SIMILARITY ANALYSIS ONLY. embed_for_classifier()
below exists for the other job and reads its encoder from the models_src
manifest, so the two can never be confused by accident.
"""

from __future__ import annotations

import re
import string
from typing import Optional, Sequence

import numpy as np

from src.offline_pipeline import (OLLAMA_URL, ollama_available,
                                  select_translation_model,
                                  translate_roman_urdu)

CLUSTER_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

#: Standard English function words. Deliberately short and generic - this
#: is not the learned clinical stop-word list from stopwords.py, which is
#: derived from the data and protects medical vocabulary. Removing words
#: here is a formatting step before similarity, not a modelling decision.
ENGLISH_STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "them", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "a", "an", "the", "and", "but", "or", "if", "of", "at",
    "by", "for", "with", "about", "into", "to", "from", "in", "on", "so",
    "than", "then", "there", "here", "when", "while", "as", "also",
}

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")

_encoder_cache = {}


def _load_encoder(name: str):
    """Load once, from the local cache where possible."""
    if name in _encoder_cache:
        return _encoder_cache[name]
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(name, device="cpu", local_files_only=True)
    except Exception:
        model = SentenceTransformer(name, device="cpu")
    _encoder_cache[name] = model
    return model


def normalize_english(text: str, drop_stop_words: bool = True) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop stop words.

    Never returns an empty string when the input had content: a complaint
    made entirely of stop words would otherwise embed to a meaningless
    point that the similarity maths would treat as a real position.
    """
    text = _PUNCT.sub(" ", str(text or "").lower())
    tokens = _WS.sub(" ", text).strip().split()
    if drop_stop_words:
        kept = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        if kept:
            tokens = kept
    return " ".join(tokens)


def embed_text(text: str, model_name: str = CLUSTER_ENCODER) -> np.ndarray:
    """One L2-normalised 384-dim vector. Raises only on a real load failure."""
    model = _load_encoder(model_name)
    vec = model.encode([text or "empty complaint"], convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)[0]
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(f"{model_name} returned {vec.shape}, "
                         f"expected ({EMBEDDING_DIM},)")
    return vec


def preprocess_and_embed(raw_roman_urdu: str, model: Optional[str] = None,
                         translate: bool = True,
                         drop_stop_words: bool = True) -> dict:
    """Full pipeline for one complaint.

    Returns a dict rather than a bare vector so a caller can see every
    intermediate stage - which is what makes the Pipeline Explorer view
    honest, and what makes a bad translation visible instead of buried.

    Never raises. `error` carries the reason and `embedding` falls back to
    the ORIGINAL text when translation fails, so a cluster analysis is not
    destroyed by one unreachable service.
    """
    out = {
        "raw": raw_roman_urdu,
        "translated": None,
        "normalized": None,
        "embedding": None,
        "l2_norm": None,
        "encoder": CLUSTER_ENCODER,
        "translated_ok": False,
        "error": None,
    }

    english = raw_roman_urdu
    if translate:
        try:
            if not ollama_available():
                out["error"] = (f"Ollama unreachable at {OLLAMA_URL} - embedding "
                                f"the untranslated text instead.")
            else:
                # preferred= must never be None: select_translation_model
                # calls .split(':') on it. Passing None here made every
                # sentence fall back to the untranslated text while the
                # error message pointed at a NoneType attribute, which is
                # a long way from "your model argument was empty".
                chosen = model or select_translation_model()
                if chosen is None:
                    out["error"] = ("No Ollama model installed - embedding the "
                                    "untranslated text instead.")
                else:
                    got = translate_roman_urdu(raw_roman_urdu, model=chosen)
                    if got:
                        english = got
                        out["translated"] = got
                        out["translated_ok"] = True
                        out["model"] = chosen
                    else:
                        out["error"] = (f"{chosen} returned nothing - embedding "
                                        f"the untranslated text instead.")
        except Exception as e:
            out["error"] = (f"Translation failed ({type(e).__name__}: {e}) - "
                            f"embedding the untranslated text instead.")

    out["normalized"] = normalize_english(english, drop_stop_words)
    try:
        vec = embed_text(out["normalized"])
        out["embedding"] = vec
        out["l2_norm"] = float(np.linalg.norm(vec))
    except Exception as e:
        out["error"] = f"Encoding failed: {type(e).__name__}: {e}"
    return out


def embed_for_classifier(text: str, model_dir: str = None) -> np.ndarray:
    """The OTHER encoder - the one models_src/ was actually fitted on.

    Separate function, and it reads the encoder name from the manifest
    rather than a constant, so the two 384-dim spaces in this project can
    never be swapped by accident.
    """
    import json
    import os
    from src.offline_pipeline import MODEL_DIR
    model_dir = model_dir or MODEL_DIR
    with open(os.path.join(model_dir, "manifest.json"), "r", encoding="utf-8") as f:
        enc = json.load(f)["encoder"]
    model = _load_encoder(enc["model"])
    prepared = enc.get("prefix", "") + (str(text).strip() or "empty complaint")
    vec = model.encode([prepared], convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(vec, dtype=np.float32)
