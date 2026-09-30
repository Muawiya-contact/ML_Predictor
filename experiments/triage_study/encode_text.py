"""Compute frozen offline embeddings once; no target labels enter this process."""

import os

for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(k, "2")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
from study_paths import OUTPUT as HERE, MODEL_CACHE
from cache_integrity import file_sha256

MODELS = {
    "sapbert_concept": (
        "Clinical_Concept",
        MODEL_CACHE / "models--cambridgeltl--SapBERT-from-PubMedBERT-fulltext",
        "cls",
        64,
    ),
    "mpnet_concept": (
        "Clinical_Concept",
        MODEL_CACHE / "models--sentence-transformers--all-mpnet-base-v2",
        "sentence_transformer",
        128,
    ),
    "minilm_complaint": (
        "chief_complaint",
        MODEL_CACHE
        / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
        "sentence_transformer",
        128,
    ),
}


def main(name):
    col, cache, pooling, maxlen = MODELS[name]
    snapshots = list((cache / "snapshots").iterdir())
    ref = cache / "refs/main"
    revision = ref.read_text().strip() if ref.exists() else snapshots[0].name
    modelpath = cache / "snapshots" / revision
    d = pd.read_csv(HERE / "dataset_with_splits.csv")
    texts = d[col].fillna("").astype(str).tolist()
    fingerprint = hashlib.sha256(
        json.dumps(texts, ensure_ascii=False).encode()
    ).hexdigest()
    outfile = HERE / f"emb_{name}.npy"
    meta = HERE / f"emb_{name}.json"
    expected = {
        "name": name,
        "text_column": col,
        "revision": revision,
        "pooling": pooling,
        "max_token_length": maxlen,
        "normalized": True,
        "text_sha256": fingerprint,
        "shape": [len(texts), 384 if name == "minilm_complaint" else 768],
    }
    if outfile.exists() or meta.exists():
        try:
            saved = json.loads(meta.read_text())
            array = np.load(outfile, mmap_mode="r", allow_pickle=False)
            matches = all(saved.get(k) == v for k, v in expected.items())
            matches = matches and list(array.shape) == expected["shape"]
            matches = matches and saved.get("array_sha256") == file_sha256(outfile)
        except (OSError, ValueError, KeyError):
            matches = False
        if not matches:
            raise ValueError(
                "Embedding cache does not match the requested encoder or is incomplete. "
                "Use a fresh TRIAGE_STUDY_OUTPUT directory; existing results were not changed."
            )
        print(name, "already complete (encoder and array verified)")
        return
    import torch

    torch.set_num_threads(2)
    unique = list(dict.fromkeys(texts))
    indices = {t: i for i, t in enumerate(unique)}
    n = len(unique)
    start = time.monotonic()
    chunkdir = HERE / f"cache_{name}"
    chunkdir.mkdir(exist_ok=True)
    if pooling == "cls":
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(str(modelpath), local_files_only=True)
        model = AutoModel.from_pretrained(str(modelpath), local_files_only=True)
        model.eval()
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(modelpath), device="cpu", local_files_only=True)
        model.max_seq_length = maxlen
    vectors = []
    for begin in range(0, n, 128):
        batch = unique[begin : begin + 128]
        key = hashlib.sha256(
            json.dumps([revision, pooling, maxlen, batch]).encode()
        ).hexdigest()
        part = chunkdir / f"{key}.npy"
        if part.exists():
            z = np.load(part)
        elif pooling == "cls":
            chunks = []
            with torch.inference_mode():
                for j in range(0, len(batch), 16):
                    inputs = tokenizer(
                        batch[j : j + 16],
                        padding=True,
                        truncation=True,
                        max_length=maxlen,
                        return_tensors="pt",
                    )
                    chunks.append(
                        model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
                    )
            z = np.vstack(chunks)
            z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-9)
            np.save(part, z)
        else:
            z = model.encode(
                batch,
                batch_size=16,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            np.save(part, z)
        vectors.append(z)
        print(
            f"{name}: {min(begin+128,n)}/{n} unique texts, elapsed {time.monotonic()-start:.0f}s",
            flush=True,
        )
    full = np.vstack(vectors)[[indices[t] for t in texts]].astype("float32")
    np.save(outfile, full)
    meta.write_text(
        json.dumps(
            {
                "name": name,
                "text_column": col,
                "model_path": str(modelpath),
                "revision": revision,
                "pooling": pooling,
                "max_token_length": maxlen,
                "normalized": True,
                "text_sha256": fingerprint,
                "shape": list(full.shape),
                "array_sha256": file_sha256(outfile),
                "unique_texts": n,
                "seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )
    print("Completed", name, full.shape, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=MODELS)
    main(p.parse_args().name)
