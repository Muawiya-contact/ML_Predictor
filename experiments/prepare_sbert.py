"""Encode the existing English translations locally with a frozen 768-D SBERT.
Only model files are downloaded. Dataset text is never sent to a remote API.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='cardiac_english_2252.csv', type=Path)
    parser.add_argument('--output', default='output/results/sbert_inputs', type=Path)
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--revision', help='Pinned model commit; resolved and recorded if omitted')
    args = parser.parse_args()
    frame = pd.read_csv(args.data)
    text = frame['English_Translation']
    if text.isna().any() or text.str.strip().eq('').any():
        raise ValueError('English translations must not be missing or blank.')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose an empty output directory.')
    revision = args.revision or HfApi().model_info(args.model).sha
    torch.set_num_threads(2)
    model = SentenceTransformer(args.model, revision=revision, device='cpu', trust_remote_code=False)
    if model.get_embedding_dimension() != 768:
        raise ValueError('This experiment requires a 768-dimensional checkpoint.')
    # Encode repeated text once, then restore the original labelled row order.
    unique = text.drop_duplicates().tolist()
    encoded = model.encode(unique, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    lookup = dict(zip(unique, encoded))
    embeddings = np.stack([lookup[value] for value in text])
    ids = np.array([f'cardiac-{i:05d}' for i in range(len(frame))])
    frame.insert(0, 'row_id', ids)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'labelled_rows.csv', index=False)
    np.savez_compressed(args.output / 'embeddings.npz', embeddings=embeddings, row_ids=ids)
    metadata = dict(model=args.model, revision=revision, dimensions=768, normalized=True,
                    source=str(args.data), source_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    text_column='English_Translation', rows=len(frame), unique_texts=len(unique),
                    pooling='checkpoint-defined mean pooling', max_sequence_length=model.max_seq_length,
                    provenance='Existing synthetic cardiac dataset; historical English translations; no new translation or fine-tuning.')
    (args.output / 'embedding_manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
