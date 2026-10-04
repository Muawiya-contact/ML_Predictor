"""Fixed third-round search restricted to the paper's three classifier families.

Reuse immutable grouped folds and frozen SapBERT. Never select on the old test
set. All candidates, including unsuccessful trials, remain in the final table.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import research_engine as engine
from cache_integrity import file_sha256
from improve_four_level import evaluate_fold


def configs():
    candidates = {}
    features = dict(view='fused', encoder='sapbert_concept', solver='full')
    for pc in (64, 128):
        for c in (30, 300):
            candidates[f'lr_refine{pc}_c{c}'] = engine.config(
                'logreg', dict(features, pca=pc), C=c, balance=True)
    for pc in (128, 256):
        for c in (100, 300):
            candidates[f'lr_quad{pc}_c{c}'] = engine.config(
                'logreg', dict(features, pca=pc, polynomial=True), C=c, balance=True)
    for leaves, iterations in ((7, 800), (31, 400)):
        candidates[f'hgb_refine64_leaf{leaves}'] = engine.config(
            'hgb', dict(features, pca=64), max_iter=iterations,
            learning_rate=0.05, max_leaf_nodes=leaves, min_samples_leaf=10,
            l2_regularization=5, early_stopping=False, balance=True)
    for pc in (64, 128):
        for leaf in (1, 3):
            candidates[f'rf_refine{pc}_leaf{leaf}'] = engine.config(
                'rf', dict(features, pca=pc), n_estimators=500,
                min_samples_leaf=leaf, max_features='sqrt', balance=True)
    return candidates


def run(original, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh refinement destination')
    candidates = configs()
    plan = json.loads((original / 'comparison_plan.json').read_text())
    data_hash = file_sha256(original / 'dataset_with_splits.csv')
    if data_hash != plan['source_data_sha256']:
        raise ValueError('Source data or split changed')
    embedding_hash = file_sha256(original / 'emb_sapbert_concept.npy')
    protocol = dict(candidates=candidates, source_data_sha256=data_hash,
                    embedding_sha256=embedding_hash, labels_changed=False,
                    encoder_finetuned=False, seed=42,
                    selection='Combine all rounds; unchanged original emergency-recall constraint and development-only ranking.',
                    supplement='Fourteen predeclared refinements within Logistic Regression, Random Forest and HistGradientBoosting.')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    df = pd.read_csv(original / 'dataset_with_splits.csv')
    dev = df.partition.eq('development')
    records = []
    probs = {name: np.full((len(df), 4), np.nan) for name in candidates}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(evaluate_fold, str(original.resolve()), fold, candidates) for fold in range(5)]
        for job in as_completed(jobs):
            rows, indices, predictions = job.result()
            records.extend(rows)
            for name, p in predictions.items():
                probs[name][indices] = p
            pd.DataFrame(records).to_csv(output / 'cross_validation.csv', index=False)
    assert file_sha256(original / 'dataset_with_splits.csv') == data_hash
    assert file_sha256(original / 'emb_sapbert_concept.npy') == embedding_hash
    np.savez_compressed(output / 'oof.npz', row_id=df.loc[dev, 'row_id'].to_numpy(),
                        **{name: p[dev] for name, p in probs.items()})
    print('COMPLETE 70 refinement fits', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.original, args.output)
