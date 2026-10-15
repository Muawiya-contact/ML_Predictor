"""Fixed complaint-detail refinement within the paper's classifier families.

Reuse immutable grouped folds and frozen SapBERT. Never select on the old test
set. All candidates, including unsuccessful trials, remain in the final table.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import sys
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import research_engine as engine
from cache_integrity import file_sha256
from improve_four_level import evaluate_fold


def configs():
    import copy
    from src.complaint_details import VERSION, FEATURE_NAMES
    base = dict(view='fused', encoder='sapbert_concept', solver='full', pca=128, polynomial=True)
    candidates = {}
    for c in (1, 10, 30, 300, 1000):
        candidates[f'lr_detail128_c{c}'] = engine.config('logreg', dict(base), C=c, balance=True)
    candidates['lr_detail128_unbalanced'] = engine.config('logreg', dict(base), C=100, balance=False)
    for pc in (64, 256):
        candidates[f'lr_detail{pc}_c100'] = engine.config('logreg', dict(base, pca=pc), C=100, balance=True)
    for leaves, l2 in ((7, 20), (15, 10), (31, 20)):
        candidates[f'hgb_detail64_leaf{leaves}_l2{l2}'] = engine.config('hgb', dict(base, pca=64, polynomial=False), max_iter=800, learning_rate=.05, max_leaf_nodes=leaves, min_samples_leaf=20, l2_regularization=l2, early_stopping=False, balance=True)
    for fraction in (.3, .7):
        candidates[f'rf_detail64_fraction{fraction}'] = engine.config('rf', dict(base, pca=64, polynomial=False), n_estimators=500, min_samples_leaf=2, max_features=fraction, balance=True)
    for config in candidates.values():
        config['text_details'] = dict(source='complaint_details', version=VERSION, feature_names=copy.deepcopy(FEATURE_NAMES))
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
                    supplement='Thirteen predeclared refinements after original-complaint details; original four labels and partitions unchanged.')
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
    print(f'COMPLETE {len(candidates)*5} detail refinement fits', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.original, args.output)
