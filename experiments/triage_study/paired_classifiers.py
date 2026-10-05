"""Compare six classifiers using frozen SapBERT on concepts plus complaints.

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
    candidates={}
    for pc in (64,128):
        for c in (1,10,100):
            cfg=engine.config('logreg',dict(view='fused',encoder='sapbert_pair',solver='full',pca=pc,polynomial=True),C=c,balance=True)
            cfg['text_details']=dict(source='complaint_details',version=VERSION,feature_names=copy.deepcopy(FEATURE_NAMES))
            candidates[f'lr_pair{pc}_c{c}']=cfg
    return candidates


def run(original, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh refinement destination')
    from importlib.metadata import version
    import platform
    environment = dict(python=platform.python_version(),
                       packages={name: version(name) for name in
                                 ('numpy', 'pandas', 'scikit-learn', 'catboost', 'xgboost-cpu')},
                       workers=2, threads_per_worker=2, device='cpu')
    (output / 'environment.json').write_text(json.dumps(environment, indent=2))
    candidates = configs()
    plan = json.loads((original / 'comparison_plan.json').read_text())
    data_hash = file_sha256(original / 'dataset_with_splits.csv')
    if data_hash != plan['source_data_sha256']:
        raise ValueError('Source data or split changed')
    embedding_hash = file_sha256(original / 'emb_sapbert_concept.npy')
    protocol = dict(candidates=candidates, source_data_sha256=data_hash,
                    embedding_sha256=embedding_hash, labels_changed=False,
                    encoder_finetuned=False, seed=42,
                    target='Accuracy, macro precision, macro recall and macro F1 >= 0.90; target is assessed, never fabricated.',
                    selection='Combine all rounds; unchanged original emergency-recall constraint and development-only ranking.',
                    supplement='Six predeclared paired-text LR candidates; frozen SapBERT now receives concept + [SEP] + original complaint with 128-token truncation. All four labels and partitions unchanged.')
    protocol['additional_embedding_sha256'] = file_sha256(original / 'emb_sapbert_pair.npy')
    protocol['additional_embedding_metadata'] = json.loads((original / 'emb_sapbert_pair.json').read_text())
    metadata = protocol['additional_embedding_metadata']
    if metadata['source_data_sha256'] != data_hash or metadata['array_sha256'] != protocol['additional_embedding_sha256']:
        raise ValueError('Paired embedding cache does not match its source or metadata')
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
    assert file_sha256(original / 'emb_sapbert_pair.npy') == protocol['additional_embedding_sha256']
    np.savez_compressed(output / 'oof.npz', row_id=df.loc[dev, 'row_id'].to_numpy(),
                        **{name: p[dev] for name, p in probs.items()})
    print(f'COMPLETE {len(candidates)*5} expanded classifier fits', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.original, args.output)
