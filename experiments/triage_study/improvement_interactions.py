"""Predeclared supplemental CV for learned quadratic vital-sign interactions.

No medical thresholds or replacement labels are introduced. Polynomial numeric
terms precede scaling inside each training fold. The original 29 configurations
remain available; the final selection compares all 35 on the same folds.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'2')
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import research_engine as engine
from improve_four_level import evaluate_fold
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
from cache_integrity import file_sha256

def configs():
    return {f'lr_quadratic64_c{c}_balanced{int(balance)}':engine.config('logreg',dict(view='fused',encoder='sapbert_concept',pca=64,solver='full',polynomial=True),C=c,balance=balance) for c in [1,10,100] for balance in [True,False]}

def run(original,output):
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use a fresh supplemental output')
    candidates=configs()
    (output/'protocol.json').write_text(json.dumps(dict(candidates=candidates,source_data_sha256=file_sha256(original/'dataset_with_splits.csv'),selection='Combine with original 29 candidates; same development ranking and emergency recall constraint.',labels_changed=False),indent=2))
    df=pd.read_csv(original/'dataset_with_splits.csv');dev=df.partition.eq('development');records=[];probs={name:np.full((len(df),4),np.nan) for name in candidates}
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate_fold,str(original.resolve()),fold,candidates) for fold in range(5)]
        for job in as_completed(jobs):
            rows,indices,pred=job.result();records.extend(rows)
            for name,p in pred.items():probs[name][indices]=p
            pd.DataFrame(records).to_csv(output/'cross_validation.csv',index=False)
    np.savez_compressed(output/'oof.npz',row_id=df.loc[dev,'row_id'].to_numpy(),**{name:p[dev] for name,p in probs.items()})
    print('COMPLETE supplemental 30 fits',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.original,a.output)
