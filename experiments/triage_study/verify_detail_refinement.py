"""Verify every refinement OOF score before any retrospective comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from four_level_study import scores
from cache_integrity import file_sha256


def verify(source, original, incumbent):
    protocol=json.loads((source/'protocol.json').read_text())
    for name,key in [('dataset_with_splits.csv','source_data_sha256'),('emb_sapbert_concept.npy','embedding_sha256')]:
        assert file_sha256(original/name)==protocol[key]
    frame=pd.read_csv(original/'dataset_with_splits.csv');dev=frame[frame.partition.eq('development')]
    test=frame[frame.partition.eq('test')]
    assert not set(dev.group)&set(test.group)
    cv=pd.read_csv(source/'cross_validation.csv');z=np.load(source/'oof.npz')
    assert len(cv)==len(protocol['candidates'])*5 and not cv.duplicated(['candidate','fold']).any()
    assert set(cv.candidate)==set(protocol['candidates'])
    np.testing.assert_array_equal(z['row_id'],dev.row_id)
    for name in protocol['candidates']:
        p=z[name];assert p.shape==(len(dev),4) and np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-10)
        for fold in range(5):
            mask=dev.cv5.eq(fold).to_numpy()
            assert not set(dev.loc[mask,'group'])&set(dev.loc[~mask,'group'])
            row=cv[cv.candidate.eq(name)&cv.fold.eq(fold+1)].iloc[0]
            for key,value in scores(dev.Labels.to_numpy()[mask],p[mask].argmax(1)).items():
                np.testing.assert_allclose(row[key],value,atol=1e-12)
    summary=cv.groupby('candidate').agg(macro_f1=('macro_f1','mean'),accuracy=('accuracy','mean'),emergency_recall=('emergency_recall','mean')).reset_index()
    old=json.loads((incumbent/'selection.json').read_text());oldcv=pd.read_csv(incumbent/'cv_summary.csv')
    threshold=float(oldcv[oldcv.candidate.eq(old['baseline'])].emergency_recall.iloc[0]-.01)
    eligible=summary[summary.emergency_recall.ge(threshold)].sort_values(['macro_f1','accuracy','candidate'],ascending=[False,False,True])
    best=eligible.iloc[0] if len(eligible) else None
    result=dict(status='passed',fits=len(cv),source_hashes_verified=True,group_separation_verified=True,all_oof_metrics_verified=True,test_rows_used=False,
                incumbent_cv_macro_f1=float(oldcv[oldcv.candidate.eq(old['candidate'])].macro_f1.iloc[0]),minimum_emergency_recall=threshold,
                best_eligible=best.to_dict() if best is not None else None)
    if best is not None:
        prior=np.load(incumbent/'development_oof.npz')
        np.testing.assert_array_equal(prior['row_id'],dev.row_id)
        groups,_=pd.factorize(dev.group,sort=True)
        y=dev.Labels.to_numpy()
        matrices=[]
        for prediction in (z[best.candidate].argmax(1),prior['selected'].argmax(1)):
            grouped=np.zeros((groups.max()+1,4,4),dtype=int)
            np.add.at(grouped,(groups,y,prediction),1)
            matrices.append(grouped)
        def macro_f1(cm):
            den=cm.sum(0)+cm.sum(1)
            return np.divide(2*np.diag(cm),den,out=np.zeros(4,dtype=float),where=den!=0).mean()
        rng=np.random.default_rng(20261001);gains=[]
        for _ in range(1000):
            sampled=rng.integers(0,len(matrices[0]),len(matrices[0]))
            gains.append(macro_f1(matrices[0][sampled].sum(0))-macro_f1(matrices[1][sampled].sum(0)))
        result['paired_bootstrap_vs_incumbent']=dict(iterations=1000,seed=20261001,
            pooled_macro_f1_gain=float(macro_f1(matrices[0].sum(0))-macro_f1(matrices[1].sum(0))),
            ci95=np.quantile(gains,[.025,.975]).tolist(),
            interpretation='Conditional paired group interval; excludes repeated model-selection uncertainty.')
    (source/'verification.json').write_text(json.dumps(result,indent=2));summary.to_csv(source/'cv_summary.csv',index=False)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ['source','original','incumbent']:parser.add_argument('--'+flag,type=Path,required=True)
    a=parser.parse_args();verify(a.source,a.original,a.incumbent)
