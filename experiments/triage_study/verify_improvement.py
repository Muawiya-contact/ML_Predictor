"""Independently verify second-round selection, predictions and paired uncertainty."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
from improve_four_level import rank
from cache_integrity import file_sha256

def verify(source,original):
    protocol=json.loads((source/'protocol.json').read_text());selection=json.loads((source/'selection.json').read_text())
    assert file_sha256(original/'dataset_with_splits.csv')==protocol['source_data_sha256']
    assert file_sha256(original/'emb_sapbert_concept.npy')==protocol['embedding_sha256']
    cv=pd.read_csv(source/'cross_validation.csv');summary=pd.read_csv(source/'cv_summary.csv')
    assert len(cv)==5*len(protocol['candidates']) and not cv.duplicated(['candidate','fold']).any()
    assert set(cv.candidate)==set(protocol['candidates'])
    for row in summary.itertuples():
        part=cv[cv.candidate.eq(row.candidate)]
        assert set(part.fold)=={1,2,3,4,5}
        for target,field in [('macro_f1','macro_f1'),('accuracy','accuracy'),('emergency_recall','emergency_recall')]:
            np.testing.assert_allclose(getattr(row,target),part[field].mean(),atol=1e-12)
    baseline=summary.set_index('candidate').loc[protocol['baseline']]
    assert rank(summary,baseline).iloc[0].candidate==selection['candidate']
    old_cv=pd.read_csv(original/'cross_validation.csv').query("candidate == 'logreg_c10'").sort_values('fold')
    current_cv=cv[cv.candidate.eq(protocol['baseline'])].sort_values('fold')
    np.testing.assert_allclose(old_cv.macro_f1,current_cv.macro_f1,atol=1e-12)
    df=pd.read_csv(original/'dataset_with_splits.csv');dev=df[df.partition.eq('development')];test=df[df.partition.eq('test')]
    assert not set(dev.group)&set(test.group)
    for fold in range(5):assert not set(dev[dev.cv5.eq(fold)].group)&set(dev[dev.cv5.ne(fold)].group)
    z=np.load(source/'development_oof.npz');np.testing.assert_array_equal(z['row_id'],dev.row_id);np.testing.assert_array_equal(z['reference'],dev.Labels)
    for name in ['selected','baseline']:
        assert np.isfinite(z[name]).all();np.testing.assert_allclose(z[name].sum(1),1,atol=1e-8)
    results=json.loads((source/'retrospective_results.json').read_text())
    for r in results:
        pred=pd.read_csv(source/(r['candidate']+'_retrospective_predictions.csv'))
        np.testing.assert_array_equal(pred.row_id,test.row_id);np.testing.assert_array_equal(pred.reference,test.Labels)
        np.testing.assert_allclose(r['metrics']['accuracy'],accuracy_score(pred.reference,pred.predicted))
        np.testing.assert_allclose(r['metrics']['macro_f1'],f1_score(pred.reference,pred.predicted,average='macro'))
        np.testing.assert_array_equal(r['confusion'],confusion_matrix(pred.reference,pred.predicted,labels=[0,1,2,3]))
    # Paired grouped bootstrap of OOF predictions: diagnostic, not independent confirmation.
    groups=dev.group.to_numpy();unique=np.unique(groups);rows=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(2026)
    gains=[];y=z['reference'];a=z['selected'].argmax(1);b=z['baseline'].argmax(1)
    for _ in range(1000):
        ids=np.concatenate([rows[i] for i in rng.integers(0,len(rows),len(rows))])
        gains.append(f1_score(y[ids],a[ids],average='macro')-f1_score(y[ids],b[ids],average='macro'))
    output=dict(status='passed',cv_fits=len(cv),baseline_reproduced=True,source_hashes_verified=True,holdout_previously_exposed=True,
        development_oof_paired_group_bootstrap=dict(iterations=1000,seed=2026,macro_f1_gain=float(f1_score(y,a,average='macro')-f1_score(y,b,average='macro')),ci95=np.quantile(gains,[.025,.975]).tolist(),interpretation='Conditional on selected OOF predictions; does not account for model-selection optimism.'))
    (source/'verification.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--original',type=Path,required=True);a=p.parse_args();verify(a.source,a.original)
