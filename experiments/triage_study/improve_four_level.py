"""Second-round development-only search; previous test set is retrospective.

Labels, encoder and original group partitions are immutable. PCA, imputation
and scaling fit on each training fold. No holdout scores enter selection.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import research_engine as engine
from four_level_study import scores, verify_embeddings
from cache_integrity import file_sha256
import hashlib


def candidates():
    configs = {}
    for pc in (64, 128, 256):
        for c in (10, 100, 1000):
            for balanced in (True, False):
                name = f'lr_pca{pc}_c{c}_balanced{int(balanced)}'
                configs[name] = engine.config('logreg', dict(view='fused', encoder='sapbert_concept', pca=pc, solver='full'), C=c, balance=balanced)
        configs[f'lr_pca{pc}_whiten'] = engine.config('logreg', dict(view='fused', encoder='sapbert_concept', pca=pc, solver='full', whiten=True), C=.1, balance=True)
    for pc in (0, 64, 128):
        view = 'structured' if pc == 0 else 'fused'
        for leaves in (7, 15):
            configs[f'hgb_{view}_{pc}_leaf{leaves}'] = engine.config('hgb', dict(view=view, encoder='sapbert_concept', pca=pc, solver='full'), max_iter=400, learning_rate=.05, max_leaf_nodes=leaves, l2_regularization=5, early_stopping=False, balance=True)
    for pc in (0, 64):
        view = 'structured' if pc == 0 else 'fused'
        configs[f'rf_{view}_{pc}'] = engine.config('rf', dict(view=view, encoder='sapbert_concept', pca=pc, solver='full'), n_estimators=300, min_samples_leaf=5, max_features=.8, balance=True)
    return configs


def rank(summary, baseline):
    # Preserve emergency recall within one percentage point of the incumbent.
    eligible = summary[summary.emergency_recall >= baseline.emergency_recall - .01]
    return eligible.sort_values(['macro_f1', 'accuracy', 'candidate'], ascending=[False, False, True])


def evaluate_fold(source, fold, configs):
    """A separate process owns its fold transforms; at most three run at once."""
    engine.HERE=Path(source)
    df=pd.read_csv(engine.HERE/'dataset_with_splits.csv')
    dev=df.partition.eq('development').to_numpy(); y=df.Labels.to_numpy(dtype=int)
    train=np.flatnonzero(dev & df.cv5.ne(fold).to_numpy());valid=np.flatnonzero(dev & df.cv5.eq(fold).to_numpy())
    if set(df.iloc[train].group)&set(df.iloc[valid].group):raise ValueError('Group overlap')
    feature_groups={}
    for name,cfg in configs.items():feature_groups.setdefault(json.dumps(cfg['features'],sort_keys=True),[]).append(name)
    records=[];predictions={}
    for key,names in feature_groups.items():
        transform=engine.Features(**json.loads(key)).fit(df.iloc[train])
        a,b=transform.transform(df.iloc[train]),transform.transform(df.iloc[valid])
        for name in names:
            start=time.monotonic();model=engine.fit_model(configs[name],a,y[train]);p=model.predict_proba(b)
            np.testing.assert_array_equal(model.classes_,[0,1,2,3]);predictions[name]=p
            row=dict(candidate=name,fold=fold+1,seconds=time.monotonic()-start,**scores(y[valid],p.argmax(1)));records.append(row)
            print(f"fold={fold+1} {name} F1={row['macro_f1']:.4f} seconds={row['seconds']:.1f}",flush=True)
    return records,valid,predictions


def run(source, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh output directory.')
    source = source.resolve(); engine.HERE = source
    plan = json.loads((source/'comparison_plan.json').read_text())
    data_path = source/'dataset_with_splits.csv'
    if file_sha256(data_path) != plan['source_data_sha256']:
        raise ValueError('Original dataset or splits changed.')
    df = pd.read_csv(data_path)
    texts = df.Clinical_Concept.fillna('').astype(str).tolist()
    fingerprint = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    meta = json.loads((source/'emb_sapbert_concept.json').read_text())
    verify_embeddings(meta, source/'emb_sapbert_concept.npy', fingerprint, len(df))
    configs = candidates()
    baseline_name = 'lr_pca64_c10_balanced1'
    protocol = dict(candidates=configs, baseline=baseline_name,
        selection='Highest mean development macro F1 subject to emergency recall no more than 1 percentage point below incumbent; accuracy breaks ties.',
        evaluation='Previously examined holdout: retrospective comparison, not a fresh independent test.',
        source_data_sha256=file_sha256(data_path), embedding_sha256=file_sha256(source/'emb_sapbert_concept.npy'),
        encoder_finetuned=False, labels_changed=False, seed=42)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    dev = df.partition.eq('development').to_numpy(); y=df.Labels.to_numpy(dtype=int)
    records=[]; probs={name:np.full((len(df),4),np.nan) for name in configs}
    # Audit contradictory identical model inputs using development data only.
    cols=['Clinical_Concept']+engine.NUM[:6]+engine.CAT+['AVPU']
    conflicts=df.loc[dev].groupby(cols, dropna=False).Labels.nunique()
    (output/'input_audit.json').write_text(json.dumps(dict(development_rows=int(dev.sum()), identical_input_groups_with_multiple_labels=int((conflicts>1).sum()), labels_modified=False), indent=2))
    feature_groups={}
    for name,cfg in configs.items():
        key=json.dumps(cfg['features'],sort_keys=True)
        feature_groups.setdefault(key,[]).append(name)
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(evaluate_fold, str(source), fold, configs) for fold in range(5)]
        for job in as_completed(jobs):
            rows, valid, predictions=job.result()
            records.extend(rows)
            for name,p in predictions.items():probs[name][valid]=p
            pd.DataFrame(records).sort_values(['fold','candidate']).to_csv(output/'cross_validation.csv',index=False)
            print(f'Completed {len(records)} of {5*len(configs)} fits',flush=True)
    summary=pd.DataFrame(records).groupby('candidate').agg(macro_f1=('macro_f1','mean'),f1_std=('macro_f1','std'),accuracy=('accuracy','mean'),precision=('precision_macro','mean'),recall=('recall_macro','mean'),emergency_recall=('emergency_recall','mean'),under_triage=('under_triage_rate','mean')).reset_index()
    base=summary.set_index('candidate').loc[baseline_name]
    ordered=rank(summary,base); chosen=ordered.iloc[0].candidate
    summary.sort_values('macro_f1',ascending=False).to_csv(output/'cv_summary.csv',index=False)
    selection=dict(candidate=chosen,config=configs[chosen],baseline=baseline_name,rule=protocol['selection'],cv_gain=float(ordered.iloc[0].macro_f1-base.macro_f1))
    (output/'selection.json').write_text(json.dumps(selection,indent=2))
    np.savez_compressed(output/'development_oof.npz',row_id=df.loc[dev,'row_id'].to_numpy(),reference=y[dev],selected=probs[chosen][dev],baseline=probs[baseline_name][dev])
    review=df.loc[dev].copy();review['oof_predicted']=probs[chosen][dev].argmax(1);review['oof_confidence']=probs[chosen][dev].max(1)
    review[review.Labels.ne(review.oof_predicted)].to_csv(output/'development_errors_for_review.csv',index=False)
    print('FROZEN SELECTION',selection,flush=True)
    # No search or tuning follows this retrospective comparison.
    tr=np.flatnonzero(dev);te=np.flatnonzero(~dev)
    if set(df.iloc[tr].group)&set(df.iloc[te].group):raise ValueError('Test group overlap')
    result=[]
    for name in dict.fromkeys([baseline_name,chosen]):
        transform=engine.Features(**configs[name]['features']).fit(df.iloc[tr]);model=engine.fit_model(configs[name],transform.transform(df.iloc[tr]),y[tr]);p=model.predict_proba(transform.transform(df.iloc[te]));pred=p.argmax(1)
        result.append(dict(candidate=name,metrics=scores(y[te],pred),report=classification_report(y[te],pred,labels=[0,1,2,3],output_dict=True,zero_division=0),confusion=confusion_matrix(y[te],pred,labels=[0,1,2,3]).tolist()))
        pd.DataFrame(dict(row_id=df.iloc[te].row_id,reference=y[te],predicted=pred)).to_csv(output/f'{name}_retrospective_predictions.csv',index=False)
        if name==chosen:
            dest=output/'serving';dest.mkdir()
            for filename,value in [('model',model),('structured',transform.structured_),('pca',transform.pca_)]:joblib.dump(value,dest/f'{filename}.pkl')
    (output/'retrospective_results.json').write_text(json.dumps(result,indent=2))
    assert file_sha256(data_path)==protocol['source_data_sha256']
    assert file_sha256(source/'emb_sapbert_concept.npy')==protocol['embedding_sha256']
    print('COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.source,args.output)
