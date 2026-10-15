"""Verify all follow-up OOF predictions, freeze joint selection, then fit once."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'2')
import sys,json,copy,argparse,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report,confusion_matrix
from cache_integrity import file_sha256
from four_level_study import scores
from improve_four_level import rank
from finalize_detail_comparison import fit_candidate
import research_engine as engine


def run(root,output):
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use a fresh destination')
    original=root/'triage_four_level';main=root/'triage_four_level_round6'
    protocol=json.loads((main/'protocol.json').read_text())
    assert json.loads((main/'verification.json').read_text())['status']=='passed'
    assert file_sha256(original/'dataset_with_splits.csv')==protocol['source_data_sha256']
    assert file_sha256(original/'emb_sapbert_concept.npy')==protocol['embedding_sha256']
    frame=pd.read_csv(original/'dataset_with_splits.csv');dev=frame[frame.partition.eq('development')];test=frame[frame.partition.eq('test')]
    assert not set(dev.group)&set(test.group)
    previous_selection=json.loads((main/'selection.json').read_text())
    old=np.load(main/'development_oof.npz');np.testing.assert_array_equal(old['row_id'],dev.row_id)
    arrays={previous_selection['candidate']:old['selected']}
    records=[pd.read_csv(main/'cross_validation.csv')]
    inputs={}
    def check(name,p,cv):
        assert p.shape==(len(dev),4) and np.isfinite(p).all() and (p>=0).all()
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
        for fold in range(5):
            mask=dev.cv5.eq(fold).to_numpy()
            assert not set(dev.loc[mask,'group'])&set(dev.loc[~mask,'group'])
            row=cv[cv.candidate.eq(name)&cv.fold.eq(fold+1)].iloc[0]
            for key,value in scores(dev.Labels.to_numpy()[mask],p[mask].argmax(1)).items():np.testing.assert_allclose(row[key],value,atol=1e-12)
    for folder in ('triage_advanced_classifiers','triage_paired_classifiers'):
        path=root/folder;plan=json.loads((path/'protocol.json').read_text())
        assert plan['source_data_sha256']==protocol['source_data_sha256'] and plan['embedding_sha256']==protocol['embedding_sha256']
        if plan.get('additional_embedding_sha256'):
            assert file_sha256(original/'emb_sapbert_pair.npy')==plan['additional_embedding_sha256']
            assert plan['additional_embedding_metadata']['source_data_sha256']==protocol['source_data_sha256']
            protocol['additional_embedding_sha256']=plan['additional_embedding_sha256']
        inputs[folder]={n:file_sha256(path/n) for n in ('protocol.json','oof.npz','cross_validation.csv')}
        cv=pd.read_csv(path/'cross_validation.csv');z=np.load(path/'oof.npz');np.testing.assert_array_equal(z['row_id'],dev.row_id)
        assert len(cv)==len(plan['candidates'])*5 and not cv.duplicated(['candidate','fold']).any()
        for name,config in plan['candidates'].items():
            assert name not in protocol['candidates'];check(name,z[name],cv)
            protocol['candidates'][name]=config;arrays[name]=z[name]
        records.append(cv)
    path=root/'triage_refined_blends';plan=json.loads((path/'protocol.json').read_text());cv=pd.read_csv(path/'cross_validation.csv');z=np.load(path/'oof.npz')
    np.testing.assert_array_equal(z['row_id'],dev.row_id)
    sources={}
    for name,(relative,key) in plan['sources'].items():
        assert file_sha256(root/relative)==plan['source_hashes'][name]
        other=np.load(root/relative);np.testing.assert_array_equal(other['row_id'],dev.row_id);sources[name]=other[key]
    base_names={'lr':'lr_detail128_c10','lr64':'lr_detail64_c100','hgb':'hgb__complaint_details','cat':'cat_detail64_depth6','xgb':'xgb_detail64_depth4','ordinal':'ordinal_d2_p128_c10'}
    for name,weights in plan['candidates'].items():
        check(name,z[name],cv)
        np.testing.assert_allclose(z[name],sum(sources[k]*w for k,w in weights.items()),atol=1e-12)
        if name=='lr':continue
        target='blend_'+name;config=copy.deepcopy(protocol['candidates']['lr_detail128_c10'])
        config.update(classifier='soft_vote',params={},components=[dict(candidate=base_names[k],weight=w,config=copy.deepcopy(protocol['candidates'][base_names[k]])) for k,w in weights.items()])
        protocol['candidates'][target]=config;arrays[target]=z[name]
    blend_cv=cv[cv.candidate.ne('lr')].copy();blend_cv['candidate']='blend_'+blend_cv.candidate;blend_cv['seconds']=0;records.append(blend_cv)
    protocol['selection']='Highest mean development macro F1 subject to emergency recall no more than one percentage point below the original reference; accuracy then name break ties.'
    protocol['followup_inputs']=inputs
    protocol['supplement']='24 nonlinear/ordinal/neural-head candidates, six paired-text candidates and 24 fixed-weight probability blends. Blends reuse OOF predictions; no meta-model is trained.'
    combined=pd.concat(records,ignore_index=True);assert len(combined)==5*len(protocol['candidates'])
    summary=combined.groupby('candidate').agg(macro_f1=('macro_f1','mean'),f1_std=('macro_f1','std'),accuracy=('accuracy','mean'),precision=('precision_macro','mean'),recall=('recall_macro','mean'),emergency_recall=('emergency_recall','mean'),under_triage=('under_triage_rate','mean')).reset_index()
    winner=rank(summary,summary.set_index('candidate').loc[protocol['baseline']]).iloc[0]
    config=protocol['candidates'][winner.candidate]
    selection=dict(candidate=winner.candidate,config=config,baseline=protocol['baseline'],rule=protocol['selection'],cv_gain=float(winner.macro_f1-summary.set_index('candidate').loc[protocol['baseline'],'macro_f1']))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'selection.json').write_text(json.dumps(selection,indent=2))
    combined.to_csv(output/'cross_validation.csv',index=False);summary.sort_values('macro_f1',ascending=False).to_csv(output/'cv_summary.csv',index=False)
    print('FROZEN WINNER',winner.candidate,flush=True)
    np.savez_compressed(output/'development_oof.npz',row_id=dev.row_id,reference=dev.Labels,selected=arrays[winner.candidate],baseline=old['baseline'])
    review=dev.copy();review['predicted']=arrays[winner.candidate].argmax(1);review['confidence']=arrays[winner.candidate].max(1)
    review.loc[review.Labels.ne(review.predicted)].to_csv(output/'development_errors_for_review.csv',index=False)
    engine.HERE=original.resolve();model,feature,scaler,p=fit_candidate(config,dev,test);pred=p.argmax(1)
    result=dict(candidate=winner.candidate,config=config,feature_count=model.n_features_in_,pca_retained_variance=float(feature.pca_.explained_variance_ratio_.sum()),metrics=scores(test.Labels.to_numpy(),pred),report=classification_report(test.Labels,pred,labels=[0,1,2,3],output_dict=True,zero_division=0),confusion=confusion_matrix(test.Labels,pred,labels=[0,1,2,3]).tolist())
    old_results=json.loads((main/'retrospective_results.json').read_text());base=next(r for r in old_results if r['candidate']==protocol['baseline']);previous=next(r for r in old_results if r['candidate']==previous_selection['candidate'])
    (output/'previous_round_result.json').write_text(json.dumps(previous,indent=2));(output/'retrospective_results.json').write_text(json.dumps([base,result],indent=2))
    for name in ('input_audit.json',protocol['baseline']+'_retrospective_predictions.csv'):shutil.copy2(main/name,output/name)
    pd.DataFrame(dict(row_id=test.row_id,reference=test.Labels,predicted=pred)).to_csv(output/(winner.candidate+'_retrospective_predictions.csv'),index=False)
    np.save(output/'selected_probabilities.npy',p);dest=output/'serving';dest.mkdir()
    for name,value in [('model',model),('structured',feature.structured_),('pca',feature.pca_),('detail_scaler',scaler)]:joblib.dump(value,dest/(name+'.pkl'))
    assert file_sha256(original/'dataset_with_splits.csv')==protocol['source_data_sha256']
    print(json.dumps(result['metrics'],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
