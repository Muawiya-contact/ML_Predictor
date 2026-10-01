"""Freeze combined development selection before any retrospective evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from src.complaint_details import detail_matrix, VERSION, FEATURE_NAMES
from cache_integrity import file_sha256
from four_level_study import scores
from improve_four_level import rank
import research_engine as engine


def fit_candidate(config, dev, test):
    feature = engine.Features(**config['features']).fit(dev)
    a,b=feature.transform(dev),feature.transform(test)
    scaler=None
    details=config.get('text_details')
    if details:
        column='chief_complaint' if details['source']=='complaint_details' else 'Clinical_Concept'
        scaler=StandardScaler().fit(detail_matrix(dev[column].fillna('').tolist()))
        a=np.hstack([a,scaler.transform(detail_matrix(dev[column].fillna('').tolist()))])
        b=np.hstack([b,scaler.transform(detail_matrix(test[column].fillna('').tolist()))])
    model=engine.fit_model(config,a,dev.Labels.to_numpy())
    return model,feature,scaler,model.predict_proba(b)


def finalize(source, original, incumbent, output):
    if json.loads((source/'verification.json').read_text())['status']!='passed':
        raise ValueError('Verify the learning audit before finalizing')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a fresh comparison destination')
    protocol=json.loads((incumbent/'protocol.json').read_text())
    audit=json.loads((source/'protocol.json').read_text())
    for key in ['source_data_sha256','embedding_sha256']:
        assert protocol[key]==audit[key]
    assert file_sha256(original/'dataset_with_splits.csv')==protocol['source_data_sha256']
    assert file_sha256(original/'emb_sapbert_concept.npy')==protocol['embedding_sha256']
    cv=pd.read_csv(incumbent/'cross_validation.csv')
    new=pd.read_csv(source/'cross_validation.csv')
    new=new[new.fraction.eq(1)&new.variant.ne('baseline')].copy()
    new['candidate']=new.classifier+'__'+new.variant
    for family,base in audit['candidates'].items():
        for variant in ['concept_details','complaint_details']:
            config=copy.deepcopy(base)
            config['text_details']=dict(source=variant,version=VERSION,feature_names=FEATURE_NAMES)
            name=family+'__'+variant
            assert name not in protocol['candidates']
            protocol['candidates'][name]=config
    protocol['supplement']='Six predeclared detail-feature conditions from a development-only audit; learning-curve subset fits are reported separately.'
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cv=pd.concat([cv,new[cv.columns]],ignore_index=True)
    assert len(cv)==5*len(protocol['candidates']) and not cv.duplicated(['candidate','fold']).any()
    cv.to_csv(output/'cross_validation.csv',index=False)
    summary=cv.groupby('candidate').agg(macro_f1=('macro_f1','mean'),f1_std=('macro_f1','std'),
        accuracy=('accuracy','mean'),precision=('precision_macro','mean'),recall=('recall_macro','mean'),
        emergency_recall=('emergency_recall','mean'),under_triage=('under_triage_rate','mean')).reset_index()
    baseline=summary.set_index('candidate').loc[protocol['baseline']]
    winner=rank(summary,baseline).iloc[0]
    selection=dict(candidate=winner.candidate,config=protocol['candidates'][winner.candidate],
                   baseline=protocol['baseline'],rule=protocol['selection'],cv_gain=float(winner.macro_f1-baseline.macro_f1))
    summary.sort_values('macro_f1',ascending=False).to_csv(output/'cv_summary.csv',index=False)
    (output/'selection.json').write_text(json.dumps(selection,indent=2))
    print('FROZEN SELECTION',json.dumps(selection),flush=True)
    old_selection=json.loads((incumbent/'selection.json').read_text())
    old_oof=np.load(incumbent/'development_oof.npz')
    new_oof=np.load(source/'development_oof.npz')
    np.testing.assert_array_equal(old_oof['row_id'],new_oof['row_id'])
    selected=old_oof['selected'] if winner.candidate==old_selection['candidate'] else new_oof[winner.candidate]
    np.savez_compressed(output/'development_oof.npz',row_id=old_oof['row_id'],reference=old_oof['reference'],baseline=old_oof['baseline'],selected=selected)
    family_choices=[]
    for family in ['logreg','hgb','rf']:
        names=[name for name,cfg in protocol['candidates'].items() if cfg['classifier']==family and cfg['features']['view']=='fused']
        best=summary[summary.candidate.isin(names)].sort_values(['macro_f1','accuracy','candidate'],ascending=[False,False,True]).iloc[0]
        family_choices.append(dict(candidate=best.candidate,config=protocol['candidates'][best.candidate],cv_macro_f1=float(best.macro_f1)))
    (output/'family_selection.json').write_text(json.dumps(family_choices,indent=2))
    old_results=json.loads((incumbent/'retrospective_results.json').read_text())
    base_result=next(r for r in old_results if r['candidate']==protocol['baseline'])
    prior_result=next(r for r in old_results if r['candidate']==old_selection['candidate'])
    (output/'previous_round_result.json').write_text(json.dumps(prior_result,indent=2))
    shutil.copy2(incumbent/'input_audit.json',output/'input_audit.json')
    shutil.copy2(incumbent/(protocol['baseline']+'_retrospective_predictions.csv'),output/(protocol['baseline']+'_retrospective_predictions.csv'))
    df=pd.read_csv(original/'dataset_with_splits.csv');dev=df[df.partition.eq('development')];test=df[df.partition.eq('test')]
    engine.HERE=original.resolve()
    results={}
    names=list(dict.fromkeys([winner.candidate]+[r['candidate'] for r in family_choices]))
    for name in names:
        config=protocol['candidates'][name]
        model,feature,scaler,prob=fit_candidate(config,dev,test)
        pred=prob.argmax(1)
        result=dict(candidate=name,config=config,feature_count=int(model.n_features_in_),
                    pca_retained_variance=float(feature.pca_.explained_variance_ratio_.sum()),
                    metrics=scores(test.Labels.to_numpy(),pred),
                    report=classification_report(test.Labels,pred,labels=[0,1,2,3],output_dict=True,zero_division=0),
                    confusion=confusion_matrix(test.Labels,pred,labels=[0,1,2,3]).tolist())
        results[name]=result
        records=pd.DataFrame(dict(row_id=test.row_id,reference=test.Labels,predicted=pred))
        records.to_csv(output/(name+'_family_predictions.csv'),index=False)
        if name==winner.candidate:
            records.to_csv(output/(name+'_retrospective_predictions.csv'),index=False)
            np.save(output/'selected_probabilities.npy',prob)
            destination=output/'serving';destination.mkdir()
            for key,obj in [('model',model),('structured',feature.structured_),('pca',feature.pca_),('detail_scaler',scaler)]:
                if obj is not None:joblib.dump(obj,destination/(key+'.pkl'))
        print(name,result['metrics'],flush=True)
    (output/'retrospective_results.json').write_text(json.dumps([base_result,results[winner.candidate]],indent=2))
    (output/'family_results.json').write_text(json.dumps([results[r['candidate']] for r in family_choices],indent=2))
    assert file_sha256(original/'dataset_with_splits.csv')==protocol['source_data_sha256']
    assert file_sha256(original/'emb_sapbert_concept.npy')==protocol['embedding_sha256']
    review=dev.copy();review['oof_predicted']=selected.argmax(1);review['oof_confidence']=selected.max(1)
    review[review.Labels.ne(review.oof_predicted)].to_csv(output/'development_errors_for_review.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['source','original','incumbent','output']:parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    finalize(args.source,args.original,args.incumbent,args.output)
