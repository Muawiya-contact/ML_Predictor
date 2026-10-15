"""Freeze a development-selected blend before retrospective evaluation."""
import sys,json,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report,confusion_matrix
import research_engine as engine
from finalize_detail_comparison import fit_candidate
from four_level_study import scores

def run(root,output,blend_name="triage_probability_blends"):
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use an empty destination')
    joint=json.loads((root/'triage_four_level_round6/protocol.json').read_text())
    joint['candidates'].update(json.loads((root/'triage_advanced_classifiers/protocol.json').read_text())['candidates'])
    blends=json.loads((root/blend_name/'protocol.json').read_text())
    summary=pd.read_csv(root/blend_name/'summary.csv')
    prior=pd.read_csv(root/'triage_four_level_round6/cv_summary.csv').set_index('candidate')
    threshold=float(prior.loc[joint['baseline'],'emergency_recall']-.01)
    winner=summary[summary.emergency_recall.ge(threshold)].sort_values(['macro_f1','accuracy','candidate'],ascending=[False,False,True]).iloc[0].candidate
    names={'lr':'lr_detail128_c10','lr64':'lr_detail64_c100','hgb':'hgb__complaint_details','cat':'cat_detail64_depth6','xgb':'xgb_detail64_depth4','ordinal':'ordinal_d2_p128_c10'}
    config=copy.deepcopy(joint['candidates']['lr_detail128_c10']);config['classifier']='soft_vote';config['params']={}
    config['components']=[dict(candidate=names[key],weight=weight,config=joint['candidates'][names[key]]) for key,weight in blends['candidates'][winner].items()]
    selection=dict(candidate='blend_'+winner,config=config,minimum_emergency_recall=threshold,selected_on='fixed five-fold development probabilities only')
    (output/'selection.json').write_text(json.dumps(selection,indent=2))
    df=pd.read_csv(root/'triage_four_level/dataset_with_splits.csv');dev=df[df.partition.eq('development')];test=df[df.partition.eq('test')]
    engine.HERE=(root/'triage_four_level').resolve()
    model,feature,scaler,p=fit_candidate(config,dev,test)
    pred=p.argmax(1);result=dict(candidate=selection['candidate'],config=config,feature_count=model.n_features_in_,pca_retained_variance=float(feature.pca_.explained_variance_ratio_.sum()),metrics=scores(test.Labels.to_numpy(),pred),report=classification_report(test.Labels,pred,labels=[0,1,2,3],output_dict=True,zero_division=0),confusion=confusion_matrix(test.Labels,pred,labels=[0,1,2,3]).tolist())
    (output/'result.json').write_text(json.dumps(result,indent=2));np.save(output/'selected_probabilities.npy',p)
    pd.DataFrame(dict(row_id=test.row_id,reference=test.Labels,predicted=pred)).to_csv(output/(selection['candidate']+'_retrospective_predictions.csv'),index=False)
    dest=output/'serving';dest.mkdir()
    for name,value in [('model',model),('structured',feature.structured_),('pca',feature.pca_),('detail_scaler',scaler)]:joblib.dump(value,dest/(name+'.pkl'))
    print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--blends',default='triage_probability_blends');p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output,a.blends)
