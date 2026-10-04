"""Fixed-weight OOF blends; no test rows or fitted meta-model are used."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from four_level_study import scores
from cache_integrity import file_sha256


def run(root,output,refine=False):
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise ValueError('Use a fresh destination')
    sources={'lr':('triage_four_level_round6/development_oof.npz','selected'),
      'lr64':('triage_detail_refinement/oof.npz','lr_detail64_c100'),
      'hgb':('triage_learning_detail_audit/development_oof.npz','hgb__complaint_details'),
      'cat':('triage_expanded_classifiers/oof.npz','cat_detail64_depth6'),
      'xgb':('triage_expanded_classifiers/oof.npz','xgb_detail64_depth4')}
    configs={'lr':{'lr':1.0}}
    for peer in ('lr64','hgb','cat','xgb'):
        for weight in (.1,.25,.5):configs[f'lr_{peer}_{weight}']={'lr':1-weight,peer:weight}
    configs['lr_hgb_cat']={'lr':.6,'hgb':.2,'cat':.2}
    if refine:
        sources['ordinal']=('triage_advanced_classifiers/oof.npz','ordinal_d2_p128_c10')
        for weight in (.35,.4,.45,.55,.6,.65,.75,.9):
            configs[f'lr_hgb_{weight}']={'lr':1-weight,'hgb':weight}
        for name,weights in [('lr_hgb_ordinal_442',{'lr':.4,'hgb':.4,'ordinal':.2}),('lr_hgb_ordinal_252',{'lr':.25,'hgb':.5,'ordinal':.25}),('lr_hgb_ordinal_154',{'lr':.1,'hgb':.5,'ordinal':.4})]:
            configs[name]=weights
    protocol=dict(candidates=configs,sources=sources,source_hashes={k:file_sha256(root/v[0]) for k,v in sources.items()},test_rows_used=False,weights_fitted=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    df=pd.read_csv(root/'triage_four_level/dataset_with_splits.csv');dev=df[df.partition.eq('development')]
    arrays={}
    for name,(path,key) in sources.items():
        z=np.load(root/path);np.testing.assert_array_equal(z['row_id'],dev.row_id);arrays[name]=z[key]
    records=[];predictions={}
    for name,weights in configs.items():
        p=sum(arrays[k]*v for k,v in weights.items());np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
        predictions[name]=p
        for fold in range(5):
            mask=dev.cv5.eq(fold).to_numpy()
            records.append(dict(candidate=name,fold=fold+1,**scores(dev.Labels.to_numpy()[mask],p[mask].argmax(1))))
    cv=pd.DataFrame(records);cv.to_csv(output/'cross_validation.csv',index=False)
    mean=cv.groupby('candidate').mean(numeric_only=True).drop(columns='fold').sort_values('macro_f1',ascending=False)
    mean.to_csv(output/'summary.csv');np.savez_compressed(output/'oof.npz',row_id=dev.row_id,**predictions)
    print(mean[['accuracy','precision_macro','recall_macro','macro_f1','emergency_recall']].to_string())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--refine',action='store_true');p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output,a.refine)
