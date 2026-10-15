"""Isolate label-task changes using identical inputs, splits and classifiers.

This is a retrospective diagnostic, not another model-selection experiment.
Old labels are never written to the current dataset or deployed model.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import research_engine as engine
from four_level_study import PARAMS
from cache_integrity import file_sha256


def run(old_source, current_source, output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty diagnostic directory')
    old = pd.read_csv(old_source / 'dataset_with_splits.csv').sort_values('source_row').reset_index(drop=True)
    new = pd.read_csv(current_source / 'dataset_with_splits.csv').sort_values('source_row').reset_index(drop=True)
    np.testing.assert_array_equal(old.source_row, new.source_row)
    inputs = ['Age','Gender','Mode_of_Arrival','chief_complaint','Clinical_Concept','Heart_Rate','Systolic_BP','Diastolic_BP','ECG_Status','Temperature','SpO2','AVPU']
    pd.testing.assert_frame_equal(old[inputs], new[inputs])
    assert file_sha256(old_source / 'emb_sapbert_concept.npy') == file_sha256(current_source / 'emb_sapbert_concept.npy')
    dev = new.partition.eq('development').to_numpy()
    assert not set(new.loc[dev,'group']) & set(new.loc[~dev,'group'])
    plan = dict(purpose='Retrospective label-task diagnostic; no model promotion or label editing.',
                comparison='Same current grouped split, PCA-64, patient preprocessing, SapBERT vectors and classifier settings; only target definitions differ.',
                source_hashes={name:file_sha256(path/'dataset_with_splits.csv') for name,path in [('old',old_source),('current',current_source)]},
                embedding_sha256=file_sha256(current_source/'emb_sapbert_concept.npy'),
                input_rows_identical=len(new),changed_partition_membership=int((old.partition!=new.partition).sum()),
                old_counts=old.Labels.value_counts().sort_index().to_dict(),new_counts=new.Labels.value_counts().sort_index().to_dict(),
                label_transition=pd.crosstab(old.Labels,new.Labels).to_dict(),
                classifier_parameters=PARAMS,labels_modified=False)
    (output/'protocol.json').write_text(json.dumps(plan,indent=2))
    engine.HERE = current_source.resolve()
    features = engine.Features(view='fused',encoder='sapbert_concept',pca=64,solver='full').fit(new.loc[dev])
    train_x, test_x = features.transform(new.loc[dev]), features.transform(new.loc[~dev])
    results=[]
    for family, params in PARAMS.items():
        config=engine.config(family,dict(view='fused',encoder='sapbert_concept',pca=64,solver='full'),**params)
        for task,frame in [('old_three_level',old),('current_four_level',new)]:
            y=frame.Labels.to_numpy(dtype=int)
            model=engine.fit_model(config,train_x,y[dev])
            pred=model.predict(test_x)
            report=classification_report(y[~dev],pred,output_dict=True,zero_division=0)
            row=dict(classifier=family,task=task,accuracy=report['accuracy'],precision_macro=report['macro avg']['precision'],recall_macro=report['macro avg']['recall'],macro_f1=report['macro avg']['f1-score'],labels=model.classes_.tolist(),confusion=confusion_matrix(y[~dev],pred,labels=model.classes_).tolist())
            if task=='current_four_level':
                saved=pd.read_csv(current_source/f'fused_64_{family}_predictions.csv')
                np.testing.assert_array_equal(new.loc[~dev,'row_id'],saved.row_id)
                np.testing.assert_array_equal(pred,saved.predicted)
                row['current_baseline_predictions_reproduced']=True
            results.append(row)
            print(json.dumps(row),flush=True)
            (output/'results.json').write_text(json.dumps(results,indent=2))
    assert file_sha256(old_source/'dataset_with_splits.csv')==plan['source_hashes']['old']
    assert file_sha256(current_source/'dataset_with_splits.csv')==plan['source_hashes']['current']
    (output/'verification.json').write_text(json.dumps(dict(status='passed',paired_conditions=6,current_baseline_predictions_reproduced=True,labels_modified=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-source',type=Path,required=True)
    parser.add_argument('--current-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.old_source,args.current_source,args.output)
