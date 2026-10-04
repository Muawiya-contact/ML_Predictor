"""Development-only error and confidence diagnostics; no automatic relabelling."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, log_loss
from cache_integrity import file_sha256


def investigate(original, incumbent, output, peers=None, bundle=None):
    protocol=json.loads((incumbent/'protocol.json').read_text())
    if file_sha256(original/'dataset_with_splits.csv')!=protocol['source_data_sha256']:
        raise ValueError('Source changed')
    if file_sha256(original/'emb_sapbert_concept.npy')!=protocol['embedding_sha256']:
        raise ValueError('Embeddings changed')
    frame=pd.read_csv(original/'dataset_with_splits.csv')
    dev=frame[frame.partition.eq('development')].copy()
    z=np.load(incumbent/'development_oof.npz')
    np.testing.assert_array_equal(dev.row_id,z['row_id'])
    np.testing.assert_array_equal(dev.Labels,z['reference'])
    p=z['selected'];np.testing.assert_allclose(p.sum(1),1)
    truth=dev.Labels.to_numpy();pred=p.argmax(1);confidence=p.max(1);wrong=pred!=truth
    matrix=confusion_matrix(truth,pred,labels=[0,1,2,3])
    bins=[];ece=0.
    for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
        mask=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=1)
        if mask.any():
            accuracy=float((pred[mask]==truth[mask]).mean());mean=float(confidence[mask].mean())
            ece+=mask.mean()*abs(accuracy-mean)
            bins.append(dict(lower=float(lo),upper=float(hi),rows=int(mask.sum()),accuracy=accuracy,mean_confidence=mean))
    # Complete rows only: matching unavailable values must not imply clinical equality.
    inputs=['Age','Heart_Rate','Systolic_BP','Diastolic_BP','Temperature','SpO2','Gender','Mode_of_Arrival','AVPU','ECG_Status','chief_complaint']
    complete=dev.dropna(subset=inputs)
    groups=complete.groupby(inputs,dropna=False).Labels.nunique()
    result=dict(candidate=json.loads((incumbent/'selection.json').read_text())['candidate'], source_data_sha256=protocol['source_data_sha256'], oof_sha256=file_sha256(incumbent/'development_oof.npz'), development_rows=len(dev),errors=int(wrong.sum()),confusion_matrix=matrix.tolist(),
                urgent_standard_errors=int(matrix[1,2]+matrix[2,1]),
                adjacent_errors=int(((abs(pred-truth)==1)&wrong).sum()),
                high_confidence_errors_90=int((wrong&(confidence>=.9)).sum()),
                multiclass_mcc=float(matthews_corrcoef(truth,pred)),
                multiclass_log_loss=float(log_loss(truth,p,labels=[0,1,2,3])),
                multiclass_brier=float(((p-np.eye(4)[truth])**2).sum(1).mean()),
                confidence_ece_10_bins=float(ece),calibration_bins=bins,
                exact_complete_input_groups_with_conflicting_labels=int((groups>1).sum()),
                class_counts=dev.Labels.value_counts().sort_index().to_dict(),
                missing_inputs=dev[inputs].isna().sum().to_dict(),
                labels_modified=False,test_rows_used=False,
                interpretation='Pooled development OOF diagnostics conditional on prior selection. Confidence is uncalibrated. No clinical label correctness or achievable accuracy ceiling is inferred.')
    numeric=['Age','Heart_Rate','Systolic_BP','Diastolic_BP','Temperature','SpO2']
    values=dev[numeric].apply(pd.to_numeric,errors='coerce')
    result['numeric_profile']=values.agg(['min','median','max']).to_dict()
    result['nonfinite_numeric_inputs']=int((~np.isfinite(values.to_numpy())).sum())
    result['diastolic_above_systolic_rows']=int((values.Diastolic_BP>values.Systolic_BP).sum())
    if peers is not None:
        other=np.load(peers/'development_oof.npz')
        np.testing.assert_array_equal(other['row_id'],dev.row_id)
        np.testing.assert_array_equal(other['reference'],truth)
        votes=np.stack([other[k].argmax(1) for k in ['logreg__complaint_details','hgb__complaint_details','rf__complaint_details']])
        result['all_three_families_wrong']=int((votes!=truth).all(0).sum())
        result['all_three_agree_on_wrong_level']=int(((votes!=truth).all(0)&(votes==votes[0]).all(0)).sum())
        result['peer_oof_sha256']=file_sha256(peers/'development_oof.npz')
    output.mkdir(parents=True,exist_ok=True)
    if bundle is not None:
        from triage_pipeline import load_artifacts
        from src.complaint_details import detail_matrix, FEATURE_NAMES
        import research_engine as engine
        artifacts=load_artifacts(str(bundle))
        config=json.loads((incumbent/'selection.json').read_text())['config']
        if artifacts['manifest']['source_config'] != config:
            raise ValueError('Coefficient diagnostic bundle differs from the incumbent')
        if not hasattr(artifacts['model'],'coef_'):
            raise ValueError('Coefficient diagnostic requires Logistic Regression')
        engine.HERE=original.resolve()
        transform=engine.Features(**config['features'])
        transform.structured_=artifacts['structured'];transform.pca_=artifacts['pca']
        values=transform.transform(dev)
        values=np.hstack([values,artifacts['detail_scaler'].transform(detail_matrix(dev.chief_complaint))])
        names=list(transform.structured_.get_feature_names_out())+[f'PCA_{i+1}' for i in range(config['features']['pca'])]+FEATURE_NAMES
        strength=(abs(artifacts['model'].coef_)*values.std(0)).mean(0)
        pd.DataFrame(dict(feature=names,mean_abs_standardized_coefficient=strength)).to_csv(output/'coefficient_diagnostics.csv',index=False)
        result['coefficient_interpretation']='Descriptive mean absolute LR coefficient times development feature standard deviation; correlated features and joint effects prevent a causal interpretation.'
        result['coefficient_model_sha256']=file_sha256(bundle/'model.pkl')
    (output/'diagnostics.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    axes[0].imshow(matrix,cmap='Blues');axes[0].set_xticks(range(4));axes[0].set_yticks(range(4))
    axes[0].set_xlabel('Predicted');axes[0].set_ylabel('Reference');axes[0].set_title('Pre-refinement model: development OOF')
    for i in range(4):
        for j in range(4):
            axes[0].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    axes[1].plot([0,1],[0,1],':',color='gray',label='Perfect agreement')
    axes[1].plot([r['mean_confidence'] for r in bins],[r['accuracy'] for r in bins],'o-',color='#2874ad',label='OOF observations')
    axes[1].set_xlim(0,1);axes[1].set_ylim(0,1);axes[1].set_xlabel('Mean maximum probability');axes[1].set_ylabel('Observed accuracy')
    axes[1].set_title('Confidence diagnostic (uncalibrated)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'current_error_confidence.png',dpi=200);plt.close(fig)

    pd.DataFrame(bins).to_csv(output/'confidence_bins.csv',index=False)
    review=dev[wrong].copy();review['predicted']=pred[wrong];review['confidence']=confidence[wrong]
    review['reviewed_label']='';review['reviewer_reason']=''
    review.sort_values('confidence',ascending=False).to_csv(output/'review_queue.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['calibration_bins','missing_inputs']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for flag in ['original','incumbent','output']:parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--peers',type=Path)
    parser.add_argument('--bundle',type=Path)
    a=parser.parse_args();investigate(a.original,a.incumbent,a.output,a.peers,a.bundle)
