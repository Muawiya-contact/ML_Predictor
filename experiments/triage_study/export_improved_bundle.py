"""Export a verified second-round fused winner without retraining or relabelling."""
import argparse,json,shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import research_engine as engine
from cache_integrity import file_sha256

def export(source,original,incumbent,output):
    selection=json.loads((source/'selection.json').read_text())
    verification=json.loads((source/'verification.json').read_text())
    if verification['status']!='passed':raise ValueError('Independent verification required')
    config=selection['config']
    if config['features']['view']!='fused':raise ValueError('Structured-only winner is a research control; cannot silently replace SapBERT deployment.')
    if selection['cv_gain']<=0:raise ValueError('No development improvement: retain incumbent deployment.')
    manifest=json.loads((incumbent/'model_manifest.json').read_text())
    if manifest['sklearn_version']!=sklearn.__version__:raise ValueError('Training/runtime version mismatch')
    result=next(r for r in json.loads((source/'retrospective_results.json').read_text()) if r['candidate']==selection['candidate'])
    df=pd.read_csv(original/'dataset_with_splits.csv');test=df[df.partition.eq('test')]
    engine.HERE=original.resolve()
    transform=engine.Features(**config['features']);transform.structured_=joblib.load(source/'serving/structured.pkl');transform.pca_=joblib.load(source/'serving/pca.pkl');model=joblib.load(source/'serving/model.pkl')
    predictions=model.predict(transform.transform(test));expected=pd.read_csv(source/(selection['candidate']+'_retrospective_predictions.csv'))
    np.testing.assert_array_equal(test.row_id,expected.row_id);np.testing.assert_array_equal(predictions,expected.predicted)
    names={'logreg':'Logistic Regression','hgb':'Hist Gradient Boosting','rf':'Random Forest'};pc=config['features']['pca']
    description = (' + quadratic patient features' if config['features'].get('polynomial') else '')
    manifest.update(method=f'SapBERT + PCA-{pc} + '+names[config['classifier']]+description,projected_embedding_dim=pc,
        feature_blocks=[dict(name='structured',dim=int(model.n_features_in_-pc)),dict(name='embedding',dim=pc,rescaled=False)],
        text_pipeline=f'English -> SapBERT CLS (768) -> fitted PCA ({pc})',source_config=config,selection=selection,
        evaluation_note='Second-round retrospective comparison on previously examined test rows; new independent data is required. Scores use supplied concepts, not live Ollama translations.',
        artifact_sha256={name:file_sha256(source/'serving'/name) for name in ['model.pkl','structured.pkl','pca.pkl']},
        improvement_verification=verification)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Use an empty export destination')
    for name in manifest['artifact_sha256']:shutil.copy2(source/'serving'/name,output/name)
    (output/'model_manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(incumbent/'learned_stopwords.json',output/'learned_stopwords.json')
    (output/'triage_metrics.json').write_text(json.dumps(dict(accuracy=result['metrics']['accuracy']*100,labels=[0,1,2,3],confusion_matrix=result['confusion'],metrics=result['metrics'],classification_report=result['report'],evaluation_note=manifest['evaluation_note']),indent=2))
    print(f'Exported {selection["candidate"]}; all {len(test)} predictions verified.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--incumbent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.source,a.original,a.incumbent,a.output)
