"""Fit component models and verify their shared feature-column mappings."""
import numpy as np
from src.complaint_details import detail_matrix,FEATURE_NAMES
from src.soft_voting import MappedSoftVotingClassifier


def matrix_and_names(config,feature,scaler,frame):
    matrix=feature.transform(frame)
    names=list(feature.structured_.get_feature_names_out())+[f'embedding_{i}' for i in range(config['features']['pca'])]
    if config.get('text_details'):
        column='chief_complaint' if config['text_details']['source']=='complaint_details' else 'Clinical_Concept'
        matrix=np.hstack([matrix,scaler.transform(detail_matrix(frame[column].fillna('').tolist()))])
        names += ['detail_'+n for n in FEATURE_NAMES]
    return matrix,names


def fit(config,dev,test):
    from finalize_detail_comparison import fit_candidate
    fitted=[]
    for component in config['components']:
        fitted.append(fit_candidate(component['config'],dev,test))
    base=config['components'][0]['config'];_,feature,scaler,_=fitted[0]
    a,names=matrix_and_names(base,feature,scaler,dev)
    b,_=matrix_and_names(base,feature,scaler,test)
    models=[];mappings=[];weights=[];expected=np.zeros((len(test),4))
    for component,(model,other,detail,probabilities) in zip(config['components'],fitted):
        x,other_names=matrix_and_names(component['config'],other,detail,dev)
        xv,_=matrix_and_names(component['config'],other,detail,test)
        mapping=[names.index(n) for n in other_names]
        np.testing.assert_allclose(a[:,mapping],x,atol=1e-9,rtol=1e-7)
        np.testing.assert_allclose(b[:,mapping],xv,atol=1e-9,rtol=1e-7)
        models.append(model);mappings.append(mapping);weights.append(component['weight'])
        expected += component['weight']*probabilities
    ensemble=MappedSoftVotingClassifier(models,mappings,weights,a.shape[1])
    np.testing.assert_allclose(ensemble.predict_proba(b),expected,atol=1e-10,rtol=1e-7)
    return ensemble,feature,scaler,ensemble.predict_proba(b)
