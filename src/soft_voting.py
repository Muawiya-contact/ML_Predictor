"""A fixed probability blend with verified mappings into a shared feature matrix."""
import numpy as np


class MappedSoftVotingClassifier:
    def __init__(self, models, mappings, weights, n_features):
        if not (len(models)==len(mappings)==len(weights)) or not models:
            raise ValueError('Each component needs a mapping and weight')
        self.weights=np.asarray(weights,dtype=float)
        if (self.weights<0).any() or not np.isclose(self.weights.sum(),1):
            raise ValueError('Weights must be nonnegative and sum to one')
        self.models=models
        self.mappings=[np.asarray(m,dtype=int) for m in mappings]
        self.n_features_in_=n_features
        self.classes_=np.asarray(models[0].classes_)
        for model,mapping in zip(models,self.mappings):
            if not np.array_equal(model.classes_,self.classes_):raise ValueError('Component class order differs')
            if len(mapping)!=model.n_features_in_ or (mapping<0).any() or (mapping>=n_features).any():raise ValueError('Invalid component feature mapping')

    def predict_proba(self,X):
        X=np.asarray(X)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_:raise ValueError('Unexpected ensemble feature dimensions')
        return sum(w*m.predict_proba(X[:,columns]) for w,m,columns in zip(self.weights,self.models,self.mappings))

    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(1)]
