"""Cumulative binary logistic models for the ordered levels 0, 1, 2, 3."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class OrdinalLogisticClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, max_iter=2000):
        self.C=C
        self.max_iter=max_iter

    def fit(self,X,y,sample_weight=None):
        X,y=check_X_y(X,y)
        self.classes_=np.unique(y)
        if not np.array_equal(self.classes_,np.arange(4)):
            raise ValueError('Expected all four levels 0, 1, 2, 3')
        self.n_features_in_=X.shape[1]
        self.models_=[LogisticRegression(C=self.C,max_iter=self.max_iter,random_state=42).fit(X,y>k,sample_weight=sample_weight) for k in range(3)]
        return self

    def predict_proba(self,X):
        check_is_fitted(self,'models_');X=check_array(X)
        cumulative=np.column_stack([m.predict_proba(X)[:,1] for m in self.models_])
        # Enforce P(Y>0) >= P(Y>1) >= P(Y>2), then recover class masses.
        cumulative=np.minimum.accumulate(cumulative,axis=1)
        return np.column_stack([1-cumulative[:,0],cumulative[:,0]-cumulative[:,1],cumulative[:,1]-cumulative[:,2],cumulative[:,2]])

    def predict(self,X):
        return self.classes_[self.predict_proba(X).argmax(1)]
