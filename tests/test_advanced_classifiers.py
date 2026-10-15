import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from src.ordinal_classifier import OrdinalLogisticClassifier
from src.soft_voting import MappedSoftVotingClassifier

class AdvancedModels(unittest.TestCase):
    def test_ordinal_probabilities_are_ordered_and_valid(self):
        x,y=make_classification(n_samples=100,n_features=8,n_informative=6,n_redundant=0,n_classes=4,random_state=9)
        model=OrdinalLogisticClassifier().fit(x,y)
        p=model.predict_proba(x)
        self.assertEqual(p.shape,(100,4));self.assertTrue((p>=0).all())
        np.testing.assert_allclose(p.sum(1),1)
        cumulative=np.column_stack([p[:,k+1:].sum(1) for k in range(3)])
        self.assertTrue((np.diff(cumulative,axis=1)<=0).all())

    def test_mapped_blend_matches_independent_models(self):
        x,y=make_classification(n_samples=100,n_features=8,n_informative=6,n_redundant=0,n_classes=4,random_state=9)
        a=LogisticRegression().fit(x,y);b=LogisticRegression().fit(x[:,[0,2,4]],y)
        model=MappedSoftVotingClassifier([a,b],[list(range(8)),[0,2,4]],[.5,.5],8)
        np.testing.assert_allclose(model.predict_proba(x),.5*a.predict_proba(x)+.5*b.predict_proba(x[:,[0,2,4]]))
        with self.assertRaisesRegex(ValueError,'feature mapping'):
            MappedSoftVotingClassifier([a,b],[list(range(8)),[0,2,20]],[.5,.5],8)
        with self.assertRaisesRegex(ValueError,'sum to one'):
            MappedSoftVotingClassifier([a],[list(range(8))],[.5],8)
