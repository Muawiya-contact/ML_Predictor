"""Additional classifiers must retain four-class probability and pickle parity."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import make_classification
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/triage_study'))
import research_engine as engine
from expanded_classifiers import configs

class ExpandedClassifiers(unittest.TestCase):
    def test_fixed_candidate_scope(self):
        planned=configs()
        self.assertEqual(len(planned),11)
        self.assertEqual({x['classifier'] for x in planned.values()},{'catboost','xgboost','svc'})
        self.assertTrue(all(x['text_details']['source']=='complaint_details' for x in planned.values()))

    def test_probability_serialization_for_all_new_families(self):
        x,y=make_classification(n_samples=80,n_features=10,n_informative=7,n_redundant=0,n_classes=4,random_state=9)
        for family,params in [('catboost',dict(iterations=8,depth=3)),('xgboost',dict(n_estimators=8,max_depth=3)),('svc',dict(C=1))]:
            with self.subTest(family=family):
                if family!='svc' and importlib.util.find_spec(family) is None:
                    continue
                cfg=engine.config(family,{},balance=True,**params)
                model=engine.fit_model(cfg,x,y)
                np.testing.assert_array_equal(model.classes_,[0,1,2,3])
                p=model.predict_proba(x)
                self.assertEqual(p.shape,(80,4))
                self.assertTrue(np.isfinite(p).all() and (p>=0).all())
                np.testing.assert_allclose(p.sum(1),1,atol=1e-6,rtol=0)
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'model.pkl';joblib.dump(model,path)
                    np.testing.assert_allclose(joblib.load(path).predict_proba(x),p,atol=1e-12)
