"""Keep complaint detail features in both CV and final fitting paths."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/triage_study'))
from src.complaint_details import FEATURE_NAMES, VERSION
from improve_four_level import evaluate_fold
from finalize_detail_comparison import fit_candidate


class DummyFeatures:
    def __init__(self, **kwargs): pass
    def fit(self, frame): return self
    def transform(self, frame): return np.zeros((len(frame), 1))

class DummyModel:
    classes_ = np.arange(4)
    def predict_proba(self, values): return np.tile([.25]*4, (len(values),1))

class DetailRefinementTests(unittest.TestCase):
    def fixture(self):
        config=dict(classifier='logreg',features={},params={},text_details=dict(source='complaint_details',version=VERSION,feature_names=FEATURE_NAMES))
        frame=pd.DataFrame(dict(row_id=range(8),Labels=[0,1,2,3]*2,cv5=[1]*4+[0]*4,partition=['development']*8,group=range(8),chief_complaint=['one hour']*4+['two hours']*4,Clinical_Concept=['three hours']*8))
        return config,frame

    def test_final_fit_preserves_raw_details_and_training_only_scaler(self):
        config,frame=self.fixture();captured=[]
        with patch('research_engine.Features',DummyFeatures), patch('research_engine.fit_model',side_effect=lambda c,x,y: captured.append(x) or DummyModel()):
            model,feature,scaler,prob=fit_candidate(config,frame.iloc[:4],frame.iloc[4:])
        self.assertEqual(captured[0].shape,(4,20))
        self.assertAlmostEqual(scaler.mean_[FEATURE_NAMES.index('log_duration_minutes')],np.log1p(60))
        self.assertEqual(prob.shape,(4,4))

    def test_cv_path_retains_detail_block(self):
        config,frame=self.fixture();captured=[]
        with tempfile.TemporaryDirectory() as tmp:
            frame.to_csv(Path(tmp)/'dataset_with_splits.csv',index=False)
            with patch('research_engine.Features',DummyFeatures), patch('research_engine.fit_model',side_effect=lambda c,x,y: captured.append(x) or DummyModel()):
                rows,ids,preds=evaluate_fold(tmp,0,{'details':config})
        self.assertEqual(captured[0].shape,(4,20))
        self.assertEqual(ids.tolist(),[4,5,6,7])
        self.assertEqual(len(rows),1)

if __name__=='__main__': unittest.main()
