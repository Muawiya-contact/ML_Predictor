"""Protect the second-round selection boundary and emergency-recall constraint."""
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/triage_study'))
from improve_four_level import candidates, rank

class ImprovementTests(unittest.TestCase):
    def test_selection_rejects_accuracy_gain_with_emergency_recall_loss(self):
        frame=pd.DataFrame([
            dict(candidate='baseline',macro_f1=.86,accuracy=.85,emergency_recall=.93),
            dict(candidate='unsafe_tradeoff',macro_f1=.95,accuracy=.95,emergency_recall=.90),
            dict(candidate='improved',macro_f1=.88,accuracy=.87,emergency_recall=.925)])
        self.assertEqual(rank(frame,frame.iloc[0]).iloc[0].candidate,'improved')
    def test_baseline_remains_eligible_when_no_candidate_improves(self):
        frame=pd.DataFrame([
            dict(candidate='baseline',macro_f1=.86,accuracy=.85,emergency_recall=.93),
            dict(candidate='worse',macro_f1=.85,accuracy=.84,emergency_recall=.94)])
        self.assertEqual(rank(frame,frame.iloc[0]).iloc[0].candidate,'baseline')
    def test_search_includes_incumbent_and_planned_dimensions(self):
        configs=candidates()
        self.assertEqual(configs['lr_pca64_c10_balanced1']['params'],{'C':10,'balance':True})
        self.assertTrue({64,128,256} <= {c['features']['pca'] for c in configs.values()})
        self.assertTrue(all(c['features']['view'] in ('fused','structured') for c in configs.values()))

    def test_quadratic_patient_preprocessing_matches_serving(self):
        import numpy as np
        import research_engine as engine
        from src.sapbert_serving import prepare
        frame=pd.DataFrame({
            **{name:[30.,40.,50.] for name in engine.NUM[:6]},
            'AVPU':['A','V','U'], 'Gender':['Male']*3,
            'Mode_of_Arrival':['Walk-in']*3,'ECG_Status':['Normal']*3})
        features=engine.Features(view='structured',polynomial=True).fit(frame)
        expected=features.transform(frame)
        actual=features.structured_.transform(prepare(frame))
        np.testing.assert_allclose(actual,expected)
        self.assertEqual(expected.shape,(3,38))  # 35 numeric terms + 3 categories.
        before=features.structured_.named_transformers_['num'].named_steps['scaler'].mean_.copy()
        shifted=frame.copy();shifted['Age']=999
        features.transform(shifted)
        np.testing.assert_array_equal(before,features.structured_.named_transformers_['num'].named_steps['scaler'].mean_)

    def test_refinement_preserves_classifier_scope_and_serving_features(self):
        from improvement_refinement import configs
        planned = configs()
        self.assertEqual(len(planned), 14)
        self.assertEqual({c['classifier'] for c in planned.values()}, {'logreg', 'rf', 'hgb'})
        for config in planned.values():
            self.assertEqual(config['features']['encoder'], 'sapbert_concept')
            self.assertEqual(config['features']['view'], 'fused')
            self.assertNotIn('text_weight', config['features'])
            self.assertFalse(config['features'].get('derived', False))
