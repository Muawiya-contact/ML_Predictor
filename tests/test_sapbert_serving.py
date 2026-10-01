"""Regression checks for the live SapBERT adapter (no model downloads)."""
import unittest
import tempfile
from pathlib import Path
import sklearn
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
import pandas as pd
from src.sapbert_serving import prepare, predict_frame, load_bundle


class ServingTests(unittest.TestCase):
    def artifacts(self):
        class Structured:
            named_transformers_ = {'cat': SimpleNamespace(named_steps={
                'encoder': SimpleNamespace(categories_=[['Male'], ['Walk-in'], ['Normal']])})}
            def transform(self, frame):
                return np.zeros((len(frame), 22))
        class Model:
            classes_ = np.array([0, 1, 2, 3])
            def predict_proba(self, features):
                assert features.shape[1] == 86
                return np.tile([.05, .1, .15, .7], (len(features), 1))
        return {'structured': Structured(), 'model': Model()}

    @patch('src.sapbert_serving.joblib.load')
    def test_version_and_integrity_fail_before_unpickling(self, load):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'requires scikit-learn'):
                load_bundle(tmp, {'sklearn_version': 'incompatible'})
            Path(tmp, 'model.pkl').write_bytes(b'changed artifact')
            with self.assertRaisesRegex(ValueError, 'evaluated manifest'):
                load_bundle(tmp, {'sklearn_version': sklearn.__version__,
                                  'artifact_sha256': {'model.pkl': 'wrong'}})
        load.assert_not_called()

    def test_gui_missing_complaint_does_not_start_prediction(self):
        from triage_gui import TriageGUI
        for value in ('', ' ', 'X', 'n/a'):
            with self.subTest(value=value):
                gui = SimpleNamespace(
                    complaint=SimpleNamespace(get=lambda *a: value),
                    _show_no_complaint=Mock(), _run_async=Mock())
                TriageGUI._do_predict(gui)
                gui._show_no_complaint.assert_called_once_with()
                gui._run_async.assert_not_called()

    def test_avpu_conversion_matches_training(self):
        frame = pd.DataFrame({'AVPU': ['A', 'voice', 'Pain', 'U', '?'],
            **{c: ['40'] * 5 for c in ['Age','Heart_Rate','Systolic_BP','Diastolic_BP','Temperature','SpO2']},
            **{c: ['x'] * 5 for c in ['Gender','Mode_of_Arrival','ECG_Status']}})
        result = prepare(frame)
        self.assertEqual(result.avpu_ord.iloc[:4].tolist(), [0,1,2,3])
        self.assertTrue(pd.isna(result.avpu_ord.iloc[4]))

    @patch('triage_pipeline.build_text_features', side_effect=lambda a, t: np.zeros((len(t),64)))
    def test_labels_missing_values_and_cap(self, encode):
        frame = pd.DataFrame({'Complaint_Text': ['Chest pain',''], 'Age':['bad',30]})
        out, notes, proba, confidence = predict_frame(self.artifacts(),frame)
        self.assertEqual(out.Predicted_Triage_Level.tolist(), [3,3])
        self.assertEqual(out.Predicted_Level_0to3.tolist(), [3,3])
        self.assertEqual(confidence,[.7,.5])
        self.assertTrue(any('Age missing' in n for n in notes[0]))
        np.testing.assert_allclose(proba[:,3], [.7,.7])

    def test_cluster_uses_supplied_encoder_and_dimension(self):
        from src.cluster_analyzer import analyze_sentence_cluster
        calls = []
        def embed(text, translate=True):
            calls.append((text, translate))
            v = np.zeros(768, dtype=np.float32)
            v[0] = 1
            return {'raw': text, 'translated': text, 'normalized': text,
                    'embedding': v, 'encoder': 'test-sapbert',
                    'l2_norm': 1., 'translated_ok': translate}
        result = analyze_sentence_cluster(['chest pain', 'chest pressure'],
                                           embedder=embed, reference='pain')
        self.assertEqual(result['vectors'].shape, (2,768))
        self.assertEqual(result['sentences'][0]['shape'], (768,))
        self.assertEqual(calls[-1], ('pain',False))
        self.assertTrue(result['diagonal_ok'])

    @patch('triage_pipeline.build_text_features')
    def test_empty_batch_does_not_load_encoder(self, encode):
        out, _, _, _ = predict_frame(self.artifacts(),pd.DataFrame())
        self.assertEqual(len(out),0)
        encode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
