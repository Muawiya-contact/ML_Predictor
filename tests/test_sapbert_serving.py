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
    @patch('src.sapbert_serving.joblib.load')
    def test_classifier_runtime_checked_before_unpickling(self, load):
        from importlib.metadata import PackageNotFoundError
        manifest = {'sklearn_version': sklearn.__version__,
                    'classifier_runtime': {'package': 'catboost', 'version': '1.2.10'}}
        for result in ('different', PackageNotFoundError('catboost')):
            with self.subTest(result=result), patch('importlib.metadata.version') as version:
                if isinstance(result, Exception):
                    version.side_effect = result
                else:
                    version.return_value = result
                with self.assertRaisesRegex(ValueError, 'requires catboost==1.2.10'):
                    load_bundle('unused', manifest)
        load.assert_not_called()

    @patch('triage_pipeline.build_text_features', side_effect=lambda a, t: np.zeros((len(t),64)))
    def test_paired_input_preserves_original_complaint(self, encode):
        from src.sapbert_serving import encoder_input
        art=self.artifacts();art['manifest']={'text_input':'concept_and_complaint'}
        frame=pd.DataFrame({'Complaint_Text':['Chest pain'],'Raw_Complaint':['seena mein dard kal raat se']})
        predict_frame(art,frame)
        expected='Chest pain [SEP] seena mein dard kal raat se'
        self.assertEqual(encode.call_args.args[1],[expected])
        self.assertEqual(encoder_input('Chest pain',frame.Raw_Complaint[0],art['manifest']),expected)
        with self.assertRaisesRegex(ValueError,'Raw_Complaint'):
            predict_frame(art,frame.drop(columns='Raw_Complaint'))

    def test_batch_stage_export_matches_paired_encoder_input(self):
        from triage_gui import TriageGUI
        gui=SimpleNamespace(active_artifacts=lambda:{'manifest':{'text_input':'concept_and_complaint','text_representation':'embeddings_raw'}})
        result=TriageGUI._stage_columns(gui,['seena mein dard'],['Chest pain'])
        self.assertEqual(result['Text_Encoded'],['Chest pain [SEP] seena mein dard'])

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

    def test_pca_manifest_dimension_mismatch_is_rejected(self):
        import json
        from triage_pipeline import resolve_model_dir
        directory, _ = resolve_model_dir()
        manifest = json.loads((Path(directory) / 'model_manifest.json').read_text())
        manifest['projected_embedding_dim'] = 128 if manifest['projected_embedding_dim'] == 64 else 64
        with self.assertRaisesRegex(ValueError, 'PCA dimensions'):
            load_bundle(directory, manifest)

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

    @patch('triage_pipeline.build_text_features', side_effect=lambda a, t: np.zeros((len(t),64)))
    def test_detail_model_uses_original_complaint_and_refuses_missing_raw(self, encode):
        from src.complaint_details import detail_matrix
        from sklearn.preprocessing import StandardScaler
        art = self.artifacts()
        art['manifest'] = {'text_details': {'source': 'complaint_details'}}
        scaler = StandardScaler().fit(detail_matrix(['aadhay ghante se', 'for an hour']))
        art['detail_scaler'] = scaler
        captured = []
        art['model'] = SimpleNamespace(classes_=np.arange(4), predict_proba=lambda x: (captured.append(x) or np.tile([.1,.2,.3,.4], (len(x),1))))
        frame = pd.DataFrame({'Complaint_Text':['pain for an hour'], 'Raw_Complaint':['seena dard aadhay ghante se']})
        predict_frame(art, frame)
        np.testing.assert_allclose(captured[0][:,-19:], scaler.transform(detail_matrix(frame.Raw_Complaint)))
        with self.assertRaisesRegex(ValueError, 'Raw_Complaint is required'):
            predict_frame(art, frame.drop(columns='Raw_Complaint'))


if __name__ == '__main__':
    unittest.main()
