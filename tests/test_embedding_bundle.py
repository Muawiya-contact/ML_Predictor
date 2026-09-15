"""Embedding bundle contracts and translation-aware batch routing."""
import json
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triage_pipeline import load_artifacts, read_manifest, resolve_model_dir
from predict_batch import predict_translated_dataframe


class EmbeddingBundleTests(unittest.TestCase):
    def test_default_bundle_matches_classifier_dimensions(self):
        model_dir, _ = resolve_model_dir()
        art = load_artifacts(model_dir)
        self.assertEqual(Path(model_dir).name, 'triage_model_embedding_english')
        self.assertEqual(art['blocks'], ('embedding',))
        self.assertEqual(art['model'].n_features_in_, 410)
        self.assertEqual([b['dim'] for b in art['manifest']['feature_blocks']], [26, 384])

    def test_missing_bundle_fails_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_model_dir(directory)

    def test_incompatible_layout_is_rejected(self):
        art = load_artifacts()
        with tempfile.TemporaryDirectory() as directory:
            manifest = dict(art['manifest'], feature_blocks=[{'name': 'other', 'dim': 410}])
            Path(directory, 'model_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_manifest(directory)

    def test_batch_only_scores_translated_rows_that_pass_gate(self):
        art = load_artifacts()
        frame = pd.DataFrame({
            'Complaint_Text': ['seena mein dard', 'pait mein dard', 'n/a'],
            'Predicted_Triage_Level': [1, 1, 1],
            'Confidence': [0.99, 0.99, 0.99],
        }, index=[7, 7, 9])
        calls = []
        def score(_art, rows):
            calls.extend(rows['Complaint_Text'].tolist())
            result = rows.copy()
            result['Predicted_Triage_Level'] = 2
            result['Confidence'] = 0.7
            result['Notes'] = ''
            return result, []
        with patch('src.offline_pipeline.ollama_models', return_value=['llama3.2']), \
             patch('src.offline_pipeline.translate_roman_urdu', side_effect=['Chest pain', 'Chest pain', None]), \
             patch('predict_batch.predict_dataframe', side_effect=score):
            result = predict_translated_dataframe(art, frame)
        self.assertEqual(calls, ['Chest pain'])
        self.assertEqual(result.index.tolist(), [7, 7, 9])
        self.assertEqual(result['Gate_Status'].tolist(), ['PASS', 'BLOCKED', 'NOT TRANSLATED'])
        self.assertEqual(result['Complaint_Text'].tolist(), frame['Complaint_Text'].tolist())
        self.assertTrue(result['Predicted_Triage_Level'].iloc[1:].isna().all())
        self.assertTrue(result['Confidence'].iloc[1:].isna().all())

    def test_empty_batch_needs_no_translator(self):
        with patch('src.offline_pipeline.ollama_models') as models:
            result = predict_translated_dataframe(load_artifacts(), pd.DataFrame({'Complaint_Text': []}))
        models.assert_not_called()
        self.assertEqual(len(result), 0)
        self.assertIn('Gate_Status', result)

    def test_training_exports_only_embedding_configurations(self):
        import train_embedding_pipeline as training
        from triage_pipeline import build_text_features
        class SmallEncoder:
            def get_sentence_embedding_dimension(self):
                return 4
            def encode(self, texts, **kwargs):
                vectors = np.array([list(hashlib.sha256(t.encode()).digest()[:4]) for t in texts], dtype=float)
                return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        source = Path(__file__).resolve().parents[1] / 'cardiac_english_2252.csv'
        sample = pd.read_csv(source).groupby('Triage_Level').head(10)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory, 'sample.csv')
            output = Path(directory, 'bundle')
            sample.to_csv(data, index=False)
            argv = ['train_embedding_pipeline.py', '--data', str(data), '--out-dir', str(output)]
            with patch.object(sys, 'argv', argv), \
                 patch.object(training, 'load_embedding_model', return_value=SmallEncoder()), \
                 contextlib.redirect_stdout(io.StringIO()):
                training.main()
            art = load_artifacts(str(output))
            art['encoder'] = SmallEncoder()
            self.assertEqual(build_text_features(art, ['Chest pain']).shape, (1, 4))
            metrics = json.loads((output / 'triage_metrics.json').read_text())
            self.assertEqual([m['text_representation'] for m in metrics['all_methods']],
                             ['embeddings_raw', 'embeddings_preprocessed'])
            self.assertEqual({p.name for p in output.glob('*.pkl')},
                             {'model.pkl', 'scaler.pkl', 'gender_enc.pkl', 'mode_enc.pkl', 'avpu_enc.pkl', 'ecg_enc.pkl'})
            self.assertTrue((output / 'learned_stopwords.json').is_file())
            self.assertTrue((output / 'embedding_pipeline_results.csv').is_file())


if __name__ == '__main__':
    unittest.main()
