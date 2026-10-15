"""Refused GUI batch rows never reach inference; exports retain original text."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from triage_gui import TriageGUI


class GuiBatchSafetyTests(unittest.TestCase):
    def run_batch(self, inputs, statuses):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'patients.csv'
            pd.DataFrame({'Complaint_Text': inputs, 'Raw_Complaint': ['stale'] * len(inputs),
                          'Confidence': ['99%'] * len(inputs)}).to_csv(path, index=False)
            gui = SimpleNamespace(_batch_target=str(path), _batch_progress={},
                                  active_artifacts=lambda: {})
            answers = iter(statuses)
            def translate(text, allow_blocked):
                status = next(answers)
                gui._last_gate = (status == 'PASS', ['mismatch'], None)
                return ('English: ' + text, None) if status != 'FAILED' else (None, 'Unavailable')
            gui.translate_complaint = translate
            gui._stage_columns = lambda raw, english: {'Input_Raw': raw, 'Text_Encoded': english}
            seen = []
            def predict(art, frame):
                seen.extend(frame.Raw_Complaint.tolist())
                out = frame.copy()
                out['Predicted_Triage_Level'] = 1
                out['Confidence'] = '80%'
                out['Notes'] = ''
                return out, []
            with patch('triage_pipeline.predict_dataframe', side_effect=predict):
                result, base = TriageGUI._batch_worker(gui)
            saved = pd.read_csv(base + '.csv')
            self.assertEqual(saved.Complaint_Text.tolist(), inputs)
            self.assertEqual(saved.Raw_Complaint.tolist(), inputs)
            return result, seen

    def test_only_accepted_rows_scored_and_original_preserved(self):
        result, seen = self.run_batch(['raw one', 'raw two', 'raw three'], ['PASS', 'BLOCKED', 'FAILED'])
        self.assertEqual(seen, ['raw one'])
        self.assertTrue(result.loc[1:, 'Confidence'].isna().all())
        self.assertTrue(result.loc[1:, 'Text_Encoded'].isna().all())
        self.assertTrue(result.loc[1:, 'Notes'].str.startswith('NOT SCORED').all())

    def test_all_failed_rows_export_reasons_without_scoring(self):
        result, seen = self.run_batch(['raw one'], ['FAILED'])
        self.assertEqual(seen, [])
        self.assertTrue(result.Confidence.isna().all())
        self.assertIn('Unavailable', result.Notes.iloc[0])


if __name__ == '__main__':
    unittest.main()
