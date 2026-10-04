"""The CLI shares current model routing and the explicit no-complaint state."""
import unittest
from unittest.mock import patch
import pandas as pd
from run_inference import infer


class CurrentCLITests(unittest.TestCase):
    def test_missing_complaint_never_calls_translator_or_classifier(self):
        with patch('run_inference.predict_translated_dataframe') as predict:
            for text in ['', 'X', 'n/a']:
                result = infer({}, text)
                self.assertEqual(result['Confidence'], '50%')
                self.assertIsNone(result['Predicted_Triage_Level'])
                self.assertIn('placeholder', result['Notes'])
            predict.assert_not_called()

    def test_patient_values_and_requested_translator_reach_shared_pipeline(self):
        output = pd.DataFrame({'Predicted_Triage_Level': [0], 'Confidence': ['90%']})
        with patch('run_inference.predict_translated_dataframe', return_value=output) as predict:
            result = infer({'manifest': {}}, 'seena mein dard', {'Age': 65}, 'qwen2.5')
        self.assertEqual(result['Predicted_Triage_Level'], 0)
        self.assertEqual(predict.call_args.args[1].Age.iloc[0], 65)
        self.assertEqual(predict.call_args.kwargs['model'], 'qwen2.5')

    def test_refused_row_serializes_no_level(self):
        output = pd.DataFrame({'Predicted_Triage_Level': [float('nan')], 'Gate_Status': ['BLOCKED']})
        with patch('run_inference.predict_translated_dataframe', return_value=output):
            result = infer({}, 'pait mein dard')
        self.assertIsNone(result['Predicted_Triage_Level'])
        self.assertEqual(result['Gate_Status'], 'BLOCKED')


if __name__ == '__main__':
    unittest.main()
