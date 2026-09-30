"""Regression checks for workbook recovery, zero-based labels and leakage controls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1] / 'experiments/triage_study'
sys.path.insert(0, str(STUDY))
import prepare_data
from prepare_four_level_source import prepare, INPUTS, LABELS
from four_level_study import scores


class FourLevelTests(unittest.TestCase):
    def frame(self, rows=80):
        return pd.DataFrame({
            **{c: [40 + i % 2 for i in range(rows)] for c in ['Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Temperature', 'SpO2']},
            'Gender': ['Male'] * rows, 'Mode_of_Arrival': ['Walk-in'] * rows,
            'ECG_Status': ['Normal'] * rows, 'AVPU': ['A'] * rows,
            'chief_complaint': [f'complaint {i//2}' for i in range(rows)],
            'Clinical_Concept': [f'concept {i//2}' for i in range(rows)],
            'Triage_Level': [(i//2) % 4 for i in range(rows)],
            'Triage_Label': [LABELS[(i//2) % 4] for i in range(rows)]})

    def test_missing_concepts_recovered_without_copying_old_targets(self):
        frame = self.frame()
        old = frame.copy()
        old['Labels'] = 99
        frame.loc[0, 'Clinical_Concept'] = None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / 'source.xlsx'
            workbook.write_bytes(b'test workbook identity')
            old.to_csv(root / 'old.csv', index=False)
            with patch('prepare_four_level_source.pd.read_excel', return_value=frame):
                prepare(workbook, root / 'prepared', root / 'old.csv')
            actual = pd.read_csv(root / 'prepared/prepared_source.csv')
            self.assertEqual(actual.Clinical_Concept.iloc[0], 'concept 0')
            np.testing.assert_array_equal(actual.Triage_Level, frame.Triage_Level)
            self.assertEqual(set(actual), set(INPUTS + ['Clinical_Concept', 'Triage_Level']))

    def test_mismatched_reference_refuses_recovery(self):
        frame = self.frame()
        old = frame.copy()
        frame.loc[0, 'Clinical_Concept'] = None
        old.loc[0, 'Age'] = 81
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / 'source.xlsx'
            workbook.write_bytes(b'test')
            old.to_csv(root / 'old.csv', index=False)
            with patch('prepare_four_level_source.pd.read_excel', return_value=frame):
                with self.assertRaisesRegex(ValueError, 'Reference inputs differ'):
                    prepare(workbook, root / 'prepared', root / 'old.csv')
            self.assertFalse((root / 'prepared').exists())

    def test_groups_do_not_cross_holdout_or_cv_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.csv'
            self.frame(400).to_csv(source, index=False)
            output = root / 'study'
            output.mkdir()
            with patch.object(prepare_data, 'HERE', output):
                prepare_data.main(source, target_column='Triage_Level', labels=[0,1,2,3], provenance='test')
            data = pd.read_csv(output / 'dataset_with_splits.csv')
            dev = data[data.partition.eq('development')]
            test = data[data.partition.eq('test')]
            self.assertEqual(data.groupby('group').size().max(), 2)
            self.assertFalse(set(dev.group) & set(test.group))
            self.assertFalse(set(dev.chief_complaint) & set(test.chief_complaint))
            for fold in range(5):
                self.assertFalse(set(dev[dev.cv5.eq(fold)].group) & set(dev[dev.cv5.ne(fold)].group))
            self.assertNotIn('Triage_Label', data)
            audit = json.loads((output / 'data_audit.json').read_text())
            self.assertNotIn('Triage_Level', audit['excluded_columns'])
            self.assertEqual(audit['labels'], [0,1,2,3])

    def test_emergency_recall_and_undertriage_use_zero_as_emergency(self):
        result = scores(np.array([0,0,1,2,3]), np.array([0,1,1,1,3]))
        self.assertEqual(result['emergency_recall'], .5)
        self.assertEqual(result['under_triage_rate'], .2)
        self.assertEqual(result['over_triage_rate'], .2)
        self.assertEqual(result['accuracy'], .6)


if __name__ == '__main__':
    unittest.main()
