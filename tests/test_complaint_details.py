import unittest
import numpy as np
from src.complaint_details import duration_minutes, detail_matrix, FEATURE_NAMES


class ComplaintDetailsTests(unittest.TestCase):
    def test_half_hour_is_not_double_counted_as_an_hour(self):
        self.assertEqual(duration_minutes('pain for half an hour'), (30.0, False))
        self.assertEqual(duration_minutes('seena dard aadhay ghante se'), (30.0, False))
        self.assertEqual(duration_minutes('for an hour'), (60.0, False))

    def test_multilingual_units_and_ambiguous_duration(self):
        self.assertEqual(duration_minutes('do ghante se'), (120.0, False))
        self.assertEqual(duration_minutes('teen din se'), (4320.0, False))
        self.assertEqual(duration_minutes('pain 2 hours, sweating 10 minutes'), (None, True))
        self.assertEqual(duration_minutes('since morning'), (None, False))

    def test_features_are_mentions_and_expose_negation(self):
        row = dict(zip(FEATURE_NAMES, detail_matrix(['no sweating, family history'])[0]))
        self.assertEqual(row['sweating_word'], 1)
        self.assertEqual(row['negation_word'], 1)
        self.assertEqual(row['family_word'], 1)
        self.assertEqual(row['duration_known'], 0)

    def test_empty_and_batch_outputs_are_finite(self):
        self.assertEqual(detail_matrix([]).shape, (0, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(detail_matrix([None, '', 'mild pain'])).all())
