"""Checks ID alignment and leakage boundaries, without using research outcomes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from experiments.compare_classifiers import load_inputs, representations, structured_features, NUMERIC, CATEGORICAL


class ResearchComparisonTests(unittest.TestCase):
    def test_embeddings_follow_ids_not_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({'row_id': ['a', 'b'], 'Triage_Level': [1, 2]}).to_csv(root/'data.csv', index=False)
            np.savez(root/'vectors.npz', embeddings=np.array([np.ones(768), np.zeros(768)]), row_ids=['b', 'a'])
            _, vectors, _ = load_inputs(root/'data.csv', root/'vectors.npz', 'row_id', 'Triage_Level')
            self.assertTrue(np.all(vectors[0] == 0))
            self.assertTrue(np.all(vectors[1] == 1))

    def test_pca_never_fits_on_test_rows(self):
        data = np.random.default_rng(42).normal(size=(90, 768))
        train, test = np.arange(70), np.arange(70, 90)
        _, first = representations(data, train, test)
        data[test] += 10000
        _, second = representations(data, train, test)
        np.testing.assert_allclose(first.mean_, second.mean_)
        np.testing.assert_allclose(first.components_, second.components_)

    def test_50_rows_cannot_be_relabelled_as_pca64(self):
        with self.assertRaisesRegex(ValueError, '65 training rows'):
            representations(np.zeros((50, 768)), np.arange(40), np.arange(40, 50))

    def test_test_only_category_does_not_expand_training_features(self):
        frame = pd.DataFrame({**{c: [1., 2., 3.] for c in NUMERIC},
                              **{c: ['known', 'known', 'unknown'] for c in CATEGORICAL}})
        train, test = structured_features(frame, [0, 1], [2])
        self.assertEqual(train.shape[1], 10)
        np.testing.assert_array_equal(test[0, 6:], np.zeros(4))


if __name__ == '__main__':
    unittest.main()
