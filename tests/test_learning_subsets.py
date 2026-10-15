import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/triage_study'))
from learning_and_detail_audit import nested_group_subsets


class LearningSubsetTests(unittest.TestCase):
    def test_subsets_keep_groups_whole_and_nested(self):
        frame = pd.DataFrame([dict(group=g, row_id=2*g+i, Labels=g % 4) for g in range(100) for i in range(2)])
        sizes = nested_group_subsets(frame, 9)
        previous = set()
        for subset in sizes.values():
            self.assertTrue(previous <= set(subset.row_id))
            self.assertTrue((subset.groupby('group').size() == 2).all())
            previous = set(subset.row_id)
        self.assertEqual(previous, set(frame.row_id))
        pd.testing.assert_frame_equal(sizes[.5], nested_group_subsets(frame, 9)[.5])
