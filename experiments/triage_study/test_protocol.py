"""Small adversarial checks independent of private input data and model downloads."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import prepare_data
from research_engine import Features


class ProtocolTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(prepare_data.normal(pd.NA), "")
        self.assertEqual(prepare_data.normal("  Chest, PAIN! "), "chest pain")

    def test_group_isolation_and_frozen_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            rows = []
            for i in range(120):
                row = {c: 10 for c in prepare_data.COLS}
                row.update(
                    Gender="Male",
                    Mode_of_Arrival="Walk-in",
                    AVPU="A",
                    ECG_Status="Normal",
                    Labels=i % 3 + 1,
                    chief_complaint=f"complaint {i}",
                    Clinical_Concept=f"concept {i}",
                )
                rows.append(row)
            # Different complaints share a concept, linking their groups.
            rows[1]["Clinical_Concept"] = rows[0]["Clinical_Concept"]
            source = root / "input.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            with patch.object(prepare_data, "HERE", output), contextlib.redirect_stdout(
                io.StringIO()
            ):
                prepare_data.main(source)
                df = pd.read_csv(output / "dataset_with_splits.csv")
                self.assertEqual(df.group.iloc[0], df.group.iloc[1])
                self.assertTrue(
                    set(df[df.partition == "test"].group).isdisjoint(
                        df[df.partition == "development"].group
                    )
                )
                original = (output / "split_manifest.csv").read_bytes()
                prepare_data.main(source)
                self.assertEqual(original, (output / "split_manifest.csv").read_bytes())
                source.write_text(source.read_text() + "\n")
                with self.assertRaises(ValueError):
                    prepare_data.main(source)

    def test_preprocessing_ignores_labels_and_validation_extremes(self):
        frame = pd.DataFrame({c: [10.0, 20.0, 30.0] for c in prepare_data.NUM})
        frame = frame.assign(
            Gender=["M", "F", "M"],
            Mode_of_Arrival="Walk-in",
            ECG_Status="Normal",
            AVPU="A",
            Labels=[1, 2, 3],
        )
        features = Features().fit(frame)
        before = features.transform(frame)
        validation = frame.copy()
        validation["Age"] = 9999
        validation["Gender"] = "new"
        validation["Labels"] = 999
        features.transform(validation)
        np.testing.assert_allclose(before, features.transform(frame))
        np.testing.assert_allclose(before, features.transform(frame.assign(Labels=999)))


if __name__ == "__main__":
    unittest.main()
