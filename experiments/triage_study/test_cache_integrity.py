"""Regression checks for encoder identity and stale downstream caches."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import encode_text
import research_engine
from cache_integrity import file_sha256, verify_study


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "hf"
        (self.cache / "snapshots/rev-a").mkdir(parents=True)
        (self.cache / "refs").mkdir()
        (self.cache / "refs/main").write_text("rev-a")
        self.df = pd.DataFrame({"row_id": [0, 1], "Clinical_Concept": ["pain", "ache"]})
        for name in [
            "trimmed_dataset.csv",
            "dataset_with_splits.csv",
            "split_manifest.csv",
        ]:
            self.df.to_csv(self.root / name, index=False)
        self.array = self.root / "emb_sapbert_concept.npy"
        self.meta = self.root / "emb_sapbert_concept.json"
        np.save(self.array, np.ones((2, 768), dtype="float32"))
        self.metadata = dict(
            name="sapbert_concept",
            text_column="Clinical_Concept",
            revision="rev-a",
            pooling="cls",
            max_token_length=64,
            normalized=True,
            shape=[2, 768],
            text_sha256=hashlib.sha256(
                json.dumps(["pain", "ache"], ensure_ascii=False).encode()
            ).hexdigest(),
            array_sha256=file_sha256(self.array),
        )
        self.save_metadata()

    def save_metadata(self):
        self.meta.write_text(json.dumps(self.metadata))

    def encode(self):
        with patch.object(encode_text, "HERE", self.root), patch.dict(
            encode_text.MODELS,
            {"sapbert_concept": ("Clinical_Concept", self.cache, "cls", 64)},
        ), contextlib.redirect_stdout(io.StringIO()):
            encode_text.main("sapbert_concept")

    def test_matching_encoder_reuses_verified_vectors(self):
        before = self.array.read_bytes()
        self.encode()
        self.assertEqual(before, self.array.read_bytes())

    def test_encoder_revision_change_is_rejected(self):
        (self.cache / "refs/main").write_text("rev-b")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.encode()
        self.assertEqual("rev-a", json.loads(self.meta.read_text())["revision"])

    def test_encoder_settings_changes_are_rejected(self):
        for key, value in [
            ("pooling", "mean"),
            ("max_token_length", 128),
            ("normalized", False),
            ("shape", [2, 384]),
        ]:
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = value
                self.save_metadata()
                with self.assertRaises(ValueError):
                    self.encode()
                self.metadata[key] = original
                self.save_metadata()

    def test_corrupt_or_partial_array_is_rejected(self):
        self.array.write_bytes(b"partial numpy file")
        with self.assertRaises(ValueError):
            self.encode()

    def test_changed_array_content_is_rejected(self):
        np.save(self.array, np.zeros((2, 768), dtype="float32"))
        with self.assertRaises(ValueError):
            self.encode()

    def test_matrix_resume_verifies_embedding_content(self):
        tr, va = np.array([0]), np.array([1])
        config = dict(view="text", encoder="sapbert_concept", pca=0)
        with patch.object(research_engine, "HERE", self.root):
            first, _ = research_engine.matrices(self.df, tr, va, config)
            again, _ = research_engine.matrices(self.df, tr, va, config)
            np.testing.assert_array_equal(first, again)
            np.save(self.array, np.full((2, 768), 10, dtype="float32"))
            # Even a legitimately regenerated array with updated metadata cannot
            # be combined with the prior study's cached feature matrices.
            self.metadata["array_sha256"] = file_sha256(self.array)
            self.save_metadata()
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                research_engine.matrices(self.df, tr, va, config)

    def test_cv_resume_rejects_changed_splits(self):
        verify_study(self.root)
        config = {
            "classifier": "logreg",
            "features": {"view": "structured"},
            "params": {},
        }
        (self.root / "cv3").mkdir()
        (self.root / "cv3" / f"{research_engine.ident(config)}.json").write_text(
            '{"cached": true}'
        )
        with patch.object(research_engine, "HERE", self.root):
            self.assertEqual({"cached": True}, research_engine.evaluate_cv(config))
            (self.root / "dataset_with_splits.csv").write_text("modified assignments")
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                research_engine.evaluate_cv(config)

    def test_legacy_results_cannot_acquire_new_identity(self):
        (self.root / "cv3").mkdir()
        with self.assertRaisesRegex(ValueError, "no input manifest"):
            verify_study(self.root)
        self.assertFalse((self.root / "study_inputs.json").exists())

    def test_final_result_shortcut_also_checks_inputs(self):
        verify_study(self.root)
        (self.root / "final_results.json").write_text("[]")
        (self.root / "split_manifest.csv").write_text("changed")
        with patch.object(research_engine, "HERE", self.root):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                research_engine.finalize()

    def test_leaderboard_rejects_changed_inputs(self):
        verify_study(self.root)
        (self.root / "trimmed_dataset.csv").write_text("changed")
        with patch.object(research_engine, "HERE", self.root):
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                research_engine.export_board(3)


if __name__ == "__main__":
    unittest.main()
