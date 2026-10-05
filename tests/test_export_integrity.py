"""A prior verification file must not authorize changed export inputs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/triage_study'))
from export_improved_bundle import export
from cache_integrity import file_sha256


class ExportIntegrityTests(unittest.TestCase):
    def test_changed_data_or_embeddings_rejected_before_loading_model(self):
        for changed in ('dataset_with_splits.csv', 'emb_sapbert_concept.npy'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root/'selection.json').write_text('{}')
                (root/'verification.json').write_text('{"status":"passed"}')
                protocol = {}
                for name, key in [('dataset_with_splits.csv', 'source_data_sha256'), ('emb_sapbert_concept.npy', 'embedding_sha256')]:
                    (root/name).write_bytes(b'verified')
                    protocol[key] = file_sha256(root/name)
                (root/'protocol.json').write_text(json.dumps(protocol))
                (root/changed).write_bytes(b'changed')
                with patch('export_improved_bundle.joblib.load') as load:
                    with self.assertRaisesRegex(ValueError, 'Export source changed'):
                        export(root, root, root, root/'out')
                    load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
