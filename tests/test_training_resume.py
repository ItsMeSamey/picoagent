import json
import tempfile
import unittest
from pathlib import Path

from picoagent.training.provenance import preflight_resume, write_json


class ResumePreflightTests(unittest.TestCase):
    def test_newer_complete_or_partial_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "checkpoint-1"
            old.mkdir()
            (root / "checkpoint-2").mkdir()
            with self.assertRaisesRegex(ValueError, "newer checkpoint"):
                preflight_resume(root, old, max_steps=3)

    def test_final_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            (root / "final-model").mkdir()
            with self.assertRaisesRegex(ValueError, "Final artifacts"):
                preflight_resume(root, checkpoint, max_steps=3)

    def test_latest_interrupted_checkpoint_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint-2"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
            preflight_resume(root, checkpoint, max_steps=3)
            with self.assertRaisesRegex(ValueError, "training budget"):
                preflight_resume(root, checkpoint, max_steps=2)

    def test_atomic_exclusive_manifest_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            write_json(path, {"original": True}, exclusive=True)
            with self.assertRaises(FileExistsError):
                write_json(path, {"overwritten": True}, exclusive=True)
            self.assertEqual(json.loads(path.read_text()), {"original": True})
            self.assertEqual([p.name for p in Path(temp).iterdir()], ["manifest.json"])

class ModelSnapshotRevisionTests(unittest.TestCase):
    def test_reads_cache_revision_without_following_blob_symlink(self):
        from picoagent.training.provenance import snapshot_revision
        commit = "a" * 40
        self.assertEqual(snapshot_revision(f"/cache/models--org--model/snapshots/{commit}/config.json"), commit)

    def test_rejects_branch_or_non_snapshot_identity(self):
        from picoagent.training.provenance import snapshot_revision
        for path in ("/cache/model/config.json", "/cache/model/snapshots/main/config.json", "/cache/model/snapshots/shortsha/config.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                snapshot_revision(path)
