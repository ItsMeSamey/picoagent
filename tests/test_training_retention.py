import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from picoagent.training.data import sha256_file
from picoagent.training.provenance import checkpoint_evidence
from picoagent.training.retention import mirror_and_prune


class TrainingRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "runtime"
        self.durable = self.root / "durable"
        self.run.mkdir()
        self.durable.mkdir()
        (self.run / "run_manifest.json").write_text(json.dumps({"run": "retention-test"}))
        self.run_hash = sha256_file(self.run / "run_manifest.json")

    def checkpoint(self, step):
        checkpoint = self.run / f"checkpoint-{step}"
        checkpoint.mkdir()
        for filename in (
            "model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt",
            "rng_state.pth", "tokenizer.json",
        ):
            (checkpoint / filename).write_text(f"test fixture {step}: {filename}")
        checkpoint_evidence(checkpoint, self.run_hash)
        return checkpoint

    def mirror(self, checkpoint, best=None, durable=True):
        return mirror_and_prune(
            self.run, self.durable if durable else None, checkpoint, best,
        )

    def checkpoint_names(self, root):
        return {path.name for path in root.glob("checkpoint-*")}

    def test_retains_latest_two_and_best_and_all_durable_copies(self):
        for step in (1, 2, 10, 20, 30):
            result = self.mirror(self.checkpoint(step), "checkpoint-1")
        self.assertEqual(self.checkpoint_names(self.run), {
            "checkpoint-1", "checkpoint-20", "checkpoint-30",
        })
        self.assertEqual(self.checkpoint_names(self.durable), {
            "checkpoint-1", "checkpoint-2", "checkpoint-10", "checkpoint-20", "checkpoint-30",
        })
        self.assertEqual(result["pruned"], ["checkpoint-10"])
        self.assertEqual(
            (self.run / "run_manifest.json").read_bytes(),
            (self.durable / "run_manifest.json").read_bytes(),
        )
        for name in result["retained"]:
            for source in (self.run / name).iterdir():
                self.assertEqual(source.read_bytes(), (self.durable / name / source.name).read_bytes())

    def test_no_durable_config_means_no_pruning(self):
        for step in range(1, 5):
            result = self.mirror(self.checkpoint(step), durable=False)
        self.assertFalse(result["enabled"])
        self.assertEqual(result["pruned"], [])
        self.assertEqual(len(self.checkpoint_names(self.run)), 4)
        self.assertEqual(list(self.durable.iterdir()), [])

    def test_unmirrored_older_checkpoints_are_never_pruned(self):
        old = self.checkpoint(1)
        self.checkpoint(2)
        self.mirror(self.checkpoint(3))
        self.assertTrue(old.exists())
        self.assertFalse((self.durable / old.name).exists())

    def test_corrupt_new_copy_prevents_any_pruning(self):
        self.mirror(self.checkpoint(1))
        self.mirror(self.checkpoint(2))
        current = self.checkpoint(3)
        copytree = shutil.copytree

        def corrupt_copy(source, destination, **kwargs):
            result = copytree(source, destination, **kwargs)
            (Path(destination) / "optimizer.pt").write_text("corrupt")
            return result

        with mock.patch("picoagent.training.retention.shutil.copytree", side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, "integrity"):
                self.mirror(current)
        self.assertEqual(len(self.checkpoint_names(self.run)), 3)
        self.assertFalse((self.durable / current.name).exists())
        self.assertFalse(list(self.durable.glob(".checkpoint-3.tmp-*")))

    def test_all_deletion_candidates_verified_before_any_deletion(self):
        for step in range(1, 5):
            self.checkpoint(step)
        # Prepare two older mirrors without invoking retention.
        for step in (1, 2):
            shutil.copytree(self.run / f"checkpoint-{step}", self.durable / f"checkpoint-{step}")
        (self.durable / "checkpoint-2/optimizer.pt").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.mirror(self.run / "checkpoint-4")
        self.assertEqual(len(self.checkpoint_names(self.run)), 4)
        self.assertTrue((self.durable / "checkpoint-4").exists())

    def test_interrupted_temporary_copy_is_not_a_verified_mirror(self):
        old = self.checkpoint(1)
        self.checkpoint(2)
        interrupted = self.durable / ".checkpoint-1.tmp-interrupted"
        interrupted.mkdir()
        (interrupted / "optimizer.pt").write_text("partial")
        self.mirror(self.checkpoint(3))
        self.assertTrue(old.exists())
        self.assertTrue(interrupted.exists())
        # A later complete copy can safely commit and then prune the old source.
        result = self.mirror(old)
        self.assertEqual(result["pruned"], ["checkpoint-1"])
        self.assertTrue((self.durable / old.name / "checkpoint_manifest.json").exists())

    def test_interrupted_copy_error_leaves_runtime_untouched(self):
        for step in (1, 2):
            self.mirror(self.checkpoint(step))
        current = self.checkpoint(3)
        with mock.patch("picoagent.training.retention.shutil.copytree", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.mirror(current)
        self.assertEqual(len(self.checkpoint_names(self.run)), 3)
        self.assertFalse((self.durable / current.name).exists())
        self.assertFalse((self.run / ".checkpoint-retention.lock").exists())

    def test_conflicting_existing_destination_is_never_overwritten(self):
        current = self.checkpoint(1)
        target = self.durable / current.name
        shutil.copytree(current, target)
        (target / "optimizer.pt").write_text("existing other data")
        with self.assertRaises(ValueError):
            self.mirror(current)
        self.assertEqual((target / "optimizer.pt").read_text(), "existing other data")
        self.assertTrue(current.exists())

    def test_identical_existing_copy_is_idempotent(self):
        checkpoint = self.checkpoint(1)
        first = self.mirror(checkpoint)
        self.assertEqual(self.mirror(checkpoint), first)

    def test_conflicting_durable_run_manifest_is_never_overwritten(self):
        current = self.checkpoint(1)
        manifest = self.durable / "run_manifest.json"
        manifest.write_text("another run")
        with self.assertRaisesRegex(ValueError, "run manifest conflicts"):
            self.mirror(current)
        self.assertEqual(manifest.read_text(), "another run")
        self.assertTrue(current.exists())
        self.assertFalse((self.durable / current.name).exists())

    def test_source_mutation_during_copy_never_commits_or_prunes(self):
        for step in (1, 2):
            self.mirror(self.checkpoint(step))
        current = self.checkpoint(3)
        copytree = shutil.copytree

        def change_source(source, destination, **kwargs):
            result = copytree(source, destination, **kwargs)
            (Path(source) / "optimizer.pt").write_text("concurrent write")
            return result

        with mock.patch("picoagent.training.retention.shutil.copytree", side_effect=change_source):
            with self.assertRaisesRegex(ValueError, "integrity"):
                self.mirror(current)
        self.assertEqual(len(self.checkpoint_names(self.run)), 3)
        self.assertFalse((self.durable / current.name).exists())

    def test_best_absolute_path_and_non_checkpoint_artifacts_preserved(self):
        trace = self.run / "traces"
        trace.mkdir()
        (trace / "trace.jsonl").write_text("immutable training trace")
        unknown = self.run / "checkpoint-not-a-step"
        unknown.mkdir()
        for step in (1, 2, 3, 4):
            self.mirror(self.checkpoint(step), str(self.run / "checkpoint-1"))
        self.assertTrue((self.run / "checkpoint-1").exists())
        self.assertTrue(unknown.exists())
        self.assertEqual((trace / "trace.jsonl").read_text(), "immutable training trace")

    def test_manifest_corruption_and_missing_resume_state_fail_closed(self):
        for missing in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            with self.subTest(missing=missing):
                checkpoint = self.checkpoint(1)
                (checkpoint / missing).unlink()
                (checkpoint / "checkpoint_manifest.json").unlink()
                checkpoint_evidence(checkpoint, self.run_hash)
                with self.assertRaisesRegex(ValueError, "resume state|RNG"):
                    self.mirror(checkpoint)
                shutil.rmtree(checkpoint)
        checkpoint = self.checkpoint(1)
        manifest = checkpoint / "checkpoint_manifest.json"
        manifest.write_text(manifest.read_text().replace(self.run_hash, "0" * 64))
        with self.assertRaisesRegex(ValueError, "another or unknown run"):
            self.mirror(checkpoint)

    def test_partial_newest_checkpoint_does_not_displace_last_complete_checkpoint(self):
        for step in (1, 2):
            self.mirror(self.checkpoint(step))
        current = self.checkpoint(3)
        (self.run / "checkpoint-4").mkdir()
        with self.assertRaises(FileNotFoundError):
            self.mirror(current)
        self.assertEqual(len(self.checkpoint_names(self.run)), 4)

    def test_hashes_include_files_normally_excluded_from_provenance(self):
        current = self.checkpoint(1)
        cache = current / "__pycache__"
        cache.mkdir()
        (cache / "extra").write_text("original")
        copytree = shutil.copytree

        def corrupt_copy(source, destination, **kwargs):
            result = copytree(source, destination, **kwargs)
            (Path(destination) / "__pycache__/extra").write_text("corrupt")
            return result

        # copytree recurses through the patched symbol, so use a wraps/side effect
        # only on the outer call by replacing the patch during the real copy.
        def outer_copy(source, destination, **kwargs):
            with mock.patch("picoagent.training.retention.shutil.copytree", copytree):
                return corrupt_copy(source, destination, **kwargs)

        with mock.patch("picoagent.training.retention.shutil.copytree", side_effect=outer_copy):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                self.mirror(current)

    def test_unsafe_paths_and_unconfigured_mount_fail_closed(self):
        current = self.checkpoint(1)
        for destination in (self.run, self.run / "backup", self.root, self.root / "missing"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                mirror_and_prune(self.run, destination, current, None)
        self.assertFalse((self.root / "missing").exists())
        (current / "external-link").symlink_to(self.run / "run_manifest.json")
        with self.assertRaisesRegex(ValueError, "link"):
            self.mirror(current)

    def test_flush_error_never_prunes(self):
        for step in (1, 2):
            self.mirror(self.checkpoint(step))
        current = self.checkpoint(3)
        with mock.patch("picoagent.training.retention.os.fsync", side_effect=OSError("flush failed")):
            with self.assertRaisesRegex(OSError, "flush failed"):
                self.mirror(current)
        self.assertEqual(len(self.checkpoint_names(self.run)), 3)

    def test_stale_lock_blocks_instead_of_deleting(self):
        current = self.checkpoint(1)
        lock = self.durable / ".checkpoint-retention.lock"
        lock.write_text("stale")
        with self.assertRaises(FileExistsError):
            self.mirror(current)
        self.assertEqual(lock.read_text(), "stale")
        self.assertTrue(current.exists())
        self.assertFalse((self.run / ".checkpoint-retention.lock").exists())

    def test_retention_cannot_reduce_below_two(self):
        current = self.checkpoint(1)
        for count in (0, 1, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                mirror_and_prune(self.run, self.durable, current, None, count)


if __name__ == "__main__":
    unittest.main()
