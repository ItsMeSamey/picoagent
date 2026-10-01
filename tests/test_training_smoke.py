"""Opt-in real ML checks: PICOAGENT_RUN_ML_TESTS=1 python -m pytest ...

Ordinary dependency-free test runs skip this module's actual ML imports/work.
All models are newly randomized and fixtures explicitly smoke-only.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("PICOAGENT_RUN_ML_TESTS") == "1", "optional local torch/Transformers integration smoke")
class TrainingMLSmokeTests(unittest.TestCase):
    def test_full_sft_interruption_resume_matches_uninterrupted_weights(self):
        import dataclasses
        import torch
        from safetensors.torch import load_file
        from picoagent.training import train
        from picoagent.training.config import TrainingConfig
        from picoagent.training.data import sha256_file
        from picoagent.training.provenance import verify_checkpoint
        from picoagent.training.smoke import run_smoke

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = run_smoke(root / "reference", device="cpu")
            original = train.checkpoint_evidence

            def interrupted_save(checkpoint, run_hash):
                original(checkpoint, run_hash)
                raise InterruptedError("test-controlled interruption after durable local checkpoint")

            with patch.object(train, "checkpoint_evidence", interrupted_save), self.assertRaises(InterruptedError):
                run_smoke(root / "interrupted", device="cpu")
            run = root / "interrupted/run"
            verify_checkpoint(run / "checkpoint-1", sha256_file(run / "run_manifest.json"))
            config = TrainingConfig.load(root / "interrupted/smoke-config.json")
            resumed = train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-1"))
            reference_weights = load_file(str(Path(reference["artifact"]) / "model.safetensors"))
            resumed_weights = load_file(str(Path(resumed["artifact"]) / "model.safetensors"))
            self.assertEqual(reference_weights.keys(), resumed_weights.keys())
            self.assertTrue(all(torch.equal(reference_weights[key], resumed_weights[key]) for key in reference_weights))
            initial_weights = load_file(str(root / "reference/random-initial-model/model.safetensors"))
            self.assertTrue(any(not torch.equal(reference_weights[key], initial_weights[key]) for key in reference_weights))
            self.assertEqual(resumed["device"], "cpu")
            self.assertEqual(resumed["training_mode"], "full")
            self.assertTrue(resumed["smoke_only"])
            self.assertIsNone(resumed["metrics"]["benchmark"])
            # Force the independent wall-clock path while the step interval is
            # larger than the whole run. Checkpoint one must still be sealed.
            wall_config = dataclasses.replace(TrainingConfig.load(root / "reference/smoke-config.json"),
                output_dir=str(root / "wall-clock-run"), save_steps=100,
                checkpoint_interval_seconds=0.000001)
            train.run_training(wall_config)
            wall_run = root / "wall-clock-run"
            verify_checkpoint(wall_run / "checkpoint-1", sha256_file(wall_run / "run_manifest.json"))
