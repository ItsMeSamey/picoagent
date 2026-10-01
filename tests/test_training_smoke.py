"""Opt-in real ML checks: PICOAGENT_RUN_ML_TESTS=1 python -m pytest ...

Ordinary dependency-free test runs skip this module's actual ML imports/work.
All models are newly randomized and fixtures explicitly smoke-only.
"""
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("PICOAGENT_RUN_ML_TESTS") == "1", "optional local torch/Transformers integration smoke")
class TrainingMLSmokeTests(unittest.TestCase):
    def test_segmented_resume_matches_uninterrupted_full_schedule(self):
        import torch
        from safetensors.torch import load_file
        from picoagent.training import train
        from picoagent.training.config import TrainingConfig
        from picoagent.training.smoke import run_smoke

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = run_smoke(root / "reference", device="cpu", train_records=3,
                                  gradient_accumulation_steps=1, save_steps=100,
                                  checkpoint_interval_seconds=3600.0)
            first = run_smoke(root / "segmented", device="cpu", segment_steps=1,
                              train_records=3, gradient_accumulation_steps=1, save_steps=100,
                              checkpoint_interval_seconds=3600.0)
            run = root / "segmented/run"
            self.assertEqual(first["status"], "paused")
            self.assertEqual(first["global_step"], 1)
            self.assertEqual(first["planned_global_steps"], 2)
            self.assertEqual(sorted(path.name for path in run.glob("checkpoint-*")), ["checkpoint-1"])
            self.assertFalse((run / "final-model").exists())
            self.assertFalse((run / "metrics.json").exists())
            self.assertFalse((run / "final_artifacts.json").exists())
            paused_status = json.loads((run / "run_status.json").read_text())
            self.assertEqual(paused_status["status"], "paused")
            self.assertEqual(paused_status["checkpoint"], "checkpoint-1")
            state = json.loads((run / "checkpoint-1/trainer_state.json").read_text())
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(state["max_steps"], 2)

            config = TrainingConfig.load(root / "segmented/smoke-config.json")
            resumed = train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-1"), segment_steps=1)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["global_step"], 2)
            self.assertEqual(resumed["planned_global_steps"], 2)
            self.assertEqual(sorted(path.name for path in run.glob("checkpoint-*")), ["checkpoint-1", "checkpoint-2"])

            reference_weights = load_file(str(Path(reference["artifact"]) / "model.safetensors"))
            segmented_weights = load_file(str(Path(resumed["artifact"]) / "model.safetensors"))
            self.assertEqual(reference_weights.keys(), segmented_weights.keys())
            self.assertTrue(all(torch.equal(reference_weights[key], segmented_weights[key]) for key in reference_weights))
            reference_scheduler = torch.load(root / "reference/run/checkpoint-2/scheduler.pt", weights_only=False)
            segmented_scheduler = torch.load(run / "checkpoint-2/scheduler.pt", weights_only=False)
            self.assertEqual(reference_scheduler, segmented_scheduler)

            wall = run_smoke(root / "wall-clock-segment", device="cpu", segment_steps=2,
                             train_records=3, gradient_accumulation_steps=1, save_steps=100,
                             checkpoint_interval_seconds=0.000001)
            self.assertEqual(wall["status"], "paused")
            self.assertEqual(wall["global_step"], 1)
            self.assertEqual(wall["planned_global_steps"], 2)
            wall_run = root / "wall-clock-segment/run"
            self.assertEqual(sorted(path.name for path in wall_run.glob("checkpoint-*")), ["checkpoint-1"])

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
