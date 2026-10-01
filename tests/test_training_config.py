import json
import tempfile
import unittest
from pathlib import Path

from picoagent.training.config import TrainingConfig, select_precision


class TrainingConfigTests(unittest.TestCase):
    def config(self, **kwargs):
        payload = dict(model_id="HuggingFaceTB/SmolLM2-360M", model_revision="a" * 40,
                       dataset_manifest="dataset/manifest.json", output_dir="runs/example")
        payload.update(kwargs)
        return TrainingConfig(**payload)

    def test_revision_must_be_immutable(self):
        for revision in (None, "main", "v1", "A" * 40, "a" * 39):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.config(model_revision=revision)

    def test_full_finetune_is_default(self):
        self.assertEqual(self.config().training_mode, "full")

    def test_invalid_config_values(self):
        for change in ({"training_mode": "lora_full"}, {"precision": "int4"},
                       {"learning_rate": float("nan")}, {"max_steps": 0},
                       {"max_seq_length": True}, {"gradient_checkpointing": "yes"},
                       {"seed": -1}, {"lora_dropout": 1.0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(**change)

    def test_auto_precision(self):
        self.assertEqual(select_precision("auto", cuda_available=True, bf16_supported=True), "bf16")
        self.assertEqual(select_precision("auto", cuda_available=True, bf16_supported=False), "fp16")
        self.assertEqual(select_precision("auto", cuda_available=False, bf16_supported=False), "fp32")

    def test_unsupported_explicit_precision_fails(self):
        for requested, cuda, bf16 in (("fp16", False, False), ("bf16", True, False), ("bf16", False, False)):
            with self.assertRaises(ValueError):
                select_precision(requested, cuda_available=cuda, bf16_supported=bf16)

    def test_unknown_fields_fail(self):
        with self.assertRaises(ValueError):
            TrainingConfig.from_dict(dict(self.config().as_dict(), train_on_benchmarks=True))

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(self.config().as_dict()))
            self.assertEqual(TrainingConfig.load(path), self.config())


if __name__ == "__main__":
    unittest.main()

class TrainingTPUConfigTests(unittest.TestCase):
    def test_xla_prefers_bf16(self):
        self.assertEqual(select_precision("auto", cuda_available=False, bf16_supported=False, xla_available=True), "bf16")
        self.assertEqual(select_precision("fp32", cuda_available=False, bf16_supported=False, xla_available=True), "fp32")
        with self.assertRaisesRegex(ValueError, "not fp16"):
            select_precision("fp16", cuda_available=False, bf16_supported=False, xla_available=True)


class IndependentEvalConfigTests(unittest.TestCase):
    def test_old_defaults_and_explicit_policy_are_frozen(self):
        from dataclasses import FrozenInstanceError
        config = TrainingConfig(model_id="org/model", model_revision="a" * 40,
                                dataset_manifest="dataset.json", output_dir="run")
        self.assertIsNone(config.eval_steps)
        self.assertFalse(config.checkpoint_before_eval)
        self.assertEqual(config.checkpoint_interval_seconds, 300.0)
        for changes in ({"eval_steps": 0}, {"eval_steps": True}, {"eval_steps": 1.5},
                        {"checkpoint_before_eval": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig.from_dict({**config.as_dict(), **changes})
        opted = TrainingConfig.from_dict({**config.as_dict(), "eval_steps": 50,
                                         "checkpoint_before_eval": True})
        self.assertEqual(opted.eval_steps, 50)
        with self.assertRaises(FrozenInstanceError):
            opted.eval_steps = 100
