import copy
import json
import tempfile
import unittest
from pathlib import Path

from picoagent.training.data import conversation_fingerprint, prepare_dataset, validate_record, verify_dataset


def record(split="train", number=1):
    # Fabricated receipts are unit-test input, never released as training data.
    return {"schema_version": "picoagent.data.v1", "trace_id": f"trace-{split}-{number}",
            "task_id": f"task-{split}-{number}", "family": f"family-{split}",
            "template_id": f"template-{split}", "split": split, "status": "success",
            "provenance": {"source": "original_procedural", "execution": "verified_environment",
                           "benchmark": False, "runtime": {"backend": "docker", "container_id": "c" * 64, "image": "mock-unit-test"}},
            "raw_attempt_sha256": "a" * 64, "task_sha256": "b" * 64,
            "verification": {"passed": True}, "tool_events": [],
            "messages": [{"role": "user", "content": f"Compute {split} {number}"},
                         {"role": "assistant", "content": f"Result {number}"}]}


class TrainingDatasetTests(unittest.TestCase):
    def write(self, path, value):
        path.write_text(json.dumps(value) + "\n")

    def test_exact_byte_snapshot_and_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root / "train.jsonl", record())
            self.write(root / "dev.jsonl", record("dev"))
            manifest = prepare_dataset(root / "train.jsonl", root / "dev.jsonl", root / "snapshot")
            self.assertEqual((root / "snapshot/train.jsonl").read_bytes(), (root / "train.jsonl").read_bytes())
            metadata, rows = verify_dataset(manifest)
            self.assertFalse(metadata["lockbox_used"])
            self.assertEqual(len(rows["train"]), 1)
            (root / "snapshot/train.jsonl").chmod(0o644)
            (root / "snapshot/train.jsonl").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_dataset(manifest)

    def test_authored_and_benchmark_traces_rejected(self):
        for patch in ({"execution": "authored_example"}, {"source": "official_benchmark"}, {"benchmark": True}):
            row = record()
            row["provenance"].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_record(row, "train")

    def test_missing_receipt_rejected(self):
        row = record()
        del row["provenance"]["runtime"]
        with self.assertRaises(ValueError):
            validate_record(row, "train")

    def test_test_split_rejected(self):
        with self.assertRaises(ValueError):
            validate_record(record("test"), "test")

    def test_family_or_template_overlap_rejected(self):
        for field in ("family", "template_id", "task_id", "trace_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                train, dev = record(), record("dev")
                dev[field] = train[field]
                self.write(root / "train.jsonl", train)
                self.write(root / "dev.jsonl", dev)
                with self.assertRaisesRegex(ValueError, "overlap"):
                    prepare_dataset(root / "train.jsonl", root / "dev.jsonl", root / "snapshot")

    def test_smoke_cannot_enter_production(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "dev"):
                row = record(split)
                row["provenance"] = {"source": "pipeline_smoke"}
                row["status"] = "unexecuted"
                row["verification"]["passed"] = False
                self.write(root / f"{split}.jsonl", row)
            manifest = prepare_dataset(root / "train.jsonl", root / "dev.jsonl", root / "snapshot", smoke_only=True)
            with self.assertRaisesRegex(ValueError, "Smoke"):
                verify_dataset(manifest)
            verify_dataset(manifest, allow_smoke=True)

    def test_tool_ids_normalized_in_fingerprint(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "python", "arguments": '{"code":"1"}'}}]},
                    {"role": "tool", "content": "1", "tool_call_id": "a"}, {"role": "assistant", "content": "1"}]
        other = copy.deepcopy(messages)
        other[0]["tool_calls"][0]["id"] = "b"
        other[1]["tool_call_id"] = "b"
        self.assertEqual(conversation_fingerprint(messages), conversation_fingerprint(other))

    def test_orphan_tool_response_rejected(self):
        row = record()
        row["messages"].insert(1, {"role": "tool", "content": "bad", "tool_call_id": "unknown"})
        with self.assertRaises(ValueError):
            validate_record(row, "train")

    def test_snapshot_will_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root / "train.jsonl", record())
            self.write(root / "dev.jsonl", record("dev"))
            (root / "snapshot").mkdir()
            with self.assertRaises(FileExistsError):
                prepare_dataset(root / "train.jsonl", root / "dev.jsonl", root / "snapshot")


if __name__ == "__main__":
    unittest.main()
