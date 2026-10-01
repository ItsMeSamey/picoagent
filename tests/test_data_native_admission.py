"""Admission unit fixtures are not actual native execution observations."""
from __future__ import annotations

import json

import pytest

from picoagent.data import authored_example, generate_task
from picoagent.data.native_admission import NATIVE_MANIFEST_SCHEMA, _validate_review, verify_native_snapshot
from picoagent.data.schema import DataValidationError, validate_trace


def test_native_category_requires_opt_in_before_any_other_claim():
    trace = authored_example(generate_task("math.cart_total", 0))
    trace["provenance"]["execution"] = "native_teacher_observed"
    with pytest.raises(DataValidationError, match="explicit admission opt-in"):
        validate_trace(trace)


def test_approved_flag_alone_cannot_admit_native_source():
    review = {"schema": "picoagent.native_teacher.source_review.v1", "source_id": "luna_cli", "reviewer": "unit test",
              "review_notes": "Fabricated fixture, intentionally not actual evidence", "oracle_id": "fixture", "command_policy": "fixture",
              "approved": True, "source_sha256": {"fixture.py": "0" * 64}, "oracle_source_sha256": "0" * 64,
              "teacher_mode": "luna_authored_program_deterministic_replay", "arbitrary_learner_execution_allowed": False}
    with pytest.raises(DataValidationError, match="independently pinned"):
        _validate_review(review)


def test_generic_jsonl_or_manifest_cannot_unlock_native_training(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": NATIVE_MANIFEST_SCHEMA}))
    with pytest.raises(DataValidationError, match="explicit opt-in"):
        verify_native_snapshot(path)
    with pytest.raises(DataValidationError, match="admission policy"):
        verify_native_snapshot(path, allow_native_teacher=True)


def test_actual_cli_sample_projection_and_tamper_binding():
    """Read an actual archived sample; never run its recorded commands."""
    import copy
    from pathlib import Path
    from picoagent.data.native_admission import extract_native_observation
    from picoagent.data.schema import _validate_model_event_replay
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/luna-cli-v1/native_teacher_observed/sample_001/observations.jsonl"
    if not sample.exists():
        pytest.skip("actual CLI observation artifact not distributed with this test checkout")
    raw = json.loads(sample.read_text().splitlines()[0])
    tasks = [json.loads(line) for line in (root / "data/luna-cli-v1/train.tasks.jsonl").read_text().splitlines()]
    task = next(row for row in tasks if row["task_id"] == raw["task_id"])
    projection = extract_native_observation("luna_cli", raw, task)
    _validate_model_event_replay(projection)
    assert projection["final"] == '{"revenue":771}'
    forged = copy.deepcopy(raw)
    forged["tool_events"][0]["execution"]["stdout_base64"] = "ZmFrZQ=="
    with pytest.raises(DataValidationError, match="captured raw stream"):
        extract_native_observation("luna_cli", forged, task)
    forged = copy.deepcopy(raw)
    forged["tool_events"][0]["arguments"]["code"] = "print('unrelated')"
    forged["tool_events"][0]["arguments_json"] = json.dumps(forged["tool_events"][0]["arguments"])
    with pytest.raises(DataValidationError, match="reviewed frozen action plan"):
        extract_native_observation("luna_cli", forged, task)


def test_actual_sealed_native_pilot_and_full_evidence_copy(tmp_path):
    from pathlib import Path
    from picoagent.data.native_admission import copy_native_snapshot
    from picoagent.training.data import read_records, verify_dataset
    root = Path(__file__).resolve().parents[1]
    original = root / "data/native-sharded-pilot-v1/manifest.json"
    if not original.exists():
        pytest.skip("reviewed native pilot snapshot not distributed with this checkout")
    with pytest.raises(ValueError, match="explicit"):
        verify_dataset(original)
    manifest, records = verify_dataset(original, allow_native_teacher=True)
    assert {split: len(rows) for split, rows in records.items()} == {"train": 1, "dev": 1}
    assert manifest["source_counts"]["admitted"] == {"luna_cli": {"train": 1, "dev": 1}}
    unsealed = tmp_path / "unsealed-native.jsonl"
    unsealed.write_text(json.dumps(records["train"][0]) + "\n")
    with pytest.raises(ValueError, match="verified"):
        read_records(unsealed, "train")
    copied = copy_native_snapshot(original, tmp_path / "copied")
    assert copied.read_bytes() == original.read_bytes()
    copied_manifest, copied_records = verify_dataset(copied, allow_native_teacher=True)
    assert copied_manifest == manifest and copied_records == records
    source_file = manifest["all_observations"]["paths"][0]
    target = copied.parent / source_file
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_dataset(copied, allow_native_teacher=True)
