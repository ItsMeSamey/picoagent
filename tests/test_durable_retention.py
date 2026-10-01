"""Only tiny generated test checkpoints are deleted by these tests."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from durable_retention import apply_retention, plan_retention  # noqa: E402
from picoagent.training.provenance import checkpoint_evidence  # noqa: E402
from picoagent.training.data import sha256_file  # noqa: E402


def checkpoint(root, step, loss):
    p = root / f"checkpoint-{step}"
    p.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (p / name).write_bytes(b"untrained-unit-fixture")
    (p / "trainer_state.json").write_text(json.dumps({"global_step": step,
        "log_history": [{"step": step, "eval_loss": loss}]}))
    checkpoint_evidence(p, sha256_file(root / "run_manifest.json"))
    (root / "receipts").mkdir(exist_ok=True)
    (root / "receipts" / (p.name + ".json")).write_text(json.dumps({
        "checkpoint": p.name, "checkpoint_manifest_sha256": sha256_file(p / "checkpoint_manifest.json"),
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"), "off_runtime_attested": True}))
    return p


@pytest.fixture
def archive(tmp_path):
    (tmp_path / "run_manifest.json").write_text('{"test":true}')
    (tmp_path / "trajectories.jsonl").write_text('{"keep":true}\n')
    for step, loss in [(1, .7), (2, .1), (3, .4), (4, .6), (5, .3)]:
        checkpoint(tmp_path, step, loss)
    return tmp_path


def test_plan_is_read_only_and_keeps_latest_two_and_best(archive):
    plan = plan_retention(archive)
    assert plan["keep"] == ["checkpoint-2", "checkpoint-4", "checkpoint-5"]
    assert [row["name"] for row in plan["delete"]] == ["checkpoint-1", "checkpoint-3"]
    assert len(list(archive.glob("checkpoint-*"))) == 5
    with pytest.raises(ValueError, match="explicit approval"):
        apply_retention(plan)


def test_explicit_test_cleanup_preserves_receipts_and_data(archive):
    removed = apply_retention(plan_retention(archive), confirm_delete_old_checkpoints=True)
    assert removed == ["checkpoint-1", "checkpoint-3"]
    assert (archive / "trajectories.jsonl").read_text() == '{"keep":true}\n'
    assert len(list((archive / "receipts").glob("*.json"))) == 5
    assert len((archive / "checkpoint_retention.jsonl").read_text().splitlines()) == 4


def test_new_checkpoint_invalidates_plan_before_deletion(archive):
    plan = plan_retention(archive)
    checkpoint(archive, 6, .05)
    with pytest.raises(ValueError, match="changed since"):
        apply_retention(plan, confirm_delete_old_checkpoints=True)
    assert (archive / "checkpoint-1").exists()


def test_archive_without_durable_receipt_is_rejected(archive):
    (archive / "receipts/checkpoint-1.json").unlink()
    with pytest.raises(ValueError, match="transfer receipt"):
        plan_retention(archive)


def test_corrupt_checkpoint_prevents_any_cleanup(archive):
    plan = plan_retention(archive)
    (archive / "checkpoint-3/optimizer.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        apply_retention(plan, confirm_delete_old_checkpoints=True)
    assert (archive / "checkpoint-1").exists()
