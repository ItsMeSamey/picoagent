"""Late evaluation evidence stays separate from immutable checkpoint bundles."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_sync import (  # noqa: E402
    BUNDLE_MANIFEST, CLITransfer, LocalTransfer, evaluation_record, pack_checkpoint,
    pull_checkpoint, sha256, upload_bundle, validate_manifest,
)
from colab_run import collect  # noqa: E402
from picoagent.training.evaluation import (  # noqa: E402
    EVALUATION_SCHEMA, checkpoint_eval_loss, read_evaluation, validate_evaluation,
)
from picoagent.training.provenance import checkpoint_evidence, verify_checkpoint  # noqa: E402


def make_checkpoint(root, step, loss=1.0):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (path / name).write_bytes(f"{name}:{step}".encode())
    (path / "trainer_state.json").write_text(json.dumps({
        "global_step": step, "log_history": [{"step": step, "eval_loss": loss}],
    }))
    checkpoint_evidence(path, sha256(root / "run_manifest.json"))
    return path


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text('{"test":"late-evaluation"}')
    make_checkpoint(root, 1)
    return root


def payload(run, step=1, loss=0.25):
    name = f"checkpoint-{step}"
    return {"schema": EVALUATION_SCHEMA, "checkpoint": name,
            "checkpoint_manifest_sha256": sha256(run / name / "checkpoint_manifest.json"),
            "global_step": step, "metrics": {"eval_loss": loss, "eval_runtime": 2.1}}


def save_evaluation(run, step=1, loss=0.25):
    path = run / "evaluations" / f"checkpoint-{step}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload(run, step, loss)) + "\n")
    return path


def pack(run, tmp_path, export="exports"):
    return pack_checkpoint(run, "checkpoint-1", tmp_path / export, chunk_bytes=17)


@pytest.mark.parametrize(("field", "value"), [
    ("schema", "other"), ("checkpoint", "checkpoint-2"), ("checkpoint", "../checkpoint-1"),
    ("global_step", 2), ("global_step", True), ("global_step", 1.0),
    ("checkpoint_manifest_sha256", "0" * 64), ("checkpoint_manifest_sha256", None),
    ("metrics", {}), ("metrics", {"eval_loss": True}), ("metrics", {"eval_loss": "0.1"}),
    ("metrics", {"eval_loss": float("nan")}), ("metrics", {"eval_loss": float("inf")}),
    ("metrics", {"eval_loss": float("-inf")}),
    ("metrics", {"eval_loss": 0.1, "eval_runtime": float("nan")}),
])
def test_sidecar_binding_rejects_invalid_evidence(run, field, value):
    evidence = payload(run)
    evidence[field] = value
    with pytest.raises(ValueError):
        validate_evaluation(evidence, run / "checkpoint-1")


def test_bounded_json_and_duplicate_keys_rejected(run):
    path = save_evaluation(run)
    path.write_text('{"schema":"wrong",' + path.read_text()[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        read_evaluation(path, run / "checkpoint-1")
    path.write_text(" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        read_evaluation(path, run / "checkpoint-1")


def test_invalid_sidecar_never_falls_back_to_valid_legacy(run, tmp_path):
    sidecar = save_evaluation(run)
    sidecar.write_text("not json")
    with pytest.raises(ValueError):
        checkpoint_eval_loss(run, "checkpoint-1")
    with pytest.raises(ValueError):
        pack(run, tmp_path)


def test_sidecar_precedes_legacy_and_checkpoint_stays_sealed(run):
    before = sha256(run / "checkpoint-1/checkpoint_manifest.json")
    assert checkpoint_eval_loss(run, "checkpoint-1") == 1.0
    save_evaluation(run, loss=0.25)
    assert checkpoint_eval_loss(run, "checkpoint-1") == 0.25
    assert sha256(run / "checkpoint-1/checkpoint_manifest.json") == before
    verify_checkpoint(run / "checkpoint-1", sha256(run / "run_manifest.json"))


def test_legacy_loss_requires_exact_step_and_finite_scalar(run):
    path = run / "checkpoint-1/trainer_state.json"
    path.write_text(json.dumps({"global_step": 1, "log_history": [{"step": 0, "eval_loss": 0.01}]}))
    assert checkpoint_eval_loss(run, "checkpoint-1") is None
    path.write_text(json.dumps({"global_step": 1, "log_history": [{"step": 1, "eval_loss": float("nan")}]}))
    with pytest.raises(ValueError, match="finite"):
        checkpoint_eval_loss(run, "checkpoint-1")


def test_late_pack_and_retry_pull_keep_original_bundle_and_receipt(run, tmp_path):
    bundle = pack(run, tmp_path)
    original = (bundle / BUNDLE_MANIFEST).read_bytes()
    destination = tmp_path / "durable"
    pull_checkpoint(str(bundle), destination, LocalTransfer(), off_runtime=True)
    receipt = (destination / "receipts/checkpoint-1.json").read_bytes()
    sidecar = save_evaluation(run)
    assert pack(run, tmp_path) == bundle
    assert (bundle / BUNDLE_MANIFEST).read_bytes() == original
    record = evaluation_record(bundle)
    assert record["sha256"] == sha256(sidecar)

    class NoCheckpointChunks(LocalTransfer):
        def download(self, remote, local):
            assert "/chunks/" not in remote
            super().download(remote, local)
    pull_checkpoint(str(bundle), destination, NoCheckpointChunks(), off_runtime=True)
    saved = destination / "evaluations/checkpoint-1.json"
    assert saved.read_bytes() == sidecar.read_bytes()
    assert (destination / "receipts/checkpoint-1.json").read_bytes() == receipt
    pull_checkpoint(str(bundle), destination, NoCheckpointChunks(), off_runtime=True)
    assert (bundle / BUNDLE_MANIFEST).read_bytes() == original
    assert not any(path.startswith("evaluations/") for path in json.loads(original)["files"])


def test_manifest_allowlist_does_not_admit_evaluations(run, tmp_path):
    manifest = json.loads((pack(run, tmp_path) / BUNDLE_MANIFEST).read_text())
    manifest["files"]["evaluations/checkpoint-1.json"] = manifest["files"]["run_manifest.json"]
    with pytest.raises(ValueError, match="Unexpected"):
        validate_manifest(manifest)


def test_changed_evaluation_refuses_pack_and_preserves_first_artifact(run, tmp_path):
    sidecar = save_evaluation(run)
    original = sidecar.read_bytes()
    bundle = pack(run, tmp_path)
    record = evaluation_record(bundle)
    manifest = (bundle / BUNDLE_MANIFEST).read_bytes()
    save_evaluation(run, loss=0.9)
    with pytest.raises(ValueError, match="collision"):
        pack(run, tmp_path)
    assert (bundle / record["path"]).read_bytes() == original
    assert (bundle / BUNDLE_MANIFEST).read_bytes() == manifest


def test_upload_then_late_upload_and_local_pull_preserve_manifest(run, tmp_path):
    bundle = pack(run, tmp_path)
    remote = tmp_path / "remote"
    upload_bundle(bundle, str(remote), LocalTransfer())
    original = (remote / BUNDLE_MANIFEST).read_bytes()
    sidecar = save_evaluation(run)
    pack(run, tmp_path)
    upload_bundle(bundle, str(remote), LocalTransfer())
    assert (remote / BUNDLE_MANIFEST).read_bytes() == original
    destination = tmp_path / "durable"
    pull_checkpoint(str(remote), destination, LocalTransfer(), off_runtime=True)
    assert (destination / "evaluations/checkpoint-1.json").read_bytes() == sidecar.read_bytes()


def test_generic_cli_roundtrip_uses_advertised_evaluation_digest(run, tmp_path):
    sidecar = save_evaluation(run)
    bundle = pack(run, tmp_path)
    fake = tmp_path / "transfer.py"
    fake.write_text("import pathlib,shutil,sys\np=pathlib.Path(sys.argv[2]);p.parent.mkdir(parents=True,exist_ok=True)\nshutil.copyfile(sys.argv[1],p)\n")
    transfer = CLITransfer([sys.executable, str(fake), "{remote}", "{local}"],
                           [sys.executable, str(fake), "{local}", "{remote}"])
    remote = tmp_path / "remote"
    upload_bundle(bundle, str(remote), transfer)
    destination = tmp_path / "durable"
    pull_checkpoint(str(remote), destination, transfer, off_runtime=True,
                    expected_evaluation_sha256=evaluation_record(bundle)["sha256"])
    assert (destination / "evaluations/checkpoint-1.json").read_bytes() == sidecar.read_bytes()


def test_download_corrupt_evaluation_never_publishes_it(run, tmp_path):
    save_evaluation(run)
    bundle = pack(run, tmp_path)
    record = evaluation_record(bundle)
    class Corrupt(LocalTransfer):
        def download(self, remote, local):
            super().download(remote, local)
            if "/evaluations/" in remote:
                local.write_bytes(b"corrupt")
    destination = tmp_path / "durable"
    with pytest.raises(ValueError, match="evaluation hash mismatch"):
        pull_checkpoint(str(bundle), destination, Corrupt(), off_runtime=True,
                        expected_evaluation_sha256=record["sha256"])
    assert not (destination / "evaluations/checkpoint-1.json").exists()
    assert not (destination / "receipts/checkpoint-1.json").exists()
    pull_checkpoint(str(bundle), destination, LocalTransfer(), off_runtime=True)
    assert (destination / "evaluations/checkpoint-1.json").exists()


def test_missing_advertised_evaluation_fails_closed(run, tmp_path):
    bundle = pack(run, tmp_path)
    destination = tmp_path / "durable"
    with pytest.raises(FileNotFoundError):
        pull_checkpoint(str(bundle), destination, LocalTransfer(), off_runtime=True,
                        expected_evaluation_sha256="0" * 64)
    assert not (destination / "receipts/checkpoint-1.json").exists()


def test_destination_evaluation_collision_does_not_overwrite(run, tmp_path):
    sidecar = save_evaluation(run)
    bundle = pack(run, tmp_path)
    destination = tmp_path / "durable"
    pull_checkpoint(str(bundle), destination, LocalTransfer(), off_runtime=True)
    original = sidecar.read_bytes()
    save_evaluation(run, loss=0.9)
    different = pack(run, tmp_path, "new-exports")
    with pytest.raises(ValueError, match="collision"):
        pull_checkpoint(str(different), destination, LocalTransfer(), off_runtime=True)
    assert (destination / "evaluations/checkpoint-1.json").read_bytes() == original


def test_remote_evaluation_collision_does_not_overwrite(run, tmp_path):
    save_evaluation(run)
    bundle = pack(run, tmp_path)
    remote = tmp_path / "remote"
    upload_bundle(bundle, str(remote), LocalTransfer())
    before = evaluation_record(remote)
    save_evaluation(run, loss=0.9)
    different = pack(run, tmp_path, "new-exports")
    with pytest.raises(ValueError, match="collision"):
        upload_bundle(different, str(remote), LocalTransfer())
    assert evaluation_record(remote) == before


def test_interrupted_evaluation_upload_is_retryable(run, tmp_path):
    save_evaluation(run)
    bundle = pack(run, tmp_path)
    remote = tmp_path / "remote"
    class Interrupted(LocalTransfer):
        def upload(self, local, target):
            if "/evaluations/" in target:
                path = Path(target)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"partial")
                raise ConnectionError("disconnected")
            super().upload(local, target)
    with pytest.raises(ConnectionError):
        upload_bundle(bundle, str(remote), Interrupted())
    assert not (remote / BUNDLE_MANIFEST).exists()
    upload_bundle(bundle, str(remote), LocalTransfer())
    assert evaluation_record(remote) == evaluation_record(bundle)


def test_uploaded_evaluation_requires_readback(run, tmp_path):
    save_evaluation(run)
    bundle = pack(run, tmp_path)
    remote = tmp_path / "remote"
    class Corrupt(LocalTransfer):
        def upload(self, local, target):
            super().upload(local, target)
            if "/evaluations/" in target:
                Path(target).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="evaluation failed read-back"):
        upload_bundle(bundle, str(remote), Corrupt())
    assert not (remote / BUNDLE_MANIFEST).exists()


def test_sidecar_symlinks_rejected(run, tmp_path):
    actual = tmp_path / "evaluation.json"
    actual.write_text(json.dumps(payload(run)))
    (run / "evaluations").mkdir()
    (run / "evaluations/checkpoint-1.json").symlink_to(actual)
    with pytest.raises(ValueError, match="symlink"):
        pack(run, tmp_path)


class FakeColab:
    def execute(self, code):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, "fake-colab.py", "exec"), {})
        return json.loads(stream.getvalue().split("PICOAGENT_RESULT=")[-1])

    def transfer(self):
        class AbsoluteLocal(LocalTransfer):
            def download(self, remote, local):
                super().download("/" + remote.lstrip("/"), local)
        return AbsoluteLocal()


def test_collect_late_sidecar_changes_best_and_preserves_it_during_prune(run, tmp_path):
    make_checkpoint(run, 2, 0.2)
    make_checkpoint(run, 3, 0.3)
    make_checkpoint(run, 4, 0.4)
    project = str(Path(__file__).resolve().parents[1])
    args = FakeColab(), project, str(run), str(tmp_path / "exports"), tmp_path / "durable", True
    first = collect(*args)
    assert first["best_checkpoint"] == "checkpoint-2"
    manifest = (tmp_path / "exports/checkpoint-1" / BUNDLE_MANIFEST).read_bytes()
    sidecar = save_evaluation(run, loss=0.01)
    second = collect(*args, prune=True)
    assert second["best_checkpoint"] == "checkpoint-1"
    assert second["pruned_runtime"] == ["checkpoint-2"]
    assert (run / "checkpoint-1").is_dir()
    assert (tmp_path / "durable/evaluations/checkpoint-1.json").read_bytes() == sidecar.read_bytes()
    assert (tmp_path / "exports/checkpoint-1" / BUNDLE_MANIFEST).read_bytes() == manifest


def test_collect_invalid_sidecar_stops_before_pruning(run, tmp_path):
    for step in (2, 3, 4):
        make_checkpoint(run, step)
    sidecar = save_evaluation(run)
    invalid = payload(run)
    invalid["checkpoint_manifest_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(invalid))
    project = str(Path(__file__).resolve().parents[1])
    with pytest.raises(ValueError, match="hash mismatch"):
        collect(FakeColab(), project, str(run), str(tmp_path / "exports"), tmp_path / "durable", True, True)
    assert len(list(run.glob("checkpoint-*"))) == 4


def test_remote_evaluation_symlink_is_not_overwritten(run, tmp_path):
    save_evaluation(run)
    bundle = pack(run, tmp_path)
    record = evaluation_record(bundle)
    remote = tmp_path / "remote"
    (remote / "evaluations").mkdir(parents=True)
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("preserve this")
    (remote / record["path"]).symlink_to(unrelated)
    with pytest.raises(ValueError, match="collision"):
        upload_bundle(bundle, str(remote), LocalTransfer())
    assert unrelated.read_text() == "preserve this"


def test_destination_sidecar_symlink_is_not_followed(run, tmp_path):
    save_evaluation(run)
    bundle = pack(run, tmp_path)
    destination = tmp_path / "durable"
    (destination / "evaluations").mkdir(parents=True)
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("preserve this")
    (destination / "evaluations/checkpoint-1.json").symlink_to(unrelated)
    with pytest.raises(ValueError, match="symlink"):
        pull_checkpoint(str(bundle), destination, LocalTransfer(), off_runtime=True)
    assert unrelated.read_text() == "preserve this"


def test_restore_includes_sidecar_without_durability_claim(run, tmp_path):
    sidecar = save_evaluation(run)
    bundle = pack(run, tmp_path)
    destination = tmp_path / "restore"
    pull_checkpoint(str(bundle), destination, LocalTransfer(), off_runtime=False, restore=True)
    assert (destination / "evaluations/checkpoint-1.json").read_bytes() == sidecar.read_bytes()
    assert not (destination / "receipts").exists()


def test_legacy_multiple_exact_step_losses_keep_original_minimum_policy(run):
    (run / "checkpoint-1/trainer_state.json").write_text(json.dumps({"global_step": 1, "log_history": [
        {"step": 0, "eval_loss": 0.001}, {"step": 1, "eval_loss": 0.1},
        {"step": 1, "eval_loss": 0.4},
    ]}))
    assert checkpoint_eval_loss(run, "checkpoint-1") == 0.1


def test_evaluation_arriving_during_collect_is_kept_for_next_pass(run, tmp_path):
    for step in (2, 3, 4):
        make_checkpoint(run, step, 0.1 if step == 2 else 0.3)
    class DelayedEvaluation(FakeColab):
        calls = 0
        def execute(self, code):
            self.calls += 1
            if self.calls == 2:
                save_evaluation(run, loss=0.01)
            return super().execute(code)
    project = str(Path(__file__).resolve().parents[1])
    args = project, str(run), str(tmp_path / "exports"), tmp_path / "durable", True
    first = collect(DelayedEvaluation(), *args, prune=True)
    assert first["pruned_runtime"] == []
    assert first["pending_evaluations"] == ["checkpoint-1"]
    assert (run / "checkpoint-1").is_dir()
    second = collect(FakeColab(), *args, prune=True)
    assert second["best_checkpoint"] == "checkpoint-1"
    assert second["pruned_runtime"] == ["checkpoint-2"]
    assert (tmp_path / "durable/evaluations/checkpoint-1.json").is_file()
