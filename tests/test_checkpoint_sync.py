"""Small fake checkpoints test transport failures without an accelerator/account."""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_sync import (  # noqa: E402
    BUNDLE_MANIFEST, CLITransfer, LocalTransfer, pack_checkpoint, pull_checkpoint,
    sha256, upload_bundle,
)
from colab_run import Colab, archive_source, collect  # noqa: E402
from picoagent.training.provenance import checkpoint_evidence, verify_checkpoint  # noqa: E402


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_manifest.json").write_text('{"identity":"test"}')
    (root / "dataset_manifest.json").write_text('{"split":"original"}')
    (root / "traces").mkdir()
    (root / "traces" / "training.jsonl").write_text('{"immutable":"trace"}\n')
    checkpoint(root, 1)
    return root


def checkpoint(root: Path, step: int, loss: float = 1.0) -> Path:
    target = root / f"checkpoint-{step}"
    target.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (target / name).write_bytes((f"{name}-{step}-" * 5).encode())
    (target / "trainer_state.json").write_text(json.dumps({"global_step": step, "log_history": [{"step": step, "eval_loss": loss}]}))
    checkpoint_evidence(target, sha256(root / "run_manifest.json"))
    return target


def bundle(run: Path, tmp_path: Path, step: int = 1) -> Path:
    return pack_checkpoint(run, f"checkpoint-{step}", tmp_path / "export", chunk_bytes=17)


def test_chunked_roundtrip_and_resume_state(run, tmp_path):
    packed = bundle(run, tmp_path)
    destination = tmp_path / "durable"
    result = pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True,
                             expected_manifest_sha256=sha256(packed / BUNDLE_MANIFEST))
    verify_checkpoint(result, sha256(destination / "run_manifest.json"))
    assert (result / "optimizer.pt").read_bytes() == (run / "checkpoint-1" / "optimizer.pt").read_bytes()
    assert json.loads((destination / "receipts/checkpoint-1.json").read_text())["off_runtime_attested"] is True
    assert (destination / "dataset_manifest.json").exists()
    assert (run / "traces/training.jsonl").exists()


def test_attestation_required(run, tmp_path):
    with pytest.raises(ValueError, match="off_runtime"):
        pull_checkpoint(str(bundle(run, tmp_path)), tmp_path / "durable", LocalTransfer(), off_runtime=False)


def test_interrupted_download_resumes_verified_chunks(run, tmp_path):
    class Interrupted(LocalTransfer):
        def __init__(self):
            self.calls = []
            self.fail = True

        def download(self, remote, local):
            self.calls.append(remote)
            if self.fail and len(self.calls) == 5:
                local.write_bytes(b"partial")
                raise ConnectionError("runtime disconnected")
            super().download(remote, local)

    transfer = Interrupted()
    packed, destination = bundle(run, tmp_path), tmp_path / "durable"
    with pytest.raises(ConnectionError):
        pull_checkpoint(str(packed), destination, transfer, off_runtime=True)
    assert not (destination / "checkpoint-1").exists()
    assert not (destination / "receipts/checkpoint-1.json").exists()
    completed_chunks = set(transfer.calls[1:4])
    transfer.fail = False
    transfer.calls.clear()
    result = pull_checkpoint(str(packed), destination, transfer, off_runtime=True)
    assert result.exists()
    assert not completed_chunks.intersection(transfer.calls)


def test_hash_mismatch_never_commits(run, tmp_path):
    packed = bundle(run, tmp_path)
    chunk = next((packed / "chunks").iterdir())
    chunk.write_bytes(b"corrupt")
    destination = tmp_path / "durable"
    with pytest.raises(ValueError, match="hash mismatch"):
        pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True)
    assert not (destination / "checkpoint-1").exists()
    assert not (destination / "receipts/checkpoint-1.json").exists()


def test_manifest_hash_mismatch_fails_before_data(run, tmp_path):
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        pull_checkpoint(str(bundle(run, tmp_path)), tmp_path / "durable", LocalTransfer(),
                        off_runtime=True, expected_manifest_sha256="0" * 64)


def test_manifest_traversal_rejected(run, tmp_path):
    packed = bundle(run, tmp_path)
    manifest = json.loads((packed / BUNDLE_MANIFEST).read_text())
    manifest["files"]["../escape"] = manifest["files"]["run_manifest.json"]
    (packed / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        pull_checkpoint(str(packed), tmp_path / "durable", LocalTransfer(), off_runtime=True)
    assert not (tmp_path / "escape").exists()


def test_corrupt_existing_checkpoint_not_overwritten(run, tmp_path):
    packed, destination = bundle(run, tmp_path), tmp_path / "durable"
    pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True)
    (destination / "checkpoint-1/optimizer.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True)
    assert (destination / "checkpoint-1/optimizer.pt").read_bytes() == b"corrupted"


def test_run_collision_preserves_previous_checkpoint(run, tmp_path):
    packed, destination = bundle(run, tmp_path), tmp_path / "durable"
    pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True)
    (run / "run_manifest.json").write_text('{"different":"run"}')
    checkpoint(run, 2)
    packed2 = bundle(run, tmp_path, 2)
    with pytest.raises(ValueError, match="different run metadata"):
        pull_checkpoint(str(packed2), destination, LocalTransfer(), off_runtime=True)
    assert (destination / "checkpoint-1").exists()
    assert not (destination / "checkpoint-2").exists()


def test_interrupted_upload_does_not_publish_manifest(run, tmp_path):
    class Interrupted(LocalTransfer):
        def upload(self, local, remote):
            if len(list((tmp_path / "remote").rglob("*"))) > 3:
                raise ConnectionError("upload disconnected")
            super().upload(local, remote)

    remote = tmp_path / "remote"
    remote.mkdir()
    packed = bundle(run, tmp_path)
    with pytest.raises(ConnectionError):
        upload_bundle(packed, str(remote), Interrupted())
    assert not (remote / BUNDLE_MANIFEST).exists()
    upload_bundle(packed, str(remote), LocalTransfer())
    assert sha256(remote / BUNDLE_MANIFEST) == sha256(packed / BUNDLE_MANIFEST)


def test_upload_readback_corruption_refuses_commit(run, tmp_path):
    class Corrupt(LocalTransfer):
        def upload(self, local, remote):
            super().upload(local, remote)
            Path(remote).write_bytes(b"corrupt")

    remote = tmp_path / "remote"
    with pytest.raises(ValueError, match="read-back"):
        upload_bundle(bundle(run, tmp_path), str(remote), Corrupt())
    assert not (remote / BUNDLE_MANIFEST).exists()


def test_cli_transfer_fake_process_handles_spaces_without_shell(run, tmp_path):
    fake = tmp_path / "fake transfer.py"
    fake.write_text("import pathlib, shutil, sys\np=pathlib.Path(sys.argv[2]); p.parent.mkdir(parents=True, exist_ok=True)\nshutil.copyfile(sys.argv[1], p)\n")
    transport = CLITransfer([sys.executable, str(fake), "{remote}", "{local}"],
                            [sys.executable, str(fake), "{local}", "{remote}"])
    packed = bundle(run, tmp_path)
    remote = tmp_path / "remote space;literal"
    upload_bundle(packed, str(remote), transport)
    result = pull_checkpoint(str(remote), tmp_path / "durable", transport, off_runtime=True)
    assert result.exists()


def test_checkpoint_symlinks_rejected(run, tmp_path):
    (run / "checkpoint-1/link").symlink_to(run / "run_manifest.json")
    with pytest.raises(ValueError, match="Symlinks"):
        bundle(run, tmp_path)


def test_source_archive_excludes_credentials_venv_git_and_preserves_traces(tmp_path):
    root = tmp_path / "project"
    for name in ("src/picoagent/main.py", "data/traces.jsonl", ".git/config", ".venv/package.py",
                 ".env", "configs/oauth.json", "scripts/token.json", "README.md"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    output = tmp_path / "source.tar.gz"
    result = archive_source(root, output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
    assert set(names) == {"src/picoagent/main.py", "data/traces.jsonl", "README.md", "SOURCE_MANIFEST.json"}
    assert result["sha256"] == sha256(output)


def test_colab_commands_always_explicit_session(monkeypatch):
    calls = []
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr("colab_run.run_cli", fake_run)
    client = Colab("picoagent-tpu")
    client.command("status")
    assert calls == [["colab", "status", "--session", "picoagent-tpu"]]
    for prohibited in ("new", "run", "stop", "pay", "ssh"):
        with pytest.raises(ValueError):
            client.command(prohibited)
    with pytest.raises(ValueError):
        Colab("")


def test_collect_prunes_only_verified_runtime_checkpoints_preserves_best_traces(run, tmp_path):
    checkpoint(run, 2, 0.1)
    checkpoint(run, 3, 0.7)
    checkpoint(run, 4, 0.8)
    # Executes only the generated local-equivalent Python snippets; no Colab calls.
    class FakeColab:
        def execute(self, code):
            import contextlib
            import io
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(code, "fake_colab.py", "exec"), {})
            return json.loads(stream.getvalue().split("PICOAGENT_RESULT=")[-1])

        def transfer(self):
            class AbsoluteLocal(LocalTransfer):
                def download(self, remote, local):
                    super().download("/" + remote.lstrip("/"), local)
            return AbsoluteLocal()

    project = str(Path(__file__).resolve().parents[1])
    result = collect(FakeColab(), project, str(run), str(tmp_path / "exports"),
                     tmp_path / "durable", True, prune=True)
    assert result["pruned_runtime"] == ["checkpoint-1"]
    assert {p.name for p in run.glob("checkpoint-*")} == {"checkpoint-2", "checkpoint-3", "checkpoint-4"}
    assert (run / "traces/training.jsonl").exists()
    assert (tmp_path / "durable/checkpoint-1").exists()


def test_retry_completed_pull_does_not_download_chunks(run, tmp_path):
    class ManifestOnly(LocalTransfer):
        def download(self, remote, local):
            assert remote.endswith(BUNDLE_MANIFEST)
            super().download(remote, local)
    packed, destination = bundle(run, tmp_path), tmp_path / "durable"
    first = pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True)
    assert pull_checkpoint(str(packed), destination, ManifestOnly(), off_runtime=True) == first


def test_restore_to_runtime_verifies_but_does_not_claim_durability(run, tmp_path):
    packed = bundle(run, tmp_path)
    destination = tmp_path / "new-runtime"
    restored = pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=False, restore=True)
    verify_checkpoint(restored, sha256(destination / "run_manifest.json"))
    assert not (destination / "receipts").exists()
    assert (restored / "rng_state.pth").exists()


def test_cli_timeout_reaps_its_process(tmp_path):
    import os
    from checkpoint_sync import run_cli
    pidfile = tmp_path / "pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_cli([sys.executable, "-c", "import os,pathlib,time,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)", str(pidfile)], timeout=0.3, capture=True)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_launch_reservation_blocks_duplicate_job(tmp_path, monkeypatch):
    from colab_run import start
    class FakeChild:
        pid = 12345
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    class FakeColab:
        def execute(self, code):
            import contextlib
            import io
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exec(compile(code, "fake_colab.py", "exec"), {})
            return json.loads(stream.getvalue().split("PICOAGENT_RESULT=")[-1])
    project = tmp_path / "project"
    project.mkdir()
    result = start(FakeColab(), str(project), ["python", "train.py"])
    assert result["supervisor_pid"] == 12345
    assert json.loads((project / ".picoagent-job.json").read_text())["status"] == "starting"
    with pytest.raises(RuntimeError, match="already running"):
        start(FakeColab(), str(project), ["python", "train.py"])


def test_low_disk_aborts_pull_before_payload_download(run, tmp_path, monkeypatch):
    import checkpoint_sync
    from types import SimpleNamespace
    packed = bundle(run, tmp_path)
    class OnlyManifest(LocalTransfer):
        def download(self, remote, local):
            assert remote.endswith(BUNDLE_MANIFEST), "must not download payload on insufficient disk"
            super().download(remote, local)
    monkeypatch.setattr(checkpoint_sync.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    destination = tmp_path / "durable"
    with pytest.raises(OSError, match="Insufficient free space"):
        pull_checkpoint(str(packed), destination, OnlyManifest(), off_runtime=True)
    assert not (destination / "checkpoint-1").exists()
    assert not (destination / "receipts/checkpoint-1.json").exists()
    assert (run / "checkpoint-1/optimizer.pt").exists()


def test_low_disk_aborts_pack_without_mutating_checkpoint(run, tmp_path, monkeypatch):
    import checkpoint_sync
    from types import SimpleNamespace
    monkeypatch.setattr(checkpoint_sync.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="Insufficient free space"):
        bundle(run, tmp_path)
    assert (run / "checkpoint-1/optimizer.pt").exists()
    assert not (tmp_path / "export/checkpoint-1" / BUNDLE_MANIFEST).exists()


def test_source_archive_keeps_public_tokenizer_metadata_but_not_credentials(tmp_path):
    root = tmp_path / "project"
    for name in ("data/pilot/tokenizer/tokenizer.json", "data/pilot/tokenizer/special_tokens_map.json",
                 "data/pilot/token_validation.json", "data/pilot/access_token.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    result = archive_source(root, tmp_path / "source.tar.gz")
    assert set(result["manifest"]["files"]) == {"data/pilot/tokenizer/tokenizer.json",
        "data/pilot/tokenizer/special_tokens_map.json", "data/pilot/token_validation.json"}


def test_source_archive_respects_git_ignored_duplicate_working_copies(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("data/raw-working/\n")
    for name in ("data/raw-working/observations.jsonl", "data/sealed/manifest.json", "src/code.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    result = archive_source(root, tmp_path / "source.tar.gz")
    assert "data/raw-working/observations.jsonl" not in result["manifest"]["files"]
    assert "data/sealed/manifest.json" in result["manifest"]["files"]
    assert "src/code.py" in result["manifest"]["files"]
