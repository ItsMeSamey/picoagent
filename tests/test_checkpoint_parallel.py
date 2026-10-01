"""Tiny CPU fakes for optional checkpoint prefetch; no account/provider calls."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_sync import (  # noqa: E402
    BUNDLE_MANIFEST, LocalTransfer, pack_checkpoint, pull_checkpoint, sha256,
)
from colab_safe_cli import report_error  # noqa: E402
from colab_sdk_watch import SDKTransfer  # noqa: E402
from picoagent.training.provenance import checkpoint_evidence, verify_checkpoint  # noqa: E402


@pytest.fixture
def packed(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (run / "run_manifest.json").write_text('{"identity":"test"}')
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"repeated" * 5 + name.encode())
    (checkpoint / "trainer_state.json").write_text('{"global_step":1,"log_history":[]}')
    checkpoint_evidence(checkpoint, sha256(run / "run_manifest.json"))
    return pack_checkpoint(run, "checkpoint-1", tmp_path / "export", chunk_bytes=17)


class ParallelTransfer(LocalTransfer):
    download_workers = 4

    def __init__(self, failure=None):
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(self.download_workers)
        self.calls = []
        self.active = self.peak = 0
        self.failure = failure

    def download(self, remote, local):
        if remote.endswith(BUNDLE_MANIFEST):
            return super().download(remote, local)
        with self.lock:
            index = len(self.calls)
            self.calls.append(remote)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if index < self.download_workers:
                self.barrier.wait(timeout=5)
            if self.failure is not None and index == 0:
                local.write_bytes(b"partial")
                raise self.failure
            time.sleep(0.002)
            super().download(remote, local)
        finally:
            with self.lock:
                self.active -= 1


def assert_unpublished(destination):
    assert not (destination / "checkpoint-1").exists()
    assert not (destination / "receipts/checkpoint-1.json").exists()
    assert not list(destination.rglob("*.partial"))
    assert not list(destination.glob(".incoming/materialize-*"))


def test_parallel_roundtrip_is_bounded_and_deduplicates_chunks(packed, tmp_path):
    transfer = ParallelTransfer()
    destination = tmp_path / "durable"
    result = pull_checkpoint(str(packed), destination, transfer, off_runtime=True)
    manifest = json.loads((packed / BUNDLE_MANIFEST).read_text())
    chunks = [c["sha256"] for info in manifest["files"].values() for c in info["chunks"]]
    assert len(chunks) > len(set(chunks)), "Fixture must exercise duplicate chunks"
    assert transfer.peak == 4
    assert transfer.active == 0
    assert len(transfer.calls) == len(set(chunks))
    assert max(Counter(transfer.calls).values()) == 1
    verify_checkpoint(result, sha256(destination / "run_manifest.json"))
    assert (destination / "receipts/checkpoint-1.json").is_file()


@pytest.mark.parametrize("failure", [ConnectionError("fake disconnect"), KeyboardInterrupt()])
def test_parallel_failure_drains_and_resumes_only_verified_cache(packed, tmp_path, failure):
    transfer = ParallelTransfer(failure)
    destination = tmp_path / "durable"
    with pytest.raises(type(failure)):
        pull_checkpoint(str(packed), destination, transfer, off_runtime=True)
    assert_unpublished(destination)
    assert transfer.active == 0
    cached = list(destination.glob(".incoming/checkpoint-1/*/chunks/*"))
    assert cached
    assert all(sha256(path) == path.name for path in cached)

    class Retry(LocalTransfer):
        download_workers = 4
        def download(self, remote, local):
            assert Path(remote).name not in {path.name for path in cached}
            super().download(remote, local)

    result = pull_checkpoint(str(packed), destination, Retry(), off_runtime=True)
    verify_checkpoint(result, sha256(destination / "run_manifest.json"))


def test_main_thread_cancellation_drains_before_return(packed, tmp_path, monkeypatch):
    import checkpoint_sync
    original_wait = checkpoint_sync.wait

    def interrupt_after_first_result(*args, **kwargs):
        original_wait(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint_sync, "wait", interrupt_after_first_result)
    destination = tmp_path / "durable"
    transfer = ParallelTransfer()
    with pytest.raises(KeyboardInterrupt):
        pull_checkpoint(str(packed), destination, transfer, off_runtime=True)
    assert transfer.active == 0
    assert_unpublished(destination)


def test_default_serial_path_never_creates_workers(packed, tmp_path, monkeypatch):
    import checkpoint_sync
    monkeypatch.setattr(checkpoint_sync, "ThreadPoolExecutor",
                        lambda **kwargs: pytest.fail("Serial default must not create threads"))
    result = pull_checkpoint(str(packed), tmp_path / "durable", LocalTransfer(), off_runtime=True)
    assert result.is_dir()


def test_parallel_low_disk_fails_before_payload(packed, tmp_path, monkeypatch):
    import checkpoint_sync

    class ManifestOnly(LocalTransfer):
        download_workers = 4
        def download(self, remote, local):
            assert remote.endswith(BUNDLE_MANIFEST)
            super().download(remote, local)

    monkeypatch.setattr(checkpoint_sync.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    destination = tmp_path / "durable"
    with pytest.raises(OSError, match="Insufficient free space"):
        pull_checkpoint(str(packed), destination, ManifestOnly(), off_runtime=True)
    assert_unpublished(destination)


@pytest.mark.parametrize("damage", ["missing", "truncated", "corrupt", "wrong-file-hash", "path"])
def test_parallel_rejects_missing_malicious_and_corrupt_payload(packed, tmp_path, damage):
    manifest = json.loads((packed / BUNDLE_MANIFEST).read_text())
    info = manifest["files"]["run_manifest.json"]
    chunk = packed / info["chunks"][0]["path"]
    if damage == "missing":
        chunk.unlink()
    elif damage == "truncated":
        chunk.write_bytes(chunk.read_bytes()[:-1])
    elif damage == "corrupt":
        chunk.write_bytes(b"x" * chunk.stat().st_size)
    elif damage == "wrong-file-hash":
        info["sha256"] = "0" * 64
    else:
        info["chunks"][0]["path"] = "../escape"
    (packed / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
    transfer = LocalTransfer()
    transfer.download_workers = 4
    destination = tmp_path / "durable"
    with pytest.raises((FileNotFoundError, ValueError)):
        pull_checkpoint(str(packed), destination, transfer, off_runtime=True)
    assert_unpublished(destination)
    assert not (tmp_path / "escape").exists()


def test_conflicting_duplicate_chunk_size_fails_before_payload(packed, tmp_path):
    manifest = json.loads((packed / BUNDLE_MANIFEST).read_text())
    info = manifest["files"]["run_manifest.json"]
    duplicate = dict(info["chunks"][0])
    duplicate["bytes"] -= 1
    info["chunks"].append(duplicate)
    info["bytes"] += duplicate["bytes"]
    (packed / BUNDLE_MANIFEST).write_text(json.dumps(manifest))

    class ManifestOnly(LocalTransfer):
        download_workers = 4
        def download(self, remote, local):
            assert remote.endswith(BUNDLE_MANIFEST)
            super().download(remote, local)

    destination = tmp_path / "durable"
    with pytest.raises(ValueError, match="Conflicting lengths"):
        pull_checkpoint(str(packed), destination, ManifestOnly(), off_runtime=True)
    assert_unpublished(destination)


@pytest.mark.parametrize("workers", [0, 5, True, 1.5, "4"])
def test_worker_bounds_fail_before_any_transfer(tmp_path, workers):
    class NoCalls(LocalTransfer):
        download_workers = workers
        def download(self, *args):
            pytest.fail("Invalid worker setting must fail before provider access")

    with pytest.raises(ValueError, match="download_workers"):
        pull_checkpoint("unused", tmp_path / "durable", NoCalls(), off_runtime=True)
    with pytest.raises(ValueError, match="download_workers"):
        SDKTransfer(None, "fake", download_workers=workers)


def test_sdk_uses_independent_official_clients_and_serialized_state(monkeypatch, tmp_path):
    calls = []
    state_lock = threading.Lock()
    barrier = threading.Barrier(4)

    class State:
        def get_session(self, name):
            assert name == "fake-session"
            assert state_lock.acquire(blocking=False)
            try:
                time.sleep(0.002)
                return object()
            finally:
                state_lock.release()

    class ContentsClient:
        def __init__(self, session):
            calls.append(self)
        def download(self, remote, local):
            barrier.wait(timeout=5)
            Path(local).write_bytes(remote.encode())

    module = ModuleType("colab_cli.contents")
    module.ContentsClient = ContentsClient
    monkeypatch.setitem(sys.modules, "colab_cli.contents", module)
    transfer = SDKTransfer(State(), "fake-session", download_workers=4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(transfer.download, str(index), tmp_path / str(index)) for index in range(4)]
        for future in futures:
            future.result()
    assert len({id(client) for client in calls}) == 4
    assert transfer.downloads == transfer.bytes == 4
    assert SDKTransfer(State(), "fake-session").download_workers == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500, "PRIVATE_STATUS", True, 999, None])
def test_http_diagnostics_expose_only_valid_status(status, capsys):
    error = RuntimeError("PRIVATE_URL?token=PRIVATE_CREDENTIAL")
    error.response = SimpleNamespace(status_code=status, text="PRIVATE_BODY", headers={"PRIVATE_HEADER": "secret"})
    report_error("transport", error)
    captured = capsys.readouterr().err
    assert "PRIVATE" not in captured
    if type(status) is int and 100 <= status <= 599:
        assert f"HTTP {status}" in captured
    else:
        assert "HTTP" not in captured
