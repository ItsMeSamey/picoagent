"""Verified transfer mappings survive interrupted downloads without a provider."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from checkpoint_sync import (  # noqa: E402
    BUNDLE_MANIFEST, LocalTransfer, pack_checkpoint, pull_checkpoint, sha256,
)
from picoagent.training.provenance import checkpoint_evidence  # noqa: E402


@pytest.fixture
def packed(tmp_path):
    root = tmp_path / "run"
    checkpoint = root / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (root / "run_manifest.json").write_text('{"identity":"manifest-cache-test"}')
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes((name + "-ordered-chunks-").encode() * 4)
    (checkpoint / "trainer_state.json").write_text('{"global_step":3,"log_history":[]}')
    checkpoint_evidence(checkpoint, sha256(root / "run_manifest.json"))
    bundle = pack_checkpoint(root, "checkpoint-3", tmp_path / "export", chunk_bytes=17)
    manifest = bundle / BUNDLE_MANIFEST
    # Noncanonical whitespace and key order must survive byte-for-byte.
    parsed = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(dict(reversed(list(parsed.items()))), indent=3) + "\n\n")
    return bundle


def cache_manifest(destination, packed):
    return destination / ".incoming/checkpoint-3" / sha256(packed / BUNDLE_MANIFEST) / BUNDLE_MANIFEST


@pytest.mark.parametrize("workers", [1, 2])
def test_interruption_preserves_exact_manifest_and_ordered_chunk_mapping(packed, tmp_path, workers):
    destination = tmp_path / "durable"
    expected = (packed / BUNDLE_MANIFEST).read_bytes()
    persisted = cache_manifest(destination, packed)
    old_cache = destination / ".incoming/checkpoint-older/untouched/chunks"
    old_cache.mkdir(parents=True)
    (old_cache / "partial-evidence").write_bytes(b"preserve")
    log = destination / "download.log"
    log.write_text("previous attempts\n")
    class Interrupted(LocalTransfer):
        download_workers = workers
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()
        def download(self, remote, local):
            if remote.endswith(BUNDLE_MANIFEST):
                return super().download(remote, local)
            # The mapping is persisted before the very first payload request.
            assert persisted.read_bytes() == expected
            with self.lock:
                self.calls += 1
                index = self.calls
            if index == 3:
                local.write_bytes(b"partial")
                raise ConnectionError("runtime disappeared")
            return super().download(remote, local)
    with pytest.raises(ConnectionError, match="runtime disappeared"):
        pull_checkpoint(str(packed), destination, Interrupted(), off_runtime=True,
                        expected_manifest_sha256=sha256(packed / BUNDLE_MANIFEST))
    assert persisted.read_bytes() == expected
    assert sha256(persisted) == persisted.parent.name
    stored, original = json.loads(persisted.read_bytes()), json.loads(expected)
    for name, info in original["files"].items():
        assert stored["files"][name]["chunks"] == info["chunks"]
    cached = list((persisted.parent / "chunks").iterdir())
    assert cached and all(path.name == sha256(path) for path in cached)
    assert not (destination / "checkpoint-3").exists()
    assert not (destination / "receipts/checkpoint-3.json").exists()
    assert (old_cache / "partial-evidence").read_bytes() == b"preserve"
    assert log.read_text() == "previous attempts\n"
    assert not list((destination / ".incoming").glob("manifest-*.json"))


def test_existing_identical_cache_manifest_is_a_safe_retry(packed, tmp_path):
    destination = tmp_path / "durable"
    persisted = cache_manifest(destination, packed)
    persisted.parent.mkdir(parents=True)
    persisted.write_bytes((packed / BUNDLE_MANIFEST).read_bytes())
    class Interrupted(LocalTransfer):
        def download(self, remote, local):
            if not remote.endswith(BUNDLE_MANIFEST):
                raise ConnectionError("stop before first chunk")
            super().download(remote, local)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            pull_checkpoint(str(packed), destination, Interrupted(), off_runtime=True)
        assert persisted.read_bytes() == (packed / BUNDLE_MANIFEST).read_bytes()


def test_different_cached_manifest_fails_without_overwriting_evidence(packed, tmp_path):
    destination = tmp_path / "durable"
    persisted = cache_manifest(destination, packed)
    persisted.parent.mkdir(parents=True)
    persisted.write_bytes(b"existing conflicting evidence")
    class ManifestOnly(LocalTransfer):
        def download(self, remote, local):
            assert remote.endswith(BUNDLE_MANIFEST)
            super().download(remote, local)
    with pytest.raises(ValueError, match="Cached transfer manifest collision"):
        pull_checkpoint(str(packed), destination, ManifestOnly(), off_runtime=True)
    assert persisted.read_bytes() == b"existing conflicting evidence"
    assert not (destination / "checkpoint-3").exists()
    assert not (destination / "receipts/checkpoint-3.json").exists()


@pytest.mark.parametrize("damage", ["invalid-schema", "wrong-pin"])
def test_invalid_or_unpinned_mismatch_is_never_saved_as_verified(packed, tmp_path, damage):
    destination = tmp_path / "durable"
    expected = sha256(packed / BUNDLE_MANIFEST)
    if damage == "invalid-schema":
        payload = json.loads((packed / BUNDLE_MANIFEST).read_text())
        payload["schema"] = "invalid"
        (packed / BUNDLE_MANIFEST).write_text(json.dumps(payload))
        expected = sha256(packed / BUNDLE_MANIFEST)
    else:
        expected = "0" * 64
    with pytest.raises(ValueError):
        pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True,
                        expected_manifest_sha256=expected)
    assert not list(destination.rglob(BUNDLE_MANIFEST))
    assert not (destination / "checkpoint-3").exists()
    assert not (destination / "receipts/checkpoint-3.json").exists()


def test_cached_manifest_symlink_is_rejected(packed, tmp_path):
    destination = tmp_path / "durable"
    persisted = cache_manifest(destination, packed)
    persisted.parent.mkdir(parents=True)
    evidence = tmp_path / "external.json"
    evidence.write_text("preserve")
    persisted.symlink_to(evidence)
    with pytest.raises(ValueError, match="symlink"):
        pull_checkpoint(str(packed), destination, LocalTransfer(), off_runtime=True)
    assert evidence.read_text() == "preserve"
