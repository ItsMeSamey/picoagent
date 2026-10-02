"""Bounded publication tests use fake GitHub only, never live credentials/network."""
from contextlib import contextmanager
from copy import deepcopy
import threading
import time

import pytest

from test_github_checkpoint_release import (
    BUNDLE_MANIFEST, FakeGitHub, FakePublicHTTP, prepared as prepared, release, upload,
)


class ConcurrentGitHub(FakeGitHub):
    def __init__(self, plan, workers):
        super().__init__()
        self.mutex = threading.Lock()
        self.active_uploads = self.active_reads = 0
        self.max_uploads = self.max_reads = 0
        self.max_files = self.max_bytes = 0
        self.finished_reads = set()
        self.chunk_names = set(plan["assets"]) - {BUNDLE_MANIFEST}
        self.workers = workers
        self.fail_name = None
        self.failure = ConnectionError
        self.leave_starter = False
        self.paths = []

    def assets(self, release_id):
        with self.mutex:
            assert self.active_uploads == 0, "Inventory must wait for sibling uploads"
            return super().assets(release_id)

    def upload(self, tag, path):
        with self.mutex:
            if path.name == BUNDLE_MANIFEST:
                assert self.finished_reads == self.chunk_names
                assert self.active_reads == self.active_uploads == 0
            self.paths.append(path)
            self.active_uploads += 1
            self.max_uploads = max(self.max_uploads, self.active_uploads)
            files = list(path.parent.iterdir())
            self.max_files = max(self.max_files, len(files))
            sizes = []
            for temporary in files:
                try:
                    sizes.append(temporary.stat().st_size)
                except FileNotFoundError:
                    pass  # A sibling may have completed and removed its chunk.
            self.max_bytes = max(self.max_bytes, sum(sizes))
            assert all(size <= release.PARALLEL_CHUNK_LIMIT for size in sizes)
            assert len(files) <= self.workers
        try:
            time.sleep(0.015)
            with self.mutex:
                if path.name == self.fail_name:
                    if self.leave_starter:
                        self.records[path.name] = {"id": 999, "name": path.name,
                                                   "size": 0, "state": "starter"}
                    raise self.failure("fake parallel interruption")
                super().upload(tag, path)
        finally:
            with self.mutex:
                self.active_uploads -= 1

    @contextmanager
    def stream(self, asset):
        with self.mutex:
            self.active_reads += 1
            self.max_reads = max(self.max_reads, self.active_reads)
        try:
            time.sleep(0.01)
            with super().stream(asset) as stream:
                yield stream
            if asset["name"] != BUNDLE_MANIFEST:
                with self.mutex:
                    self.finished_reads.add(asset["name"])
        finally:
            with self.mutex:
                self.active_reads -= 1


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_bounded_parallel_publish_manifest_last_and_complete_readback(prepared, workers):
    client = ConcurrentGitHub(prepared[2], workers)
    result = upload(prepared, client, upload_workers=workers, publish=True,
                    public_opener=FakePublicHTTP(client))
    assert result["published"] and result["independent_readback_verified"]
    assert set(result["assets"]) == set(prepared[2]["assets"])
    assert set(client.reads) == set(prepared[2]["assets"])
    assert 1 <= client.max_files <= workers
    assert 0 < client.max_bytes <= workers * 32 * 1024**2
    assert client.max_uploads == client.max_reads == workers
    assert client.active_reads == client.active_uploads == 0
    assert all(not path.exists() and not path.parent.exists() for path in client.paths)
    assert client.writes[-2] == ("upload", BUNDLE_MANIFEST)
    before = list(client.writes)
    ids = deepcopy(client.records)
    upload(prepared, client, upload_workers=workers, public_opener=FakePublicHTTP(client))
    assert before == client.writes
    assert {name: value["id"] for name, value in ids.items()} == {
        name: value["id"] for name, value in client.records.items()}


@pytest.mark.parametrize("workers", [0, 5, -1, 1.5, True, "2", None])
def test_invalid_workers_fail_before_github(prepared, workers):
    client = FakeGitHub()
    client.api = lambda _: pytest.fail("Invalid worker count must fail before remote access")
    with pytest.raises(ValueError, match="upload_workers"):
        upload(prepared, client, upload_workers=workers)


def test_parallel_chunk_limit_fails_before_github(prepared, monkeypatch):
    client = FakeGitHub()
    client.api = lambda _: pytest.fail("Oversized parallel chunk must fail before remote access")
    monkeypatch.setattr(release, "PARALLEL_CHUNK_LIMIT", 1)
    with pytest.raises(ValueError, match="32 MiB"):
        upload(prepared, client, upload_workers=2)
    # Legacy serial accepts the original valid manifest without this new limit.
    upload(prepared, FakeGitHub())


@pytest.mark.parametrize("failure", [ConnectionError, KeyboardInterrupt])
def test_failure_drains_workers_and_resume_reuses_immutable_assets(prepared, failure):
    client = ConcurrentGitHub(prepared[2], 4)
    client.fail_name = sorted(client.chunk_names)[0]
    client.failure = failure
    with pytest.raises(failure, match="interruption"):
        upload(prepared, client, upload_workers=4, publish=True)
    assert client.active_reads == client.active_uploads == 0
    assert BUNDLE_MANIFEST not in client.records and client.current["draft"]
    assert all(not path.exists() and not path.parent.exists() for path in client.paths)
    writes = list(client.writes)
    time.sleep(0.04)
    assert writes == client.writes, "No background writes after exceptional return"
    previous = deepcopy(client.records)
    client.fail_name = None
    upload(prepared, client, upload_workers=4)
    for name, record in previous.items():
        assert client.records[name]["id"] == record["id"]
        assert client.writes.count(("upload", name)) == 1


def test_partial_starter_upload_requires_inspection_not_clobber(prepared):
    client = ConcurrentGitHub(prepared[2], 4)
    client.fail_name = sorted(client.chunk_names)[0]
    client.leave_starter = True
    with pytest.raises(ConnectionError):
        upload(prepared, client, upload_workers=4)
    before = list(client.writes)
    client.fail_name = None
    with pytest.raises(ValueError, match="collision"):
        upload(prepared, client, upload_workers=4)
    assert before == client.writes
    assert BUNDLE_MANIFEST not in client.records


@pytest.mark.parametrize("damage", ["corrupt", "truncated", "oversized"])
def test_parallel_readback_failure_never_uploads_manifest(prepared, damage):
    client = ConcurrentGitHub(prepared[2], 4)
    client.damage = damage
    with pytest.raises(ValueError, match="Read-back"):
        upload(prepared, client, upload_workers=4, publish=True)
    assert client.active_reads == client.active_uploads == 0
    assert BUNDLE_MANIFEST not in client.records
    assert client.current["draft"]
    assert not any(action == "publish" for action, _ in client.writes)


def test_missing_digest_parallel_still_reads_every_byte(prepared):
    client = ConcurrentGitHub(prepared[2], 3)
    client.missing_api_digest = True
    assert upload(prepared, client, upload_workers=3)["independent_readback_verified"]
    assert set(client.reads) == set(prepared[2]["assets"])


def test_main_thread_interrupt_waits_for_active_batch(monkeypatch):
    started, finished = threading.Event(), threading.Event()

    def operation(_):
        started.set()
        time.sleep(0.06)
        finished.set()

    def interrupt(_):
        assert started.wait(timeout=2)
        raise KeyboardInterrupt

    monkeypatch.setattr(release, "as_completed", interrupt)
    with pytest.raises(KeyboardInterrupt):
        release.run_batch(["chunk"], operation, 2)
    assert finished.is_set()


def test_parallel_inventory_detects_changed_preexisting_id(prepared):
    client = ConcurrentGitHub(prepared[2], 2)
    client.fail_name = sorted(client.chunk_names)[2]
    with pytest.raises(ConnectionError):
        upload(prepared, client, upload_workers=2)
    client.fail_name = None
    original_upload = client.upload
    name = sorted(client.records)[0]

    def mutate(tag, path):
        original_upload(tag, path)
        with client.mutex:
            client.records[name]["id"] += 1000

    client.upload = mutate
    with pytest.raises(ValueError, match="changed during upload"):
        upload(prepared, client, upload_workers=2)
    assert BUNDLE_MANIFEST not in client.records


def test_manifest_upload_cannot_mask_changed_verified_chunk(prepared):
    client = ConcurrentGitHub(prepared[2], 2)
    original_upload = client.upload

    def mutate(tag, path):
        original_upload(tag, path)
        if path.name == BUNDLE_MANIFEST:
            name = sorted(client.chunk_names)[0]
            client.records[name]["id"] += 1000

    client.upload = mutate
    with pytest.raises(ValueError, match="changed during manifest upload"):
        upload(prepared, client, upload_workers=2, publish=True)
    assert client.current["draft"]
    assert not any(action == "publish" for action, _ in client.writes)
