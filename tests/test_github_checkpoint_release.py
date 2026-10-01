"""Tiny fake-gh and HTTP-only fixtures; never contact GitHub or any provider."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import github_checkpoint_release as release  # noqa: E402
from checkpoint_sync import BUNDLE_MANIFEST, pack_checkpoint, sha256  # noqa: E402
from picoagent.training.provenance import checkpoint_evidence, verify_checkpoint  # noqa: E402


@pytest.fixture
def prepared(tmp_path):
    run = tmp_path / "run"
    source = run / "source_snapshot/src/picoagent/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# original source fixture\n")
    files = {"src/picoagent/fixture.py": sha256(source)}
    source_hash = release.digest_json(files)
    (run / "run_manifest.json").write_text(json.dumps({
        "schema": "picoagent.training.run.v1", "identity": {"source_tree_sha256": source_hash},
        "code": {"files": files, "tree_sha256": source_hash, "git": {"commit": "a" * 40}},
    }))
    checkpoint = run / "checkpoint-7"
    checkpoint.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes((name + "-payload-").encode() * 3)
    (checkpoint / "trainer_state.json").write_text('{"global_step":7,"log_history":[]}')
    checkpoint_evidence(checkpoint, sha256(run / "run_manifest.json"))
    bundle = pack_checkpoint(run, "checkpoint-7", tmp_path / "export", chunk_bytes=113)
    manifest = bundle / BUNDLE_MANIFEST
    return run, manifest, release.make_plan(run, manifest, "test-owner/test-repo")


class FakeGitHub:
    prefix = "repos/test-owner/test-repo"

    def __init__(self):
        self.current = None
        self.records = {}
        self.payloads = {}
        self.writes = []
        self.reads = []
        self.private = False
        self.tag_exists = False
        self.wrong_tag = False
        self.fail_upload_number = None
        self.damage = None
        self.missing_api_digest = False

    def api(self, path):
        assert path == self.prefix
        return {"private": self.private}

    def check_tag(self, tag, commit, required=False):
        if self.wrong_tag or (required and not self.tag_exists):
            raise ValueError("Source tag collision")

    def release(self, tag):
        return deepcopy(self.current)

    def create(self, plan, body):
        assert self.current is None
        self.writes.append(("create-draft", plan["tag"]))
        self.current = {"id": 123, "tag_name": plan["tag"], "body": body, "draft": True,
                        "target_commitish": plan["identity"]["source_commit"]}

    def assets(self, release_id):
        assert release_id == 123
        return deepcopy(list(self.records.values()))

    def upload(self, tag, path):
        assert self.current["draft"], "Must never upload into a published release"
        assert path.name not in self.records, "Must never clobber an asset"
        if self.fail_upload_number == len(self.records) + 1:
            raise ConnectionError("fake interrupted upload")
        self.writes.append(("upload", path.name))
        data = path.read_bytes()
        self.payloads[path.name] = data
        self.records[path.name] = {"id": len(self.records) + 1, "name": path.name,
                                   "size": len(data), "state": "uploaded", "download_count": 0}
        if not self.missing_api_digest:
            self.records[path.name]["digest"] = "sha256:" + sha256(path)

    @contextmanager
    def stream(self, asset):
        self.reads.append(asset["name"])
        self.records[asset["name"]]["download_count"] += 1
        data = self.payloads[asset["name"]]
        if self.damage == "corrupt":
            data = b"x" * len(data)
        elif self.damage == "truncated":
            data = data[:-1]
        elif self.damage == "oversized":
            data += b"extra"
        yield io.BytesIO(data)

    def publish(self, tag):
        assert list(self.records)[-1] == BUNDLE_MANIFEST
        self.writes.append(("publish", tag))
        self.current["draft"] = False
        self.tag_exists = True


class FakePublicHTTP:
    def __init__(self, client):
        self.client = client
        self.requests = []

    def open(self, request, timeout):
        assert timeout == 60
        assert not any(key.lower() == "authorization" for key in request.headers)
        assert urlparse(request.full_url).hostname == "github.com"
        assert not self.client.current["draft"], "Draft assets must not be called public"
        self.requests.append(request.full_url)
        return io.BytesIO(self.client.payloads[request.full_url.rsplit("/", 1)[1]])


def upload(prepared, client, **kwargs):
    run, manifest, plan = prepared
    return release.upload_checkpoint(run, manifest, plan,
                                     approved_plan_sha256=release.digest_json(plan), client=client, **kwargs)


def test_plan_is_deterministic_exact_identity_and_local_only(prepared, monkeypatch):
    run, manifest, plan = prepared
    monkeypatch.setattr(release, "GitHub", lambda _: pytest.fail("Plan must never create a GitHub client"))
    assert release.make_plan(run, manifest, "test-owner/test-repo") == plan
    assert plan["identity"]["run_manifest_sha256"] == sha256(run / "run_manifest.json")
    assert plan["identity"]["transfer_manifest_sha256"] == sha256(manifest)
    assert plan["identity"]["source_commit"] == "a" * 40
    assert plan["identity"]["source_tree_sha256"] in plan["tag"]
    assert "checkpoint-7" in plan["tag"]
    assert len(plan["assets"]) < 1000


def test_exact_approval_required_before_any_github_access(prepared, monkeypatch):
    run, manifest, plan = prepared
    monkeypatch.setattr(release, "GitHub", lambda _: pytest.fail("No remote calls without approval"))
    with pytest.raises(ValueError, match="approved plan"):
        release.upload_checkpoint(run, manifest, plan, approved_plan_sha256="0" * 64)


def test_draft_staging_is_not_a_durable_release_receipt(prepared):
    client = FakeGitHub()
    result = upload(prepared, client)
    assert result["schema"] == "picoagent.github-draft-staging.v1"
    assert result["published"] is False
    assert client.writes[0][0] == "create-draft"
    assert client.writes[-1] == ("upload", BUNDLE_MANIFEST)
    assert not any(action == "publish" for action, _ in client.writes)
    assert set(client.reads) == set(prepared[2]["assets"])


def test_publish_verified_assets_and_anonymous_restore(prepared, tmp_path):
    client = FakeGitHub()
    opener = FakePublicHTTP(client)
    result = upload(prepared, client, publish=True, public_opener=opener)
    assert result["published"] and result["schema"] == "picoagent.github-release-receipt.v1"
    assert client.writes[-1][0] == "publish"
    assert len(opener.requests) == 1 and opener.requests[0].endswith(BUNDLE_MANIFEST)
    assert result["independent_readback_verified"]
    destination = tmp_path / "restore"
    restored = release.restore_checkpoint(prepared[2], destination,
                                          expected_plan_sha256=release.digest_json(prepared[2]), opener=opener)
    verify_checkpoint(restored, sha256(destination / "run_manifest.json"))
    assert sha256(restored / "optimizer.pt") == sha256(prepared[0] / "checkpoint-7/optimizer.pt")
    assert not (destination / "receipts").exists(), "Runtime restore must not claim durability"


def test_interrupted_upload_resumes_without_replacing_assets(prepared):
    client = FakeGitHub()
    client.fail_upload_number = 3
    with pytest.raises(ConnectionError):
        upload(prepared, client)
    assert BUNDLE_MANIFEST not in client.records
    existing = deepcopy(client.records)
    client.fail_upload_number = None
    result = upload(prepared, client)
    assert not result["published"]
    for name, record in existing.items():
        assert client.records[name]["id"] == record["id"]
        assert [entry for entry in client.writes if entry == ("upload", name)] == [("upload", name)]


@pytest.mark.parametrize("collision", ["digest", "size", "state", "body", "source", "extra", "missing", "tag"])
def test_existing_identity_or_asset_collision_fails_without_writes(prepared, collision):
    client = FakeGitHub()
    upload(prepared, client)
    name = next(iter(client.records))
    if collision == "digest":
        client.records[name]["digest"] = "sha256:" + "0" * 64
    elif collision == "size":
        client.records[name]["size"] += 1
    elif collision == "state":
        client.records[name]["state"] = "starter"
    elif collision == "body":
        client.current["body"] = "another run"
    elif collision == "source":
        client.current["target_commitish"] = "b" * 40
    elif collision == "extra":
        client.records["unexpected"] = {**client.records[name], "name": "unexpected"}
    elif collision == "missing":
        del client.records[name]
    else:
        client.wrong_tag = True
    before = list(client.writes)
    with pytest.raises(ValueError):
        upload(prepared, client)
    assert client.writes == before


@pytest.mark.parametrize("damage", ["corrupt", "truncated", "oversized"])
def test_independent_readback_required_even_with_matching_api_digest(prepared, damage):
    client = FakeGitHub()
    client.damage = damage
    with pytest.raises(ValueError, match="Read-back"):
        upload(prepared, client, publish=True)
    assert BUNDLE_MANIFEST not in client.records
    assert client.current["draft"] is True
    assert not any(action == "publish" for action, _ in client.writes)


def test_missing_api_digest_still_requires_full_readback(prepared):
    client = FakeGitHub()
    client.missing_api_digest = True
    result = upload(prepared, client)
    assert result["independent_readback_verified"]
    assert set(client.reads) == set(prepared[2]["assets"])


def test_public_approval_does_not_allow_different_repository_visibility(prepared):
    client = FakeGitHub()
    client.private = True
    with pytest.raises(ValueError, match="visibility"):
        upload(prepared, client)
    assert client.writes == []


def test_changed_checkpoint_fails_before_remote_creation(prepared):
    run, _, _ = prepared
    (run / "checkpoint-7/optimizer.pt").write_bytes(b"changed")
    client = FakeGitHub()
    with pytest.raises(ValueError, match="Local file"):
        upload(prepared, client)
    assert client.writes == []


@pytest.mark.parametrize("limit", ["count", "size"])
def test_github_asset_limits_fail_before_remote_creation(prepared, monkeypatch, limit):
    run, manifest, _ = prepared
    monkeypatch.setattr(release, "MAX_ASSETS" if limit == "count" else "ASSET_LIMIT", 1)
    with pytest.raises(ValueError, match="limits"):
        release.make_plan(run, manifest, "test-owner/test-repo")


def test_missing_source_snapshot_cannot_claim_exact_source_identity(prepared):
    run, manifest, _ = prepared
    data = json.loads(manifest.read_text())
    del data["files"]["source_snapshot/src/picoagent/fixture.py"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source snapshot"):
        release.make_plan(run, manifest, "test-owner/test-repo")


@pytest.mark.parametrize("which", ["manifest", "chunk"])
def test_corrupt_public_restore_does_not_complete(prepared, tmp_path, which):
    client = FakeGitHub()
    opener = FakePublicHTTP(client)
    upload(prepared, client, publish=True, public_opener=opener)
    name = BUNDLE_MANIFEST if which == "manifest" else next(iter(client.payloads))
    client.payloads[name] = b"corrupt"
    destination = tmp_path / "restore"
    with pytest.raises(ValueError, match="Read-back"):
        release.restore_checkpoint(prepared[2], destination,
                                   expected_plan_sha256=release.digest_json(prepared[2]), opener=opener)
    assert not (destination / "checkpoint-7").exists()
    assert not (destination / "receipts").exists()


def test_restore_requires_independently_pinned_plan(prepared, tmp_path):
    with pytest.raises(ValueError, match="trusted plan"):
        release.restore_checkpoint(prepared[2], tmp_path / "restore", expected_plan_sha256="0" * 64)


def test_gh_adapter_uses_exact_host_target_draft_and_no_clobber(prepared, monkeypatch, tmp_path):
    calls = []
    client = release.GitHub("test-owner/test-repo")
    monkeypatch.setattr(client, "call", lambda *args: calls.append(args) or "[]")
    client.check_tag(prepared[2]["tag"], "a" * 40)
    client.create(prepared[2], "pinned body")
    client.upload(prepared[2]["tag"], tmp_path / "chunk.bin")
    client.publish(prepared[2]["tag"])
    assert calls[0][:5] == ("api", "--hostname", "github.com", "--method", "GET")
    assert "--draft" in calls[1] and "--target" in calls[1] and "a" * 40 in calls[1]
    assert all("--clobber" not in args and "delete" not in args for args in calls)
    assert "--draft=false" in calls[-1]


def test_gh_binary_stream_uses_fake_executable_only(tmp_path, monkeypatch):
    fake = tmp_path / "gh"
    fake.write_text(f"#!{sys.executable}\nimport sys\nassert '--hostname' in sys.argv\nsys.stdout.buffer.write(b'checkpoint')\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    with release.GitHub("test-owner/test-repo").stream({"id": 12}) as stream:
        assert stream.read() == b"checkpoint"


@pytest.mark.parametrize("url", ["http://github.com/asset", "https://evil.example/asset", "file:///tmp/file"])
def test_anonymous_restore_rejects_unexpected_redirects(url):
    with pytest.raises(ValueError, match="redirect"):
        release.GitHubRedirects().redirect_request(None, None, 302, "", {}, url)
