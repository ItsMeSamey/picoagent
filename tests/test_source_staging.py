"""Local-only source staging: no provider calls or learner execution."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from colab_run import Colab, archive_source, stage  # noqa: E402
from source_staging import (commit_chunk, file_hash, materialize_source, pack_archive,
                            probe_chunks, relative_path)  # noqa: E402


def put(root, name, payload=b"fixture"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def record(path):
    return {"bytes": path.stat().st_size, "sha256": file_hash(path)}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    put(root, "src/picoagent/main.py", b"# reviewed fixture\n")
    put(root, "README.md")
    put(root, "data/history/large.raw", b"historical working copy")
    put(root, "data/chosen/train.jsonl", b"observed fixture\n")
    put(root, "data/chosen/tokenizer/tokenizer.json", b'{"public":"tokenizer"}')
    manifest = put(root, "data/chosen/manifest.json")
    files = {name: record(manifest.parent / name) for name in ("train.jsonl", "tokenizer/tokenizer.json")}
    manifest.write_text(json.dumps({"files": files}))
    return root, manifest


class FakeColab:
    """Run only the reviewed staging helpers against a local temporary directory."""
    def __init__(self):
        self.uploads = []
        self.fail_after = None
        self.corrupt = False

    def execute(self, code):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(code, "local_staging_fixture", "exec"), {})
        return json.loads(stream.getvalue().split("PICOAGENT_RESULT=")[-1])

    def command(self, command, local, remote, capture=False):
        assert command == "upload" and capture
        if self.fail_after is not None and len(self.uploads) >= self.fail_after:
            raise ConnectionError("fixture interruption")
        destination = Path("/" + remote.lstrip("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, destination)
        if self.corrupt:
            destination.write_bytes(b"incomplete")
        self.uploads.append((Path(local).stat().st_size, remote))
        return subprocess.CompletedProcess([], 0, "", "")


def test_selected_snapshot_excludes_history_and_includes_public_tokenizer(source, tmp_path):
    root, manifest = source
    output = tmp_path / "source.tar.gz"
    result = archive_source(root, output, manifest)
    assert set(result["manifest"]["files"]) == {
        "README.md", "src/picoagent/main.py", "data/chosen/manifest.json",
        "data/chosen/train.jsonl", "data/chosen/tokenizer/tokenizer.json"}
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == set(result["manifest"]["files"]) | {"SOURCE_MANIFEST.json"}
    digest = file_hash(output)
    archive_source(root, output, manifest)
    assert file_hash(output) == digest
    assert (root / "data/history/large.raw").read_bytes() == b"historical working copy"


def test_selected_required_ignored_files_still_included(source, tmp_path):
    root, manifest = source
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    put(root, ".gitignore", b"data/\n")
    result = archive_source(root, tmp_path / "s.tar.gz", manifest)
    assert "data/chosen/train.jsonl" in result["manifest"]["files"]


@pytest.mark.parametrize("failure", ["missing", "changed", "secret", "symlink", "ancestor_symlink", "escape", "absolute"])
def test_required_dataset_cannot_be_silently_filtered_or_escape(source, tmp_path, failure):
    root, manifest = source
    body = json.loads(manifest.read_text())
    path = root / "data/chosen/train.jsonl"
    if failure == "missing":
        path.unlink()
    elif failure == "changed":
        path.write_text("different")
    elif failure == "secret":
        bad = put(root, "data/chosen/credentials.json")
        body["files"][bad.name] = record(bad)
    elif failure == "symlink":
        path.unlink()
        path.symlink_to(root / "README.md")
    elif failure == "ancestor_symlink":
        (manifest.parent / "linked").symlink_to(root, target_is_directory=True)
        body["files"]["linked/README.md"] = record(root / "README.md")
    else:
        name = "../history/large.raw" if failure == "escape" else str(root / "README.md")
        body["files"][name] = record(root / "README.md")
    manifest.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        archive_source(root, tmp_path / "s.tar.gz", manifest)
    assert not (tmp_path / "s.tar.gz").exists()


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../b", "a//b", "./a", "a\\b", "a\x00b"])
def test_unsafe_relative_paths(name):
    with pytest.raises(ValueError):
        relative_path(name)


def test_stage_bounded_upload_resume_and_exact_repeat(source, tmp_path):
    root, manifest = source
    client = FakeColab()
    client.fail_after = 2
    project, cache = tmp_path / "remote_project", tmp_path / "remote_cache"
    kwargs = dict(dataset_manifest=manifest, chunk_bytes=128, transfer_root=str(cache))
    with pytest.raises(ConnectionError):
        stage(client, root, str(project), tmp_path / "s.tar.gz", **kwargs)
    assert not project.exists()
    prior = list(client.uploads)
    client.fail_after = None
    result = stage(client, root, str(project), tmp_path / "s.tar.gz", **kwargs)
    assert result["verified"] and all(size <= 128 for size, _ in client.uploads)
    assert len({remote for _, remote in client.uploads}) == len(client.uploads)
    assert client.uploads[:2] == prior
    assert (project / "data/chosen/train.jsonl").read_bytes() == (manifest.parent / "train.jsonl").read_bytes()
    uploaded = len(client.uploads)
    repeat = stage(client, root, str(project), tmp_path / "s.tar.gz", **kwargs)
    assert repeat["resumed"] and repeat["uploaded_chunks"] == 0
    assert len(client.uploads) == uploaded
    put(project, "run_status.json")
    with pytest.raises(RuntimeError):
        stage(client, root, str(project), tmp_path / "s.tar.gz", **kwargs)
    assert (project / "run_status.json").exists()


def test_corrupt_upload_not_published_then_repaired(source, tmp_path):
    root, manifest = source
    client = FakeColab()
    client.corrupt = True
    project = tmp_path / "remote"
    kwargs = dict(dataset_manifest=manifest, chunk_bytes=128, transfer_root=str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="staging failed"):
        stage(client, root, str(project), tmp_path / "s.tar.gz", **kwargs)
    assert not project.exists()
    client.corrupt = False
    assert stage(client, root, str(project), tmp_path / "s.tar.gz", **kwargs)["verified"]


def test_valid_partial_upload_is_recovered_without_reupload(source, tmp_path):
    root, manifest = source
    archive = tmp_path / "s.tar.gz"
    source_manifest = archive_source(root, archive, manifest)["manifest"]
    chunks, bundle = pack_archive(archive, source_manifest, 128)
    cache = tmp_path / "cache"
    cache.mkdir()
    first = bundle["chunks"][0]
    shutil.copyfile(chunks / first["sha256"], cache / (first["sha256"] + ".partial"))
    assert probe_chunks(str(cache), bundle)["verified_chunks"] == [first["sha256"]]
    assert (cache / first["sha256"]).is_file()


def test_symlink_upload_target_rejected(tmp_path):
    payload = b"bounded"
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / (digest + ".partial")).symlink_to(tmp_path / "victim")
    with pytest.raises(ValueError, match="symlink"):
        commit_chunk(str(tmp_path), {"sha256": digest, "bytes": len(payload)})


@pytest.mark.parametrize("mutation", ["hash", "extra", "duplicate", "escape", "symlink"])
def test_bad_archive_never_published(source, tmp_path, mutation):
    root, manifest = source
    archive = tmp_path / "original.tar.gz"
    source_manifest = archive_source(root, archive, manifest)["manifest"]
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(archive) as original, tarfile.open(bad, "w:gz") as output:
        for member in original:
            payload = original.extractfile(member).read()
            if mutation == "hash" and member.name == "README.md":
                payload = b"changed"
                member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))
            if mutation == "duplicate" and member.name == "README.md":
                output.addfile(member, io.BytesIO(payload))
        if mutation in {"extra", "escape", "symlink"}:
            entry = tarfile.TarInfo("../escape" if mutation == "escape" else "unexpected")
            if mutation == "symlink":
                entry.type, entry.linkname = tarfile.SYMTYPE, "../../victim"
            output.addfile(entry, io.BytesIO(b""))
    chunks, bundle = pack_archive(bad, source_manifest, 128)
    with pytest.raises(ValueError):
        materialize_source(str(chunks), str(tmp_path / "published"), bundle)
    assert not (tmp_path / "published").exists()
    assert not (tmp_path / "escape").exists()


def test_colab_missing_result_does_not_echo_private_output(monkeypatch):
    client = Colab("fixture")
    monkeypatch.setattr(client, "command", lambda *a, **kw: subprocess.CompletedProcess([], 0, "PRIVATE_URL", "PRIVATE_TOKEN"))
    with pytest.raises(RuntimeError) as exc:
        client.execute("pass")
    assert "PRIVATE" not in str(exc.value)


def test_archive_cannot_overwrite_selected_evidence(source):
    root, manifest = source
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="outside approved"):
        archive_source(root, manifest, manifest)
    assert manifest.read_bytes() == before


def test_clean_git_root_archive_still_resumes(source, tmp_path):
    root, manifest = source
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "fixture"], cwd=root, check=True)
    output = root / "source.tar.gz"
    first = archive_source(root, output, manifest)
    pack_archive(output, first["manifest"], 128)
    second = archive_source(root, output, manifest)
    assert first["sha256"] == second["sha256"]
    assert first["manifest"]["git_dirty"] is second["manifest"]["git_dirty"] is False


def test_low_disk_does_not_create_project(source, tmp_path, monkeypatch):
    import source_staging
    from types import SimpleNamespace
    root, manifest = source
    output = tmp_path / "s.tar.gz"
    info = archive_source(root, output, manifest)
    chunks, bundle = pack_archive(output, info["manifest"], 128)
    monkeypatch.setattr(source_staging.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="Insufficient free space"):
        materialize_source(str(chunks), str(tmp_path / "project"), bundle)
    assert not (tmp_path / "project").exists()


def test_corrupt_local_chunk_is_repaired(source, tmp_path):
    root, manifest = source
    output = tmp_path / "s.tar.gz"
    info = archive_source(root, output, manifest)
    chunks, bundle = pack_archive(output, info["manifest"], 128)
    first = bundle["chunks"][0]
    (chunks / first["sha256"]).write_bytes(b"corrupt")
    again, repeated = pack_archive(output, info["manifest"], 128)
    assert repeated == bundle and again == chunks
    assert file_hash(chunks / first["sha256"]) == first["sha256"]


@pytest.mark.parametrize("size", [0, -1, 256 * 1024 * 1024 + 1, True])
def test_invalid_chunk_size_fails_locally(source, tmp_path, size):
    root, manifest = source
    client = FakeColab()
    with pytest.raises(ValueError, match="chunk_bytes"):
        stage(client, root, str(tmp_path / "project"), tmp_path / "s.tar.gz",
              dataset_manifest=manifest, chunk_bytes=size)
    assert client.uploads == []


def test_staging_never_reads_payloads_with_read_bytes(source, tmp_path, monkeypatch):
    root, manifest = source
    def forbidden(*args, **kwargs):
        raise AssertionError("Whole payload read_bytes is forbidden")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = stage(FakeColab(), root, str(tmp_path / "project"), tmp_path / "s.tar.gz",
                   dataset_manifest=manifest, chunk_bytes=128,
                   transfer_root=str(tmp_path / "remote_cache"))
    assert result["verified"]


def test_mounted_archive_extractor_needs_no_transfer_chunks(source, tmp_path):
    from source_staging import extract_source_archive
    root, manifest = source
    archive = tmp_path / "s.tar.gz"
    info = archive_source(root, archive, manifest)
    payload = json.dumps(info["manifest"], sort_keys=True, indent=2).encode() + b"\n"
    identity = {"schema": "picoagent.source-transfer.v1", "archive": record(archive),
                "source_manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "source_manifest_bytes": len(payload),
                "extracted_bytes": sum(v["bytes"] for v in info["manifest"]["files"].values())}
    result = extract_source_archive(archive, str(tmp_path / "project"), identity)
    assert result["verified"] and not result["resumed"]
    assert extract_source_archive(archive, str(tmp_path / "project"), identity)["resumed"]
