"""Historical preservation tests operate only on inert fixture bytes."""
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preserve_data  # noqa: E402
from source_staging import file_hash  # noqa: E402


def fixture(tmp_path):
    source = tmp_path / "data/raw"
    source.mkdir(parents=True)
    sealed = tmp_path / "data/sealed"
    sealed.mkdir()
    (source / "failed.jsonl").write_bytes(b'{"failed":true}\n')
    (source / "receipt.txt").write_bytes(b"exact\r\nraw\x00bytes\n")
    (sealed / "receipt.bin").write_bytes((source / "receipt.txt").read_bytes())
    with gzip.open(sealed / "partial.gz", "wb") as stream:
        stream.write(b"piece one\n")
    (source / "partial").write_bytes(b"piece one\n")
    with tarfile.open(sealed / "evidence.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo("raw/stdout")
        member.size = 4
        archive.addfile(member, io.BytesIO(b"out\n"))
    (source / "stdout").write_bytes(b"out\n")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)} for p in sealed.iterdir()}
    manifest = sealed / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    return source, manifest


def test_byte_preservation_reuses_canonical_and_restores_failed_attempt(tmp_path, monkeypatch):
    source, _ = fixture(tmp_path)
    monkeypatch.setattr(preserve_data, "PIECE", 4)
    result = preserve_data.preserve(tmp_path, ["data/raw"], ["data/sealed/manifest.json"], "data/history")
    assert result["source_file_count"] == 4 and result["reused_files"] == 3
    assert result["newly_archived_unique_contents"] == 1
    manifest = tmp_path / "data/history/manifest.json"
    for path in source.iterdir():
        restored = tmp_path / "restored" / path.name
        preserve_data.restore_file(tmp_path, manifest, f"data/raw/{path.name}", restored)
        assert restored.read_bytes() == path.read_bytes()
    before = file_hash(source / "failed.jsonl")
    with pytest.raises(FileExistsError):
        preserve_data.restore_file(tmp_path, manifest, "data/raw/failed.jsonl", source / "failed.jsonl")
    assert file_hash(source / "failed.jsonl") == before


def test_archives_are_deterministic(tmp_path):
    fixture(tmp_path)
    first = preserve_data.preserve(tmp_path, ["data/raw"], ["data/sealed/manifest.json"], "data/history1")
    second = preserve_data.preserve(tmp_path, ["data/raw"], ["data/sealed/manifest.json"], "data/history2")
    assert [x["sha256"] for x in first["archives"]] == [x["sha256"] for x in second["archives"]]


def test_credential_pattern_blocks_without_echo(tmp_path):
    source, _ = fixture(tmp_path)
    secret = "hf_" + "x" * 40
    (source / "private").write_text(secret)
    with pytest.raises(ValueError) as exc:
        preserve_data.preserve(tmp_path, ["data/raw"], ["data/sealed/manifest.json"], "data/history")
    assert secret not in str(exc.value)
    assert not (tmp_path / "data/history").exists()


def test_canonical_tamper_blocks_archive(tmp_path):
    fixture(tmp_path)
    (tmp_path / "data/sealed/receipt.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Canonical evidence"):
        preserve_data.preserve(tmp_path, ["data/raw"], ["data/sealed/manifest.json"], "data/history")
    assert not (tmp_path / "data/history").exists()
