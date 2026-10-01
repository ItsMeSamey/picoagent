"""Bounded source transport, bootstrapped verbatim on an explicitly named runtime.

Standard library only. No provider calls, credentials, or dataset execution.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
MAX_CHUNK_BYTES = 256 * 1024 * 1024
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value: str) -> Path:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("Unsafe source path")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("Unsafe source path")
    return Path(*path.parts)


def no_symlink_path(path: Path) -> Path:
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError("Unsafe source location")
    for parent in [*reversed(path.parents), path]:
        if parent.is_symlink():
            raise ValueError("Source staging does not accept symlinks")
    return path


def integrity_record(info: dict) -> None:
    if (not isinstance(info, dict) or not DIGEST.fullmatch(str(info.get("sha256", "")))
            or type(info.get("bytes")) is not int or info["bytes"] < 0):
        raise ValueError("Invalid source integrity record")


def matches(path: Path, info: dict) -> bool:
    no_symlink_path(path)
    return path.is_file() and path.stat().st_size == info["bytes"] and file_hash(path) == info["sha256"]


def pack_archive(archive: Path, source_manifest: dict, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> tuple[Path, dict]:
    """Stream to content-addressed chunks; reuse only hash-verified local chunks."""
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= MAX_CHUNK_BYTES:
        raise ValueError("chunk_bytes must be in (0, 256MiB]")
    archive = no_symlink_path(archive)
    digest = file_hash(archive)
    root = no_symlink_path(Path(str(archive) + ".chunks") / digest)
    root.mkdir(parents=True, exist_ok=True)
    chunks, total, overall = [], 0, hashlib.sha256()
    with archive.open("rb") as stream:
        while data := stream.read(chunk_bytes):
            chunk = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            path = root / chunk["sha256"]
            if not matches(path, chunk):
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=root, prefix=".chunk-", delete=False) as target:
                        temporary = Path(target.name)
                        target.write(data)
                        target.flush()
                        os.fsync(target.fileno())
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            chunks.append(chunk)
            overall.update(data)
            total += len(data)
    if overall.hexdigest() != digest:
        raise ValueError("Source archive changed while chunking")
    payload = json.dumps(source_manifest, sort_keys=True, indent=2).encode() + b"\n"
    bundle = {"schema": "picoagent.source-transfer.v1", "archive": {"sha256": digest, "bytes": total},
              "chunks": chunks, "source_manifest_sha256": hashlib.sha256(payload).hexdigest(),
              "source_manifest_bytes": len(payload),
              "extracted_bytes": sum(info["bytes"] for info in source_manifest["files"].values())}
    validate_bundle(bundle)
    return root, bundle


def validate_archive_identity(bundle: dict) -> None:
    if bundle.get("schema") != "picoagent.source-transfer.v1":
        raise ValueError("Unknown source transfer schema")
    integrity_record(bundle["archive"])
    if (not DIGEST.fullmatch(str(bundle.get("source_manifest_sha256", "")))
            or type(bundle.get("source_manifest_bytes")) is not int
            or not 0 < bundle["source_manifest_bytes"] <= 64 * 1024 * 1024
            or type(bundle.get("extracted_bytes")) is not int or bundle["extracted_bytes"] < 0):
        raise ValueError("Invalid source manifest bounds")


def validate_bundle(bundle: dict) -> None:
    validate_archive_identity(bundle)
    if not isinstance(bundle.get("chunks"), list) or not bundle["chunks"]:
        raise ValueError("Source transfer requires chunks")
    for chunk in bundle["chunks"]:
        integrity_record(chunk)
        if not 0 < chunk["bytes"] <= MAX_CHUNK_BYTES:
            raise ValueError("Invalid source chunk size")
    if sum(chunk["bytes"] for chunk in bundle["chunks"]) != bundle["archive"]["bytes"]:
        raise ValueError("Source chunk sizes do not match archive")


def probe_chunks(cache: str, bundle: dict) -> dict:
    """Recover verified uploads after disconnect; incomplete parts are retried."""
    validate_bundle(bundle)
    root = no_symlink_path(Path(cache))
    root.mkdir(parents=True, exist_ok=True)
    verified = []
    for chunk in bundle["chunks"]:
        target = root / chunk["sha256"]
        partial = root / (chunk["sha256"] + ".partial")
        if matches(target, chunk):
            verified.append(chunk["sha256"])
        elif matches(partial, chunk):
            os.replace(partial, target)
            verified.append(chunk["sha256"])
    return {"verified_chunks": sorted(set(verified))}


def commit_chunk(cache: str, chunk: dict) -> dict:
    integrity_record(chunk)
    if not 0 < chunk["bytes"] <= MAX_CHUNK_BYTES:
        raise ValueError("Invalid source chunk size")
    root = no_symlink_path(Path(cache))
    target = root / chunk["sha256"]
    partial = root / (chunk["sha256"] + ".partial")
    if not matches(partial, chunk):
        raise ValueError("Uploaded source chunk failed verification")
    no_symlink_path(target)
    os.replace(partial, target)
    return {"verified_chunk": chunk["sha256"]}


def verify_tree(target: Path, bundle: dict) -> dict:
    manifest_path = target / "SOURCE_MANIFEST.json"
    if not matches(manifest_path, {"sha256": bundle["source_manifest_sha256"],
                                   "bytes": bundle["source_manifest_bytes"]}):
        raise ValueError("Source manifest failed verification")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "picoagent.source-archive.v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid source archive manifest")
    expected = {"SOURCE_MANIFEST.json"}
    for name, info in manifest["files"].items():
        relative_path(name)
        integrity_record(info)
        if name == "SOURCE_MANIFEST.json" or not matches(target / name, info):
            raise ValueError("Extracted source file failed verification")
        expected.add(name)
    actual = set()
    for path in target.rglob("*"):
        no_symlink_path(path)
        if path.is_file():
            actual.add(path.relative_to(target).as_posix())
        elif not path.is_dir():
            raise ValueError("Unexpected source tree entry")
    if actual != expected or sum(info["bytes"] for info in manifest["files"].values()) != bundle["extracted_bytes"]:
        raise ValueError("Extracted source tree differs from manifest")
    return manifest


def materialize_source(cache: str, project: str, bundle: dict) -> dict:
    """Verify archive and every file, then publish atomically; never overwrite a run."""
    validate_bundle(bundle)
    root, target = no_symlink_path(Path(cache)), no_symlink_path(Path(project))
    if not Path(project).is_absolute() or target == Path(target.anchor):
        raise ValueError("Project must be a non-root absolute path")
    if root == target or root in target.parents or target in root.parents:
        raise ValueError("Transfer cache and project must be separate")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        # An exact completed retry is harmless. Any run outputs or changes fail.
        manifest = verify_tree(target, bundle)
        return {"project": str(target), "source_files": len(manifest["files"]), "verified": True, "resumed": True}
    root.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    required = bundle["archive"]["bytes"] + bundle["extracted_bytes"] + bundle["source_manifest_bytes"] + 16 * 1024 * 1024
    if shutil.disk_usage(root).free < required or shutil.disk_usage(target.parent).free < required:
        raise OSError("Insufficient free space for verified source staging")
    archive = root / "source.tar.gz"
    if not matches(archive, bundle["archive"]):
        partial = no_symlink_path(root / "source.tar.gz.partial")
        digest = hashlib.sha256()
        with partial.open("wb") as output:
            for chunk in bundle["chunks"]:
                source = root / chunk["sha256"]
                if not matches(source, chunk):
                    raise ValueError("Missing or corrupt source chunk")
                with source.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != bundle["archive"]["sha256"] or partial.stat().st_size != bundle["archive"]["bytes"]:
            raise ValueError("Reconstructed source archive failed verification")
        os.replace(partial, archive)
    return extract_source_archive(archive, str(target), bundle)


def extract_source_archive(archive: Path, project: str, identity: dict) -> dict:
    """Verify a mounted/local archive and publish its exact tree atomically.

    identity uses source-transfer.v1's archive, source_manifest_sha256,
    source_manifest_bytes and extracted_bytes fields; chunks are not required.
    This is also usable by explicit input-dataset package staging.
    """
    validate_archive_identity(identity)
    bundle = identity
    archive, target = no_symlink_path(Path(archive)), no_symlink_path(Path(project))
    if not Path(project).is_absolute() or target == Path(target.anchor):
        raise ValueError("Project must be a non-root absolute path")
    if not matches(archive, bundle["archive"]):
        raise ValueError("Source archive failed verification")
    if archive.is_relative_to(target):
        raise ValueError("Source archive and destination must be separate")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        manifest = verify_tree(target, bundle)
        return {"project": str(target), "source_files": len(manifest["files"]), "verified": True, "resumed": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < bundle["extracted_bytes"] + bundle["source_manifest_bytes"] + 16 * 1024 * 1024:
        raise OSError("Insufficient free space for verified source extraction")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        seen, total = set(), 0
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                relative = relative_path(member.name)
                if not member.isfile() or member.name in seen or member.size < 0:
                    raise ValueError("Unsafe or duplicate source archive entry")
                seen.add(member.name)
                total += member.size
                if total > bundle["extracted_bytes"] + bundle["source_manifest_bytes"]:
                    raise ValueError("Source archive exceeds declared bounds")
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                path.chmod(0o755 if member.mode & 0o100 else 0o644)
        manifest = verify_tree(temporary, bundle)
        no_symlink_path(target)
        if target.exists() and any(target.iterdir()):
            raise ValueError("Project changed during source staging")
        os.rename(temporary, target)
        return {"project": str(target), "source_files": len(manifest["files"]), "verified": True, "resumed": False}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
