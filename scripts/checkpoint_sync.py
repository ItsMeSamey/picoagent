#!/usr/bin/env python3
"""Chunked, checksum-verified checkpoint transfer. No credentials or cloud SDKs.

A local path is NOT proof of durable storage. The operator must explicitly
attest that the destination lives outside the training runtime before receipts
are issued or retention is run. External commands are argv arrays, never shells.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from picoagent.training.provenance import verify_checkpoint  # noqa: E402

BUNDLE_MANIFEST = "transfer_manifest.json"
SCHEMA = "picoagent.transfer.v1"
CHECKPOINT = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError("Unsafe manifest path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in value.split("/")):
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return Path(*path.parts)


def checkpoint_name(value: str) -> str:
    if not CHECKPOINT.fullmatch(value):
        raise ValueError(f"Invalid checkpoint name: {value!r}")
    return value


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Symlinks are not allowed in checkpoint transfers")


def run_cli(argv: list[str], *, timeout: float, capture: bool = False) -> subprocess.CompletedProcess:
    """Bound and reap only this invocation's process group, including CLI threads.

    CLI 0.7.4 can leak reconnecting client threads when initial connection setup
    raises before its cleanup block. A timeout/interrupt must not strand clients.
    """
    process = subprocess.Popen(argv, text=True, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE if capture else None,
                               stderr=subprocess.PIPE if capture else None,
                               start_new_session=True)
    def terminate_group() -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        # Parent may have exited while a descendant ignored SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            terminate_group()
            raise subprocess.CalledProcessError(process.returncode, argv, stdout, stderr)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    except BaseException:
        terminate_group()
        raise


class LocalTransfer:
    """Same API as CLITransfer, useful for mounted storage and tests."""

    def download(self, remote: str, local: Path) -> None:
        source = Path(remote)
        if source.is_symlink():
            raise ValueError("Refusing symlink transfer")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local)

    def upload(self, local: Path, remote: str) -> None:
        target = Path(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)


class CLITransfer:
    """A trusted CLI argv template containing literal {local}/{remote} tokens.

    Example download: ["colab", "download", "--session", "training", "{remote}", "{local}"]
    Command configuration is executable code; never accept it from downloaded data.
    """

    def __init__(self, download: list[str], upload: list[str] | None = None, timeout: int = 600):
        for command in (download, upload):
            if command is not None and (
                not isinstance(command, list) or not command
                or not all(isinstance(arg, str) for arg in command)
                or "{local}" not in command or "{remote}" not in command
            ):
                raise ValueError("Transfer command must be a JSON argv array with {local} and {remote}")
        self.download_command, self.upload_command, self.timeout = download, upload, timeout

    def _run(self, command: list[str] | None, local: Path, remote: str) -> None:
        if command is None:
            raise ValueError("No upload command configured")
        substitutions = {"{local}": str(local), "{remote}": remote}
        argv = [substitutions.get(arg, arg) for arg in command]
        run_cli(argv, timeout=self.timeout)

    def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._run(self.download_command, local, remote)

    def upload(self, local: Path, remote: str) -> None:
        self._run(self.upload_command, local, remote)


def remote_join(root: str, name: str) -> str:
    safe_relative(name)
    return root.rstrip("/") + "/" + name


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown transfer schema")
    name = checkpoint_name(manifest.get("checkpoint", ""))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "run_manifest.json" not in files:
        raise ValueError("Transfer requires run manifest and checkpoint files")
    if f"{name}/checkpoint_manifest.json" not in files:
        raise ValueError("Missing checkpoint completion manifest")
    for relative, info in files.items():
        path = safe_relative(relative)
        if path.parts[0] not in {name, "run_manifest.json", "dataset_manifest.json", "source_snapshot", "tokenizer_snapshot"}:
            raise ValueError("Unexpected file in checkpoint transfer")
        if not DIGEST.fullmatch(str(info.get("sha256", ""))) or not isinstance(info.get("bytes"), int) or info["bytes"] < 0:
            raise ValueError("Invalid file integrity record")
        chunks = info.get("chunks")
        if not isinstance(chunks, list) or sum(chunk.get("bytes", -1) for chunk in chunks) != info["bytes"]:
            raise ValueError("Invalid chunk sizes")
        for chunk in chunks:
            digest = chunk.get("sha256", "")
            if not DIGEST.fullmatch(str(digest)) or chunk.get("path") != f"chunks/{digest}":
                raise ValueError("Invalid chunk identity")
            if not isinstance(chunk.get("bytes"), int) or not 0 < chunk["bytes"] <= 256 * 1024 * 1024:
                raise ValueError("Invalid chunk length")


def pack_checkpoint(run_dir: Path, name: str, export_root: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Path:
    """Publish an immutable export manifest only after every bounded chunk exists."""
    name = checkpoint_name(name)
    if not 0 < chunk_bytes <= 256 * 1024 * 1024:
        raise ValueError("chunk_bytes must be in (0, 256MiB]")
    run_dir, export_root = run_dir.resolve(), export_root.resolve()
    if export_root == run_dir or run_dir in export_root.parents:
        raise ValueError("Export directory must be outside the run directory")
    checkpoint = run_dir / name
    reject_symlinks(checkpoint)
    run_hash = sha256(run_dir / "run_manifest.json")
    verify_checkpoint(checkpoint, run_hash)
    target = export_root / name
    if (target / BUNDLE_MANIFEST).is_file():
        old = json.loads((target / BUNDLE_MANIFEST).read_text())
        validate_manifest(old)
        if old["files"][f"{name}/checkpoint_manifest.json"]["sha256"] != sha256(checkpoint / "checkpoint_manifest.json"):
            raise ValueError("Existing export is for another checkpoint")
        return target
    target.mkdir(parents=True, exist_ok=True)
    files = [run_dir / "run_manifest.json", *sorted(checkpoint.rglob("*"))]
    for optional in ("dataset_manifest.json", "source_snapshot", "tokenizer_snapshot"):
        source = run_dir / optional
        if source.is_dir():
            reject_symlinks(source)
            files.extend(sorted(source.rglob("*")))
        elif source.is_file():
            files.append(source)
    required = sum(source.stat().st_size for source in files if source.is_file()) + 256 * 1024 * 1024
    if shutil.disk_usage(export_root).free < required:
        raise OSError("Insufficient free space for checkpoint export; existing checkpoints remain untouched")
    records = {}
    for source in files:
        if not source.is_file():
            continue
        if source.is_symlink():
            raise ValueError("Refusing symlink in snapshot")
        relative = source.relative_to(run_dir).as_posix()
        chunks, size, file_hash = [], 0, hashlib.sha256()
        with source.open("rb") as handle:
            while data := handle.read(chunk_bytes):
                digest = hashlib.sha256(data).hexdigest()
                path = target / "chunks" / digest
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or sha256(path) != digest:
                    temporary = path.with_suffix(".partial")
                    with temporary.open("wb") as output:
                        output.write(data)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, path)
                chunks.append({"path": f"chunks/{digest}", "sha256": digest, "bytes": len(data)})
                size += len(data)
                file_hash.update(data)
        records[relative] = {"sha256": file_hash.hexdigest(), "bytes": size, "chunks": chunks}
    # Catch a save changing while copied. A checkpoint becomes immutable after its manifest.
    verify_checkpoint(checkpoint, run_hash)
    manifest = {"schema": SCHEMA, "checkpoint": name, "files": records}
    validate_manifest(manifest)
    sync_directory(target / "chunks")
    write_json_atomic(target / BUNDLE_MANIFEST, manifest)
    return target


def upload_bundle(bundle: Path, remote: str, transfer: LocalTransfer | CLITransfer) -> str:
    """Publish manifest LAST, after read-back checks. Use a unique remote prefix.

    A failed upload leaves only orphan chunks and no valid completion marker.
    An existing valid destination must never be reused for a different bundle.
    """
    manifest_path = bundle / BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    chunks = {chunk["path"]: chunk for info in manifest["files"].values() for chunk in info["chunks"]}
    with tempfile.TemporaryDirectory(prefix="picoagent-readback-") as directory:
        readback = Path(directory) / "verify"
        for relative, chunk in chunks.items():
            source = bundle / safe_relative(relative)
            if sha256(source) != chunk["sha256"]:
                raise ValueError("Local export chunk is corrupt")
            transfer.upload(source, remote_join(remote, relative))
            transfer.download(remote_join(remote, relative), readback)
            if sha256(readback) != chunk["sha256"] or readback.stat().st_size != chunk["bytes"]:
                raise ValueError("Uploaded chunk failed read-back verification")
        transfer.upload(manifest_path, remote_join(remote, BUNDLE_MANIFEST))
        transfer.download(remote_join(remote, BUNDLE_MANIFEST), readback)
        if sha256(readback) != sha256(manifest_path):
            raise ValueError("Uploaded completion manifest failed read-back verification")
    return sha256(manifest_path)


def pull_checkpoint(remote: str, destination: Path, transfer: LocalTransfer | CLITransfer,
                    *, off_runtime: bool, expected_manifest_sha256: str | None = None,
                    restore: bool = False) -> Path:
    """Resume at chunk granularity, commit checkpoint atomically, then issue receipt.

    Immutable original run metadata is preserved alongside the checkpoint. A
    failed transfer never overwrites a committed checkpoint or creates a receipt.
    Single writer per destination is required (the Colab watcher holds a lock).
    """
    if not off_runtime and not restore:
        raise ValueError("Explicit off_runtime attestation required; same-VM storage is not durable")
    if off_runtime and restore:
        raise ValueError("Restore mode cannot attest a runtime destination as durable")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Destination must not be a symlink")
    destination = destination.resolve()
    incoming = destination / ".incoming"
    incoming.mkdir(exist_ok=True)
    # Unique manifest probe means a failed download cannot be mistaken for complete metadata.
    probe = incoming / f"manifest-{uuid.uuid4().hex}.json"
    try:
        transfer.download(remote_join(remote, BUNDLE_MANIFEST), probe)
        manifest_digest = sha256(probe)
        if expected_manifest_sha256 and manifest_digest != expected_manifest_sha256:
            raise ValueError("Transfer manifest hash mismatch")
        manifest = json.loads(probe.read_text())
        validate_manifest(manifest)
    finally:
        probe.unlink(missing_ok=True)
    name = manifest["checkpoint"]
    receipt_path = destination / "receipts" / f"{name}.json"
    final = destination / name
    if not restore and receipt_path.is_file() and final.is_dir():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("transfer_manifest_sha256") == manifest_digest:
            reject_symlinks(final)
            verify_checkpoint(final, sha256(destination / "run_manifest.json"))
            if sha256(final / "checkpoint_manifest.json") == receipt.get("checkpoint_manifest_sha256"):
                for relative, info in manifest["files"].items():
                    saved = destination / safe_relative(relative)
                    if saved.is_symlink() or not saved.is_file() or sha256(saved) != info["sha256"]:
                        raise ValueError("Previously saved run snapshot failed integrity verification")
                return final
    # Keep room for both downloaded chunks and atomic materialization. Existing
    # checkpoints are never deleted implicitly to make an incoming transfer fit.
    required = 2 * sum(info["bytes"] for info in manifest["files"].values()) + 256 * 1024 * 1024
    if shutil.disk_usage(destination).free < required:
        raise OSError("Insufficient free space for checkpoint download and atomic reconstruction")
    cache = incoming / name / manifest_digest / "chunks"
    cache.mkdir(parents=True, exist_ok=True)
    stage = incoming / f"materialize-{name}-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        for relative, info in manifest["files"].items():
            output = stage / safe_relative(relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                for chunk in info["chunks"]:
                    cached = cache / chunk["sha256"]
                    if not cached.is_file() or cached.stat().st_size != chunk["bytes"] or sha256(cached) != chunk["sha256"]:
                        partial = cached.with_suffix(".partial")
                        transfer.download(remote_join(remote, chunk["path"]), partial)
                        if partial.stat().st_size != chunk["bytes"] or sha256(partial) != chunk["sha256"]:
                            raise ValueError("Downloaded chunk hash mismatch")
                        with partial.open("rb") as verified:
                            os.fsync(verified.fileno())
                        os.replace(partial, cached)
                        sync_directory(cache)
                    with cached.open("rb") as piece:
                        shutil.copyfileobj(piece, handle)
                handle.flush()
                os.fsync(handle.fileno())
            if output.stat().st_size != info["bytes"] or sha256(output) != info["sha256"]:
                raise ValueError("Reassembled file hash mismatch")
        run_hash = sha256(stage / "run_manifest.json")
        verify_checkpoint(stage / name, run_hash)
        for folder in sorted((path for path in stage.rglob("*") if path.is_dir()), reverse=True):
            sync_directory(folder)
        sync_directory(stage)
        # Any existing metadata must be identical: never mix independent runs.
        for relative, info in manifest["files"].items():
            if safe_relative(relative).parts[0] == name:
                continue
            target = destination / safe_relative(relative)
            if target.exists() and (target.is_symlink() or not target.is_file() or sha256(target) != info["sha256"]):
                raise ValueError(f"Destination contains different run metadata: {relative}")
        final = destination / name
        if final.exists():
            reject_symlinks(final)
            verify_checkpoint(final, run_hash)
            if sha256(final / "checkpoint_manifest.json") != sha256(stage / name / "checkpoint_manifest.json"):
                raise ValueError("Destination checkpoint collision")
        for relative in manifest["files"]:
            if safe_relative(relative).parts[0] == name:
                continue
            target = destination / safe_relative(relative)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / safe_relative(relative), target)
                sync_directory(target.parent)
        if not final.exists():
            os.rename(stage / name, final)
        sync_directory(destination)
        verify_checkpoint(final, run_hash)
        receipt = {"schema": "picoagent.durable-receipt.v1", "checkpoint": name,
                   "checkpoint_manifest_sha256": sha256(final / "checkpoint_manifest.json"),
                   "run_manifest_sha256": run_hash, "transfer_manifest_sha256": manifest_digest,
                   "destination": str(final), "off_runtime_attested": True}
        if not restore:
            write_json_atomic(destination / "receipts" / f"{name}.json", receipt)
        shutil.rmtree(cache.parent)  # Only transient chunk cache; never training traces.
        return final
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--run-dir", type=Path, required=True)
    pack.add_argument("--checkpoint", required=True)
    pack.add_argument("--export-root", type=Path, required=True)
    pack.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    pull = sub.add_parser("pull")
    pull.add_argument("--remote", required=True)
    pull.add_argument("--destination", type=Path, required=True)
    pull.add_argument("--off-runtime", action="store_true", required=True)
    pull.add_argument("--expected-manifest-sha256")
    restore = sub.add_parser("restore")
    restore.add_argument("--remote", required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--expected-manifest-sha256")
    push = sub.add_parser("push")
    push.add_argument("--bundle", type=Path, required=True)
    push.add_argument("--remote", required=True)
    for command in (pull, push, restore):
        command.add_argument("--download-command-json", help="JSON argv array, or omit for filesystem copy")
        command.add_argument("--upload-command-json")
        command.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.action == "pack":
        result = pack_checkpoint(args.run_dir, args.checkpoint, args.export_root, args.chunk_bytes)
    else:
        transfer = (CLITransfer(json.loads(args.download_command_json),
                               json.loads(args.upload_command_json) if args.upload_command_json else None,
                               args.timeout) if args.download_command_json else LocalTransfer())
        if args.action in {"pull", "restore"}:
            from picoagent.training.retention import _exclusive_lock
            args.destination.mkdir(parents=True, exist_ok=True)
            with _exclusive_lock(args.destination):
                result = pull_checkpoint(args.remote, args.destination, transfer,
                                         off_runtime=args.action == "pull", restore=args.action == "restore",
                                         expected_manifest_sha256=args.expected_manifest_sha256)
        else:
            result = upload_bundle(args.bundle, args.remote, transfer)
    print(result)


if __name__ == "__main__":
    main()
