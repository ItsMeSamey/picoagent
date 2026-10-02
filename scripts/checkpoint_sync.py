#!/usr/bin/env python3
"""Chunked, checksum-verified checkpoint transfer. No credentials or cloud SDKs.

A local path is NOT proof of durable storage. The operator must explicitly
attest that the destination lives outside the training runtime before receipts
are issued or retention is run. External commands are argv arrays, never shells.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from picoagent.training.provenance import verify_checkpoint  # noqa: E402
from picoagent.training.evaluation import (  # noqa: E402
    MAX_EVALUATION_BYTES, parse_evaluation, read_evaluation,
)

BUNDLE_MANIFEST = "transfer_manifest.json"
SCHEMA = "picoagent.transfer.v1"
CHECKPOINT = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024
MAX_DOWNLOAD_WORKERS = 4


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


def runtime_export_roots(run_dir: Path, export_root: Path) -> tuple[Path, Path]:
    """Check lexical paths before resolution or export creation can hide aliases."""
    roots = []
    for path in (Path(run_dir), Path(export_root)):
        if ".." in path.parts:
            raise ValueError("Runtime retention paths must not contain parent traversal")
        path = path.absolute()
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise ValueError("Runtime retention paths must not contain symlinks")
            if parent.exists() and not parent.is_dir():
                raise ValueError("Runtime retention roots must be directories")
        roots.append(path)
    run, exports = roots
    if run.is_relative_to(exports) or exports.is_relative_to(run):
        raise ValueError("Runtime run and export roots must not overlap")
    if not run.is_dir():
        raise ValueError("Runtime run directory must exist")
    return run, exports


def _retention_tree(root: Path, *, reject_traces: bool = True) -> dict[str, tuple[int, ...]]:
    """Inventory every entry without following links or ignoring hidden files."""
    entries = {}
    pending = [root]
    while pending:
        path = pending.pop()
        info = path.lstat()
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError("Runtime retention refuses symlinks and special files")
        relative = path.relative_to(root).as_posix()
        if reject_traces and relative != "." and (
                path.suffix in {".jsonl", ".ndjson"} or "trace" in path.name.lower()):
            raise ValueError("Runtime retention refuses directories containing traces")
        entries[relative] = (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                             info.st_mtime_ns, info.st_ctime_ns)
        if stat.S_ISDIR(info.st_mode):
            pending.extend(path.iterdir())
    if not stat.S_ISDIR(entries["."][2]):
        raise ValueError("Runtime retention target must be a directory")
    return entries


def preflight_export_trees(export_root: Path, names: list[str]) -> None:
    """Check every existing collection target before any bundle is written.

    Regular partial files are retryable and remain untouched by this preflight;
    links and special files must never reach pack_checkpoint's write paths.
    The caller first validates the export root with runtime_export_roots.
    """
    for name in names:
        bundle = export_root / checkpoint_name(name)
        if bundle.exists() or bundle.is_symlink():
            _retention_tree(bundle, reject_traces=False)


def _exact_retention_tree(root: Path, files: set[str]) -> dict[str, tuple[int, ...]]:
    directories = {"."}
    for name in files:
        path = safe_relative(name)
        if path.as_posix() != name:
            raise ValueError("Runtime retention requires canonical manifest paths")
        directories.update(parent.as_posix() for parent in path.parents)
    tree = _retention_tree(root)
    if files & directories or set(tree) != files | directories:
        raise ValueError("Runtime retention target contains missing or unexpected entries")
    if any(not stat.S_ISREG(tree[name][2]) for name in files):
        raise ValueError("Runtime retention requires regular manifest files")
    return tree


def _runtime_prune_pair(run: Path, exports: Path, name: str,
                        ack: dict[str, Any]) -> tuple[dict, dict]:
    """Bind both exact trees to the controller's durable acknowledgement."""
    checkpoint, bundle = run / name, exports / name
    # Inventory first: neither hashing nor JSON reads may follow special files.
    _retention_tree(checkpoint)
    _retention_tree(bundle)
    manifest_path = bundle / BUNDLE_MANIFEST
    if not manifest_path.is_file() or sha256(manifest_path) != ack["transfer_manifest_sha256"]:
        raise ValueError("Export transfer manifest changed after durable acknowledgement")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    checkpoint_key = f"{name}/checkpoint_manifest.json"
    if (manifest["checkpoint"] != name
            or manifest["files"]["run_manifest.json"]["sha256"] != ack["run_manifest_sha256"]
            or manifest["files"].get(checkpoint_key, {}).get("sha256") != ack["checkpoint_manifest_sha256"]):
        raise ValueError("Export checkpoint/run identity differs from durable acknowledgement")
    for relative in manifest["files"]:
        path = safe_relative(relative)
        if path.as_posix() != relative or any(
                "trace" in part.lower() or Path(part).suffix in {".jsonl", ".ndjson"}
                for part in path.parts):
            raise ValueError("Runtime retention refuses unsafe or trace-containing manifests")
    checkpoint_files = {relative[len(name) + 1:]: info for relative, info in manifest["files"].items()
                        if relative.startswith(name + "/")}
    checkpoint_tree = _exact_retention_tree(checkpoint, set(checkpoint_files))
    seal = checkpoint / "checkpoint_manifest.json"
    if sha256(seal) != ack["checkpoint_manifest_sha256"]:
        raise ValueError("Checkpoint changed after durable acknowledgement")
    sealed_files = json.loads(seal.read_text()).get("files")
    if sealed_files != {relative: info["sha256"] for relative, info in checkpoint_files.items()
                        if relative != "checkpoint_manifest.json"}:
        raise ValueError("Export file identities differ from acknowledged checkpoint seal")
    # verify_checkpoint hashes every payload against this exact sealed mapping;
    # compare export lengths separately without hashing the same payload twice.
    verify_checkpoint(checkpoint, ack["run_manifest_sha256"])
    for relative, info in checkpoint_files.items():
        if checkpoint_tree[relative][3] != info["bytes"]:
            raise ValueError("Checkpoint changed after durable acknowledgement")
    chunks = {chunk["path"]: chunk for info in manifest["files"].values() for chunk in info["chunks"]}
    expected = {BUNDLE_MANIFEST, *chunks}
    evaluation_hash = ack["evaluation_sha256"]
    if evaluation_hash is not None:
        expected.add(f"evaluations/{evaluation_hash}.json")
    bundle_tree = _exact_retention_tree(bundle, expected)
    for relative, chunk in chunks.items():
        path = bundle / relative
        if path.stat().st_size != chunk["bytes"] or sha256(path) != chunk["sha256"]:
            raise ValueError("Export chunk changed after durable acknowledgement")
    if evaluation_hash is not None:
        evaluation = evaluation_record(bundle, manifest)
        if evaluation is None or evaluation["sha256"] != evaluation_hash:
            raise ValueError("Export evaluation differs from durable acknowledgement")
    if _retention_tree(checkpoint) != checkpoint_tree or _retention_tree(bundle) != bundle_tree:
        raise ValueError("Runtime retention target changed during validation")
    return checkpoint_tree, bundle_tree


def _changed_runtime_evaluations(run: Path, acknowledged: dict[str, dict]) -> list[str]:
    changed = []
    for name, ack in acknowledged.items():
        sidecar = run / "evaluations" / f"{name}.json"
        if any(path.is_symlink() for path in (sidecar, *sidecar.parents)):
            raise ValueError("Evaluation sidecar path must not contain symlinks")
        current = hashlib.sha256(_regular_evaluation_bytes(sidecar)).hexdigest() if sidecar.exists() else None
        if current != ack["evaluation_sha256"]:
            changed.append(name)
    return changed


def prune_runtime_exports(run_dir: Path, export_root: Path, acknowledged: dict[str, dict],
                          best_checkpoint: str | None, *,
                          latest_published_repository: str | None = None) -> dict[str, Any]:
    """Prune only paired, acknowledged checkpoint/export trees, latest two + best.

    Validate the whole plan before deleting anything and recheck each pair at
    its deletion boundary. Sealed checkpoints are immutable; racing new saves
    never enter this plan. This is not a multi-directory deletion transaction:
    failure after checkpoint removal deliberately leaves its export untouched.
    """
    run, exports = runtime_export_roots(run_dir, export_root)
    if best_checkpoint is not None:
        checkpoint_name(best_checkpoint)
    for name, ack in acknowledged.items():
        checkpoint_name(name)
        for key in ("checkpoint_manifest_sha256", "transfer_manifest_sha256", "run_manifest_sha256"):
            if not DIGEST.fullmatch(str(ack.get(key, ""))):
                raise ValueError("Runtime retention requires complete durable acknowledgement")
        if "evaluation_sha256" not in ack or (ack["evaluation_sha256"] is not None
                and not DIGEST.fullmatch(str(ack["evaluation_sha256"]))):
            raise ValueError("Runtime retention requires acknowledged evaluation identity")
    run_manifest = run / "run_manifest.json"
    if run_manifest.is_symlink() or not stat.S_ISREG(run_manifest.stat().st_mode):
        raise ValueError("Runtime retention requires a regular run manifest")
    run_hash = sha256(run_manifest)
    if any(ack["run_manifest_sha256"] != run_hash for ack in acknowledged.values()):
        raise ValueError("Run changed after durable acknowledgement")

    def checkpoints() -> list[Path]:
        return sorted((path for path in run.iterdir() if CHECKPOINT.fullmatch(path.name)),
                      key=lambda path: int(path.name.split("-")[1]))

    current = checkpoints()
    latest_public = current[-1].name if current and latest_published_repository else None

    def verify_public(name):
        from picoagent.training.durability import acknowledgement_from_receipt
        path = run / 'durability' / (name + '.json')
        if (path.parent.is_symlink() or path.is_symlink() or not path.is_file()
                or path.stat().st_size > 2 * 1024**2):
            raise ValueError('Latest-only retention requires verified public acknowledgements')
        ack = json.loads(path.read_text())
        if ack != acknowledgement_from_receipt(ack.get('receipt', {})):
            raise ValueError('Public acknowledgement integrity mismatch')
        identity = ack['receipt']['identity']
        if (identity['repository'] != latest_published_repository or identity['checkpoint'] != name
                or any(identity[key] != acknowledged[name][key] for key in
                       ('run_manifest_sha256', 'checkpoint_manifest_sha256', 'transfer_manifest_sha256'))):
            raise ValueError('Public acknowledgement does not match collected checkpoint')
        _runtime_prune_pair(run, exports, name, acknowledged[name])

    if latest_public is not None:
        # A newer save racing collection is unverified. Do not delete anything
        # until a subsequent collection publishes that exact replacement.
        if latest_public not in acknowledged:
            return {'pruned_runtime': [], 'pending_evaluations': []}
        for name in acknowledged:
            verify_public(name)
    keep = {path.name for path in current[-2:]} | ({best_checkpoint} if best_checkpoint else set())
    if latest_public is not None:
        keep = {latest_public}
    candidates = [path.name for path in current if path.name in acknowledged and path.name not in keep]
    result = {"pruned_runtime": [], "pending_evaluations": []}

    def deferred() -> bool:
        result["pending_evaluations"] = _changed_runtime_evaluations(run, acknowledged)
        return bool(result["pending_evaluations"])

    if deferred():
        return result
    plan = {name: _runtime_prune_pair(run, exports, name, acknowledged[name]) for name in candidates}
    root_ids = [(path.stat().st_dev, path.stat().st_ino) for path in (run, exports)] if plan else []
    for name, identity in plan.items():
        runtime_export_roots(run, exports)
        if ([(path.stat().st_dev, path.stat().st_ino) for path in (run, exports)] != root_ids
                or sha256(run_manifest) != run_hash):
            raise ValueError("Runtime retention roots changed before deletion")
        # Never expand a plan when newer checkpoints appear, or delete a path
        # that became one of the latest two after a concurrent removal.
        now = checkpoints()
        if latest_public is not None:
            if not now or now[-1].name != latest_public:
                break
            verify_public(latest_public)
            verify_public(name)
        if name in {path.name for path in now[-(1 if latest_public else 2):]}:
            continue
        if deferred():
            break
        if _runtime_prune_pair(run, exports, name, acknowledged[name]) != identity:
            raise ValueError("Runtime retention target changed before deletion")
        if deferred():
            break
        shutil.rmtree(run / name)
        # The export is removed ONLY after its original checkpoint was removed.
        # A replacement/new bundle is never swept up by the paired deletion.
        runtime_export_roots(run, exports)
        if _retention_tree(exports / name) != identity[1]:
            raise ValueError("Export changed before deletion")
        shutil.rmtree(exports / name)
        result["pruned_runtime"].append(name)
    return result


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

    def evaluation_digest(self, remote: str, manifest: dict[str, Any]) -> str | None:
        """Filesystem discovery can distinguish absence from transport failure."""
        record = evaluation_record(Path(remote), manifest)
        return record["sha256"] if record else None


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
    chunk_lengths = {}
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
            if chunk_lengths.setdefault(digest, chunk["bytes"]) != chunk["bytes"]:
                raise ValueError("Conflicting lengths for duplicate chunk identity")


def _regular_evaluation_bytes(path: Path) -> bytes:
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Evaluation artifact path must not contain symlinks")
    if not path.is_file() or path.stat().st_size > MAX_EVALUATION_BYTES:
        raise ValueError("Evaluation artifact must be a bounded regular file")
    with path.open("rb") as handle:
        data = handle.read(MAX_EVALUATION_BYTES + 1)
    if len(data) > MAX_EVALUATION_BYTES:
        raise ValueError("Evaluation artifact exceeds size limit")
    return data


def _publish_evaluation(path: Path, data: bytes) -> None:
    """Atomic no-clobber publication, accepting only byte-identical retries."""
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Evaluation artifact path must not contain symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _regular_evaluation_bytes(path) != data:
                raise ValueError("Evaluation artifact collision") from None
        sync_directory(path.parent)
        if _regular_evaluation_bytes(path) != data:
            raise ValueError("Evaluation artifact failed read-back verification")
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_record(bundle: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Inspect the independent content-addressed evaluation, if one is present.

    This directory is deliberately outside transfer_manifest.json. A bundle may
    acquire one evaluation after its immutable checkpoint transfer was sealed.
    Multiple evaluations are a collision, never a last-writer-wins selection.
    """
    directory = bundle / "evaluations"
    if any(parent.is_symlink() for parent in (directory, *directory.parents)):
        raise ValueError("Evaluation artifact path must not contain symlinks")
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise ValueError("Evaluation artifacts require a directory")
    files = [path for path in directory.iterdir() if not (path.name.startswith(".") and path.name.endswith(".tmp"))]
    if not files:
        return None
    if len(files) != 1:
        raise ValueError("Evaluation artifact collision")
    path = files[0]
    if path.suffix != ".json" or not DIGEST.fullmatch(path.stem):
        raise ValueError("Invalid evaluation artifact identity")
    if manifest is None:
        manifest = json.loads((bundle / BUNDLE_MANIFEST).read_text())
        validate_manifest(manifest)
    name = manifest["checkpoint"]
    data = _regular_evaluation_bytes(path)
    digest = hashlib.sha256(data).hexdigest()
    if digest != path.stem:
        raise ValueError("Evaluation artifact hash mismatch")
    payload = parse_evaluation(data, name, manifest["files"][f"{name}/checkpoint_manifest.json"]["sha256"])
    return {"path": f"evaluations/{digest}.json", "sha256": digest, "bytes": len(data),
            "eval_loss": payload["metrics"]["eval_loss"]}


def pack_evaluation(run_dir: Path, name: str, bundle: Path) -> dict[str, Any] | None:
    """Add late evidence without touching the checkpoint or transfer manifest."""
    checkpoint = run_dir / name
    source = run_dir / "evaluations" / f"{name}.json"
    existing = evaluation_record(bundle)
    if read_evaluation(source, checkpoint) is None:
        return existing
    data = _regular_evaluation_bytes(source)
    parse_evaluation(data, name, sha256(checkpoint / "checkpoint_manifest.json"))
    digest = hashlib.sha256(data).hexdigest()
    if existing is not None and existing["sha256"] != digest:
        raise ValueError("Evaluation artifact collision")
    _publish_evaluation(bundle / "evaluations" / f"{digest}.json", data)
    return evaluation_record(bundle)


def pull_evaluation(remote: str, destination: Path, transfer: LocalTransfer | CLITransfer,
                    manifest: dict[str, Any], expected_sha256: str | None = None) -> None:
    """Fetch advertised evidence even when the checkpoint already has a receipt.

    Generic command transports must be given the advertised digest: a failed CLI
    download is not proof an optional file is absent. Local filesystems support
    safe discovery. Both flows validate binding and publish without overwriting.
    """
    if expected_sha256 is None and hasattr(transfer, "evaluation_digest"):
        expected_sha256 = transfer.evaluation_digest(remote, manifest)
    if expected_sha256 is None:
        return
    if not isinstance(expected_sha256, str) or not DIGEST.fullmatch(expected_sha256):
        raise ValueError("Invalid expected evaluation SHA256")
    name = manifest["checkpoint"]
    checkpoint = destination / name
    checkpoint_hash = manifest["files"][f"{name}/checkpoint_manifest.json"]["sha256"]
    if sha256(checkpoint / "checkpoint_manifest.json") != checkpoint_hash:
        raise ValueError("Evaluation destination checkpoint hash mismatch")
    saved = destination / "evaluations" / f"{name}.json"
    if saved.exists() or saved.is_symlink():
        existing = _regular_evaluation_bytes(saved)
        parse_evaluation(existing, name, checkpoint_hash)
        if hashlib.sha256(existing).hexdigest() != expected_sha256:
            raise ValueError("Destination evaluation collision")
        return
    incoming = destination / ".incoming"
    incoming.mkdir(exist_ok=True)
    temporary = incoming / f"evaluation-{uuid.uuid4().hex}.json"
    try:
        transfer.download(remote_join(remote, f"evaluations/{expected_sha256}.json"), temporary)
        data = _regular_evaluation_bytes(temporary)
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("Downloaded evaluation hash mismatch")
        parse_evaluation(data, name, checkpoint_hash)
        _publish_evaluation(saved, data)
    finally:
        temporary.unlink(missing_ok=True)


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
        pack_evaluation(run_dir, name, target)
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
    pack_evaluation(run_dir, name, target)
    return target


def upload_bundle(bundle: Path, remote: str, transfer: LocalTransfer | CLITransfer) -> str:
    """Publish manifest LAST, after read-back checks. Use a unique remote prefix.

    A failed upload leaves only orphan chunks and no valid completion marker.
    An existing valid destination must never be reused for a different bundle.
    """
    manifest_path = bundle / BUNDLE_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    evaluation = evaluation_record(bundle, manifest)
    if isinstance(transfer, LocalTransfer) and evaluation is not None:
        remote_evaluations = Path(remote) / "evaluations"
        if any(path.is_symlink() for path in (remote_evaluations, *remote_evaluations.parents)):
            raise ValueError("Remote evaluation artifact path must not contain symlinks")
        # The same content address may contain an interrupted upload. Retry it;
        # a different content identity must never silently replace an evaluation.
        if remote_evaluations.exists() and (not remote_evaluations.is_dir() or any(
                path.name != Path(evaluation["path"]).name or path.is_symlink() or not path.is_file()
                for path in remote_evaluations.iterdir())):
            raise ValueError("Remote evaluation artifact collision")
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
        if evaluation is not None:
            source = bundle / evaluation["path"]
            data = _regular_evaluation_bytes(source)
            if hashlib.sha256(data).hexdigest() != evaluation["sha256"]:
                raise ValueError("Local evaluation artifact changed during upload")
            snapshot = Path(directory) / "evaluation.json"
            snapshot.write_bytes(data)
            transfer.upload(snapshot, remote_join(remote, evaluation["path"]))
            transfer.download(remote_join(remote, evaluation["path"]), readback)
            if _regular_evaluation_bytes(readback) != data:
                raise ValueError("Uploaded evaluation failed read-back verification")
        transfer.upload(manifest_path, remote_join(remote, BUNDLE_MANIFEST))
        transfer.download(remote_join(remote, BUNDLE_MANIFEST), readback)
        if sha256(readback) != sha256(manifest_path):
            raise ValueError("Uploaded completion manifest failed read-back verification")
    return sha256(manifest_path)


def validate_download_workers(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_DOWNLOAD_WORKERS:
        raise ValueError(f"download_workers must be an integer in [1, {MAX_DOWNLOAD_WORKERS}]")
    return value


def _cache_chunk(remote: str, cache: Path, chunk: dict[str, Any], transfer: Any) -> Path:
    """Publish only a complete, fsynced, checksum-verified cache object."""
    cached = cache / chunk["sha256"]
    if cached.is_symlink():
        raise ValueError("Refusing symlink in chunk cache")
    if cached.is_file() and cached.stat().st_size == chunk["bytes"] and sha256(cached) == chunk["sha256"]:
        return cached
    partial = cached.with_name(f"{cached.name}.{uuid.uuid4().hex}.partial")
    try:
        transfer.download(remote_join(remote, chunk["path"]), partial)
        if (partial.is_symlink() or not partial.is_file()
                or partial.stat().st_size != chunk["bytes"] or sha256(partial) != chunk["sha256"]):
            raise ValueError("Downloaded chunk hash mismatch")
        with partial.open("rb") as verified:
            os.fsync(verified.fileno())
        os.replace(partial, cached)
        sync_directory(cache)
        return cached
    finally:
        partial.unlink(missing_ok=True)


def _prefetch_chunks(remote: str, cache: Path, manifest: dict[str, Any], transfer: Any,
                     workers: int) -> None:
    """Bound both active and queued downloads; drain before releasing the lock.

    The transport must support independent concurrent downloads. Interrupts and
    failures cancel pending work, wait for in-flight calls, and never publish a
    checkpoint or receipt. Completed verified cache objects remain resumable.
    """
    chunks = {chunk["sha256"]: chunk for info in manifest["files"].values()
              for chunk in info["chunks"]}
    remaining = iter(chunks.values())
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="checkpoint-download")
    pending = set()
    try:
        for chunk in remaining:
            pending.add(executor.submit(_cache_chunk, remote, cache, chunk, transfer))
            if len(pending) == workers:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Surface every completed failure before submitting replacement work.
            for future in done:
                future.result()
            for _ in done:
                chunk = next(remaining, None)
                if chunk is not None:
                    pending.add(executor.submit(_cache_chunk, remote, cache, chunk, transfer))
    finally:
        for future in pending:
            future.cancel()
        # Official SDK requests are not interruptible: never return while a
        # worker can still write cache files outside the caller's writer lock.
        executor.shutdown(wait=True, cancel_futures=True)


def _preserve_transfer_manifest(probe: Path, data: bytes, incoming: Path,
                                name: str, digest: str) -> None:
    """Keep the exact verified mapping with partial chunks after runtime loss.

    Link the fsynced probe atomically without clobbering existing evidence. JSON
    is intentionally not reserialized: the advertised digest identifies bytes,
    and file/chunk order must remain available for offline reconstruction.
    """
    target = incoming / name / digest / BUNDLE_MANIFEST
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ValueError("Transfer manifest cache must not contain symlinks")
    if probe.is_symlink() or not probe.is_file() or sha256(probe) != digest:
        raise ValueError("Transfer manifest probe changed during validation")
    target.parent.mkdir(parents=True, exist_ok=True)
    with probe.open("rb") as handle:
        os.fsync(handle.fileno())
    try:
        os.link(probe, target)
    except FileExistsError:
        if (target.is_symlink() or not target.is_file() or target.stat().st_size != len(data)
                or sha256(target) != digest or target.read_bytes() != data):
            raise ValueError("Cached transfer manifest collision") from None
    sync_directory(target.parent)
    sync_directory(target.parent.parent)
    sync_directory(incoming)
    sync_directory(incoming.parent)
    if sha256(target) != digest:
        raise ValueError("Cached transfer manifest failed read-back verification")


def pull_checkpoint(remote: str, destination: Path, transfer: LocalTransfer | CLITransfer,
                    *, off_runtime: bool, expected_manifest_sha256: str | None = None,
                    restore: bool = False, expected_evaluation_sha256: str | None = None) -> Path:
    """Resume at chunk granularity, commit checkpoint atomically, then issue receipt.

    Immutable original run metadata is preserved alongside the checkpoint. A
    failed transfer never overwrites a committed checkpoint or creates a receipt.
    Single writer per destination is required (the Colab watcher holds a lock).
    """
    if not off_runtime and not restore:
        raise ValueError("Explicit off_runtime attestation required; same-VM storage is not durable")
    if off_runtime and restore:
        raise ValueError("Restore mode cannot attest a runtime destination as durable")
    workers = validate_download_workers(getattr(transfer, "download_workers", 1))
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("Destination must not be a symlink")
    destination = destination.resolve()
    incoming = destination / ".incoming"
    if incoming.is_symlink():
        raise ValueError("Incoming transfer cache must not be a symlink")
    incoming.mkdir(exist_ok=True)
    # Unique manifest probe means a failed download cannot be mistaken for complete metadata.
    probe = incoming / f"manifest-{uuid.uuid4().hex}.json"
    try:
        transfer.download(remote_join(remote, BUNDLE_MANIFEST), probe)
        if probe.is_symlink() or not probe.is_file():
            raise ValueError("Transfer manifest probe must be a regular file")
        manifest_bytes = probe.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_manifest_sha256 and manifest_digest != expected_manifest_sha256:
            raise ValueError("Transfer manifest hash mismatch")
        manifest = json.loads(manifest_bytes)
        validate_manifest(manifest)
        _preserve_transfer_manifest(probe, manifest_bytes, incoming, manifest["checkpoint"], manifest_digest)
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
                pull_evaluation(remote, destination, transfer, manifest, expected_evaluation_sha256)
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
        if workers > 1:
            _prefetch_chunks(remote, cache, manifest, transfer, workers)
        for relative, info in manifest["files"].items():
            output = stage / safe_relative(relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as handle:
                for chunk in info["chunks"]:
                    cached = _cache_chunk(remote, cache, chunk, transfer)
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
        pull_evaluation(remote, destination, transfer, manifest, expected_evaluation_sha256)
        receipt = {"schema": "picoagent.durable-receipt.v1", "checkpoint": name,
                   "checkpoint_manifest_sha256": sha256(final / "checkpoint_manifest.json"),
                   "run_manifest_sha256": run_hash, "transfer_manifest_sha256": manifest_digest,
                   "destination": str(final), "off_runtime_attested": True}
        if not restore:
            write_json_atomic(destination / "receipts" / f"{name}.json", receipt)
        # Keep the exact pinned manifest for public-release publication/recovery.
        # Only chunk payloads are transient after successful materialization.
        shutil.rmtree(cache)
        sync_directory(cache.parent)
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
    pull.add_argument("--expected-evaluation-sha256", help="Advertised late evaluation digest (required for generic CLI transports)")
    restore = sub.add_parser("restore")
    restore.add_argument("--remote", required=True)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument("--expected-manifest-sha256")
    restore.add_argument("--expected-evaluation-sha256", help="Advertised late evaluation digest (required for generic CLI transports)")
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
                                         expected_manifest_sha256=args.expected_manifest_sha256,
                                         expected_evaluation_sha256=args.expected_evaluation_sha256)
        else:
            result = upload_bundle(args.bundle, args.remote, transfer)
    print(result)


if __name__ == "__main__":
    main()
