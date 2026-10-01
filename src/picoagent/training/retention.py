"""Fail-closed checkpoint mirroring and runtime retention.

The caller must explicitly supply an *existing*, authorized, run-specific durable
root on independently persistent storage. A pathname, a different device number,
or a successful fsync cannot prove that storage is physically off-runtime. None
therefore disables both copying and pruning; this module never chooses a default
backup location. Run/code/tokenizer/dataset archives are the caller's responsibility.

A checkpoint_manifest.json written after Trainer's full-state save is the seal.
The training process must not mutate sealed checkpoints, and all synchronizers
must honor this module's exclusive locks. After an interrupted process, inspect
and remove stale .checkpoint-retention.lock files before retrying. Hidden staging
directories are never considered checkpoints or used as evidence for deletion.
"""
from __future__ import annotations

import contextlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .data import sha256_file
from .provenance import verify_checkpoint

_CHECKPOINT = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
_LOCK = ".checkpoint-retention.lock"


def _hash_tree(root: Path) -> dict[str, str]:
    """Hash every regular file, including the manifest; never follow links."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Checkpoint must be a real directory: {root}")
    hashes = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"Checkpoint contains a link or special file: {path}")
        if stat.S_ISREG(mode):
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def _verified_tree(checkpoint: Path, run_hash: str) -> dict[str, str]:
    hashes = _hash_tree(checkpoint)
    verify_checkpoint(checkpoint, run_hash)
    if _hash_tree(checkpoint) != hashes:
        raise ValueError(f"Checkpoint changed during verification: {checkpoint}")
    return hashes


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


@contextlib.contextmanager
def _exclusive_lock(root: Path) -> Iterator[None]:
    lock = root / _LOCK
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink()


def _copy_run_manifest(run_dir: Path, durable_dir: Path, run_hash: str) -> None:
    source = run_dir / "run_manifest.json"
    target = durable_dir / source.name
    if target.is_symlink():
        raise ValueError("Durable run manifest must not be a symlink")
    if target.exists():
        if not target.is_file() or sha256_file(target) != run_hash:
            raise ValueError("Durable run manifest conflicts with the source run")
        with target.open("rb") as handle:
            os.fsync(handle.fileno())
        return
    descriptor, name = tempfile.mkstemp(prefix=".run-manifest-", dir=durable_dir)
    temporary = Path(name)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != run_hash or sha256_file(source) != run_hash:
            raise ValueError("Run manifest changed or was corrupted during copying")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Durable run manifest appeared during copying: {target}")
        temporary.rename(target)
        _fsync_directory(durable_dir)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_dirs(run_dir: Path) -> list[Path]:
    checkpoints = []
    for path in run_dir.iterdir():
        if _CHECKPOINT.fullmatch(path.name):
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Checkpoint path must be a real directory: {path}")
            checkpoints.append(path)
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[1]))


def _best_name(best_checkpoint: str | None) -> str | None:
    if best_checkpoint is None:
        return None
    name = Path(best_checkpoint).name
    if not _CHECKPOINT.fullmatch(name):
        raise ValueError("best_checkpoint must identify a checkpoint-N directory")
    return name


def mirror_and_prune(
    run_dir: Path,
    durable_dir: Path | None,
    checkpoint: Path,
    best_checkpoint: str | None,
    keep_last: int = 2,
) -> dict[str, Any]:
    """Mirror one sealed checkpoint, then prune verified runtime duplicates.

    ``durable_dir`` is the run-specific destination (checkpoint-N is placed
    directly beneath it), not its parent. Passing it is explicit opt-in and the
    caller's attestation that it is already mounted, authorized durable storage.
    It must exist, and neither root may contain the other. The immutable run
    manifest is also copied and must agree with any existing destination copy.

    Retain at least the latest two checkpoints plus ``best_checkpoint``. Older
    checkpoints without a final verified mirror remain on the runtime. Verify
    *all* proposed deletions before removing any; corruption/copy/flush/conflict
    errors raise and prevent pruning. This is not a multi-directory deletion
    transaction: an OS error during deletion may leave some verified duplicates
    deleted, but their durable copies remain intact. Durable copies are not pruned.
    """
    if not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 2:
        raise ValueError("keep_last must be an integer of at least two")
    if durable_dir is None:
        return {"enabled": False, "mirrored": None, "pruned": [], "retained": []}
    best = _best_name(best_checkpoint)
    run_dir, durable_dir, checkpoint = Path(run_dir), Path(durable_dir), Path(checkpoint)
    if any(path.is_symlink() for path in (run_dir, durable_dir, checkpoint)):
        raise ValueError("Run, durable root, and checkpoint must not be symlinks")
    if not run_dir.is_dir() or not durable_dir.is_dir():
        raise ValueError("Run and explicitly configured durable root must already exist")
    run_dir, durable_dir = run_dir.resolve(), durable_dir.resolve()
    checkpoint = checkpoint.resolve()
    if run_dir.is_relative_to(durable_dir) or durable_dir.is_relative_to(run_dir):
        raise ValueError("Runtime and durable roots must not overlap")
    if checkpoint.parent != run_dir or not _CHECKPOINT.fullmatch(checkpoint.name):
        raise ValueError("checkpoint must be an immediate checkpoint-N child of run_dir")
    run_manifest = run_dir / "run_manifest.json"
    if run_manifest.is_symlink() or not run_manifest.is_file():
        raise ValueError("Run requires a regular run_manifest.json")

    # Exclusive creation refuses concurrent synchronizers and stale crash locks.
    with _exclusive_lock(run_dir), _exclusive_lock(durable_dir):
        run_hash = sha256_file(run_manifest)
        source_hashes = _verified_tree(checkpoint, run_hash)
        _copy_run_manifest(run_dir, durable_dir, run_hash)
        target = durable_dir / checkpoint.name
        if target.exists() or target.is_symlink():
            if _verified_tree(target, run_hash) != source_hashes:
                raise ValueError(f"Durable checkpoint conflicts with source: {target}")
            _fsync_tree(target)
            _fsync_directory(durable_dir)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint.name}.tmp-", dir=durable_dir))
            try:
                shutil.copytree(checkpoint, temporary, dirs_exist_ok=True, symlinks=True)
                if _verified_tree(temporary, run_hash) != source_hashes:
                    raise ValueError("Copied checkpoint failed SHA256 verification")
                if _verified_tree(checkpoint, run_hash) != source_hashes:
                    raise ValueError("Source checkpoint changed during copying")
                _fsync_tree(temporary)
                if target.exists() or target.is_symlink():
                    raise FileExistsError(f"Durable checkpoint appeared during copying: {target}")
                temporary.rename(target)
                _fsync_directory(durable_dir)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        # No source deletion happens until the entire plan is verified.
        checkpoints = _checkpoint_dirs(run_dir)
        keep = {path.name for path in checkpoints[-keep_last:]}
        if best is not None:
            keep.add(best)
        candidates = []
        for path in checkpoints:
            hashes = _verified_tree(path, run_hash)
            if path.name in keep:
                continue
            mirror = durable_dir / path.name
            if not mirror.exists() and not mirror.is_symlink():
                continue
            if _verified_tree(mirror, run_hash) != hashes:
                raise ValueError(f"Durable checkpoint conflicts with source: {mirror}")
            _fsync_tree(mirror)
            candidates.append(path)
        _fsync_directory(durable_dir)
        if sha256_file(run_manifest) != run_hash:
            raise ValueError("Run manifest changed before pruning")
        if sha256_file(durable_dir / "run_manifest.json") != run_hash:
            raise ValueError("Durable run manifest changed before pruning")
        pruned = []
        for path in candidates:
            shutil.rmtree(path)
            pruned.append(path.name)
        if pruned:
            _fsync_directory(run_dir)
        return {
            "enabled": True,
            "mirrored": str(target),
            "pruned": pruned,
            "retained": [path.name for path in checkpoints if path.name not in pruned],
        }
