"""Monotonic checkpoint deadlines checked only after completed optimizer steps."""
from __future__ import annotations

import math
import json
import time
import os
import tempfile
from pathlib import Path
from typing import Callable


class CheckpointTimer:
    """A disabled timer never saves; slow steps save as soon as they finish.

    This cannot interrupt a long compilation/backward pass safely. Step-based
    checkpointing remains active independently of the wall-clock interval.
    """

    def __init__(self, interval_seconds: float | None, *, now: float | None = None):
        if interval_seconds is not None and (
            isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds) or interval_seconds <= 0
        ):
            raise ValueError("Checkpoint interval must be positive finite seconds or null")
        self.interval_seconds = interval_seconds
        self.last_saved_at = time.monotonic() if now is None else now

    def due(self, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return self.interval_seconds is not None and current - self.last_saved_at >= self.interval_seconds

    def mark_saved(self, *, now: float | None = None) -> None:
        self.last_saved_at = time.monotonic() if now is None else now


def save_checkpoint_transaction(output: Path, step: int, save: Callable[[Path], None],
                                run_manifest_sha256: str) -> Path:
    """Publish a complete checkpoint on one local POSIX filesystem.

    Trainer writes below a unique hidden staging root. A killed process leaves
    only that hidden root, never a misleading checkpoint-N or overwritten good
    checkpoint. Staging remnants are preserved for inspection, not auto-deleted.
    This provides local crash safety, not off-runtime backup durability.
    """
    from .provenance import checkpoint_evidence, validate_resume_files

    destination = output / f"checkpoint-{step}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    staging_root = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=output))
    staged = staging_root / destination.name
    save(staging_root)
    validate_resume_files(staged)
    state = json.loads((staged / "trainer_state.json").read_text())
    if state.get("global_step") != step:
        raise ValueError("Staged checkpoint step does not agree with trainer state")
    checkpoint_evidence(staged, run_manifest_sha256)
    # Flush content as well as names before making this checkpoint discoverable.
    for path in sorted(staged.rglob("*")):
        if path.is_symlink():
            raise ValueError("Checkpoint files must not be symbolic links")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [p for p in staged.rglob("*") if p.is_dir()]
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True) + [staged]:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    # The audited trainer permits only one writer per run. Never reserve a
    # visible checkpoint-N early: a crash there would block last-good resume.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    os.rename(staged, destination)
    descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    staging_root.rmdir()
    return destination


class AsyncCheckpointSchedule:
    """Bound sealed local snapshots without touching an upload's immutable input.

    The external uploader owns publication and verified retention. When it is
    slow or unavailable we coalesce intermediate requests, rather than block
    training, delete snapshots, or queue mutable tensor references. One extra
    local slot is reserved for a final (or explicit segment-end) snapshot.
    """

    def __init__(self, output: Path, interval_seconds: float, pending_limit: int):
        self.output = output
        self.timer = CheckpointTimer(interval_seconds)
        self.pending_limit = pending_limit
        self.coalesced_requests = 0
        self.last_checkpoint: str | None = None
        self.last_error: str | None = None
        self.blocked_reason: str | None = None

    def begin(self) -> None:
        self.timer.mark_saved()

    def local_checkpoints(self) -> list[Path]:
        import re
        return sorted((p for p in self.output.glob("checkpoint-*")
                       if re.fullmatch(r"checkpoint-[0-9]+", p.name)
                       and (p.is_dir() or p.is_symlink())),
                      key=lambda p: int(p.name.split("-")[1]))

    def should_save(self, *, boundary: bool = False) -> bool:
        if boundary:
            return True
        if not self.timer.due():
            return False
        if len(self.local_checkpoints()) >= self.pending_limit:
            self.coalesced_requests += 1
            self.blocked_reason = "local_checkpoint_backlog"
            # Coalesce one interval at a time, not one status write per step.
            self.timer.mark_saved()
            return False
        self.blocked_reason = None
        return True

    def coalesce_space(self, error: OSError) -> None:
        self.coalesced_requests += 1
        self.blocked_reason = "insufficient_checkpoint_space"
        self.last_error = str(error)
        self.timer.mark_saved()

    def saved(self, checkpoint: str) -> None:
        self.last_checkpoint = checkpoint
        self.blocked_reason = None
        self.timer.mark_saved()

    def status(self) -> dict:
        checkpoints = self.local_checkpoints()
        return {"mode": "asynchronous_external_upload", "status": "coalesced" if self.blocked_reason else "ready",
                "local_checkpoint_count": len(checkpoints), "local_checkpoint_limit": self.pending_limit,
                "final_checkpoint_reserved_slots": 1, "coalesced_checkpoint_requests": self.coalesced_requests,
                "blocked_reason": self.blocked_reason, "last_space_error": self.last_error,
                "last_sealed_checkpoint": self.last_checkpoint or (checkpoints[-1].name if checkpoints else None),
                "backup_status": "not_asserted; consult independently verified controller receipts"}
