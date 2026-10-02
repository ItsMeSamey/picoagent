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
