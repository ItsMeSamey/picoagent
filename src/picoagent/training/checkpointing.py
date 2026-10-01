"""Monotonic checkpoint deadlines checked only after completed optimizer steps."""
from __future__ import annotations

import math
import time


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
