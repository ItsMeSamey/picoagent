"""Validate evaluation evidence bound to an already sealed checkpoint.

Evaluation may finish after checkpoint publication. Its separate, immutable
sidecar never changes the full-state checkpoint or its completion manifest.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .data import sha256_file

EVALUATION_SCHEMA = "picoagent.training.evaluation.v1"
MAX_EVALUATION_BYTES = 64 * 1024
_CHECKPOINT = re.compile(r"checkpoint-(0|[1-9][0-9]*)\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def validate_evaluation(payload: Any, checkpoint_path: Path) -> dict[str, Any]:
    """Return a validated payload or raise; do not verify the checkpoint's tree.

    Callers must separately verify the sealed checkpoint before trusting its
    contents. Name, step, schema, manifest digest and finite scalar metrics are
    checked here, including an obligatory eval_loss. Booleans are not numbers.
    """
    checkpoint_path = Path(checkpoint_path)
    name = checkpoint_path.name
    manifest = checkpoint_path / "checkpoint_manifest.json"
    if checkpoint_path.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Evaluation requires a sealed regular checkpoint manifest")
    return validate_evaluation_binding(payload, name, sha256_file(manifest))


def validate_evaluation_binding(payload: Any, name: str, manifest_sha256: str) -> dict[str, Any]:
    """Validate against a checkpoint manifest digest already verified by transport."""
    if not _CHECKPOINT.fullmatch(name):
        raise ValueError("Invalid evaluation checkpoint name")
    if not isinstance(payload, dict) or payload.get("schema") != EVALUATION_SCHEMA:
        raise ValueError("Unknown evaluation schema")
    if payload.get("checkpoint") != name:
        raise ValueError("Evaluation checkpoint name mismatch")
    step = payload.get("global_step")
    if type(step) is not int or step != int(name.split("-")[1]):
        raise ValueError("Evaluation checkpoint step mismatch")
    digest = payload.get("checkpoint_manifest_sha256")
    if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
            or digest != manifest_sha256):
        raise ValueError("Evaluation checkpoint manifest hash mismatch")
    metrics = payload.get("metrics")
    if (not isinstance(metrics, dict) or "eval_loss" not in metrics
            or any(not isinstance(key, str) or not finite_number(value)
                   for key, value in metrics.items())):
        raise ValueError("Evaluation requires finite numeric metrics and eval_loss")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate evaluation JSON key")
        result[key] = value
    return result


def parse_evaluation(data: bytes, name: str, manifest_sha256: str) -> dict[str, Any]:
    if len(data) > MAX_EVALUATION_BYTES:
        raise ValueError("Evaluation sidecar exceeds size limit")
    payload = json.loads(data, object_pairs_hook=_unique_object)
    return validate_evaluation_binding(payload, name, manifest_sha256)


def read_evaluation(path: Path, checkpoint_path: Path) -> dict[str, Any] | None:
    """Read bounded regular-file evidence; only a missing sidecar returns None."""
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Evaluation sidecar path must not contain symlinks")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_size > MAX_EVALUATION_BYTES:
        raise ValueError("Evaluation sidecar must be a bounded regular file")
    with path.open("rb") as handle:
        data = handle.read(MAX_EVALUATION_BYTES + 1)
    if len(data) > MAX_EVALUATION_BYTES:
        raise ValueError("Evaluation sidecar exceeds size limit")
    return validate_evaluation(json.loads(data, object_pairs_hook=_unique_object), checkpoint_path)


def checkpoint_eval_loss(run_dir: Path, name: str) -> float | None:
    """Prefer bound evaluation evidence; fall back only to exact-step legacy logs."""
    checkpoint = Path(run_dir) / name
    if not _CHECKPOINT.fullmatch(name):
        raise ValueError("Invalid evaluation checkpoint name")
    sidecar = read_evaluation(Path(run_dir) / "evaluations" / f"{name}.json", checkpoint)
    if sidecar is not None:
        return float(sidecar["metrics"]["eval_loss"])
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    step = int(name.split("-")[1])
    if (not isinstance(trainer, dict) or type(trainer.get("global_step")) is not int
            or trainer["global_step"] != step or not isinstance(trainer.get("log_history", []), list)):
        raise ValueError("Legacy evaluation trainer state does not match checkpoint")
    loss = None
    for row in trainer.get("log_history", []):
        if not isinstance(row, dict):
            raise ValueError("Malformed legacy evaluation history")
        if type(row.get("step")) is int and row["step"] == step and "eval_loss" in row:
            if not finite_number(row["eval_loss"]):
                raise ValueError("Legacy evaluation requires finite eval_loss")
            loss = min(loss, float(row["eval_loss"])) if loss is not None else float(row["eval_loss"])
    return loss
