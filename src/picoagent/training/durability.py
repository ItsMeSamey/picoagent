"""Fail-closed rendezvous with an off-runtime public-backup controller.

Acknowledgements are integrity evidence from the trusted controller, not a
cryptographic authentication mechanism. Never expose the run directory to
untrusted writers. No GitHub credentials are needed on the training runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

from .data import canonical_json, sha256_file
from .provenance import verify_checkpoint, write_json

SCHEMA = "picoagent.durable-ack.v1"
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
CHECKPOINT = re.compile(r"checkpoint-[0-9]+\Z")


def acknowledgement_from_receipt(receipt: dict) -> dict:
    """Accept only a published, fully read-back GitHub release receipt."""
    if (receipt.get("schema") != "picoagent.github-release-receipt.v1"
            or receipt.get("published") is not True
            or receipt.get("independent_readback_verified") is not True):
        raise ValueError("A verified published release receipt is required")
    identity = receipt.get("identity", {})
    if (identity.get("visibility") != "public"
            or not CHECKPOINT.fullmatch(str(identity.get("checkpoint")))
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", str(identity.get("repository")))
            or type(receipt.get("release_id")) is not int or receipt["release_id"] <= 0
            or not isinstance(receipt.get("tag"), str) or not receipt["tag"]):
        raise ValueError("Invalid public release identity")
    for key in ("run_manifest_sha256", "checkpoint_manifest_sha256", "transfer_manifest_sha256",
                "source_tree_sha256"):
        if not DIGEST.fullmatch(str(identity.get(key))):
            raise ValueError("Invalid release identity digest")
    if not DIGEST.fullmatch(str(receipt.get("plan_sha256"))):
        raise ValueError("Invalid release plan digest")
    assets = receipt.get("assets")
    if not isinstance(assets, dict) or not assets or "transfer_manifest.json" not in assets:
        raise ValueError("Missing release asset verification")
    for name, record in assets.items():
        if (not isinstance(record, dict) or not DIGEST.fullmatch(str(record.get("sha256")))
                or type(record.get("bytes")) is not int or record["bytes"] <= 0
                or type(record.get("id")) is not int or record["id"] <= 0
                or (name != "transfer_manifest.json" and name != f"chunk-{record['sha256']}.bin")):
            raise ValueError("Invalid verified release asset")
    if assets["transfer_manifest.json"]["sha256"] != identity["transfer_manifest_sha256"]:
        raise ValueError("Release manifest digest mismatch")
    return {"schema": SCHEMA, "receipt": receipt,
            "receipt_sha256": hashlib.sha256(canonical_json(receipt).encode()).hexdigest()}


def wait_for_durable_ack(output: Path, checkpoint: Path, timeout_seconds: float,
                         *, poll_seconds: float = 1.0) -> dict:
    """Do not continue training/evaluation on missing, partial or stale evidence.

    Timeout leaves the sealed checkpoint intact and fails the training process.
    A resumed process must call this again before taking another optimizer step.
    """
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Durability timeout must be positive finite seconds")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("Durability polling interval must be positive")
    if checkpoint.parent != output or not CHECKPOINT.fullmatch(checkpoint.name):
        raise ValueError("Durability checkpoint must belong to the run")
    run_hash = sha256_file(output / "run_manifest.json")
    checkpoint_hash = sha256_file(checkpoint / "checkpoint_manifest.json")
    verify_checkpoint(checkpoint, run_hash)
    directory = output / "durability"
    if directory.is_symlink():
        raise ValueError("Durability directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    path = directory / f"{checkpoint.name}.json"
    status = directory / "status.json"
    write_json(status, {"status": "waiting", "checkpoint": checkpoint.name,
                        "checkpoint_manifest_sha256": checkpoint_hash})
    deadline = time.monotonic() + timeout_seconds
    while True:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024**2:
                raise ValueError("Unsafe durability acknowledgement")
            ack = json.loads(path.read_text())
            expected = acknowledgement_from_receipt(ack.get("receipt", {}))
            if ack != expected:
                raise ValueError("Durability acknowledgement integrity mismatch")
            identity = ack["receipt"]["identity"]
            if (identity["checkpoint"] != checkpoint.name
                    or identity["run_manifest_sha256"] != run_hash
                    or identity["checkpoint_manifest_sha256"] != checkpoint_hash):
                raise ValueError("Durability acknowledgement belongs to another run or checkpoint")
            write_json(status, {"status": "verified", "checkpoint": checkpoint.name,
                                "receipt_sha256": ack["receipt_sha256"]})
            return ack
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            write_json(status, {"status": "timed_out", "checkpoint": checkpoint.name})
            raise TimeoutError(f"No verified public backup acknowledgement for {checkpoint.name}; training stopped")
        time.sleep(min(poll_seconds, remaining))
