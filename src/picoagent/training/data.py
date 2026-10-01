"""Fail-closed trace admission and content-addressed, read-only dataset snapshots.

A provenance claim is not independent proof of execution or clean pretraining. The
collector is responsible for receipts; this module verifies the declared contract,
exact input bytes, IDs, and cross-split duplicate messages before training.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

DATASET_SCHEMA = "picoagent.training.dataset.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record(record: dict[str, Any], expected_split: str, *, smoke_only: bool = False) -> None:
    if not isinstance(record, dict):
        raise ValueError("Each trace must be a JSON object")
    if expected_split not in {"train", "dev"} or record.get("split") != expected_split:
        raise ValueError("Only explicitly labeled train/dev records are accepted; lockbox/test is never admitted")
    for key in ("trace_id", "task_id", "family", "template_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"Trace requires nonempty {key}")
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    if smoke_only:
        if provenance.get("source") != "pipeline_smoke":
            raise ValueError("Smoke data must be labeled pipeline_smoke")
    elif (provenance.get("source") != "original_procedural"
          or provenance.get("execution") != "verified_environment"
          or record.get("status") != "success"
          or not isinstance(record.get("verification"), dict)
          or record["verification"].get("passed") is not True):
        raise ValueError("Training requires successful, verified original_procedural traces; authored/benchmark data is excluded")
    if not smoke_only:
        from picoagent.data.schema import validate_trace
        validate_trace(record)
        for key in ("raw_attempt_sha256", "task_sha256"):
            if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
                raise ValueError(f"{key} must be a SHA256 hex digest")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Trace requires a nonempty messages array")
    assistant_count = 0
    pending: set[str] = set()
    seen_call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid message role")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("Message content must be a string or null")
        role = message["role"]
        if role == "assistant":
            if pending:
                raise ValueError("Assistant followed unresolved tool calls")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be an array")
            if not calls and not content:
                raise ValueError("Assistant messages require content or tool calls")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise ValueError("Only canonical function tool calls are accepted")
                call_id = call.get("id")
                function = call.get("function", {})
                if not isinstance(call_id, str) or not call_id or call_id in seen_call_ids:
                    raise ValueError("Tool call IDs must be nonempty and unique")
                if not isinstance(function, dict) or function.get("name") not in {"bash", "python", "write_file", "search", "knowledge"}:
                    raise ValueError("Unknown tool; training and inference must use the same harness allowlist")
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    raise ValueError("Function arguments must be JSON-encoded strings")
                try:
                    decoded = json.loads(arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("Function arguments must be valid JSON") from exc
                if not isinstance(decoded, dict):
                    raise ValueError("Function arguments must encode an object")
                pending.add(call_id)
                seen_call_ids.add(call_id)
            assistant_count += 1
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("Orphan or repeated tool response")
            pending.remove(call_id)
        elif pending:
            raise ValueError("Unresolved tool calls before a user/system message")
    if pending or not assistant_count or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("Training traces must end with an assistant final answer after all tool responses")


def read_records(path: str | Path, expected_split: str, *, smoke_only: bool = False) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank line at {path}:{line_number}")
            try:
                record = json.loads(line)
                validate_record(record, expected_split, smoke_only=smoke_only)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"Empty {expected_split} dataset")
    return records


def conversation_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Normalize ephemeral tool IDs so renaming calls cannot evade duplicate checks."""
    normalized = json.loads(canonical_json(messages))
    call_ids: dict[str, str] = {}
    for message in normalized:
        for call in message.get("tool_calls", []) or []:
            call_ids[call["id"]] = f"call-{len(call_ids)}"
            call["id"] = call_ids[call["id"]]
        if message.get("role") == "tool":
            message["tool_call_id"] = call_ids.get(message["tool_call_id"], message["tool_call_id"])
    return sha256_bytes(canonical_json(normalized).encode())


def _check_disjoint(records: dict[str, list[dict[str, Any]]]) -> None:
    # Templates/families are split upstream. Reject collisions here as well so a
    # copied or mislabeled split cannot be silently used for development loss.
    for key in ("trace_id", "task_id", "template_id", "family"):
        seen: dict[str, str] = {}
        for split, rows in records.items():
            within: set[str] = set()
            for row in rows:
                value = row[key]
                if key in {"trace_id", "task_id"} and value in within:
                    raise ValueError(f"Duplicate {key} {value!r} within {split}")
                within.add(value)
                if value in seen and seen[value] != split:
                    raise ValueError(f"Cross-split {key} overlap: {value!r}")
                seen[value] = split
    fingerprints: dict[str, str] = {}
    for split, rows in records.items():
        for row in rows:
            fingerprint = conversation_fingerprint(row["messages"])
            if fingerprint in fingerprints:
                raise ValueError(f"Duplicate conversation in {fingerprints[fingerprint]} and {split}")
            fingerprints[fingerprint] = split


def prepare_dataset(train_path: str | Path, dev_path: str | Path, output_dir: str | Path, *, smoke_only: bool = False) -> Path:
    """Snapshot original bytes under exclusive creation; no in-place overwrites."""
    sources = {"train": Path(train_path), "dev": Path(dev_path)}
    # Read once, validate those same bytes, and write precisely the validated bytes.
    raw = {split: path.read_bytes() for split, path in sources.items()}
    records: dict[str, list[dict[str, Any]]] = {}
    for split, data in raw.items():
        rows = []
        for index, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"Blank line in {split}:{index}")
            row = json.loads(line)
            validate_record(row, split, smoke_only=smoke_only)
            rows.append(row)
        if not rows:
            raise ValueError(f"Empty {split} dataset")
        records[split] = rows
    _check_disjoint(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "smoke_only": smoke_only,
        "admission": "pipeline_smoke_only" if smoke_only else "successful_verified_original_procedural_only",
        "lockbox_used": False,
        "splits": {},
    }
    for split, data in raw.items():
        target = output / f"{split}.jsonl"
        with target.open("xb") as handle:
            handle.write(data)
        os.chmod(target, 0o444)
        manifest["splits"][split] = {
            "path": target.name, "sha256": sha256_bytes(data), "bytes": len(data),
            "records": len(records[split]),
            "source_path": str(sources[split].resolve()),
            "families": sorted({row["family"] for row in records[split]}),
            "templates": sorted({row["template_id"] for row in records[split]}),
        }
    path = output / "manifest.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(manifest) + "\n")
    os.chmod(path, 0o444)
    return path


def verify_dataset(manifest_path: str | Path, *, allow_smoke: bool = False) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("lockbox_used") is not False:
        raise ValueError("Invalid dataset manifest or lockbox admission")
    smoke = manifest.get("smoke_only")
    if not isinstance(smoke, bool) or (smoke and not allow_smoke):
        raise ValueError("Smoke datasets are forbidden in production training")
    if set(manifest.get("splits", {})) != {"train", "dev"}:
        raise ValueError("Manifest must contain exactly train and dev; never test/lockbox")
    records = {}
    for split, entry in manifest["splits"].items():
        target = (path.parent / entry["path"]).resolve()
        if target.parent != path.parent or target.name != f"{split}.jsonl":
            raise ValueError("Dataset paths must be sibling train.jsonl/dev.jsonl snapshots")
        raw = target.read_bytes()
        if sha256_bytes(raw) != entry.get("sha256") or len(raw) != entry.get("bytes"):
            raise ValueError(f"Dataset integrity check failed: {split}")
        rows = []
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                raise ValueError("Blank line in immutable dataset")
            row = json.loads(line)
            validate_record(row, split, smoke_only=smoke)
            rows.append(row)
        if not rows or len(rows) != entry.get("records"):
            raise ValueError(f"Dataset record count mismatch: {split}")
        if sorted({row["family"] for row in rows}) != entry.get("families") or sorted({row["template_id"] for row in rows}) != entry.get("templates"):
            raise ValueError(f"Dataset family/template metadata mismatch: {split}")
        records[split] = rows
    _check_disjoint(records)
    return manifest, records
