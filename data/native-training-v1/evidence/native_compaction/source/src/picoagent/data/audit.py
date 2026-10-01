"""Deterministic manifests, split leakage checks, and append-only attempt archives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Iterable

from .schema import canonical_json, content_hash, validate_task, validate_trace


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path: str | Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank record at {path}:{line_no}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"non-object record at {path}:{line_no}")
        rows.append(row)
    return rows


def audit_tasks(tasks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(tasks)
    seen: dict[str, dict[str, str]] = {name: {} for name in ("family", "template_id", "input_sha256", "task_id")}
    counts = {split: 0 for split in ("train", "dev", "test")}
    for task in rows:
        validate_task(task)
        split = task["split"]
        for field, values in seen.items():
            value = task[field]
            if value in values and (values[value] != split or field in {"task_id", "input_sha256"}):
                raise ValueError(f"duplicate/cross-split {field}: {value}")
            values[value] = split
        counts[split] += 1
    return {"passed": True, "counts": counts, "families": {split: sorted(k for k, v in seen["family"].items() if v == split) for split in counts},
            "checks": ["unique_task_ids", "unique_inputs", "disjoint_families", "disjoint_templates", "original_provenance"]}


def write_curriculum(output_dir: str | Path, tasks: list[dict[str, Any]], *, configuration: dict[str, Any], split_policy: dict[str, str] | None = None) -> Path:
    from .generators import GENERATOR_VERSION, SPLIT_POLICY, authored_example
    report = audit_tasks(tasks)
    policy = SPLIT_POLICY if split_policy is None else split_policy
    if any(policy.get(task["family"]) != task["split"] for task in tasks):
        raise ValueError("tasks do not match supplied split policy")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        rows = [task for task in tasks if task["split"] == split]
        for kind, records in (("tasks", rows), ("authored", [authored_example(task) for task in rows])):
            path = output / f"{split}.{kind}.jsonl"
            with path.open("x", encoding="utf-8") as handle:
                for row in records:
                    handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size, "records": len(records), "kind": kind, "split": split}
    source_files = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"schema": "picoagent.curriculum.manifest.v1", "generator_version": GENERATOR_VERSION,
                "configuration": configuration, "split_policy": policy, "split_policy_sha256": content_hash(policy),
                "files": files, "source_sha256": source_files, "audit": report,
                "execution": "unexecuted", "verified_trace_count": 0,
                "limitations": ["Authored references are not observed tool outputs or model rollouts.",
                                "Published held-out families assess original curriculum generalization, not private benchmark performance.",
                                "This manifest cannot establish the contents of a base model's pretraining corpus."]}
    write_new_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def verify_curriculum(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "picoagent.curriculum.manifest.v1":
        raise ValueError("unsupported curriculum manifest")
    tasks, authored = [], []
    for name, entry in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("manifest paths must be sibling filenames")
        target = path.parent / name
        if file_hash(target) != entry["sha256"] or target.stat().st_size != entry["bytes"]:
            raise ValueError(f"dataset hash mismatch: {name}")
        records = read_jsonl(target)
        if len(records) != entry["records"] or any(record.get("split") != entry["split"] for record in records):
            raise ValueError(f"dataset count/split mismatch: {name}")
        (tasks if entry["kind"] == "tasks" else authored).extend(records)
    report = audit_tasks(tasks)
    by_id = {task["task_id"]: task for task in tasks}
    for trace in authored:
        validate_trace(trace)
        if trace["task_sha256"] != content_hash(by_id[trace["task_id"]]):
            raise ValueError("authored example does not match task hash")
    if manifest.get("split_policy_sha256") != content_hash(manifest["split_policy"]):
        raise ValueError("split policy hash mismatch")
    for task in tasks:
        if manifest["split_policy"].get(task["family"]) != task["split"]:
            raise ValueError("task violates declared family split policy")
    return report


class AttemptArchive:
    """An exclusive directory per attempt, including failures and interrupted runs.

    No method deletes/overwrites attempts. Existing archive roots are append-only;
    each attempt has a random opaque ID so retries retain earlier evidence. Event
    chaining detects accidental alteration (it is not an adversarial signature).
    """
    def __init__(self, root: str | Path, task: dict[str, Any], *, teacher: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempt_id = uuid.uuid4().hex
        self.path = self.root / self.attempt_id
        self.path.mkdir(exist_ok=False)
        self._sequence = 0
        self._previous = "0" * 64
        self._finished = False
        write_new_json(self.path / "task.json", task)
        write_new_json(self.path / "request.json", {"attempt_id": self.attempt_id, "task_sha256": content_hash(task), "teacher": teacher})
        self.event("attempt_started", {"task_id": task["task_id"], "teacher": teacher})

    def event(self, kind: str, payload: Any) -> None:
        if self._finished:
            raise RuntimeError("attempt is already finalized")
        row = {"sequence": self._sequence, "kind": kind, "payload": payload, "previous_sha256": self._previous}
        row["sha256"] = content_hash(row)
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._sequence += 1
        self._previous = row["sha256"]

    def finalize(self, raw: dict[str, Any], trace: dict[str, Any]) -> Path:
        self.event("attempt_finished", {"status": trace["status"]})
        write_new_json(self.path / "raw.json", raw)
        trace["raw_attempt_sha256"] = file_hash(self.path / "raw.json")
        # Archive the raw outcome before validation. A malformed trace does not
        # erase provider output, failed tools, model errors, or partial evidence.
        try:
            validate_trace(trace)
        except (ValueError, TypeError, KeyError) as exc:
            write_new_json(self.path / "invalid_trace.json", {"trace": trace, "validation_error": str(exc)})
            self._seal()
            raise
        write_new_json(self.path / "trace.json", trace)
        self._seal()
        return self.path / "trace.json"

    def _seal(self) -> None:
        entries = {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                   for p in sorted(self.path.iterdir()) if p.is_file() and p.name != "manifest.json"}
        write_new_json(self.path / "manifest.json", {"schema": "picoagent.attempt.manifest.v1", "attempt_id": self.attempt_id,
                                                    "event_count": self._sequence, "event_chain_sha256": self._previous, "files": entries})
        self._finished = True


def verify_attempt(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("invalid attempt manifest path")
        target = directory / name
        if file_hash(target) != info["sha256"] or target.stat().st_size != info["bytes"]:
            raise ValueError(f"attempt integrity failure: {name}")
    previous = "0" * 64
    events = read_jsonl(directory / "events.jsonl")
    for i, event in enumerate(events):
        claimed = event.pop("sha256")
        if event["sequence"] != i or event["previous_sha256"] != previous or content_hash(event) != claimed:
            raise ValueError("attempt event chain mismatch")
        previous = claimed
    if len(events) != manifest["event_count"] or previous != manifest["event_chain_sha256"]:
        raise ValueError("attempt event chain length mismatch")
    if (directory / "trace.json").exists():
        trace = json.loads((directory / "trace.json").read_text(encoding="utf-8"))
        validate_trace(trace)
        if trace["raw_attempt_sha256"] != file_hash(directory / "raw.json"):
            raise ValueError("trace raw evidence mismatch")
        task = json.loads((directory / "task.json").read_text(encoding="utf-8"))
        if trace["task_sha256"] != content_hash(task):
            raise ValueError("trace task evidence mismatch")
        for key in ("task_id", "family", "template_id", "split"):
            if trace[key] != task[key]:
                raise ValueError("trace task identity mismatch")
        raw = json.loads((directory / "raw.json").read_text(encoding="utf-8"))
        if "events" in raw:
            if trace.get("tool_events", []) != [event for event in raw["events"] if event.get("type") == "tool_execution"]:
                raise ValueError("trace tool events differ from raw execution")
            if trace.get("model_events", []) != [event for event in raw["events"] if event.get("type") in {"assistant", "compaction"}]:
                raise ValueError("trace model events differ from raw execution")
    return {"passed": True, "attempt_id": manifest["attempt_id"], "files": len(manifest["files"])}
