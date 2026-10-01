"""Small, dependency-free contracts for auditable original task data.

Validation checks evidence structure, not whether a remote claim is truthful.
Only the collector observing an actual isolated runtime may issue execution claims.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

SCHEMA_VERSION = "picoagent.data.v1"
SPLITS = ("train", "dev", "test")
EXECUTION_KINDS = ("authored_example", "unexecuted", "verified_environment")
STATUSES = ("success", "failed", "error", "unexecuted")


class DataValidationError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and str(path) != "." and not value.startswith("-") and not path.is_absolute() and ".." not in path.parts and "\\" not in value and "\x00" not in value


def validate_task(task: dict[str, Any]) -> None:
    _require(isinstance(task, dict), "task must be an object")
    _require(task.get("schema_version") == SCHEMA_VERSION, "unsupported task schema")
    for key in ("task_id", "family", "template_id", "prompt"):
        _require(isinstance(task.get(key), str) and bool(task[key]), f"missing task {key}")
    _require(task.get("split") in SPLITS, "invalid task split")
    _require(type(task.get("seed")) is int, "task seed must be an integer")
    source = task.get("provenance", {})
    _require(source.get("source") == "original_procedural", "only original procedural tasks are supported")
    _require(source.get("benchmark") is False, "benchmark tasks cannot enter this dataset")
    env = task.get("environment", {})
    _require(isinstance(env, dict), "environment must be an object")
    _require(isinstance(env.get("files", {}), dict), "files must be a mapping")
    for path, value in env.get("files", {}).items():
        _require(isinstance(path, str) and safe_relative_path(path), "unsafe task fixture path")
        _require(isinstance(value, str), "fixture content must be text")
    _require(isinstance(env.get("kv", {}), dict), "kv fixtures must be a mapping")
    _require(isinstance(env.get("docs", []), list), "docs fixtures must be a list")
    oracle = task.get("oracle", {})
    _require(oracle.get("kind") in {"json_exact", "text_exact", "svg"}, "unsupported oracle")
    _require("expected" in oracle, "oracle expected value is required")
    _require(task.get("input_sha256") == content_hash({"prompt": task["prompt"], "environment": env}), "task input hash mismatch")


def validate_messages(messages: list[dict[str, Any]], *, allow_incomplete: bool = False) -> None:
    _require(isinstance(messages, list) and bool(messages), "messages must be nonempty")
    pending: set[str] = set()
    used: set[str] = set()
    for index, message in enumerate(messages):
        _require(isinstance(message, dict), f"message {index} must be an object")
        role = message.get("role")
        _require(role in {"system", "user", "assistant", "tool"}, f"invalid message role at {index}")
        _require(isinstance(message.get("content"), str) or (role == "assistant" and message.get("content") is None), f"invalid content at {index}")
        calls = message.get("tool_calls", [])
        _require(isinstance(calls, list), "tool_calls must be a list")
        _require(not calls or role == "assistant", "only assistant can call tools")
        if role == "tool":
            call_id = message.get("tool_call_id")
            _require(call_id in pending, "tool response lacks a pending unique call")
            _require(not calls, "tool message cannot call tools")
            pending.remove(call_id)
        else:
            _require(not pending, "all pending tool calls must receive results before next non-tool message")
            _require("tool_call_id" not in message, "tool_call_id only belongs on tool messages")
        for call in calls:
            _require(isinstance(call, dict), "tool call must be an object")
            call_id = call.get("id")
            _require(isinstance(call_id, str) and bool(call_id) and call_id not in used, "tool call IDs must be unique")
            _require(call.get("type") == "function", "tool call type must be function")
            fn = call.get("function", {})
            _require(isinstance(fn.get("name"), str) and bool(fn["name"]), "missing tool function name")
            _require(isinstance(fn.get("arguments"), str), "tool arguments must be serialized JSON")
            try:
                args = json.loads(fn["arguments"])
            except (ValueError, TypeError) as exc:
                raise DataValidationError("invalid tool argument JSON") from exc
            _require(isinstance(args, dict), "tool arguments must decode to an object")
            pending.add(call_id)
            used.add(call_id)
    _require(allow_incomplete or not pending, "unanswered tool calls")


def validate_trace(trace: dict[str, Any]) -> None:
    _require(trace.get("schema_version") == SCHEMA_VERSION, "unsupported trace schema")
    for key in ("trace_id", "task_id", "family", "template_id"):
        _require(isinstance(trace.get(key), str) and bool(trace[key]), f"missing trace {key}")
    _require(trace.get("split") in SPLITS, "invalid trace split")
    status = trace.get("status")
    _require(status in STATUSES, "invalid trace status")
    provenance = trace.get("provenance", {})
    _require(provenance.get("source") == "original_procedural", "trace must have original provenance")
    _require(provenance.get("benchmark") is False, "benchmark trace rejected")
    execution = provenance.get("execution")
    _require(execution in EXECUTION_KINDS, "unknown trace execution kind")
    validate_messages(trace.get("messages", []), allow_incomplete=status in {"error", "failed", "unexecuted"})
    verification = trace.get("verification", {})
    _require(type(verification.get("passed")) is bool, "verification.passed must be a boolean")
    events = trace.get("tool_events", [])
    _require(isinstance(events, list), "tool_events must be a list")
    if execution != "verified_environment":
        _require(status != "success" and not verification["passed"], "unexecuted/authored traces cannot claim success")
        _require(not events, "authored traces cannot contain executed tool events")
    else:
        runtime = provenance.get("runtime", {})
        _require(runtime.get("backend") in {"docker", "podman"}, "verified trace requires isolated runtime backend")
        _require(isinstance(runtime.get("container_id"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", runtime["container_id"])), "verified trace requires a real 64-character container receipt")
        _require(bool(runtime.get("image")), "verified trace requires an image reference")
        for key in ("raw_attempt_sha256", "task_sha256"):
            _require(isinstance(trace.get(key), str) and bool(re.fullmatch(r"[0-9a-f]{64}", trace[key])), f"verified trace requires valid {key}")
        calls = {call["id"]: call["function"] for message in trace["messages"] for call in message.get("tool_calls", [])}
        replies = {message["tool_call_id"]: message for message in trace["messages"] if message["role"] == "tool"}
        _require(len(events) == len(replies), "every observed tool reply needs exactly one execution event")
        event_ids: set[str] = set()
        for event in events:
            _require(isinstance(event, dict), "tool event must be an object")
            call_id = event.get("tool_call_id")
            _require(call_id in calls and call_id in replies and call_id not in event_ids, "tool execution event has no matching unique call/reply")
            event_ids.add(call_id)
            _require(event.get("name") == calls[call_id]["name"], "tool execution name mismatch")
            _require(event.get("arguments") == calls[call_id]["arguments"], "tool execution arguments mismatch")
            _require(type(event.get("verified")) is bool, "tool event requires verified boolean")
            _require(canonical_json(event.get("result")) == replies[call_id]["content"], "tool execution result differs from observed reply")
            if event["verified"] and event["name"] in {"bash", "python", "write_file"}:
                receipt = event["result"]
                _require(receipt.get("backend") == "container" and receipt.get("runtime") in {"docker", "podman"}, "executed code needs a container runtime receipt")
                _require(isinstance(receipt.get("container_id"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", receipt["container_id"])), "executed code needs actual container ID")
                _require(bool(receipt.get("image")), "executed code needs image identity")
        if status == "success":
            _require(verification["passed"], "success trace requires passed oracle")
            _require(trace["messages"][-1].get("role") == "assistant" and not trace["messages"][-1].get("tool_calls"), "success trace needs final assistant response")
    _require(not verification["passed"] or status == "success", "passed oracle must match success status")


def training_eligible(trace: dict[str, Any]) -> bool:
    """Strict default: failures and illustrative answers remain archived only."""
    validate_trace(trace)
    return (trace["split"] == "train" and trace["status"] == "success"
            and trace["provenance"]["execution"] == "verified_environment"
            and trace["verification"]["passed"])
