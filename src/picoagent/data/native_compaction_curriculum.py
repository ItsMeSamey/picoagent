"""Observed native CPU pilot for reviewed, deterministic compaction teachers.

This is a narrowly gated producer of diagnostic observations, not a learner
runtime or an admission path. The only task command is a reviewed linked-record
``cat``. Every mode reuses the same base task identity. No model weights load.
"""
from __future__ import annotations

import argparse
import base64
import copy
from functools import lru_cache
import gzip
import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from picoagent.harness.agent import AgentHarness
from picoagent.harness.context import ContextManager
from picoagent.harness.protocol import render_messages
from picoagent.harness.tools import TOOL_SCHEMAS

from .audit import file_hash, write_new_json
from .compaction_curriculum import GOAL_MARKER, VisibleContextTeacher
from .compaction_curriculum import generate_compaction_task, visible_memory
from .generators import SYSTEM_PROMPT
from .oracles import check_task_result
from .native_storage import ShardWriter, metadata
from .schema import _validate_model_event_replay, canonical_json, content_hash, validate_task

SCHEMA = "picoagent.native_observation.v1"
SOURCE_ID = "native_compaction"
TEACHER_MODE = "reviewed_procedural_replay"
TOKENIZER_MODEL = "HuggingFaceTB/SmolLM2-360M"
TOKENIZER_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
MODES = ("full", "half", "manual")
SOURCE_PATHS = (
    "src/picoagent/data/native_compaction_curriculum.py",
    "src/picoagent/data/compaction_curriculum.py",
    "src/picoagent/data/generators.py", "src/picoagent/data/schema.py",
    "src/picoagent/data/oracles.py", "src/picoagent/data/audit.py",
    "src/picoagent/data/native_storage.py",
    "src/picoagent/harness/agent.py", "src/picoagent/harness/context.py",
    "src/picoagent/harness/protocol.py", "src/picoagent/harness/tools.py",
    "src/picoagent/harness/sandbox.py", "src/picoagent/harness/knowledge.py",
)
PILOT_BASES = (
    ("compaction.running_balance", 0), ("compaction.latest_status", 0),
    ("compaction.running_balance", 1), ("compaction.threshold_count", 0),
    ("compaction.first_status", 0), ("compaction.threshold_count", 1),
)


def independent_oracle_check(task: dict, final: str) -> dict:
    """Recompute from frozen linked files; never read oracle/reference answers."""
    goal = json.loads(task["prompt"].split(GOAL_MARKER, 1)[1])
    path, rows, visited = goal["initial_path"], [], set()
    files = task["environment"]["files"]
    while path is not None:
        if path in visited or path not in files:
            raise ValueError("broken or cyclic frozen record chain")
        visited.add(path)
        packet = json.loads(files[path])
        if packet["task_id"] != task["task_id"] or packet["index"] != len(rows):
            raise ValueError("frozen record identity/order mismatch")
        rows.append(packet["observation"])
        path = packet["next"]
    if len(rows) != goal["length"] or visited != set(files):
        raise ValueError("frozen chain length/coverage mismatch")
    operation = goal["operation"]
    if operation == "running_balance":
        value = goal["initial_balance"]
        for amount, _ in rows:
            value += amount
    elif operation == "threshold_count":
        value = 0
        for amount, _ in rows:
            if amount >= goal["threshold"]:
                value += 1
    elif operation in {"first_status", "latest_status"}:
        value = None
        iterator = rows if operation == "first_status" else reversed(rows)
        for amount, status in iterator:
            if amount >= goal["threshold"]:
                value = status
                break
    else:
        raise ValueError("native pilot does not execute test-family oracles")
    expected = {"result": value, "observations": len(rows)}
    try:
        passed = canonical_json(json.loads(final)) == canonical_json(expected)
    except (ValueError, TypeError):
        passed = False
    return {"passed": passed, "oracle": "native_compaction_fixture_recomputed_v1",
            "expected": expected, "independent_recomputation": True,
            "task_sha256": content_hash(task), "source_sha256": file_hash(__file__)}


class Journal:
    def __init__(self, path: Path):
        self.path, self.sequence, self.previous = path, 0, "0" * 64
        path.touch(exist_ok=False)

    def write(self, kind: str, payload: Any) -> None:
        row = {"sequence": self.sequence, "previous_sha256": self.previous,
               "kind": kind, "payload": payload}
        row["sha256"] = content_hash(row)
        data = (canonical_json(row) + "\n").encode("utf-8")
        with self.path.open("ab") as stream:
            if self.path.suffix == ".gz":
                # A complete member per event keeps every flushed receipt
                # independently readable after an interrupted later write.
                with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as member:
                    member.write(data)
            else:
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.previous, self.sequence = row["sha256"], self.sequence + 1


def _write_evidence(path: Path, value: dict, *, compressed: bool) -> Path:
    if not compressed:
        write_new_json(path, value)
        return path
    path = path.with_name(path.name + ".gz")
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6) as member:
            member.write((canonical_json(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _stream_fields(name: str, value: bytes) -> dict:
    return {name + "_b64": base64.b64encode(value).decode("ascii"),
            name + "_sha256": hashlib.sha256(value).hexdigest(),
            name + "_bytes": len(value)}


def _capture(argv: list[str], cwd: Path, env: dict, journal: Journal) -> dict:
    """Private reviewed-call capture; stdout/stderr are durable before checks."""
    journal.write("subprocess_requested", {"argv": argv, "cwd": str(cwd),
                                           "environment": env, **_stream_fields("stdin", b"")})
    start = time.monotonic_ns()
    started_at = time.time_ns()
    error = None
    try:
        process = subprocess.run(argv, cwd=cwd, env=env, input=b"", capture_output=True,
                                 timeout=10, check=False)
        out, err, code, timed_out = process.stdout, process.stderr, process.returncode, False
    except subprocess.TimeoutExpired as exc:
        out, err = exc.stdout or b"", exc.stderr or b""
        code, timed_out = 124, True
        error = {"type": type(exc).__name__, "message": str(exc)}
    except OSError as exc:
        out, err, code, timed_out = b"", b"", 125, False
        error = {"type": type(exc).__name__, "message": str(exc), "process_started": False}
    receipt = {"execution_kind": "subprocess", "argv": argv, "cwd": str(cwd),
               "stdin": "", "environment": env, **_stream_fields("stdin", b""),
               **_stream_fields("stdout", out), **_stream_fields("stderr", err),
               "exit_code": code, "timed_out": timed_out, "truncated": False,
               "duration_seconds": (time.monotonic_ns() - start) / 1e9,
               "started_at_unix_ns": started_at, "capture_error": error}
    journal.write("subprocess_observed", receipt)
    return receipt


def _fixture_hashes(workspace: Path) -> dict:
    return {str(path.relative_to(workspace)): file_hash(path)
            for path in sorted(workspace.rglob("*")) if path.is_file()}


class ReviewedCatRegistry:
    """No generic dispatch: only the expected observed chain's next read."""
    schemas = TOOL_SCHEMAS

    def __init__(self, task: dict, workspace: Path, journal: Journal, executables: dict):
        self.task, self.workspace, self.journal = task, workspace, journal
        self.executables = executables
        self.next_path = json.loads(task["prompt"].split(GOAL_MARKER, 1)[1])["initial_path"]
        self.receipts: list[dict] = []
        self.pending_call = None

    def dispatch(self, name: str, arguments: str) -> dict:
        requested = {"name": name, "arguments": arguments, "call": self.pending_call}
        self.journal.write("tool_requested", requested)
        try:
            args = json.loads(arguments)
            expected = "cat -- " + str(self.next_path)
            if (name != "bash" or set(args) != {"command"} or args["command"] != expected
                    or re.fullmatch(r"cat -- records/[0-9a-f]{16}\.json", args["command"]) is None):
                raise ValueError("native producer accepts only the next exact reviewed cat command")
            if self.pending_call != {"id": f"record_{len(self.receipts)}", "type": "function",
                                     "function": {"name": name, "arguments": arguments}}:
                raise ValueError("dispatch differs from the fixed teacher's recorded call")
            file = self.workspace / self.next_path
            if file.is_symlink() or file.resolve().parent != (self.workspace / "records").resolve():
                raise ValueError("record path leaves its frozen fixture directory")
            if file_hash(file) != hashlib.sha256(self.task["environment"]["files"][self.next_path].encode()).hexdigest():
                raise ValueError("fixture changed before its observed read")
            for executable in ("bash", "cat"):
                if file_hash(self.executables[executable]["path"]) != self.executables[executable]["sha256"]:
                    raise ValueError("reviewed executable changed")
        except Exception as exc:
            self.journal.write("tool_rejected", {**requested, "type": type(exc).__name__, "error": str(exc)})
            raise
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.workspace),
               "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
        capture = _capture([self.executables["bash"]["path"], "--noprofile", "--norc", "-c", expected],
                           self.workspace, env, self.journal)
        result = {"stdout": base64.b64decode(capture["stdout_b64"]).decode("utf-8", errors="replace"),
                  "stderr": base64.b64decode(capture["stderr_b64"]).decode("utf-8", errors="replace"),
                  "exit_code": capture["exit_code"], "timed_out": capture["timed_out"],
                  "truncated": False, "backend": "native_teacher", "runtime": "reviewed_native_cat",
                  "execution": "native_teacher_observed"}
        if capture["capture_error"] is not None:
            result["error"] = capture["capture_error"]["type"]
        receipt = {"sequence": len(self.receipts), "tool_call_id": self.pending_call["id"],
                   "name": name, "arguments": arguments, "result": result, **capture}
        self.receipts.append(receipt)
        self.journal.write("tool_observed", receipt)
        if capture["exit_code"] == 0 and not capture["timed_out"]:
            packet = json.loads(result["stdout"])
            self.next_path = packet["next"]
        return result


def _snapshot(output: Path, tokenizer_path: Path) -> dict:
    root = Path(__file__).resolve().parents[3]
    snapshot = output / "source_snapshot"
    source_hashes, files = {}, {}
    for name in SOURCE_PATHS:
        contents = (root / name).read_bytes()
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(contents)
        source_hashes[name] = hashlib.sha256(contents).hexdigest()
        files[name] = {"sha256": source_hashes[name], "bytes": len(contents)}
        destination.chmod(0o444)
    tokenizer_files = {}
    for source in sorted(tokenizer_path.iterdir()):
        if source.name not in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                               "vocab.json", "merges.txt", "config.json"}:
            continue
        name = "tokenizer/" + source.name
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
        files[name] = {"sha256": file_hash(destination), "bytes": destination.stat().st_size}
        tokenizer_files[source.name] = files[name]
        destination.chmod(0o444)
    write_new_json(snapshot / "tool_protocol.json", {"tools": TOOL_SCHEMAS,
                                                    "sha256": content_hash(TOOL_SCHEMAS)})
    files["tool_protocol.json"] = {"sha256": file_hash(snapshot / "tool_protocol.json"),
                                   "bytes": (snapshot / "tool_protocol.json").stat().st_size}
    manifest = {"schema": "picoagent.native_compaction.source_snapshot.v1", "files": files,
                "source_sha256": source_hashes, "tokenizer": {"model": TOKENIZER_MODEL,
                    "revision": TOKENIZER_REVISION, "files": tokenizer_files},
                "created_before_execution": True}
    write_new_json(snapshot / "manifest.json", manifest)
    return manifest


def _runtime(output: Path, journal: Journal) -> dict:
    executables = {}
    for name in ("bash", "cat"):
        path = Path(shutil.which(name, path="/usr/bin:/bin") or "")
        if not path.is_file():
            raise RuntimeError("reviewed executable is unavailable: " + name)
        probe = _capture([str(path), "--version"], output,
                         {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}, journal)
        executables[name] = {"path": str(path), "resolved_path": str(path.resolve()),
                             "sha256": file_hash(path), "version_probe": probe,
                             "version": base64.b64decode(probe["stdout_b64"]).decode().splitlines()[0]}
        if probe["exit_code"] != 0:
            raise RuntimeError("reviewed executable version probe failed")
    executables["python"] = {"path": sys.executable, "resolved_path": str(Path(sys.executable).resolve()),
                             "sha256": file_hash(sys.executable), "version": platform.python_version()}
    return {"backend": "native_teacher", "python_version": platform.python_version(),
            "platform": platform.platform(), "locale": locale.setlocale(locale.LC_ALL, None),
            "executables": executables, "environment_policy": "fixed PATH; fixture-local HOME; C.UTF-8; no network commands",
            "container_semantic_replay": "not_verified", "outer_exec_receipt_id": None,
            "arbitrary_learner_execution_allowed": False}


def _candidate(task: dict, source_hashes: dict) -> dict:
    return {"task_id": task["task_id"], "base_task_id": task["task_id"], "execution": "unexecuted",
            "teacher_mode": TEACHER_MODE, "teacher_decision_mode": TEACHER_MODE,
            "teacher_model": None, "teacher": "VisibleContextTeacher",
            "source_sha256": source_hashes, "mode_variants": list(MODES),
            "command_policy": "one exact cat -- records/<16 lowercase hex>.json per call, following only the observed next path",
            "arbitrary_learner_execution_allowed": False, "private_oracle_access": False}


def _run_one(task: dict, mode: str, output: Path, snapshot: dict, runtime: dict, tokenizer,
             candidate: dict, *, compressed: bool = False) -> dict:
    attempt = output / "attempts" / (content_hash({"task": task["task_id"], "mode": mode})[:20])
    attempt.mkdir(parents=True, exist_ok=False)
    journal = Journal(attempt / ("journal.jsonl.gz" if compressed else "journal.jsonl"))
    journal.write("attempt_started", {"task_id": task["task_id"], "base_task_id": task["task_id"],
                                      "mode": mode, "task_sha256": content_hash(task),
                                      "candidate_sha256": content_hash(candidate)})
    workspace = attempt / "workspace"
    workspace.mkdir()
    for relative, text in task["environment"]["files"].items():
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(text)
        destination.chmod(0o444)
    before = _fixture_hashes(workspace)
    journal.write("fixture_before", before)
    registry = ReviewedCatRegistry(task, workspace.resolve(), journal, runtime["executables"])
    teacher = VisibleContextTeacher()

    @lru_cache(maxsize=4096)
    def count_text(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count(messages, tools=TOOL_SCHEMAS):
        return count_text(render_messages(messages, tools, add_generation_prompt=True))

    def model(messages, tools):
        journal.write("teacher_request", {"messages": messages, "tools": tools})
        try:
            response = teacher(messages, tools)
        except Exception as exc:
            journal.write("teacher_exception", {"type": type(exc).__name__, "message": str(exc)})
            raise
        journal.write("teacher_response", response)
        if response.get("tool_calls"):
            if len(response["tool_calls"]) != 1:
                raise ValueError("fixed teacher must make exactly one read per step")
            registry.pending_call = copy.deepcopy(response["tool_calls"][0])
        return response

    context = ContextManager(model, mode=mode, max_tokens=4096, reserve_tokens=768,
                              token_counter=count, request_token_counter=lambda rows: count(rows, []))
    harness = AgentHarness(model, registry, context=context, max_steps=task["compaction"]["horizon"] + 1,
                           trace_path=None if compressed else attempt / "harness.jsonl", system_prompt=SYSTEM_PROMPT)
    error = None
    try:
        result = harness.run(task["prompt"])
        events = result.events
        effective, final, stop = result.messages, result.final, result.stop_reason
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        journal.write("attempt_exception", error)
        events = [json.loads(line) for line in (attempt / "harness.jsonl").read_text().splitlines()] if (attempt / "harness.jsonl").exists() else []
        effective, final, stop = [], "", "exception"
    after = _fixture_hashes(workspace)
    journal.write("fixture_after", after)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]
    for event in events:
        if event["type"] == "assistant":
            messages.append(event["message"])
        elif event["type"] == "tool_execution":
            messages.append({"role": "tool", "name": event["name"], "tool_call_id": event["tool_call_id"],
                             "content": canonical_json(event["result"])})
    raw = {"schema": SCHEMA, "source_id": SOURCE_ID, "execution_kind": "native_teacher_observed",
           "sft_admissible": False, "task": task, "task_id": task["task_id"], "base_task_id": task["task_id"],
           "task_sha256": content_hash(task), "candidate": candidate, "candidate_sha256": content_hash(candidate),
           "mode": mode, "source_sha256": snapshot["source_sha256"], "source_module_sha256": file_hash(__file__),
           "source_snapshot_manifest_sha256": file_hash(output / "source_snapshot/manifest.json"),
           "teacher_mode": TEACHER_MODE, "teacher_decision_mode": TEACHER_MODE,
           "teacher_model": None, "model_identity": None, "provider_generation": None,
           "runtime": runtime, "tools": TOOL_SCHEMAS, "tool_schemas_sha256": content_hash(TOOL_SCHEMAS),
           "messages": messages, "effective_messages": effective,
           "model_events": [event for event in events if event["type"] in {"assistant", "compaction", "compaction_error"}],
           "tool_events": [event for event in events if event["type"] == "tool_execution"],
           "receipts": registry.receipts, "final": final, "stop_reason": stop, "error": error,
           "fixture_sha256_before": before, "fixture_sha256_after": after, "fixture_unchanged": before == after,
           "artifacts": {}, "kv": {}, "context_budget": {"max_tokens": 4096, "reserve_tokens": 768,
               "summary_tokens": context.summary_tokens, "compaction_headroom_tokens": context.compaction_headroom_tokens,
               "trigger_budget": context.trigger_budget}, "tokenizer": snapshot["tokenizer"],
           "attempt_path": str(attempt.relative_to(output)), "all_harness_events": events}
    # Freeze all observations before any semantic/replay assertion can reject them.
    captured_path = _write_evidence(attempt / "raw_before_validation.json", raw, compressed=compressed)
    journal.write("raw_captured", {"sha256": file_hash(captured_path),
                                   "path": str(captured_path.relative_to(output))})
    validation_errors = []
    independent = independent_oracle_check(task, final)
    shared = check_task_result(task, final)
    try:
        _validate_model_event_replay(raw)
        if independent["passed"] != shared["passed"]:
            raise ValueError("independent fixture oracle disagrees with task oracle")
        if before != after:
            raise ValueError("read-only fixture changed")
        if not any(event["type"] == "compaction" and event["accepted"] for event in raw["model_events"]):
            raise ValueError("pilot did not actually compact")
        for event in raw["model_events"]:
            if event["type"] == "compaction":
                if visible_memory(event["before_messages"]) != visible_memory(event["result_messages"]):
                    raise ValueError("compaction lost or invented observed essentials")
                if count(event["summary_request"], []) > 4096 - 768:
                    raise ValueError("compactor request exceeds actual reserved context")
                if count(event["summary_request"] + [event["summary_response"]], []) > 4096:
                    raise ValueError("complete compactor example exceeds context")
            elif event["type"] == "assistant" and count(event["input_messages"]) > 4096 - 768:
                raise ValueError("ordinary action input exceeds reserved context")
    except Exception as exc:
        validation_errors.append({"type": type(exc).__name__, "message": str(exc)})
        journal.write("validation_rejected", validation_errors[-1])
    raw.update(independent_oracle=independent, shared_oracle=shared, validation_errors=validation_errors,
               status="observed_success" if stop == "final" and independent["passed"] and not validation_errors else "observed_failure")
    journal.write("attempt_completed", {"status": raw["status"], "independent_oracle": independent,
                                        "validation_errors": validation_errors})
    raw["journal"] = {"path": str(journal.path.relative_to(output)), "event_count": journal.sequence,
                       "chain_sha256": journal.previous, "file_sha256": file_hash(journal.path)}
    _write_evidence(attempt / "observation.json", raw, compressed=compressed)
    return raw


def scaling_bases() -> tuple[tuple[str, int], ...]:
    """Fixed authorized expansion: 192 train + 24 dev, no test instances."""
    return tuple((family, seed) for family, count in (
        ("compaction.running_balance", 96), ("compaction.latest_status", 96),
        ("compaction.threshold_count", 12), ("compaction.first_status", 12))
        for seed in range(count))


def run_native_compaction_pilot(output_dir: str | Path, *, tokenizer_path: str | Path,
                               scale: bool = False) -> dict:
    """Fixed original task sets only; no caller task, model or command injection.

    `scale=True` collects 216 bases/648 variants in bounded gzip shards. Receipt
    journals and pre-validation raw evidence are compressed at first creation;
    no original is deleted. Full harness events remain in every raw envelope.
    """
    if type(scale) is not bool:
        raise ValueError("scale must be a boolean selecting the fixed reviewed curriculum")
    bases = scaling_bases() if scale else PILOT_BASES
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "batch_journal.jsonl")
    journal.write("batch_started", {"source_id": SOURCE_ID, "bases": bases, "modes": MODES,
                                    "compressed": scale})
    try:
        snapshot = _snapshot(output, Path(tokenizer_path))
        journal.write("sources_frozen", {"manifest_sha256": file_hash(output / "source_snapshot/manifest.json")})
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(output / "source_snapshot/tokenizer"),
                                                  local_files_only=True, trust_remote_code=False)
        tasks = [generate_compaction_task(family, seed) for family, seed in bases]
        candidates = [_candidate(task, snapshot["source_sha256"]) for task in tasks]
        for task in tasks:
            validate_task(task)
            if task["split"] not in {"train", "dev"}:
                raise ValueError("test tasks cannot enter the native pilot")
        sidecar_paths = {}
        for kind, rows in (("tasks", tasks), ("candidates", candidates)):
            if scale:
                writer = ShardWriter(output, kind)
                try:
                    for row in rows:
                        writer.add(row)
                finally:
                    writer.close()
                sidecar_paths[kind] = writer.paths
            else:
                filename = kind + ".jsonl"
                with (output / filename).open("x", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(canonical_json(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                sidecar_paths[kind] = [filename]
        runtime = _runtime(output, journal)
        write_new_json(output / "runtime.json", runtime)
        summaries = []
        writer = ShardWriter(output, "observations") if scale else None
        stream = None if scale else (output / "observations.jsonl").open("x", encoding="utf-8")
        try:
            for index, (task, candidate) in enumerate(zip(tasks, candidates)):
                # Scale balances each mode within each family independently.
                offset = task["seed"] % 3 if scale else index % 3
                for mode in MODES[offset:] + MODES[:offset]:
                    raw = _run_one(task, mode, output, snapshot, runtime, tokenizer, candidate, compressed=scale)
                    if writer is not None:
                        writer.add(raw)
                    else:
                        stream.write(canonical_json(raw) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    summaries.append({"status": raw["status"], "mode": mode,
                                      "receipts": len(raw["receipts"]),
                                      "compactions": sum(event["type"] == "compaction" and event["accepted"]
                                                         for event in raw["model_events"])})
                    journal.write("variant_preserved", {"task_id": task["task_id"], "mode": mode,
                                                         "observations": len(summaries), **summaries[-1]})
        finally:
            if writer is not None:
                writer.close()
            if stream is not None:
                stream.close()
        observation_paths = writer.paths if scale else ["observations.jsonl"]
        report = {"schema": "picoagent.native_compaction.pilot.v1", "execution_kind": "native_teacher_observed",
                  "sft_admissible": False, "unique_base_tasks": len(tasks), "mode_variant_observations": len(summaries),
                  "unique_bases_by_split": {split: sum(task["split"] == split for task in tasks) for split in ("train", "dev")},
                  "passed": sum(row["status"] == "observed_success" for row in summaries),
                  "failed": sum(row["status"] != "observed_success" for row in summaries),
                  "actual_cat_subprocesses": sum(row["receipts"] for row in summaries),
                  "compactions_by_mode": {mode: sum(row["compactions"] for row in summaries if row["mode"] == mode) for mode in MODES},
                  "observation_paths": observation_paths, **sidecar_paths,
                  "observation_files": {name: metadata(output / name) for name in observation_paths},
                  "storage": "gzip_sharded_v1" if scale else "plain_jsonl",
                  "source_snapshot_sha256": file_hash(output / "source_snapshot/manifest.json"),
                  "limitations": ["Native reviewed CPU teacher observations; container parity is not verified.",
                      "Deterministic visible-context teacher; no adaptive model inference or learner capability claim.",
                      "Mode variants reuse base task IDs and are not extra unique tasks.",
                      "Separate independently reviewed native admission is required before SFT."]}
        journal.write("batch_completed", report)
        write_new_json(output / "manifest.json", report)
        return report
    except Exception as exc:
        journal.write("batch_failed", {"type": type(exc).__name__, "message": str(exc)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--scale", action="store_true", help="Collect the fixed 216-base train/dev expansion into gzip shards")
    args = parser.parse_args(argv)
    print(canonical_json(run_native_compaction_pilot(args.output_dir, tokenizer_path=args.tokenizer_path, scale=args.scale)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
