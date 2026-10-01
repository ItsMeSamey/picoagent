"""Capture reviewed in-process knowledge/search tool observations.

This adapter uses AgentHarness and ToolRegistry.dispatch, with a per-task
KnowledgeStore file and LocalCorpusSearch over original fixtures. It has no
subprocess execution path, no external search client, and no test split.
"""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import json
import locale
import platform
from pathlib import Path
import sys
import time

from picoagent.data.collector import LocalCorpusSearch
from picoagent.data.luna_knowledge_search_curriculum import (
    FAMILY_SPLIT_POLICY,
    KnowledgeSearchTeacher,
    authored_unexecuted_candidate,
    generate_knowledge_search_task,
    independent_knowledge_search_oracle,
)
from picoagent.data.oracles import check_task_result
from picoagent.data.schema import canonical_json, content_hash, validate_task
from picoagent.harness.agent import AgentHarness, DEFAULT_SYSTEM_PROMPT
from picoagent.harness.knowledge import KnowledgeStore
from picoagent.harness.tools import ToolRegistry


REPO = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "src/picoagent/data/luna_knowledge_search_curriculum.py",
    "data/luna-knowledge-search-v1/record_native.py",
    "src/picoagent/data/collector.py",
    "src/picoagent/data/generators.py",
    "src/picoagent/data/oracles.py",
    "src/picoagent/data/schema.py",
    "src/picoagent/harness/agent.py",
    "src/picoagent/harness/protocol.py",
    "src/picoagent/harness/tools.py",
    "src/picoagent/harness/knowledge.py",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _append_fsync(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((canonical_json(value) + "\n").encode("utf-8"))
        handle.flush()
        import os
        os.fsync(handle.fileno())


def _file_snapshot(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    raw = path.read_bytes()
    return {"exists": True, "content_b64": base64.b64encode(raw).decode("ascii"),
            "sha256": _sha(raw), "bytes": len(raw)}


def _runtime() -> dict:
    executable = Path(sys.executable).resolve()
    return {
        "backend": "native_teacher",
        "python_version": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": _file_sha(executable),
        "platform": platform.platform(),
        "locale": str(locale.getlocale()),
        "environment_policy": "in-process only; no subprocess, no network search client",
    }


class ObservedNativeToolRegistry(ToolRegistry):
    """Real ToolRegistry dispatch with fsynced host-function receipts."""

    def __init__(self, knowledge: KnowledgeStore, local_search: LocalCorpusSearch,
                 corpus: list[dict[str, str]], *, attempt_dir: Path, global_events: Path,
                 attempt_events: Path):
        super().__init__(backend=None, knowledge=knowledge, search_client=local_search)
        self.corpus = copy.deepcopy(corpus)
        self.attempt_dir = attempt_dir
        self.global_events = global_events
        self.attempt_events = attempt_events
        self.receipts: list[dict] = []
        self.pending_call_id: str | None = None

    def journal(self, event: dict) -> None:
        row = {"observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **event}
        _append_fsync(self.attempt_events, row)
        _append_fsync(self.global_events, {"attempt_id": self.attempt_dir.name, **row})

    def dispatch(self, name, arguments):
        if name not in {"knowledge", "search"}:
            raise PermissionError(f"native fixture runner allows only knowledge/search, not {name}")
        decoded = json.loads(arguments) if isinstance(arguments, str) else copy.deepcopy(arguments)
        if not isinstance(decoded, dict):
            decoded = {"invalid_arguments": repr(decoded)}
        call_id = self.pending_call_id
        if not call_id:
            raise RuntimeError("tool dispatch has no recorded assistant call ID")
        is_knowledge = name == "knowledge"
        state_before = self.knowledge.list() if is_knowledge else None
        store_before = _file_snapshot(self.knowledge.path) if is_knowledge else None
        corpus_before = copy.deepcopy(self.search_client.documents) if name == "search" else None
        corpus_before_sha = content_hash(corpus_before) if corpus_before is not None else None
        self.journal({
            "type": "host_function_dispatch_started",
            "tool_call_id": call_id,
            "name": name,
            "arguments": canonical_json(decoded),
            "operation": copy.deepcopy(decoded),
            "state_before": copy.deepcopy(state_before),
            "state_before_sha256": content_hash(state_before) if state_before is not None else None,
            "store_file_before": copy.deepcopy(store_before),
            "corpus_before_sha256": corpus_before_sha,
            "search_source": "original_fixture_corpus" if name == "search" else None,
        })

        # This is the actual shared ToolRegistry dispatch. The result is journaled
        # before it is returned to AgentHarness or any oracle is called.
        dispatch_started = time.monotonic()
        result = super().dispatch(name, arguments)
        dispatch_duration = time.monotonic() - dispatch_started
        raw_return = {
            "sequence": len(self.receipts),
            "tool_call_id": call_id,
            "name": name,
            "arguments": canonical_json(decoded),
            "execution_kind": "host_function",
            "result": copy.deepcopy(result),
            "duration_seconds": dispatch_duration,
        }
        # Persist the exact return immediately, before any post-state reads can fail.
        _append_fsync(self.attempt_dir / "host_function_returns.jsonl", raw_return)
        self.journal({"type": "host_function_return_observed", "return": raw_return})
        state_after = self.knowledge.list() if is_knowledge else None
        store_after = _file_snapshot(self.knowledge.path) if is_knowledge else None
        corpus_after = copy.deepcopy(self.search_client.documents) if name == "search" else None
        corpus_after_sha = content_hash(corpus_after) if corpus_after is not None else None
        receipt = {
            "sequence": len(self.receipts),
            "tool_call_id": call_id,
            "name": name,
            "arguments": canonical_json(decoded),
            "operation": copy.deepcopy(decoded),
            "execution_kind": "host_function",
            "result": copy.deepcopy(result),
            "state_before": copy.deepcopy(state_before),
            "state_before_sha256": content_hash(state_before) if state_before is not None else None,
            "state_after": copy.deepcopy(state_after),
            "state_after_sha256": content_hash(state_after) if state_after is not None else None,
            "store_file_before": copy.deepcopy(store_before),
            "store_file_after": copy.deepcopy(store_after),
            "search_source": "original_fixture_corpus" if name == "search" else None,
            "corpus_snapshot": copy.deepcopy(corpus_before) if corpus_before is not None else None,
            "corpus_sha256": corpus_before_sha,
            "corpus_before_sha256": corpus_before_sha,
            "corpus_after_sha256": corpus_after_sha,
            "duration_seconds": dispatch_duration,
        }
        self.receipts.append(receipt)
        # Each raw host return and its post-state are durable before dispatch ends.
        _append_fsync(self.attempt_dir / "native_receipts.jsonl", receipt)
        self.journal({"type": "host_function_receipt_observed", "receipt": receipt})
        self.pending_call_id = None
        return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    raw = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha(raw)


def _build_attempt(task: dict, candidate: dict, source_hashes: dict[str, str],
                   output: Path, global_events: Path, runtime: dict) -> dict:
    validate_task(task)
    attempt_id = "attempt-" + hashlib.sha256(task["task_id"].encode("utf-8")).hexdigest()[:16]
    attempt_dir = output / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    _write_json(attempt_dir / "frozen_task.json", task)
    _write_json(attempt_dir / "unexecuted_candidate.json", candidate)
    _append_fsync(global_events, {"attempt_id": attempt_id, "type": "attempt_started",
                                  "task_id": task["task_id"], "task_sha256": content_hash(task),
                                  "candidate_sha256": content_hash(candidate)})
    events_path = attempt_dir / "event_journal.jsonl"
    store = KnowledgeStore(attempt_dir / "knowledge.json")
    for key, value in task["environment"].get("kv", {}).items():
        store.set(key, value)
    initial_kv = store.list()
    initial_store_file = _file_snapshot(store.path)
    local_search = LocalCorpusSearch(task["environment"].get("docs", []))
    frozen_corpus = copy.deepcopy(local_search.documents)
    corpus_sha = content_hash(frozen_corpus)
    tools = ObservedNativeToolRegistry(store, local_search, frozen_corpus,
                                       attempt_dir=attempt_dir, global_events=global_events,
                                       attempt_events=events_path)
    callback = KnowledgeSearchTeacher()
    callback_inputs: list[dict] = []

    def recorded_callback(messages, schemas):
        request = copy.deepcopy(messages)
        schema_copy = copy.deepcopy(schemas)
        response = callback(request, schema_copy)
        if response.get("tool_calls"):
            if len(response["tool_calls"]) != 1:
                raise ValueError("reviewed callback must emit one host-function action")
            tools.pending_call_id = response["tool_calls"][0]["id"]
        call = {"type": "callback_decision", "input_messages": request,
                "tool_schemas": schema_copy, "message": copy.deepcopy(response)}
        callback_inputs.append(call)
        # Persist the visible prefix and decision before harness dispatch.
        tools.journal(call)
        return response

    harness = AgentHarness(recorded_callback, tools, max_steps=16, context=None,
                           trace_path=attempt_dir / "harness_events.jsonl",
                           system_prompt=DEFAULT_SYSTEM_PROMPT)
    started = time.monotonic()
    result = None
    exception = None
    try:
        result = harness.run(task["prompt"])
    except Exception as exc:  # Preserve even partial attempts.
        exception = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        tools.journal({"type": "harness_exception", "exception": exception})
    elapsed = time.monotonic() - started

    if result is not None:
        messages = copy.deepcopy(result.messages)
        harness_events = copy.deepcopy(result.events)
        final = result.final
        stop_reason = result.stop_reason
    else:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]}]
        for call in callback_inputs:
            messages.append(copy.deepcopy(call["message"]))
            for tool_call in call["message"].get("tool_calls", []):
                receipt = next((r for r in tools.receipts if r["tool_call_id"] == tool_call["id"]), None)
                if receipt is not None:
                    messages.append({"role": "tool", "name": receipt["name"],
                                     "tool_call_id": receipt["tool_call_id"],
                                     "content": canonical_json(receipt["result"])})
        harness_events = []
        final = None
        stop_reason = "exception"

    final_kv = store.list()
    final_store_file = _file_snapshot(store.path)
    observed = {
        "type": "observation_before_oracle",
        "messages": messages,
        "effective_messages": messages,
        "final": final,
        "stop_reason": stop_reason,
        "receipts": copy.deepcopy(tools.receipts),
        "kv": copy.deepcopy(final_kv),
        "knowledge_file_initial": initial_store_file,
        "knowledge_file_final": final_store_file,
        "corpus_fixture": frozen_corpus,
        "corpus_sha256": corpus_sha,
        "exception": exception,
    }
    _append_fsync(attempt_dir / "observations_before_oracle.jsonl", observed)
    tools.journal({"type": "observation_before_oracle", "sha256": content_hash(observed),
                   "receipt_count": len(tools.receipts), "final": final,
                   "stop_reason": stop_reason, "exception": exception})

    # Only now may independent/shared checks run. Neither callback nor policy
    # received this oracle or any fixture reference.
    independent = independent_knowledge_search_oracle(task, final, final_kv) if exception is None else {
        "passed": False, "failures": ["harness raised before completion"],
        "method": "independent_fixture_recomputation"}
    shared = check_task_result(task, final or "", kv=final_kv) if exception is None else {
        "passed": False, "failures": ["harness raised before completion"], "checks": []}
    passed = exception is None and stop_reason == "final" and independent.get("passed") is True and shared.get("passed") is True
    model_events = []
    if result is not None:
        for event in harness_events:
            if event.get("type") == "assistant":
                decision = callback_inputs[len(model_events)] if len(model_events) < len(callback_inputs) else {}
                model_events.append({"type": "assistant", "step": event["step"],
                                     "input_messages": copy.deepcopy(event["input_messages"]),
                                     "tool_schemas": copy.deepcopy(decision.get("tool_schemas", tools.schemas)),
                                     "message": copy.deepcopy(event["message"])})
    tool_events = [copy.deepcopy(event) for event in harness_events if event.get("type") == "tool_execution"]
    evidence = {
        "schema": "picoagent.native_observation.v1",
        "source_id": "native_knowledge_search",
        "task": task,
        "candidate": candidate,
        "task_sha256": content_hash(task),
        "candidate_sha256": content_hash(candidate),
        "source_sha256": source_hashes,
        "teacher": {"model": None, "mode": "reviewed_procedural_replay",
                    "identity": "KnowledgeSearchTeacher"},
        "runtime": runtime,
        "tool_schemas": tools.schemas,
        "tool_schemas_sha256": content_hash(tools.schemas),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_prompt_sha256": _sha(DEFAULT_SYSTEM_PROMPT.encode("utf-8")),
        "messages": messages,
        "effective_messages": messages,
        "callback_events": callback_inputs,
        "model_events": model_events,
        "tool_events": tool_events,
        "receipts": copy.deepcopy(tools.receipts),
        "initial_kv": initial_kv,
        "initial_knowledge_file": initial_store_file,
        "kv": final_kv,
        "knowledge_file_final": final_store_file,
        "search_source": "original_fixture_corpus",
        "corpus_fixture": frozen_corpus,
        "corpus_sha256": corpus_sha,
        "final": final,
        "artifacts": {},
        "independent_oracle": independent,
        "shared_oracle": shared,
        "attempt_status": "success" if passed else ("failed" if exception is None else "incomplete"),
        "exception": exception,
        "elapsed_seconds": elapsed,
        "sft_admissible": False,
        "outer_exec_call_id": None,
    }
    record = {
        "schema": "picoagent.native_observation.v1",
        "source_id": "native_knowledge_search",
        "attempt_id": attempt_id,
        "task_id": task["task_id"],
        "task_sha256": content_hash(task),
        "candidate_sha256": content_hash(candidate),
        "family": task["family"],
        "template_id": task["template_id"],
        "split": task["split"],
        "status": "success" if passed else ("failed" if exception is None else "error"),
        "execution": "native_teacher_observed",
        "teacher_model": None,
        "teacher_mode": "reviewed_procedural_replay",
        "sft_admissible": False,
        "native_evidence": evidence,
        "raw_attempt_sha256": content_hash(evidence),
        "verification": {"passed": passed, "independent": independent, "shared": shared},
    }
    _write_json(attempt_dir / "native_observation.json", record)
    _append_fsync(global_events, {"attempt_id": attempt_id, "type": "attempt_completed",
                                  "task_id": task["task_id"], "status": record["status"],
                                  "raw_attempt_sha256": record["raw_attempt_sha256"]})
    return record


def record_batch(output_dir: Path, *, train_seeds_per_family: int = 16,
                 dev_seeds_per_family: int = 8) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if train_seeds_per_family < 1 or dev_seeds_per_family < 1:
        raise ValueError("seed counts must be positive")
    tasks = []
    for family, split in sorted(FAMILY_SPLIT_POLICY.items()):
        count = train_seeds_per_family if split == "train" else dev_seeds_per_family
        tasks.extend(generate_knowledge_search_task(family, seed) for seed in range(count))
    output_dir.mkdir(parents=True)
    (output_dir / "attempts").mkdir()
    snapshots = output_dir / "source_snapshot"
    source_hashes = {}
    for rel in SOURCE_PATHS:
        source = REPO / rel
        raw = source.read_bytes()
        source_hashes[rel] = _sha(raw)
        target = snapshots / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    runtime = _runtime()
    split_task_rows = {"train": [], "dev": []}
    candidates = []
    for task in tasks:
        split_task_rows[task["split"]].append(task)
        candidates.append(authored_unexecuted_candidate(task))
    task_hashes = {}
    for split, rows in split_task_rows.items():
        task_hashes[f"{split}.tasks.jsonl"] = _write_jsonl(output_dir / "tasks" / f"{split}.tasks.jsonl", rows)
    candidate_hash = _write_jsonl(output_dir / "candidate_plans.unexecuted.jsonl", candidates)
    task_manifest = {"schema": "picoagent.native_knowledge_search.tasks.v1",
                     "configuration": {"train_seeds_per_family": train_seeds_per_family,
                                       "dev_seeds_per_family": dev_seeds_per_family,
                                       "test_execution": False},
                     "counts": {split: len(rows) for split, rows in split_task_rows.items()},
                     "family_split_policy": FAMILY_SPLIT_POLICY,
                     "files": task_hashes,
                     "source_sha256": source_hashes}
    _write_json(output_dir / "tasks" / "manifest.json", task_manifest)
    _write_json(output_dir / "manifest.json", {
        "schema": "picoagent.native_knowledge_search.batch.v1",
        "status": "in_progress",
        "source_id": "native_knowledge_search",
        "execution": "native_teacher_observed",
        "teacher_model": None,
        "teacher_mode": "reviewed_procedural_replay",
        "sft_admissible": False,
        "configuration": {"train_seeds_per_family": train_seeds_per_family,
                           "dev_seeds_per_family": dev_seeds_per_family,
                           "test_execution": False},
        "source_sha256": source_hashes,
        "candidate_file_sha256": candidate_hash,
        "task_manifest_sha256": _file_sha(output_dir / "tasks" / "manifest.json"),
        "runtime": runtime,
        "records": 0,
        "failures": 0,
    })

    events_path = output_dir / "events.jsonl"
    records_path = output_dir / "records.jsonl"
    failures_path = output_dir / "failures.jsonl"
    failures_path.write_bytes(b"")
    records = []
    failures = []
    for task, candidate in zip(tasks, candidates):
        try:
            record = _build_attempt(task, candidate, source_hashes, output_dir, events_path, runtime)
            records.append(record)
            _append_fsync(records_path, record)
            if record["status"] != "success":
                failures.append({"task_id": task["task_id"], "attempt_id": record["attempt_id"],
                                 "status": record["status"], "verification": record["verification"]})
                _append_fsync(failures_path, failures[-1])
        except Exception as exc:
            failure = {"task_id": task["task_id"], "candidate_sha256": content_hash(candidate),
                       "status": "error", "exception_type": type(exc).__name__, "message": str(exc)[:2000]}
            failures.append(failure)
            _append_fsync(failures_path, failure)
            raise
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="complete", records=len(records), failures=len(failures),
                    success_by_split={split: sum(row["split"] == split and row["status"] == "success" for row in records)
                                      for split in ("train", "dev")},
                    events_sha256=_file_sha(events_path), records_sha256=_file_sha(records_path),
                    failures_sha256=_file_sha(failures_path) if failures_path.exists() else _sha(b""),
                    task_files_sha256=task_hashes,
                    source_snapshots={rel: _file_sha(snapshots / rel) for rel in source_hashes})
    _write_json(manifest_path, manifest)
    return {"output": str(output_dir), "status": manifest["status"], "records": len(records),
            "failures": len(failures), "success_by_split": manifest["success_by_split"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seeds-per-family", type=int, default=16)
    parser.add_argument("--dev-seeds-per-family", type=int, default=8)
    args = parser.parse_args()
    result = record_batch(args.output_dir, train_seeds_per_family=args.train_seeds_per_family,
                         dev_seeds_per_family=args.dev_seeds_per_family)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
