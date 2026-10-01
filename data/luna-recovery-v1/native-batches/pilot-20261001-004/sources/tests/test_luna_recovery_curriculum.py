from __future__ import annotations

import base64
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

import pytest

from picoagent.data.audit import read_jsonl
from picoagent.data.generators import GENERATOR_VERSION
from picoagent.data.luna_recovery_curriculum import (
    FAMILY_SPLIT_POLICY,
    RecoveryCandidateTeacher,
    authored_recovery_example,
    generate_recovery_task,
    generate_recovery_tasks,
    independent_recovery_oracle,
    verify_recovery_curriculum,
    write_recovery_curriculum,
)
from picoagent.data.schema import canonical_json, content_hash, safe_relative_path, validate_task, validate_trace
from picoagent.harness import AgentHarness, KnowledgeStore, TOOL_SCHEMAS
from picoagent.harness.agent import DEFAULT_SYSTEM_PROMPT


_TESTED_FAMILIES = tuple(sorted(FAMILY_SPLIT_POLICY))
_EXECUTED_FAMILIES = tuple(family for family in _TESTED_FAMILIES if FAMILY_SPLIT_POLICY[family] in {"train", "dev"})
_DENIED_COMMAND_WORDS = re.compile(
    r"\b(rm|curl|wget|nc|netcat|ssh|scp|sudo|su|printenv|env|chmod|chown|kill|dd|mkfs|shutdown|reboot|mount|umount)\b|https?://|(?:^|\s)/(?:etc|root|home|proc|sys|dev)(?:/|\s|$)|\.\.",
    re.I,
)


def test_candidate_track_has_24_family_disjoint_unexecuted_tasks(tmp_path):
    tasks = generate_recovery_tasks(seeds_per_family=2)
    assert len(tasks) == 48
    assert len({task["family"] for task in tasks}) == 24
    assert {task["family"] for task in tasks if task["split"] == "train"}.isdisjoint(
        {task["family"] for task in tasks if task["split"] == "dev"}
    )
    assert {task["family"] for task in tasks if task["split"] == "dev"}.isdisjoint(
        {task["family"] for task in tasks if task["split"] == "test"}
    )
    assert all(task["candidate_status"] == "unexecuted" and task["training_eligible"] is False for task in tasks)
    assert all(task["provenance"]["benchmark"] is False for task in tasks)
    assert all(task["provenance"]["generator_version"] == GENERATOR_VERSION for task in tasks)
    for task in tasks:
        validate_task(task)
        trace = authored_recovery_example(task)
        validate_trace(trace)
        assert trace["status"] == "unexecuted"
        assert trace["tool_events"] == []
        assert trace["verification"]["passed"] is False
    no_oracle = copy_json(tasks[0])
    no_oracle.pop("oracle")
    no_oracle.pop("reference")
    assert independent_recovery_oracle(no_oracle) == independent_recovery_oracle(tasks[0])
    first = generate_recovery_task("file_missing_csv", 3)
    assert first == generate_recovery_task("file_missing_csv", 3)
    with pytest.raises(ValueError):
        generate_recovery_task("cli_sort_bad_option", -1)


def test_written_candidate_bundle_hashes_and_family_splits(tmp_path):
    tasks = generate_recovery_tasks(seeds_per_family=1)
    manifest = write_recovery_curriculum(tmp_path / "luna", tasks, seeds_per_family=1)
    report = verify_recovery_curriculum(manifest)
    assert report["counts"] == {"train": 8, "dev": 8, "test": 8}
    assert len(report["families"]["train"]) == len(report["families"]["dev"]) == len(report["families"]["test"]) == 8
    manifest_obj = json.loads(manifest.read_text())
    assert manifest_obj["execution"] == "unexecuted"
    assert manifest_obj["training_eligible"] is False
    assert manifest_obj["native_teacher_observed_records"] == 0
    assert (manifest.parent / "native_teacher_observed.jsonl").read_text() == ""
    with pytest.raises(FileExistsError):
        write_recovery_curriculum(manifest.parent, tasks, seeds_per_family=1)


class _NativeFixtureTools:
    """Allowlisted, per-episode native fixture runner for deterministic tests only.

    This is not a production backend or training-data collector. It executes only
    the fixed RecoveryCandidateTeacher over freshly regenerated original tasks,
    with a small PATH/locale-only environment inside one disposable task dir.
    """

    def __init__(self, root: Path, task: dict[str, Any], notes_path: Path, receipt_path: Path | None = None):
        self.root = root.resolve()
        self.task = task
        self.schemas = TOOL_SCHEMAS
        self.store = KnowledgeStore(notes_path)
        for key, value in task["environment"]["kv"].items():
            self.store.set(key, value)
        self.receipts: list[dict[str, Any]] = []
        self._last_result: dict[str, Any] | None = None
        self.pending_call_id: str | None = None
        self.receipt_path = receipt_path
        if receipt_path is not None:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.open("x", encoding="utf-8").close()
        for name, content in task["environment"]["files"].items():
            assert safe_relative_path(name)
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert self.root in target.resolve().parents
            target.write_text(content, encoding="utf-8")
        self.environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if name == "bash":
            return self._bash(arguments)
        if name == "python":
            return self._python(arguments)
        if name == "write_file":
            return self._write_file(arguments)
        if name == "knowledge":
            return self._knowledge(arguments)
        raise AssertionError(f"unexpected test tool {name}")

    def _record_host_function(self, name: str, arguments: dict[str, Any], result: dict[str, Any], *, operation: str,
                              host_effect: dict[str, Any] | None = None,
                              state_before: dict[str, Any] | None = None,
                              state_after: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {"sequence": len(self.receipts), "tool_call_id": self.pending_call_id,
               "name": name, "arguments": canonical_json(arguments),
               "execution_kind": "host_function", "operation": operation,
               "argv": None, "stdin": "", "cwd": str(self.root),
               "stdout_b64": base64.b64encode(b"").decode("ascii"),
               "stderr_b64": base64.b64encode(b"").decode("ascii"),
               "stdout_sha256": hashlib.sha256(b"").hexdigest(),
               "stderr_sha256": hashlib.sha256(b"").hexdigest(), "exit_code": 0,
               "timed_out": False, "truncated": False, "duration_seconds": 0.0,
               "result": copy_json(result)}
        if host_effect is not None:
            row["host_effect"] = copy_json(host_effect)
        if state_before is not None:
            row["state_before_sha256"] = content_hash(state_before)
        if state_after is not None:
            row["state_after_sha256"] = content_hash(state_after)
        self._persist_receipt(row)
        return result

    def _tool_result(self, name: str, arguments: dict[str, Any], argv: list[str], stdin: bytes = b"") -> dict[str, Any]:
        if not stdin and name == "python":
            stdin = arguments["code"].encode("utf-8")
        started = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(argv, cwd=self.root, env=self.environment, input=stdin,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout if isinstance(error.stdout, bytes) else str(error.stdout or "").encode()
            stderr = error.stderr if isinstance(error.stderr, bytes) else str(error.stderr or "").encode()
            code = 124
        duration = time.monotonic() - started
        truncated = False
        if len(stdout) > 128 * 1024:
            stdout, truncated = stdout[:128 * 1024], True
        if len(stderr) > 128 * 1024:
            stderr, truncated = stderr[:128 * 1024], True
        result = {"stdout": stdout.decode("utf-8", errors="replace"), "stderr": stderr.decode("utf-8", errors="replace"),
                  "exit_code": code, "timed_out": timed_out, "truncated": truncated, "backend": "native_teacher_observed"}
        self._last_result = result
        row = {"sequence": len(self.receipts), "tool_call_id": self.pending_call_id,
                              "name": name, "arguments": canonical_json(arguments),
                              "execution_kind": "subprocess", "argv": argv, "stdin": stdin.decode("utf-8", errors="replace"),
                              "cwd": str(self.root), "stdout_b64": base64.b64encode(stdout).decode("ascii"),
                              "stderr_b64": base64.b64encode(stderr).decode("ascii"),
                              "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                              "exit_code": code, "timed_out": timed_out, "truncated": truncated,
                              "duration_seconds": duration, "result": copy_json(result)}
        self._persist_receipt(row)
        return result

    def _persist_receipt(self, row: dict[str, Any]) -> None:
        self.receipts.append(copy_json(row))
        if self.receipt_path is not None:
            with self.receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _bash(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments["command"]
        assert isinstance(command, str) and _DENIED_COMMAND_WORDS.search(command) is None, command
        assert "$(" not in command and "`" not in command, command
        assert not re.search(r"\b(eval|exec|source|\. |trap|function)\b", command), command
        argv = ["bash", "--noprofile", "--norc", "-c", command]
        return self._tool_result("bash", arguments, argv)

    def _python(self, arguments: dict[str, Any]) -> dict[str, Any]:
        code = arguments["code"]
        assert isinstance(code, str)
        assert not re.search(r"\b(socket|urllib|requests|subprocess|os\.system|exec\(|eval\()", code), code
        assert "../" not in code and "/etc/" not in code and "/root/" not in code
        return self._tool_result("python", arguments, ["python", "-I", "-c", code])

    def _write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path, content = arguments["path"], arguments["content"]
        assert safe_relative_path(path)
        assert path.startswith("scripts/") or path.startswith("output/")
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.is_symlink() and self.root in target.resolve().parents
        target.write_text(content, encoding="utf-8")
        result = {"path": path, "bytes": len(content.encode("utf-8")), "backend": "native_teacher_observed"}
        raw = target.read_bytes()
        host_effect = {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                       "readback_verified": raw == content.encode("utf-8")}
        return self._record_host_function("write_file", arguments, result,
                                          operation="write_only_within_task_workspace", host_effect=host_effect)

    def _knowledge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        op = arguments["operation"]
        before = self.kv_snapshot()
        if op == "list":
            result = {"items": self.store.list(arguments.get("prefix", "")), "untrusted": True}
        elif op == "get":
            key = arguments["key"]
            result = {"key": key, "value": self.store.get(key), "untrusted": True}
        elif op == "set":
            self.store.set(arguments["key"], arguments["value"])
            result = {"stored": arguments["key"]}
        else:
            result = {"deleted": self.store.delete(arguments["key"])}
        after = self.kv_snapshot()
        return self._record_host_function("knowledge", arguments, result, operation=op,
                                          state_before=before, state_after=after)

    def kv_snapshot(self) -> dict[str, Any]:
        return self.store.list()


def copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


class _RecordingTeacher:
    """Persist each visible prefix/action before the harness dispatches it."""

    def __init__(self, candidate: RecoveryCandidateTeacher, task: dict[str, Any], tools: _NativeFixtureTools,
                 action_path: Path | None = None):
        self.candidate = candidate
        self.task = task
        self.tools = tools
        self.action_path = action_path
        self.actions: list[dict[str, Any]] = []
        # Keep the exact callback inputs separately from annotation packets.
        # The full contexts are part of native observations; only the
        # annotation-only view strips the shared system message.
        self.model_calls: list[dict[str, Any]] = []
        if action_path is not None:
            action_path.parent.mkdir(parents=True, exist_ok=True)
            action_path.open("x", encoding="utf-8").close()

    def __call__(self, messages: list[dict], tool_schemas: list[dict]) -> dict[str, Any]:
        response = self.candidate(messages, tool_schemas)
        self.model_calls.append({"input_messages": copy_json(messages),
                                 "tool_schemas": copy_json(tool_schemas),
                                 "response": copy_json(response)})
        if response.get("tool_calls"):
            if len(response["tool_calls"]) != 1:
                raise ValueError("reviewed candidate callback emitted an unexpected multi-call action")
            call = response["tool_calls"][0]
            visible = _visible_messages(messages)
            row = {"schema": "picoagent.annotation.prefix.v1", "task_id": self.task["task_id"],
                   "family": self.task["family"], "split": self.task["split"],
                   "prefix_index": len(self.actions) + 1, "visible_prefix": visible,
                   "proposed_action": copy_json(response), "annotation_only": True,
                   "training_eligible": False, "excludes_oracle": True,
                   "excludes_future_tool_results": True, "excludes_final_answer": True}
            self.actions.append(row)
            self.tools.pending_call_id = call["id"]
            if self.action_path is not None:
                with self.action_path.open("a", encoding="utf-8") as handle:
                    handle.write(canonical_json(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        return response


def _independent_expected(task: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Call the separately importable, fixture-only recovery oracle."""
    return independent_recovery_oracle(task)

def _visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotation-only view of context, omitting the shared system prompt."""
    return [copy_json(message) for message in messages if message.get("role") != "system"]


def _prefix_packets(task: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packets = []
    ordinal = 0
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        if not message.get("tool_calls"):
            continue
        ordinal += 1
        packets.append({"schema": "picoagent.annotation.prefix.v1", "task_id": task["task_id"],
                        "family": task["family"], "split": task["split"], "prefix_index": ordinal,
                        "visible_prefix": _visible_messages(event.get("input_messages", [])),
                        "proposed_action": copy_json(message), "annotation_only": True,
                        "training_eligible": False, "excludes_oracle": True,
                        "excludes_future_tool_results": True, "excludes_final_answer": True})
    return packets


def _executable_facts(environment: dict[str, str]) -> dict[str, Any]:
    names = ("bash", "python", "sort", "cut", "uniq", "wc", "cat", "find", "sed", "nl", "tr")
    facts: dict[str, Any] = {}
    for name in names:
        path = shutil.which(name, path=environment.get("PATH"))
        if path is None:
            facts[name] = None
            continue
        binary = Path(path).resolve()
        probe = subprocess.run([str(binary), "--version"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=environment, timeout=3, check=False)
        facts[name] = {
            "path": str(binary),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "version_argv": [str(binary), "--version"],
            "version_stdout_b64": base64.b64encode(probe.stdout).decode("ascii"),
            "version_stderr_b64": base64.b64encode(probe.stderr).decode("ascii"),
            "version_stdout_sha256": hashlib.sha256(probe.stdout).hexdigest(),
            "version_stderr_sha256": hashlib.sha256(probe.stderr).hexdigest(),
            "version_exit_code": probe.returncode,
        }
    return facts


def _frozen_source_paths() -> tuple[Path, ...]:
    return tuple(Path(path) for path in (
        "src/picoagent/data/luna_recovery_curriculum.py",
        "tests/test_luna_recovery_curriculum.py",
        "src/picoagent/harness/agent.py",
        "src/picoagent/harness/tools.py",
        "src/picoagent/harness/protocol.py",
        "src/picoagent/harness/knowledge.py",
    ))


def _attempt_messages(model_calls: list[dict[str, Any]], receipts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild a partial exception transcript only from captured calls/results."""
    if not model_calls:
        return []
    messages = copy_json(model_calls[0]["input_messages"])
    by_call_id = {row["tool_call_id"]: row for row in receipts if row.get("tool_call_id")}
    for call_row in model_calls:
        assistant = call_row["response"]
        messages.append(copy_json(assistant))
        for call in assistant.get("tool_calls", []):
            receipt = by_call_id.get(call["id"])
            if receipt is not None:
                messages.append({"role": "tool", "name": receipt["name"],
                                 "tool_call_id": receipt["tool_call_id"],
                                 "content": canonical_json(receipt["result"])})
    return messages


def _run_episode(task: dict[str, Any], temp_root: Path, *, attempt_root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Recompute from the only approved generator entrypoint; do not accept an
    # arbitrary changed reference plan carrying a provenance label.
    family = task["family"].removeprefix("luna_recovery.")
    frozen = generate_recovery_task(family, task["seed"])
    assert content_hash(task) == content_hash(frozen)
    assert task["candidate_plan_sha256"] == content_hash(task["reference"]["plan"])
    episode = temp_root / task["task_id"].replace(":", "_")
    work = episode / "workspace"
    work.mkdir(parents=True)
    recorded = (attempt_root / task["task_id"].replace(":", "_")) if attempt_root is not None else (episode / "native_attempt")
    recorded.mkdir(parents=True, exist_ok=False)
    (recorded / "frozen_task.json").write_text(canonical_json(frozen) + "\n", encoding="utf-8")
    (recorded / "candidate_plan.json").write_text(canonical_json({"plan": frozen["reference"]["plan"],
                                                                     "plan_sha256": frozen["candidate_plan_sha256"],
                                                                     "training_eligible": False}) + "\n", encoding="utf-8")
    tools = _NativeFixtureTools(work, frozen, episode / "knowledge.json", recorded / "native_receipts.jsonl")
    callback = _RecordingTeacher(RecoveryCandidateTeacher(frozen), frozen, tools, recorded / "annotation_prefixes.jsonl")
    harness = AgentHarness(callback, tools, max_steps=14, context=None)
    tool_schemas = copy_json(tools.schemas)
    tool_schemas_hash = content_hash(tool_schemas)
    source_hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                     for source in _frozen_source_paths()}
    started = time.monotonic()
    try:
        result = harness.run(frozen["prompt"])
    except Exception as error:
        incident = {"schema": "picoagent.native_attempt_incident.v1", "attempt_id": recorded.name,
                    "task_id": frozen["task_id"], "status": "excluded_incomplete_runtime_exception",
                    "exception_type": type(error).__name__, "exception": str(error)[:2000],
                    "receipt_count": len(tools.receipts), "action_count": len(callback.actions),
                    "training_eligible": False}
        (recorded / "attempt_incident.json").write_text(canonical_json(incident) + "\n", encoding="utf-8")
        messages = _attempt_messages(callback.model_calls, tools.receipts)
        model_events = [{"type": "callback_observation", "call_index": index,
                         "input_messages": row["input_messages"], "tool_schemas": row["tool_schemas"],
                         "message": row["response"], "harness_event_status": "not_confirmed_due_to_exception"}
                        for index, row in enumerate(callback.model_calls, start=1)]
        envelope = {"schema": "picoagent.native_observation.v1", "task": frozen,
                    "candidate": {"status": "unexecuted", "training_eligible": False,
                                  "plan": frozen["reference"]["plan"], "plan_sha256": frozen["candidate_plan_sha256"]},
                    "teacher": {"model": "not_sampled", "mode": "procedural_candidate_callback_deterministic_replay",
                                "identity": "RecoveryCandidateTeacher"},
                    "source_sha256": source_hashes,
                    "tool_schemas": tool_schemas, "tool_schemas_sha256": tool_schemas_hash,
                    "system_prompt": DEFAULT_SYSTEM_PROMPT,
                    "system_prompt_sha256": hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                    "runtime": {"backend": "native_teacher_observed", "python_version": sys.version,
                                "platform": platform.platform(), "locale": str(locale.getlocale()),
                                "python_executable": str(Path(sys.executable).resolve()),
                                "python_executable_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                                "executables": _executable_facts(tools.environment),
                                "environment_policy": "PATH, LC_ALL=C, LANG=C, PYTHONIOENCODING only",
                                "environment_values": copy_json(tools.environment)},
                    "outer_exec_call_id": None, "messages": messages, "effective_messages": messages,
                    "model_events": model_events, "tool_events": [], "receipts": tools.receipts,
                    "final": None, "artifacts": {}, "kv": tools.kv_snapshot(),
                    "independent_oracle": {"passed": False, "status": "not_reached"},
                    "execution": "native_teacher_observed", "attempt_status": "incomplete",
                    "sft_admissible": False,
                    "limitation": "No Docker/Podman receipt; exception and partial native actions retained."}
        (recorded / "native_observation.json").write_text(canonical_json(envelope) + "\n", encoding="utf-8")
        return envelope, callback.actions
    expected, expected_artifacts, expected_kv = _independent_expected(frozen)
    if frozen["oracle"]["kind"] == "json_exact":
        try:
            final_value = canonical_json(json.loads(result.final))
        except (ValueError, TypeError):
            final_value = None
        final_pass = final_value == expected
    else:
        final_pass = result.final.strip() == expected
    actual_artifacts = {path: (work / path).read_text(encoding="utf-8") for path in expected_artifacts if (work / path).is_file()}
    artifacts_pass = actual_artifacts == expected_artifacts
    actual_kv = tools.kv_snapshot()
    kv_pass = actual_kv == expected_kv
    oracle_pass = final_pass and artifacts_pass and kv_pass
    for event in result.events:
        if event.get("type") == "tool_execution":
            assert event["result"].get("backend") == "native_teacher_observed" or event["name"] == "knowledge"
    assert tools.receipts
    tool_events = [copy_json(e) for e in result.events if e.get("type") == "tool_execution"]
    assert len(tools.receipts) == len(tool_events)
    for receipt, event in zip(tools.receipts, tool_events):
        assert receipt["tool_call_id"] == event["tool_call_id"]
        assert receipt["name"] == event["name"]
        assert receipt["arguments"] == event["arguments"]
        assert canonical_json(receipt["result"]) == canonical_json(event["result"])
    envelope = {
        "schema": "picoagent.native_observation.v1",
        "task": frozen,
        "candidate": {"status": "unexecuted", "training_eligible": False,
                      "plan": frozen["reference"]["plan"], "plan_sha256": frozen["candidate_plan_sha256"]},
        "teacher": {"model": "not_sampled", "mode": "procedural_candidate_callback_deterministic_replay",
                    "identity": "RecoveryCandidateTeacher"},
        "source_sha256": source_hashes,
        "tool_schemas": tool_schemas,
        "tool_schemas_sha256": tool_schemas_hash,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "runtime": {"backend": "native_teacher_observed", "python_version": sys.version,
                    "platform": platform.platform(), "locale": str(locale.getlocale()),
                    "python_executable": str(Path(sys.executable).resolve()),
                    "python_executable_sha256": hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                    "executables": _executable_facts(tools.environment), "cwd_policy": "one temporary workspace per task",
                    "environment_policy": "PATH, LC_ALL=C, LANG=C, PYTHONIOENCODING only",
                    "environment_values": copy_json(tools.environment)},
        "outer_exec_call_id": None,
        "messages": copy_json(result.messages),
        "effective_messages": copy_json(result.messages),
        "model_events": [{"type": "assistant", "step": e["step"],
                          "input_messages": copy_json(e["input_messages"]),
                          "tool_schemas": copy_json(callback.model_calls[index]["tool_schemas"]),
                          "message": copy_json(e["message"])}
                         for index, e in enumerate((e for e in result.events if e.get("type") == "assistant"))],
        "tool_events": tool_events,
        "receipts": tools.receipts,
        "final": result.final,
        "artifacts": actual_artifacts,
        "kv": actual_kv,
        "independent_oracle": {"passed": oracle_pass,
                               "source": "src/picoagent/data/luna_recovery_curriculum.py::independent_recovery_oracle",
                               "checked_from_fixtures": True,
                               "checks": {"final": final_pass, "artifacts": artifacts_pass, "kv": kv_pass}},
        "attempt_status": "complete" if oracle_pass else "complete_failed_oracle",
        "elapsed_seconds": time.monotonic() - started,
        "execution": "native_teacher_observed",
        "sft_admissible": False,
        "limitation": "No Docker/Podman receipt; local deterministic fixture replay only.",
    }
    (recorded / "native_observation.json").write_text(canonical_json(envelope) + "\n", encoding="utf-8")
    return envelope, callback.actions


@pytest.mark.parametrize("family", _EXECUTED_FAMILIES)
@pytest.mark.skipif(not os.environ.get("PICOAGENT_RUN_LUNA_NATIVE_SMOKES"), reason="native fixture execution is opt-in")
def test_recovery_callback_runs_through_shared_harness_with_real_local_results(family, tmp_path):
    """Actual fixture commands run once through AgentHarness in isolated temp dirs."""
    task = generate_recovery_task(family, 0)
    envelope, packets = _run_episode(task, tmp_path)
    assert envelope["execution"] == "native_teacher_observed"
    assert envelope["sft_admissible"] is False
    assert envelope["receipts"]
    assert packets
    assert all(packet["training_eligible"] is False for packet in packets)
    assert envelope["messages"][0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert envelope["effective_messages"][0] == envelope["messages"][0]
    assert envelope["model_events"][0]["input_messages"] == envelope["messages"][:2]
    assert envelope["model_events"][0]["tool_schemas"] == envelope["tool_schemas"]
    assert all(all(message["role"] != "system" for message in packet["visible_prefix"]) for packet in packets)
    assert envelope["tool_schemas_sha256"] == content_hash(envelope["tool_schemas"])
    assert all(details is None or details["binary_sha256"] for details in envelope["runtime"]["executables"].values())
    # Every task intended to begin with a tool failure has a genuine observed
    # nonzero/error response in its actual tool results.
    initial = envelope["tool_events"][0]["result"]
    if family != "python_output_shape":
        assert initial.get("exit_code", 0) != 0 or "error" in initial
    assert "container_id" not in envelope["runtime"] and "image" not in envelope["runtime"]


@pytest.mark.skipif(not os.environ.get("PICOAGENT_NATIVE_BATCH_DIR"), reason="native receipt batch is opt-in")
def test_capture_reviewed_native_train_dev_batch(tmp_path):
    """Create a raw native batch with per-action fsync before any success checks."""
    output = Path(os.environ["PICOAGENT_NATIVE_BATCH_DIR"])
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    sources_to_freeze = _frozen_source_paths()
    for source in sources_to_freeze:
        target = sources / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    observations_path = output / "native_teacher_observed.jsonl"
    prefixes_path = output / "annotation_prefixes.jsonl"
    attempts_root = output / "attempts"
    attempts_root.mkdir()
    selected = os.environ.get("PICOAGENT_NATIVE_FAMILIES")
    family_names = tuple(part.strip() for part in selected.split(",") if part.strip()) if selected else _EXECUTED_FAMILIES
    assert family_names and len(set(family_names)) == len(family_names)
    assert set(family_names).issubset(_EXECUTED_FAMILIES)
    tasks = [generate_recovery_task(family, 0) for family in family_names]
    assert not any(task["split"] == "test" for task in tasks)
    observations_path.open("x", encoding="utf-8").close()
    prefixes_path.open("x", encoding="utf-8").close()
    failures: list[dict[str, Any]] = []

    def append_synced(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    try:
        for task in tasks:
            attempt_dir = attempts_root / task["task_id"].replace(":", "_")
            try:
                envelope, prefix_rows = _run_episode(task, tmp_path / "episodes", attempt_root=attempts_root)
            except Exception as error:
                # Preserve a capture-stage incident even if setup fails before
                # the per-task envelope exists; never synthesize a tool reply.
                incident = {"schema": "picoagent.native_attempt_incident.v1", "attempt_id": attempt_dir.name,
                            "task_id": task["task_id"], "status": "excluded_incomplete_capture_exception",
                            "exception_type": type(error).__name__, "exception": str(error)[:2000],
                            "training_eligible": False}
                attempt_dir.mkdir(parents=True, exist_ok=True)
                incident_path = attempt_dir / "attempt_incident.json"
                if not incident_path.exists():
                    incident_path.write_text(canonical_json(incident) + "\n", encoding="utf-8")
                failures.append(incident)
                continue
            # Persist full actual attempt result and annotation-only prefixes
            # immediately. Raw per-action receipts were already flushed inside
            # the adapter, before the next policy step or any oracle check.
            append_synced(observations_path, envelope)
            for row in prefix_rows:
                append_synced(prefixes_path, row)
            receipts = envelope.get("receipts", [])
            tool_events = envelope.get("tool_events", [])
            oracle_pass = envelope.get("independent_oracle", {}).get("passed") is True
            if (len(receipts) != len(tool_events) or not oracle_pass or
                    envelope.get("attempt_status") != "complete"):
                failure = {"task_id": task["task_id"], "status": envelope.get("attempt_status"),
                           "independent_oracle": envelope.get("independent_oracle"),
                           "receipt_count": len(receipts), "tool_event_count": len(tool_events)}
                failures.append(failure)
    finally:
        source_hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                         for source in sources_to_freeze}
        files = {}
        for file in sorted(output.rglob("*")):
            if not file.is_file() or file.name == "manifest.json":
                continue
            relative = str(file.relative_to(output))
            records = len(read_jsonl(file)) if file.suffix == ".jsonl" and file.stat().st_size else 0 if file.suffix == ".jsonl" else None
            files[relative] = {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                               "bytes": file.stat().st_size, "records": records,
                               "training_eligible": False}
        observed = read_jsonl(observations_path)
        packet_rows = read_jsonl(prefixes_path)
        manifest = {"schema": "picoagent.native_observation_batch.v1", "batch_id": output.name,
                    "status": "complete" if not failures else "incomplete_or_failed_attempts_preserved",
                    "execution": "native_teacher_observed", "sft_admissible": False,
                    "included_splits": sorted({task["split"] for task in tasks}), "excluded_splits": ["test"],
                    "task_count": len(observed), "annotation_prefix_count": len(packet_rows),
                    "source_sha256": source_hashes,
                    "tool_schemas_sha256": content_hash(TOOL_SCHEMAS),
                    "full_callback_context_preserved": True,
                    "files": files, "failures": failures,
                    "limitations": ["Deterministic procedural candidate replay, not a Luna-sampled rollout.",
                                    "No test family execution, container identity, production receipt, or SFT eligibility.",
                                    "Prefix packets are derived from these same attempts; no duplicate annotation execution."]}
        (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    assert not failures, f"Native pilot preserved all attempts but recorded failures: {failures}"
