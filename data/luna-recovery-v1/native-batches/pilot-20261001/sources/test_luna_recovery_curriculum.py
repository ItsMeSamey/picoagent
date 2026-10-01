from __future__ import annotations

import base64
import csv
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
    verify_recovery_curriculum,
    write_recovery_curriculum,
)
from picoagent.data.schema import canonical_json, content_hash, safe_relative_path, validate_task, validate_trace
from picoagent.harness import AgentHarness, KnowledgeStore, TOOL_SCHEMAS


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

    def __init__(self, root: Path, task: dict[str, Any], notes_path: Path):
        self.root = root.resolve()
        self.task = task
        self.schemas = TOOL_SCHEMAS
        self.store = KnowledgeStore(notes_path)
        for key, value in task["environment"]["kv"].items():
            self.store.set(key, value)
        self.receipts: list[dict[str, Any]] = []
        self._last_result: dict[str, Any] | None = None
        for name, content in task["environment"]["files"].items():
            assert safe_relative_path(name)
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert self.root in target.resolve().parents
            target.write_text(content, encoding="utf-8")
        self.environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", "LANG": "C", "PYTHONIOENCODING": "utf-8"}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "bash":
            return self._bash(arguments)
        if name == "python":
            return self._python(arguments)
        if name == "write_file":
            return self._write_file(arguments)
        if name == "knowledge":
            return self._knowledge(arguments)
        raise AssertionError(f"unexpected test tool {name}")

    def _record_host_function(self, name: str, arguments: dict[str, Any], result: dict[str, Any], *, operation: str) -> dict[str, Any]:
        self.receipts.append({"sequence": len(self.receipts), "name": name, "arguments": canonical_json(arguments),
                              "execution_kind": "host_function", "operation": operation,
                              "argv": None, "stdin": "", "cwd": str(self.root),
                              "stdout_b64": base64.b64encode(b"").decode("ascii"),
                              "stderr_b64": base64.b64encode(b"").decode("ascii"),
                              "stdout_sha256": hashlib.sha256(b"").hexdigest(),
                              "stderr_sha256": hashlib.sha256(b"").hexdigest(), "exit_code": 0,
                              "timed_out": False, "truncated": False, "duration_seconds": 0.0,
                              "result": copy_json(result)})
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
        self.receipts.append({"sequence": len(self.receipts), "name": name, "arguments": canonical_json(arguments),
                              "execution_kind": "subprocess", "argv": argv, "stdin": stdin.decode("utf-8", errors="replace"),
                              "cwd": str(self.root), "stdout_b64": base64.b64encode(stdout).decode("ascii"),
                              "stderr_b64": base64.b64encode(stderr).decode("ascii"),
                              "stdout_sha256": hashlib.sha256(stdout).hexdigest(), "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                              "exit_code": code, "timed_out": timed_out, "truncated": truncated,
                              "duration_seconds": duration, "result": copy_json(result)})
        return result

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
        return self._record_host_function("write_file", arguments, result, operation="write_only_within_task_workspace")

    def _knowledge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        op = arguments["operation"]
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
        return self._record_host_function("knowledge", arguments, result, operation=op)

    def kv_snapshot(self) -> dict[str, Any]:
        return self.store.list()


def copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _independent_expected(task: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Recompute from visible task fixtures, never reading oracle/reference values."""
    family = task["family"].removeprefix("luna_recovery.")
    files = task["environment"]["files"]
    artifacts: dict[str, str] = {}
    kv: dict[str, Any] = {}
    if family.startswith("cli_"):
        path = next(name for name in files if name.startswith("input/"))
        if family == "cli_sort_bad_option":
            nums = [int(line) for line in files[path].splitlines()]
            final = "\n".join(map(str, sorted(set(nums), reverse=True)))
            artifacts["output/result.txt"] = final + "\n"
        elif family == "cli_cut_bad_option":
            final = "\n".join(line.split(";")[1] for line in files[path].splitlines())
            artifacts["output/result.txt"] = final + "\n"
        elif family == "cli_uniq_bad_option":
            values = files[path].splitlines()
            counts = {value: values.count(value) for value in set(values)}
            final = "\n".join(f"{counts[key]:7} {key}" for key in sorted(counts))
            artifacts["output/result.txt"] = final + "\n"
        else:
            final = str(len(files[path].splitlines()))
            artifacts["output/result.txt"] = final + "\n"
        return final, artifacts, kv
    if family in {"pydoc_wrong_callable", "source_index_origin", "source_range_endpoint", "pydoc_wrong_signature"}:
        numbers = json.loads(files["input/values.json"])
        if family == "pydoc_wrong_callable":
            output = {"values": [number * number for number in numbers]}
        elif family == "source_index_origin":
            output = {"value": numbers[2]}
        elif family == "source_range_endpoint":
            output = {"values": numbers[1:4]}
        else:
            offset = int(re.search(r"offset=([0-9]+)", task["prompt"]).group(1))
            output = {"values": [number + offset for number in numbers]}
        text = canonical_json(output)
        artifacts["output/result.json"] = text + "\n"
        return text, artifacts, kv
    if family.startswith("kv_"):
        source_key, source = next(iter(task["environment"]["kv"].items()))
        total = source["total"] + (2 if family == "kv_wrong_key" else 0)
        target = re.search(r"at ([A-Za-z0-9_./-]+)", task["prompt"]).group(1) if family == "kv_wrong_key" else re.search(r"at ([A-Za-z0-9_./-]+)", task["prompt"]).group(1)
        kv = {**task["environment"]["kv"], target: {"total": total, "checked": True}}
        return canonical_json({"total": total}), artifacts, kv
    if family == "docs_untrusted_hint":
        doc = files["docs/local-guide.md"]
        total = int(re.search(r"accepted total is ([0-9]+)", doc).group(1))
        return canonical_json({"total": total}), artifacts, kv
    if family == "file_nested_path":
        value = next(content for path, content in files.items() if path.startswith("input/") and path.endswith(".txt"))
        total = sum(int(line) for line in value.splitlines() if line.strip())
        output = {"total": total}
    elif family in {"jsonl_shape"}:
        rows = [json.loads(line) for line in files["input/events.jsonl"].splitlines() if line.strip()]
        output = {"total": sum(int(row.get("score", row.get("value", 0))) for row in rows)}
    elif family in {"file_missing_json", "json_wrong_key", "python_keyerror_schema", "json_mixed_numeric"}:
        data_path = next(path for path in files if path.startswith("input/") and path.endswith(".json"))
        data = json.loads(files[data_path])
        if isinstance(data, list):
            rows = data
        else:
            rows = next(value for value in data.values() if isinstance(value, list))
        if rows and isinstance(rows[0], dict):
            total = sum(int(row.get("amount", row.get("value", 0))) for row in rows)
        else:
            total = sum(int(value) for value in rows)
        output = {"total": total}
    else:
        data_path = next(path for path in files if path.startswith("input/") and path.endswith(".csv"))
        raw = files[data_path]
        try:
            dialect = csv.Sniffer().sniff(raw[:2048], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(raw.splitlines(), dialect=dialect))
        field = next(key for key in rows[0] if key.lower() in {"amount", "quantity", "qty", "units", "value", "score"})
        output = {"total": sum(int(row[field]) for row in rows if row.get(field, "").strip())}
    text = canonical_json(output)
    output_path = "output/report.json" if family == "scope_preserve_input" else "output/result.json"
    artifacts[output_path] = text + "\n"
    return text, artifacts, kv


def _visible_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep task/action/result content while omitting the shared system prompt."""
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


def _executable_facts() -> dict[str, Any]:
    names = ("bash", "python", "sort", "cut", "uniq", "wc", "cat", "find", "sed", "nl", "tr")
    return {name: shutil.which(name) for name in names}


def _run_episode(task: dict[str, Any], temp_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Recompute from the only approved generator entrypoint; do not accept an
    # arbitrary changed reference plan carrying a provenance label.
    family = task["family"].removeprefix("luna_recovery.")
    frozen = generate_recovery_task(family, task["seed"])
    assert content_hash(task) == content_hash(frozen)
    assert task["candidate_plan_sha256"] == content_hash(task["reference"]["plan"])
    episode = temp_root / task["task_id"].replace(":", "_")
    work = episode / "workspace"
    work.mkdir(parents=True)
    tools = _NativeFixtureTools(work, frozen, episode / "knowledge.json")
    callback = RecoveryCandidateTeacher(frozen)
    harness = AgentHarness(callback, tools, max_steps=14, context=None)
    started = time.monotonic()
    result = harness.run(frozen["prompt"])
    assert result.stop_reason == "final", (task["task_id"], result.stop_reason, result.error, result.final)
    expected, expected_artifacts, expected_kv = _independent_expected(frozen)
    if frozen["oracle"]["kind"] == "json_exact":
        assert canonical_json(json.loads(result.final)) == expected, (task["task_id"], result.final, expected)
    else:
        assert result.final.strip() == expected, (task["task_id"], result.final, expected)
    for path, content in expected_artifacts.items():
        assert (work / path).read_text(encoding="utf-8") == content, (task["task_id"], path)
    if expected_kv:
        assert tools.kv_snapshot() == expected_kv
    for event in result.events:
        if event.get("type") == "tool_execution":
            assert event["result"].get("backend") == "native_teacher_observed" or event["name"] == "knowledge"
    assert tools.receipts
    envelope = {
        "schema": "picoagent.native_observation.v1",
        "task": frozen,
        "candidate": {"status": "unexecuted", "training_eligible": False,
                      "plan": frozen["reference"]["plan"], "plan_sha256": frozen["candidate_plan_sha256"]},
        "teacher": {"model": "not_sampled", "mode": "procedural_candidate_callback_deterministic_replay",
                    "identity": "RecoveryCandidateTeacher"},
        "source_sha256": {"src/picoagent/data/luna_recovery_curriculum.py": hashlib.sha256(
            Path("src/picoagent/data/luna_recovery_curriculum.py").read_bytes()).hexdigest()},
        "runtime": {"backend": "native_teacher_observed", "python_version": sys.version,
                    "platform": platform.platform(), "locale": str(locale.getlocale()),
                    "executables": _executable_facts(), "cwd_policy": "one temporary workspace per task",
                    "environment_policy": "PATH, LC_ALL=C, LANG=C, PYTHONIOENCODING only"},
        "outer_exec_call_id": None,
        "messages": _visible_messages(result.messages),
        "effective_messages": _visible_messages(result.messages),
        "model_events": [{"type": "assistant", "input_messages": _visible_messages(e["input_messages"]),
                          "message": copy_json(e["message"])} for e in result.events if e.get("type") == "assistant"],
        "tool_events": [copy_json(e) for e in result.events if e.get("type") == "tool_execution"],
        "receipts": tools.receipts,
        "final": result.final,
        "artifacts": {name: content for name, content in expected_artifacts.items()},
        "kv": tools.kv_snapshot(),
        "independent_oracle": {"passed": True, "source": "tests/test_luna_recovery_curriculum.py::_independent_expected",
                               "checked_from_fixtures": True},
        "elapsed_seconds": time.monotonic() - started,
        "execution": "native_teacher_observed",
        "sft_admissible": False,
        "limitation": "No Docker/Podman receipt; local deterministic fixture replay only.",
    }
    return envelope, _prefix_packets(frozen, result.events)


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
    # Every task intended to begin with a tool failure has a genuine observed
    # nonzero/error response in its actual tool results.
    initial = envelope["tool_events"][0]["result"]
    if family != "python_output_shape":
        assert initial.get("exit_code", 0) != 0 or "error" in initial
    assert "container_id" not in envelope["runtime"] and "image" not in envelope["runtime"]


@pytest.mark.skipif(not os.environ.get("PICOAGENT_NATIVE_BATCH_DIR"), reason="native receipt batch is opt-in")
def test_capture_reviewed_native_train_dev_batch(tmp_path):
    """Create one exclusive raw native batch and prefix packets from the same runs."""
    output = Path(os.environ["PICOAGENT_NATIVE_BATCH_DIR"])
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    generator_source = Path("src/picoagent/data/luna_recovery_curriculum.py")
    test_source = Path("tests/test_luna_recovery_curriculum.py")
    for source in (generator_source, test_source):
        target = sources / source.name
        target.write_bytes(source.read_bytes())
    observations_path = output / "native_teacher_observed.jsonl"
    prefixes_path = output / "annotation_prefixes.jsonl"
    observations = []
    packets = []
    selected = os.environ.get("PICOAGENT_NATIVE_FAMILIES")
    family_names = tuple(part.strip() for part in selected.split(",") if part.strip()) if selected else _EXECUTED_FAMILIES
    assert family_names and len(set(family_names)) == len(family_names)
    assert set(family_names).issubset(_EXECUTED_FAMILIES)
    tasks = [generate_recovery_task(family, 0) for family in family_names]
    assert not any(task["split"] == "test" for task in tasks)
    for task in tasks:
        envelope, prefix_rows = _run_episode(task, tmp_path / "episodes")
        assert len(envelope["receipts"]) == len(envelope["tool_events"])
        assert len(prefix_rows) == sum(bool(event["message"].get("tool_calls")) for event in envelope["model_events"])
        assert all("reference" not in packet and "oracle" not in packet for packet in prefix_rows)
        assert all(not any(m["role"] == "assistant" and m.get("content") for m in packet["visible_prefix"])
                   for packet in prefix_rows)
        observations.append(envelope)
        packets.extend(prefix_rows)
    for target, rows in ((observations_path, observations), (prefixes_path, packets)):
        with target.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
    source_hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest() for source in (generator_source, test_source)}
    files = {}
    for target in (observations_path, prefixes_path):
        files[target.name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size,
                              "records": len(read_jsonl(target)), "training_eligible": False}
    for source in sources.iterdir():
        files[f"sources/{source.name}"] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                           "bytes": source.stat().st_size, "records": None,
                                           "training_eligible": False}
    manifest = {"schema": "picoagent.native_observation_batch.v1", "batch_id": output.name,
                "execution": "native_teacher_observed", "sft_admissible": False,
                "included_splits": sorted({task["split"] for task in tasks}), "excluded_splits": ["test"],
                "task_count": len(observations), "annotation_prefix_count": len(packets),
                "source_sha256": source_hashes, "files": files,
                "limitations": ["One deterministic procedural candidate replay per train/dev family.",
                                "No test family execution, container identity, production receipt, or SFT eligibility.",
                                "Prefix packets are derived from the same run events; no duplicate annotation execution."]}
    (output / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
