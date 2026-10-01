from __future__ import annotations

import copy
import json

import pytest

from picoagent.data import authored_example, generate_task
from picoagent.data.audit import AttemptArchive, read_jsonl, verify_attempt
from picoagent.data.collector import LocalCorpusSearch, ScriptedTeacher, collect_task, export_attempts
from picoagent.data.schema import DataValidationError, canonical_json, validate_messages, validate_trace


def tool_call(call_id="c1"):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "python", "arguments": '{"code":"print(1)"}'}}]}


def test_canonical_tool_call_roundtrip():
    messages = [{"role": "user", "content": "compute"}, tool_call(), {"role": "tool", "content": "1", "tool_call_id": "c1"}, {"role": "assistant", "content": "1"}]
    validate_messages(messages)
    with pytest.raises(DataValidationError, match="pending|unanswered"):
        validate_messages(messages[:2])
    validate_messages(messages[:2], allow_incomplete=True)
    with pytest.raises(DataValidationError):
        validate_messages(messages + [tool_call()])
    malformed = copy.deepcopy(messages)
    malformed[1]["tool_calls"][0]["function"]["arguments"] = {"code": "print(1)"}
    with pytest.raises(DataValidationError, match="serialized"):
        validate_messages(malformed)


def test_authored_trace_cannot_be_relabelled_as_success():
    trace = authored_example(generate_task("math.cart_total", 0))
    trace["status"] = "success"
    trace["verification"]["passed"] = True
    with pytest.raises(DataValidationError, match="unexecuted|authored"):
        validate_trace(trace)
    trace["provenance"]["execution"] = "verified_environment"
    with pytest.raises(DataValidationError, match="runtime"):
        validate_trace(trace)


def test_archive_keeps_failed_raw_and_detects_edit(tmp_path):
    task = generate_task("math.cart_total", 0)
    archive = AttemptArchive(tmp_path, task, teacher="unit_test_authored")
    trace = authored_example(task)
    trace["trace_id"] = archive.attempt_id
    archive.event("test_failure", {"bad_answer": "wrong"})
    archive.finalize({"raw_output": "wrong", "error": "intentional test fixture"}, trace)
    assert verify_attempt(archive.path)["passed"]
    with pytest.raises(RuntimeError):
        archive.event("late", {})
    (archive.path / "raw.json").write_text('{"modified":true}')
    with pytest.raises(ValueError, match="integrity"):
        verify_attempt(archive.path)


def test_invalid_trace_is_archived_before_error(tmp_path):
    task = generate_task("math.cart_total", 0)
    archive = AttemptArchive(tmp_path, task, teacher="unit_test_authored")
    trace = authored_example(task)
    trace["messages"] = []
    with pytest.raises(DataValidationError):
        archive.finalize({"raw_output": "malformed"}, trace)
    assert (archive.path / "raw.json").exists()
    assert (archive.path / "invalid_trace.json").exists()
    assert verify_attempt(archive.path)["passed"]


def test_collection_fail_closed_without_runtime_preserves_attempt(monkeypatch, tmp_path):
    from picoagent.harness import sandbox
    def unavailable(*args, **kwargs):
        raise sandbox.SandboxUnavailable("intentional unavailable runtime for unit test")
    monkeypatch.setattr(sandbox, "ContainerSandbox", unavailable)
    def never_called(messages, tools):
        raise AssertionError("model must never run without a verified container")
    result = collect_task(generate_task("math.cart_total", 0), tmp_path / "archive", model=never_called)
    assert result["trace"]["status"] == "error"
    assert result["trace"]["provenance"]["execution"] == "unexecuted"
    assert not result["trace"]["verification"]["passed"]
    assert verify_attempt(result["attempt_path"])["passed"]
    exported = export_attempts(tmp_path / "archive", tmp_path / "export")
    assert exported["attempts"] == 1 and exported["admitted"] == 0
    assert len(read_jsonl(tmp_path / "export" / "all_attempts.jsonl")) == 1
    assert not (tmp_path / "export" / "train.jsonl").read_text()


def test_local_search_is_explicitly_fixture_retrieval():
    task = generate_task("search.fact_lookup", 0)
    search = LocalCorpusSearch(task["environment"]["docs"])
    query = task["reference"]["plan"][0]["arguments"]["query"]
    result = search.search(query)
    assert result["source"] == "original_fixture_corpus"
    assert result["results"][0]["id"] == task["oracle"]["expected"]["source"]
    assert result["untrusted"] is True
    assert search.search("absent_token_12345")["results"] == []


def test_scripted_teacher_only_requests_tools_and_stops_on_failure():
    task = generate_task("python.square_sort", 0)
    teacher = ScriptedTeacher(task)
    first = teacher([{"role": "user", "content": task["prompt"]}], [])
    assert first["role"] == "assistant" and first["tool_calls"]
    failed = teacher([{"role": "tool", "content": canonical_json({"exit_code": 1, "stderr": "failed"}), "tool_call_id": "teacher_call_1"}], [])
    assert failed["role"] == "assistant" and not failed.get("tool_calls")
    assert failed["content"] != task["reference"]["final"]
