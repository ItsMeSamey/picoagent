from __future__ import annotations

import copy

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


def test_teacher_uses_observed_docs_not_private_reference_answer():
    task = generate_task("docs.default_override", 0)
    task["reference"]["final"] = '{"hidden_oracle":"must not be copied"}'
    teacher = ScriptedTeacher(task)
    observed = [{"role": "tool", "content": canonical_json({"stdout": "weave config defaults: workers=2, format=text, retries=19."}), "tool_call_id": "test"}]
    teacher.step = len(teacher.plan)
    answer = teacher(observed, [])
    assert '"retries":19' in answer["content"]
    assert "hidden_oracle" not in answer["content"]


def test_teacher_derives_knowledge_write_from_observed_value():
    task = generate_task("kv.copy_value", 0)
    teacher = ScriptedTeacher(task)
    teacher.step = 1
    observed = [{"role": "tool", "content": canonical_json({"key": "source", "value": 731}), "tool_call_id": "test"}]
    result = teacher(observed, [])
    assert '"value":731' in result["tool_calls"][0]["function"]["arguments"]
    assert "value" not in task["reference"]["plan"][-1]["arguments"]


def test_full_transcript_retains_calls_lost_from_compacted_context():
    from picoagent.data.collector import _full_transcript
    first = tool_call()
    response = {"stdout": "1", "exit_code": 0}
    events = [{"type": "assistant", "message": first, "input_messages": []},
              {"type": "tool_execution", "name": "python", "tool_call_id": "c1", "result": response},
              {"type": "compaction", "result_messages": [{"role": "user", "content": "summary"}]},
              {"type": "assistant", "message": {"role": "assistant", "content": "done"}, "input_messages": [{"role": "user", "content": "summary"}]}]
    full = _full_transcript("compute", events)
    validate_messages(full)
    assert full[2] == first
    assert full[3]["tool_call_id"] == "c1"
    assert full[-1]["content"] == "done"


def test_old_generator_specs_rejected_before_collection(tmp_path):
    task = generate_task("math.cart_total", 0)
    task["provenance"]["generator_version"] = "original-curriculum-v1"
    with pytest.raises(ValueError, match="excluded"):
        collect_task(task, tmp_path)


def _verified_unit_trace():
    # Structural validator fixture only; never exported as real collected data.
    trace = authored_example(generate_task("math.cart_total", 0))
    trace["status"] = "success"
    trace["verification"] = {"passed": True}
    trace["provenance"]["execution"] = "verified_environment"
    trace["provenance"]["runtime"] = {"backend": "docker", "container_id": "0" * 64, "image": "unit-test-only"}
    trace["raw_attempt_sha256"] = "0" * 64
    trace["model_events"] = [{"type": "assistant", "input_messages": trace["messages"][:-1], "message": trace["messages"][-1]}]
    trace["effective_messages"] = copy.deepcopy(trace["messages"])
    return trace


def test_model_events_are_bound_to_verified_transcript():
    trace = _verified_unit_trace()
    validate_trace(trace)
    trace["model_events"] = copy.deepcopy(trace["model_events"])
    trace["model_events"][0]["message"]["content"] = "Unrelated injected training content"
    with pytest.raises(DataValidationError, match="response differs"):
        validate_trace(trace)
    trace = _verified_unit_trace()
    trace["model_events"] = copy.deepcopy(trace["model_events"])
    trace["model_events"][0]["input_messages"][-1]["content"] = "Unrelated injected instruction"
    with pytest.raises(DataValidationError, match="input does not match"):
        validate_trace(trace)


def test_extra_or_omitted_model_events_rejected():
    trace = _verified_unit_trace()
    trace["model_events"].append(copy.deepcopy(trace["model_events"][0]))
    with pytest.raises(DataValidationError, match="no matching"):
        validate_trace(trace)
    trace = _verified_unit_trace()
    trace["effective_messages"][-1]["content"] = "fake final context"
    with pytest.raises(DataValidationError, match="final effective"):
        validate_trace(trace)


def test_compaction_replay_binds_summary_to_source_and_recent_suffix():
    from picoagent.data.schema import _validate_model_event_replay
    from picoagent.harness.context import ContextManager
    system = {"role": "system", "content": "Unit-test system"}
    user = {"role": "user", "content": "Solve this unit-test task. " * 60}
    c1, c2 = tool_call("c1"), tool_call("c2")
    r1 = {"role": "tool", "content": "1", "tool_call_id": "c1"}
    r2 = {"role": "tool", "content": "2", "tool_call_id": "c2"}
    prefix = [system, user, c1, r1, c2, r2]
    context = ContextManager(lambda messages, tools: {"role": "assistant", "content": "Solve the task."}, max_tokens=8192)
    compacted = context.compact(prefix, force=True)
    final = {"role": "assistant", "content": "done"}
    trace = {"messages": prefix + [final], "effective_messages": compacted.messages + [final],
             "model_events": [{"type": "assistant", "input_messages": prefix[:2], "message": c1},
                              {"type": "assistant", "input_messages": prefix[:4], "message": c2},
                              compacted.event,
                              {"type": "assistant", "input_messages": compacted.messages, "message": final}]}
    _validate_model_event_replay(trace)
    corrupted = copy.deepcopy(trace)
    corrupted["model_events"][2]["summary_request"][1]["content"] = "unrelated private prompt"
    with pytest.raises(DataValidationError, match="exact recorded source"):
        _validate_model_event_replay(corrupted)
    corrupted = copy.deepcopy(trace)
    corrupted["model_events"][2]["retained_messages"] = []
    with pytest.raises(DataValidationError, match="reconstruct"):
        _validate_model_event_replay(corrupted)


@pytest.mark.parametrize("mode", ["full", "manual"])
def test_full_and_manual_compaction_replay_preserves_exact_transition(mode):
    from picoagent.data.schema import _validate_model_event_replay
    from picoagent.harness.context import ContextManager
    system = {"role": "system", "content": "Pinned test system"}
    user = {"role": "user", "content": "Solve the original task and preserve tool facts. " * 60}
    c1, c2 = tool_call("x1"), tool_call("x2")
    r1 = {"role": "tool", "content": "old result", "tool_call_id": "x1"}
    r2 = {"role": "tool", "content": "recent result", "tool_call_id": "x2"}
    prefix = [system, user, c1, r1, c2, r2]
    def summarizer(messages, tools):
        text = canonical_json({"keep_groups": [2], "summary": "Solve the task; the first operation completed."}) if mode == "manual" else "Solve the task; both operations completed."
        return {"role": "assistant", "content": text}
    context = ContextManager(summarizer, max_tokens=8192, mode=mode)
    compacted = context.compact(prefix, force=True)
    final = {"role": "assistant", "content": "done"}
    trace = {"messages": prefix + [final], "effective_messages": compacted.messages + [final],
             "model_events": [{"type": "assistant", "input_messages": prefix[:2], "message": c1},
                              {"type": "assistant", "input_messages": prefix[:4], "message": c2},
                              compacted.event,
                              {"type": "assistant", "input_messages": compacted.messages, "message": final}]}
    _validate_model_event_replay(trace)
    forged = copy.deepcopy(trace)
    forged["model_events"][2]["before_messages"][-1]["content"] = "invented prior observation"
    with pytest.raises(DataValidationError, match="before_messages"):
        _validate_model_event_replay(forged)
    forged = copy.deepcopy(trace)
    forged["model_events"][2]["retained_messages"] = [r2]
    with pytest.raises(DataValidationError, match="atomic|retain no"):
        _validate_model_event_replay(forged)
    if mode == "manual":
        forged = copy.deepcopy(trace)
        forged["model_events"][2]["keep_group_indices"] = [1]
        with pytest.raises(DataValidationError, match="IDs differ"):
            _validate_model_event_replay(forged)
        forged = copy.deepcopy(trace)
        forged["model_events"][2]["summary_request"][1]["content"] = canonical_json({"groups": [], "retained_context_budget": 7680})
        with pytest.raises(DataValidationError, match="numbered groups"):
            _validate_model_event_replay(forged)


def test_legacy_half_compaction_replay_remains_valid():
    from picoagent.data.schema import _validate_model_event_replay
    from picoagent.harness.context import ContextManager
    prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "Long earlier task " * 100},
              tool_call("legacy"), {"role": "tool", "content": "result", "tool_call_id": "legacy"}]
    context = ContextManager(lambda messages, tools: {"role": "assistant", "content": "Keep solving."}, max_tokens=8192)
    result = context.compact(prefix, force=True)
    for field in ("mode", "before_messages", "keep_group_indices", "retained_context_budget", "trigger_budget"):
        result.event.pop(field, None)
    final = {"role": "assistant", "content": "done"}
    trace = {"messages": prefix + [final], "effective_messages": result.messages + [final],
             "model_events": [{"type": "assistant", "input_messages": prefix[:2], "message": prefix[2]},
                              result.event, {"type": "assistant", "input_messages": result.messages, "message": final}]}
    _validate_model_event_replay(trace)
