"""Contract tests use explicitly unexecuted in-memory replies, never fake containers."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shlex

import pytest

from picoagent.data.audit import audit_tasks, read_jsonl, verify_attempt, verify_curriculum
from picoagent.data.collector import _full_transcript
from picoagent.data.compaction_curriculum import (
    GOAL_MARKER, MEMORY_PREFIX, SPLIT_POLICY, TRACK, VisibleContextTeacher,
    audit_compaction_trace, authored_start, collect_compaction_task,
    export_compaction_attempts, generate_compaction_task, generate_compaction_tasks,
    visible_memory, write_compaction_curriculum,
)
from picoagent.data.generators import SYSTEM_PROMPT
from picoagent.data.oracles import check_task_result
from picoagent.data.schema import DataValidationError, canonical_json, validate_trace
from picoagent.harness.agent import AgentHarness
from picoagent.harness.context import ContextManager, conservative_token_count
from picoagent.harness.tools import TOOL_SCHEMAS
from picoagent.training.data import validate_record
from picoagent.training.encoding import event_examples


class UnexecutedFixtureRegistry:
    """Read fixture strings without invoking shell, Python, or any real backend."""
    schemas = TOOL_SCHEMAS

    def __init__(self, task):
        self.files = task["environment"]["files"]
        self.read_count = 0

    def dispatch(self, name, arguments):
        assert name == "bash"
        command = shlex.split(json.loads(arguments)["command"])
        assert command[:2] == ["cat", "--"] and len(command) == 3
        self.read_count += 1
        return {"stdout": self.files[command[2]], "stderr": "", "exit_code": 0,
                "backend": "unexecuted_unit_fixture", "execution": "unexecuted",
                "container_id": "unexecuted-unit-fixture-" + str(self.read_count),
                "image": "unexecuted-unit-fixture-image",
                "timed_out": False, "truncated": False}


def unit_rollout(task, *, max_tokens=12000, teacher=None,
                 token_counter=conservative_token_count, reserve_tokens=1000):
    teacher = teacher or VisibleContextTeacher()
    manager = ContextManager(teacher, max_tokens=max_tokens, reserve_tokens=reserve_tokens,
                             token_counter=token_counter)
    harness = AgentHarness(teacher, UnexecutedFixtureRegistry(task), context=manager,
                           max_steps=task["compaction"]["horizon"] + 1,
                           system_prompt=SYSTEM_PROMPT)
    result = harness.run(task["prompt"])
    trace = authored_start(task)
    trace["messages"] = _full_transcript(task["prompt"], result.events)
    trace["effective_messages"] = result.messages
    trace["model_events"] = [event for event in result.events
                             if event["type"] in {"assistant", "compaction"}]
    trace["provenance"].update(
        execution="unexecuted", context_compaction_enabled=True,
        context_budget={"max_tokens": max_tokens, "reserve_tokens": reserve_tokens},
        accepted_compactions=sum(event.get("type") == "compaction" and event["accepted"]
                                 for event in result.events))
    # Harness fixture events are deliberately not relabelled as execution evidence.
    trace["tool_events"] = []
    return result, trace, manager


def test_generation_is_original_deterministic_and_family_disjoint():
    tasks = generate_compaction_tasks(seeds_per_family=2)
    assert tasks == generate_compaction_tasks(seeds_per_family=2)
    report = audit_tasks(tasks)
    assert report["counts"] == {"train": 4, "dev": 4, "test": 4}
    for task in tasks:
        assert task["provenance"]["benchmark"] is False
        assert task["provenance"]["execution"] == "unexecuted"
        assert task["provenance"]["curriculum_track"] == TRACK
        assert task["split"] == SPLIT_POLICY[task["family"]]
        assert "final" not in task["reference"]
        assert len(task["environment"]["files"]) == 16
    modified = generate_compaction_task(tasks[0]["family"], 0, horizon=24)
    assert modified["task_id"] != tasks[0]["task_id"]
    assert modified["split"] == tasks[0]["split"]


@pytest.mark.parametrize("family", list(SPLIT_POLICY))
def test_long_horizon_actually_triggers_repeated_compaction(family):
    task = generate_compaction_task(family, 3)
    result, trace, manager = unit_rollout(task)
    assert result.stop_reason == "final", result.error
    assert check_task_result(task, result.final)["passed"]
    assert len(manager.events) >= 2  # No force=True or fabricated compaction event.
    validate_trace(trace)
    report = audit_compaction_trace(trace, token_counter=conservative_token_count)
    assert report["passed"], report
    assert not report["eligible"]  # Unit fixtures can NEVER qualify for production SFT.
    for event in manager.events:
        assert event["tokens_before"] > manager.input_budget
        assert event["tokens_after"] < event["tokens_before"]
        assert event["result_messages"] == event["pinned_messages"] + [event["summary_message"]] + event["retained_messages"]
        expected_source = event["pinned_messages"] + event["source_messages"] + event["retained_messages"]
        assert canonical_json(expected_source[event["split_index"]:]) == canonical_json(event["retained_messages"])
        # The source excludes the pinned system prefix; the summary callback sees no suffix.
        assert json.loads(event["summary_request"][1]["content"]) == event["source_messages"]
        assert json.loads(event["summary_response"]["content"]) == visible_memory(event["source_messages"])
        assert "audit_detail" not in event["summary_response"]["content"]


def test_teacher_has_no_private_task_or_history_and_resumes_with_fresh_instance():
    task = generate_compaction_task("compaction.running_balance", 5)
    result, trace, manager = unit_rollout(task)
    first = manager.events[0]
    teacher = VisibleContextTeacher()
    assert teacher.__dict__ == {}
    assert teacher(first["summary_request"], []) == first["summary_response"]
    next_event = next(event for event in trace["model_events"]
                      if event["type"] == "assistant" and event["input_messages"] == first["result_messages"])
    assert teacher(first["result_messages"], TOOL_SCHEMAS) == next_event["message"]
    # A private oracle change cannot affect the callback, which has no task argument.
    task["oracle"]["expected"] = {"secret_future_answer": "NEVER_VISIBLE_ORACLE"}
    assert "NEVER_VISIBLE_ORACLE" not in teacher(first["summary_request"], [])["content"]
    assert teacher(first["summary_request"], []) == first["summary_response"]
    assert result.stop_reason == "final"


def test_withheld_suffix_cannot_leak_into_summary():
    task = generate_compaction_task("compaction.latest_status", 1)
    _, _, manager = unit_rollout(task)
    event = manager.events[0]
    before = event["pinned_messages"] + event["source_messages"] + copy.deepcopy(event["retained_messages"])
    suffix = before[event["split_index"]:]
    last_tool = next(message for message in reversed(suffix) if message["role"] == "tool")
    result = json.loads(last_tool["content"])
    packet = json.loads(result["stdout"])
    packet["audit_detail"] = "WITHHELD_SUFFIX_MARKER " + packet["audit_detail"]
    result["stdout"] = canonical_json(packet)
    last_tool["content"] = canonical_json(result)
    # Rebuild with changed suffix; length/message boundaries remain identical.
    before = event["pinned_messages"] + event["source_messages"] + suffix
    ctx = ContextManager(VisibleContextTeacher(), max_tokens=12000, reserve_tokens=1000,
                         token_counter=conservative_token_count)
    changed = ctx.compact(before)
    assert changed.event is not None
    assert changed.event["summary_response"] == event["summary_response"]
    assert "WITHHELD_SUFFIX_MARKER" not in changed.event["summary_response"]["content"]
    assert changed.messages[len(event["pinned_messages"]) + 1:] == suffix


def test_summary_preserves_goals_observed_facts_paths_and_actual_receipt_fields():
    task = generate_compaction_task("compaction.running_balance", 0)
    _, trace, manager = unit_rollout(task)
    final_memory = visible_memory(trace["effective_messages"])
    assert len(final_memory["observations"]) == 16
    goal = json.loads(task["prompt"].split(GOAL_MARKER)[1])
    assert final_memory["goal"] == goal
    assert final_memory["next_path"] is None
    receipts = final_memory["receipts"]
    assert len(receipts) == 1
    assert receipts[0]["calls"] == [f"record_{i}" for i in range(16)]
    assert receipts[0]["result"]["backend"] == "unexecuted_unit_fixture"
    assert "container_id" not in receipts[0]["result"]
    first_summary = json.loads(manager.events[0]["summary_response"]["content"])
    assert first_summary["next_path"] in task["environment"]["files"]
    assert first_summary["observations"] == final_memory["observations"][:len(first_summary["observations"])]


def test_failed_receipt_remains_failed_after_summary_and_stops_teacher():
    task = generate_compaction_task("compaction.running_balance", 0)
    start = authored_start(task)["messages"]
    failure = {"exit_code": 7, "stderr": "fixture permission failure", "stdout": "",
               "backend": "unexecuted_unit_fixture", "timed_out": False, "truncated": False}
    rows = start + [{"role": "tool", "tool_call_id": "record_0", "content": canonical_json(failure)}]
    memory = visible_memory(rows)
    assert memory["observations"] == [] and memory["failures"][0]["result"]["exit_code"] == 7
    after = [{"role": "user", "content": MEMORY_PREFIX + canonical_json(memory)}]
    answer = VisibleContextTeacher()(after, TOOL_SCHEMAS)
    assert "incomplete" in answer["content"] and not answer.get("tool_calls")
    assert "fixture permission failure" in canonical_json(memory)


def test_replay_rejects_suffix_or_next_input_corruption():
    _, trace, _ = unit_rollout(generate_compaction_task("compaction.running_balance", 0))
    mutated = copy.deepcopy(trace)
    event = next(event for event in mutated["model_events"] if event["type"] == "compaction")
    event["retained_messages"][-1]["content"] = "CHANGED"
    with pytest.raises(DataValidationError, match="reconstruct|context mismatch"):
        validate_trace(mutated)
    mutated = copy.deepcopy(trace)
    after_index = next(i for i, event in enumerate(mutated["model_events"]) if event["type"] == "compaction") + 1
    mutated["model_events"][after_index]["input_messages"][1]["content"] = "LOST SUMMARY"
    with pytest.raises(DataValidationError, match="replayed effective"):
        validate_trace(mutated)


def test_replay_valid_but_fact_losing_summary_fails_semantic_audit():
    class ForgetfulTeacher(VisibleContextTeacher):
        def __call__(self, messages, tools):
            result = super().__call__(messages, tools)
            if not tools and messages[0].get("content", "").startswith("Summarize the supplied"):
                memory = json.loads(result["content"])
                memory["observations"][0][0] += 1000
                result["content"] = canonical_json(memory)
            return result
    _, trace, _ = unit_rollout(generate_compaction_task("compaction.running_balance", 0), teacher=ForgetfulTeacher())
    validate_trace(trace)  # Exact replay alone cannot establish summary faithfulness.
    report = audit_compaction_trace(trace)
    assert not report["passed"]
    assert any("unsupported facts" in failure for failure in report["failures"])


def test_sft_expansion_includes_real_summary_decisions_with_tools_disabled():
    _, trace, manager = unit_rollout(generate_compaction_task("compaction.running_balance", 0))
    examples = event_examples(trace)
    summaries = [example for example, last_only in examples if example["tools"] == []]
    assert len(summaries) == len(manager.events)
    assert all(last_only for _, last_only in examples)
    for example, event in zip(summaries, manager.events):
        assert example["messages"] == event["summary_request"] + [event["summary_response"]]
        assert len(example["messages"]) == 3
    with pytest.raises(ValueError, match="verified"):
        validate_record(trace, "train")


def test_no_compaction_or_forced_under_budget_trace_is_not_admitted():
    task = generate_compaction_task("compaction.running_balance", 0, detail_words=16)
    _, trace, _ = unit_rollout(task, max_tokens=100000)
    assert not audit_compaction_trace(trace)["passed"]
    _, compacted, _ = unit_rollout(generate_compaction_task("compaction.running_balance", 0))
    compacted["provenance"]["context_budget"]["max_tokens"] = 1000000
    report = audit_compaction_trace(compacted)
    assert not report["passed"] and "compaction was not budget-triggered" in report["failures"]


def test_unavailable_runtime_archives_error_and_strict_export_preserves_it(monkeypatch, tmp_path):
    from picoagent.harness import sandbox
    def unavailable(*args, **kwargs):
        raise sandbox.SandboxUnavailable("explicitly unavailable in this unit test")
    monkeypatch.setattr(sandbox, "ContainerSandbox", unavailable)
    task = generate_compaction_task("compaction.running_balance", 0)
    result = collect_compaction_task(task, tmp_path / "attempts", token_counter=conservative_token_count)
    assert result["trace"]["status"] == "error"
    assert result["trace"]["provenance"]["execution"] == "unexecuted"
    assert not result["compaction_audit"]["eligible"]
    assert verify_attempt(result["attempt_path"])["passed"]
    report = export_compaction_attempts(tmp_path / "attempts", tmp_path / "export",
                                        token_counter=conservative_token_count)
    assert report["attempts"] == 1 and report["admitted"] == 0
    assert read_jsonl(tmp_path / "export" / "all_attempts.jsonl")[0] == result["trace"]
    assert read_jsonl(tmp_path / "export" / "compaction_audits.jsonl")[0]["eligible"] is False
    assert (tmp_path / "export" / "train.jsonl").read_text() == ""


def test_test_collection_requires_explicit_unlock(tmp_path):
    with pytest.raises(ValueError, match="include_test"):
        collect_compaction_task(generate_compaction_task("compaction.amount_range", 0), tmp_path,
                                token_counter=conservative_token_count)


def test_manifest_is_immutable_and_authored_examples_have_no_invented_results(tmp_path):
    manifest = write_compaction_curriculum(tmp_path / "corpus", seeds_per_family=1,
                                           horizon=8, detail_words=16)
    assert verify_curriculum(manifest)["counts"] == {"train": 2, "dev": 2, "test": 2}
    with pytest.raises(FileExistsError):
        write_compaction_curriculum(tmp_path / "corpus", seeds_per_family=1)
    for split in ("train", "dev", "test"):
        for example in read_jsonl(tmp_path / "corpus" / f"{split}.authored.jsonl"):
            assert example["status"] == "unexecuted"
            assert not example["tool_events"] and not example["verification"]["passed"]
            assert not any(message["role"] == "tool" for message in example["messages"])
            assert example["messages"][-1]["tool_calls"]


@pytest.fixture(scope="module")
def pinned_smol_counter():
    """Optional cached-tokenizer integration; never downloads or executes a model."""
    transformers = pytest.importorskip("transformers")
    cache = Path(os.environ.get("HF_HOME", Path(__file__).resolve().parents[2] / ".hf-cache"))
    revision = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
    snapshot = cache / "hub/models--HuggingFaceTB--SmolLM2-360M/snapshots" / revision
    if not (snapshot / "tokenizer.json").exists():
        pytest.skip("pinned SmolLM2 tokenizer is not cached; no network fallback")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    from picoagent.harness.protocol import render_messages
    def count(rows):
        return len(tokenizer.encode(render_messages(rows, TOOL_SCHEMAS, add_generation_prompt=True),
                                    add_special_tokens=False))
    return count


@pytest.mark.parametrize("family", list(SPLIT_POLICY))
def test_pinned_tokenizer_4096_budget_with_unique_unexecuted_receipt_ids(family, pinned_smol_counter):
    task = generate_compaction_task(family, 0)
    result, trace, manager = unit_rollout(task, max_tokens=4096, reserve_tokens=512,
                                         token_counter=pinned_smol_counter)
    assert result.stop_reason == "final", result.error
    assert check_task_result(task, result.final)["passed"]
    assert len(manager.events) >= 2
    assert audit_compaction_trace(trace, token_counter=pinned_smol_counter)["passed"]
    assert not audit_compaction_trace(trace, token_counter=pinned_smol_counter)["eligible"]
    ids = [json.loads(message["content"])["container_id"] for message in trace["messages"]
           if message["role"] == "tool"]
    assert len(set(ids)) == task["compaction"]["horizon"]
    for event in trace["model_events"]:
        if event["type"] == "assistant":
            assert pinned_smol_counter(event["input_messages"]) <= 4096 - 512
        else:
            assert "unexecuted-unit-fixture-" not in event["summary_response"]["content"]
