"""Metric contract fixtures, not model evaluation evidence."""
import copy

import pytest

from picoagent.evaluation import MODES, selection_key, summarize


def rows():
    return [{"task_id": task, "family": family, "mode": mode, "split": "dev",
             "checkpoint": "fixture", "execution": "unexecuted_unit_fixture",
             "task_success": mode != "manual" or task == "hard", "protocol_valid": True,
             "compactions": 1, "compaction_attempts": 1, "compaction_failures": 0,
             "model_invocations": 2, "model_responses": 2, "model_errors": 0,
             "context_overflows": 0, "attempt_errors": 0, "input_tokens": 100}
            for task, family in [("easy1", "easy"), ("easy2", "easy"), ("hard", "hard")]
            for mode in MODES]


def test_weak_mode_is_visible_and_family_balanced():
    result = summarize(rows())
    assert result["per_mode"]["manual"]["task_success"] == 1 / 3
    assert result["per_mode"]["manual"]["macro_family_success"] == 0.5
    assert result["worst_mode_macro_success"] == 0.5
    assert result["learned_model_score_verified"] is False
    with pytest.raises(ValueError, match="verified dev"):
        selection_key(result)


def test_verified_invocation_coverage_requires_every_mode():
    values = rows()
    for row in values:
        row["execution"] = "verified_environment"
        if row["mode"] == "full":
            row["model_invocations"] = 0
    report = summarize(values)
    assert report["per_mode"]["manual"]["model_invocation_coverage"] == 1
    assert report["per_mode"]["full"]["model_invocation_coverage"] == 0
    assert report["learned_model_score_verified"] is False


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "family", "checkpoint", "split"])
def test_unpaired_or_mixed_reports_rejected(mutation):
    values = rows()
    if mutation == "omit":
        values.pop()
    elif mutation == "duplicate":
        values.append(copy.deepcopy(values[0]))
    else:
        values[0][mutation] = "test" if mutation == "split" else "different"
    with pytest.raises(ValueError):
        summarize(values)


def test_selection_requires_every_mode_to_exercise_compaction():
    report = summarize(rows())
    # Pure selection logic fixture, never persisted as measured performance.
    report["learned_model_score_verified"] = True
    assert selection_key(report)[0] == 0.5
    report["per_mode"]["manual"]["compaction_exercised"] = 0
    with pytest.raises(ValueError, match="exercise compaction"):
        selection_key(report)
    report["per_mode"]["manual"]["compaction_exercised"] = 1
    report["per_mode"]["half"]["model_invocation_coverage"] = 0
    with pytest.raises(ValueError, match="invoke the policy"):
        selection_key(report)
    report["split"] = "test"
    with pytest.raises(ValueError, match="verified dev"):
        selection_key(report)


def test_runner_preserves_invalid_output_as_failure_for_each_mode(monkeypatch, tmp_path):
    import json
    from picoagent.evaluation import evaluate
    modes, archive_roots, supplied_policies = [], [], []
    def collect(task, archive_root, **kwargs):
        mode = kwargs["context_mode"]
        modes.append(mode)
        archive_roots.append(archive_root)
        supplied_policies.append(kwargs["model"])
        attempt = archive_root / "unit-fixture"
        attempt.mkdir(parents=True)
        (attempt / "events.jsonl").write_text(json.dumps({"kind": "model_request", "payload": {
            "messages": [{"role": "user", "content": "prompt"}], "tools": [{"type": "function"}]}}) + "\n" +
            json.dumps({"kind": "model_exception", "payload": {"message": "malformed fixture reply"}}) + "\n")
        (attempt / "raw.json").write_text(json.dumps({"result": {
            "stop_reason": "invalid_model_output",
            "events": [{"type": "model_error", "message": "malformed fixture reply"}]}}))
        return {"attempt_path": str(attempt), "trace": {
            "status": "failed", "verification": {"passed": False},
            "provenance": {"execution": "unexecuted_unit_fixture"}, "model_events": []}}
    monkeypatch.setattr("picoagent.data.collector.collect_task", collect)
    class Policy:
        seed = 123
        def __call__(self, messages, tools):
            raise AssertionError("unit test does not call a model")
        def count_tokens(self, messages, tools):
            return 0
    policy = Policy()
    report = evaluate([{"task_id": "a", "family": "f", "split": "dev"}],
                      policy=policy, checkpoint="fixture", archive_root=tmp_path / "eval",
                      image="unexecuted-fixture")
    assert modes == list(MODES)
    assert archive_roots == [tmp_path / "eval" / mode for mode in MODES]
    assert all(value is policy for value in supplied_policies)
    assert report["worst_mode_macro_success"] == 0
    assert all(row["protocol_valid"] == 0 for row in report["per_mode"].values())
    assert all(row["model_invocation_tasks"] == 1 for row in report["per_mode"].values())
    assert all(row["model_error_tasks"] == 1 for row in report["per_mode"].values())
    assert not report["learned_model_score_verified"]
    assert len((tmp_path / "eval/results.jsonl").read_text().splitlines()) == 3


def test_runner_refuses_test_unlock_before_any_attempt(tmp_path):
    from picoagent.evaluation import evaluate
    class Policy:
        def count_tokens(self, messages, tools):
            return 0
    with pytest.raises(ValueError, match="explicit unlock"):
        evaluate([{"task_id": "a", "family": "f", "split": "test"}], policy=Policy(),
                 checkpoint="fixture", archive_root=tmp_path / "eval", image="unused")
    assert not (tmp_path / "eval").exists()


def test_compaction_and_context_overflow_failures_remain_in_each_denominator(monkeypatch, tmp_path):
    import json
    from picoagent.evaluation import evaluate

    def collect(task, archive_root, **kwargs):
        mode = kwargs["context_mode"]
        attempt = archive_root / "unit-fixture"
        attempt.mkdir(parents=True)
        (attempt / "events.jsonl").write_text(json.dumps({"kind": "model_request", "payload": {
            "messages": [{"role": "user", "content": "prompt"}], "tools": []}}) + "\n")
        if mode == "manual":
            events = [{"type": "model_error", "message": "prompt plus generation exceeds model context"}]
            stop = "invalid_model_output"
        elif mode == "half":
            events = []
            stop = "context_budget"
        else:
            events = [{"type": "compaction_error", "error": "compaction request exceeds context budget"}]
            stop = "context_budget"
        (attempt / "raw.json").write_text(json.dumps({"result": {"events": events, "stop_reason": stop}}))
        return {"attempt_path": str(attempt), "trace": {
            "status": "failed", "verification": {"passed": False},
            "provenance": {"execution": "unexecuted_unit_fixture"}, "model_events": []}}

    monkeypatch.setattr("picoagent.data.collector.collect_task", collect)
    class Policy:
        def count_tokens(self, messages, tools):
            return len(messages)

    report = evaluate([{"task_id": "a", "family": "f", "split": "dev"}], policy=Policy(),
                      checkpoint="fixture", archive_root=tmp_path / "eval", image="unexecuted-fixture")
    assert report["paired_tasks"] == 1
    assert all(row["tasks"] == 1 and row["successes"] == 0 and row["task_failures"] == 1
               for row in report["per_mode"].values())
    assert report["per_mode"]["full"]["compaction_failure_tasks"] == 1
    assert report["per_mode"]["half"]["compaction_failure_tasks"] == 1
    assert report["per_mode"]["half"]["context_overflow_tasks"] == 1
    assert report["per_mode"]["manual"]["context_overflow_tasks"] == 1
    assert all(row["protocol_failures"] == 1 for row in report["per_mode"].values())
