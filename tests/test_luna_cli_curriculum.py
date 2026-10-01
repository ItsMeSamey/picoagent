from __future__ import annotations

import ast
import json
import pytest

from picoagent.data.audit import audit_tasks, read_jsonl, verify_curriculum
from picoagent.data.luna_cli_curriculum import (
    FAMILY_SPLITS,
    LunaCLITeacher,
    candidate_plan_rows,
    generate_task,
    generate_tasks,
    iter_tasks,
    build_annotation_packets,
    independent_expected_from_fixtures,
    independent_oracle_check,
    run_native_teacher_observed,
    run_native_teacher_observed_batch,
    verify_luna_cli_dataset,
    write_luna_cli_dataset,
)
from picoagent.data.schema import canonical_json, validate_task


def test_luna_cli_families_are_original_deterministic_and_family_split():
    tasks = generate_tasks()
    assert len(FAMILY_SPLITS) == len(tasks) == 36
    assert {task["family"] for task in tasks} == set(FAMILY_SPLITS)
    assert audit_tasks(tasks)["counts"] == {"train": 12, "dev": 12, "test": 12}
    assert tasks == generate_tasks()
    assert generate_task("cli.csv_revenue_total", 1) == generate_task("cli.csv_revenue_total", 1)
    assert generate_task("cli.csv_revenue_total", 10)["split"] == "train"
    assert generate_task("cli.help_flag_lookup", 10)["split"] == "dev"
    assert generate_task("cli.help_recovery_recipe", 10)["split"] == "test"
    for task in tasks:
        validate_task(task)
        assert task["provenance"]["source"] == "original_procedural"
        assert task["provenance"]["benchmark"] is False
        assert independent_expected_from_fixtures(task) == task["oracle"]["expected"] if task["oracle"]["kind"] == "json_exact" else independent_oracle_check(task, task["reference"]["final"])["passed"]


def test_plans_are_fixture_derived_and_candidates_are_not_fake_traces():
    for task in generate_tasks():
        plan_text = canonical_json(task["reference"]["plan"])
        assert canonical_json(task["oracle"]["expected"]) not in plan_text
        assert not any(key in plan_text for key in ("stdout", "stderr", "container_id", "tool_events"))
        for action in task["reference"]["plan"]:
            if action["name"] == "python":
                ast.parse(action["arguments"]["code"])
                assert "input/" in action["arguments"]["code"] or "Path(\"input\")" in action["arguments"]["code"]
            elif action["name"] == "bash":
                assert action["arguments"]["command"].startswith("cat ")
    rows = candidate_plan_rows(generate_tasks())
    assert [len(rows[split]) for split in ("train", "dev", "test")] == [12, 12, 12]
    assert all(row["status"] == "unexecuted_candidate_plan" and row["not_a_trace"] for split in rows.values() for row in split)


def test_teacher_uses_observed_results_not_private_expected_values():
    task = generate_task("cli.csv_revenue_total")
    task["oracle"]["expected"] = {"secret": "poisoned"}
    task["reference"]["final"] = "poisoned final"
    teacher = LunaCLITeacher(task)
    call = teacher([{"role": "user", "content": task["prompt"]}], [])
    assert call["tool_calls"][0]["function"]["name"] == "python"
    tool_message = {"role": "tool", "content": canonical_json({"stdout": '{"revenue":31}', "stderr": "", "exit_code": 0})}
    response = teacher([{"role": "user", "content": task["prompt"]}, call, tool_message], [])
    assert response["content"] == '{"revenue":31}'
    assert "poisoned" not in response["content"]


def test_recovery_and_help_finalizers_consume_observed_outputs():
    task = generate_task("cli.recover_missing_csv")
    teacher = LunaCLITeacher(task)
    first = teacher([{"role": "user", "content": task["prompt"]}], [])
    assert first["tool_calls"][0]["function"]["arguments"] == '{"command":"cat input/catalog-old.csv"}'
    second = teacher([{"role": "user", "content": task["prompt"]}, first,
                      {"role": "tool", "content": canonical_json({"stdout": "", "stderr": "not found", "exit_code": 1})}], [])
    code = json.loads(second["tool_calls"][0]["function"]["arguments"])["code"]
    assert "Path(\"input\").glob(\"*.csv\")" in code
    expected = independent_expected_from_fixtures(task)
    final = teacher([{"role": "user", "content": task["prompt"]}, first,
                     {"role": "tool", "content": canonical_json({"stdout": "", "stderr": "not found", "exit_code": 1})},
                     second, {"role": "tool", "content": canonical_json({"stdout": canonical_json(expected), "exit_code": 0})}], [])
    assert json.loads(final["content"]) == expected

    docs = generate_task("cli.help_flag_lookup")
    helper = LunaCLITeacher(docs)
    search_call = helper([{"role": "user", "content": docs["prompt"]}], [])
    manual = next(value for path, value in docs["environment"]["files"].items() if path.startswith("docs/"))
    cat_call = helper([{"role": "user", "content": docs["prompt"]}, search_call,
                       {"role": "tool", "content": canonical_json({"results": [{"id": "guide"}]})}], [])
    assert cat_call["tool_calls"][0]["function"]["arguments"]
    result = helper([{"role": "user", "content": docs["prompt"]}, search_call,
                     {"role": "tool", "content": canonical_json({"results": [{"id": "guide"}]})}, cat_call,
                     {"role": "tool", "content": canonical_json({"stdout": manual, "exit_code": 0})}], [])
    assert json.loads(result["content"]) == docs["oracle"]["expected"]


def test_luna_dataset_manifest_and_candidate_plan_audit(tmp_path):
    output = tmp_path / "luna-cli-v1"
    summary = write_luna_cli_dataset(output)
    assert summary["families"] == 36
    assert verify_luna_cli_dataset(output)["passed"]
    assert verify_curriculum(output / "manifest.json")["passed"]
    plans = read_jsonl(output / "train.candidate_plans.unexecuted.jsonl")
    assert len(plans) == 12
    assert all(row["execution"] == "unexecuted" and row["not_a_trace"] for row in plans)


def test_native_teacher_receipts_are_separate_and_require_container_replay(tmp_path):
    tasks = [generate_task("cli.csv_revenue_total"), generate_task("cli.recover_missing_csv")]
    result = run_native_teacher_observed(tasks, tmp_path / "native_teacher_observed")
    assert result["records"] == result["passed"] == 2
    records = read_jsonl(tmp_path / "native_teacher_observed" / "observations.jsonl")
    assert len(records) == 2
    for record in records:
        assert record["execution_kind"] == "native_teacher_observed"
        assert record["not_a_trace"] is True
        assert record["sft_admissible"] is False
        assert record["requires_container_replay"] is True
        assert record["model_identity"] is None
        assert record["native_execution_metadata"]["container_id"] is None
        assert record["oracle_checks"]["independent"]["passed"]
        assert record["fixture_unchanged"] is True
        assert record["fixture_sha256_before"] == record["fixture_sha256_after"]
        assert record["tool_events"]
        assert all("stdout_base64" in action["execution"] for action in record["tool_events"] if action["name"] != "search")
        assert all("container_id" not in action["execution"] for action in record["tool_events"])
        assert record["source_snapshot_manifest_sha256"]
    packets = build_annotation_packets(records)
    all_packets = packets["train"] + packets["dev"] + packets["test"]
    assert len(all_packets) == 3
    assert len({packet["task_id"] for packet in all_packets}) == 2
    for packet in all_packets:
        assert packet["variant"] == "artificial_reasoning_annotation"
        assert packet["token_hint"] == {"minimum": 8, "maximum": 24}
        assert "oracle" not in canonical_json(packet).lower()
        assert "final_response" not in canonical_json(packet)
        assert packet["prefix_messages"][-1]["role"] in {"user", "tool"}


def test_native_batch_is_streaming_hash_chained_resumable_and_train_dev_gated(tmp_path):
    output = tmp_path / "append_only"
    one_task = generate_task("cli.csv_revenue_total")
    first = run_native_teacher_observed_batch(iter([one_task]), output)
    assert first["completed_attempts"] == first["passed"] == 1
    rows = read_jsonl(output / "attempt_index.jsonl")
    assert len(rows) == 1 and rows[0]["sequence"] == 0
    group = output / rows[0]["attempt_dir"]
    journal = read_jsonl(group / "events.jsonl")
    assert journal[0]["kind"] == "attempt_started"
    assert journal[-1]["kind"] == "attempt_completed"
    previous = "0" * 64
    for sequence, row in enumerate(journal):
        claimed = row.pop("sha256")
        assert row["sequence"] == sequence and row["previous_sha256"] == previous
        from picoagent.data.schema import content_hash
        assert content_hash(row) == claimed
        previous = claimed
    resumed = run_native_teacher_observed_batch(iter([one_task]), output)
    assert resumed["completed_attempts"] == 0 and resumed["skipped_completed"] == 1
    assert len(read_jsonl(output / "attempt_index.jsonl")) == 1
    with pytest.raises(ValueError, match="not authorized"):
        run_native_teacher_observed_batch(iter([generate_task("cli.help_recovery_recipe")]), tmp_path / "blocked")


def test_seed_iterator_scales_without_materializing_10008_train_rows():
    train_count = sum(1 for _ in iter_tasks(seeds_by_split={"train": 834, "dev": 1, "test": 1}, splits=("train",)))
    assert train_count == 12 * 834 == 10008
