from __future__ import annotations

import ast
import copy
import csv
import io
import json
import shlex

import pytest

from picoagent.data import authored_example, check_task_result, generate_task, generate_tasks
from picoagent.data.audit import audit_tasks, verify_curriculum, write_curriculum
from picoagent.data.generators import SPLIT_POLICY, svg_reference
from picoagent.data.schema import DataValidationError, canonical_json, training_eligible, validate_task


def test_generation_deterministic_family_held_out():
    first = generate_tasks(seeds_per_family=5, holdout_seeds_per_family=2)
    assert first == generate_tasks(seeds_per_family=5, holdout_seeds_per_family=2)
    assert len(SPLIT_POLICY) == 24
    assert audit_tasks(first)["counts"] == {"train": 40, "dev": 16, "test": 16}
    for task in first:
        assert task["split"] == SPLIT_POLICY[task["family"]]
        assert generate_task(task["family"], task["seed"] + 100)["split"] == task["split"]


def test_all_references_pass_pure_author_checks_but_are_not_training_data():
    for task in generate_tasks(seeds_per_family=7):
        trace = authored_example(task)
        assert trace["verification"]["author_answer_matches"] is True
        assert trace["verification"]["passed"] is False
        assert trace["status"] == "unexecuted"
        assert trace["provenance"]["execution"] == "authored_example"
        assert not trace["tool_events"]
        assert not any(message["role"] == "tool" for message in trace["messages"])
        assert training_eligible(trace) is False


def test_fixture_oracles_independently_recomputed():
    for seed in range(12):
        task = generate_task("bash.csv_sum", seed)
        rows = csv.DictReader(io.StringIO(task["environment"]["files"]["input/ledger.csv"]))
        total = sum(int(row["amount"]) for row in rows)
        assert check_task_result(task, json.dumps({"total": total}))["passed"]
        assert not check_task_result(task, json.dumps({"total": total + 1}))["passed"]
        task = generate_task("python.square_sort", seed)
        numbers = json.loads(task["environment"]["files"]["input/values.json"])
        expected = {"values": sorted({value ** 2 for value in numbers})}
        assert check_task_result(task, json.dumps(expected))["passed"]


def test_reference_programs_parse_without_executing_on_host():
    for task in generate_tasks(seeds_per_family=3):
        for action in task["reference"]["plan"]:
            if action["name"] == "python":
                ast.parse(action["arguments"]["code"])
            elif action["name"] == "bash" and action["arguments"]["command"].startswith("python -c "):
                argv = shlex.split(action["arguments"]["command"])
                assert argv[:2] == ["python", "-c"]
                ast.parse(argv[2])


def test_docs_semantics_change_between_seeds():
    flag_answers = [generate_task("docs.flag_lookup", seed)["oracle"]["expected"]["argv"][1] for seed in range(10)]
    defaults = [generate_task("docs.default_override", seed)["oracle"]["expected"]["retries"] for seed in range(10)]
    orders = [tuple(generate_task("docs.ordered_recipe", seed)["oracle"]["expected"]["steps"]) for seed in range(10)]
    assert len(set(flag_answers)) > 1
    assert len(set(defaults)) > 1
    assert len(set(orders)) > 1


def test_roundtrip_preserves_svg_order_and_oracle():
    task = json.loads(canonical_json(generate_task("visualization.bar_sorted", 2)))
    svg = svg_reference(task["oracle"]["expected"])
    assert check_task_result(task, "done", artifacts={"output/chart.svg": svg})["passed"]
    assert not check_task_result(task, "done", artifacts={"output/chart.svg": svg.replace('y="20"', 'y="500"')})["passed"]
    assert not check_task_result(task, "done")["passed"]
    assert not check_task_result(task, "done", artifacts={"output/chart.svg": svg.replace("</svg>", "<script>alert(1)</script></svg>")})["passed"]


def test_kv_oracle_checks_actual_postcondition():
    task = generate_task("kv.counter_update", 2)
    answer = task["reference"]["final"]
    assert not check_task_result(task, answer)["passed"]
    assert check_task_result(task, answer, kv=task["oracle"]["kv_expected"])["passed"]


def test_json_oracle_strict_types_and_no_markdown():
    task = generate_task("math.cart_total", 0)
    assert not check_task_result(task, "```json\n" + task["reference"]["final"] + "\n```")["passed"]
    altered = copy.deepcopy(task)
    altered["oracle"]["expected"] = {"answer": 1}
    assert not check_task_result(altered, '{"answer":true}')["passed"]
    assert not check_task_result(altered, '{"answer":NaN}')["passed"]


def test_leakage_fails_closed():
    tasks = generate_tasks(seeds_per_family=1)
    assert audit_tasks(tasks)["passed"]
    with pytest.raises(ValueError, match="task_id|input_sha256"):
        audit_tasks(tasks + [tasks[0]])
    bad = copy.deepcopy(tasks[0])
    bad["task_id"] += "-renamed"
    bad["split"] = "dev" if bad["split"] != "dev" else "train"
    with pytest.raises(ValueError, match="family"):
        audit_tasks(tasks + [bad])
    bad = copy.deepcopy(tasks[0])
    bad["provenance"]["source"] = "benchmark"
    with pytest.raises(DataValidationError, match="original"):
        validate_task(bad)


def test_curriculum_manifest_is_reproducible_and_detects_tamper(tmp_path):
    tasks = generate_tasks(seeds_per_family=1)
    paths = [write_curriculum(tmp_path / name, tasks, configuration={"seeds_per_family": 1}) for name in ("first", "second")]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert verify_curriculum(paths[0])["passed"]
    with pytest.raises(FileExistsError):
        write_curriculum(tmp_path / "first", tasks, configuration={})
    target = paths[0].parent / "train.tasks.jsonl"
    target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        verify_curriculum(paths[0])


def test_installed_help_and_original_source_curriculum_is_separate(tmp_path):
    from picoagent.data.tool_curriculum import TOOL_SPLIT_POLICY, generate_tool_tasks
    tasks = generate_tool_tasks(seeds_per_family=3)
    assert audit_tasks(tasks)["counts"] == {"train": 6, "dev": 6, "test": 6}
    assert not set(TOOL_SPLIT_POLICY) & set(SPLIT_POLICY)
    for task in tasks:
        assert authored_example(task)["verification"]["author_answer_matches"]
        first = task["reference"]["plan"][0]["arguments"]["command"]
        assert "--help" in first or "-m pydoc" in first or first.startswith("cat local_api_")
        for path, source in task["environment"]["files"].items():
            if path.endswith(".py"):
                ast.parse(source)  # Never execute generated module on the host.
    manifest = write_curriculum(tmp_path / "tools", tasks, configuration={"track": "tool-docs"}, split_policy=TOOL_SPLIT_POLICY)
    assert verify_curriculum(manifest)["passed"]


def test_doc_driven_api_arguments_follow_observed_semantics():
    from picoagent.data.collector import ScriptedTeacher
    from picoagent.data.tool_curriculum import generate_tool_task
    for family in ("tool_docs.pydoc_affine", "tool_docs.source_index", "tool_docs.pydoc_window"):
        for seed in range(5):
            task = generate_tool_task(family, seed)
            teacher = ScriptedTeacher(task)
            teacher.step = 1
            source = next(text for path, text in task["environment"]["files"].items() if path.endswith(".py"))
            observed = [{"role": "tool", "tool_call_id": "read_docs", "content": canonical_json({"stdout": source, "exit_code": 0})}]
            output = teacher(observed, [])
            code = json.loads(output["tool_calls"][0]["function"]["arguments"])["code"]
            ast.parse(code)
            assert "input/values.json" in code
            if family.endswith("source_index"):
                expected_position = 3 if "positions are 1-based" in source else 2
                assert f"lookup(values, {expected_position})" in code
            elif family.endswith("pydoc_window"):
                endpoint = 4 if "stop is inclusive" in source else 5
                assert f"select(values, 2, {endpoint})" in code
