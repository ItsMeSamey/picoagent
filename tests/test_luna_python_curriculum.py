from __future__ import annotations

import ast
import csv
import io
import json

import pytest

from picoagent.data.generators import svg_reference
from picoagent.data.luna_python_curriculum import (
    DOC_FAMILIES,
    FAMILY_SPLITS,
    LunaPythonCallback,
    SVG_FAMILIES,
    candidate_record,
    generate_luna_python_task,
    generate_luna_python_tasks,
    independent_expected,
    independent_oracle,
    verify_independent_result,
    write_luna_python_dataset,
)
from picoagent.data.oracles import check_task_result
from picoagent.data.schema import canonical_json, content_hash, training_eligible, validate_task


def test_families_are_original_deterministic_and_split_by_whole_template():
    assert len(FAMILY_SPLITS) == 32
    first = generate_luna_python_tasks(seeds_per_family=2)
    second = generate_luna_python_tasks(seeds_per_family=2)
    assert first == second
    assert len(first) == 64
    assert len({row["family"] for row in first}) == 32
    assert {split: sum(row["split"] == split for row in first) for split in ("train", "dev", "test")} == {
        "train": 32, "dev": 16, "test": 16
    }
    for row in first:
        validate_task(row)
        assert FAMILY_SPLITS[row["family"]] == row["split"]
        assert row["input_sha256"] == content_hash({"prompt": row["prompt"], "environment": row["environment"]})
        assert row["provenance"]["source"] == "original_procedural"
        assert row["provenance"]["benchmark"] is False
    for family in FAMILY_SPLITS:
        a, b = generate_luna_python_task(family, 0), generate_luna_python_task(family, 1)
        assert a["split"] == b["split"] == FAMILY_SPLITS[family]
        assert a["task_id"] != b["task_id"]
        assert a["input_sha256"] != b["input_sha256"]


def test_scaled_generation_can_keep_held_out_splits_smaller():
    rows = generate_luna_python_tasks(train_seeds_per_family=3, holdout_seeds_per_family=1)
    assert len(rows) == 64
    assert {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "test")} == {
        "train": 48, "dev": 8, "test": 8
    }
    # This configuration scales to 10,000 train episodes with 128 each in dev/test.
    assert 16 * 625 == 10_000 and 8 * 16 == 128


def test_authored_reference_answers_pass_pure_oracle_but_are_not_training_data():
    for family in FAMILY_SPLITS:
        task = generate_luna_python_task(family)
        if family in SVG_FAMILIES:
            artifacts = {"output/chart.svg": svg_reference(task["oracle"]["expected"])}
            result = check_task_result(task, "Wrote output/chart.svg.", artifacts=artifacts)
        else:
            result = check_task_result(task, task["reference"]["final"])
        assert result["passed"], (family, result)
        assert not training_eligible(_unexecuted_trace(task))


def test_independent_fixture_oracle_covers_all_families_without_reading_expected():
    for family in FAMILY_SPLITS:
        for seed in range(5):
            task = generate_luna_python_task(family, seed)
            expected = independent_expected(task)
            if family in SVG_FAMILIES:
                assert canonical_json(expected) == canonical_json(task["oracle"]["expected"])
                artifacts = {"output/chart.svg": svg_reference(expected)}
                result = verify_independent_result(task, "Wrote output/chart.svg.", artifacts)
            else:
                final = "Computed result: " + canonical_json(expected) + "."
                assert final == task["oracle"]["expected"]
                result = verify_independent_result(task, final)
                assert independent_oracle(task, final)["passed"]
            assert result["passed"], (family, seed, result)


def _unexecuted_trace(task):
    from picoagent.data.generators import authored_example

    return authored_example(task)


def test_fixture_sets_cover_csv_svg_and_unfamiliar_python_docs():
    tables = [generate_luna_python_task(f) for f in FAMILY_SPLITS if f.startswith("py_table.")]
    assert all("input/task.csv" in row["environment"]["files"] for row in tables)
    assert "input/products.csv" in generate_luna_python_task("py_table.left_join")["environment"]["files"]
    for family in SVG_FAMILIES:
        task = generate_luna_python_task(family)
        assert task["oracle"]["kind"] == "svg"
        assert "input/task.json" in task["environment"]["files"]
        assert task["oracle"]["artifact_path"] == "output/chart.svg"
    for family in DOC_FAMILIES:
        task = generate_luna_python_task(family, 7)
        modules = [path for path in task["environment"]["files"] if path.endswith(".py")]
        assert len(modules) == 1
        ast.parse(task["environment"]["files"][modules[0]])
        assert "python -m pydoc" in task["prompt"]
    rounding_task = generate_luna_python_task("py_docs.rounding_mode", 7)
    rounding_input = json.loads(rounding_task["environment"]["files"]["input/task.json"])
    assert set(rounding_input) == {"value", "places"}


def test_callback_uses_shared_protocol_and_only_observed_tool_stdout():
    task = generate_luna_python_task("py_math.invoice_total")
    policy = LunaPythonCallback()
    request = policy([{"role": "system", "content": "system"}, {"role": "user", "content": task["prompt"]}], [])
    call = request["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "python"
    code = json.loads(call["function"]["arguments"])["code"]
    ast.parse(code)  # Parse only; generated code is never run on the host.
    observed = {"stdout": "Computed result: {\"demo\":7}.", "stderr": "", "exit_code": 0}
    response = policy([{"role": "user", "content": task["prompt"]},
                       {"role": "tool", "content": canonical_json(observed)}], [])
    assert response["content"] == observed["stdout"]
    assert "oracle" not in response["content"]


@pytest.mark.parametrize("family", sorted(DOC_FAMILIES))
def test_docs_callback_adapts_arguments_to_observed_documentation(family):
    task = generate_luna_python_task(family, 3)
    module = next(name[:-3] for name in task["environment"]["files"] if name.endswith(".py"))
    policy = LunaPythonCallback()
    first = policy([{"role": "user", "content": task["prompt"]}], [])
    assert first["tool_calls"][0]["function"]["name"] == "bash"
    assert module in json.loads(first["tool_calls"][0]["function"]["arguments"])["command"]
    source = next(value for name, value in task["environment"]["files"].items() if name.endswith(".py"))
    second = policy([{"role": "user", "content": task["prompt"]},
                     {"role": "tool", "content": canonical_json({"stdout": source, "stderr": "", "exit_code": 0})}], [])
    assert second["tool_calls"][0]["function"]["name"] == "python"
    code = json.loads(second["tool_calls"][0]["function"]["arguments"])["code"]
    ast.parse(code)
    assert "input/task.json" in code and module in code
    assert "Computed result:" in code
    if family == "py_docs.window_stop":
        if "stop is exclusive" in source:
            assert "d['stop']+1" in code
        else:
            assert "d['stop']" in code


def test_candidate_records_are_not_transcripts_or_training_data():
    task = generate_luna_python_task("py_count.choose")
    record = candidate_record(task, raw_response="candidate source", plan=[{"tool": "python", "code": "read input/task.json"}])
    assert record["status"] == "unexecuted_candidate_plan"
    assert record["execution"] == "unexecuted"
    assert record["training_eligible"] is False
    assert record["raw_candidate_entry"] == "candidate source"
    assert record["planned_tool_calls"][0]["code"] == "read input/task.json"
    assert record["receipts"] == [] and record["tool_events"] == []
    assert "messages" not in record


def test_dataset_writer_hashes_unexecuted_specs_and_never_claims_execution(tmp_path):
    manifest = write_luna_python_dataset(tmp_path / "pilot", seeds_per_family=1)
    payload = json.loads(manifest.read_text())
    assert payload["execution"] == "unexecuted"
    assert payload["verified_trace_count"] == 0
    assert payload["counts"] == {"train": 16, "dev": 8, "test": 8}
    assert payload["candidate_policy"]["training_eligible"] is False
    candidate_rows = (manifest.parent / "candidate_plans.jsonl").read_text().splitlines()
    assert len(candidate_rows) == 0
    from picoagent.data.audit import verify_curriculum
    assert verify_curriculum(manifest)["passed"]


def test_prompts_expose_output_keys_svg_structure_and_pivot_axes():
    for family in FAMILY_SPLITS:
        task = generate_luna_python_task(family, 460)
        if family in SVG_FAMILIES:
            for clause in ("title element", "data-label", "numeric width", "output/chart.svg", "no scripts"):
                assert clause in task["prompt"]
        else:
            assert "top-level key(s)" in task["prompt"]
    pivot = generate_luna_python_task("py_table.pivot_counts", 460)
    axes = json.loads(pivot["environment"]["files"]["input/task.json"])
    assert axes == {"regions": ["east", "north", "west"], "states": ["done", "queued", "ready"]}
    assert "input/task.json" in pivot["prompt"] and "zero counts" in pivot["prompt"]


def test_reported_audit_edge_seeds_have_public_contracts_and_matching_oracles():
    pivot = generate_luna_python_task("py_table.pivot_counts", 460)
    axes = json.loads(pivot["environment"]["files"]["input/task.json"])
    observed_pairs = {(row["region"], row["state"]) for row in
                      csv.DictReader(io.StringIO(pivot["environment"]["files"]["input/task.csv"]))}
    assert axes["states"] == ["done", "queued", "ready"]
    assert any((region, "queued") not in observed_pairs for region in axes["regions"])
    assert independent_expected(pivot)["counts"]["east"]["queued"] >= 0
    pivot_final = "Computed result: " + canonical_json(independent_expected(pivot)) + "."
    assert independent_oracle(pivot, pivot_final)["passed"]

    task = generate_luna_python_task("py_docs.window_stop", 3)
    module = next(path[:-3].split("/")[-1] for path in task["environment"]["files"] if path.endswith(".py"))
    source = next(value for path, value in task["environment"]["files"].items() if path.endswith(".py"))
    callback = LunaPythonCallback()
    user = {"role": "user", "content": task["prompt"]}
    first = callback([user], [])
    assert first["tool_calls"][0]["function"]["name"] == "bash"
    observed = {"stdout": source, "stderr": "", "exit_code": 0}
    second = callback([user, {"role": "tool", "content": canonical_json(observed)}], [])
    code = json.loads(second["tool_calls"][0]["function"]["arguments"])["code"]
    assert module in code
    if "stop is exclusive" in source:
        assert "d['stop']+1" in code
    else:
        assert "d['stop']" in code
    data = json.loads(task["environment"]["files"]["input/task.json"])
    assert independent_expected(task)["values"] == data["values"][data["start"]:data["stop"] + 1]


def test_luna_subagent_candidates_are_preserved_unexecuted_and_split_by_family():
    from pathlib import Path

    directory = Path(__file__).parents[1] / "data" / "luna-python-v1"
    rows = [json.loads(line) for line in (directory / "candidate_plans.jsonl").read_text().splitlines()]
    assert len(rows) == 20
    assert {row["model"] for row in rows} == {"gpt-6-luna"}
    assert all(row["status"] == "unexecuted_candidate_plan" for row in rows)
    assert all(row["execution"] == "unexecuted" and row["training_eligible"] is False for row in rows)
    assert all(row["compatibility"] == "unverified_against_frozen_fixture" for row in rows)
    assert all(not row["receipts"] and not row["tool_events"] and not row["has_final_response"] for row in rows)
    for row in rows:
        response_path = directory / row["raw_response_file"]
        assert response_path.is_file()
        response_rows = [json.loads(line) for line in response_path.read_text().splitlines()]
        source = next(item for item in response_rows if row["family"] in item["families"])
        candidates = json.loads(source["raw_response"])["candidate_plans"]
        original = next(item for item in candidates if item["family"] == row["family"])
        assert row["candidate"] == original
        assert row["task_sha256"] == content_hash(generate_luna_python_task(row["family"], 0))
        for action in row["planned_tool_calls"]:
            if action.get("tool") == "python":
                ast.parse(action["code"])  # Static syntax inspection only.
                assert "input/task.json" in action["code"]
                assert "oracle" not in action["code"].lower()
                assert "reference" not in action["code"].lower()
