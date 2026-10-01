from __future__ import annotations

import json
from pathlib import Path

from picoagent.data.luna_knowledge_search_curriculum import (
    DEV_FAMILIES,
    FAMILY_SPLIT_POLICY,
    KnowledgeSearchTeacher,
    TRAIN_FAMILIES,
    _fixture_search,
    _task_fixtures,
    authored_unexecuted_candidate,
    generate_knowledge_search_task,
    generate_knowledge_search_tasks,
    independent_knowledge_search_oracle,
    verify_knowledge_search_curriculum,
    write_knowledge_search_curriculum,
)
from picoagent.data.schema import content_hash, validate_task


def test_default_curriculum_has_train_dev_only_family_disjoint_tasks():
    tasks = generate_knowledge_search_tasks()
    report = verify_knowledge_search_curriculum(tasks)
    assert report["passed"]
    assert report["counts"] == {"train": 128, "dev": 32, "test": 0}
    assert len(TRAIN_FAMILIES) == 8
    assert len(DEV_FAMILIES) == 4
    assert set(TRAIN_FAMILIES).isdisjoint(DEV_FAMILIES)
    assert all(FAMILY_SPLIT_POLICY[family] == "train" for family in TRAIN_FAMILIES)
    assert all(FAMILY_SPLIT_POLICY[family] == "dev" for family in DEV_FAMILIES)
    assert {task["split"] for task in tasks} == {"train", "dev"}
    assert len({task["task_id"] for task in tasks}) == len(tasks)
    assert len({task["input_sha256"] for task in tasks}) == len(tasks)
    for task in tasks:
        validate_task(task)


def test_seeded_fixtures_are_deterministic_and_vary_by_seed():
    for family in FAMILY_SPLIT_POLICY:
        first = generate_knowledge_search_task(family, 0)
        assert first == generate_knowledge_search_task(family, 0)
        second = generate_knowledge_search_task(family, 1)
        assert first["input_sha256"] != second["input_sha256"]
        assert first["environment"] != second["environment"] or first["prompt"] != second["prompt"]


def test_independent_oracle_uses_fixtures_without_reference_or_oracle():
    for family in FAMILY_SPLIT_POLICY:
        task = generate_knowledge_search_task(family, 3)
        expected_final, expected_kv = _task_fixtures(task)
        assert expected_final == task["oracle"]["expected"]
        no_hidden_answers = {key: value for key, value in task.items() if key not in {"oracle", "reference"}}
        result = independent_knowledge_search_oracle(no_hidden_answers, expected_final, expected_kv)
        assert result["passed"], (family, result)


def test_actual_search_fixture_retrieval_matches_independent_local_recomputation():
    from picoagent.data.collector import LocalCorpusSearch

    for family in ("search.fact_lookup", "search.ranked_titles", "search.two_hop",
                   "search.no_match", "search.catalog_detail", "search.active_summary"):
        task = generate_knowledge_search_task(family, 5)
        query = _first_search_query(task)
        limit = 2 if family == "search.ranked_titles" else (6 if family == "search.active_summary" else 5)
        real_local = LocalCorpusSearch(task["environment"]["docs"]).search(query, limit=limit)
        expected = _fixture_search(task["environment"]["docs"], query, limit)
        assert real_local["results"] == expected
        assert real_local["source"] == "original_fixture_corpus"
        assert real_local["untrusted"] is True


def test_teacher_initial_actions_are_visible_prompt_derived_and_use_only_host_tools():
    for family in FAMILY_SPLIT_POLICY:
        task = generate_knowledge_search_task(family, 2)
        teacher = KnowledgeSearchTeacher()
        schemas = [
            {"type": "function", "function": {"name": "knowledge", "parameters": {}}},
            {"type": "function", "function": {"name": "search", "parameters": {}}},
        ]
        response = teacher([{"role": "user", "content": task["prompt"]}], schemas)
        assert response["role"] == "assistant"
        assert response["tool_calls"]
        action = response["tool_calls"][0]
        assert action["function"]["name"] in {"knowledge", "search"}
        assert action["type"] == "function"
        args = json.loads(action["function"]["arguments"])
        assert isinstance(args, dict)
        assert "oracle" not in response and "reference" not in response


def test_candidate_sidecars_are_unexecuted_and_contain_no_fake_outputs():
    task = generate_knowledge_search_task("kv.note_write_verify", 1)
    candidate = authored_unexecuted_candidate(task)
    assert candidate["status"] == "unexecuted_candidate_plan"
    assert candidate["execution"] == "unexecuted"
    assert candidate["training_eligible"] is False
    assert candidate["receipts"] == []
    assert "final" not in candidate
    assert candidate["task_sha256"] == content_hash(task)


def test_writer_freezes_160_tasks_and_candidate_sidecar(tmp_path: Path):
    manifest_path = write_knowledge_search_curriculum(tmp_path / "curriculum")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"train": 128, "dev": 32, "test": 0}
    assert manifest["configuration"]["test_families_generated"] is False
    assert manifest["candidate_sidecar"]["records"] == 160
    for row in (manifest_path.parent / "train.tasks.jsonl").read_text().splitlines():
        assert json.loads(row)["split"] == "train"
    for row in (manifest_path.parent / "dev.tasks.jsonl").read_text().splitlines():
        assert json.loads(row)["split"] == "dev"
    candidates = [json.loads(line) for line in (manifest_path.parent / "candidates" / "unexecuted_plans.jsonl").read_text().splitlines()]
    assert len(candidates) == 160
    assert all(row["execution"] == "unexecuted" and row["receipts"] == [] for row in candidates)


def _first_search_query(task):
    prompt = task["prompt"]
    family = task["family"]
    if family == "search.fact_lookup":
        return prompt.split("this query: ", 1)[1].split(". From", 1)[0]
    if family == "search.ranked_titles":
        return prompt.split("query: ", 1)[1].split(". Request", 1)[0]
    if family == "search.two_hop":
        import re
        site = re.search(r"for current ([a-z]+) incident", prompt).group(1)
        return f"current {site} incident brief"
    if family == "search.no_match":
        code = prompt.split("exact code ", 1)[1].split(".", 1)[0]
        return "missing code " + code
    if family == "search.catalog_detail":
        import re
        asset = re.search(r"Search the local catalog for asset ([A-Za-z0-9_]+),", prompt).group(1)
        return f"asset {asset} technician assignment"
    import re
    project = re.search(r"for project ([a-z]+) inventory", prompt).group(1)
    return f"project {project} inventory record score"
