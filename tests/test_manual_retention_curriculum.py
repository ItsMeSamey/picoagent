"""Mixed-record contract fixtures are unexecuted; actual observations stay separate."""
import json
from pathlib import Path

import pytest

from picoagent.data.compaction_curriculum import generate_compaction_task, visible_memory
from picoagent.data.manual_retention_curriculum import FAMILIES, TokenizerAwareRetentionTeacher
from picoagent.data.manual_retention_curriculum import generate_retention_task
from picoagent.data.oracles import check_task_result
from picoagent.data.schema import canonical_json, _validate_model_event_replay
from picoagent.harness.protocol import END_MESSAGE, render_messages
from picoagent.harness.tools import TOOL_SCHEMAS


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    path = Path(__file__).resolve().parents[1] / "data/native-compaction-v1/source_snapshot/tokenizer"
    if not path.exists():
        pytest.skip("pinned tokenizer unavailable; no network fallback")
    return transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def test_extension_is_separate_deterministic_and_mixed_length():
    for family in FAMILIES:
        original = generate_compaction_task(family, 0)
        mixed = generate_retention_task(family, 0)
        assert mixed == generate_retention_task(family, 0)
        assert mixed["task_id"] != original["task_id"]
        assert mixed["family"].startswith("retention.")
        assert mixed["split"] in {"train", "dev"}
        assert original == generate_compaction_task(family, 0)
        sizes = {len(json.loads(text)["audit_detail"].split()) for text in mixed["environment"]["files"].values()}
        assert sizes == {32, 96}
    with pytest.raises(ValueError, match="train/dev"):
        generate_retention_task("compaction.amount_range", 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_actual_tokenizer_fixture_retains_and_discards_under_real_trigger(family, tokenizer):
    from test_compaction_curriculum import unit_rollout

    def count(rows, tools=TOOL_SCHEMAS):
        return len(tokenizer.encode(render_messages(rows, tools, add_generation_prompt=True), add_special_tokens=False))
    task = generate_retention_task(family, 0)
    teacher = TokenizerAwareRetentionTeacher(tokenizer)
    result, trace, manager = unit_rollout(task, mode="manual", teacher=teacher,
                                         max_tokens=4096, reserve_tokens=768,
                                         token_counter=count, request_token_counter=lambda rows: count(rows, []))
    assert result.stop_reason == "final", result.error
    assert check_task_result(task, result.final)["passed"]
    _validate_model_event_replay(trace)
    assert any(event["keep_group_indices"] for event in manager.events)
    assert any(not event["keep_group_indices"] for event in manager.events)
    for event in manager.events:
        assert event["tokens_before"] > manager.trigger_budget
        groups = json.loads(event["summary_request"][1]["content"])["groups"]
        keep = event["keep_group_indices"]
        assert len(keep) < len(groups)
        expected = [message for group in groups if group["id"] in keep for message in group["messages"]]
        assert canonical_json(expected) == canonical_json(event["retained_messages"])
        assert visible_memory(event["before_messages"]) == visible_memory(event["result_messages"])
        assert count(event["summary_request"], []) <= 4096 - 768
        assert count(event["summary_request"] + [event["summary_response"]], []) <= 4096
        assert len(tokenizer.encode(canonical_json(event["summary_response"]) + END_MESSAGE, add_special_tokens=False)) <= 768
        # Recreate the teacher from tokenizer alone: no hidden episode state.
        assert TokenizerAwareRetentionTeacher(tokenizer)(event["summary_request"], []) == event["summary_response"]
    assert trace["provenance"]["execution"] == "unexecuted"


def test_selector_cannot_use_changed_private_oracle(tokenizer):
    task = generate_retention_task("compaction.running_balance", 0)
    rows = [{"role": "user", "content": task["prompt"]}]
    teacher = TokenizerAwareRetentionTeacher(tokenizer)
    first = teacher(rows, TOOL_SCHEMAS)
    task["oracle"]["expected"] = {"hidden": "SECRET_ORACLE"}
    assert teacher(rows, TOOL_SCHEMAS) == first
    assert set(teacher.__dict__) == {"tokenizer"}


def test_actual_retention_variants_bind_origin_and_show_nonempty_choices():
    from picoagent.data.audit import file_hash
    from picoagent.data.native_storage import iter_rows
    from picoagent.data.schema import content_hash
    root = Path(__file__).resolve().parents[1] / "data/native-compaction-v1/manual-retention-v2"
    if not (root / "derivation.json").exists():
        pytest.skip("actual retention variant pilot has not been produced")
    manifest = json.loads((root / "manifest.json").read_text())
    derivation = json.loads((root / "derivation.json").read_text())
    assert derivation["retention_manifest_sha256"] == file_hash(root / "manifest.json")
    assert derivation["context_retention_variants"] == 16
    assert derivation["independent_new_fact_problems"] == 0
    origins = {row["task_id"]: row for row in derivation["records"]}
    tasks = {task["task_id"]: task for path in manifest["tasks"] for task in iter_rows(root / path)}
    assert set(origins) == set(tasks)
    for key, task in tasks.items():
        assert origins[key]["task_sha256"] == content_hash(task)
        assert origins[key]["origin_base_task_id"].startswith("compaction.")
    rows = [row for path in manifest["observation_paths"] for row in iter_rows(root / path)]
    assert len(rows) == 16
    for raw in rows:
        assert raw["source_id"] == "native_compaction_retention"
        assert raw["sft_admissible"] is False and raw["status"] == "observed_success"
        assert raw["retention_coverage"]["retained_and_discarded"] > 0
        assert raw["retention_coverage"]["empty"] > 0
        assert raw["independent_oracle"]["passed"]
        _validate_model_event_replay(raw)
