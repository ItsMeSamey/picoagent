"""Synthetic view contracts; these fixtures are never admitted training data."""
import copy

import pytest

from picoagent.data.artificial_plans import _apply, _review, augment_record
from picoagent.data.schema import content_hash
from picoagent.training.encoding import event_examples


def source():
    start = [{"role": "system", "content": "test"}, {"role": "user", "content": "read x"}]
    action = {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function",
              "function": {"name": "bash", "arguments": '{"command":"cat x"}'}}]}
    reply = {"role": "tool", "tool_call_id": "a", "content": "fixture only"}
    final = {"role": "assistant", "content": "fixture only"}
    messages = start + [action, reply, final]
    return {"task_id": "train-a", "trace_id": "fixture", "family": "read", "messages": messages,
            "effective_messages": messages, "provenance": {"context_compaction_enabled": False,
            "execution": "unexecuted_unit_fixture"}, "model_events": [
                {"type": "assistant", "input_messages": start, "message": action},
                {"type": "assistant", "input_messages": start + [action, reply], "message": final}]}


def annotation(record):
    event = record["model_events"][0]
    return {"task_id": record["task_id"], "tool_call_id": "a", "note": "read(x)", "note_tokens": 4,
            "template_id": "read", "template_sha256": content_hash({"note": "read(x)", "note_tokens": 4, "family": "read", "action_index": 0, "tool_name": "bash"}),
            "prefix_messages_sha256": content_hash(event["input_messages"]),
            "source_action_sha256": content_hash({"name": "bash", "arguments_json": '{"command":"cat x"}', "tool_call_id": "a"})}


def test_annotation_changes_only_assistant_plan_content_and_replayed_context():
    record = source()
    original = copy.deepcopy(record)
    augmented = augment_record(record, [annotation(record)])
    assert record == original
    assert augmented["messages"][2]["content"] == "read(x)"
    assert augmented["messages"][2]["tool_calls"] == record["messages"][2]["tool_calls"]
    assert augmented["messages"][3:] == record["messages"][3:]
    assert augmented["model_events"][1]["input_messages"][2]["content"] == "read(x)"
    assert augmented["provenance"]["model_context_origin"] == "synthetically_augmented_not_observed"
    assert len(event_examples(augmented)) == 2


@pytest.mark.parametrize("field", ["prefix_messages_sha256", "source_action_sha256", "task_id", "tool_call_id"])
def test_unbound_annotations_rejected(field):
    record = source()
    note = annotation(record)
    note[field] = "wrong"
    with pytest.raises(ValueError):
        augment_record(record, [note])


def test_compaction_annotation_is_not_silently_reconstructed():
    record = source()
    note = annotation(record)
    record["model_events"].append({"type": "compaction"})
    with pytest.raises(ValueError, match="compaction"):
        augment_record(record, [note])


def test_one_view_per_base_and_dev_unchanged():
    record = source()
    note = annotation(record)
    rows = {"train": [record], "dev": [{"task_id": "dev"}]}
    result = _apply(rows, [note], {"read": {"note": "read(x)", "note_tokens": 4, "family": "read", "action_index": 0, "tool_name": "bash"}}, ["train-a"], lambda note: 4)
    assert len(result["train"]) == 1
    assert result["dev"] == rows["dev"]
    assert result["train"][0]["task_id"] == record["task_id"]
    bad = dict(note, note="invent(future_answer)")
    with pytest.raises(ValueError, match="literal template"):
        _apply(rows, [bad], {"read": {"note": "read(x)", "note_tokens": 4, "family": "read", "action_index": 0, "tool_name": "bash"}}, ["train-a"])


def test_arbitrary_approved_flag_is_not_template_review_authority():
    templates = {"read": {"note": "read(x)", "note_tokens": 4, "family": "read", "action_index": 0, "tool_name": "bash"}}
    review = {"approved": True, "schema": "picoagent.artificial_action_plan.review.v1",
              "template_bodies_sha256": content_hash(templates)}
    with pytest.raises(ValueError, match="pinned review"):
        _review(templates, review, [])


def test_pinned_review_binds_exact_annotation_rows(monkeypatch):
    from picoagent.data import artificial_plans
    templates = {"read": {"note": "read(x)", "note_tokens": 4, "family": "read", "action_index": 0, "tool_name": "bash"}}
    notes = [annotation(source())]
    review = {"schema": "picoagent.artificial_action_plan.review.v1",
              "template_bodies_sha256": content_hash(templates), "annotation_rows_sha256": content_hash(notes),
              "tokenizer_identity": {"model_id": "fixture", "revision": "a" * 40,
                                     "tokenizer_json_sha256": "b" * 64}}
    monkeypatch.setattr(artificial_plans, "APPROVED_TEMPLATE_REVIEWS", frozenset({content_hash(review)}))
    assert _review(templates, review, notes) == content_hash(review)
    forged = copy.deepcopy(notes)
    forged[0]["task_id"] = "another-otherwise-valid-prefix"
    with pytest.raises(ValueError, match="exact annotation corpus"):
        _review(templates, review, forged)


def test_template_self_hash_has_one_explicit_definition():
    from picoagent.data.artificial_plans import template_body
    body = {"note": "read(x)", "note_tokens": 4}
    assert template_body({**body, "template_sha256": content_hash(body)}) == body
    with pytest.raises(ValueError, match="self-hash"):
        template_body({**body, "template_sha256": "0" * 64})


def test_actual_training_tokenizer_recounts_notes(tmp_path):
    import json
    from picoagent.data.artificial_plans import validate_note_tokenizer
    from picoagent.data.audit import file_hash
    vocab = tmp_path / "tokenizer.json"
    vocab.write_text("fixture vocabulary")
    identity = {"model_id": "fixture", "revision": "a" * 40,
                "tokenizer_json_sha256": file_hash(vocab)}
    (tmp_path / "manifest.json").write_text(json.dumps({"tokenizer_identity": identity}))
    (tmp_path / "template_review.json").write_text(json.dumps({"tokenizer_identity": identity}))
    (tmp_path / "templates.json").write_text(json.dumps({"read": {"note": "read(x)", "note_tokens": 4}}))
    class WrongTokenizer:
        def encode(self, text, add_special_tokens):
            return [1] * 30
    with pytest.raises(ValueError, match="compact-note token"):
        validate_note_tokenizer(tmp_path / "manifest.json", WrongTokenizer(), model_id="fixture",
                                revision="a" * 40, tokenizer_json_path=vocab)


def test_verified_copy_binds_prior_manifest_and_every_evidence_byte(tmp_path):
    import json
    from picoagent.data.artificial_plans import SCHEMA, _copy_verified_plan_snapshot
    from picoagent.data.audit import file_hash
    root = tmp_path / "source"
    root.mkdir()
    asset = root / "fixture.txt"
    asset.write_text("unexecuted fixture")
    manifest = {"schema": SCHEMA, "files": {"fixture.txt": {
        "bytes": asset.stat().st_size, "sha256": file_hash(asset)}}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    expected = file_hash(path)
    copied = _copy_verified_plan_snapshot(path, tmp_path / "copy", verified_manifest=manifest,
                                         expected_manifest_sha256=expected)
    assert file_hash(copied) == expected
    assert (copied.parent / "fixture.txt").read_bytes() == asset.read_bytes()
    with pytest.raises(ValueError, match="since strict verification"):
        _copy_verified_plan_snapshot(path, tmp_path / "wrong", verified_manifest={},
                                     expected_manifest_sha256=expected)
    asset.write_text("changed fixture!!!")
    with pytest.raises(ValueError, match="evidence changed"):
        _copy_verified_plan_snapshot(path, tmp_path / "tampered", verified_manifest=manifest,
                                     expected_manifest_sha256=expected)


def test_public_copy_still_performs_strict_semantic_verification(tmp_path, monkeypatch):
    import json
    from picoagent.data import artificial_plans
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": artificial_plans.SCHEMA}))
    def reject(*args, **kwargs):
        raise ValueError("strict verification was called")
    monkeypatch.setattr(artificial_plans, "verify_plan_snapshot", reject)
    with pytest.raises(ValueError, match="strict verification was called"):
        artificial_plans.copy_plan_snapshot(path, tmp_path / "copy")
