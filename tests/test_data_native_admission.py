"""Admission unit fixtures are not actual native execution observations."""
from __future__ import annotations

import json

import pytest

from picoagent.data import authored_example, generate_task
from picoagent.data.native_admission import NATIVE_MANIFEST_SCHEMA, _validate_review, verify_native_snapshot
from picoagent.data.schema import DataValidationError, validate_trace


def test_native_category_requires_opt_in_before_any_other_claim():
    trace = authored_example(generate_task("math.cart_total", 0))
    trace["provenance"]["execution"] = "native_teacher_observed"
    with pytest.raises(DataValidationError, match="explicit admission opt-in"):
        validate_trace(trace)


def test_approved_flag_alone_cannot_admit_native_source():
    review = {"schema": "picoagent.native_teacher.source_review.v1", "source_id": "luna_cli", "reviewer": "unit test",
              "review_notes": "Fabricated fixture, intentionally not actual evidence", "oracle_id": "fixture", "command_policy": "fixture",
              "approved": True, "source_sha256": {"fixture.py": "0" * 64}, "oracle_source_sha256": "0" * 64,
              "teacher_mode": "luna_authored_program_deterministic_replay", "arbitrary_learner_execution_allowed": False}
    with pytest.raises(DataValidationError, match="independently pinned"):
        _validate_review(review)


def test_generic_jsonl_or_manifest_cannot_unlock_native_training(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": NATIVE_MANIFEST_SCHEMA}))
    with pytest.raises(DataValidationError, match="explicit opt-in"):
        verify_native_snapshot(path)
    with pytest.raises(DataValidationError, match="admission policy"):
        verify_native_snapshot(path, allow_native_teacher=True)


def test_actual_cli_sample_projection_and_tamper_binding():
    """Read an actual archived sample; never run its recorded commands."""
    import copy
    from pathlib import Path
    from picoagent.data.native_admission import extract_native_observation
    from picoagent.data.schema import _validate_model_event_replay
    root = Path(__file__).resolve().parents[1]
    sample = root / "data/luna-cli-v1/native_teacher_observed/sample_001/observations.jsonl"
    if not sample.exists():
        pytest.skip("actual CLI observation artifact not distributed with this test checkout")
    raw = json.loads(sample.read_text().splitlines()[0])
    tasks = [json.loads(line) for line in (root / "data/luna-cli-v1/train.tasks.jsonl").read_text().splitlines()]
    task = next(row for row in tasks if row["task_id"] == raw["task_id"])
    projection = extract_native_observation("luna_cli", raw, task)
    _validate_model_event_replay(projection)
    assert projection["final"] == '{"revenue":771}'
    forged = copy.deepcopy(raw)
    forged["tool_events"][0]["execution"]["stdout_base64"] = "ZmFrZQ=="
    with pytest.raises(DataValidationError, match="captured raw stream"):
        extract_native_observation("luna_cli", forged, task)
    forged = copy.deepcopy(raw)
    forged["tool_events"][0]["arguments"]["code"] = "print('unrelated')"
    forged["tool_events"][0]["arguments_json"] = json.dumps(forged["tool_events"][0]["arguments"])
    with pytest.raises(DataValidationError, match="reviewed frozen action plan"):
        extract_native_observation("luna_cli", forged, task)


def test_actual_sealed_native_pilot_and_full_evidence_copy(tmp_path):
    from pathlib import Path
    from picoagent.data.native_admission import copy_native_snapshot
    from picoagent.training.data import read_records, verify_dataset
    root = Path(__file__).resolve().parents[1]
    original = root / "data/native-sharded-pilot-v1/manifest.json"
    if not original.exists():
        pytest.skip("reviewed native pilot snapshot not distributed with this checkout")
    with pytest.raises(ValueError, match="explicit"):
        verify_dataset(original)
    manifest, records = verify_dataset(original, allow_native_teacher=True)
    assert {split: len(rows) for split, rows in records.items()} == {"train": 1, "dev": 1}
    assert manifest["source_counts"]["admitted"] == {"luna_cli": {"train": 1, "dev": 1}}
    unsealed = tmp_path / "unsealed-native.jsonl"
    unsealed.write_text(json.dumps(records["train"][0]) + "\n")
    with pytest.raises(ValueError, match="verified"):
        read_records(unsealed, "train")
    copied = copy_native_snapshot(original, tmp_path / "copied")
    assert copied.read_bytes() == original.read_bytes()
    copied_manifest, copied_records = verify_dataset(copied, allow_native_teacher=True)
    assert copied_manifest == manifest and copied_records == records
    source_file = manifest["all_observations"]["paths"][0]
    target = copied.parent / source_file
    target.chmod(0o644)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity mismatch"):
        verify_dataset(copied, allow_native_teacher=True)


@pytest.mark.parametrize(('directory', 'source'), [
    ('native-python-pilot-v3', 'luna_python'),
    ('native-compaction-pilot-v1', 'native_compaction'),
    ('native-recovery-pilot-v5', 'luna_recovery'),
])
def test_reviewed_new_adapters_bind_actual_raw_bytes(directory, source):
    import copy
    from pathlib import Path
    from picoagent.data.native_admission import extract_native_observation
    from picoagent.data.native_storage import iter_rows
    from picoagent.data.schema import _validate_model_event_replay
    root = Path(__file__).resolve().parents[1] / 'data' / directory
    if not (root / 'manifest.json').exists():
        pytest.skip('optional actual native pilot is not distributed')
    manifest = json.loads((root / 'manifest.json').read_text())
    row = next(iter_rows(root / manifest['all_observations']['paths'][0]))
    evidence = row['native_evidence']
    projection = extract_native_observation(source, evidence['raw_record'], evidence['task'])
    _validate_model_event_replay(projection)
    validate_trace(row, allow_native_teacher=True)
    forged = copy.deepcopy(evidence['raw_record'])
    if source == 'luna_python':
        forged['native_evidence']['receipts'][0]['stdout_bytes_b64'] = 'ZmFrZQ=='
    elif source == 'native_compaction':
        forged['receipts'][0]['stdout_b64'] = 'ZmFrZQ=='
    else:
        forged['receipts'][0]['stdin_b64'] = 'ZmFrZQ=='
    with pytest.raises(DataValidationError, match='hash|mismatch|differs'):
        extract_native_observation(source, forged, evidence['task'])


def test_recovery_host_write_requires_exact_readback():
    import copy
    from pathlib import Path
    from picoagent.data.native_storage import iter_rows
    root = Path(__file__).resolve().parents[1] / 'data/native-recovery-pilot-v5'
    if not (root / 'manifest.json').exists():
        pytest.skip('optional actual recovery pilot is not distributed')
    manifest = json.loads((root / 'manifest.json').read_text())
    row = next(iter_rows(root / manifest['all_observations']['paths'][0]))
    forged = copy.deepcopy(row)
    receipt = next(v for v in forged['native_evidence']['receipts'] if v['name'] == 'write_file')
    receipt['host_effect']['readback_verified'] = False
    with pytest.raises(DataValidationError, match='readback'):
        validate_trace(forged, allow_native_teacher=True)


def test_native_combination_requires_opt_in_and_rejects_same_source(tmp_path):
    from pathlib import Path
    from picoagent.data.native_admission import combine_native_snapshots
    source = Path(__file__).resolve().parents[1] / 'data/native-sharded-pilot-v1/manifest.json'
    with pytest.raises(DataValidationError, match='explicit opt-in'):
        combine_native_snapshots([source], tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()
    with pytest.raises(DataValidationError, match='repeated native sources'):
        combine_native_snapshots([source, source], tmp_path / 'not-created', allow_native_teacher=True)
    assert not (tmp_path / 'not-created').exists()


def test_native_problem_origin_cannot_be_added_to_unreviewed_source():
    import copy
    from pathlib import Path
    from picoagent.data.native_admission import native_problem_identity
    from picoagent.data.native_storage import iter_rows
    root = Path(__file__).resolve().parents[1] / 'data/native-sharded-pilot-v1'
    manifest = json.loads((root / 'manifest.json').read_text())
    row = next(iter_rows(root / manifest['all_observations']['paths'][0]))
    assert native_problem_identity(row) == row['task_id']
    forged = copy.deepcopy(row)
    forged['provenance']['origin_base_task_id'] = 'unrelated-task'
    with pytest.raises(DataValidationError, match='canonical problem identity'):
        validate_trace(forged, allow_native_teacher=True)
    forged = copy.deepcopy(row)
    forged['provenance']['origin_base_task_sha256'] = '0' * 64
    with pytest.raises(DataValidationError, match='origin task hash'):
        validate_trace(forged, allow_native_teacher=True)


def test_native_selection_deduplicates_views_without_hiding_cross_split_overlap():
    from picoagent.data.native_admission import _select_native_success
    problems, conversations = {}, {}
    def row(task, split='train', text='same', origin=None):
        return {'task_id': task, 'split': split, 'messages': [{'role': 'user', 'content': text}],
                'provenance': {'origin_base_task_id': origin} if origin else {}}
    assert _select_native_success(row('a'), problems, conversations, deduplicate=True)
    assert not _select_native_success(row('b'), problems, conversations, deduplicate=True)
    assert not _select_native_success(row('c', text='variant', origin='a'), problems, conversations, deduplicate=True)
    assert _select_native_success(row('d', text='different'), problems, conversations, deduplicate=True)
    with pytest.raises(DataValidationError, match='cross-split native conversation'):
        _select_native_success(row('e', split='dev'), problems, conversations, deduplicate=True)
    with pytest.raises(DataValidationError, match='cross-split native canonical'):
        _select_native_success(row('f', split='dev', text='new', origin='a'), problems, conversations, deduplicate=True)


def test_actual_knowledge_store_bytes_are_bound_to_state_and_have_no_process_claims():
    import copy
    from pathlib import Path
    from picoagent.data.native_admission import extract_native_observation
    from picoagent.data.native_storage import iter_rows
    from picoagent.data.schema import content_hash
    root = Path(__file__).resolve().parents[1] / 'data/native-knowledge-search-pilot-v1'
    if not (root / 'manifest.json').exists():
        pytest.skip('optional actual knowledge/search pilot is not distributed')
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = list(iter_rows(root / manifest['all_observations']['paths'][0]))
    row = next(r for r in rows if r['family'] == 'kv.note_write_verify')
    validate_trace(row, allow_native_teacher=True)
    raw = row['native_evidence']['raw_record']
    assert raw['native_evidence']['initial_knowledge_file'] == {'exists': False}
    forged = copy.deepcopy(raw)
    forged['native_evidence']['initial_knowledge_file']['content_b64'] = ''
    forged['raw_attempt_sha256'] = content_hash(forged['native_evidence'])
    with pytest.raises(DataValidationError, match='missing native store'):
        extract_native_observation('native_knowledge_search', forged, row['native_evidence']['task'])
    forged = copy.deepcopy(raw)
    forged['native_evidence']['receipts'][0]['argv'] = ['invented-process']
    forged['raw_attempt_sha256'] = content_hash(forged['native_evidence'])
    with pytest.raises(DataValidationError, match='cannot claim subprocess'):
        extract_native_observation('native_knowledge_search', forged, row['native_evidence']['task'])
