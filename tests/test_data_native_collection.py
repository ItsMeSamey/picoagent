"""Read/copy actual sealed evidence; never execute recorded teacher actions."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from picoagent.data import native_admission as admission
from picoagent.data.audit import file_hash
from picoagent.data.native_collection import STORAGE, _aggregate, seal_native_collection
from picoagent.data.schema import DataValidationError

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'data/native-sharded-pilot-v1/manifest.json'
KNOWLEDGE = ROOT / 'data/native-knowledge-search-pilot-v1/manifest.json'


def test_private_verified_copy_checks_bytes_without_repeating_semantics(tmp_path, monkeypatch):
    digest = file_hash(CLI)
    manifest, _ = admission.verify_native_snapshot(CLI, allow_native_teacher=True)
    with monkeypatch.context() as patch:
        patch.setattr(admission, 'verify_native_snapshot', lambda *a, **k: pytest.fail('semantic verification repeated'))
        copied = admission._copy_verified_native_snapshot(CLI, tmp_path / 'copy', verified_manifest=manifest,
                                                          expected_manifest_sha256=digest)
    assert copied.read_bytes() == CLI.read_bytes()
    for relative, info in manifest['files'].items():
        assert file_hash(copied.parent / relative) == info['sha256']
    admission.verify_native_snapshot(copied, allow_native_teacher=True)
    victim = copied.parent / manifest['all_observations']['paths'][0]
    victim.chmod(0o644)
    victim.write_bytes(victim.read_bytes() + b'corruption')
    with pytest.raises(DataValidationError, match='source integrity mismatch'):
        admission._copy_verified_native_snapshot(copied, tmp_path / 'bad-copy', verified_manifest=manifest,
                                                  expected_manifest_sha256=digest)
    assert not (tmp_path / 'bad-copy/manifest.json').exists()


def test_private_copy_requires_original_manifest_object_and_hash(tmp_path):
    manifest, _ = admission.verify_native_snapshot(CLI, allow_native_teacher=True)
    with pytest.raises(DataValidationError, match='changed since validation'):
        admission._copy_verified_native_snapshot(CLI, tmp_path / 'stale', verified_manifest=manifest,
                                                  expected_manifest_sha256='0' * 64)
    forged = copy.deepcopy(manifest)
    forged['lockbox_used'] = True
    with pytest.raises(DataValidationError, match='prior validated object'):
        admission._copy_verified_native_snapshot(CLI, tmp_path / 'forged', verified_manifest=forged,
                                                  expected_manifest_sha256=file_hash(CLI))
    assert not (tmp_path / 'stale').exists() and not (tmp_path / 'forged').exists()


def test_collection_preserves_components_and_checks_global_inventory(tmp_path):
    if not KNOWLEDGE.exists():
        pytest.skip('optional actual knowledge/search pilot is not distributed')
    with pytest.raises(DataValidationError, match='explicit opt-in'):
        seal_native_collection([CLI, KNOWLEDGE], tmp_path / 'collection')
    output = seal_native_collection([CLI, KNOWLEDGE], tmp_path / 'collection', allow_native_teacher=True)
    with pytest.raises(DataValidationError, match='explicit opt-in'):
        admission.verify_native_snapshot(output)
    manifest, rows = admission.verify_native_snapshot(output, allow_native_teacher=True)
    assert manifest['storage'] == STORAGE
    assert {k: len(v) for k, v in rows.items()} == {'train': 9, 'dev': 5}
    for i, original in enumerate((CLI, KNOWLEDGE)):
        assert (output.parent / f'components/{i:03d}/manifest.json').read_bytes() == original.read_bytes()
        source = json.loads(original.read_text())
        for relative, info in source['files'].items():
            assert manifest['files'][f'components/{i:03d}/{relative}'] == {'sha256': info['sha256'], 'bytes': info['bytes']}
    assert all(r['native_evidence_reference']['path'].startswith('components/') for group in rows.values() for r in group)
    forged = copy.deepcopy(manifest)
    forged['splits']['train']['records'] += 1
    output.chmod(0o644)
    output.write_text(json.dumps(forged))
    with pytest.raises(DataValidationError, match='splits does not match'):
        admission.verify_native_snapshot(output, allow_native_teacher=True)


def test_collection_rejects_repeated_source_before_creation(tmp_path):
    with pytest.raises(DataValidationError, match='repeats a source'):
        seal_native_collection([CLI, CLI], tmp_path / 'duplicate', allow_native_teacher=True)
    assert not (tmp_path / 'duplicate').exists()


def test_collection_global_origin_and_conversation_guards():
    def component(source, task, text, origin=None):
        row = {'trace_id': task, 'task_id': task, 'template_id': task, 'family': task,
               'provenance': {'origin_base_task_id': origin} if origin else {},
               'messages': [{'role': 'user', 'content': text}]}
        return ({'source_counts': {'observed': {source: {}}, 'admitted': {source: {}}}},
                {'train': [row], 'dev': []}, source)
    with pytest.raises(DataValidationError, match='canonical underlying problem'):
        _aggregate([component('one', 'a', 'first'), component('two', 'b', 'variant', origin='a')])
    with pytest.raises(ValueError, match='Duplicate conversation'):
        _aggregate([component('one', 'a', 'same'), component('two', 'b', 'same')])
