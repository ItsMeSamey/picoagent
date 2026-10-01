"""Immutable collections of already sealed native datasets, without row rewrites."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .audit import file_hash, write_new_json
from .native_storage import MAX_FILE_BYTES

STORAGE = 'sealed_native_collection_v1'


def _aggregate(components: list[tuple[dict, dict[str, list[dict]], str]]) -> tuple[dict, dict[str, list[dict]]]:
    from .native_admission import _need, native_problem_identity, selected_compaction_modes
    from picoagent.training.data import _check_disjoint
    records: dict[str, list[dict]] = {'train': [], 'dev': []}
    source_counts: dict[str, dict] = {'observed': {}, 'admitted': {}}
    origins: set[str] = set()
    for manifest, rows, prefix in components:
        sources = set(manifest['source_counts']['observed'])
        _need(not sources.intersection(source_counts['observed']), 'native collection repeats a source identity/version')
        for kind in source_counts:
            source_counts[kind].update(copy.deepcopy(manifest['source_counts'][kind]))
        for split in ('train', 'dev'):
            for original in rows[split]:
                identity = native_problem_identity(original)
                _need(identity not in origins, 'native collection repeats a canonical underlying problem')
                origins.add(identity)
                row = dict(original)
                if 'native_evidence_reference' in row:
                    reference = dict(row['native_evidence_reference'])
                    reference['path'] = prefix + '/' + reference['path']
                    reference['component_manifest'] = prefix + '/manifest.json'
                    row['native_evidence_reference'] = reference
                records[split].append(row)
    _check_disjoint(records)
    return {'source_counts': source_counts, 'selected_compaction_modes': selected_compaction_modes(records),
            'splits': {split: {'records': len(rows), 'families': sorted({row['family'] for row in rows}),
                               'templates': sorted({row['template_id'] for row in rows})}
                       for split, rows in records.items()}}, records


def seal_native_collection(component_manifests: list[str | Path], destination: str | Path, *,
                           allow_native_teacher: bool = False) -> Path:
    """Strictly verify each input once, then byte-copy its complete sealed tree.

    Components must have disjoint selected tasks, canonical problem origins,
    normalized conversations, and train/dev families/templates. The collection
    never selects new variants, rewrites a raw trace, or enables host inference.
    """
    from . import native_admission as admission
    admission._need(allow_native_teacher is True, 'native collection creation requires explicit opt-in')
    admission._need(bool(component_manifests), 'native collection requires components')
    verified = []
    for original in component_manifests:
        path = Path(original).resolve()
        digest = file_hash(path)
        manifest, records = admission.verify_native_snapshot(path, allow_native_teacher=True)
        admission._need(file_hash(path) == digest, 'native component manifest changed during verification')
        admission._need(path.stat().st_size <= MAX_FILE_BYTES and all(info['bytes'] <= MAX_FILE_BYTES for info in manifest['files'].values()),
                        'native collection components must already use bounded files')
        verified.append((path, digest, manifest, records))
    aggregate, _ = _aggregate([(m, r, f'components/{i:03d}') for i, (_, _, m, r) in enumerate(verified)])
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    descriptors, inventory = [], {}
    for index, (original, digest, manifest, _) in enumerate(verified):
        prefix = f'components/{index:03d}'
        target = admission._copy_verified_native_snapshot(original, root / prefix,
                    verified_manifest=manifest, expected_manifest_sha256=digest)
        relative = prefix + '/manifest.json'
        descriptors.append({'path': relative, 'sha256': digest, 'name': original.parent.name})
        inventory[relative] = {'sha256': digest, 'bytes': target.stat().st_size}
        for name, info in manifest['files'].items():
            inventory[prefix + '/' + name] = {'sha256': info['sha256'], 'bytes': info['bytes']}
    manifest = {'schema': admission.NATIVE_MANIFEST_SCHEMA, 'storage': STORAGE,
                'admission': 'audited_native_teacher_observed_only', 'teacher_mode': admission.TEACHER_MODE,
                'lockbox_used': False, 'container_semantic_replay': 'not_verified',
                'arbitrary_learner_execution_allowed': False, 'components': descriptors,
                'files': dict(sorted(inventory.items())), **aggregate,
                'selection': 'unchanged component selections; global duplicate rejection',
                'limitations': ['Reviewed procedural native observations; no sampled-model or container-parity claim.',
                                'Each component is preserved byte-for-byte; the collection creates no new observations.']}
    write_new_json(root / 'manifest.json', manifest)
    os.chmod(root / 'manifest.json', 0o444)
    # Each component was strictly verified above, then every copied source and
    # destination byte/hash was checked. Repeating semantic audits here would
    # add no assurance. Standalone verification below is always strict.
    return root / 'manifest.json'


def verify_native_collection(path: Path, manifest: dict[str, Any]) -> tuple[dict, dict[str, list[dict]]]:
    from . import native_admission as admission
    root = path.parent
    admission._need(manifest.get('schema') == admission.NATIVE_MANIFEST_SCHEMA and manifest.get('storage') == STORAGE,
                    'unknown native collection format')
    admission._need(manifest.get('admission') == 'audited_native_teacher_observed_only' and manifest.get('lockbox_used') is False
                    and manifest.get('teacher_mode') == admission.TEACHER_MODE
                    and manifest.get('container_semantic_replay') == 'not_verified'
                    and manifest.get('arbitrary_learner_execution_allowed') is False, 'invalid native collection admission policy')
    admission._need(set(manifest.get('splits', {})) == {'train', 'dev'}, 'native collection requires train/dev only')
    descriptors, inventory = manifest.get('components'), manifest.get('files')
    admission._need(isinstance(descriptors, list) and bool(descriptors) and isinstance(inventory, dict), 'native collection lacks component inventory')
    for name, info in inventory.items():
        target = admission._inside(root, name)
        admission._need(set(info) == {'sha256', 'bytes'} and target.is_file() and type(info['bytes']) is int
                        and target.stat().st_size == info['bytes'] <= MAX_FILE_BYTES and file_hash(target) == info['sha256'],
                        f'native collection file integrity mismatch: {name}')
    components, expected_inventory, seen_paths = [], {}, set()
    for descriptor in descriptors:
        relative = descriptor.get('path')
        admission._need(isinstance(relative, str) and relative.endswith('/manifest.json') and relative not in seen_paths,
                        'invalid or repeated native collection component path')
        seen_paths.add(relative)
        component = admission._inside(root, relative)
        admission._need(relative in inventory and file_hash(component) == descriptor.get('sha256'), 'native component manifest hash mismatch')
        child, rows = admission.verify_native_snapshot(component, allow_native_teacher=True)
        prefix = str(Path(relative).parent)
        components.append((child, rows, prefix))
        expected_inventory[relative] = {'sha256': descriptor['sha256'], 'bytes': component.stat().st_size}
        for name, info in child['files'].items():
            key = prefix + '/' + name
            admission._need(key not in expected_inventory, 'native collection components overlap files')
            expected_inventory[key] = {'sha256': info['sha256'], 'bytes': info['bytes']}
    admission._need(inventory == expected_inventory, 'native collection inventory omits or adds component evidence')
    aggregate, records = _aggregate(components)
    for key, expected in aggregate.items():
        admission._need(manifest.get(key) == expected, f'native collection {key} does not match verified components')
    return manifest, records
