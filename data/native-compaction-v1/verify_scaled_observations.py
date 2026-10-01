"""Read-only verification of real native observations; no task commands execute."""
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

from picoagent.data.audit import file_hash, write_new_json
from picoagent.data.compaction_curriculum import GOAL_MARKER, visible_memory
from picoagent.data.native_compaction_curriculum import independent_oracle_check
from picoagent.data.schema import _validate_model_event_replay, canonical_json, content_hash

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'manifest.json').read_text())
base_modes = defaultdict(list)
base_splits = {}
preferred = defaultdict(Counter)
counts = Counter()
source_hashes = json.loads((root / 'source_snapshot/manifest.json').read_text())['source_sha256']
for name, digest in source_hashes.items():
    assert file_hash(root / 'source_snapshot' / name) == digest
for shard_name in manifest['observation_paths']:
    shard = root / shard_name
    descriptor = manifest['observation_files'][shard_name]
    assert file_hash(shard) == descriptor['sha256']
    assert shard.stat().st_size == descriptor['bytes'] < 25 * 1024 * 1024
    logical = hashlib.sha256()
    rows = 0
    with gzip.open(shard, 'rb') as stream:
        for line in stream:
            logical.update(line)
            raw = json.loads(line)
            rows += 1
            counts['observations'] += 1
            task = raw['task']
            task_id = raw['task_id']
            assert task_id == task['task_id'] == raw['base_task_id']
            assert raw['task_sha256'] == content_hash(task)
            assert raw['source_sha256'] == source_hashes
            assert raw['teacher_decision_mode'] == 'reviewed_procedural_replay'
            assert raw['teacher_model'] is None and raw['sft_admissible'] is False
            assert raw['status'] == 'observed_success' and raw['stop_reason'] == 'final'
            assert task['split'] in ('train', 'dev')
            if task_id not in base_modes:
                preferred[task['family']][raw['mode']] += 1
            base_modes[task_id].append(raw['mode'])
            base_splits[task_id] = task['split']
            _validate_model_event_replay(raw)
            assert independent_oracle_check(task, raw['final'])['passed']
            assert raw['fixture_sha256_before'] == raw['fixture_sha256_after']
            assert file_hash(root / raw['journal']['path']) == raw['journal']['file_sha256']
            goal = json.loads(task['prompt'].split(GOAL_MARKER, 1)[1])
            path = goal['initial_path']
            assert len(raw['receipts']) == len(raw['tool_events']) == goal['length']
            for index, (receipt, event) in enumerate(zip(raw['receipts'], raw['tool_events'])):
                counts['actual_cat_receipts'] += 1
                assert receipt['sequence'] == index and receipt['tool_call_id'] == f'record_{index}'
                assert receipt['tool_call_id'] == event['tool_call_id']
                assert receipt['result'] == event['result']
                assert receipt['arguments'] == event['arguments']
                command = 'cat -- ' + path
                assert json.loads(receipt['arguments']) == {'command': command}
                assert receipt['argv'] == [raw['runtime']['executables']['bash']['path'], '--noprofile', '--norc', '-c', command]
                assert receipt['exit_code'] == 0 and receipt['timed_out'] is False and receipt['truncated'] is False
                assert 'container_id' not in receipt['result'] and receipt['result']['backend'] == 'native_teacher'
                decoded = {}
                for name in ('stdin', 'stdout', 'stderr'):
                    data = base64.b64decode(receipt[name + '_b64'], validate=True)
                    assert len(data) == receipt[name + '_bytes']
                    assert hashlib.sha256(data).hexdigest() == receipt[name + '_sha256']
                    decoded[name] = data
                assert decoded['stdin'] == decoded['stderr'] == b''
                expected = task['environment']['files'][path].encode()
                assert decoded['stdout'] == expected
                assert file_hash(root / raw['attempt_path'] / 'workspace' / path) == hashlib.sha256(expected).hexdigest()
                packet = json.loads(decoded['stdout'])
                assert packet['index'] == index and packet['task_id'] == task_id
                path = packet['next']
            assert path is None
            for event in raw['model_events']:
                if event['type'] == 'compaction':
                    assert event['accepted'] and event['mode'] == raw['mode']
                    assert visible_memory(event['before_messages']) == visible_memory(event['result_messages'])
                    counts['accepted_compactions'] += 1
                    counts['compactions_' + raw['mode']] += 1
    assert rows == descriptor['records']
    assert logical.hexdigest() == descriptor['logical_sha256']
assert len(base_modes) == 216 and counts['observations'] == 648
assert all(sorted(values) == ['full', 'half', 'manual'] for values in base_modes.values())
assert Counter(base_splits.values()) == {'train': 192, 'dev': 24}
assert all(len(set(values.values())) == 1 and set(values) == {'full','half','manual'} for values in preferred.values())
report = {'schema': 'picoagent.native_compaction.scaled_integrity.v1', 'passed': True,
          'execution_kind': 'native_teacher_observed', 'sft_admissible': False,
          'source_manifest_sha256': file_hash(root / 'manifest.json'),
          'verifier_sha256': file_hash(__file__), 'counts': dict(counts),
          'unique_base_tasks': len(base_modes), 'base_splits': dict(Counter(base_splits.values())),
          'preferred_modes_by_family': {key: dict(value) for key,value in preferred.items()},
          'checks': ['all_shard_hashes_and_logical_hashes', 'all_frozen_source_hashes',
                     'all_real_subprocess_byte_receipts', 'read_order_and_exact_frozen_stdout',
                     'fixture_disk_bytes_before_after', 'exact_model_event_replay',
                     'compaction_preserves_observed_essentials', 'independent_fixture_oracle',
                     'all_compressed_journal_file_hashes', 'mode_balance_without_task_inflation']}
write_new_json(root / 'integrity_report.json', report)
print(canonical_json(report))
