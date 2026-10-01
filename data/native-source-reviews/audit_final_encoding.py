"""Read-only production-encoding audit of an already admitted immutable wrapper.

This is not an admission shortcut: the root builder has already run strict native
and artificial-view verification. This script independently binds every consumed
byte to that exact wrapper manifest, streams the sealed selection, replays event
contexts, and audits actual production token IDs/labels without retaining them.
No command, subprocess, network request, or oracle from a trace is executed.

Run from repository root with PYTHONPATH=src and the pinned local ML environment:
  .venv/bin/python data/native-source-reviews/audit_final_encoding.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from picoagent.data.artificial_plans import _apply, _inside, _notes, _review, validate_note_tokenizer  # noqa: E402
from picoagent.data.audit import file_hash  # noqa: E402
from picoagent.data.native_storage import iter_rows  # noqa: E402
from picoagent.data.schema import _validate_model_event_replay, content_hash  # noqa: E402
from picoagent.harness.protocol import END_MESSAGE, canonical_json, render_messages  # noqa: E402
from picoagent.training.encoding import IGNORE_INDEX, encode_trace, event_examples  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MANIFEST = Path('data/native-training-plans-v1/manifest.json')
EXPECTED = 'e5774693bc42815ab44858754303b3970c7e34f26a1140425dccdd5853ccfbd9'
OUTPUT = Path('data/native-source-reviews/native-training-plans-v1-encoding-audit.json')
MAX_LENGTH = 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_rows(path, prefix=''):
    """Mirror verified native loader projection and collection reference prefixes."""
    manifest = json.loads(path.read_text())
    root = path.parent
    if manifest['storage'] == 'sealed_native_collection_v1':
        for component in manifest['components']:
            relative = component['path']
            target = _inside(root, relative)
            require(file_hash(target) == component['sha256'], 'component identity changed')
            child_prefix = '/'.join(filter(None, (prefix, str(Path(relative).parent))))
            yield from selected_rows(target, child_prefix)
        return
    require(manifest['storage'] == 'gzip_sharded_v1', 'unsupported sealed storage')
    wanted = {}
    for split, descriptor in manifest['splits'].items():
        count = 0
        for relative in descriptor['index_paths']:
            for selection in iter_rows(_inside(root, relative)):
                key = (selection['path'], selection['row_index'])
                require(key not in wanted and selection['split'] == split, 'selection duplicate/split')
                wanted[key] = selection
                count += 1
        require(count == descriptor['records'], 'selection count changed')
    for relative in manifest['all_observations']['paths']:
        for index, row in enumerate(iter_rows(_inside(root, relative))):
            selection = wanted.pop((relative, index), None)
            if selection is None:
                continue
            require(content_hash(row) == selection['trace_sha256'], 'selected trace content changed')
            require(all(row[key] == selection[key] for key in ('split', 'trace_id', 'task_id')), 'selected trace identity changed')
            require(row['status'] == 'success' and row['verification']['passed'], 'selected unsuccessful record')
            _validate_model_event_replay(row)
            source = row['native_evidence']['source_id']
            light = {key: value for key, value in row.items() if key not in {'native_evidence', 'tool_events', 'effective_messages'}}
            reference = {'path': relative, 'row_index': index, 'trace_sha256': selection['trace_sha256']}
            if prefix:
                reference['path'] = prefix + '/' + relative
                reference['component_manifest'] = prefix + '/manifest.json'
            light['native_evidence_reference'] = reference
            yield source, light
    require(not wanted, 'selection references missing rows')


class ObservedTokenizer:
    """Capture the actual production call for an independent mask cross-check."""
    is_fast = True

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.last = None

    def __call__(self, text, **kwargs):
        require(kwargs == {'add_special_tokens': False, 'return_offsets_mapping': True, 'truncation': False}, 'production tokenization contract changed')
        result = self.tokenizer(text, **kwargs)
        self.last = (text, result)
        return result


class Stats:
    def __init__(self):
        self.tasks = 0
        self.annotated_tasks = 0
        self.events = Counter()
        self.tools = Counter()
        self.lengths = []
        self.targets = []
        self.prompts = []
        self.total_tokens = 0
        self.supervised = 0
        self.boundary_crossing_tokens_masked = 0
        self.manual_keep = Counter()

    def result(self, tool_names):
        def distribution(values):
            ordered = sorted(values)
            if not ordered:
                return {'minimum': 0, 'median': 0, 'p95': 0, 'maximum': 0}
            return {'minimum': ordered[0], 'median': ordered[len(ordered)//2], 'p95': ordered[math.ceil(len(ordered)*.95)-1], 'maximum': ordered[-1]}
        return {'tasks': self.tasks, 'annotated_tasks': self.annotated_tasks, 'examples': len(self.lengths),
                'total_tokens': self.total_tokens, 'supervised_tokens': self.supervised,
                'event_types': dict(sorted(self.events.items())), 'tool_calls': {key: self.tools[key] for key in sorted(tool_names)},
                'length': distribution(self.lengths), 'supervised_target_length': distribution(self.targets),
                'input_token_positions': distribution(self.prompts),
                'boundary_crossing_tokens_masked': self.boundary_crossing_tokens_masked,
                'manual_keep_decisions': dict(sorted(self.manual_keep.items()))}


def main():
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    require(not OUTPUT.exists(), 'audit output already exists; preserve previous results')
    require(file_hash(MANIFEST) == EXPECTED, 'wrapper differs from requested identity')
    manifest = json.loads(MANIFEST.read_text())
    root = MANIFEST.parent
    source_files = [Path(__file__), *map(Path, [
        'src/picoagent/training/encoding.py', 'src/picoagent/harness/protocol.py',
        'src/picoagent/data/artificial_plans.py', 'src/picoagent/data/native_storage.py',
        'src/picoagent/data/native_collection.py', 'src/picoagent/data/schema.py'])]
    source_hashes = {str(path): file_hash(path) for path in source_files}
    for relative, info in manifest['files'].items():
        target = _inside(root, relative)
        require(target.is_file() and target.stat().st_size == info['bytes'] and file_hash(target) == info['sha256'], 'wrapper physical evidence changed: ' + relative)
    print('All wrapper inventory bytes and hashes verified', flush=True)
    base = _inside(root, 'base/manifest.json')
    require(file_hash(base) == manifest['base_manifest_sha256'], 'base identity changed')
    tokenizer_dir = root / 'base/components/000/evidence/native_compaction/source/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    require(tokenizer.is_fast, 'fast tokenizer required')
    identity = manifest['tokenizer_identity']
    validate_note_tokenizer(MANIFEST, tokenizer, model_id=identity['model_id'], revision=identity['revision'], tokenizer_json_path=tokenizer_dir/'tokenizer.json')
    tokenizer_hashes = {str(path.relative_to(root)): file_hash(path) for path in sorted(tokenizer_dir.iterdir()) if path.is_file()}
    templates = json.loads((root/'templates.json').read_text())
    review = json.loads((root/'template_review.json').read_text())
    notes = _notes(root/'annotations.jsonl.gz')
    require(_review(templates, review, notes) == manifest['template_review_sha256'], 'review identity changed')
    by_task = defaultdict(list)
    for note in notes:
        by_task[note['task_id']].append(note)
    selected = set(manifest['selected_task_ids'])
    require(len(selected) == len(manifest['selected_task_ids']), 'duplicate annotated task')
    seen_notes, seen_selected = set(), set()
    adapter = ObservedTokenizer(tokenizer)
    groups = defaultdict(Stats)
    tool_names = {'bash', 'python', 'write_file', 'search', 'knowledge'}
    all_tasks, all_origins = set(), set()
    encoded_digest = hashlib.sha256()
    for source, original in selected_rows(base):
        split = original['split']
        task_id = original['task_id']
        require(split in {'train', 'dev'} and task_id not in all_tasks, 'duplicate/test task')
        all_tasks.add(task_id)
        origin = original['provenance'].get('origin_base_task_id', task_id)
        require(origin not in all_origins, 'duplicate canonical problem origin')
        all_origins.add(origin)
        record = original
        if task_id in by_task:
            require(split == 'train', 'dev annotation')
            record = _apply({'train': [original], 'dev': []}, by_task[task_id], templates,
                            [task_id] if task_id in selected else [],
                            note_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)))['train'][0]
            seen_notes.add(task_id)
        if task_id in selected:
            require('artificial_action_plan_view' in record, 'selected annotation absent')
            seen_selected.add(task_id)
        _validate_model_event_replay(record)
        mode = record['provenance'].get('context_mode', 'none')
        keys = [('all',), ('split', split), ('source', source), ('mode', mode), ('source_split_mode', source, split, mode)]
        stats = [groups[key] for key in keys]
        for stat in stats:
            stat.tasks += 1
            stat.annotated_tasks += task_id in selected
        tool_names.update(tool['function']['name'] for tool in record['tools'])
        events = [event for event in record['model_events'] if event['type'] == 'assistant' or (event['type'] == 'compaction' and event.get('accepted') is True)]
        examples = event_examples(record)
        require(len(events) == len(examples), 'production dropped a model event')
        transcript_calls = Counter(call['function']['name'] for message in record['messages'] if message['role'] == 'assistant' for call in message.get('tool_calls', []))
        event_calls = Counter()
        for event, (example, last_only) in zip(events, examples):
            require(last_only, 'exact-event last-only supervision required')
            output = event['message'] if event['type'] == 'assistant' else event['summary_response']
            inputs = event['input_messages'] if event['type'] == 'assistant' else event['summary_request']
            require(example['messages'] == inputs + [output], 'production altered event context/output')
            require(example['tools'] == (record['tools'] if event['type'] == 'assistant' else []), 'production changed tool schema')
            encoded = encode_trace(example, adapter, MAX_LENGTH, supervise_last_only=True)
            text, observed = adapter.last
            prefix = render_messages(inputs, tools=example['tools']) + 'ASSISTANT:\n'
            target = canonical_json(output) + END_MESSAGE
            require(text == prefix + target, 'production rendered text differs from exact event')
            ids = list(observed['input_ids'])
            offsets = list(observed['offset_mapping'])
            begin, end = len(prefix), len(text)
            expected_labels = [token if b > a and begin <= a and b <= end else IGNORE_INDEX for token, (a, b) in zip(ids, offsets)]
            require(encoded['input_ids'] == ids and encoded['labels'] == expected_labels, 'assistant-only mask mismatch')
            require(encoded['attention_mask'] == [1] * len(ids), 'unexpected attention mask')
            require(len(ids) <= MAX_LENGTH and any(label != IGNORE_INDEX for label in expected_labels[1:]), 'overflow/empty shifted target')
            require(offsets[-1][1] == len(text) and expected_labels[-1] == ids[-1], 'final response terminator clipped/masked')
            supervised = sum(label != IGNORE_INDEX for label in expected_labels)
            crossing = sum(a < begin < b for a, b in offsets)
            kind = event['type']
            if kind == 'assistant':
                kind = 'tool_action' if output.get('tool_calls') else 'final_answer'
            calls = Counter(call['function']['name'] for call in output.get('tool_calls', []))
            event_calls.update(calls)
            for stat in stats:
                stat.events[kind] += 1
                stat.tools.update(calls)
                stat.lengths.append(len(ids))
                stat.targets.append(supervised)
                stat.prompts.append(len(ids) - supervised)
                stat.total_tokens += len(ids)
                stat.supervised += supervised
                stat.boundary_crossing_tokens_masked += crossing
                if event['type'] == 'compaction' and event.get('mode', 'half') == 'manual':
                    stat.manual_keep['nonempty' if event['keep_group_indices'] else 'empty'] += 1
            encoded_digest.update((canonical_json({'trace_id': example['trace_id'], **encoded}) + '\n').encode())
        require(event_calls == transcript_calls, 'tool calls omitted/duplicated in event supervision')
        if len(all_tasks) % 2000 == 0:
            print(f'Audited {len(all_tasks)} tasks / {len(groups[("all",)].lengths)} events', flush=True)
    require(seen_notes == set(by_task) and seen_selected == selected, 'annotation coverage mismatch')
    require({split: groups[('split', split)].tasks for split in ('train', 'dev')} == manifest['counts'], 'selected counts mismatch')
    require(len(selected) == manifest['annotated_train_tasks'], 'annotated count mismatch')
    require(file_hash(MANIFEST) == EXPECTED, 'wrapper manifest changed during audit')
    require(all(file_hash(Path(name)) == digest for name, digest in source_hashes.items()), 'audit implementation changed during audit')
    report = {
        'schema': 'picoagent.artificial_action_plan.production_encoding_audit.v1',
        'passed': True, 'manifest': str(MANIFEST), 'manifest_sha256': EXPECTED,
        'base_manifest_sha256': manifest['base_manifest_sha256'], 'scope': 'encoding audit of previously strictly admitted immutable snapshot; native oracles were not rerun',
        'physical_inventory_files_checked': len(manifest['files']), 'physical_inventory_bytes_checked': sum(info['bytes'] for info in manifest['files'].values()),
        'tokenizer_identity': identity, 'tokenizer_files': tokenizer_hashes,
        'versions': {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')}},
        'execution': {'cpu_only': True, 'gpu_allocated': False, 'network_used': False, 'trace_commands_executed': False},
        'max_seq_length': MAX_LENGTH, 'no_truncation': True, 'mask_policy': 'only exact current assistant output; all prefix/input/tool schema/result tokens masked; crossing tokens masked',
        'every_event_replayed': True, 'every_label_independently_offset_checked': True,
        'every_final_terminator_present_and_supervised': True,
        'encoded_stream_sha256': encoded_digest.hexdigest(), 'encoded_stream_order': 'component order, normalized observation shard order, selected row order, model event order',
        'source_hashes': source_hashes, 'started_at': started_at,
        'completed_at': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': round(time.monotonic()-start, 3),
        'counts': manifest['counts'], 'annotated_train_tasks': len(selected), 'unique_canonical_problems': len(all_origins),
        'overall': groups[('all',)].result(tool_names),
        'splits': {split: groups[('split', split)].result(tool_names) for split in ('train', 'dev')},
        'sources': {key[1]: value.result(tool_names) for key, value in sorted(groups.items()) if key[0] == 'source'},
        'modes': {key[1]: value.result(tool_names) for key, value in sorted(groups.items()) if key[0] == 'mode'},
        'source_split_mode': [dict(source=key[1], split=key[2], mode=key[3], **value.result(tool_names)) for key, value in sorted(groups.items()) if key[0] == 'source_split_mode'],
    }
    with OUTPUT.open('x') as handle:
        handle.write(canonical_json(report)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({'output': str(OUTPUT), 'sha256': file_hash(OUTPUT), 'overall': report['overall'], 'elapsed_seconds': report['elapsed_seconds']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
