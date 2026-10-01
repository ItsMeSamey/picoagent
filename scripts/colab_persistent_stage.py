#!/usr/bin/env python3
"""Stage an already prepared archive using one official Colab CLI connection.

Uses existing CLI authentication/session only. No allocation, auth change, TLS
changes, keepalive, or training. A failed transfer is resumed by verified hashes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from colab_run import OUTPUT_SENTINEL, staging_call
from source_staging import matches, validate_bundle
from colab_safe_cli import invoke, report_error


def parse_result(outputs: list[dict]) -> dict:
    if any(item.get('output_type') == 'error' for item in outputs):
        raise RuntimeError('Remote staging execution failed; private output suppressed')
    text = ''.join(item.get('text', '') for item in outputs if item.get('output_type') == 'stream')
    values = [json.loads(line[len(OUTPUT_SENTINEL):]) for line in text.splitlines()
              if line.startswith(OUTPUT_SENTINEL)]
    if len(values) != 1 or not isinstance(values[0], dict):
        raise RuntimeError('Expected one structured remote staging result')
    return values[0]


class PersistentSession:
    def __init__(self, name: str):
        from colab_cli.common import state
        from colab_cli.runtime import ColabRuntime
        self.state, self.name = state, name
        session = state.get_session(name)
        def save_kernel(value):
            session.kernel_id = value
            state.store.add(session)
        def save_session(value):
            session.session_id = value
            state.store.add(session)
        self.runtime = ColabRuntime(session.url, session.token,
            kernel_id=session.kernel_id, session_id=session.session_id,
            on_kernel_started=save_kernel, on_session_started=save_session)

    def execute(self, code: str) -> dict:
        return parse_result(self.runtime.execute_code(code, timeout=120))

    def command(self, action, local, remote, *, capture=True):
        if action != 'upload':
            raise ValueError('Persistent source adapter supports upload only')
        from colab_cli.contents import ContentsClient
        ContentsClient(self.state.get_session(self.name)).upload(local, remote)


def stage_prepared(client, prepared: dict, *, project: str) -> dict:
    bundle = prepared['bundle']
    validate_bundle(bundle)
    if not matches(Path(prepared['source']['archive']), bundle['archive']):
        raise ValueError('Prepared archive bytes changed')
    remote = '/content/picoagent-source-uploads/' + bundle['archive']['sha256']
    verified = set(staging_call(client, 'probe_chunks', remote, bundle)['verified_chunks'])
    print(json.dumps({'phase':'upload', 'verified_chunks':len(verified), 'total_chunks':len(bundle['chunks'])}), flush=True)
    for chunk in bundle['chunks']:
        digest = chunk['sha256']
        if digest in verified:
            continue
        local = Path(prepared['chunks']) / digest
        if not matches(local, chunk):
            raise ValueError('Prepared source chunk changed')
        client.command('upload', str(local), (remote + '/' + digest + '.partial').lstrip('/'))
        receipt = staging_call(client, 'commit_chunk', remote, chunk)
        if receipt.get('verified_chunk') != digest:
            raise ValueError('Remote source acknowledgement mismatch')
        verified.add(digest)
        print(json.dumps({'verified_chunks':len(verified), 'total_chunks':len(bundle['chunks'])}), flush=True)
    return staging_call(client, 'materialize_source', remote, project, bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--project', default='/content/picoagent')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('Receipt path must be new; inspect previous transfer first')
    client = PersistentSession(args.session)
    result = stage_prepared(client, json.loads(args.prepared.read_text()), project=args.project)
    with args.receipt.open('x') as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({'phase':'source_ready', 'result':result}), flush=True)


if __name__ == '__main__':
    import websocket
    from colab_cli.runtime import ColabRuntime
    try:
        invoke(main, ColabRuntime, websocket.WebSocketApp)
    except Exception as error:
        report_error('persistent-stage', error)
        raise SystemExit(1) from None
