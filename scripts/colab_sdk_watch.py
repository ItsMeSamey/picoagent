#!/usr/bin/env python3
"""Collect verified checkpoints using the official CLI's existing session SDK.

One process avoids reimporting the CLI for every bounded file chunk. Integrity,
atomic publication and retention remain in the existing checkpoint workflow.
No allocation, authentication grant, TLS/proxy changes or teacher execution.
"""
import argparse
import json
from pathlib import Path
import threading
import time

from checkpoint_sync import MAX_DOWNLOAD_WORKERS, validate_download_workers
from colab_persistent_stage import PersistentSession, parse_result
from colab_run import collect, status, validate_publication_options
from colab_safe_cli import (invoke, report_error, safe_http_status,
                            retryable_transport_error as _retryable_transport_error)


class SDKTransfer:
    def __init__(self, state, session_name, *, download_workers=1):
        self.state, self.session_name = state, session_name
        self.download_workers = validate_download_workers(download_workers)
        self.downloads = self.bytes = 0
        self._state_lock = threading.Lock()
        self._progress_lock = threading.Lock()

    def download(self, remote, local):
        from colab_cli.contents import ContentsClient
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        # Keep the official existing-session lookup/refresh serialized. Each
        # concurrent transfer gets its own unmodified official ContentsClient.
        with self._state_lock:
            client = ContentsClient(self.state.get_session(self.session_name))
        client.download(remote, str(local))
        with self._progress_lock:
            self.downloads += 1
            self.bytes += local.stat().st_size
            if self.downloads == 1 or self.downloads % 8 == 0:
                print(json.dumps({'downloaded_files':self.downloads, 'downloaded_bytes':self.bytes}), flush=True)

    def upload(self, local, remote):
        raise ValueError('Checkpoint collector is download-only')


class SDKController(PersistentSession):
    def __init__(self, name, *, download_workers=1):
        validate_download_workers(download_workers)
        super().__init__(name)
        self._transfer = SDKTransfer(self.state, name, download_workers=download_workers)

    def execute(self, code):
        return parse_result(self.runtime.execute_code(code, timeout=600))

    def transfer(self):
        return self._transfer


def _terminal_status(snapshot):
    return any((snapshot.get(key) or {}).get('status') in {'completed', 'failed'}
               for key in ('run_status', 'job_status'))


def _training_exit_code(snapshot):
    run_status = snapshot.get('run_status') or {}
    job_status = snapshot.get('job_status') or {}
    if run_status.get('status') != 'failed' and job_status.get('status') != 'failed':
        return 0
    returncode = job_status.get('returncode') if job_status.get('status') == 'failed' else None
    if type(returncode) is not int or returncode == 0:
        return 1
    if returncode < 0:
        return min(255, 128 + abs(returncode))
    return min(255, returncode)


def watch(client, *, project, run_dir, export_root, destination, prune=False,
          interval=120, collect_fn=None, status_fn=None, sleep_fn=None, emit=None,
          publish_repository=None, approve_public_checkpoints=False, prune_local_published_cache=False,
          upload_workers=1, stop_file=None, retry_max_seconds=300,
          prune_to_latest_published=False):
    """Collect until terminal status, then collect once more before returning.

    A checkpoint can be sealed after an earlier collection but before the
    supervisor records completion. The terminal-triggered collection closes
    that race; the following status read ensures the terminal outcome is still
    current before deciding whether training succeeded.
    """
    validate_publication_options(publish_repository, approve_public_checkpoints, prune_local_published_cache, upload_workers)
    if type(retry_max_seconds) not in (int, float) or not 10 <= retry_max_seconds <= 3600:
        raise ValueError('retry_max_seconds must be between 10 and 3600')
    if prune_to_latest_published and (not prune or publish_repository is None):
        raise ValueError('--prune-to-latest-published requires --prune and --publish-repository')
    publication = ({"publish_repository": publish_repository,
                    "approve_public_checkpoints": approve_public_checkpoints}
                   if publish_repository is not None else {})
    if prune_to_latest_published:
        publication['prune_to_latest_published'] = True
    if upload_workers != 1:
        publication["upload_workers"] = upload_workers
    if prune_local_published_cache:
        publication["prune_local_published_cache"] = True
    collect_fn = collect if collect_fn is None else collect_fn
    status_fn = status if status_fn is None else status_fn
    sleep_fn = time.sleep if sleep_fn is None else sleep_fn
    emit = (lambda payload: print(payload, flush=True)) if emit is None else emit
    def stop_requested():
        if stop_file is not None and Path(stop_file).exists():
            emit(json.dumps({"controller_stopped": True, "reason": "stop_file",
                             "collection_completed": True, "stop_file": str(stop_file)}))
            return True
        return False

    def retry(operation, phase):
        delay = min(10, retry_max_seconds)
        while True:
            try:
                return operation()
            except Exception as error:
                if not _retryable_transport_error(error):
                    raise
                # A stop requested during failure must not imply a completed
                # collection or acknowledgement. Leave resumable chunks intact.
                if stop_file is not None and Path(stop_file).exists():
                    raise
                emit(json.dumps({'controller_retry': True, 'phase': phase,
                                 'error_type': type(error).__name__, 'retry_seconds': delay,
                                 'http_status': safe_http_status(error),
                                 'adapter_phase': (getattr(error, 'phase', None) if
                                     getattr(error, 'phase', None) in
                                     {'reconnect', 'refresh', 'execute', 'download'} else None)}))
                sleep_fn(delay)
                delay = min(retry_max_seconds, delay * 2)

    def collect_once():
        return retry(lambda: collect_fn(client, project, run_dir, export_root, destination,
                                       True, prune, **publication), 'collection')

    def status_once():
        return retry(lambda: status_fn(client, project, run_dir), 'status')

    while True:
        result = collect_once()
        emit(json.dumps(result))
        if stop_requested():
            return 0
        current = status_once()
        emit(json.dumps(current))
        if _terminal_status(current):
            final_result = collect_once()
            emit(json.dumps(final_result))
            if stop_requested():
                return 0
            final_status = status_once()
            emit(json.dumps(final_status))
            if _terminal_status(final_status):
                return _training_exit_code(final_status)
        sleep_fn(max(10, interval))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--project', default='/content/picoagent')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--export-root', default='/content/picoagent-checkpoint-exports')
    parser.add_argument('--off-runtime', action='store_true', required=True)
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune-to-latest-published', action='store_true',
                        help='Retain only latest runtime checkpoint after verified public publication')
    parser.add_argument('--retry-max-seconds', type=int, default=300,
                        help='Maximum transient transport retry delay (10–3600 seconds)')
    parser.add_argument('--prune-local-published-cache', action='store_true',
                        help='Approve eviction of older published local checkpoint payloads; retain latest')
    parser.add_argument('--publish-repository', help='Publish exact verified checkpoints to public OWNER/REPO')
    parser.add_argument('--approve-public-checkpoints', action='store_true',
                        help='Approve public disclosure of checkpoints and pinned run/source data')
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--download-workers', type=int, default=1,
                        choices=range(1, MAX_DOWNLOAD_WORKERS + 1),
                        help='Bounded concurrent checkpoint chunk downloads (default: serial)')
    parser.add_argument('--upload-workers', type=int, choices=range(1, 5), default=1,
                        help='Bounded GitHub upload/read-back concurrency (default: serial)')
    parser.add_argument('--stop-file', type=Path,
                        help='Stop controller after a completed collection if this local file exists')
    args = parser.parse_args()
    validate_publication_options(args.publish_repository, args.approve_public_checkpoints, args.prune_local_published_cache, args.upload_workers)
    from picoagent.training.retention import _exclusive_lock
    args.destination.mkdir(parents=True, exist_ok=True)
    client = SDKController(args.session, download_workers=args.download_workers)
    with _exclusive_lock(args.destination):
        exit_code = watch(client, project=args.project, run_dir=args.run_dir,
                          export_root=args.export_root, destination=args.destination,
                          prune=args.prune, interval=args.interval,
                          prune_to_latest_published=args.prune_to_latest_published,
                          retry_max_seconds=args.retry_max_seconds,
                          publish_repository=args.publish_repository,
                          approve_public_checkpoints=args.approve_public_checkpoints,
                          prune_local_published_cache=args.prune_local_published_cache,
                          **({"upload_workers": args.upload_workers} if args.upload_workers != 1 else {}),
                          **({"stop_file": args.stop_file} if args.stop_file is not None else {}))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == '__main__':
    import websocket
    from colab_cli.runtime import ColabRuntime
    try:
        invoke(main, ColabRuntime, websocket.WebSocketApp)
    except Exception as error:
        report_error('sdk-checkpoint-watch', error)
        raise SystemExit(1) from None
