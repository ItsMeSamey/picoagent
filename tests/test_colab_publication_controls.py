"""Local-only publication concurrency plumbing and graceful controller stopping."""
import json
from pathlib import Path
import sys

import pytest

from test_release_collector import (
    collection as collection, controller, prepared as prepared, release, watcher,
)


@pytest.mark.parametrize("workers", [1, 4])
def test_collect_forwards_only_explicit_parallel_workers(collection, monkeypatch, workers):
    collect, client, github, run, destination = collection
    original = release.upload_checkpoint
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(release, 'upload_checkpoint', record)
    result = collect(approve_public_checkpoints=True, upload_workers=workers)
    assert len(calls) == 1
    if workers == 1:
        assert 'upload_workers' not in calls[0]
    else:
        assert calls[0]['upload_workers'] == workers
    ack = json.loads((run / 'durability/checkpoint-7.json').read_text())
    assert ack['receipt'] == json.loads(Path(result['release_receipts']['checkpoint-7']).read_text())
    assert ack['receipt']['published']


@pytest.mark.parametrize('workers', [0, 5, True, '4', None])
def test_collect_bad_workers_rejected_before_remote_access(collection, workers):
    collect, client, github, run, destination = collection
    with pytest.raises(ValueError, match='upload_workers'):
        collect(approve_public_checkpoints=True, upload_workers=workers)
    assert client.calls == github.writes == []
    assert not destination.exists()


def test_nondefault_workers_require_repository_before_collect_or_watch(tmp_path):
    class NoRemote:
        def execute(self, code):
            pytest.fail('Must validate before remote execution')

    with pytest.raises(ValueError, match='publish-repository'):
        controller.collect(NoRemote(), '/project', '/run', '/export', tmp_path, True, upload_workers=4)
    with pytest.raises(ValueError, match='publish-repository'):
        watcher.watch(NoRemote(), project='/project', run_dir='/run', export_root='/export',
                      destination=tmp_path, upload_workers=4,
                      collect_fn=lambda *a, **k: pytest.fail('Must validate before collection'))


def test_watch_forwards_workers_on_initial_and_terminal_collection(tmp_path):
    calls = []
    assert watcher.watch(
        object(), project='/project', run_dir='/run', export_root='/export', destination=tmp_path,
        publish_repository='test-owner/test-repo', approve_public_checkpoints=True, upload_workers=4,
        collect_fn=lambda *args, **kwargs: calls.append(kwargs) or {},
        status_fn=lambda *args: {'run_status': {'status': 'completed'}}, emit=lambda _: None,
    ) == 0
    assert calls == [{'publish_repository': 'test-owner/test-repo',
                      'approve_public_checkpoints': True, 'upload_workers': 4}] * 2


@pytest.mark.parametrize('terminal_collection', [False, True])
def test_stop_marker_waits_for_complete_collect_before_stopping(tmp_path, terminal_collection):
    marker = tmp_path / 'stop-controller'
    events, output = [], []
    target = 2 if terminal_collection else 1
    calls = 0

    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == target:
            marker.touch()
            assert not any(item.get('controller_stopped') for item in output)
        events.extend(['upload_finished', 'receipt_persisted', 'acknowledged'])
        return {'verified_checkpoints': ['checkpoint-7']}

    def status(*args):
        assert terminal_collection
        return {'run_status': {'status': 'completed'}}

    assert watcher.watch(object(), project='/project', run_dir='/run', export_root='/exports',
                         destination=tmp_path, collect_fn=collect, status_fn=status,
                         sleep_fn=lambda _: pytest.fail('No sleep after stop marker'),
                         emit=lambda value: output.append(json.loads(value)), stop_file=marker) == 0
    assert calls == target and events[-1] == 'acknowledged'
    assert output[-1] == {'controller_stopped': True, 'reason': 'stop_file',
                          'collection_completed': True, 'stop_file': str(marker)}
    assert marker.exists(), 'Never delete operator-owned markers'


def test_collect_error_does_not_report_safe_stop(tmp_path):
    marker = tmp_path / 'stop-controller'
    marker.touch()
    output = []

    def failed(*args, **kwargs):
        raise ConnectionError('publication incomplete')

    with pytest.raises(ConnectionError, match='incomplete'):
        watcher.watch(object(), project='/project', run_dir='/run', export_root='/export',
                      destination=tmp_path, collect_fn=failed, stop_file=marker, emit=output.append)
    assert output == []


@pytest.mark.parametrize('module', [controller, watcher])
@pytest.mark.parametrize('workers', [1, 4])
def test_cli_forwards_workers_and_stop_marker_releases_lock(tmp_path, monkeypatch, module, workers):
    from picoagent.training.retention import _exclusive_lock
    destination = tmp_path / 'archive'
    marker = tmp_path / 'stop-controller'
    calls = []
    monkeypatch.setattr(module, 'SDKController' if module is watcher else 'Colab', lambda *a, **k: object())

    def collect(*args, **kwargs):
        calls.append(kwargs)
        marker.touch()
        return {'verified_checkpoints': ['checkpoint-7']}

    monkeypatch.setattr(module, 'collect', collect)
    monkeypatch.setattr(module, 'status', lambda *a: pytest.fail('Must stop at collection boundary'))
    argv = ['watch.py', '--session', 'existing']
    if module is controller:
        argv.append('watch')
    argv += ['--destination', str(destination), '--run-dir', '/run', '--off-runtime',
             '--publish-repository', 'test-owner/test-repo', '--approve-public-checkpoints',
             '--upload-workers', str(workers), '--stop-file', str(marker)]
    monkeypatch.setattr(sys, 'argv', argv)
    module.main()
    assert len(calls) == 1
    if workers == 1:
        assert 'upload_workers' not in calls[0]
    else:
        assert calls[0]['upload_workers'] == 4
    with _exclusive_lock(destination):
        pass  # Normal stop has actually released the controller's lock.


@pytest.mark.parametrize('module', [controller, watcher])
def test_cli_parallel_without_repository_rejected_before_client(tmp_path, monkeypatch, module):
    monkeypatch.setattr(module, 'SDKController' if module is watcher else 'Colab',
                        lambda *a, **k: pytest.fail('No client before validation'))
    argv = ['watch.py', '--session', 'existing']
    if module is controller:
        argv.append('collect')
    argv += ['--destination', str(tmp_path), '--run-dir', '/run', '--off-runtime', '--upload-workers', '4']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='publish-repository'):
        module.main()
