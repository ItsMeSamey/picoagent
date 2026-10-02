"""CPU-only controller/public-release barrier integration; all network is fake."""
import contextlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_run as controller
import colab_sdk_watch as watcher
import github_checkpoint_release as release
from checkpoint_sync import LocalTransfer
from test_github_checkpoint_release import prepared, FakeGitHub, FakePublicHTTP  # noqa: F401


class Controller:
    def __init__(self):
        self.calls = []

    def execute(self, code):
        self.calls.append(code)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return json.loads(output.getvalue().split(controller.OUTPUT_SENTINEL)[-1])

    def transfer(self):
        class Transfer(LocalTransfer):
            def download(self, remote, local):
                return super().download('/' + remote.lstrip('/'), local)
        return Transfer()


@pytest.fixture
def collection(prepared, monkeypatch, tmp_path):  # noqa: F811
    run, manifest, plan = prepared
    client, github = Controller(), FakeGitHub()
    original = release.upload_checkpoint
    monkeypatch.setattr(release, 'upload_checkpoint', lambda *args, **kwargs:
                        original(*args, **kwargs, client=github, public_opener=FakePublicHTTP(github)))
    destination = tmp_path / 'controller'

    def collect(**kwargs):
        return controller.collect(client, str(Path(__file__).resolve().parents[1]), str(run),
                                  str(manifest.parent.parent), destination, True,
                                  publish_repository='test-owner/test-repo', **kwargs)
    return collect, client, github, run, destination


def test_publication_requires_approval_before_any_write(collection):
    collect, client, github, run, destination = collection
    with pytest.raises(ValueError, match='approve-public-checkpoints'):
        collect()
    assert not client.calls and not github.writes and not destination.exists()


def test_failed_upload_does_not_acknowledge_or_delete(collection):
    collect, client, github, run, destination = collection
    github.fail_upload_number = 2
    with pytest.raises(ConnectionError):
        collect(approve_public_checkpoints=True)
    assert not (run / 'durability/checkpoint-7.json').exists()
    assert (run / 'checkpoint-7/model.safetensors').is_file()
    assert list(destination.glob('release-backups/checkpoint-7/*/plan.json'))
    assert not list(destination.glob('release-backups/checkpoint-7/*/receipt.json'))
    assert not any('prune_runtime_exports' in code for code in client.calls)


def test_valid_published_release_acknowledges_without_prune_and_retries(collection):
    collect, client, github, run, destination = collection
    result = collect(approve_public_checkpoints=True)
    ack_path = run / 'durability/checkpoint-7.json'
    ack = json.loads(ack_path.read_text())
    receipt_path = Path(result['release_receipts']['checkpoint-7'])
    assert ack['receipt'] == json.loads(receipt_path.read_text())
    assert ack['receipt']['published'] is True
    assert (receipt_path.parent / 'plan.json').is_file()
    cached_manifest = (destination / '.incoming/checkpoint-7' /
                       ack['receipt']['identity']['transfer_manifest_sha256'] / 'transfer_manifest.json')
    assert cached_manifest.is_file()
    assert not (cached_manifest.parent / 'chunks').exists()
    assert result['pruned_runtime'] == []
    assert (run / 'checkpoint-7/model.safetensors').is_file()
    writes = list(github.writes)
    reads = list(github.reads)
    collect(approve_public_checkpoints=True)
    assert github.writes == writes
    assert github.reads == reads
    assert json.loads(ack_path.read_text()) == ack


def test_interrupted_upload_retry_reuses_existing_assets(collection):
    collect, client, github, run, destination = collection
    github.fail_upload_number = 2
    with pytest.raises(ConnectionError):
        collect(approve_public_checkpoints=True)
    first = next(iter(github.records))
    github.fail_upload_number = None
    collect(approve_public_checkpoints=True)
    assert github.writes.count(('upload', first)) == 1
    assert (run / 'durability/checkpoint-7.json').is_file()


def test_watcher_passes_publication_on_final_collection(tmp_path):
    calls = []
    terminal = {'run_status': {'status': 'completed'}}
    watcher.watch(object(), project='/project', run_dir='/run', export_root='/exports',
                  destination=tmp_path, publish_repository='test-owner/test-repo',
                  approve_public_checkpoints=True,
                  collect_fn=lambda *args, **kwargs: calls.append(kwargs) or {},
                  status_fn=lambda *args: terminal, emit=lambda _: None)
    assert calls == [{'publish_repository': 'test-owner/test-repo',
                      'approve_public_checkpoints': True}] * 2


def test_unpublished_receipt_cannot_acknowledge(collection, monkeypatch):
    collect, client, github, run, destination = collection
    original = release.upload_checkpoint

    def unpublished(*args, **kwargs):
        receipt = original(*args, **kwargs)
        receipt['published'] = False
        return receipt

    monkeypatch.setattr(release, 'upload_checkpoint', unpublished)
    with pytest.raises(ValueError, match='published release'):
        collect(approve_public_checkpoints=True)
    assert not (run / 'durability/checkpoint-7.json').exists()
    assert not list(destination.glob('release-backups/checkpoint-7/*/receipt.json'))


def test_receipt_persistence_failure_does_not_acknowledge(collection, monkeypatch):
    from picoagent.training import provenance
    collect, client, github, run, destination = collection
    original = provenance.write_json

    def fail_receipt(path, payload, **kwargs):
        if path.name == 'receipt.json':
            raise OSError('fake full controller disk')
        return original(path, payload, **kwargs)

    monkeypatch.setattr(provenance, 'write_json', fail_receipt)
    with pytest.raises(OSError, match='full controller disk'):
        collect(approve_public_checkpoints=True)
    assert not (run / 'durability/checkpoint-7.json').exists()


def test_restarted_controller_reverifies_existing_receipt(collection):
    collect, client, github, run, destination = collection
    collect(approve_public_checkpoints=True)
    reads = len(github.reads)
    del client._confirmed_release_receipts
    collect(approve_public_checkpoints=True)
    assert len(github.reads) > reads


def test_missing_runtime_ack_reverifies_before_retry(collection):
    collect, client, github, run, destination = collection
    collect(approve_public_checkpoints=True)
    reads = len(github.reads)
    (run / 'durability/checkpoint-7.json').unlink()
    collect(approve_public_checkpoints=True)
    assert len(github.reads) > reads
    assert (run / 'durability/checkpoint-7.json').is_file()
