"""No live cache deletion or network: tiny CPU-only publication fixtures."""
# ruff: noqa: F811
import copy
import json
import shutil

import pytest

from test_release_collector import collection  # noqa: F401
from test_github_checkpoint_release import prepared, FakeGitHub, FakePublicHTTP  # noqa: F401
import colab_run as controller
import local_checkpoint_cache as cache
from picoagent.training.provenance import checkpoint_evidence
from checkpoint_sync import sha256


def advance(collection):
    collect, client, github, run, destination = collection
    collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    old_github = copy.deepcopy(github)
    github.__dict__.update(FakeGitHub().__dict__)
    new = run / 'checkpoint-8'
    shutil.copytree(run / 'checkpoint-7', new)
    (new / 'trainer_state.json').write_text('{"global_step":8,"log_history":[]}')
    (new / 'checkpoint_manifest.json').unlink()
    checkpoint_evidence(new, sha256(run / 'run_manifest.json'))
    return old_github


def test_requires_explicit_publication():
    with pytest.raises(ValueError, match='requires'):
        controller.validate_publication_options(None, False, True)


def test_eviction_preserves_metadata_partials_and_skips_redownload(collection, monkeypatch):
    collect, client, github, run, destination = collection
    advance(collection)
    partial = destination / '.incoming/unique.partial'
    partial.write_bytes(b'unique interrupted payload')
    result = collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert result['pruned_local_published_cache'] == ['checkpoint-7']
    assert not (destination / 'checkpoint-7/model.safetensors').exists()
    assert (destination / 'checkpoint-7/checkpoint_manifest.json').exists()
    assert (destination / 'checkpoint-7/trainer_state.json').exists()
    assert (destination / 'checkpoint-8/model.safetensors').exists()
    assert partial.read_bytes() == b'unique interrupted payload'
    assert list(destination.glob('receipts/*.json'))
    original = controller.pull_checkpoint
    def pull(remote, *args, **kwargs):
        assert not remote.endswith('checkpoint-7'), 'Evicted checkpoint must not be downloaded again'
        return original(remote, *args, **kwargs)
    monkeypatch.setattr(controller, 'pull_checkpoint', pull)
    assert collect(approve_public_checkpoints=True, prune_local_published_cache=True)['pruned_local_published_cache'] == []


def test_upload_failure_keeps_previous_complete_cache(collection):
    collect, client, github, run, destination = collection
    advance(collection)
    github.fail_upload_number = 2
    with pytest.raises(ConnectionError):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert (destination / 'checkpoint-7/model.safetensors').is_file()
    assert not list(destination.glob('release-backups/**/local-payload-eviction.json'))


@pytest.mark.parametrize('kind', ['extra', 'trace', 'symlink', 'corrupt'])
def test_unsafe_tree_never_deleted(collection, kind):
    collect, client, github, run, destination = collection
    advance(collection)
    target = destination / 'checkpoint-7'
    if kind == 'extra':
        (target / 'unique.partial').write_bytes(b'unique')
    elif kind == 'trace':
        (target / 'trace.jsonl').write_text('{}\n')
    elif kind == 'symlink':
        (target / 'unexpected').symlink_to(run / 'checkpoint-7/model.safetensors')
    else:
        (target / 'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert (target / 'optimizer.pt').is_file()
    assert not list(destination.glob('release-backups/**/local-payload-eviction.json'))


@pytest.mark.parametrize('field', ['identity', 'published', 'plan_sha256', 'independent_readback_verified'])
def test_invalid_receipt_never_deleted(collection, field):
    collect, client, github, run, destination = collection
    advance(collection)
    path = next(destination.glob('release-backups/checkpoint-7/*/receipt.json'))
    receipt = json.loads(path.read_text())
    if field == 'identity':
        receipt[field]['checkpoint'] = 'checkpoint-99'
    elif field == 'plan_sha256':
        receipt[field] = '0' * 64
    else:
        receipt[field] = False
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert (destination / 'checkpoint-7/model.safetensors').is_file()


def test_restart_reverifies_evicted_public_release(collection, monkeypatch):
    collect, client, github, run, destination = collection
    old_github = advance(collection)
    collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    client._confirmed_release_receipts.clear()
    monkeypatch.setattr(cache.release, 'GitHub', lambda repository: old_github)
    public_transfer = cache.release.PublicTransfer
    monkeypatch.setattr(cache.release, 'PublicTransfer', lambda plan, opener=None: public_transfer(plan, opener or FakePublicHTTP(old_github)))
    reads = len(old_github.reads)
    collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert len(old_github.reads) > reads
    assert not (destination / 'checkpoint-7/model.safetensors').exists()


def test_bad_runtime_ack_cannot_skip(collection, monkeypatch):
    collect, client, github, run, destination = collection
    advance(collection)
    collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    (run / 'durability/checkpoint-7.json').write_text('{}')
    def forbidden(*args, **kwargs):
        raise RuntimeError('Fresh download required')
    monkeypatch.setattr(controller, 'pull_checkpoint', forbidden)
    with pytest.raises(RuntimeError, match='Fresh download required'):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)


def test_no_opt_in_retains_previous_payload(collection, monkeypatch):
    collect, client, github, run, destination = collection
    advance(collection)
    # Avoid fake single-release fixture rejecting a known old publication.
    original = controller.publish_collected_checkpoint
    def publish(*args):
        if args[4]['name'] == 'checkpoint-7':
            return str(next(destination.glob('release-backups/checkpoint-7/*/receipt.json')))
        return original(*args)
    monkeypatch.setattr(controller, 'publish_collected_checkpoint', publish)
    collect(approve_public_checkpoints=True)
    assert (destination / 'checkpoint-7/model.safetensors').is_file()


def test_restart_remote_failure_does_not_trust_receipt(collection, monkeypatch):
    collect, client, github, run, destination = collection
    old_github = advance(collection)
    collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    client._confirmed_release_receipts.clear()
    old_github.private = True
    monkeypatch.setattr(cache.release, 'GitHub', lambda repository: old_github)
    with pytest.raises(ValueError, match='visibility'):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert (destination / 'checkpoint-8/model.safetensors').is_file()


def test_watcher_passes_cleanup_on_final_collect(tmp_path):
    import colab_sdk_watch as watcher
    calls = []
    watcher.watch(object(), project='/project', run_dir='/run', export_root='/exports',
                  destination=tmp_path, publish_repository='test-owner/test-repo',
                  approve_public_checkpoints=True, prune_local_published_cache=True,
                  collect_fn=lambda *args, **kwargs: calls.append(kwargs) or {},
                  status_fn=lambda *args: {'run_status': {'status': 'completed'}}, emit=lambda _: None)
    assert len(calls) == 2 and all(call['prune_local_published_cache'] is True for call in calls)


def test_manifest_listed_trace_is_retained(collection):
    collect, client, github, run, destination = collection
    (run / 'checkpoint-7/trace.jsonl').write_text('{"important":"trace"}\n')
    (run / 'checkpoint-7/checkpoint_manifest.json').unlink()
    checkpoint_evidence(run / 'checkpoint-7', sha256(run / 'run_manifest.json'))
    shutil.rmtree(run.parent / 'export/checkpoint-7')  # Tiny prebuilt test export only.
    advance(collection)
    with pytest.raises(ValueError, match='traces'):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert (destination / 'checkpoint-7/trace.jsonl').is_file()
    assert (destination / 'checkpoint-7/model.safetensors').is_file()


def test_interrupted_eviction_is_reported_and_retained(collection, monkeypatch):
    from pathlib import Path
    collect, client, github, run, destination = collection
    advance(collection)
    original = Path.unlink
    def interrupted(path, *args, **kwargs):
        if path == destination / 'checkpoint-7/optimizer.pt':
            raise OSError('simulated unlink interruption')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', interrupted)
    with pytest.raises(OSError, match='simulated unlink'):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    monkeypatch.setattr(Path, 'unlink', original)
    with pytest.raises(RuntimeError, match='Interrupted local payload eviction'):
        collect(approve_public_checkpoints=True, prune_local_published_cache=True)
    assert (destination / 'checkpoint-7/optimizer.pt').is_file()
    assert (destination / 'checkpoint-8/model.safetensors').is_file()
