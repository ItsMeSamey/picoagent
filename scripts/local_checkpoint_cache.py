"""Opt-in eviction of recoverable checkpoint payloads; never deletes releases."""
import hashlib
import json
import re
import tempfile
from pathlib import Path

from checkpoint_sync import (BUNDLE_MANIFEST, checkpoint_name, sha256, validate_manifest,
                             _exact_retention_tree, sync_directory, pull_evaluation)
from source_staging import no_symlink_path
from picoagent.training.data import canonical_json
from picoagent.training.durability import acknowledgement_from_receipt
from picoagent.training.provenance import write_json
import github_checkpoint_release as release


PAYLOAD = re.compile(r'(?:model(?:-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin|optimizer\.pt|scheduler\.pt|scaler\.pt|rng_state(?:_\d+)?\.pth)\Z')


def evidence(destination, name, digest, repository):
    checkpoint_name(name)
    if not release.DIGEST.fullmatch(digest):
        raise ValueError('Invalid cache identity digest')
    root = no_symlink_path(destination / 'release-backups' / name / digest)
    plan = json.loads(no_symlink_path(root / 'plan.json').read_text())
    receipt = json.loads(no_symlink_path(root / 'receipt.json').read_text())
    release.validate_plan(plan)
    ack = acknowledgement_from_receipt(receipt)
    identity = plan['identity']
    if (identity['checkpoint'] != name or identity['repository'] != repository
            or identity['transfer_manifest_sha256'] != digest
            or identity['run_manifest_sha256'] != sha256(no_symlink_path(destination / 'run_manifest.json'))
            or receipt['identity'] != identity or receipt['plan_sha256'] != release.digest_json(plan)
            or receipt['tag'] != plan['tag']
            or {key: {field: value[field] for field in ('sha256', 'bytes')}
                for key, value in receipt['assets'].items()} != plan['assets']):
        raise ValueError('Published cache evidence identity mismatch')
    manifest_path = no_symlink_path(destination / '.incoming' / name / digest / BUNDLE_MANIFEST)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    if (sha256(manifest_path) != digest or manifest['checkpoint'] != name
            or release.asset_records(manifest, manifest_path.stat().st_size, digest) != plan['assets']
            or manifest['files'][f'{name}/checkpoint_manifest.json']['sha256'] != identity['checkpoint_manifest_sha256']
            or manifest['files']['run_manifest.json']['sha256'] != identity['run_manifest_sha256']):
        raise ValueError('Published cache manifest mismatch')
    return root, plan, receipt, ack, manifest_path, manifest


def verify_published(plan, receipt):
    """Read-only restart verification, including every asset, without local payloads."""
    client = release.GitHub(plan['identity']['repository'])
    if client.api(client.prefix).get('private') is not False:
        raise ValueError('Public repository visibility mismatch')
    body = canonical_json({'schema': release.SCHEMA, 'plan_sha256': release.digest_json(plan),
                           'identity': plan['identity']})
    current = client.release(plan['tag'])
    release.check_release(current, plan, body)
    if current['draft'] or current['id'] != receipt['release_id']:
        raise ValueError('Published release identity changed')
    client.check_tag(plan['tag'], plan['identity']['source_commit'], required=True)
    assets = release.asset_inventory(client, current, plan)
    if set(assets) != set(plan['assets']):
        raise ValueError('Published release is incomplete')
    for name, record in plan['assets'].items():
        if assets[name]['id'] != receipt['assets'][name]['id']:
            raise ValueError('Published release asset identity changed')
        with client.stream(assets[name]) as stream:
            release.readback(stream, record)
    latest = client.release(plan['tag'])
    release.check_release(latest, plan, body)
    if (latest['draft'] or latest['id'] != current['id']
            or release.asset_identity(release.asset_inventory(client, latest, plan)) != release.asset_identity(assets)
            or client.api(client.prefix).get('private') is not False):
        raise ValueError('Published release changed during readback')
    client.check_tag(plan['tag'], plan['identity']['source_commit'], required=True)
    with tempfile.TemporaryDirectory(prefix='checkpoint-public-cache-check-') as directory:
        release.PublicTransfer(plan).download('release/' + BUNDLE_MANIFEST, Path(directory) / BUNDLE_MANIFEST)


def skip_published(client, destination, bundle, repository):
    """Return pinned evidence only with exact runtime ack, never a bare filename."""
    root = no_symlink_path(destination / 'release-backups' / checkpoint_name(bundle['name']) / bundle['sha256'])
    if not (root / 'receipt.json').exists():
        return None
    root, plan, receipt, ack, manifest_path, manifest = evidence(
        destination, bundle['name'], bundle['sha256'], repository)
    ack_hash = hashlib.sha256((canonical_json(ack) + '\n').encode()).hexdigest()
    if bundle.get('durability_ack_sha256') != ack_hash:
        return None
    cache = getattr(client, '_confirmed_release_receipts', {})
    digest = release.digest_json(plan)
    if cache.get(digest) != release.digest_json(receipt):
        verify_published(plan, receipt)
        cache[digest] = release.digest_json(receipt)
        client._confirmed_release_receipts = cache
    # Evaluation sidecars can appear after publication; always collect them.
    pull_evaluation(bundle['path'].lstrip('/'), destination, client.transfer(), manifest,
                    bundle['evaluation_sha256'])
    return str(root / 'receipt.json'), plan['identity']


def prune_published(destination, latest_bundle, repository, client):
    """Verify newest copy and old complete trees before deleting only payload files.

    JSON metadata is kept in place. Unexpected entries, traces, symlinks and any
    incomplete checkpoint prevent eviction. Incoming partials are never visited.
    Caller owns the destination's exclusive collection lock.
    """
    latest = evidence(destination, latest_bundle['name'], latest_bundle['sha256'], repository)
    _, plan, receipt, _, manifest_path, _ = latest
    if release.make_plan(destination, manifest_path, repository) != plan:
        raise ValueError('Latest local checkpoint is not complete')
    cache = getattr(client, '_confirmed_release_receipts', {})
    if cache.get(release.digest_json(plan)) != release.digest_json(receipt):
        raise ValueError('Latest public backup was not verified by this controller')
    pruned = []
    for path in sorted((destination / 'release-backups').glob('checkpoint-*/*/receipt.json')):
        name, digest = path.parent.parent.name, path.parent.name
        if int(checkpoint_name(name).split('-')[1]) >= int(latest_bundle['name'].split('-')[1]):
            continue
        root, old_plan, old_receipt, _, old_manifest_path, manifest = evidence(destination, name, digest, repository)
        files = {str(Path(relative).relative_to(name)) for relative in manifest['files']
                 if relative.startswith(name + '/')}
        payloads = sorted(relative for relative in files if PAYLOAD.fullmatch(relative))
        marker = no_symlink_path(root / 'local-payload-eviction.json')
        if marker.exists():
            saved = json.loads(marker.read_text())
            if (saved.get('plan_sha256') != release.digest_json(old_plan)
                    or saved.get('receipt_sha256') != release.digest_json(old_receipt)
                    or saved.get('payloads') != payloads):
                raise ValueError('Local eviction record identity mismatch')
            for relative in saved['payloads']:
                if (not isinstance(relative, str) or Path(relative).name != relative
                        or f'{name}/{relative}' not in manifest['files']):
                    raise ValueError('Unsafe local eviction record payload')
                if no_symlink_path(destination / name / relative).exists():
                    raise RuntimeError(f'Interrupted local payload eviction for {name}; remaining payloads retained for recovery')
            continue
        target = no_symlink_path(destination / name)
        if not target.exists():
            continue
        # Validate the whole tree first: even a trace listed in a manifest blocks deletion.
        tree = _exact_retention_tree(target, files)
        if release.make_plan(destination, old_manifest_path, repository) != old_plan:
            raise ValueError('Older checkpoint integrity mismatch')
        if cache.get(release.digest_json(old_plan)) != release.digest_json(old_receipt):
            verify_published(old_plan, old_receipt)
            cache[release.digest_json(old_plan)] = release.digest_json(old_receipt)
            client._confirmed_release_receipts = cache
        if _exact_retention_tree(target, files) != tree:
            raise ValueError('Checkpoint changed before eviction')
        # A durable intent preserves exact evidence if interrupted during unlink.
        write_json(marker, {'plan_sha256': release.digest_json(old_plan),
                            'receipt_sha256': release.digest_json(old_receipt), 'payloads': payloads}, exclusive=True)
        for relative in payloads:
            no_symlink_path(target / relative).unlink()
            sync_directory((target / relative).parent)
        pruned.append(name)
    return pruned
