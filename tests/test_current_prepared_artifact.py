"""Cheap production-source regression: checkpoint changes require new token approval.

Historical configs remain archived; this test protects the designated current
full-run config before any accelerator allocation, without loading ML packages.
"""
import hashlib
import json
from pathlib import Path

from picoagent.training.prepared_approvals import APPROVED_PREPARED_MANIFESTS


def test_current_full_run_prepared_artifact_matches_entire_core_source():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/smol360m_native_t4_async_30min_v4.json').read_text())
    assert config['async_checkpoint_upload'] is True
    assert config['async_checkpoint_max_local'] == 2
    assert config['checkpoint_interval_seconds'] == 1800.0
    assert config['output_dir'] == 'runs/smol360m-native-t4-async-30min-v4'
    path = root / config['prepared_manifest']
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == config['prepared_manifest_sha256']
    assert digest in APPROVED_PREPARED_MANIFESTS
    manifest = json.loads(data)
    package = root / 'src/picoagent'
    source = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob('*.py')
              if p.relative_to(package).as_posix() != 'training/prepared_approvals.py'}
    assert manifest['transform']['source'] == source, (
        'Current production token artifact is stale after a core source change; '
        'rebuild, independently audit, approve and repin before GPU allocation')
    for relative, record in manifest['files'].items():
        payload = path.parent / relative
        assert not payload.is_symlink()
        assert payload.stat().st_size == record['bytes']
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == record['sha256']
