"""Mocked bootstrap contracts only; no provider, package or network operations."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location('colab_public_bootstrap_test', Path(__file__).resolve().parents[1] / 'scripts/colab_public_bootstrap.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'Path', lambda value: tmp_path / Path(value).name)
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda name: 'fixture-only')
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs['timeout'] == 600
        assert kwargs['env']['GIT_TERMINAL_PROMPT'] == '0'
        if argv[:2] == ['git','init']:
            (kwargs['cwd'] / '.git').mkdir()
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda argv, **kwargs:
                        module.REPOSITORY if argv[1] == 'remote' else module.DEFAULT_COMMIT)
    return calls


def test_pins_revision_keeps_torch_and_never_starts_training(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    result = module.main()
    assert result['phase'] == 'ready' and result['training_started'] is False
    assert ['git','fetch','--depth=1','origin',module.DEFAULT_COMMIT] in calls
    assert not any('torch' in argument for call in calls for argument in call)
    assert not (tmp_path / 'picoagent-bootstrap/bootstrap.lock').exists()


def test_existing_run_prevents_checkout_changes(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path)
    run = tmp_path / 'picoagent/runs/original'
    run.mkdir(parents=True)
    with pytest.raises(ValueError, match='run output'):
        module.main()
    assert not calls and run.is_dir()


def test_failed_bootstrap_has_sanitized_status_and_releases_lock(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(9, ['private-fixture-marker'])
    monkeypatch.setattr(module.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    raw = (tmp_path / 'picoagent-bootstrap/status.json').read_text()
    assert 'private-fixture-marker' not in raw
    assert json.loads(raw)['returncode'] == 9
    assert not (tmp_path / 'picoagent-bootstrap/bootstrap.lock').exists()
