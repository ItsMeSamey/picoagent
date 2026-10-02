"""Generated launch-code execution tests; no runtime/provider calls or training."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import colab_run as controller  # noqa: E402


class LocalExecution:
    def __init__(self):
        self.calls = 0
        self.output = ''

    def execute(self, code):
        self.calls += 1
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, 'local-launch-fixture.py', 'exec'), {})
        self.output = output.getvalue()
        lines = [line for line in self.output.splitlines() if line.startswith(controller.OUTPUT_SENTINEL)]
        assert len(lines) == 1
        return json.loads(lines[0][len(controller.OUTPUT_SENTINEL):])


@pytest.mark.parametrize('foreground', [False, True])
@pytest.mark.parametrize('returncode', [0, 17])
def test_foreground_waits_only_after_lock_release_preserving_supervisor_state(
        tmp_path, monkeypatch, foreground, returncode):
    state = tmp_path / '.picoagent-job.json'
    lock = tmp_path / '.picoagent-launch.lock'
    events = []
    terminal = {'status': 'completed' if returncode == 0 else 'failed', 'returncode': returncode,
                'finished_at': 123.5}

    class Child:
        pid = 12345

        def wait(self):
            assert not lock.exists(), 'Waiting must not hold the launch lock'
            assert json.loads(state.read_text())['status'] == 'starting'
            events.append('wait')
            state.write_text(json.dumps(terminal))  # Represents independent supervisor completion.
            return returncode

    def popen(argv, **kwargs):
        assert lock.exists()
        assert json.loads(state.read_text()) == {'status': 'starting', 'command': ['python', 'train.py']}
        assert kwargs['start_new_session'] is True
        events.append('spawn')
        return Child()

    monkeypatch.setattr(subprocess, 'Popen', popen)
    client = LocalExecution()
    result = controller.start(client, str(tmp_path), ['python', 'train.py'], wait_for_completion=foreground)
    assert client.calls == 1
    assert events == (['spawn', 'wait'] if foreground else ['spawn'])
    assert not lock.exists()
    assert result['supervisor_pid'] == 12345
    if foreground:
        assert result['returncode'] == returncode and result['waited_for_completion'] is True
        assert json.loads(state.read_text()) == terminal
    else:
        assert set(result) == {'supervisor_pid', 'log'}
        assert json.loads(state.read_text())['status'] == 'starting'


def test_wait_interruption_does_not_relaunch_or_rewrite_job_state(tmp_path, monkeypatch):
    state = tmp_path / '.picoagent-job.json'
    running = {'status': 'running', 'pid': 12345, 'command': ['python', 'train.py']}
    spawns = []

    class Child:
        pid = 12345

        def wait(self):
            assert not (tmp_path / '.picoagent-launch.lock').exists()
            state.write_text(json.dumps(running))
            raise TimeoutError('ambiguous foreground disconnect')

    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: spawns.append(1) or Child())
    client = LocalExecution()
    with pytest.raises(TimeoutError, match='ambiguous'):
        controller.start(client, str(tmp_path), ['python', 'train.py'], wait_for_completion=True)
    assert client.calls == len(spawns) == 1
    assert json.loads(state.read_text()) == running
    assert not (tmp_path / '.picoagent-launch.lock').exists()
    # A mistaken explicit repeat is still stopped by the existing reservation.
    with pytest.raises(RuntimeError, match='already running'):
        controller.start(client, str(tmp_path), ['python', 'train.py'], wait_for_completion=True)
    assert len(spawns) == 1


def test_spawn_failure_preserves_failed_state_and_does_not_wait(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('spawn denied')

    monkeypatch.setattr(subprocess, 'Popen', fail)
    with pytest.raises(OSError, match='spawn denied'):
        controller.start(LocalExecution(), str(tmp_path), ['python', 'train.py'], wait_for_completion=True)
    assert json.loads((tmp_path / '.picoagent-job.json').read_text()) == {
        'status': 'failed', 'error': 'Supervisor launch failed'}
    assert not (tmp_path / '.picoagent-launch.lock').exists()


@pytest.mark.parametrize('value', [None, 1, 'true'])
def test_invalid_foreground_option_fails_before_execution(tmp_path, value):
    client = LocalExecution()
    with pytest.raises(ValueError, match='boolean'):
        controller.start(client, str(tmp_path), ['python', 'train.py'], wait_for_completion=value)
    assert client.calls == 0


@pytest.mark.parametrize('foreground', [False, True])
def test_cli_foreground_forwarding_keeps_default_call_compatible(tmp_path, monkeypatch, foreground):
    calls = []
    monkeypatch.setattr(controller, 'Colab', lambda *a: object())
    monkeypatch.setattr(controller, 'start', lambda *a, **k: calls.append(k) or {})
    argv = ['colab_run.py', '--session', 'existing', 'start', '--command-json', '["python", "train.py"]']
    if foreground:
        argv.append('--wait-for-completion')
    monkeypatch.setattr(sys, 'argv', argv)
    controller.main()
    assert calls == ([{'wait_for_completion': True}] if foreground else [{}])


@pytest.mark.parametrize('status', ['running', 'starting'])
def test_status_changed_before_lock_acquisition_cannot_launch_duplicate(tmp_path, monkeypatch, status):
    state = tmp_path / '.picoagent-job.json'
    lock = tmp_path / '.picoagent-launch.lock'
    original_open = Path.open
    existing = {'status': status, 'pid': 98765}

    def race(path, *args, **kwargs):
        if path == lock and args == ('x',):
            state.write_text(json.dumps(existing))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', race)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Never duplicate a reserved job'))
    with pytest.raises(RuntimeError, match='already running'):
        controller.start(LocalExecution(), str(tmp_path), ['python', 'train.py'], wait_for_completion=True)
    assert json.loads(state.read_text()) == existing
    assert not lock.exists()


@pytest.mark.parametrize('returncode', [0, 17])
def test_real_local_supervisor_propagates_training_exit_without_overwriting_state(tmp_path, returncode):
    # Both supervisor and training fixture are tiny real local Python processes.
    # No model, network, provider, credential, or live runtime is involved.
    result = controller.start(LocalExecution(), str(tmp_path),
                              [sys.executable, '-c', f'raise SystemExit({returncode})'],
                              wait_for_completion=True)
    assert result['returncode'] == returncode
    assert result['waited_for_completion'] is True
    saved = json.loads((tmp_path / '.picoagent-job.json').read_text())
    assert saved['status'] == ('completed' if returncode == 0 else 'failed')
    assert saved['returncode'] == returncode
    assert 'error' not in saved, 'SystemExit must not overwrite the recorded training outcome'
    assert not (tmp_path / '.picoagent-launch.lock').exists()
