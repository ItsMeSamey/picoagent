"""CPU-only lifecycle tests for the SDK checkpoint watcher; no provider calls."""
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("colab_sdk_watch_test", SCRIPTS / "colab_sdk_watch.py")
watch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch_module)


def test_terminal_status_triggers_final_collection_after_new_checkpoint_is_sealed():
    events = []
    collected = iter([
        {"verified_checkpoints": ["checkpoint-1"]},
        {"verified_checkpoints": ["checkpoint-1"]},
        {"verified_checkpoints": ["checkpoint-1", "checkpoint-2"]},
    ])
    statuses = iter([
        {"job_status": {"status": "running"}, "run_status": {"status": "paused"},
         "checkpoints": ["checkpoint-1"]},
        {"job_status": {"status": "completed", "returncode": 0},
         "run_status": {"status": "completed"}, "checkpoints": ["checkpoint-1", "checkpoint-2"]},
        {"job_status": {"status": "completed", "returncode": 0},
         "run_status": {"status": "completed"}, "checkpoints": ["checkpoint-1", "checkpoint-2"]},
    ])

    def collect_fn(*args):
        result = next(collected)
        events.append(("collect", result))
        return result

    def status_fn(*args):
        result = next(statuses)
        events.append(("status", result))
        return result

    exit_code = watch_module.watch(
        object(), project="/fixture", run_dir="/fixture/run", export_root="/fixture/exports",
        destination=Path("/fixture/durable"), collect_fn=collect_fn, status_fn=status_fn,
        sleep_fn=lambda seconds: events.append(("sleep", seconds)), emit=lambda _: None,
    )

    assert exit_code == 0
    assert [kind for kind, _ in events] == ["collect", "status", "sleep", "collect", "status",
                                            "collect", "status"]
    assert events[-2][1]["verified_checkpoints"] == ["checkpoint-1", "checkpoint-2"]


def test_terminal_training_failure_returns_actual_nonzero_code_after_final_collect():
    events = []
    failed = {"job_status": {"status": "failed", "returncode": 23},
              "run_status": {"status": "failed"}, "checkpoints": ["checkpoint-4"]}

    def collect_fn(*args):
        events.append("collect")
        return {"verified_checkpoints": ["checkpoint-4"]}

    def status_fn(*args):
        events.append("status")
        return failed

    exit_code = watch_module.watch(
        object(), project="/fixture", run_dir="/fixture/run", export_root="/fixture/exports",
        destination=Path("/fixture/durable"), collect_fn=collect_fn, status_fn=status_fn,
        sleep_fn=lambda _: events.append("sleep"), emit=lambda _: None,
    )

    assert exit_code == 23
    assert events == ["collect", "status", "collect", "status"]


def test_run_failure_without_supervisor_return_code_still_fails_watch():
    assert watch_module._training_exit_code({
        "job_status": {"status": "completed", "returncode": 0},
        "run_status": {"status": "failed"},
    }) == 1
    assert watch_module._training_exit_code({
        "job_status": {"status": "failed", "returncode": -9},
        "run_status": {"status": "failed"},
    }) == 137


def test_transient_collection_and_terminal_status_retry_without_losing_final_drain():
    import json
    attempts, sleeps, output = [], [], []
    failures = [ConnectionError, TimeoutError, ConnectionError, ConnectionError, None, None]
    statuses = [ConnectionError, {'run_status': {'status': 'completed'}},
                {'run_status': {'status': 'completed'}}]

    def collect(*args):
        failure = failures.pop(0)
        attempts.append('collect')
        if failure:
            raise failure('PRIVATE_URL_NOT_LOGGED')
        return {'verified_checkpoints': ['checkpoint-9']}

    def status(*args):
        value = statuses.pop(0)
        if value is ConnectionError:
            raise value('PRIVATE_URL_NOT_LOGGED')
        return value

    assert watch_module.watch(object(), project='/p', run_dir='/r', export_root='/e',
                              destination=Path('/d'), collect_fn=collect, status_fn=status,
                              sleep_fn=sleeps.append, emit=output.append, retry_max_seconds=25) == 0
    assert sleeps == [10, 20, 25, 25, 10]
    assert len(attempts) == 6
    assert 'PRIVATE_URL_NOT_LOGGED' not in ''.join(output)
    assert sum(json.loads(value).get('controller_retry', False) for value in output) == 5


def test_permission_quota_and_integrity_errors_never_retry():
    import pytest
    from types import SimpleNamespace
    for error in [ValueError('hash mismatch'), PermissionError('denied'),
                  RuntimeError('remote execution failed')]:
        assert not watch_module._retryable_transport_error(error)
    for status in (401, 403, 404, 429):
        error = ConnectionError('private response')
        error.response = SimpleNamespace(status_code=status)
        assert not watch_module._retryable_transport_error(error)
    with pytest.raises(ValueError):
        watch_module.watch(object(), project='/p', run_dir='/r', export_root='/e',
                           destination=Path('/d'), prune=True, prune_to_latest_published=True)
