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
