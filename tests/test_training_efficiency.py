"""Opt-in local CPU proofs for explicit new-run checkpoint/eval policy."""
import dataclasses
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("PICOAGENT_RUN_ML_TESTS") != "1",
                                reason="optional local torch/Transformers integration smoke")


def assert_same_training(reference, resumed, reference_run, resumed_run, step):
    import torch
    from safetensors.torch import load_file
    a = load_file(str(Path(reference["artifact"]) / "model.safetensors"))
    b = load_file(str(Path(resumed["artifact"]) / "model.safetensors"))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[key], b[key]) for key in a)
    def equal_state(left, right):
        import numpy as np
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_state(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert type(left) is type(right) and len(left) == len(right)
            for x, y in zip(left, right):
                equal_state(x, y)
        else:
            assert left == right

    for filename in ("scheduler.pt", "optimizer.pt", "rng_state.pth"):
        equal_state(torch.load(reference_run / f"checkpoint-{step}" / filename, weights_only=False),
                    torch.load(resumed_run / f"checkpoint-{step}" / filename, weights_only=False))


def test_seal_before_eval_exact_cap_and_dropout_resume_equality(tmp_path):
    from transformers import Trainer
    from picoagent.training import train
    from picoagent.training.config import TrainingConfig
    from picoagent.training.smoke import run_smoke
    from picoagent.training.provenance import verify_checkpoint
    from picoagent.training.evaluation import read_evaluation

    options = dict(device="cpu", train_records=7, gradient_accumulation_steps=1,
                   save_steps=2, eval_steps=1, max_steps=5, dropout=0.2,
                   checkpoint_interval_seconds=None, checkpoint_before_eval=True)
    original_evaluate = Trainer.evaluate
    evaluated = []

    def verify_then_evaluate(trainer, *args, **kwargs):
        run = Path(trainer.args.output_dir)
        step = trainer.state.global_step
        checkpoint = run / f"checkpoint-{step}"
        before = None
        if checkpoint.exists():
            verify_checkpoint(checkpoint, train.sha256_file(run / "run_manifest.json"))
            before = train.sha256_file(checkpoint / "checkpoint_manifest.json")
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            # Coincident eval must not be inside the checkpoint sealed first.
            assert not any(row.get("step") == step and "eval_loss" in row
                           for row in state.get("log_history", []))
        elif step % 2 == 0 or step == 5:
            pytest.fail("Evaluation ran before the scheduled checkpoint was sealed")
        result = original_evaluate(trainer, *args, **kwargs)
        if before is not None:
            assert train.sha256_file(checkpoint / "checkpoint_manifest.json") == before
            verify_checkpoint(checkpoint, train.sha256_file(run / "run_manifest.json"))
        evaluated.append((str(run), step))
        return result

    with patch.object(Trainer, "evaluate", verify_then_evaluate):
        reference = run_smoke(tmp_path / "reference", **options)
        first = run_smoke(tmp_path / "segmented", segment_steps=3,
                          continue_through_checkpoints=True, **options)
        run = tmp_path / "segmented/run"
        assert first["status"] == "paused"
        assert first["global_step"] == 3
        assert sorted(p.name for p in run.glob("checkpoint-*")) == ["checkpoint-2", "checkpoint-3"]
        status = json.loads((run / "run_status.json").read_text())
        assert status["new_checkpoints"] == ["checkpoint-2", "checkpoint-3"]
        assert status["continue_through_checkpoints"] is True
        for step in (2, 3):
            assert read_evaluation(run / f"evaluations/checkpoint-{step}.json",
                                   run / f"checkpoint-{step}")["global_step"] == step
        config = TrainingConfig.load(tmp_path / "segmented/smoke-config.json")
        with pytest.raises(ValueError, match="Resume identity mismatch"):
            train.run_training(dataclasses.replace(config, eval_steps=3),
                               resume_from_checkpoint=str(run / "checkpoint-3"), segment_steps=2,
                               continue_through_checkpoints=True)
        resumed = train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-3"),
                                     segment_steps=2, continue_through_checkpoints=True)
    assert resumed["status"] == "completed" and resumed["global_step"] == 5
    assert_same_training(reference, resumed, tmp_path / "reference/run", run, 5)
    assert any(step == 1 for _, step in evaluated)  # independent eval-only step exercised
    # Final metrics reuse the just-finished evaluation instead of doing it twice.
    assert sum(step == 5 and path == str(tmp_path / "reference/run")
               for path, step in evaluated) == 1
    assert sum(step == 5 and path == str(run) for path, step in evaluated) == 1


def test_interruption_after_seal_before_eval_preserves_dropout_rng(tmp_path):
    from picoagent.training import train
    from picoagent.training.config import TrainingConfig
    from picoagent.training.smoke import run_smoke
    from picoagent.training.provenance import verify_checkpoint

    options = dict(device="cpu", train_records=5, gradient_accumulation_steps=1,
                   save_steps=1, eval_steps=1, max_steps=3, dropout=0.2,
                   checkpoint_interval_seconds=None, checkpoint_before_eval=True)
    reference = run_smoke(tmp_path / "reference", **options)
    original = train.checkpoint_evidence

    def interrupt(checkpoint, run_hash):
        original(checkpoint, run_hash)
        raise InterruptedError("controlled interruption before eval")

    with patch.object(train, "checkpoint_evidence", interrupt), pytest.raises(InterruptedError):
        run_smoke(tmp_path / "interrupted", **options)
    run = tmp_path / "interrupted/run"
    verify_checkpoint(run / "checkpoint-1", train.sha256_file(run / "run_manifest.json"))
    assert not (run / "evaluations/checkpoint-1.json").exists()
    config = TrainingConfig.load(tmp_path / "interrupted/smoke-config.json")
    resumed = train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-1"))
    assert_same_training(reference, resumed, tmp_path / "reference/run", run, 3)


def test_timer_saves_without_forcing_eval_and_rechecks_budget_before_each_save(tmp_path):
    from picoagent.training import train
    from picoagent.training.config import TrainingConfig
    from picoagent.training.smoke import run_smoke

    first = run_smoke(tmp_path / "run", device="cpu", max_steps=4, segment_steps=2,
                      train_records=5, gradient_accumulation_steps=1, save_steps=100,
                      eval_steps=100, checkpoint_interval_seconds=0.000001,
                      checkpoint_before_eval=True, continue_through_checkpoints=True)
    run = tmp_path / "run/run"
    assert first["global_step"] == 2
    assert sorted(p.name for p in run.glob("checkpoint-*")) == ["checkpoint-1", "checkpoint-2"]
    assert not (run / "evaluations").exists()
    for step in (1, 2):
        state = json.loads((run / f"checkpoint-{step}/trainer_state.json").read_text())
        assert not any("eval_loss" in row for row in state["log_history"])

    config = TrainingConfig.load(tmp_path / "run/smoke-config.json")
    real_reservation = train._output_budget_reservation
    calls = []
    exports = tmp_path / "exports"
    exports.mkdir()

    def reserve(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:  # initial preflight passes, next save must re-check
            # Sparse fixture models a separately generated export under the
            # chosen budget root without allocating hundreds of MiB in RAM.
            with (exports / "concurrent-export.bin").open("wb") as handle:
                handle.truncate(700 * 1024 * 1024)
        return real_reservation(*args, **kwargs)

    with patch.object(train, "_output_budget_reservation", reserve), pytest.raises(OSError, match="Insufficient bounded output budget"):
        train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-2"),
                           segment_steps=2, continue_through_checkpoints=True,
                           output_budget_bytes=2_000_000_000, output_budget_root=str(tmp_path))
    assert len(calls) == 2
    assert not (run / "checkpoint-3").exists()
    assert json.loads((run / "run_status.json").read_text())["status"] == "failed"
    assert (run / "checkpoint-2/checkpoint_manifest.json").is_file()


def test_legacy_boundary_does_not_add_timer_evaluation(tmp_path):
    from picoagent.training.smoke import run_smoke
    result = run_smoke(tmp_path / "legacy", device="cpu", max_steps=3, segment_steps=1,
                       train_records=5, gradient_accumulation_steps=1, save_steps=100,
                       checkpoint_interval_seconds=0.000001)
    assert result["status"] == "paused" and result["global_step"] == 1
    state = json.loads((tmp_path / "legacy/run/checkpoint-1/trainer_state.json").read_text())
    assert not any("eval_loss" in row for row in state["log_history"])
    assert not (tmp_path / "legacy/run/evaluations").exists()


def test_final_step_eval_failure_preserves_checkpoint_but_requires_separate_finalization(tmp_path):
    from transformers import Trainer
    from picoagent.training import train
    from picoagent.training.config import TrainingConfig
    from picoagent.training.smoke import run_smoke
    from picoagent.training.provenance import verify_checkpoint

    with patch.object(Trainer, "evaluate", side_effect=RuntimeError("final evaluation failed")), \
            pytest.raises(RuntimeError, match="final evaluation failed"):
        run_smoke(tmp_path / "final-failure", device="cpu", max_steps=1,
                  checkpoint_before_eval=True, eval_steps=1)
    run = tmp_path / "final-failure/run"
    verify_checkpoint(run / "checkpoint-1", train.sha256_file(run / "run_manifest.json"))
    assert not (run / "evaluations/checkpoint-1.json").exists()
    assert not (run / "final-model").exists()
    config = TrainingConfig.load(tmp_path / "final-failure/smoke-config.json")
    with pytest.raises(ValueError, match="already reaches the configured training budget"):
        train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-1"))


def test_accumulated_cross_epoch_dropout_resume_exact_full_state(tmp_path):
    from picoagent.training import train
    from picoagent.training.config import TrainingConfig
    from picoagent.training.smoke import run_smoke

    # Three rows and accumulation two create a short final accumulation group
    # in each epoch. Split inside epoch two after its first complete update.
    options = dict(device="cpu", train_records=3, gradient_accumulation_steps=2,
                   save_steps=2, eval_steps=1, max_steps=4, dropout=0.2,
                   checkpoint_interval_seconds=None, checkpoint_before_eval=True)
    reference = run_smoke(tmp_path / "reference", **options)
    first = run_smoke(tmp_path / "segmented", segment_steps=3,
                      continue_through_checkpoints=True, **options)
    assert first["global_step"] == 3 and first["status"] == "paused"
    run = tmp_path / "segmented/run"
    state = json.loads((run / "checkpoint-3/trainer_state.json").read_text())
    assert 1 < state["epoch"] < 2
    config = TrainingConfig.load(tmp_path / "segmented/smoke-config.json")
    resumed = train.run_training(config, resume_from_checkpoint=str(run / "checkpoint-3"),
                                 segment_steps=1, continue_through_checkpoints=True)
    assert resumed["global_step"] == 4 and resumed["status"] == "completed"
    final_state = json.loads((run / "checkpoint-4/trainer_state.json").read_text())
    assert final_state["epoch"] == 2
    assert_same_training(reference, resumed, tmp_path / "reference/run", run, 4)
