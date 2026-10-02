"""Local crash-safe publication and genuine fresh-process Trainer restore."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from picoagent.training.checkpointing import save_checkpoint_transaction
from picoagent.training.provenance import preflight_resume, verify_checkpoint


def write_fixture(root, step):
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"fixture")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))


def test_failed_save_preserves_last_good_and_does_not_block_resume(tmp_path):
    good = save_checkpoint_transaction(tmp_path, 1, lambda root: write_fixture(root, 1), "run")
    manifest = (good / "checkpoint_manifest.json").read_bytes()

    def fail(root):
        checkpoint = root / "checkpoint-2"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_checkpoint_transaction(tmp_path, 2, fail, "run")
    assert not (tmp_path / "checkpoint-2").exists()
    assert list(tmp_path.glob(".checkpoint-2-*"))
    assert (good / "checkpoint_manifest.json").read_bytes() == manifest
    verify_checkpoint(good, "run")
    preflight_resume(tmp_path, good, max_steps=3)
    newer = save_checkpoint_transaction(tmp_path, 2, lambda root: write_fixture(root, 2), "run")
    verify_checkpoint(newer, "run")


def test_incomplete_save_never_becomes_visible(tmp_path):
    def incomplete(root):
        (root / "checkpoint-1").mkdir()
    with pytest.raises(ValueError, match="weights"):
        save_checkpoint_transaction(tmp_path, 1, incomplete, "run")
    assert not (tmp_path / "checkpoint-1").exists()


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    good = save_checkpoint_transaction(tmp_path, 1, lambda root: write_fixture(root, 1), "run")
    with pytest.raises(FileExistsError):
        save_checkpoint_transaction(tmp_path, 1, lambda root: pytest.fail("must not save"), "run")
    verify_checkpoint(good, "run")


@pytest.mark.skipif(os.environ.get("PICOAGENT_RUN_ML_TESTS") != "1", reason="optional CPU ML integration")
def test_fresh_process_recovers_after_killed_partial_save(tmp_path):
    """No live model/optimizer/RNG objects survive any of these invocations."""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                       USE_TORCH_XLA="0", OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false")

    def run(code, expected=0):
        result = subprocess.run([sys.executable, "-c", code], env=environment,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == expected, result.stdout + result.stderr

    options = "device='cpu', max_steps=3, train_records=5, gradient_accumulation_steps=1, save_steps=1, dropout=0.2, checkpoint_before_eval=True, eval_steps=1"
    run(f"from picoagent.training.smoke import run_smoke; run_smoke({str(tmp_path / 'reference')!r}, {options})")
    run(f"from picoagent.training.smoke import run_smoke; run_smoke({str(tmp_path / 'resumed')!r}, segment_steps=1, {options})")
    checkpoint = tmp_path / "resumed/run/checkpoint-1"
    config = tmp_path / "resumed/smoke-config.json"
    resume = f"run_training(TrainingConfig.load({str(config)!r}), resume_from_checkpoint={str(checkpoint)!r})"
    imports = "from picoagent.training.train import run_training; from picoagent.training.config import TrainingConfig; "
    run("import os; from transformers import Trainer; "
        "Trainer._save_optimizer_and_scheduler = lambda *a, **kw: os._exit(86); "
        + imports + resume, expected=86)
    assert not (tmp_path / "resumed/run/checkpoint-2").exists()
    assert list((tmp_path / "resumed/run").glob(".checkpoint-2-*"))
    preflight_resume(checkpoint.parent, checkpoint, max_steps=3)
    run(imports + resume)

    import torch
    from safetensors.torch import load_file

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    reference = tmp_path / "reference/run"
    resumed = tmp_path / "resumed/run"
    equal(load_file(str(reference / "final-model/model.safetensors")),
          load_file(str(resumed / "final-model/model.safetensors")))
    for filename in ("optimizer.pt", "scheduler.pt"):
        equal(torch.load(reference / "checkpoint-3" / filename, weights_only=False),
              torch.load(resumed / "checkpoint-3" / filename, weights_only=False))
    for root in (reference, resumed):
        state = json.loads((root / "checkpoint-3/trainer_state.json").read_text())
        assert state["global_step"] == state["max_steps"] == 3


@pytest.mark.skipif(os.environ.get("PICOAGENT_RUN_ML_TESTS") != "1", reason="optional CPU ML integration")
def test_durability_gate_survives_resume_and_precedes_evaluation(tmp_path, monkeypatch):
    from dataclasses import replace
    from picoagent.training import durability
    from picoagent.training.config import TrainingConfig
    from picoagent.training.data import sha256_file
    from picoagent.training.smoke import run_smoke
    from picoagent.training.train import run_training

    run_smoke(tmp_path / "fixture", device="cpu", max_steps=2, checkpoint_before_eval=True)
    config = replace(TrainingConfig.load(tmp_path / "fixture/smoke-config.json"),
                     output_dir=str(tmp_path / "gated"), save_steps=100, eval_steps=500,
                     checkpoint_interval_seconds=600.0)
    waited = []

    def timeout(output, checkpoint, timeout_seconds):
        verify_checkpoint(checkpoint, sha256_file(output / "run_manifest.json"))
        waited.append(checkpoint.name)
        raise TimeoutError("backup has not completed")

    monkeypatch.setattr(durability, "wait_for_durable_ack", timeout)
    with pytest.raises(TimeoutError, match="backup"):
        run_training(config, durability_timeout_seconds=1)
    checkpoint = tmp_path / "gated/checkpoint-1"
    assert waited == ["checkpoint-1"]
    assert not (tmp_path / "gated/evaluations").exists()
    assert not (tmp_path / "gated/checkpoint-2").exists()
    manifest = json.loads((tmp_path / "gated/run_manifest.json").read_text())
    assert manifest["durability_required"] is True
    with pytest.raises(ValueError, match="requires --durability"):
        run_training(config, resume_from_checkpoint=str(checkpoint))
    with pytest.raises(TimeoutError, match="backup"):
        run_training(config, resume_from_checkpoint=str(checkpoint), durability_timeout_seconds=1)
    assert waited == ["checkpoint-1", "checkpoint-1"]
    assert not (tmp_path / "gated/checkpoint-2").exists()


def test_finalize_preflight_requires_exact_final_step(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    state = checkpoint / "trainer_state.json"
    state.write_text(json.dumps({"global_step": 2, "max_steps": 3}))
    with pytest.raises(ValueError, match="exact final"):
        preflight_resume(tmp_path, checkpoint, max_steps=3, finalize_only=True)
    state.write_text(json.dumps({"global_step": 2, "max_steps": 2}))
    with pytest.raises(ValueError, match="configured training budget"):
        preflight_resume(tmp_path, checkpoint, max_steps=3, finalize_only=True)
    preflight_resume(tmp_path, checkpoint, max_steps=2, finalize_only=True)
    preflight_resume(tmp_path, checkpoint, max_steps=-1, finalize_only=True)
    with pytest.raises(ValueError, match="training budget"):
        preflight_resume(tmp_path, checkpoint, max_steps=2)


@pytest.mark.skipif(os.environ.get("PICOAGENT_RUN_ML_TESTS") != "1", reason="optional CPU ML integration")
def test_finalize_after_final_backup_timeout_never_trains(tmp_path, monkeypatch):
    from dataclasses import replace
    import torch
    from safetensors.torch import load_file
    from transformers import Trainer
    from picoagent.training import durability
    from picoagent.training.config import TrainingConfig
    from picoagent.training.data import sha256_file
    from picoagent.training.smoke import run_smoke
    from picoagent.training.train import run_training

    reference = run_smoke(tmp_path / "reference", device="cpu", max_steps=2,
                          checkpoint_before_eval=True, dropout=0.2)
    config = replace(TrainingConfig.load(tmp_path / "reference/smoke-config.json"),
                     output_dir=str(tmp_path / "gated"))
    acknowledged = []

    def gate(output, checkpoint, timeout_seconds):
        verify_checkpoint(checkpoint, sha256_file(output / "run_manifest.json"))
        if checkpoint.name == "checkpoint-2":
            raise TimeoutError("final backup incomplete")
        acknowledged.append(checkpoint.name)

    monkeypatch.setattr(durability, "wait_for_durable_ack", gate)
    with pytest.raises(TimeoutError, match="final backup"):
        run_training(config, durability_timeout_seconds=1)
    checkpoint = tmp_path / "gated/checkpoint-2"
    manifest_before = (checkpoint / "checkpoint_manifest.json").read_bytes()
    assert not (tmp_path / "gated/final-model").exists()
    assert not (tmp_path / "gated/evaluations/checkpoint-2.json").exists()
    with pytest.raises(ValueError, match="training budget"):
        run_training(config, resume_from_checkpoint=str(checkpoint), durability_timeout_seconds=1)
    with pytest.raises(TimeoutError, match="final backup"):
        run_training(config, resume_from_checkpoint=str(checkpoint), durability_timeout_seconds=1,
                     finalize_only=True)

    def ack(output, checkpoint, timeout_seconds):
        verify_checkpoint(checkpoint, sha256_file(output / "run_manifest.json"))
        acknowledged.append(checkpoint.name)

    monkeypatch.setattr(durability, "wait_for_durable_ack", ack)
    monkeypatch.setattr(Trainer, "train", lambda *a, **kw: pytest.fail("finalize must not train"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("must not update optimizer"))
    result = run_training(config, resume_from_checkpoint=str(checkpoint), durability_timeout_seconds=1,
                          finalize_only=True)
    assert result["status"] == "completed"
    assert result["global_step"] == result["planned_global_steps"] == 2
    assert result["metrics"]["training"]["optimizer_updates_this_invocation"] == 0
    assert acknowledged == ["checkpoint-1", "checkpoint-2"]
    assert (checkpoint / "checkpoint_manifest.json").read_bytes() == manifest_before
    expected = load_file(str(Path(reference["artifact"]) / "model.safetensors"))
    actual = load_file(str(Path(result["artifact"]) / "model.safetensors"))
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    with pytest.raises(ValueError, match="Final artifacts"):
        run_training(config, resume_from_checkpoint=str(checkpoint), durability_timeout_seconds=1,
                     finalize_only=True)
