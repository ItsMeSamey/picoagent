"""Segment-boundary and output-budget checks without launching ML training."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from picoagent.training import train
from picoagent.training.config import TrainingConfig


def config():
    return TrainingConfig(model_id="org/model", model_revision="a" * 40,
                          dataset_manifest="unused.json", output_dir="unused")


def test_segment_runtime_arguments_are_positive_and_outside_config_identity():
    with pytest.raises(ValueError, match="positive integer"):
        train.run_training(config(), segment_steps=True)
    with pytest.raises(ValueError, match="positive integer"):
        train.run_training(config(), segment_steps=0)
    with pytest.raises(ValueError, match="requires an explicit segmented run"):
        train.run_training(config(), output_budget_bytes=20_000_000_000)
    with pytest.raises(ValueError, match="requires output_budget_bytes"):
        train.run_training(config(), segment_steps=10, output_budget_root="/tmp")
    assert "segment_steps" not in config().as_dict()
    assert "output_budget_bytes" not in config().as_dict()


def test_output_budget_counts_whole_saved_tree_and_reserves_checkpoint_final_and_margin(tmp_path):
    saved = tmp_path / "working"
    output = saved / "picoagent-training"
    output.mkdir(parents=True)
    (saved / "training.log").write_bytes(b"log")
    (output / "run_manifest.json").write_bytes(b"manifest")
    with patch.object(train.shutil, "disk_usage", return_value=SimpleNamespace(free=100_000_000_000)):
        result = train._output_budget_reservation(
            output, parameters=1_000, budget_bytes=10_000_000_000, budget_root=saved)
    assert result["existing_output_bytes"] == len(b"log") + len(b"manifest")
    assert result["checkpoint_reserve_bytes"] > 3 * 1_000 * 4
    assert result["final_model_reserve_bytes"] == 1_000 * 4
    assert result["projected_output_bytes"] <= result["output_budget_bytes"]
    assert result["filesystem_free_bytes"] == 100_000_000_000


def test_output_budget_fails_closed_on_projected_cap_or_filesystem_free_space(tmp_path):
    saved = tmp_path / "working"
    output = saved / "run"
    output.mkdir(parents=True)
    (saved / "existing.bin").write_bytes(b"existing output")
    with patch.object(train.shutil, "disk_usage", return_value=SimpleNamespace(free=100_000_000_000)):
        with pytest.raises(OSError, match="Insufficient bounded output budget"):
            train._output_budget_reservation(output, parameters=1_000, budget_bytes=100,
                                             budget_root=saved)
    with patch.object(train.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
        with pytest.raises(OSError, match="Insufficient filesystem free space"):
            train._output_budget_reservation(output, parameters=1_000, budget_bytes=10_000_000_000,
                                             budget_root=saved)


def test_output_budget_refuses_symlinks_and_outside_root(tmp_path):
    saved = tmp_path / "working"
    output = saved / "run"
    output.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("not saved output")
    wrong_root = tmp_path / "outside-root"
    wrong_root.mkdir()
    (saved / "link").symlink_to(outside)
    with patch.object(train.shutil, "disk_usage", return_value=SimpleNamespace(free=100_000_000_000)):
        with pytest.raises(ValueError, match="symlink"):
            train._output_budget_reservation(output, parameters=1, budget_bytes=10_000_000_000,
                                             budget_root=saved)
        with pytest.raises(ValueError, match="must contain"):
            train._output_budget_reservation(output, parameters=1, budget_bytes=10_000_000_000,
                                             budget_root=wrong_root)


def test_continue_through_requires_explicit_segment_and_boolean():
    with pytest.raises(ValueError, match="requires an explicit segmented run"):
        train.run_training(config(), continue_through_checkpoints=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        train.run_training(config(), segment_steps=1, continue_through_checkpoints=1)
    assert "continue_through_checkpoints" not in config().as_dict()


def test_training_cli_preserves_legacy_default_and_wires_continuation(tmp_path, capsys):
    import json
    from picoagent.training.__main__ import main
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config().as_dict()))
    base = ["train", "--config", str(config_path), "--segment-steps", "7"]
    with patch.object(train, "run_training", return_value={}) as run:
        assert main(base) == 0
        assert run.call_args.kwargs["continue_through_checkpoints"] is False
        assert main(base + ["--continue-through-checkpoints"]) == 0
        assert run.call_args.kwargs["continue_through_checkpoints"] is True
        assert run.call_args.kwargs["segment_steps"] == 7
    capsys.readouterr()
