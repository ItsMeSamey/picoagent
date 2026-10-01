"""Training dispatch tests; mocks here are not native execution evidence."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from picoagent.training.config import TrainingConfig
from picoagent.training.data import verify_dataset


def config(**kwargs):
    return TrainingConfig(model_id="example/model", model_revision="a" * 40,
                          dataset_manifest="unused.json", output_dir="unused", **kwargs)


def test_native_config_is_opt_in_and_strictly_boolean():
    assert config().allow_native_teacher_observed is False
    assert config(allow_native_teacher_observed=True).allow_native_teacher_observed is True
    with pytest.raises(ValueError, match="boolean"):
        config(allow_native_teacher_observed="yes")


def test_native_manifest_rejected_before_verifier_without_opt_in(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "picoagent.native_teacher.dataset.v1"}))
    with pytest.raises(ValueError, match="explicit"):
        verify_dataset(path)
    with pytest.raises(ValueError, match="smoke"):
        verify_dataset(path, allow_native_teacher=True, allow_smoke=True)


def test_native_dispatch_uses_strict_evidence_verifier(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    manifest = {"schema": "picoagent.native_teacher.dataset.v1", "lockbox_used": False}
    path.write_text(json.dumps(manifest))
    records = {split: [{"trace_id": split, "task_id": split, "family": split,
                       "template_id": split, "messages": [{"role": "assistant", "content": split}]}]
               for split in ("train", "dev")}
    verifier = Mock(return_value=(manifest, records))
    monkeypatch.setitem(sys.modules, "picoagent.data.native_admission",
                        SimpleNamespace(verify_native_snapshot=verifier))
    assert verify_dataset(path, allow_native_teacher=True) == (manifest, records)
    verifier.assert_called_once_with(path.resolve(), allow_native_teacher=True)
    verifier.side_effect = ValueError("raw receipt integrity failed")
    with pytest.raises(ValueError, match="receipt integrity"):
        verify_dataset(path, allow_native_teacher=True)


def test_native_dispatch_still_rejects_test_split(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    manifest = {"schema": "picoagent.native_teacher.dataset.v1", "lockbox_used": False}
    path.write_text(json.dumps(manifest))
    monkeypatch.setitem(sys.modules, "picoagent.data.native_admission", SimpleNamespace(
        verify_native_snapshot=lambda *args, **kwargs: (manifest, {"train": [], "dev": [], "test": []})))
    with pytest.raises(ValueError, match="lockbox/test"):
        verify_dataset(path, allow_native_teacher=True)


def test_artificial_plan_config_needs_both_boolean_opt_ins():
    assert config().allow_artificial_action_plans is False
    with pytest.raises(ValueError, match="native-evidence"):
        config(allow_artificial_action_plans=True)
    with pytest.raises(ValueError, match="boolean"):
        config(allow_native_teacher_observed=True, allow_artificial_action_plans="yes")
    assert config(allow_native_teacher_observed=True, allow_artificial_action_plans=True).allow_artificial_action_plans


def test_artificial_manifest_requires_explicit_opt_ins(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "picoagent.artificial_action_plan.dataset.v1"}))
    with pytest.raises(ValueError, match="explicit"):
        verify_dataset(path)
    with pytest.raises(ValueError, match="explicit"):
        verify_dataset(path, allow_native_teacher=True)
    with pytest.raises(ValueError, match="smoke"):
        verify_dataset(path, allow_smoke=True, allow_native_teacher=True, allow_artificial_action_plans=True)
