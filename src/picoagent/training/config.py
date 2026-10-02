"""Strict, serializable training configuration."""
from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    model_revision: str | None
    dataset_manifest: str
    output_dir: str
    training_mode: str = "full"
    precision: str = "auto"
    device: str = "auto"
    max_seq_length: int = 2048
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    save_steps: int = 10
    eval_steps: int | None = None
    checkpoint_before_eval: bool = False
    checkpoint_interval_seconds: float | None = 300.0
    async_checkpoint_upload: bool = False
    async_checkpoint_max_local: int = 2
    logging_steps: int = 10
    seed: int = 20261001
    deterministic: bool = True
    smoke_test: bool = False
    allow_native_teacher_observed: bool = False
    allow_artificial_action_plans: bool = False
    prepared_manifest: str | None = None
    prepared_manifest_sha256: str | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    def __post_init__(self) -> None:
        for key in ("model_id", "dataset_manifest", "output_dir"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("gradient_checkpointing", "deterministic", "smoke_test", "allow_native_teacher_observed", "allow_artificial_action_plans", "checkpoint_before_eval", "async_checkpoint_upload"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a boolean")
        if type(self.async_checkpoint_max_local) is not int or self.async_checkpoint_max_local < 2:
            raise ValueError("async_checkpoint_max_local must be an integer >= 2")
        if self.async_checkpoint_upload and self.checkpoint_interval_seconds is None:
            raise ValueError("async_checkpoint_upload requires checkpoint_interval_seconds")
        if self.async_checkpoint_upload and not self.checkpoint_before_eval:
            raise ValueError("async_checkpoint_upload requires checkpoint_before_eval")
        if self.device not in {"auto", "cpu", "cuda", "xla"}:
            raise ValueError("device must be auto, cpu, cuda, or xla")
        if self.eval_steps is not None and (type(self.eval_steps) is not int or self.eval_steps <= 0):
            raise ValueError("eval_steps must be a positive integer or null")
        if (self.prepared_manifest is None) != (self.prepared_manifest_sha256 is None):
            raise ValueError("prepared_manifest and prepared_manifest_sha256 must be supplied together")
        if self.prepared_manifest is not None:
            if not isinstance(self.prepared_manifest, str) or not self.prepared_manifest.strip():
                raise ValueError("prepared_manifest must be a nonempty path")
            if not isinstance(self.prepared_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.prepared_manifest_sha256):
                raise ValueError("prepared_manifest_sha256 must be an explicitly pinned SHA256 digest")
        if self.training_mode not in {"full", "qlora"}:
            raise ValueError("training_mode must be full or qlora; QLoRA is not full fine-tuning")
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision must be auto, bf16, fp16, or fp32")
        if not self.smoke_test:
            if not isinstance(self.model_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
                raise ValueError("Production model_revision must be an immutable 40-character lowercase commit SHA")
            if Path(self.model_id).exists():
                raise ValueError("Production model_id must be a Hub ID with a pinned revision; local paths are smoke-only")
        elif not Path(self.model_id).is_dir():
            raise ValueError("smoke_test requires an existing local random-model directory")
        for key in ("max_seq_length", "per_device_batch_size", "gradient_accumulation_steps", "save_steps", "logging_steps", "lora_rank", "lora_alpha"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        if self.checkpoint_interval_seconds is not None and (not isinstance(self.checkpoint_interval_seconds, (int, float)) or isinstance(self.checkpoint_interval_seconds, bool) or not math.isfinite(self.checkpoint_interval_seconds) or self.checkpoint_interval_seconds <= 0):
            raise ValueError("checkpoint_interval_seconds must be finite positive seconds or null")
        for key in ("learning_rate", "num_train_epochs"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("weight_decay", "warmup_ratio", "lora_dropout"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{key} must be finite and in [0, 1)")
        if self.allow_artificial_action_plans and not self.allow_native_teacher_observed:
            raise ValueError("Artificial plans require the underlying native-evidence opt-in")
        if self.smoke_test and self.training_mode != "full":
            raise ValueError("CPU pipeline smoke tests use full mode only")
        if self.smoke_test and self.allow_native_teacher_observed:
            raise ValueError("Smoke fixtures cannot opt into native teacher production data")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        if not isinstance(payload, dict):
            raise ValueError("Training config must be a JSON object")
        unknown = set(payload) - {field.name for field in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training config keys: {sorted(unknown)}")
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def select_precision(requested: str, *, cuda_available: bool, bf16_supported: bool, xla_available: bool = False) -> str:
    """CUDA uses BF16 when supported, FP16 otherwise; CPU/MPS stays FP32."""
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"Unknown precision: {requested}")
    if xla_available:
        if requested == "fp16":
            raise ValueError("XLA TPU supports bf16 or fp32 here, not fp16")
        return "bf16" if requested in {"auto", "bf16"} else "fp32"
    if requested == "auto":
        return "bf16" if cuda_available and bf16_supported else "fp16" if cuda_available else "fp32"
    if requested == "bf16" and not (cuda_available and bf16_supported):
        raise ValueError("bf16 requested but CUDA hardware does not report BF16 support")
    if requested == "fp16" and not cuda_available:
        raise ValueError("fp16 requested without CUDA; use fp32 for the CPU smoke test")
    return requested
