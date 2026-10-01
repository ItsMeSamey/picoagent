"""Explicit backend resolution. A requested TPU can never silently become CPU."""
from __future__ import annotations

import importlib.util
import os
from typing import Any


def resolve_device(requested: str, torch: Any) -> tuple[str, dict[str, Any]]:
    if requested not in {"auto", "cpu", "cuda", "xla"}:
        raise ValueError(f"Unsupported device: {requested}")
    tpu_environment = any(os.environ.get(key) for key in ("TPU_NAME", "COLAB_TPU_ADDR", "TPU_PROCESS_ADDRESSES")) or os.environ.get("PJRT_DEVICE") == "TPU"
    try_xla = requested == "xla" or (requested == "auto" and (tpu_environment or importlib.util.find_spec("torch_xla") is not None))
    if try_xla:
        if importlib.util.find_spec("torch_xla") is None:
            raise RuntimeError("TPU requested/detected but matched torch_xla is missing; do not upgrade torch independently")
        if requested == "xla":
            os.environ.setdefault("PJRT_DEVICE", "TPU")
        import torch_xla
        import torch_xla.runtime as xr
        if xr.device_type() == "TPU":
            if int(xr.world_size()) != 1:
                raise ValueError("This baseline currently supports a single TPU process/device")
            if any(os.environ.get(name, "0") not in {"0", "", "false", "False"} for name in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16", "XLA_USE_F16")):
                raise ValueError("Unset global XLA dtype-casting variables; this trainer uses explicit operation-level BF16 autocast")
            os.environ["USE_TORCH_XLA"] = "1"
            os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
            return "xla", {"device": "TPU", "xla_version": torch_xla.__version__, "xla_device_type": xr.device_type(), "xla_world_size": xr.world_size()}
        if requested == "xla" or tpu_environment:
            raise RuntimeError("Requested TPU backend was not available; no silent CPU fallback")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    device = "cuda" if requested in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    os.environ["USE_TORCH_XLA"] = "0"
    return device, {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu"}
