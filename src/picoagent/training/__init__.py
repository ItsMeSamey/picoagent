"""Reproducible supervised training; heavyweight dependencies are imported lazily."""

from .config import TrainingConfig
from .data import prepare_dataset, verify_dataset

__all__ = ["TrainingConfig", "prepare_dataset", "verify_dataset"]
