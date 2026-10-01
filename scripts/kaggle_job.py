#!/usr/bin/env python3
"""Build a local private Kaggle kernel package; never submit it automatically.

Large/private inputs are packaged as a separate chunked Kaggle Dataset and
referenced by the kernel. Small inline archives remain only as a fixture path.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import os
import pathlib
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from colab_run import archive_source
from source_staging import matches, pack_archive

INLINE_ARCHIVE_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_KAGGLE_CHUNK_BYTES = 16 * 1024 * 1024
KAGGLE_CHUNK_BYTES_LIMIT = 32 * 1024 * 1024
MAX_GPU_SMOKE_SECONDS = 60
KAGGLE_OUTPUT_BUDGET_BYTES = 20_000_000_000
KAGGLE_ACCELERATOR = "NvidiaTeslaT4"
SUPPORTED_DATASET_LICENSES = {"unknown", "copyright-authors", "other"}
HANDLE_SEGMENT = re.compile(r"[A-Za-z0-9_-]{3,50}\Z")
PINNED_PACKAGES = ("transformers==5.18.0", "accelerate==1.15.0", "tokenizers==0.23.2")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _validate_segment(value: str, label: str) -> None:
    if not isinstance(value, str) or not HANDLE_SEGMENT.fullmatch(value):
        raise ValueError(f"{label} must be a simple Kaggle identifier (3–50 letters, digits, '_' or '-')")


def _validate_handle(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or value.count("/") != 1:
        raise ValueError("Private input dataset must be specified as owner/dataset-slug")
    owner, slug = value.split("/", 1)
    _validate_segment(owner, "dataset owner")
    _validate_segment(slug, "dataset slug")
    return owner, slug


def _relative_inside_root(root: Path, value: str | Path, label: str) -> tuple[Path, str]:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    # Resolve only after rejecting a symlink in every existing component.
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"{label} must be an existing file inside the source root")
    relative = resolved.relative_to(root).as_posix()
    if relative.startswith("../") or "\\" in relative:
        raise ValueError(f"{label} has an unsafe relative path")
    return resolved, relative


def _validate_native_sft_inputs(root: Path, config: str, dataset_manifest: str | Path) -> tuple[Path, str, str, dict[str, Any]]:
    config_path, config_relative = _relative_inside_root(root, config, "Training config")
    manifest_path, manifest_relative = _relative_inside_root(root, dataset_manifest, "Selected dataset manifest")
    if not manifest_relative.startswith("data/"):
        raise ValueError("Selected dataset manifest must be under data/")
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    if config_value.get("dataset_manifest") != manifest_relative:
        raise ValueError("Training config dataset_manifest must match the selected private input snapshot")
    if config_value.get("smoke_test") is True:
        raise ValueError("Private-input package mode is for native SFT, not a smoke-test config")
    if config_value.get("allow_native_teacher_observed") is not True:
        raise ValueError("Native SFT config must explicitly allow native_teacher_observed data")
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest_value.get("schema")
    if schema not in {"picoagent.native_teacher.dataset.v1", "picoagent.artificial_action_plan.dataset.v1"}:
        raise ValueError("Private-input dataset must be a sealed native-teacher snapshot")
    if manifest_value.get("lockbox_used") is not False or manifest_value.get("smoke_only") is True:
        raise ValueError("Native SFT input must exclude lockbox/test data and pipeline-smoke data")
    if schema == "picoagent.native_teacher.dataset.v1":
        splits = manifest_value.get("splits")
        if not isinstance(splits, dict) or set(splits) != {"train", "dev"}:
            raise ValueError("Native-teacher SFT snapshot must have exactly train and dev splits")
        if any(not isinstance(record, dict) or type(record.get("records")) is not int or record["records"] <= 0
               for record in splits.values()):
            raise ValueError("Native-teacher train/dev splits must have positive integer record counts")
        if "counts" in manifest_value:
            raise ValueError("Native-teacher snapshot must use the splits train/dev contract")
    else:
        counts = manifest_value.get("counts")
        if not isinstance(counts, dict) or set(counts) != {"train", "dev"}:
            raise ValueError("Artificial-action-plan SFT snapshot must have exactly train and dev counts")
        if any(type(count) is not int or count <= 0 for count in counts.values()):
            raise ValueError("Artificial-action-plan train/dev counts must be positive integers")
        if "splits" in manifest_value:
            raise ValueError("Artificial-action-plan snapshot must use the counts train/dev contract")
    if schema == "picoagent.artificial_action_plan.dataset.v1" and config_value.get("allow_artificial_action_plans") is not True:
        raise ValueError("Artificial action-plan input requires explicit allow_artificial_action_plans opt-in")
    return config_path, config_relative, manifest_relative, manifest_value


def _safe_tree_no_symlinks(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Resume source must be a real directory, not a symlink")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Resume source contains a symlink")


def _safe_path_no_symlink_components(path: Path) -> Path:
    candidate = path.absolute()
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("Resume path traverses a symlink")
    return candidate


def _code_tree_sha256(root: Path) -> str:
    """Match training.provenance.code_evidence() for a supplied source tree."""
    files: dict[str, str] = {}
    package = root / "src/picoagent"
    for path in sorted(package.rglob("*.py")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha256_file(path)
    project_file = root / "pyproject.toml"
    if project_file.is_file():
        files[project_file.relative_to(root).as_posix()] = _sha256_file(project_file)
    for path in sorted((root / "configs").glob("*.json")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _sha256_file(path)
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(canonical.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_resume_inventory(directory: Path, expected: dict[str, Any], label: str) -> int:
    _safe_tree_no_symlinks(directory)
    normalized: dict[str, str] = {}
    for relative, record in expected.items():
        posix = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
        digest = record.get("sha256") if isinstance(record, dict) else record
        if (posix is None or posix.is_absolute() or "\\" in relative
                or any(part in {"", ".", ".."} for part in relative.split("/"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"{label} inventory has unsafe paths or hashes")
        normalized[relative] = digest
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != set(normalized):
        raise ValueError(f"{label} evidence tree has missing or unexpected files")
    total = 0
    for relative, digest in normalized.items():
        path = directory / relative
        if not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)) or _sha256_file(path) != digest:
            raise ValueError(f"{label} evidence file changed: {relative}")
        total += path.stat().st_size
    return total


def _validate_resume_artifact(root: Path, config: str, manifest_relative: str, *,
                              run_dir: Path, checkpoint_name: str,
                              run_manifest_sha256: str,
                              checkpoint_manifest_sha256: str) -> dict[str, Any]:
    if not re.fullmatch(r"checkpoint-[1-9][0-9]*", checkpoint_name):
        raise ValueError("Resume checkpoint must use a positive checkpoint-N name")
    for value, label in ((run_manifest_sha256, "run manifest"),
                         (checkpoint_manifest_sha256, "checkpoint manifest")):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"Caller must supply an exact lowercase SHA256 for the {label}")
    run_dir = _safe_path_no_symlink_components(run_dir).resolve(strict=True)
    _safe_tree_no_symlinks(run_dir)
    run_manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "run_status.json"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    checkpoint = run_dir / checkpoint_name
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    for path in (run_manifest_path, status_path, dataset_manifest_path, checkpoint_manifest_path):
        if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(run_dir):
            raise ValueError(f"Missing or unsafe resume artifact: {path.name}")
    if _sha256_file(run_manifest_path) != run_manifest_sha256:
        raise ValueError("Caller-supplied run-manifest SHA256 does not match retrieved output")
    if _sha256_file(checkpoint_manifest_path) != checkpoint_manifest_sha256:
        raise ValueError("Caller-supplied checkpoint-manifest SHA256 does not match retrieved output")

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_manifest.get("schema") != "picoagent.training.run.v1":
        raise ValueError("Resume run manifest has an unsupported schema")
    if run_status.get("status") != "paused":
        raise ValueError("Resume source must be paused, not running, failed, or completed")
    checkpoint_step = int(checkpoint_name.split("-")[1])
    if run_status.get("checkpoint") != checkpoint_name or run_status.get("global_step") != checkpoint_step:
        raise ValueError("Paused run status does not identify the selected checkpoint step")
    planned_steps = run_status.get("planned_global_steps")
    if type(planned_steps) is not int or planned_steps <= checkpoint_step:
        raise ValueError("Paused run status must show that the original schedule is incomplete")
    if run_status.get("run_manifest_sha256") != run_manifest_sha256:
        raise ValueError("Paused run status is not bound to the pinned run manifest")
    if run_status.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha256:
        raise ValueError("Paused run status is not bound to the pinned checkpoint manifest")
    if any((run_dir / name).exists() for name in ("final-model", "final-adapter", "final_artifacts.json")):
        raise ValueError("Completed final artifacts cannot be used as a paused segment")

    numeric_checkpoints = []
    for path in run_dir.iterdir():
        match = re.fullmatch(r"checkpoint-([0-9]+)", path.name)
        if match:
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Resume run contains an unsafe checkpoint path")
            numeric_checkpoints.append(int(match.group(1)))
    if not numeric_checkpoints or max(numeric_checkpoints) != checkpoint_step:
        raise ValueError("Selected checkpoint must be the latest checkpoint; unexpected newer state exists")

    current_dataset_sha = _sha256_file(root / manifest_relative)
    if _sha256_file(dataset_manifest_path) != current_dataset_sha:
        raise ValueError("Retrieved run's immutable dataset manifest differs from the current pinned input")
    identity = run_manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("dataset_manifest_sha256") != current_dataset_sha:
        raise ValueError("Run identity is bound to a different dataset manifest")
    if identity.get("source_tree_sha256") != _code_tree_sha256(root):
        raise ValueError("Run identity is bound to different source code/configuration")

    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from picoagent.training.config import TrainingConfig

    effective_config = TrainingConfig.load(root / config).as_dict()
    effective_config["output_dir"] = "/kaggle/working/picoagent-training"
    effective_config["dataset_manifest"] = f"/kaggle/temp/picoagent/{manifest_relative}"
    effective_config["device"] = "cuda"
    expected_config_identity = {key: value for key, value in effective_config.items()
                                if key not in {"output_dir", "dataset_manifest"}}
    original_config = run_manifest.get("original_config")
    original_config_identity = ({key: value for key, value in original_config.items()
                                 if key not in {"output_dir", "dataset_manifest"}}
                                if isinstance(original_config, dict) else None)
    if (original_config_identity != expected_config_identity
            or identity.get("config") != expected_config_identity):
        raise ValueError("Resume config differs from the original immutable training schedule")

    code_files = run_manifest.get("code", {}).get("files")
    tokenizer_files = run_manifest.get("tokenizer_files")
    if not isinstance(code_files, dict) or not code_files or not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise ValueError("Run manifest lacks immutable source/tokenizer snapshot inventories")
    snapshot_bytes = 0
    snapshot_bytes += _verify_resume_inventory(run_dir / "source_snapshot", code_files, "source")
    snapshot_bytes += _verify_resume_inventory(run_dir / "tokenizer_snapshot", tokenizer_files, "tokenizer")
    dataset_manifest_value = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("dataset") != dataset_manifest_value:
        raise ValueError("Run manifest dataset metadata differs from the frozen dataset manifest")
    dataset_files = dataset_manifest_value.get("files")
    if isinstance(dataset_files, dict):
        dataset_inventory = {name: info for name, info in dataset_files.items()}
    else:
        splits = dataset_manifest_value.get("splits")
        if not isinstance(splits, dict) or not splits:
            raise ValueError("Frozen dataset manifest has no verifiable file inventory")
        dataset_inventory = {entry["path"]: {"sha256": entry["sha256"]} for entry in splits.values()}
    dataset_inventory["manifest.json"] = {"sha256": current_dataset_sha}
    snapshot_bytes += _verify_resume_inventory(run_dir / "dataset_snapshot", dataset_inventory, "dataset")

    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    if (checkpoint_manifest.get("schema") != "picoagent.checkpoint.v1"
            or checkpoint_manifest.get("run_manifest_sha256") != run_manifest_sha256
            or not isinstance(checkpoint_manifest.get("files"), dict)):
        raise ValueError("Checkpoint manifest is invalid or belongs to another run")
    expected_files = checkpoint_manifest["files"]
    actual_files: set[str] = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("Checkpoint contains a symlink")
        relative = path.relative_to(checkpoint).as_posix()
        if path.is_file() and relative != "checkpoint_manifest.json":
            actual_files.add(relative)
    if actual_files != set(expected_files):
        raise ValueError("Checkpoint has missing or unexpected files")
    for relative, digest in expected_files.items():
        if (not isinstance(relative, str) or "\\" in relative
                or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in relative.split("/"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Checkpoint manifest contains an unsafe path or hash")
        file_path = checkpoint / relative
        if not file_path.resolve(strict=True).is_relative_to(checkpoint.resolve(strict=True)) or _sha256_file(file_path) != digest:
            raise ValueError("Checkpoint file failed hash/path verification")
    for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        if required not in expected_files:
            raise ValueError(f"Full resume checkpoint is missing {required}")
    if not any(name in expected_files for name in ("model.safetensors", "pytorch_model.bin",
                                                    "adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Full resume checkpoint is missing model/adapter weights")
    if not any(name.startswith("rng_state") and name.endswith(".pth") for name in expected_files):
        raise ValueError("Full resume checkpoint is missing RNG state")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if trainer_state.get("global_step") != checkpoint_step or trainer_state.get("max_steps") != planned_steps:
        raise ValueError("Checkpoint trainer state does not match paused run schedule/status")

    return {
        "kernel_source": None,
        "run_manifest_sha256": run_manifest_sha256,
        "run_status_sha256": _sha256_file(status_path),
        "dataset_manifest_sha256": current_dataset_sha,
        "checkpoint": checkpoint_name,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "global_step": checkpoint_step,
        "planned_global_steps": planned_steps,
        "checkpoint_bytes": sum((checkpoint / name).stat().st_size for name in expected_files),
        "snapshot_bytes": snapshot_bytes,
    }


def _restore_resume_tree(input_root, output_dir, pins):
    """Verify a mounted prior kernel output and copy only its latest resumable state."""
    input_root = pathlib.Path(input_root).absolute()
    output_dir = pathlib.Path(output_dir).absolute()
    input_component = pathlib.Path(input_root.anchor)
    for part in input_root.parts[1:]:
        input_component = input_component / part
        if input_component.is_symlink():
            raise ValueError("Mounted kernel-source input traverses a symlink")
    if not input_root.is_dir():
        raise ValueError("Mounted kernel-source input is missing or unsafe")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Resume destination must be a new output run directory")
    current = pathlib.Path(output_dir.anchor)
    for part in output_dir.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("Resume destination traverses a symlink")

    def file_hash(path):
        digest = hashlib.sha256()
        with pathlib.Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def reject_symlinks(root):
        root = pathlib.Path(root)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Prior run directory is missing or unsafe")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Prior run output contains a symlink")

    run_dirs = []
    for candidate in input_root.rglob("run_manifest.json"):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if not candidate.resolve(strict=True).is_relative_to(input_root.resolve(strict=True)):
            raise ValueError("Mounted run manifest escapes the kernel-source input")
        if file_hash(candidate) == pins["run_manifest_sha256"]:
            run_dirs.append(candidate.parent)
    if len(run_dirs) != 1:
        raise ValueError("Expected exactly one mounted prior run matching the pinned run-manifest hash")
    run_dir = run_dirs[0]
    reject_symlinks(run_dir)
    if (run_dir == output_dir or run_dir.is_relative_to(output_dir)
            or output_dir.is_relative_to(run_dir)):
        raise ValueError("Resume destination must be separate from immutable prior output")

    run_manifest_path = run_dir / "run_manifest.json"
    status_path = run_dir / "run_status.json"
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    checkpoint_name = pins["checkpoint"]
    if not re.fullmatch(r"checkpoint-[1-9][0-9]*", checkpoint_name):
        raise ValueError("Pinned resume checkpoint name is invalid")
    checkpoint_step = int(checkpoint_name.split("-")[1])
    checkpoint = run_dir / checkpoint_name
    checkpoint_manifest_path = checkpoint / "checkpoint_manifest.json"
    for path in (status_path, dataset_manifest_path, checkpoint_manifest_path):
        if not path.is_file() or not path.resolve(strict=True).is_relative_to(run_dir.resolve(strict=True)):
            raise ValueError(f"Prior run is missing or escaping required artifact {path.name}")
    if file_hash(run_manifest_path) != pins["run_manifest_sha256"]:
        raise ValueError("Prior run manifest changed after source selection")
    if file_hash(status_path) != pins["run_status_sha256"]:
        raise ValueError("Prior run status changed after source selection")
    if file_hash(dataset_manifest_path) != pins["dataset_manifest_sha256"]:
        raise ValueError("Prior immutable dataset manifest changed after source selection")
    if file_hash(checkpoint_manifest_path) != pins["checkpoint_manifest_sha256"]:
        raise ValueError("Prior checkpoint manifest changed after source selection")

    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_manifest.get("schema") != "picoagent.training.run.v1":
        raise ValueError("Prior run manifest schema is unsupported")
    if (status.get("status") != "paused" or status.get("checkpoint") != checkpoint_name
            or status.get("global_step") != checkpoint_step
            or status.get("planned_global_steps") != pins["planned_global_steps"]
            or status.get("global_step") >= status.get("planned_global_steps", 0)
            or status.get("run_manifest_sha256") != pins["run_manifest_sha256"]
            or status.get("checkpoint_manifest_sha256") != pins["checkpoint_manifest_sha256"]):
        raise ValueError("Prior run is not in the exact pinned, incomplete paused state")
    if any((run_dir / name).exists() for name in ("final-model", "final-adapter", "final_artifacts.json")):
        raise ValueError("Completed/finalized run cannot be restored as an incomplete segment")

    def verify_inventory(directory, expected, label):
        directory = pathlib.Path(directory)
        reject_symlinks(directory)
        normalized = {}
        for relative, record in expected.items():
            posix = pathlib.PurePosixPath(relative) if isinstance(relative, str) else None
            digest = record.get("sha256") if isinstance(record, dict) else record
            if (posix is None or posix.is_absolute() or "\\" in relative
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError(label + " inventory has unsafe paths or hashes")
            normalized[relative] = digest
        actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
        if actual != set(normalized):
            raise ValueError(label + " evidence tree has missing or unexpected files")
        total = 0
        for relative, digest in normalized.items():
            item = directory / relative
            if not item.resolve(strict=True).is_relative_to(directory.resolve(strict=True)) or file_hash(item) != digest:
                raise ValueError(label + " evidence file changed: " + relative)
            total += item.stat().st_size
        return total

    code_files = run_manifest.get("code", {}).get("files")
    tokenizer_files = run_manifest.get("tokenizer_files")
    if not isinstance(code_files, dict) or not code_files or not isinstance(tokenizer_files, dict) or not tokenizer_files:
        raise ValueError("Prior run lacks source/tokenizer snapshot inventories")
    snapshot_bytes = verify_inventory(run_dir / "source_snapshot", code_files, "source")
    snapshot_bytes += verify_inventory(run_dir / "tokenizer_snapshot", tokenizer_files, "tokenizer")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("dataset") != dataset_manifest:
        raise ValueError("Prior run dataset metadata differs from its frozen dataset manifest")
    dataset_files = dataset_manifest.get("files")
    if isinstance(dataset_files, dict):
        dataset_inventory = dict(dataset_files)
    else:
        splits = dataset_manifest.get("splits")
        if not isinstance(splits, dict) or not splits:
            raise ValueError("Prior dataset manifest has no verifiable file inventory")
        dataset_inventory = {entry["path"]: {"sha256": entry["sha256"]} for entry in splits.values()}
    dataset_inventory["manifest.json"] = {"sha256": pins["dataset_manifest_sha256"]}
    snapshot_bytes += verify_inventory(run_dir / "dataset_snapshot", dataset_inventory, "dataset")

    checkpoint_steps = []
    for path in run_dir.iterdir():
        match = re.fullmatch(r"checkpoint-([0-9]+)", path.name)
        if match:
            if path.is_symlink() or not path.is_dir():
                raise ValueError("Prior run contains an unsafe checkpoint path")
            checkpoint_steps.append(int(match.group(1)))
    if not checkpoint_steps or max(checkpoint_steps) != checkpoint_step:
        raise ValueError("Prior run contains an unexpected newer checkpoint")

    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    if (checkpoint_manifest.get("schema") != "picoagent.checkpoint.v1"
            or checkpoint_manifest.get("run_manifest_sha256") != pins["run_manifest_sha256"]
            or not isinstance(checkpoint_manifest.get("files"), dict)):
        raise ValueError("Prior checkpoint manifest is invalid or belongs to another run")
    expected_files = checkpoint_manifest["files"]
    actual_files = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError("Prior checkpoint contains a symlink")
        relative = path.relative_to(checkpoint).as_posix()
        if path.is_file() and relative != "checkpoint_manifest.json":
            actual_files.add(relative)
    if actual_files != set(expected_files):
        raise ValueError("Prior checkpoint has missing or unexpected files")
    for relative, expected_hash in expected_files.items():
        if (not isinstance(relative, str) or "\\" in relative or pathlib.PurePosixPath(relative).is_absolute()
                or any(part in {"", ".", ".."} for part in relative.split("/"))
                or not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)):
            raise ValueError("Checkpoint manifest contains an unsafe path/hash")
        source = checkpoint / relative
        if not source.resolve(strict=True).is_relative_to(checkpoint.resolve(strict=True)) or file_hash(source) != expected_hash:
            raise ValueError("Prior checkpoint file failed hash/path verification")
    for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        if required not in expected_files:
            raise ValueError(f"Prior full checkpoint is missing {required}")
    if not any(name in expected_files for name in ("model.safetensors", "pytorch_model.bin",
                                                    "adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Prior checkpoint is missing model/adapter weights")
    if not any(name.startswith("rng_state") and name.endswith(".pth") for name in expected_files):
        raise ValueError("Prior checkpoint is missing RNG state")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != checkpoint_step or state.get("max_steps") != pins["planned_global_steps"]:
        raise ValueError("Prior checkpoint trainer state differs from the paused schedule")

    restore_bytes = (run_manifest_path.stat().st_size + dataset_manifest_path.stat().st_size
                     + status_path.stat().st_size + checkpoint_manifest_path.stat().st_size
                     + sum((checkpoint / name).stat().st_size for name in expected_files) + snapshot_bytes)
    working_root = output_dir.parent
    budget = pins.get("output_budget_bytes", 20_000_000_000)
    existing_working_bytes = 0
    for path in working_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Output working tree contains a symlink before resume staging")
        if path.is_file():
            existing_working_bytes += path.stat().st_size
    if existing_working_bytes + restore_bytes > budget:
        raise OSError("Restored evidence/checkpoint would exceed the bounded Kaggle output budget")
    if shutil.disk_usage(working_root).free < restore_bytes:
        raise OSError("Insufficient free space to restore prior run evidence and checkpoint")
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(run_manifest_path, output_dir / "run_manifest.json")
    shutil.copyfile(dataset_manifest_path, output_dir / "dataset_manifest.json")
    shutil.copyfile(status_path, output_dir / "run_status.json")
    shutil.copytree(run_dir / "source_snapshot", output_dir / "source_snapshot")
    shutil.copytree(run_dir / "tokenizer_snapshot", output_dir / "tokenizer_snapshot")
    shutil.copytree(run_dir / "dataset_snapshot", output_dir / "dataset_snapshot")
    shutil.copytree(checkpoint, output_dir / checkpoint_name)
    staged_checkpoint = output_dir / checkpoint_name
    if file_hash(output_dir / "run_manifest.json") != pins["run_manifest_sha256"]:
        raise ValueError("Staged run manifest failed pinned-hash verification")
    if file_hash(output_dir / "dataset_manifest.json") != pins["dataset_manifest_sha256"]:
        raise ValueError("Staged dataset manifest failed pinned-hash verification")
    if file_hash(output_dir / "run_status.json") != pins["run_status_sha256"]:
        raise ValueError("Staged paused status failed pinned-hash verification")
    if file_hash(staged_checkpoint / "checkpoint_manifest.json") != pins["checkpoint_manifest_sha256"]:
        raise ValueError("Staged checkpoint manifest failed pinned-hash verification")
    verify_inventory(output_dir / "source_snapshot", code_files, "staged source")
    verify_inventory(output_dir / "tokenizer_snapshot", tokenizer_files, "staged tokenizer")
    verify_inventory(output_dir / "dataset_snapshot", dataset_inventory, "staged dataset")
    for relative, expected_hash in expected_files.items():
        if file_hash(staged_checkpoint / relative) != expected_hash:
            raise ValueError("Staged checkpoint file failed hash verification: " + relative)
    staged_names = {path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()}
    if staged_names != {checkpoint_name}:
        raise ValueError("Resume staging copied an unexpected checkpoint")
    return str(staged_checkpoint)


def _dataset_metadata(handle: str, slug: str, license_name: str, license_description: str | None) -> dict[str, Any]:
    if license_name not in SUPPORTED_DATASET_LICENSES:
        raise ValueError("Dataset license must be explicitly set to unknown, copyright-authors, or other")
    if license_name == "other" and not license_description:
        raise ValueError("The 'other' license requires --dataset-license-description")
    title = (slug.replace("-", " ") + " input bundle")[:50]
    description = "Private input archive for an authorized Picoagent native-SFT kernel."
    if license_name == "other":
        description += f" License description: {license_description}"
    return {
        "id": handle,
        "title": title,
        "description": description,
        "licenses": [{"name": license_name}],
    }


def _source_staging_bootstrap() -> tuple[str, str]:
    helper_path = Path(__file__).with_name("source_staging.py")
    helper = helper_path.read_text(encoding="utf-8")
    return helper, _sha256_bytes(helper.encode("utf-8"))


def _base_program_prefix(helper: str, helper_sha256: str) -> str:
    return f'''#!/usr/bin/env python3
"""Generated Kaggle runner. It stages verified source before training."""
import hashlib, json, os, pathlib, re, shutil, signal, subprocess, sys, tarfile, tempfile, time
EXPECTED_STAGING_HELPER_SHA256 = {helper_sha256!r}
STAGING_HELPER_TEXT = {helper!r}
if hashlib.sha256(STAGING_HELPER_TEXT.encode("utf-8")).hexdigest() != EXPECTED_STAGING_HELPER_SHA256:
    raise RuntimeError("Bundled staging helper hash mismatch")
exec(compile(STAGING_HELPER_TEXT, "<pinned-source-staging>", "exec"), globals())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TORCH_XLA"] = "0"
output = pathlib.Path("/kaggle/working")
output.mkdir(parents=True, exist_ok=True)
project = pathlib.Path("/kaggle/temp/picoagent")
archive_path = pathlib.Path("/kaggle/temp/picoagent-source.tar.gz")
'''


def _inline_source_program(payload: str, source_info: dict[str, Any], config: str | None,
                           smoke_timeout_seconds: int, helper: str, helper_sha256: str) -> str:
    manifest = source_info["manifest"]
    source_manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    identity = {
        "schema": "picoagent.source-transfer.v1",
        "archive": {"sha256": source_info["sha256"], "bytes": source_info["bytes"]},
        "source_manifest_sha256": _sha256_bytes(source_manifest_bytes),
        "source_manifest_bytes": len(source_manifest_bytes),
        "extracted_bytes": sum(item["bytes"] for item in manifest["files"].values()),
    }
    prefix = _base_program_prefix(helper, helper_sha256)
    source_stage = f'''\n# Inline mode is reserved for tiny fixtures; archive size was checked by the builder.
import base64
blob = base64.b64decode({payload!r})
if hashlib.sha256(blob).hexdigest() != {source_info["sha256"]!r}:
    raise ValueError("Inline fixture archive failed SHA256 verification")
archive_path.parent.mkdir(parents=True, exist_ok=True)
with archive_path.open("xb") as target:
    target.write(blob)
    target.flush()
    os.fsync(target.fileno())
extract_source_archive(archive_path, str(project), {identity!r})
'''
    if config is None:
        return prefix + source_stage + _gpu_smoke_tail(smoke_timeout_seconds)
    return prefix + source_stage + _training_tail(config)


def _gpu_smoke_tail(timeout_seconds: int) -> str:
    runner = f'''import json, os, pathlib, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TORCH_XLA"] = "0"
output = pathlib.Path("/kaggle/working")
root = pathlib.Path("/kaggle/temp/picoagent")
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", {PINNED_PACKAGES[0]!r}, {PINNED_PACKAGES[1]!r}, {PINNED_PACKAGES[2]!r}], check=True)
import torch
hardware = {{"torch": torch.__version__, "cuda": torch.cuda.is_available(), "devices": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
(output / "hardware.json").write_text(json.dumps(hardware, indent=2))
if not hardware["cuda"]:
    raise RuntimeError("GPU unavailable; refusing silent CPU fallback")
os.environ["PYTHONPATH"] = str(root / "src")
command = [sys.executable, "-m", "picoagent.training", "smoke", "--device", "cuda", "--output-dir", "/kaggle/working/picoagent-smoke"]
with (output / "training.log").open("w") as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
sys.exit(result.returncode)
'''
    return f'''\n# Includes package install, hardware check, and one smoke command under a strict total budget.
smoke_runner = {runner!r}
smoke_path = output / "picoagent-gpu-smoke-runner.py"
smoke_path.write_text(smoke_runner, encoding="utf-8")
log_path = (output / "gpu-smoke-controller.log").open("w")
process = subprocess.Popen([sys.executable, str(smoke_path)], cwd=project, stdout=log_path, stderr=subprocess.STDOUT, start_new_session=True)
try:
    returncode = process.wait(timeout={timeout_seconds})
except subprocess.TimeoutExpired:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    (output / "job_status.json").write_text(json.dumps({{"returncode": 124, "timed_out": True, "smoke_only": True, "timeout_seconds": {timeout_seconds}}}))
    raise TimeoutError("The complete GPU smoke stage exceeded its hard timeout")
finally:
    log_path.close()
(output / "job_status.json").write_text(json.dumps({{"returncode": returncode, "smoke_only": True, "timeout_seconds": {timeout_seconds}}}))
if returncode:
    raise RuntimeError("GPU smoke failed; inspect gpu-smoke-controller.log and preserved outputs")
print((output / "training.log").read_text()[-12000:] if (output / "training.log").exists() else "GPU smoke completed")
'''


def _training_tail(config: str) -> str:
    return f'''\n# Native SFT only: this process trains from the selected snapshot and never executes learner tool calls.
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", {PINNED_PACKAGES[0]!r}, {PINNED_PACKAGES[1]!r}, {PINNED_PACKAGES[2]!r}], check=True)
import torch
hardware = {{"torch": torch.__version__, "cuda": torch.cuda.is_available(), "devices": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
print(json.dumps(hardware), flush=True)
(output / "hardware.json").write_text(json.dumps(hardware, indent=2))
if not hardware["cuda"]:
    raise RuntimeError("GPU unavailable; refusing silent CPU fallback")
os.environ["PYTHONPATH"] = str(project / "src")
config_path = project / {config!r}
training_config = json.loads(config_path.read_text())
training_config["output_dir"] = "/kaggle/working/picoagent-training"
training_config["device"] = "cuda"
resolved_config = output / "resolved_training_config.json"
resolved_config.write_text(json.dumps(training_config, indent=2))
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(output / "environment.txt").write_text(freeze.stdout)
command = [sys.executable, "-m", "picoagent.training", "train", "--config", str(resolved_config)]
with (output / "training.log").open("w") as log:
    result = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=False)
print((output / "training.log").read_text()[-12000:], flush=True)
(output / "job_status.json").write_text(json.dumps({{"returncode": result.returncode, "native_sft_only": True, "smoke_only": False}}))
if result.returncode:
    raise RuntimeError("Native SFT command failed; see preserved training.log")
'''


def _resume_restore_snippet(resume: dict[str, Any] | None) -> str:
    if resume is None:
        return ""
    function_source = inspect.getsource(_restore_resume_tree)
    pins = {key: resume[key] for key in (
        "run_manifest_sha256", "run_status_sha256", "dataset_manifest_sha256",
        "checkpoint", "checkpoint_manifest_sha256", "global_step", "planned_global_steps",
    )}
    pins["output_budget_bytes"] = resume["output_budget_bytes"]
    return f'''\n# Restore only the one locally verified prior checkpoint. The source handle is
# attached in kernel_sources; all bytes are checked against caller-pinned hashes.
_restore_resume_source = {function_source!r}
exec(compile(_restore_resume_source, "<pinned-resume-restore>", "exec"), globals())
_resume_pins = {pins!r}
resume_checkpoint_path = _restore_resume_tree(
    pathlib.Path("/kaggle/input"), pathlib.Path("/kaggle/working/picoagent-training"), _resume_pins)
'''


def _private_source_program(config: str, transfer_manifest: dict[str, Any], manifest_bytes: bytes,
                            helper: str, helper_sha256: str, *, segment_steps: int,
                            resume: dict[str, Any] | None = None) -> str:
    _validate_handle(transfer_manifest["dataset_handle"])
    manifest_sha = _sha256_bytes(manifest_bytes)
    expected_json = json.dumps(transfer_manifest, sort_keys=True, indent=2) + "\n"
    prefix = _base_program_prefix(helper, helper_sha256)
    prefix += f'''\n# The expected transfer manifest is pinned in kernel code, not trusted from input.
EXPECTED_INPUT_DATASET_HANDLE = {transfer_manifest["dataset_handle"]!r}
EXPECTED_TRANSFER_MANIFEST_SHA256 = {manifest_sha!r}
EXPECTED_TRANSFER_MANIFEST_BYTES = {len(manifest_bytes)}
EXPECTED_TRANSFER_MANIFEST = json.loads({expected_json!r})
input_root = pathlib.Path("/kaggle/input")
manifest_matches = []
manifest_record = {{"sha256": EXPECTED_TRANSFER_MANIFEST_SHA256, "bytes": EXPECTED_TRANSFER_MANIFEST_BYTES}}
for candidate in input_root.rglob("transfer_manifest.json"):
    if candidate.is_symlink() or candidate.stat().st_size != EXPECTED_TRANSFER_MANIFEST_BYTES:
        continue
    if matches(candidate, manifest_record):
        manifest_matches.append(candidate)
if len(manifest_matches) != 1:
    raise ValueError("Expected exactly one mounted private input package matching the pinned transfer manifest")
manifest_path = manifest_matches[0]
input_root = manifest_path.parent
if manifest_path.stat().st_size != EXPECTED_TRANSFER_MANIFEST_BYTES:
    raise ValueError("Private input transfer manifest size mismatch")
manifest_raw = manifest_path.read_bytes()
if hashlib.sha256(manifest_raw).hexdigest() != EXPECTED_TRANSFER_MANIFEST_SHA256:
    raise ValueError("Private input transfer manifest hash mismatch")
transfer = json.loads(manifest_raw)
if transfer != EXPECTED_TRANSFER_MANIFEST or transfer.get("dataset_handle") != EXPECTED_INPUT_DATASET_HANDLE:
    raise ValueError("Private input identity differs from the pinned kernel package")
if transfer.get("schema") != "picoagent.kaggle-private-input.v1":
    raise ValueError("Unknown Kaggle private input schema")
bundle = transfer["bundle"]
validate_bundle(bundle)
if any(chunk["bytes"] > {KAGGLE_CHUNK_BYTES_LIMIT} for chunk in bundle["chunks"]):
    raise ValueError("Private input chunk exceeds the bounded upload size")
paths = transfer["chunk_paths_by_sha256"]
archive_path.parent.mkdir(parents=True, exist_ok=True)
required_space = bundle["archive"]["bytes"] + bundle["extracted_bytes"] + bundle["source_manifest_bytes"] + 16 * 1024 * 1024
if shutil.disk_usage(archive_path.parent).free < required_space:
    raise OSError("Insufficient disk space for private archive reconstruction and extraction")
total = 0
digest = hashlib.sha256()
with archive_path.open("xb") as output_stream:
    for chunk in bundle["chunks"]:
        filename = paths[chunk["sha256"]]
        if pathlib.PurePath(filename).name != filename or filename.startswith("."):
            raise ValueError("Unsafe Kaggle input chunk path")
        source_chunk = input_root / filename
        if not matches(source_chunk, chunk):
            raise ValueError("Mounted Kaggle input chunk failed verification")
        with source_chunk.open("rb") as input_stream:
            while True:
                block = input_stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                output_stream.write(block)
                total += len(block)
    output_stream.flush()
    os.fsync(output_stream.fileno())
if total != bundle["archive"]["bytes"] or digest.hexdigest() != bundle["archive"]["sha256"]:
    raise ValueError("Reconstructed private source archive failed verification")
stage_result = extract_source_archive(archive_path, str(project), bundle)
if stage_result.get("verified") is not True:
    raise RuntimeError("Source staging did not verify extracted files")
'''
    return prefix + _resume_restore_snippet(resume) + _training_tail(
        config, segment_steps, resume_checkpoint=resume["checkpoint"] if resume else None)


def _training_tail(config: str, segment_steps: int, *, resume_checkpoint: str | None = None) -> str:
    resume_argument = (
        f', "--resume", str(pathlib.Path(training_config["output_dir"]) / {resume_checkpoint!r})'
        if resume_checkpoint else ""
    )
    return f'''\nsubprocess.run([sys.executable, "-m", "pip", "install", "--quiet", {PINNED_PACKAGES[0]!r}, {PINNED_PACKAGES[1]!r}, {PINNED_PACKAGES[2]!r}], check=True)
import torch
hardware = {{"torch": torch.__version__, "cuda": torch.cuda.is_available(), "devices": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
print(json.dumps(hardware), flush=True)
(output / "hardware.json").write_text(json.dumps(hardware, indent=2))
if not hardware["cuda"]:
    raise RuntimeError("GPU unavailable; refusing silent CPU fallback")
os.environ["PYTHONPATH"] = str(project / "src")
config_path = project / {config!r}
training_config = json.loads(config_path.read_text())
training_config["output_dir"] = "/kaggle/working/picoagent-training"
training_config["device"] = "cuda"
resolved_config = output / "resolved_training_config.json"
resolved_config.write_text(json.dumps(training_config, indent=2))
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(output / "environment.txt").write_text(freeze.stdout)
command = [sys.executable, "-m", "picoagent.training", "train", "--config", str(resolved_config),
           "--segment-steps", {str(segment_steps)!r}, "--output-budget-bytes", {str(KAGGLE_OUTPUT_BUDGET_BYTES)!r},
           "--output-budget-root", "/kaggle/working"{resume_argument}]
with (output / "training.log").open("w") as log:
    result = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=False)
print((output / "training.log").read_text()[-12000:], flush=True)
run_status_path = pathlib.Path(training_config["output_dir"]) / "run_status.json"
if not run_status_path.is_file():
    raise RuntimeError("Training exited without a run_status.json receipt")
run_status = json.loads(run_status_path.read_text())
if run_status.get("status") not in {{"paused", "completed"}}:
    raise RuntimeError("Training exited without a valid paused/completed status")
job_status = {{"returncode": result.returncode, "training_status": run_status["status"],
               "global_step": run_status.get("global_step"),
               "planned_global_steps": run_status.get("planned_global_steps"),
               "checkpoint": run_status.get("checkpoint"),
               "checkpoint_manifest_sha256": run_status.get("checkpoint_manifest_sha256"),
               "native_sft_only": True, "smoke_only": False}}
(output / "job_status.json").write_text(json.dumps(job_status, sort_keys=True, indent=2))
if result.returncode:
    raise RuntimeError("Native SFT command failed; see preserved training.log")
'''


def _kernel_metadata(owner: str, slug: str, dataset_source: str | None,
                     kernel_sources: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": f"{owner}/{slug}", "title": slug.replace("-", " "),
        "code_file": "main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": KAGGLE_ACCELERATOR,
        "dataset_sources": [dataset_source] if dataset_source else [],
        "competition_sources": [], "kernel_sources": kernel_sources or [],
    }


def build(root: Path, destination: Path, owner: str, slug: str, config: str | None = None, *,
          input_dataset: str | None = None, data_output: Path | None = None,
          dataset_manifest: str | Path | None = None, dataset_license: str | None = None,
          dataset_license_description: str | None = None,
          chunk_bytes: int = DEFAULT_KAGGLE_CHUNK_BYTES,
          segment_steps: int | None = None,
          resume_kernel: str | None = None, resume_run_dir: Path | None = None,
          resume_checkpoint: str | None = None,
          resume_run_manifest_sha256: str | None = None,
          resume_checkpoint_manifest_sha256: str | None = None,
          inline_archive_max_bytes: int = INLINE_ARCHIVE_MAX_BYTES,
          smoke_timeout_seconds: int = MAX_GPU_SMOKE_SECONDS) -> None:
    _validate_segment(owner, "kernel owner")
    _validate_segment(slug, "kernel slug")
    if type(smoke_timeout_seconds) is not int or not 0 < smoke_timeout_seconds <= MAX_GPU_SMOKE_SECONDS:
        raise ValueError(f"GPU smoke timeout must be in (0, {MAX_GPU_SMOKE_SECONDS} seconds]")
    if type(inline_archive_max_bytes) is not int or not 0 < inline_archive_max_bytes <= INLINE_ARCHIVE_MAX_BYTES:
        raise ValueError(f"Inline fixture archive limit must be in (0, {INLINE_ARCHIVE_MAX_BYTES} bytes]")
    root = root.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(root):
        raise ValueError("Kernel output must be outside the source root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    helper, helper_sha = _source_staging_bootstrap()

    if input_dataset is None:
        if data_output is not None or dataset_manifest is not None or dataset_license is not None:
            raise ValueError("Private dataset options require --input-dataset owner/slug")
        if segment_steps is not None:
            raise ValueError("--segment-steps requires private native-SFT input-dataset mode")
        if any(value is not None for value in (resume_kernel, resume_run_dir, resume_checkpoint,
                                                resume_run_manifest_sha256, resume_checkpoint_manifest_sha256)):
            raise ValueError("Resume options require private native-SFT input-dataset mode")
        if config is not None:
            raise ValueError("Training configs require private-input-dataset mode; inline mode is fixture-smoke-only")
        with tempfile.TemporaryDirectory(prefix="picoagent-kaggle-inline-") as temporary:
            archive = Path(temporary) / "source.tar.gz"
            source_info = archive_source(root, archive)
            archive_bytes = archive.stat().st_size
            if archive_bytes > inline_archive_max_bytes:
                raise ValueError(f"Inline archive is {archive_bytes} bytes, exceeding the {inline_archive_max_bytes}-byte fixture limit; use private-input-dataset mode")
            payload = base64.b64encode(archive.read_bytes()).decode("ascii")
            program = _inline_source_program(payload, {**source_info, "bytes": archive_bytes}, None,
                                             smoke_timeout_seconds, helper, helper_sha)
            _write_kernel_package(destination, owner, slug, program, None, {
                "schema": "picoagent.kaggle-kernel-package-receipt.v1",
                "mode": "legacy-inline-small-fixture", "source_archive_sha256": source_info["sha256"],
                "source_archive_bytes": archive_bytes, "inline_limit_bytes": inline_archive_max_bytes,
                "staging_helper_sha256": helper_sha, "smoke_timeout_seconds": smoke_timeout_seconds,
                "kernel_privacy": "requested_private_structurally_only; provider status unverified",
                "provider_calls_made": False,
            }, source_info)
        return

    if data_output is None or dataset_manifest is None or dataset_license is None:
        raise ValueError("Private dataset mode requires --data-output, --dataset-manifest, and an explicit --dataset-license")
    if config is None:
        raise ValueError("Private-input mode requires a native SFT config; GPU smoke is a separate small-fixture-only path")
    if type(segment_steps) is not int or segment_steps <= 0:
        raise ValueError("Private-input native-SFT mode requires a positive --segment-steps operational limit")
    _validate_handle(input_dataset)
    _dataset_metadata(input_dataset, input_dataset.split("/", 1)[1], dataset_license, dataset_license_description)
    _, config_relative, manifest_relative, _ = _validate_native_sft_inputs(root, config, dataset_manifest)
    resume_values = (resume_kernel, resume_run_dir, resume_checkpoint,
                     resume_run_manifest_sha256, resume_checkpoint_manifest_sha256)
    resume_info = None
    if any(value is not None for value in resume_values):
        if not all(value is not None for value in resume_values):
            raise ValueError("Resume requires kernel handle, retrieved run directory, checkpoint-N, and both exact manifest SHA256 pins")
        _validate_handle(resume_kernel)
        if resume_kernel == f"{owner}/{slug}":
            raise ValueError("Use a new unique kernel slug for each segment; a kernel cannot consume its own mutable output")
        resume_info = _validate_resume_artifact(
            root, config_relative, manifest_relative, run_dir=Path(resume_run_dir),
            checkpoint_name=resume_checkpoint, run_manifest_sha256=resume_run_manifest_sha256,
            checkpoint_manifest_sha256=resume_checkpoint_manifest_sha256,
        )
        resume_info["kernel_source"] = resume_kernel
        resume_info["output_budget_bytes"] = KAGGLE_OUTPUT_BUDGET_BYTES
    dataset_output = Path(data_output).absolute()
    if dataset_output.exists():
        raise FileExistsError(dataset_output)
    resolved_dataset_output = dataset_output.resolve()
    resolved_destination = destination.resolve()
    if (resolved_dataset_output.is_relative_to(root) or resolved_dataset_output == resolved_destination
            or resolved_dataset_output in resolved_destination.parents
            or resolved_destination in resolved_dataset_output.parents):
        raise ValueError("Private dataset package output must be outside the source root and separate from the kernel output")
    dataset_output.parent.mkdir(parents=True, exist_ok=True)

    # Stage data first under a temporary sibling; only publish after the small
    # kernel file and both hash manifests have been successfully generated.
    with tempfile.TemporaryDirectory(prefix=f".{dataset_output.name}.staging-", dir=dataset_output.parent) as temporary:
        staged_data = Path(temporary) / "dataset"
        staged_data.mkdir()
        with tempfile.TemporaryDirectory(prefix="picoagent-kaggle-archive-") as archive_tmp:
            archive = Path(archive_tmp) / "source.tar.gz"
            source_info = archive_source(root, archive, root / manifest_relative)
            if config_relative not in source_info["manifest"]["files"]:
                raise ValueError("Native SFT config was not included in the selected source archive")
            chunk_root, bundle = pack_archive(archive, source_info["manifest"], chunk_bytes)
            _, dataset_slug = _validate_handle(input_dataset)
            chunk_paths = {record["sha256"]: f"chunk-{record['sha256']}.bin" for record in bundle["chunks"]}
            transfer = {
                "schema": "picoagent.kaggle-private-input.v1",
                "dataset_handle": input_dataset,
                "source_manifest_path": manifest_relative,
                "bundle": bundle,
                "chunk_paths_by_sha256": chunk_paths,
            }
            transfer_bytes = _write_json(staged_data / "transfer_manifest.json", transfer)
            copied: set[str] = set()
            for record in bundle["chunks"]:
                digest = record["sha256"]
                if digest in copied:
                    continue
                source_chunk = chunk_root / digest
                if not matches(source_chunk, record):
                    raise ValueError("Local source chunk failed verification before packaging")
                target = staged_data / chunk_paths[digest]
                shutil.copyfile(source_chunk, target)
                if not matches(target, record):
                    raise ValueError("Packaged source chunk failed verification")
                copied.add(digest)
            metadata = _dataset_metadata(input_dataset, dataset_slug, dataset_license, dataset_license_description)
            metadata_bytes = _write_json(staged_data / "dataset-metadata.json", metadata)
            dataset_files = {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256_bytes(path.read_bytes())}
                for path in sorted(staged_data.iterdir()) if path.is_file()
            }
            data_receipt = {
                "schema": "picoagent.kaggle-private-input-package-receipt.v1",
                "dataset_handle": input_dataset,
                "transfer_manifest_sha256": _sha256_bytes(transfer_bytes),
                "dataset_metadata_sha256": _sha256_bytes(metadata_bytes),
                "source_archive_sha256": source_info["sha256"],
                "source_file_count": len(source_info["manifest"]["files"]),
                "source_manifest_sha256": bundle["source_manifest_sha256"],
                "chunk_count_with_repeats": len(bundle["chunks"]),
                "unique_chunk_files": len(copied),
                "max_chunk_bytes": max(chunk["bytes"] for chunk in bundle["chunks"]),
                "chunk_bytes_limit": KAGGLE_CHUNK_BYTES_LIMIT,
                "dataset_visibility": "not provider-verified; create without --public to request CLI default private",
                "dataset_license": dataset_license,
                "provider_calls_made": False,
                "files": dataset_files,
            }
            _write_json(staged_data / "package-receipt.json", data_receipt)
            program = _private_source_program(config_relative, transfer, transfer_bytes, helper, helper_sha,
                                              segment_steps=segment_steps, resume=resume_info)
            package_receipt = {
                "schema": "picoagent.kaggle-kernel-package-receipt.v1",
                "mode": "private-input-dataset-native-sft",
                "input_dataset_handle": input_dataset,
                "input_transfer_manifest_sha256": _sha256_bytes(transfer_bytes),
                "source_archive_sha256": source_info["sha256"],
                "source_archive_bytes": bundle["archive"]["bytes"],
                "source_manifest_sha256": bundle["source_manifest_sha256"],
                "selected_snapshot_manifest": manifest_relative,
                "staging_helper_sha256": helper_sha,
                "smoke_policy": "disabled_for_production_native_sft; optional fixture smoke is hard-capped at 60 seconds",
                "kernel_privacy": "requested_private_structurally_only; provider status unverified",
                "dataset_privacy": "not provider-verified; upload workflow omits --public",
                "dataset_license": dataset_license,
                "native_sft_configuration": True,
                "segment_steps": segment_steps,
                "output_budget_bytes": KAGGLE_OUTPUT_BUDGET_BYTES,
                "requested_accelerator": KAGGLE_ACCELERATOR,
                "resume_source": resume_info,
                "learner_tools_executed_in_kernel": False,
                "provider_calls_made": False,
            }
            # Keep package construction metadata in the kernel package so the
            # pinned bundle/helper identity is inspectable before any upload.
            staged_kernel = Path(temporary) / "kernel"
            _write_kernel_package(staged_kernel, owner, slug, program, input_dataset, package_receipt,
                                  source_info=None,
                                  kernel_sources=[resume_kernel] if resume_info else None)
        # Publish both packages only after all local validation and hashing pass.
        os.rename(staged_data, dataset_output)
        try:
            os.rename(staged_kernel, destination)
        except BaseException:
            shutil.rmtree(dataset_output)
            raise


def _write_kernel_package(destination: Path, owner: str, slug: str, program: str,
                          dataset_source: str | None, receipt: dict[str, Any],
                          source_info: dict[str, Any] | None,
                          kernel_sources: list[str] | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    main_path = destination / "main.py"
    main_path.write_text(program, encoding="utf-8")
    metadata = _kernel_metadata(owner, slug, dataset_source, kernel_sources)
    metadata_bytes = _write_json(destination / "kernel-metadata.json", metadata)
    package_receipt = {**receipt,
                       "kernel_main_sha256": _sha256_bytes(main_path.read_bytes()),
                       "kernel_metadata_sha256": _sha256_bytes(metadata_bytes)}
    _write_json(destination / "kernel-package-receipt.json", package_receipt)
    if source_info is not None:
        _write_json(destination / "source-receipt.json", source_info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Kernel owner slug")
    parser.add_argument("--slug", required=True, help="Kernel slug")
    parser.add_argument("--output", type=Path, required=True, help="Local kernel package directory; no submission occurs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", help="Native SFT config relative to --root; required for private input dataset mode")
    parser.add_argument("--input-dataset", help="Private input dataset handle owner/slug; creates local package only")
    parser.add_argument("--data-output", type=Path, help="Separate local upload folder for chunked dataset; no upload occurs")
    parser.add_argument("--dataset-manifest", help="Selected self-contained dataset manifest under --root/data")
    parser.add_argument("--dataset-license", choices=sorted(SUPPORTED_DATASET_LICENSES), help="Explicit Kaggle license value; no CC0 default is applied")
    parser.add_argument("--dataset-license-description", help="Required only when --dataset-license other")
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_KAGGLE_CHUNK_BYTES, help=f"Chunk size <= {KAGGLE_CHUNK_BYTES_LIMIT} bytes")
    parser.add_argument("--segment-steps", type=int, help="Required native-SFT operational max optimizer updates per Kaggle invocation; full schedule is unchanged")
    parser.add_argument("--resume-kernel", help="Unique prior Kaggle kernel source handle owner/slug; output is hash-pinned")
    parser.add_argument("--resume-run-dir", type=Path, help="Locally retrieved prior /kaggle/working/picoagent-training output; verified before package build")
    parser.add_argument("--resume-checkpoint", help="Exact latest paused checkpoint-N from the retrieved output")
    parser.add_argument("--resume-run-manifest-sha256", help="Caller-supplied exact run_manifest.json SHA256")
    parser.add_argument("--resume-checkpoint-manifest-sha256", help="Caller-supplied exact checkpoint-N/checkpoint_manifest.json SHA256")
    parser.add_argument("--inline-archive-max-bytes", type=int, default=INLINE_ARCHIVE_MAX_BYTES, help="Fixture-only inline archive limit; absolute maximum 8 MiB")
    parser.add_argument("--smoke-timeout-seconds", type=int, default=MAX_GPU_SMOKE_SECONDS, help="Whole optional fixture GPU smoke stage; hard maximum 60 seconds")
    args = parser.parse_args()
    build(args.root, args.output, args.owner, args.slug, args.config,
          input_dataset=args.input_dataset, data_output=args.data_output,
          dataset_manifest=args.dataset_manifest, dataset_license=args.dataset_license,
          dataset_license_description=args.dataset_license_description,
          chunk_bytes=args.chunk_bytes, segment_steps=args.segment_steps,
          resume_kernel=args.resume_kernel, resume_run_dir=args.resume_run_dir,
          resume_checkpoint=args.resume_checkpoint,
          resume_run_manifest_sha256=args.resume_run_manifest_sha256,
          resume_checkpoint_manifest_sha256=args.resume_checkpoint_manifest_sha256,
          inline_archive_max_bytes=args.inline_archive_max_bytes,
          smoke_timeout_seconds=args.smoke_timeout_seconds)


if __name__ == "__main__":
    main()
