#!/usr/bin/env python3
"""Build a local private Kaggle kernel package; never submit it automatically.

Large/private inputs are packaged as a separate chunked Kaggle Dataset and
referenced by the kernel. Small inline archives remain only as a fixture path.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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
import hashlib, json, os, pathlib, signal, subprocess, sys, tarfile, tempfile, time
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


def _private_source_program(config: str, transfer_manifest: dict[str, Any], manifest_bytes: bytes,
                            helper: str, helper_sha256: str, *, segment_steps: int) -> str:
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
    return prefix + _training_tail(config, segment_steps)


def _training_tail(config: str, segment_steps: int) -> str:
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
           "--output-budget-root", "/kaggle/working"]
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


def _kernel_metadata(owner: str, slug: str, dataset_source: str | None) -> dict[str, Any]:
    return {
        "id": f"{owner}/{slug}", "title": slug.replace("-", " "),
        "code_file": "main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": KAGGLE_ACCELERATOR,
        "dataset_sources": [dataset_source] if dataset_source else [],
        "competition_sources": [], "kernel_sources": [],
    }


def build(root: Path, destination: Path, owner: str, slug: str, config: str | None = None, *,
          input_dataset: str | None = None, data_output: Path | None = None,
          dataset_manifest: str | Path | None = None, dataset_license: str | None = None,
          dataset_license_description: str | None = None,
          chunk_bytes: int = DEFAULT_KAGGLE_CHUNK_BYTES,
          segment_steps: int | None = None,
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
                                              segment_steps=segment_steps)
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
                "learner_tools_executed_in_kernel": False,
                "provider_calls_made": False,
            }
            # Keep package construction metadata in the kernel package so the
            # pinned bundle/helper identity is inspectable before any upload.
            staged_kernel = Path(temporary) / "kernel"
            _write_kernel_package(staged_kernel, owner, slug, program, input_dataset, package_receipt, source_info=None)
        # Publish both packages only after all local validation and hashing pass.
        os.rename(staged_data, dataset_output)
        try:
            os.rename(staged_kernel, destination)
        except BaseException:
            shutil.rmtree(dataset_output)
            raise


def _write_kernel_package(destination: Path, owner: str, slug: str, program: str,
                          dataset_source: str | None, receipt: dict[str, Any],
                          source_info: dict[str, Any] | None) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    main_path = destination / "main.py"
    main_path.write_text(program, encoding="utf-8")
    metadata = _kernel_metadata(owner, slug, dataset_source)
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
    parser.add_argument("--inline-archive-max-bytes", type=int, default=INLINE_ARCHIVE_MAX_BYTES, help="Fixture-only inline archive limit; absolute maximum 8 MiB")
    parser.add_argument("--smoke-timeout-seconds", type=int, default=MAX_GPU_SMOKE_SECONDS, help="Whole optional fixture GPU smoke stage; hard maximum 60 seconds")
    args = parser.parse_args()
    build(args.root, args.output, args.owner, args.slug, args.config,
          input_dataset=args.input_dataset, data_output=args.data_output,
          dataset_manifest=args.dataset_manifest, dataset_license=args.dataset_license,
          dataset_license_description=args.dataset_license_description,
          chunk_bytes=args.chunk_bytes, segment_steps=args.segment_steps,
          inline_archive_max_bytes=args.inline_archive_max_bytes,
          smoke_timeout_seconds=args.smoke_timeout_seconds)


if __name__ == "__main__":
    main()
