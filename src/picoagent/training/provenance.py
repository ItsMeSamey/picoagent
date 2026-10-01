"""Run evidence and checkpoint integrity without importing ML frameworks."""
from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .data import canonical_json, sha256_bytes, sha256_file


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, payload: Any, *, exclusive: bool = False) -> None:
    text = canonical_json(payload) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            # Atomic no-clobber publication on a local POSIX filesystem.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def tree_hashes(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    exclude = exclude or set()
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith(".publish-") and str(path.relative_to(root)) not in exclude
        and not any(part in {"__pycache__", ".git"} for part in path.relative_to(root).parts)
    }


def code_evidence() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    project = package.parent.parent
    files = {f"src/picoagent/{name}": digest for name, digest in tree_hashes(package).items() if name.endswith(".py")}
    for file in (project / "pyproject.toml",):
        if file.is_file():
            files[str(file.relative_to(project))] = sha256_file(file)
    for file in sorted((project / "configs").glob("*.json")):
        files[str(file.relative_to(project))] = sha256_file(file)
    git: dict[str, Any] = {}
    for name, command in (("commit", ["git", "rev-parse", "HEAD"]), ("status", ["git", "status", "--porcelain=v1"])):
        result = subprocess.run(command, cwd=project, capture_output=True, text=True, check=False)
        git[name] = result.stdout.strip() if result.returncode == 0 else None
    return {"files": files, "tree_sha256": sha256_bytes(canonical_json(files).encode()), "git": git}


def environment_evidence() -> dict[str, Any]:
    packages = {distribution.metadata["Name"]: distribution.version for distribution in importlib.metadata.distributions() if distribution.metadata.get("Name")}
    return {"python": sys.version, "platform": platform.platform(), "packages": dict(sorted(packages.items()))}


def checkpoint_evidence(checkpoint: Path, run_manifest_sha256: str) -> Path:
    payload = {"schema": "picoagent.checkpoint.v1", "run_manifest_sha256": run_manifest_sha256,
               "files": tree_hashes(checkpoint, exclude={"checkpoint_manifest.json"})}
    path = checkpoint / "checkpoint_manifest.json"
    write_json(path, payload, exclusive=True)
    return path


def verify_checkpoint(checkpoint: Path, run_manifest_sha256: str) -> None:
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if manifest.get("schema") != "picoagent.checkpoint.v1" or manifest.get("run_manifest_sha256") != run_manifest_sha256:
        raise ValueError("Checkpoint belongs to another or unknown run")
    if tree_hashes(checkpoint, exclude={"checkpoint_manifest.json"}) != manifest.get("files"):
        raise ValueError("Checkpoint integrity verification failed")
    weight_files = {"model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json",
                    "adapter_model.safetensors", "adapter_model.bin"}
    if not any((checkpoint / filename).is_file() for filename in weight_files):
        raise ValueError("Checkpoint missing model/adapter weights")
    for required in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
        if not (checkpoint / required).is_file():
            raise ValueError(f"Checkpoint missing resume state: {required}")
    if not list(checkpoint.glob("rng_state*.pth")):
        raise ValueError("Checkpoint missing RNG state; exact resume is unavailable")


def preflight_resume(output: Path, checkpoint: Path, *, max_steps: int) -> None:
    """Reject old/colliding checkpoints before Trainer can overwrite future state."""
    import re

    output, checkpoint = output.resolve(), checkpoint.resolve()
    if checkpoint.parent != output or not re.fullmatch(r"checkpoint-[0-9]+", checkpoint.name):
        raise ValueError("Resume checkpoint must belong to output_dir and use checkpoint-N naming")
    selected = int(checkpoint.name.split("-")[1])
    for child in output.iterdir():
        if re.fullmatch(r"checkpoint-[0-9]+", child.name) and int(child.name.split("-")[1]) > selected:
            raise ValueError("A newer checkpoint directory exists; restore the latest complete checkpoint or archive incomplete future state before resuming")
    if any((output / name).exists() for name in ("final-model", "final-adapter", "final_artifacts.json")):
        raise ValueError("Final artifacts already exist; preserve the run rather than overwrite it on resume")
    status_path = output / "run_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "completed":
        raise ValueError("This run is already completed; create a separately identified new experiment")
    state_path = checkpoint / "trainer_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("global_step") != selected:
            raise ValueError("Checkpoint step does not agree with trainer_state.json")
        if max_steps > 0 and selected >= max_steps:
            raise ValueError("Checkpoint already reaches the configured training budget; it can be used directly for inference without retraining")
