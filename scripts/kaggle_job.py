#!/usr/bin/env python3
"""Build a private, self-contained Kaggle batch job; never submit it automatically."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tempfile

from colab_run import archive_source


def build(root: Path, destination: Path, owner: str, slug: str, config: str | None = None) -> None:
    if any(not value or "/" in value or ".." in value for value in (owner, slug)):
        raise ValueError("owner and slug must be simple identifiers")
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.tar.gz"
        info = archive_source(root, archive)
        payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    command = (["train", "--config", config] if config else
               ["smoke", "--device", "cuda", "--output-dir", "/kaggle/working/picoagent-smoke"])
    program = f'''import base64, hashlib, io, json, os, pathlib, subprocess, sys, tarfile
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TORCH_XLA"] = "0"
output = pathlib.Path("/kaggle/working")
output.mkdir(parents=True, exist_ok=True)
root = pathlib.Path("/kaggle/temp/picoagent")
root.mkdir(parents=True, exist_ok=False)
blob = base64.b64decode({payload!r})
assert hashlib.sha256(blob).hexdigest() == {info['sha256']!r}
with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
    for member in archive:
        relative = pathlib.PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts or not member.isfile():
            raise ValueError("Unsafe source archive")
        target = root / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as src, target.open("xb") as dst:
            import shutil
            shutil.copyfileobj(src, dst)
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
for name, record in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == record["sha256"], name
(output / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "transformers==5.18.0", "accelerate==1.15.0", "tokenizers==0.23.2"], check=True)
import torch
hardware = {{"torch":torch.__version__, "cuda":torch.cuda.is_available(), "devices":torch.cuda.device_count(), "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
print(json.dumps(hardware), flush=True)
(output / "hardware.json").write_text(json.dumps(hardware, indent=2))
if not hardware["cuda"]:
    raise RuntimeError("GPU unavailable; refusing silent CPU fallback")
os.environ["PYTHONPATH"] = str(root / "src")
command = {command!r}
if {config is not None!r}:
    config_path = root / {config!r}
    training_config = json.loads(config_path.read_text())
    training_config["output_dir"] = "/kaggle/working/picoagent-training"
    training_config["device"] = "cuda"
    resolved_config = output / "resolved_training_config.json"
    resolved_config.write_text(json.dumps(training_config, indent=2))
    command = ["train", "--config", str(resolved_config)]
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(output / "environment.txt").write_text(freeze.stdout)
with (output / "training.log").open("w") as log:
    result = subprocess.run([sys.executable, "-m", "picoagent.training", *command], cwd=root, stdout=log, stderr=subprocess.STDOUT)
print((output / "training.log").read_text()[-12000:], flush=True)
(output / "job_status.json").write_text(json.dumps({{"returncode":result.returncode,"smoke_only":{config is None!r}}}))
if result.returncode:
    raise RuntimeError("Training command failed; see preserved training.log")
'''
    (destination / "main.py").write_text(program, encoding="utf-8")
    metadata = {"id": f"{owner}/{slug}", "title": slug.replace("-", " "),
                "code_file": "main.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "source-receipt.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", help="Optional training config; default is random-model GPU smoke only")
    args = parser.parse_args()
    build(args.root, args.output, args.owner, args.slug, args.config)
