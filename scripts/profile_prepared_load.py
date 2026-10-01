"""Measure approved token loading only, without model or accelerator creation."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start = time.perf_counter()
    from picoagent.training.config import TrainingConfig
    from picoagent.training.prepared import load_prepared_dataset

    config = TrainingConfig.load(args.config)
    before_load = time.perf_counter()
    prepared = load_prepared_dataset(config)
    finished = time.perf_counter()
    torch = sys.modules.get("torch")
    cuda_initialized = torch is not None and torch.cuda.is_initialized()
    if cuda_initialized:
        raise RuntimeError("CPU profiling unexpectedly initialized CUDA")
    raw = json.loads(args.raw_reference.read_text())
    baseline = raw["measured"]["approximately_comparable_cpu_preparation_seconds"]
    elapsed = finished - before_load
    total = finished - start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_mib = usage.ru_maxrss / (1024**2 if platform.system() == "Darwin" else 1024)
    result = {
        "schema": "picoagent.prepared_tokens.approved_load_profile.v1",
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "prepared_manifest_sha256": prepared.manifest_sha256,
        "source_manifest_sha256": prepared.manifest["source_manifest_sha256"],
        "approved_public_loader": True,
        "loader_wall_seconds": elapsed,
        "imports_config_and_loader_wall_seconds": total,
        "peak_process_rss_mib": peak_rss_mib,
        "packed_token_buffer_bytes": sum(len(buffer) for dataset in prepared.datasets.values() for buffer in dataset._buffers),
        "examples": {split: len(data) for split, data in prepared.datasets.items()},
        "stats": {split: stats.as_dict() for split, stats in prepared.stats.items()},
        "raw_reference": {"path": str(args.raw_reference), "sha256": hashlib.sha256(args.raw_reference.read_bytes()).hexdigest(), "comparable_preparation_seconds": baseline},
        "approximate_local_comparison": {
            "seconds_less": baseline - total,
            "speed_ratio": baseline / total,
            "fraction_less": 1 - total / baseline,
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "cuda_initialized": bool(cuda_initialized)},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "caveats": [
            "Fresh process, but filesystem cache is warm from the build/audit; not a controlled cold-cache benchmark.",
            "Host load, page cache and runtime conditions differ from the earlier raw profile and from Colab; this ratio is not a remote startup guarantee.",
            "Both comparisons exclude model download/load, run snapshot copying, optimizer allocation and all training/evaluation.",
            "Peak RSS is measured with token buffers retained. The earlier raw profile streamed/discarded encoded examples, so its RSS is not the ordinary trainer's retained-array memory.",
            "No model was loaded or constructed; CUDA remained uninitialized.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output), "loader_wall_seconds": elapsed, "total_wall_seconds": total, "peak_rss_mib": peak_rss_mib, "raw_reference_seconds": baseline, "approximate_speed_ratio": baseline / total, "cuda_initialized": bool(cuda_initialized)}, indent=2))


if __name__ == "__main__":
    main()
