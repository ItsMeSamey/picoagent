"""Read-only timed wrappers around unchanged c2f82ad CPU admission/encoding."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import functools
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from picoagent.training import data as training_data, encoding
from picoagent.data import (
    audit,
    artificial_plans,
    native_admission,
    native_collection,
    native_storage,
    schema,
)

ROOT = Path.cwd()
OUT = Path("/tmp/picoagent-preparation-profile")
OUT.mkdir(exist_ok=True)
MANIFEST = ROOT / "data/native-training-plans-v1/manifest.json"
EXPECTED = "e5774693bc42815ab44858754303b3970c7e34f26a1140425dccdd5853ccfbd9"
assert audit.file_hash(MANIFEST) == EXPECTED
modules = [
    training_data,
    encoding,
    audit,
    artificial_plans,
    native_admission,
    native_collection,
    native_storage,
    schema,
]
identities = {m.__name__: audit.file_hash(Path(m.__file__)) for m in modules}
head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
metrics = defaultdict(lambda: {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0})
stack = []
start = time.perf_counter()
progress = time.perf_counter()


def instrument(module, name):
    original = getattr(module, name)
    label = module.__name__ + "." + name

    @functools.wraps(original)
    def wrapper(*a, **kw):
        global progress
        entered = time.perf_counter()
        frame = [entered, 0.0]
        stack.append(frame)
        try:
            return original(*a, **kw)
        finally:
            duration = time.perf_counter() - entered
            stack.pop()
            if stack:
                stack[-1][1] += duration
            m = metrics[label]
            m["calls"] += 1
            m["inclusive_seconds"] += duration
            m["exclusive_seconds"] += duration - frame[1]
            if name == "verify_native_context_tokens" and a:
                sid = a[0]["native_evidence"]["source_id"]
                extra = metrics[label + ":" + sid]
                extra["calls"] += 1
                extra["inclusive_seconds"] += duration
                extra["exclusive_seconds"] += duration - frame[1]
            if name == "validate_trace" and time.perf_counter() - progress > 30:
                progress = time.perf_counter()
                print(
                    json.dumps(
                        {
                            "phase": "strict_verify",
                            "rows_validated": m["calls"],
                            "elapsed_seconds": round(progress - start, 2),
                            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / 1024,
                        }
                    ),
                    flush=True,
                )

    for alias_module in modules:
        for alias, value in list(vars(alias_module).items()):
            if value is original:
                setattr(alias_module, alias, wrapper)


for mod, name in [
    (training_data, "verify_dataset"),
    (training_data, "_check_disjoint"),
    (artificial_plans, "verify_plan_snapshot"),
    (artificial_plans, "_apply"),
    (native_admission, "verify_native_snapshot"),
    (native_collection, "verify_native_collection"),
    (native_storage, "verify_sharded_snapshot"),
    (native_storage, "metadata"),
    (audit, "file_hash"),
    (schema, "validate_trace"),
    (schema, "_validate_model_event_replay"),
    (native_admission, "validate_native_evidence"),
    (native_admission, "extract_native_observation"),
    (native_admission, "verify_native_context_tokens"),
    (native_admission, "_independent_oracle"),
]:
    instrument(mod, name)

phase = time.perf_counter()
manifest, rows = training_data.verify_dataset(
    MANIFEST, allow_native_teacher=True, allow_artificial_action_plans=True
)
verify_seconds = time.perf_counter() - phase
print(
    json.dumps(
        {
            "phase": "strict_verify_complete",
            "seconds": verify_seconds,
            "counts": {k: len(v) for k, v in rows.items()},
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "timers": dict(metrics),
        },
        sort_keys=True,
    ),
    flush=True,
)
(OUT / "verify_profile.json").write_text(
    json.dumps(
        {
            "head": head,
            "manifest_sha256": EXPECTED,
            "seconds": verify_seconds,
            "metrics": dict(metrics),
        },
        sort_keys=True,
        indent=2,
    )
    + "\n"
)
from transformers import AutoTokenizer  # noqa: E402 - deliberately after timed verification

phase = time.perf_counter()
tokenizer_dir = MANIFEST.parent / "base/components/000/evidence/native_compaction/source/tokenizer"
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir, use_fast=True, local_files_only=True, trust_remote_code=False
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
identity = manifest["tokenizer_identity"]
artificial_plans.validate_note_tokenizer(
    MANIFEST,
    tokenizer,
    model_id=identity["model_id"],
    revision=identity["revision"],
    tokenizer_json_path=tokenizer_dir / "tokenizer.json",
)
tokenizer_seconds = time.perf_counter() - phase
phase = time.perf_counter()
encoded = {}
estimated_python_storage = 0
for split, records in rows.items():
    split_start = time.perf_counter()
    totals = {
        "tasks": len(records),
        "examples": 0,
        "total_tokens": 0,
        "assistant_tokens": 0,
        "maximum_length": 0,
        "estimated_python_container_and_int_bytes": 0,
    }
    digest = hashlib.sha256()
    event_expansion = 0.0
    encoding_seconds = 0.0
    fingerprint_seconds = 0.0
    for r in records:
        t = time.perf_counter()
        examples = encoding.event_examples(r)
        event_expansion += time.perf_counter() - t
        for ex, last in examples:
            t = time.perf_counter()
            item = encoding.encode_trace(ex, tokenizer, 4096, supervise_last_only=last)
            encoding_seconds += time.perf_counter() - t
            n = len(item["input_ids"])
            totals["examples"] += 1
            totals["total_tokens"] += n
            totals["assistant_tokens"] += sum(v != -100 for v in item["labels"])
            totals["maximum_length"] = max(totals["maximum_length"], n)
            # Logical upper estimate: label positives reuse the ids' integer objects;
            # labels -100 and attention 1 are shared CPython small integers.
            totals["estimated_python_container_and_int_bytes"] += (
                sys.getsizeof(item)
                + sum(sys.getsizeof(v) for v in item.values())
                + sum(sys.getsizeof(v) for v in item["input_ids"])
            )
            t = time.perf_counter()
            digest.update(
                (schema.canonical_json({"trace_id": ex["trace_id"], **item}) + "\n").encode()
            )
            fingerprint_seconds += time.perf_counter() - t
    totals.update(
        seconds=time.perf_counter() - split_start,
        event_expansion_seconds=event_expansion,
        production_encode_seconds=encoding_seconds,
        fingerprint_seconds=fingerprint_seconds,
        encoded_stream_sha256=digest.hexdigest(),
    )
    encoded[split] = totals
    print(
        json.dumps({"phase": "encoding_split_complete", "split": split, **totals}, sort_keys=True),
        flush=True,
    )
encoding_seconds = time.perf_counter() - phase
assert audit.file_hash(MANIFEST) == EXPECTED
assert all(audit.file_hash(Path(m.__file__)) == identities[m.__name__] for m in modules)
report = {
    "schema": "picoagent.preparation.cpu_profile.v1",
    "head": head,
    "manifest_sha256": EXPECTED,
    "source_hashes": identities,
    "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": platform.python_version(),
    "strict_verify_seconds": verify_seconds,
    "local_tokenizer_seconds": tokenizer_seconds,
    "streaming_production_encoding_wall_seconds": encoding_seconds,
    "encoded": encoded,
    "timers": dict(metrics),
    "total_elapsed_seconds": time.perf_counter() - start,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "profiling_caveats": [
        "CPU-only local run; do not attribute remote delay without remote stage/process evidence.",
        "Timed wrappers add minor overhead and page cache differs from Colab.",
        "Encoding uses unchanged event_examples+encode_trace but discards each completed item; production encode_records currently retains them all.",
        "No model download/load, GPU, provider or trace-command execution occurred.",
        "Integer/container storage is an estimate; excludes retained trace rows, tokenizer, model and framework.",
    ],
}
(OUT / "profile.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
print(
    json.dumps(
        {
            "phase": "complete",
            "path": str(OUT / "profile.json"),
            "seconds": report["total_elapsed_seconds"],
            "peak_rss_mib": report["peak_rss_mib"],
        },
        sort_keys=True,
    ),
    flush=True,
)
