"""Strict CPU preparation and separately approved, immutable token artifacts.

No pickle, model/accelerator allocation, provider actions or trace commands.
Production fast-loading requires an independently reviewed manifest digest and
an explicit matching config pin. Ordinary raw admission remains unchanged.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TrainingConfig
from .data import canonical_json, sha256_bytes, sha256_file, verify_dataset
from .encoding import EncodingStats, IGNORE_INDEX, encode_trace, event_examples

SCHEMA = "picoagent.prepared_tokens.v1"
MAX_FILE_BYTES = 25 * 1024**2
SHARD_BYTES = 16 * 1024**2
MAX_MANIFEST_BYTES = 4 * 1024**2
MAGIC = b"PICOTOK1\n"
# An approval cannot hash itself. Everything else in the core Python tree is
# bound conservatively, including dependencies that admission imports lazily.
EXCLUDED_TRANSFORM_FILES = frozenset({"training/prepared_approvals.py"})


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("non-finite JSON value")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _inside(root: Path, relative: str) -> Path:
    _need(isinstance(relative, str) and bool(relative), "empty artifact path")
    parts = Path(relative).parts
    _need(
        not Path(relative).is_absolute()
        and all(p not in {"..", "."} for p in parts)
        and "\\" not in relative,
        "unsafe artifact path",
    )
    current = root
    _need(not current.is_symlink(), "artifact root is a symlink")
    for part in parts:
        current = current / part
        _need(not current.is_symlink(), "artifact path is a symlink")
    _need(current.resolve().is_relative_to(root.resolve()), "artifact path escapes root")
    return current


def _read(path: Path, limit: int = MAX_FILE_BYTES) -> bytes:
    _need(not path.is_symlink() and path.is_file(), "artifact must be a regular non-symlink file")
    _need(path.stat().st_size <= limit, "artifact file exceeds byte bound")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    _need(len(raw) <= limit, "artifact grew beyond byte bound")
    return raw


def _identity() -> dict:
    package = Path(__file__).resolve().parents[1]
    source = {}
    for path in sorted(package.rglob("*")):
        _need(not path.is_symlink(), "transform source tree contains a symlink")
        relative = str(path.relative_to(package))
        if path.is_file() and path.suffix == ".py" and relative not in EXCLUDED_TRANSFORM_FILES:
            source[relative] = sha256_file(path)
    return {
        "source": source,
        "packages": {
            name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")
        },
        "protocol": "picoagent-text-v1",
        "ignore_index": IGNORE_INDEX,
        "policy": "exact_event_current_assistant_only_else_legacy_all_assistant;no_packing;no_truncation",
    }


def _effective_tokenizer(tokenizer) -> dict:
    _need(tokenizer.is_fast, "prepared assistant masks require a fast tokenizer")
    return {
        "class": tokenizer.__class__.__module__ + "." + tokenizer.__class__.__name__,
        "backend_sha256": sha256_bytes(tokenizer.backend_tokenizer.to_str().encode()),
        "vocab_size": len(tokenizer),
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": tokenizer.all_special_ids,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "padding_side": tokenizer.padding_side,
        "added_vocab": tokenizer.get_added_vocab(),
        "use_fast": True,
        "trust_remote_code": False,
        "add_special_tokens": False,
        "truncation": False,
    }


def _tokenizer(config: TrainingConfig, path: str | Path | None = None):
    from transformers import AutoTokenizer

    local = path is not None or config.smoke_test
    kwargs = {"use_fast": True, "trust_remote_code": False, "local_files_only": local}
    if path is None and not config.smoke_test:
        kwargs["revision"] = config.model_revision
    value = AutoTokenizer.from_pretrained(str(path or config.model_id), **kwargs)
    _need(value.is_fast, "prepared tokenizer must be fast")
    if value.pad_token_id is None:
        _need(value.eos_token_id is not None, "tokenizer needs existing pad or EOS")
        value.pad_token = value.eos_token
    value.padding_side = "right"
    return value


def _source_inventory(manifest: dict) -> dict:
    if "files" in manifest:
        result = {
            name: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
            for name, entry in manifest["files"].items()
        }
    else:
        result = {
            entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
            for entry in manifest["splits"].values()
        }
    _need("manifest.json" not in result, "source inventory includes its own root manifest")
    return dict(sorted(result.items()))


def _flags(config: TrainingConfig) -> dict:
    return {
        "smoke_test": config.smoke_test,
        "allow_native_teacher_observed": config.allow_native_teacher_observed,
        "allow_artificial_action_plans": config.allow_artificial_action_plans,
    }


def _write(root: Path, relative: str, raw: bytes, files: dict) -> None:
    _need(len(raw) <= MAX_FILE_BYTES, "prepared file exceeds bound")
    target = _inside(root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(target, 0o444)
    files[relative] = {"bytes": len(raw), "sha256": sha256_bytes(raw)}


def _compress(raw: bytes) -> bytes:
    return gzip.compress(raw, compresslevel=6, mtime=0)


def _decompress(raw: bytes, limit: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as handle:
        logical = handle.read(limit + 1)
    _need(len(logical) <= limit, "prepared logical shard exceeds bound")
    return logical


def _pack(item: dict) -> bytes:
    n = len(item["input_ids"])
    return struct.pack("<I", n) + struct.pack(
        "<" + "i" * (3 * n), *(item["input_ids"] + item["attention_mask"] + item["labels"])
    )


def _unpack(data: bytes, offset: int, n: int) -> dict:
    values = struct.unpack_from("<" + "i" * (3 * n), data, offset + 4)
    return {
        "input_ids": list(values[:n]),
        "attention_mask": list(values[n : 2 * n]),
        "labels": list(values[2 * n :]),
    }


class TokenDataset(Sequence):
    """Immutable verified byte buffers, avoiding millions of retained Python ints."""

    def __init__(self, buffers: list[bytes], locations: list[tuple[int, int, int]]):
        self._buffers, self._locations = tuple(buffers), tuple(locations)

    def __len__(self):
        return len(self._locations)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        shard, offset, size = self._locations[index]
        return _unpack(self._buffers[shard], offset, size)


@dataclass(frozen=True)
class PreparedDataset:
    manifest: dict
    source_manifest: dict
    datasets: dict[str, TokenDataset]
    stats: dict[str, EncodingStats]
    identity: dict
    path: Path
    source_path: Path
    manifest_sha256: str


def build_prepared_dataset(
    config: TrainingConfig,
    output_dir: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    shard_bytes: int = SHARD_BYTES,
) -> Path:
    """Strict CPU builder. Output remains unapproved until independent review."""
    _need(
        config.prepared_manifest is None, "builder requires raw config without a prepared artifact"
    )
    _need(
        type(shard_bytes) is int and 1024 <= shard_bytes <= SHARD_BYTES,
        "invalid prepared shard bound",
    )
    # Bind the implementation before any admission work. Capturing this only
    # afterward could record new file bytes while the verifier used an already
    # imported older implementation during a concurrent source edit.
    transform = _identity()
    source_path = Path(config.dataset_manifest)
    source_raw = _read(source_path, MAX_MANIFEST_BYTES)
    source_sha = sha256_bytes(source_raw)
    manifest, rows = verify_dataset(
        source_path,
        allow_smoke=config.smoke_test,
        allow_native_teacher=config.allow_native_teacher_observed,
        allow_artificial_action_plans=config.allow_artificial_action_plans,
    )
    _need(
        _read(source_path, MAX_MANIFEST_BYTES) == source_raw,
        "source manifest changed during CPU admission",
    )
    _need(_identity() == transform, "transform implementation changed during CPU admission")
    tokenizer = _tokenizer(config, tokenizer_path)
    if manifest["schema"] == "picoagent.artificial_action_plan.dataset.v1":
        from picoagent.data.artificial_plans import validate_note_tokenizer

        if tokenizer_path is None:
            from huggingface_hub import hf_hub_download

            tokenizer_json = Path(
                hf_hub_download(
                    config.model_id,
                    "tokenizer.json",
                    revision=config.model_revision,
                    local_files_only=True,
                    token=False,
                )
            )
        else:
            tokenizer_json = Path(tokenizer_path) / "tokenizer.json"
        validate_note_tokenizer(
            source_path,
            tokenizer,
            model_id=config.model_id,
            revision=config.model_revision,
            tokenizer_json_path=tokenizer_json,
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    files = {}
    _write(root, "source_manifest.json", source_raw, files)
    tokenizer.save_pretrained(root / "tokenizer")
    tokenizer_files = {}
    for path in sorted((root / "tokenizer").rglob("*")):
        if path.is_file():
            raw = _read(path)
            relative = str(path.relative_to(root))
            tokenizer_files[relative] = {"bytes": len(raw), "sha256": sha256_bytes(raw)}
            files[relative] = tokenizer_files[relative]
            os.chmod(path, 0o444)
    effective = _effective_tokenizer(_tokenizer(config, root / "tokenizer"))
    _need(
        effective == _effective_tokenizer(tokenizer),
        "tokenizer snapshot changed effective behavior",
    )
    splits = {}
    for split in ("train", "dev"):
        descriptors, data, index_rows = [], bytearray(MAGIC), []
        stats = {"examples": 0, "total_tokens": 0, "assistant_tokens": 0, "maximum_length": 0}
        stream_hash = hashlib.sha256()

        def flush():
            nonlocal data, index_rows
            if not index_rows:
                return
            number = len(descriptors)
            binary_path = f"{split}/part-{number:05d}.tokens.gz"
            index_path = f"{split}/part-{number:05d}.index.jsonl.gz"
            raw = bytes(data)
            index_raw = ("".join(canonical_json(row) + "\n" for row in index_rows)).encode()
            _need(len(index_raw) <= SHARD_BYTES, "index shard exceeds bound")
            _write(root, binary_path, _compress(raw), files)
            _write(root, index_path, _compress(index_raw), files)
            descriptors.append(
                {
                    "data": binary_path,
                    "index": index_path,
                    "records": len(index_rows),
                    "logical_bytes": len(raw),
                    "logical_sha256": sha256_bytes(raw),
                    "index_logical_bytes": len(index_raw),
                    "index_logical_sha256": sha256_bytes(index_raw),
                }
            )
            data, index_rows = bytearray(MAGIC), []

        for row_ordinal, row in enumerate(rows[split]):
            row_sha256 = sha256_bytes(canonical_json(row).encode())
            event_indices = [
                i
                for i, event in enumerate(row.get("model_events", []))
                if event.get("type") == "assistant"
                or (event.get("type") == "compaction" and event.get("accepted") is True)
            ]
            for example_number, (example, last_only) in enumerate(event_examples(row)):
                item = encode_trace(
                    example, tokenizer, config.max_seq_length, supervise_last_only=last_only
                )
                packed = _pack(item)
                _need(
                    len(packed) + len(MAGIC) <= shard_bytes,
                    "one example exceeds logical shard bound",
                )
                if len(data) + len(packed) > shard_bytes:
                    flush()
                _need(
                    _unpack(packed, 0, len(item["input_ids"])) == item,
                    "encoded arrays failed exact binary roundtrip",
                )
                event_index = event_indices[example_number] if event_indices else None
                entry = {
                    "ordinal": stats["examples"],
                    "row_ordinal": row_ordinal,
                    "event_index": event_index,
                    "trace_id": row["trace_id"],
                    "task_id": row["task_id"],
                    "example_id": example["trace_id"],
                    "split": split,
                    "example_sha256": sha256_bytes(canonical_json(example).encode()),
                    "row_sha256": row_sha256,
                    "last_only": last_only,
                    "encoded_sha256": sha256_bytes(packed),
                    "length": len(item["input_ids"]),
                }
                data.extend(packed)
                index_rows.append(entry)
                stream_hash.update(
                    (canonical_json({"trace_id": example["trace_id"], **item}) + "\n").encode()
                )
                n = len(item["input_ids"])
                stats["examples"] += 1
                stats["total_tokens"] += n
                stats["assistant_tokens"] += sum(v != IGNORE_INDEX for v in item["labels"])
                stats["maximum_length"] = max(stats["maximum_length"], n)
        flush()
        splits[split] = {
            "shards": descriptors,
            "stats": stats,
            "source_rows": len(rows[split]),
            "encoded_stream_sha256": stream_hash.hexdigest(),
        }
    _need(_identity() == transform, "transform implementation changed during build")
    _need(
        _read(source_path, MAX_MANIFEST_BYTES) == source_raw, "source manifest changed during build"
    )
    result = {
        "schema": SCHEMA,
        "admission": "strict_cpu_build_pending_independent_digest_approval",
        "source_manifest_sha256": source_sha,
        "builder_python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "source_inventory": _source_inventory(manifest),
        "flags": _flags(config),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "max_seq_length": config.max_seq_length,
        "transform": transform,
        "tokenizer": effective,
        "tokenizer_files": tokenizer_files,
        "logical_shard_limit": shard_bytes,
        "files": dict(sorted(files.items())),
        "splits": splits,
    }
    _write(root, "manifest.json", (canonical_json(result) + "\n").encode(), {})
    import dataclasses

    probe = dataclasses.replace(
        config,
        prepared_manifest=str(root / "manifest.json"),
        prepared_manifest_sha256=sha256_file(root / "manifest.json"),
    )
    # This internal roundtrip does not admit the result for production training.
    _load(probe, require_approval=False)
    return root / "manifest.json"


def load_prepared_dataset(config: TrainingConfig) -> PreparedDataset:
    return _load(config, require_approval=True)


def _load(config: TrainingConfig, *, require_approval: bool) -> PreparedDataset:
    _need(config.prepared_manifest is not None, "prepared config pin is required")
    path = Path(config.prepared_manifest)
    raw = _read(path, MAX_MANIFEST_BYTES)
    digest = sha256_bytes(raw)
    _need(
        digest == config.prepared_manifest_sha256,
        "prepared manifest SHA256 differs from explicit pin",
    )
    if require_approval and not config.smoke_test:
        from .prepared_approvals import APPROVED_PREPARED_MANIFESTS

        _need(
            digest in APPROVED_PREPARED_MANIFESTS,
            "prepared manifest has no independent production approval",
        )
    manifest = _json(raw)
    _need(
        set(manifest)
        == {
            "schema",
            "admission",
            "source_manifest_sha256",
            "builder_python",
            "source_inventory",
            "flags",
            "model_id",
            "model_revision",
            "max_seq_length",
            "transform",
            "tokenizer",
            "tokenizer_files",
            "logical_shard_limit",
            "files",
            "splits",
        },
        "invalid prepared manifest fields",
    )
    _need(manifest.get("schema") == SCHEMA, "unknown prepared artifact schema")
    _need(manifest.get("flags") == _flags(config), "prepared admission flags differ from config")
    _need(manifest.get("transform") == _identity(), "prepared encoder or package identity changed")
    _need(
        manifest.get("model_id") == config.model_id
        and manifest.get("model_revision") == config.model_revision
        and manifest.get("max_seq_length") == config.max_seq_length,
        "prepared model/tokenizer/length identity changed",
    )
    _need(
        type(manifest.get("logical_shard_limit")) is int
        and 1024 <= manifest["logical_shard_limit"] <= SHARD_BYTES,
        "invalid logical shard bound",
    )
    root = path.parent
    files = manifest["files"]
    _need(isinstance(files, dict) and len(files) <= 10000, "invalid prepared file inventory")
    actual_paths = set()
    for candidate in root.rglob("*"):
        _need(not candidate.is_symlink(), "prepared tree contains a symlink")
        if candidate.is_file():
            actual_paths.add(str(candidate.relative_to(root)))
    _need(
        actual_paths == set(files) | {"manifest.json"},
        "prepared inventory has missing or extra files",
    )
    for name, info in files.items():
        _need(
            set(info) == {"bytes", "sha256"} and type(info["bytes"]) is int,
            "invalid file descriptor",
        )
        data = _read(_inside(root, name))
        _need(
            len(data) == info["bytes"] and sha256_bytes(data) == info["sha256"],
            "prepared file bytes changed",
        )
    source_raw = _read(_inside(root, "source_manifest.json"), MAX_MANIFEST_BYTES)
    source_path = Path(config.dataset_manifest)
    _need(
        _read(source_path, MAX_MANIFEST_BYTES) == source_raw
        and sha256_bytes(source_raw) == manifest["source_manifest_sha256"],
        "prepared source manifest differs from raw dataset",
    )
    source = _json(source_raw)
    _need(
        _source_inventory(source) == manifest["source_inventory"],
        "source inventory differs from immutable dataset",
    )
    _need(
        bool(source.get("smoke_only", False)) == config.smoke_test,
        "prepared smoke/production mismatch",
    )
    for name, info in manifest["source_inventory"].items():
        source_file = _inside(source_path.parent, name)
        _need(
            source_file.is_file()
            and source_file.stat().st_size == info["bytes"]
            and sha256_file(source_file) == info["sha256"],
            "raw source evidence bytes changed",
        )
    _need(
        manifest["tokenizer_files"]
        == {k: v for k, v in files.items() if k.startswith("tokenizer/")},
        "tokenizer inventory changed",
    )
    tokenizer = _tokenizer(config, root / "tokenizer")
    _need(_effective_tokenizer(tokenizer) == manifest["tokenizer"], "effective tokenizer changed")
    vocabulary_size = len(tokenizer)
    _need(set(manifest["splits"]) == {"train", "dev"}, "prepared artifact requires train/dev only")
    datasets, all_stats, consumed = {}, {}, {"source_manifest.json", *manifest["tokenizer_files"]}
    global_tasks, global_traces = set(), set()
    for split in ("train", "dev"):
        descriptor = manifest["splits"][split]
        _need(
            set(descriptor) == {"shards", "stats", "source_rows", "encoded_stream_sha256"},
            "invalid prepared split descriptor",
        )
        source_count = (
            source["counts"][split] if "counts" in source else source["splits"][split]["records"]
        )
        _need(
            type(descriptor["source_rows"]) is int and descriptor["source_rows"] == source_count,
            "prepared source row count differs from raw manifest",
        )
        buffers, locations = [], []
        stats = {"examples": 0, "total_tokens": 0, "assistant_tokens": 0, "maximum_length": 0}
        stream_hash = hashlib.sha256()
        last_row, last_event, row_identity = -1, -1, None
        seen_tasks, seen_traces = set(), set()
        for shard in descriptor["shards"]:
            _need(
                set(shard)
                == {
                    "data",
                    "index",
                    "records",
                    "logical_bytes",
                    "logical_sha256",
                    "index_logical_bytes",
                    "index_logical_sha256",
                },
                "invalid prepared shard descriptor",
            )
            for name in (shard["data"], shard["index"]):
                _need(
                    name in files and name not in consumed and name.startswith(split + "/"),
                    "repeated/wrong-split prepared shard",
                )
                consumed.add(name)
            packed = _decompress(
                _read(_inside(root, shard["data"])), manifest["logical_shard_limit"]
            )
            index_data = _decompress(_read(_inside(root, shard["index"])), SHARD_BYTES)
            _need(
                len(packed) == shard["logical_bytes"]
                and sha256_bytes(packed) == shard["logical_sha256"]
                and len(index_data) == shard["index_logical_bytes"]
                and sha256_bytes(index_data) == shard["index_logical_sha256"],
                "logical prepared shard mismatch",
            )
            _need(packed.startswith(MAGIC), "invalid token binary header")
            offset, count = len(MAGIC), 0
            for line in index_data.splitlines():
                index = _json(line)
                _need(
                    set(index)
                    == {
                        "ordinal",
                        "row_ordinal",
                        "event_index",
                        "trace_id",
                        "task_id",
                        "example_id",
                        "split",
                        "example_sha256",
                        "row_sha256",
                        "last_only",
                        "encoded_sha256",
                        "length",
                    },
                    "invalid event index schema",
                )
                _need(
                    index["split"] == split
                    and type(index["ordinal"]) is int
                    and index["ordinal"] == stats["examples"],
                    "prepared example order changed",
                )
                row_number, event = index["row_ordinal"], index["event_index"]
                _need(
                    type(row_number) is int and row_number in {last_row, last_row + 1},
                    "prepared source row order changed",
                )
                current_identity = (index["task_id"], index["trace_id"], index["row_sha256"])
                if row_number != last_row:
                    _need(
                        index["task_id"] not in global_tasks
                        and index["trace_id"] not in global_traces,
                        "duplicate prepared source task",
                    )
                    seen_tasks.add(index["task_id"])
                    seen_traces.add(index["trace_id"])
                    global_tasks.add(index["task_id"])
                    global_traces.add(index["trace_id"])
                    last_row, last_event, row_identity = row_number, -1, current_identity
                _need(
                    current_identity == row_identity, "prepared row identity changed within events"
                )
                _need(type(index["last_only"]) is bool, "invalid last-only supervision flag")
                if event is None:
                    _need(not index["last_only"] and last_event == -1, "invalid legacy event order")
                    last_event = 0
                    _need(
                        index["example_id"] == index["trace_id"], "legacy example identity changed"
                    )
                else:
                    _need(
                        type(event) is int and event > last_event and index["last_only"],
                        "prepared event order changed",
                    )
                    last_event = event
                    _need(
                        index["example_id"] == f"{index['trace_id']}:event-{event}",
                        "prepared event ID changed",
                    )
                n = index["length"]
                _need(
                    type(n) is int
                    and 1 <= n <= config.max_seq_length
                    and offset + 4 + 12 * n <= len(packed),
                    "invalid/truncated token length",
                )
                _need(
                    struct.unpack_from("<I", packed, offset)[0] == n, "token index length mismatch"
                )
                end = offset + 4 + 12 * n
                _need(
                    sha256_bytes(packed[offset:end]) == index["encoded_sha256"],
                    "token array binding changed",
                )
                item = _unpack(packed, offset, n)
                _need(
                    all(0 <= token < vocabulary_size for token in item["input_ids"]),
                    "token ID outside tokenizer vocabulary",
                )
                _need(item["attention_mask"] == [1] * n, "prepared attention mask changed")
                _need(
                    all(
                        label == IGNORE_INDEX or label == token
                        for label, token in zip(item["labels"], item["input_ids"])
                    )
                    and any(label != IGNORE_INDEX for label in item["labels"][1:]),
                    "invalid assistant labels",
                )
                locations.append((len(buffers), offset, n))
                offset = end
                count += 1
                stats["examples"] += 1
                stats["total_tokens"] += n
                stats["assistant_tokens"] += sum(v != IGNORE_INDEX for v in item["labels"])
                stats["maximum_length"] = max(stats["maximum_length"], n)
                stream_hash.update(
                    (canonical_json({"trace_id": index["example_id"], **item}) + "\n").encode()
                )
            _need(
                offset == len(packed) and count == shard["records"],
                "token shard trailing data/count mismatch",
            )
            buffers.append(packed)
        _need(
            last_row + 1 == descriptor["source_rows"]
            and stats == descriptor["stats"]
            and stream_hash.hexdigest() == descriptor["encoded_stream_sha256"],
            "prepared counts/order/tensors differ",
        )
        _need(stats["examples"] > 0, "empty prepared split")
        datasets[split], all_stats[split] = TokenDataset(buffers, locations), EncodingStats(**stats)
    _need(consumed == set(files), "unconsumed prepared files")
    identity = {
        "manifest_sha256": digest,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "transform": manifest["transform"],
        "tokenizer": manifest["tokenizer"],
    }
    return PreparedDataset(
        manifest, source, datasets, all_stats, identity, path, source_path, digest
    )


def validate_training_tokenizer(prepared: PreparedDataset, tokenizer) -> None:
    _need(
        _effective_tokenizer(tokenizer) == prepared.manifest["tokenizer"],
        "training tokenizer differs from approved prepared tokenizer",
    )


def _copy_files(source: Path, target: Path, files: dict, root_manifest: bytes) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name, info in files.items():
        raw = _read(_inside(source, name), max(MAX_FILE_BYTES, info["bytes"]))
        _need(
            len(raw) == info["bytes"] and sha256_bytes(raw) == info["sha256"],
            "approved source changed while copying",
        )
        destination = _inside(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(destination, 0o444)
        _need(
            destination.stat().st_size == info["bytes"]
            and sha256_file(destination) == info["sha256"],
            "copied prepared/source bytes differ",
        )
    _write(target, "manifest.json", root_manifest, {})


def copy_prepared_snapshots(
    prepared: PreparedDataset, *, source_destination: Path, prepared_destination: Path
) -> None:
    """Byte-copy after THIS explicit approved-artifact admission, not raw bypass."""
    raw = _read(prepared.path, MAX_MANIFEST_BYTES)
    _need(
        sha256_bytes(raw) == prepared.manifest_sha256, "prepared manifest changed during snapshot"
    )
    source_raw = _read(prepared.source_path, MAX_MANIFEST_BYTES)
    _need(
        sha256_bytes(source_raw) == prepared.manifest["source_manifest_sha256"],
        "raw source changed during snapshot",
    )
    _copy_files(
        prepared.source_path.parent,
        source_destination,
        prepared.manifest["source_inventory"],
        source_raw,
    )
    _copy_files(prepared.path.parent, prepared_destination, prepared.manifest["files"], raw)
