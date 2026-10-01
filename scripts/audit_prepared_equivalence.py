"""Read-only independent byte/array audit; does not grant production approval.

Uses only Python's standard library and deliberately does not import the prepared
builder, its binary decoder, Transformers, or Torch. The raw reference must come
from a separately reviewed complete run of the original admission/encoder path.
"""
from __future__ import annotations

import argparse
import array
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path, limit: int) -> bytes:
    require(not path.is_symlink() and path.is_file(), f"Not a regular file: {path}")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    require(len(raw) <= limit, f"Oversized file: {path}")
    return raw


def payload(root: Path, name: str) -> Path:
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe payload path")
    current = root
    for part in relative.parts:
        current /= part
        require(not current.is_symlink(), "Symlink in payload path")
    require(current.resolve().is_relative_to(root.resolve()), "Payload escaped root")
    return current


def logical(root: Path, name: str) -> bytes:
    with gzip.open(payload(root, name), "rb") as handle:
        raw = handle.read(16 * 1024**2 + 1)
    require(len(raw) <= 16 * 1024**2, "Oversized logical shard")
    return raw


def decode_streams(path: Path, reference: dict) -> tuple[dict, dict]:
    raw = read(path, 4 * 1024**2)
    manifest = json.loads(raw)
    root = path.parent
    require(manifest["schema"] == "picoagent.prepared_tokens.v1", "Wrong prepared schema")
    require(manifest["source_manifest_sha256"] == reference["manifest_sha256"], "Wrong raw source")
    actual = set()
    for entry in root.rglob("*"):
        require(not entry.is_symlink(), "Artifact contains a symlink")
        if entry.is_file():
            actual.add(str(entry.relative_to(root)))
    require(actual == {"manifest.json", *manifest["files"]}, "Artifact inventory mismatch")
    for name, info in manifest["files"].items():
        file_bytes = read(payload(root, name), 25 * 1024**2)
        require(len(file_bytes) == info["bytes"] and sha(file_bytes) == info["sha256"], "Payload bytes changed")

    streams = {}
    for split in ("train", "dev"):
        result = {"examples": 0, "total_tokens": 0, "assistant_tokens": 0, "maximum_length": 0}
        stream = hashlib.sha256()
        descriptor = manifest["splits"][split]
        for shard in descriptor["shards"]:
            packed = logical(root, shard["data"])
            index_bytes = logical(root, shard["index"])
            require(packed[:9] == b"PICOTOK1\n", "Wrong binary header")
            require(len(packed) == shard["logical_bytes"] and sha(packed) == shard["logical_sha256"], "Binary logical hash mismatch")
            require(len(index_bytes) == shard["index_logical_bytes"] and sha(index_bytes) == shard["index_logical_sha256"], "Index logical hash mismatch")
            entries = [json.loads(line) for line in index_bytes.splitlines()]
            require(len(entries) == shard["records"], "Index record count mismatch")
            cursor = 9
            for entry in entries:
                require(entry["ordinal"] == result["examples"] and entry["split"] == split, "Wrong example order")
                require(cursor + 4 <= len(packed), "Missing record length")
                length = struct.unpack_from("<I", packed, cursor)[0]
                end = cursor + 4 + 12 * length
                require(1 <= length <= manifest["max_seq_length"] and length == entry["length"] and end <= len(packed), "Invalid record length")
                require(sha(packed[cursor:end]) == entry["encoded_sha256"], "Record hash mismatch")
                values = array.array("i")
                require(values.itemsize == 4, "Independent decoder needs 32-bit C int")
                values.frombytes(packed[cursor + 4:end])
                if sys.byteorder != "little":
                    values.byteswap()
                inputs, attention, labels = (values[n * length:(n + 1) * length].tolist() for n in range(3))
                require(attention == [1] * length, "Wrong attention mask")
                require(all(0 <= value < manifest["tokenizer"]["vocab_size"] for value in inputs), "Invalid token ID")
                require(all(label == -100 or label == value for label, value in zip(labels, inputs)), "Invalid labels")
                require(any(value != -100 for value in labels[1:]), "Missing causal supervision")
                record = {"trace_id": entry["example_id"], "input_ids": inputs, "attention_mask": attention, "labels": labels}
                stream.update((canonical(record) + "\n").encode())
                result["examples"] += 1
                result["total_tokens"] += length
                result["assistant_tokens"] += sum(value != -100 for value in labels)
                result["maximum_length"] = max(result["maximum_length"], length)
                cursor = end
            require(cursor == len(packed), "Trailing binary bytes")
        expected = reference["encoded"][split]
        require(result == descriptor["stats"], "Prepared statistics disagree")
        require(all(value == expected[key] for key, value in result.items()), "Raw statistics disagree")
        require(stream.hexdigest() == descriptor["encoded_stream_sha256"] == expected["encoded_stream_sha256"], "Full ordered raw/prepared arrays disagree")
        require(descriptor["source_rows"] == expected["tasks"], "Source row count changed")
        streams[split] = {**result, "source_rows": descriptor["source_rows"], "encoded_stream_sha256": stream.hexdigest()}
    return manifest, {"path": str(path), "sha256": sha(raw), "streams": streams}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--raw-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile_bytes = read(args.raw_profile, 4 * 1024**2)
    profile = json.loads(profile_bytes)
    for name, expected in profile["source_hashes"].items():
        path = Path("src") / (name.replace(".", "/") + ".py")
        require(sha(read(path, 4 * 1024**2)) == expected, f"Raw-reference implementation changed: {name}")
    old_manifest, old = decode_streams(args.candidate, profile)
    new_manifest, new = decode_streams(args.rebuilt, profile)
    require(old_manifest["files"] == new_manifest["files"], "Rebuild payload bytes are not identical")
    require(old["streams"] == new["streams"], "Rebuilt logical arrays differ")
    core = Path("src/picoagent")
    source_files = {}
    for path in sorted(core.rglob("*")):
        require(not path.is_symlink(), "Core source tree contains a symlink")
        name = str(path.relative_to(core))
        if path.is_file() and path.suffix == ".py" and name != "training/prepared_approvals.py":
            source_files[name] = sha(read(path, 4 * 1024**2))
    require(new_manifest["transform"]["source"] == source_files, "Rebuilt core source identity is stale")
    result = {
        "schema": "picoagent.prepared_tokens.independent_equivalence.v1",
        "audited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "raw_profile": {"path": str(args.raw_profile), "sha256": sha(profile_bytes), "source_manifest_sha256": profile["manifest_sha256"], "source_hashes": profile["source_hashes"]},
        "candidate": old,
        "hardened_rebuild": new,
        "all_payload_files_byte_identical": True,
        "payload_files": len(new_manifest["files"]),
        "payload_bytes": sum(item["bytes"] for item in new_manifest["files"].values()),
        "independent_decoder": "stdlib.array little-endian int32; no prepared module imports",
        "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "production_approval_granted": False,
        "interpretation": "Complete ordered token/attention/assistant-label streams match independently recorded raw production encoding. A separate source/data review and exact digest approval remain required; this report cannot approve itself.",
    }
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(args.output), "hardened_manifest_sha256": new["sha256"], "exact_equivalence": True}, indent=2))


if __name__ == "__main__":
    main()
