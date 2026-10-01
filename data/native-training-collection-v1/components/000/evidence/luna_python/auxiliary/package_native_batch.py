"""Create deterministic bounded archives from a completed native Python batch.

This utility only packages recorded bytes. It never executes a task program or
changes the source batch. It keeps original JSONL files as the authoritative
raw evidence and writes all shards/archives to a new output directory.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from picoagent.data.schema import canonical_json

MAX_LOGICAL_SHARD = 16 * 1024 * 1024
MAX_AUX_MEMBER = 8 * 1024 * 1024
MAX_TAR_GZ = 25 * 1024 * 1024
ARCHIVE_GROUP_RAW = 10 * 1024 * 1024


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_jsonl_sha256(raw: bytes) -> str:
    h = hashlib.sha256()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("JSONL input must end with a newline")
    for line in raw.splitlines():
        if line:
            h.update((canonical_json(json.loads(line)) + "\n").encode("utf-8"))
    return h.hexdigest()


def canonical_jsonl_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                h.update((canonical_json(json.loads(line)) + "\n").encode("utf-8"))
    return h.hexdigest()


def rows_and_counts(raw: bytes) -> tuple[int, dict[str, int]]:
    rows = 0
    counts: dict[str, int] = {}
    for line in raw.splitlines():
        if not line:
            continue
        row = json.loads(line)
        rows += 1
        split = row.get("split")
        if isinstance(split, str):
            counts[split] = counts.get(split, 0) + 1
    return rows, counts


def gzip_bytes(name: str, raw: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename=name, mode="wb", fileobj=output, compresslevel=9, mtime=0) as stream:
        stream.write(raw)
    return output.getvalue()


def split_lines(path: Path, limit: int):
    """Yield exact line-boundary byte chunks no larger than limit."""
    buffer = bytearray()
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > limit:
                raise ValueError(f"single line exceeds bounded shard: {path}")
            if buffer and len(buffer) + len(line) > limit:
                yield bytes(buffer)
                buffer.clear()
            buffer.extend(line)
    if buffer:
        yield bytes(buffer)


def add_shards(source: Path, package: Path, category: str, rel: str) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    canonical_hash = canonical_jsonl_sha256(raw)
    row_count, split_counts = rows_and_counts(raw)
    shard_dir = package / "shards" / category
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    logical_total = 0
    for index, chunk in enumerate(split_lines(source, MAX_LOGICAL_SHARD), start=1):
        name = f"{category}-{index:05d}.jsonl.gz"
        compressed = gzip_bytes("", chunk)
        (shard_dir / name).write_bytes(compressed)
        restored = gzip.decompress(compressed)
        if restored != chunk:
            raise AssertionError(f"gzip read-back mismatch: {name}")
        shards.append({
            "path": f"shards/{category}/{name}",
            "logical_bytes": len(chunk),
            "logical_sha256": sha256(chunk),
            "canonical_sha256": canonical_jsonl_sha256(chunk),
            "compressed_bytes": len(compressed),
            "compressed_sha256": sha256(compressed),
            "rows": sum(1 for line in chunk.splitlines() if line),
        })
        logical_total += len(chunk)
    if logical_total != len(raw):
        raise AssertionError(f"shards do not cover source bytes: {rel}")
    return {
        "source_path": rel,
        "source_bytes": len(raw),
        "source_sha256": sha256(raw),
        "canonical_sha256": canonical_hash,
        "rows": row_count,
        "split_counts": split_counts,
        "shards": shards,
    }


def split_aux_file(source: Path, archive_name: str):
    """Yield archive member names and bytes; line-split oversized text files."""
    if source.stat().st_size <= MAX_AUX_MEMBER:
        yield archive_name, source.read_bytes()
        return
    if source.suffix != ".jsonl":
        raise ValueError(f"oversized non-JSONL auxiliary file: {source}")
    for index, chunk in enumerate(split_lines(source, MAX_AUX_MEMBER), start=1):
        yield f"{archive_name}.part-{index:05d}.jsonl", chunk


def tar_gz_bytes(members: list[tuple[str, bytes]]) -> bytes:
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, compresslevel=9, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, data in members:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    return compressed.getvalue()


def create_aux_archives(source_root: Path, package: Path) -> tuple[list[dict], list[dict]]:
    files = [
        source_root / "events.jsonl",
        source_root / "progress.jsonl",
        source_root / "failures.jsonl",
        source_root / "manifest.json",
        source_root / "tasks" / "manifest.json",
    ]
    snapshot = source_root / "source_snapshot"
    if snapshot.exists():
        files.extend(sorted(p for p in snapshot.rglob("*") if p.is_file()))
    members: list[tuple[str, bytes]] = []
    aux_sources = []
    member_source: dict[str, str] = {}
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        rel = path.relative_to(source_root).as_posix()
        source_members = list(split_aux_file(path, rel))
        metadata = {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "canonical_sha256": canonical_jsonl_file_sha256(path) if path.suffix == ".jsonl" else None,
            "members": [],
        }
        for member_name, data in source_members:
            if member_name in member_source:
                raise ValueError(f"duplicate auxiliary member path: {member_name}")
            member_source[member_name] = rel
            members.append((member_name, data))
            metadata["members"].append({"path": member_name, "bytes": len(data), "sha256": sha256(data)})
        aux_sources.append(metadata)

    archives_dir = package / "aux_archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    groups: list[list[tuple[str, bytes]]] = []
    current: list[tuple[str, bytes]] = []
    current_bytes = 0
    for member in members:
        if current and current_bytes + len(member[1]) > ARCHIVE_GROUP_RAW:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(member)
        current_bytes += len(member[1])
    if current:
        groups.append(current)

    archives = []
    readback_state = {
        item["path"]: {
            "raw": hashlib.sha256(),
            "canonical": hashlib.sha256(),
            "bytes": 0,
        }
        for item in aux_sources
    }
    source_by_path = {item["path"]: item for item in aux_sources}
    for index, group in enumerate(groups, start=1):
        name = f"aux-{index:04d}.tar.gz"
        packed = tar_gz_bytes(group)
        if len(packed) > MAX_TAR_GZ:
            raise ValueError(f"compressed archive exceeds 25 MiB: {name} ({len(packed)})")
        target = archives_dir / name
        target.write_bytes(packed)
        member_rows = []
        group_by_name = dict(group)
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(packed)), mode="r:") as tar:
            for info in tar.getmembers():
                extracted = tar.extractfile(info)
                if extracted is None:
                    raise AssertionError(f"missing archive member: {info.name}")
                data = extracted.read()
                original = group_by_name.get(info.name)
                if original is None or data != original:
                    raise AssertionError(f"tar read-back mismatch: {info.name}")
                member_rows.append({"name": info.name, "bytes": len(data), "sha256": sha256(data)})
                source_path = member_source[info.name]
                state = readback_state[source_path]
                state["raw"].update(data)
                state["bytes"] += len(data)
                if source_by_path[source_path]["canonical_sha256"] is not None:
                    for line in data.splitlines():
                        if line:
                            state["canonical"].update((canonical_json(json.loads(line)) + "\n").encode("utf-8"))
        archives.append({
            "path": f"aux_archives/{name}",
            "compressed_bytes": len(packed),
            "compressed_sha256": sha256(packed),
            "member_count": len(member_rows),
            "members": member_rows,
        })
    for source in aux_sources:
        state = readback_state[source["path"]]
        if state["bytes"] != source["bytes"] or state["raw"].hexdigest() != source["sha256"]:
            raise AssertionError(f"auxiliary archive aggregation mismatch: {source['path']}")
        if source["canonical_sha256"] is not None and state["canonical"].hexdigest() != source["canonical_sha256"]:
            raise AssertionError(f"auxiliary canonical aggregation mismatch: {source['path']}")
    return archives, aux_sources


def package_batch(source_root: Path, package: Path) -> dict:
    if package.exists():
        raise FileExistsError(f"package output already exists: {package}")
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    batch_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if batch_manifest.get("status") != "complete":
        raise ValueError("source batch must be complete")
    if batch_manifest.get("configuration", {}).get("test_execution") is not False:
        raise ValueError("source batch must have test execution disabled")
    if batch_manifest.get("execution") != "native_teacher_observed":
        raise ValueError("source batch execution provenance is not native_teacher_observed")
    if batch_manifest.get("teacher_model") is not None or batch_manifest.get("sft_admissible") is not False:
        raise ValueError("source batch must preserve null teacher and ineligible status")
    expected_hashes = {
        "records.jsonl": "records_sha256",
        "events.jsonl": "events_sha256",
        "progress.jsonl": "progress_sha256",
        "failures.jsonl": "failures_sha256",
        "candidate_plans.reviewed_unexecuted.jsonl": "candidate_file_sha256",
        "tasks/manifest.json": "task_manifest_sha256",
    }
    for rel, manifest_key in expected_hashes.items():
        path = source_root / rel
        if path.is_symlink():
            raise ValueError(f"symbolic link is not permitted in source batch: {rel}")
        actual = file_sha256(path)
        if actual != batch_manifest.get(manifest_key):
            raise ValueError(f"source batch hash mismatch for {rel}")
    source_snapshot = source_root / "source_snapshot"
    for rel, digest in batch_manifest.get("source_sha256", {}).items():
        path = source_snapshot / rel
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError(f"frozen source snapshot mismatch for {rel}")
    task_manifest = json.loads((source_root / "tasks" / "manifest.json").read_text(encoding="utf-8"))
    for rel, metadata in task_manifest.get("files", {}).items():
        digest = metadata.get("sha256") if isinstance(metadata, dict) else metadata
        path = source_root / "tasks" / rel
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError(f"task sidecar hash mismatch for {rel}")
    package.mkdir(parents=True)
    shards = {
        "records": add_shards(source_root / "records.jsonl", package, "records", "records.jsonl"),
        "train_tasks": add_shards(source_root / "tasks" / "train.tasks.jsonl", package, "train_tasks", "tasks/train.tasks.jsonl"),
        "dev_tasks": add_shards(source_root / "tasks" / "dev.tasks.jsonl", package, "dev_tasks", "tasks/dev.tasks.jsonl"),
        "candidate_plans": add_shards(source_root / "candidate_plans.reviewed_unexecuted.jsonl", package, "candidate_plans", "candidate_plans.reviewed_unexecuted.jsonl"),
    }
    archives, aux_sources = create_aux_archives(source_root, package)
    output = {
        "schema": "picoagent.native_batch_package.v1",
        "source_batch_manifest_sha256": file_sha256(manifest_path),
        "source_batch_execution": batch_manifest.get("execution"),
        "packager_sha256": file_sha256(Path(__file__)),
        "source_teacher_model": batch_manifest.get("teacher_model"),
        "source_teacher_decision_mode": batch_manifest.get("teacher_decision_mode"),
        "source_sft_admissible": batch_manifest.get("sft_admissible"),
        "source_hashes": batch_manifest.get("source_sha256"),
        "shards": shards,
        "aux_archives": archives,
        "aux_source_files": aux_sources,
        "verification": {
            "gzip_readback": True,
            "tar_member_readback": True,
            "aux_aggregate_readback": True,
            "bounded_logical_shards": all(
                shard["logical_bytes"] <= MAX_LOGICAL_SHARD
                for category in shards.values() for shard in category["shards"]
            ),
            "bounded_aux_archives": all(archive["compressed_bytes"] <= MAX_TAR_GZ for archive in archives),
            "test_execution": False,
        },
    }
    manifest_out = package / "package_manifest.json"
    manifest_out.write_text(canonical_json(output) + "\n", encoding="utf-8")
    # Read the package manifest and all emitted file hashes back from disk.
    reread = json.loads(manifest_out.read_text(encoding="utf-8"))
    if reread != output:
        raise AssertionError("package manifest read-back mismatch")
    for group in shards.values():
        for shard in group["shards"]:
            packed = (package / shard["path"]).read_bytes()
            if sha256(packed) != shard["compressed_sha256"]:
                raise AssertionError(f"compressed shard hash mismatch: {shard['path']}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = package_batch(args.source_dir, args.output_dir)
    print(canonical_json({
        "output": str(args.output_dir),
        "schema": manifest["schema"],
        "categories": {key: value["rows"] for key, value in manifest["shards"].items()},
        "aux_archives": len(manifest["aux_archives"]),
        "verification": manifest["verification"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
