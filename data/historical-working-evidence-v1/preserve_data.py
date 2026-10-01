#!/usr/bin/env python3
"""Lossless local historical-byte preservation; never execute or admit data.

Existing sealed files, gzip streams and tar members are reused only by verified
byte hash. Unrepresented bytes enter deterministic <=16 MiB logical tar parts.
No originals are deleted. The report can verify or restore an individual file.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

from source_staging import file_hash, no_symlink_path, relative_path

LIMIT = 16 * 1024 * 1024
PIECE = 8 * 1024 * 1024
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bhf_[A-Za-z0-9]{30,}\b")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest_stream(stream, *, inspect=False):
    digest, size, tail = hashlib.sha256(), 0, b""
    while block := stream.read(1024 * 1024):
        if inspect and SECRET.search(tail + block):
            raise ValueError("Possible credential pattern; preservation stopped without exposing content")
        tail = block[-256:]
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def blocks(root, ref):
    if ref["kind"] == "concat":
        for part in ref["parts"]:
            yield from blocks(root, part)
        return
    path = no_symlink_path(root / relative_path(ref["path"]))
    if file_hash(path) != ref["container_sha256"]:
        raise ValueError("Preserved evidence container changed")
    if ref["kind"] == "tar":
        with tarfile.open(path, "r:gz") as archive:
            member = archive.getmember(ref["member"])
            if not member.isfile():
                raise ValueError("Historical member is not a regular file")
            with archive.extractfile(member) as stream:
                while chunk := stream.read(1024 * 1024):
                    yield chunk
    else:
        opener = gzip.open if ref["kind"] == "gzip" else open
        with opener(path, "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                yield chunk


def ref_record(root, ref):
    digest, size = hashlib.sha256(), 0
    for block in blocks(root, ref):
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def evidence_index(root, manifests):
    index, containers, documents = {}, {}, []
    def remember(info, ref):
        index.setdefault((info["sha256"], info["bytes"]), ref)
    for name in manifests:
        relative_path(name)
        path = no_symlink_path(root / name)
        manifest = json.loads(path.read_text())
        if not isinstance(manifest.get("files"), dict):
            raise ValueError("Canonical evidence requires a files map")
        listed = {name: {"sha256": file_hash(path), "bytes": path.stat().st_size}}
        for relative, info in manifest["files"].items():
            relative_path(relative)
            listed[(path.parent / relative).relative_to(root).as_posix()] = info
        for relative, info in listed.items():
            source = no_symlink_path(root / relative)
            key = info["sha256"]
            if source.stat().st_size != info["bytes"] or file_hash(source) != key:
                raise ValueError("Canonical evidence failed byte verification")
            ref = {"kind": "file", "path": relative, "container_sha256": key}
            remember(info, ref)
            # Identical physical containers need only one decompression pass.
            if key in containers:
                continue
            containers[key] = ref
            if source.name.endswith(".tar.gz"):
                with tarfile.open(source, "r|gz") as archive:
                    seen = set()
                    for member in archive:
                        relative_path(member.name)
                        if not member.isfile() or member.name in seen:
                            raise ValueError("Unsafe historical archive entry")
                        seen.add(member.name)
                        with archive.extractfile(member) as stream:
                            actual = digest_stream(stream, inspect=True)
                        remember(actual, {**ref, "kind": "tar", "member": member.name})
            elif source.name.endswith(".gz"):
                with gzip.open(source, "rb") as stream:
                    actual = digest_stream(stream, inspect=True)
                remember(actual, {**ref, "kind": "gzip"})
            elif source.name == "package_manifest.json":
                documents.append(json.loads(source.read_text()))
    # Python packages declare exact source reconstruction from raw gzip shards
    # and auxiliary member chunks, without JSON parsing/reserialization of data.
    for document in documents:
        groups = []
        for row in document.get("shards", {}).values():
            groups.append(({"sha256": row["source_sha256"], "bytes": row["source_bytes"]},
                           [{"sha256": p["logical_sha256"], "bytes": p["logical_bytes"]} for p in row["shards"]]))
        for row in document.get("aux_source_files", []):
            groups.append(({"sha256": row["sha256"], "bytes": row["bytes"]}, row["members"]))
        for source, pieces in groups:
            key = source["sha256"], source["bytes"]
            if key in index:
                continue
            refs = [index.get((part["sha256"], part["bytes"])) for part in pieces]
            if all(ref is not None for ref in refs):
                ref = {"kind": "concat", "parts": refs}
                if ref_record(root, ref) != source:
                    raise ValueError("Raw source concatenation differs from declared bytes")
                remember(source, ref)
    return index


def inventory(root, sources):
    rows = []
    for name in sorted(sources):
        directory = no_symlink_path(root / relative_path(name))
        if not directory.is_dir():
            raise ValueError("Historical source directory is missing")
        for path in sorted(directory.rglob("*")):
            no_symlink_path(path)
            if path.is_file():
                with path.open("rb") as stream:
                    info = digest_stream(stream, inspect=True)
                rows.append({"path": path.relative_to(root).as_posix(), **info, "mode": path.stat().st_mode & 0o777})
            elif not path.is_dir():
                raise ValueError("Historical source is not a regular file")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Historical source directories overlap")
    return rows


def preserve(root, sources, manifests, output):
    root = no_symlink_path(root)
    output = no_symlink_path(root / relative_path(output))
    if any(output.is_relative_to(root / source) for source in sources):
        raise ValueError("Historical output must be outside originals")
    before = inventory(root, sources)
    index = evidence_index(root, manifests)
    output.mkdir(parents=True, exist_ok=False)
    archive_rows, pending, cost = [], [], 0
    part_refs = {}
    def flush():
        nonlocal pending, cost
        if not pending:
            return
        path = output / f"unrepresented-{len(archive_rows):05d}.tar.gz"
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for name, payload, digest in pending:
                        member = tarfile.TarInfo(name)
                        member.size, member.mode = len(payload), 0o644
                        archive.addfile(member, io.BytesIO(payload))
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError("Historical archive exceeded physical bound")
        container_hash = file_hash(path)
        expected = {name: (digest, len(payload)) for name, payload, digest in pending}
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                with archive.extractfile(member) as stream:
                    actual = digest_stream(stream)
                if (actual["sha256"], actual["bytes"]) != expected.pop(member.name):
                    raise ValueError("New archive readback differs from source bytes")
                part_refs[member.name] = {"kind": "tar", "path": path.relative_to(root).as_posix(),
                                          "container_sha256": container_hash, "member": member.name}
        if expected:
            raise ValueError("New archive omitted source bytes")
        archive_rows.append({"path": path.relative_to(root).as_posix(), "sha256": container_hash, "bytes": path.stat().st_size})
        path.chmod(0o444)
        pending, cost = [], 0
    unresolved = {}
    for row in before:
        key = row["sha256"], row["bytes"]
        if key in index or key in unresolved:
            continue
        pieces = []
        with (root / row["path"]).open("rb") as stream:
            number = 0
            while True:
                payload = stream.read(PIECE)
                if not payload and number:
                    break
                name = f"payload/{row['sha256']}/part-{number:05d}"
                part_cost = 512 + ((len(payload) + 511) // 512) * 512
                if pending and cost + part_cost > LIMIT:
                    flush()
                pending.append((name, payload, hashlib.sha256(payload).hexdigest()))
                cost += part_cost
                pieces.append(name)
                number += 1
                if not payload:
                    break
        unresolved[key] = pieces
    flush()
    for key, pieces in unresolved.items():
        refs = [part_refs[name] for name in pieces]
        index[key] = refs[0] if len(refs) == 1 else {"kind": "concat", "parts": refs}
    after = inventory(root, sources)
    if after != before:
        raise ValueError("Historical originals changed; ignore rules must not be added")
    for row in before:
        row["representation"] = index[row["sha256"], row["bytes"]]
    inventory_path = output / "files.jsonl"
    with inventory_path.open("xb") as stream:
        for row in before:
            stream.write(encoded(row) + b"\n")
    if inventory_path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("Historical inventory exceeded physical bound")
    manifest = {"schema": "picoagent.historical_byte_preservation.v1", "training_eligible": False,
                "originals_unchanged": True, "source_directories": sorted(sources),
                "source_file_count": len(before), "source_bytes": sum(row["bytes"] for row in before),
                "source_inventory_sha256": hashlib.sha256(encoded(after)).hexdigest(),
                "reused_files": sum((row["sha256"], row["bytes"]) not in unresolved for row in before),
                "newly_archived_unique_contents": len(unresolved), "archives": archive_rows,
                "canonical_manifests": {name: file_hash(root / name) for name in manifests},
                "inventory": {"path": inventory_path.relative_to(root).as_posix(), "sha256": file_hash(inventory_path), "bytes": inventory_path.stat().st_size},
                "packager_sha256": file_hash(Path(__file__)),
                "archive_metadata": "fixed uid/gid/mtime zero; regular members; gzip mtime zero; original file permission bits recorded in inventory",
                "credential_screen": "No private-key, AWS access-key or Hugging Face token patterns found; name/content screening is not a guarantee against arbitrary pasted secrets"}
    (output / "manifest.json").write_bytes(encoded(manifest) + b"\n")
    inventory_path.chmod(0o444)
    (output / "manifest.json").chmod(0o444)
    return manifest


def restore_file(root, manifest_path, original, destination):
    """Restore one exact file into a new location; never execute it."""
    manifest = json.loads(manifest_path.read_text())
    info = manifest["inventory"]
    inventory_path = root / relative_path(info["path"])
    if file_hash(inventory_path) != info["sha256"]:
        raise ValueError("Historical inventory hash mismatch")
    row = next((json.loads(line) for line in inventory_path.open() if json.loads(line)["path"] == original), None)
    if row is None:
        raise ValueError("Original path is absent from preservation inventory")
    destination = no_symlink_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with destination.open("xb") as output:
        for block in blocks(root, row["representation"]):
            output.write(block)
            digest.update(block)
            size += len(block)
    if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
        raise ValueError("Restored bytes failed verification")
    destination.chmod(row["mode"])
    return {"sha256": digest.hexdigest(), "bytes": size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("preserve")
    build.add_argument("--source", action="append", required=True)
    build.add_argument("--canonical-manifest", action="append", required=True)
    build.add_argument("--output", required=True)
    restore = sub.add_parser("restore-file")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--original", required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preserve":
        result = preserve(args.root, args.source, args.canonical_manifest, args.output)
        print(json.dumps({k: result[k] for k in ("source_file_count", "source_bytes", "reused_files", "newly_archived_unique_contents", "archives")}))
    else:
        print(json.dumps(restore_file(args.root, args.manifest, args.original, args.destination)))


if __name__ == "__main__":
    main()
