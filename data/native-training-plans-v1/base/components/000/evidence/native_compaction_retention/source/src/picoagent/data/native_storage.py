"""Bounded gzip shards for reviewed native evidence; no tool execution."""
from __future__ import annotations

import copy
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterator

from .audit import file_hash, write_new_json
from .schema import canonical_json, content_hash, validate_trace

STORAGE = "gzip_sharded_v1"
MAX_FILE_BYTES = 25 * 1024 * 1024
DEFAULT_SHARD_BYTES = 16 * 1024 * 1024


def iter_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL records must be objects")
            yield row


def metadata(path: Path) -> dict[str, Any]:
    """Compressed bytes and exact/canonical logical row hashes are separate."""
    info: dict[str, Any] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    if path.name.endswith(".jsonl.gz"):
        exact, canonical, size, rows = hashlib.sha256(), hashlib.sha256(), 0, 0
        with gzip.open(path, "rb") as handle:
            for line in handle:
                exact.update(line)
                size += len(line)
                canonical.update((canonical_json(json.loads(line)) + "\n").encode())
                rows += 1
        info.update(compression="gzip", logical_sha256=exact.hexdigest(), logical_bytes=size,
                    canonical_rows_sha256=canonical.hexdigest(), records=rows)
    elif path.name.endswith(".jsonl"):
        canonical, rows = hashlib.sha256(), 0
        with path.open("rb") as handle:
            for line in handle:
                canonical.update((canonical_json(json.loads(line)) + "\n").encode())
                rows += 1
        info.update(logical_sha256=info["sha256"], logical_bytes=info["bytes"], canonical_rows_sha256=canonical.hexdigest(), records=rows)
    return info


class ShardWriter:
    def __init__(self, root: Path, prefix: str, *, limit: int = DEFAULT_SHARD_BYTES):
        if not 1024 <= limit <= DEFAULT_SHARD_BYTES:
            raise ValueError("shard logical byte limit must be 1KiB..16MiB")
        self.root, self.prefix, self.limit = root, prefix, limit
        self.paths: list[str] = []
        self.count = 0
        self._stream = None
        self._raw = None
        self._size = self._rows = 0

    def _open(self) -> None:
        relative = f"{self.prefix}/part-{len(self.paths):05d}.jsonl.gz"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = path.open("xb")
        self._stream = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0, compresslevel=6)
        self.paths.append(relative)
        self._size = self._rows = 0

    def add(self, row: dict[str, Any]) -> tuple[str, int]:
        data = (canonical_json(row) + "\n").encode("utf-8")
        if len(data) > self.limit:
            raise ValueError("one evidence record exceeds the bounded shard size")
        if self._stream is None or self._size + len(data) > self.limit:
            self.close()
            self._open()
        location = (self.paths[-1], self._rows)
        self._stream.write(data)
        self._size += len(data)
        self._rows += 1
        self.count += 1
        return location

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            if (self.root / self.paths[-1]).stat().st_size > MAX_FILE_BYTES:
                raise ValueError("compressed native shard exceeds 25MiB")
            self._stream = self._raw = None


def _copy_file(source: str | Path, root: Path, relative: str) -> str:
    from .native_admission import _inside, _need
    source = Path(source)
    _need(source.stat().st_size <= MAX_FILE_BYTES, "source evidence exceeds 25MiB; freeze it into bounded gzip shards/archives first")
    compress_jsonl = source.name.endswith(".jsonl") and ("/tasks/" in relative or "/candidates/" in relative)
    if compress_jsonl:
        relative += ".gz"
    target = _inside(root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        if compress_jsonl:
            with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0, compresslevel=6) as compressed:
                shutil.copyfileobj(incoming, compressed)
        else:
            shutil.copyfileobj(incoming, outgoing)
    return relative


def _trace(task, candidate, raw, projection, review, references, tools, oracle):
    from .native_admission import NATIVE_EVIDENCE_SCHEMA, TEACHER_MODE
    from .oracles import check_task_result
    verification = check_task_result(task, projection["final"], artifacts=projection["artifacts"], kv=projection["kv"])
    if verification["passed"] is not oracle["passed"]:
        raise ValueError("independent native oracle disagrees with frozen task oracle")
    audit = {"independent_recomputation": True, "passed": oracle["passed"], "details": oracle,
             "task_sha256": content_hash(task), "oracle_id": review["oracle_id"], "source_sha256": review["oracle_source_sha256"]}
    evidence = {"schema": NATIVE_EVIDENCE_SCHEMA, "source_id": review["source_id"], "source_sha256": review["source_sha256"],
                "review": review, "review_sha256": content_hash(review), **references,
                "raw_record": raw, "raw_record_sha256": content_hash(raw), "task": task, "candidate": candidate,
                "candidate_sha256": content_hash(candidate), "raw_projection": projection, "raw_projection_sha256": content_hash(projection),
                "receipts": projection["receipts"], "artifacts": projection["artifacts"], "kv": projection["kv"],
                "artifacts_sha256": content_hash(projection["artifacts"]), "kv_sha256": content_hash(projection["kv"]), "oracle_audit": audit}
    return {"schema_version": task["schema_version"], "trace_id": f"native:{review['source_id']}:{content_hash(raw)}",
            **{key: task[key] for key in ("task_id", "family", "template_id", "split")},
            "task_sha256": content_hash(task), "raw_attempt_sha256": content_hash(raw),
            "status": "success" if oracle["passed"] else "failed", "verification": verification,
            "provenance": {**task["provenance"], "execution": "native_teacher_observed", "teacher_model": None,
                           "teacher_decision_mode": TEACHER_MODE, "teacher_mode": TEACHER_MODE, "teacher": review["source_id"] + ":procedural_callback", "runtime": projection["runtime"],
                           "container_semantic_replay": "not_verified", "context_compaction_enabled": any(event.get("type") == "compaction" for event in projection["model_events"]),
                           "accepted_compactions": sum(event.get("type") == "compaction" and event.get("accepted") is True for event in projection["model_events"]), "source_review_sha256": content_hash(review)},
            "tools": tools, "native_evidence": evidence,
            **{key: projection[key] for key in ("messages", "effective_messages", "model_events", "tool_events")}}


def seal_sharded_snapshot(sources: list[dict], destination: str | Path, *, shard_bytes: int = DEFAULT_SHARD_BYTES) -> Path:
    from . import native_admission as admission
    from picoagent.harness.tools import TOOL_SCHEMAS
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    all_writer = ShardWriter(root, "observations", limit=shard_bytes)
    selections = {split: ShardWriter(root, f"selection/{split}", limit=shard_bytes) for split in ("train", "dev")}
    observed, admitted, family_sets, template_sets = {}, {}, {"train": set(), "dev": set()}, {"train": set(), "dev": set()}
    seen_tasks: set[str] = set()
    try:
        for spec in sources:
            review = copy.deepcopy(spec["review"])
            admission._validate_review(review)
            source_id, source_hash = review["source_id"], content_hash(review)
            admission._need(source_id not in observed, "one specification per source required")
            prefix = f"evidence/{source_id}"
            tools = spec.get("tool_schemas", TOOL_SCHEMAS)
            code_paths = {}
            admission._need(set(spec["code_paths"]) == set(review["source_sha256"]), "source file set differs from pinned review")
            for name, original in spec["code_paths"].items():
                admission._need(file_hash(original) == review["source_sha256"][name], "reviewed source changed before sealing")
                code_paths[name] = _copy_file(original, root, f"{prefix}/source/{name}")
            for name, original in spec.get("extra_paths", {}).items():
                _copy_file(original, root, f"{prefix}/auxiliary/{name}")
            review_path = f"{prefix}/review.json"
            write_new_json(root / review_path, review)
            indexes = {"task": {}, "candidate": {}}
            for kind in indexes:
                for file_index, original in enumerate(spec[kind + "_paths"]):
                    suffix = ".jsonl.gz" if str(original).endswith(".gz") else ".jsonl"
                    relative = _copy_file(original, root, f"{prefix}/{kind}s/{file_index:03d}{suffix}")
                    for index, item in enumerate(iter_rows(root / relative)):
                        identity = item["task_id"]
                        admission._need(identity not in indexes[kind], "duplicate source task/candidate identity")
                        indexes[kind][identity] = (item, relative, index)
            observed[source_id] = {"observed": 0, "oracle_failed": 0, "duplicate_success": 0, "admitted": 0}
            admitted[source_id] = {"train": 0, "dev": 0}
            oracle_name = next(name for name, digest in review["source_sha256"].items() if digest == review["oracle_source_sha256"])
            for file_index, original in enumerate(spec["observation_paths"]):
                admission._need(file_hash(original) in admission.APPROVED_OBSERVATION_FILES[source_hash], "actual observation shard has not been reviewed")
                suffix = ".jsonl.gz" if str(original).endswith(".gz") else ".jsonl"
                raw_path = _copy_file(original, root, f"{prefix}/observations/{file_index:03d}{suffix}")
                for raw_index, raw in enumerate(iter_rows(root / raw_path)):
                    identity = raw.get("task_id") or raw.get("task", {}).get("task_id")
                    task, task_path, task_index = indexes["task"][identity]
                    candidate, candidate_path, candidate_index = indexes["candidate"][identity]
                    admission._need(task["split"] in selections, "test/benchmark data cannot enter a native snapshot")
                    projection = admission.extract_native_observation(source_id, raw, task)
                    admission._need(content_hash(tools) == projection["tool_schemas_sha256"], "native tool schema mismatch")
                    oracle = admission._independent_oracle(source_id, task, projection["final"], projection["artifacts"], projection["kv"],
                                                          source_sha256=review["oracle_source_sha256"], source_path=root / code_paths[oracle_name])
                    references = {"source_paths": code_paths, "review_path": review_path, "raw_path": raw_path, "raw_record_index": raw_index,
                                  "task_path": task_path, "task_record_index": task_index, "candidate_path": candidate_path, "candidate_record_index": candidate_index}
                    trace = _trace(task, candidate, raw, projection, review, references, tools, oracle)
                    validate_trace(trace, allow_native_teacher=True)
                    location, index = all_writer.add(trace)
                    observed[source_id]["observed"] += 1
                    if not oracle["passed"]:
                        observed[source_id]["oracle_failed"] += 1
                    elif identity in seen_tasks:
                        observed[source_id]["duplicate_success"] += 1
                    else:
                        seen_tasks.add(identity)
                        split = task["split"]
                        selections[split].add({"path": location, "row_index": index, "trace_sha256": content_hash(trace),
                                               "trace_id": trace["trace_id"], "task_id": identity, "split": split})
                        observed[source_id]["admitted"] += 1
                        admitted[source_id][split] += 1
                        family_sets[split].add(task["family"])
                        template_sets[split].add(task["template_id"])
    finally:
        all_writer.close()
        for writer in selections.values():
            writer.close()
    admission._need(all(writer.count for writer in selections.values()), "native snapshot needs nonempty train/dev")
    files = {str(path.relative_to(root)): metadata(path) for path in sorted(root.rglob("*")) if path.is_file()}
    admission._need(all(entry["bytes"] <= MAX_FILE_BYTES for entry in files.values()), "native archive file exceeds 25MiB")
    manifest = {"schema": admission.NATIVE_MANIFEST_SCHEMA, "storage": STORAGE, "admission": "audited_native_teacher_observed_only",
                "lockbox_used": False, "teacher_mode": admission.TEACHER_MODE, "container_semantic_replay": "not_verified",
                "arbitrary_learner_execution_allowed": False, "source_counts": {"observed": observed, "admitted": admitted},
                "all_observations": {"paths": all_writer.paths, "records": all_writer.count},
                "splits": {split: {"index_paths": writer.paths, "records": writer.count, "families": sorted(family_sets[split]),
                                   "templates": sorted(template_sets[split])} for split, writer in selections.items()},
                "files": files, "max_file_bytes": MAX_FILE_BYTES, "logical_shard_limit": shard_bytes,
                "selection": "first successful original variant per task in declared source order",
                "limitations": ["Native observed deterministic teacher replay, not adaptive model sampling or container parity.",
                                "Training verification returns lightweight views only after checking the complete external evidence."]}
    write_new_json(root / "manifest.json", manifest)
    admission.verify_native_snapshot(root / "manifest.json", allow_native_teacher=True)
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    return root / "manifest.json"


def verify_sharded_snapshot(path: Path, manifest: dict) -> tuple[dict, dict[str, list[dict]]]:
    from . import native_admission as admission
    root, files = path.parent, manifest["files"]
    admission._need(manifest.get("admission") == "audited_native_teacher_observed_only" and manifest.get("lockbox_used") is False, "invalid sharded native admission")
    admission._need(manifest.get("teacher_mode") == admission.TEACHER_MODE and manifest.get("arbitrary_learner_execution_allowed") is False and manifest.get("container_semantic_replay") == "not_verified", "invalid sharded native execution claims")
    admission._need(set(manifest["splits"]) == {"train", "dev"}, "native shards require only train/dev")
    for relative, info in files.items():
        target = admission._inside(root, relative)
        admission._need(target.is_file() and type(info.get("bytes")) is int
                        and target.stat().st_size == info["bytes"] <= MAX_FILE_BYTES
                        and file_hash(target) == info.get("sha256"),
                        f"native shard integrity mismatch: {relative}")
        admission._need(info == metadata(target), f"native shard integrity mismatch: {relative}")
    wanted = {}
    for split, descriptor in manifest["splits"].items():
        count = 0
        for relative in descriptor["index_paths"]:
            admission._need(relative in files, "unhashed native selection shard")
            for selection in iter_rows(root / relative):
                key = (selection["path"], selection["row_index"])
                admission._need(key not in wanted and selection["split"] == split, "duplicate or mislabeled native selection")
                wanted[key] = selection
                count += 1
        admission._need(count == descriptor["records"], "native selection count mismatch")

    # At most three source shards are materialized at once; no whole 10k raw
    # dataset is cached. Returned rows omit heavy audit copies after validation.
    @lru_cache(maxsize=3)
    def original_rows(relative: str) -> list[dict]:
        admission._need(relative in files and files[relative].get("logical_bytes", files[relative]["bytes"]) <= MAX_FILE_BYTES, "oversized or unhashed native source shard")
        return list(iter_rows(admission._inside(root, relative)))

    @lru_cache(maxsize=16)
    def frozen_review(relative: str) -> dict:
        return json.loads((root / relative).read_text())

    records = {"train": [], "dev": []}
    observed, admitted, seen, covered = {}, {}, set(), set()
    total = 0
    for relative in manifest["all_observations"]["paths"]:
        admission._need(relative in files, "unhashed native observation shard")
        for index, row in enumerate(iter_rows(root / relative)):
            total += 1
            validate_trace(row, allow_native_teacher=True)
            evidence, split = row["native_evidence"], row["split"]
            admission._need(split in records, "test/lockbox row in native shards")
            source_id, review = evidence["source_id"], evidence["review"]
            observed.setdefault(source_id, {"observed": 0, "oracle_failed": 0, "duplicate_success": 0, "admitted": 0})
            admitted.setdefault(source_id, {"train": 0, "dev": 0})
            observed[source_id]["observed"] += 1
            for kind, key in (("raw_record", "raw"), ("task", "task"), ("candidate", "candidate")):
                source_path, source_index = evidence[key + "_path"], evidence[key + "_record_index"]
                originals = original_rows(source_path)
                admission._need(0 <= source_index < len(originals) and originals[source_index] == evidence[kind], "native source row differs from frozen original")
            raw_path = evidence["raw_path"]
            admission._need(files[raw_path]["sha256"] in admission.APPROVED_OBSERVATION_FILES[evidence["review_sha256"]], "native source shard was never approved")
            raw_key = (raw_path, evidence["raw_record_index"])
            admission._need(raw_key not in covered, "raw native observation projected more than once")
            covered.add(raw_key)
            review_path = evidence["review_path"]
            admission._need(review_path in files and frozen_review(review_path) == review, "native frozen source review changed")
            for name, digest in evidence["source_sha256"].items():
                source_path = evidence["source_paths"][name]
                admission._need(source_path in files and files[source_path]["sha256"] == digest, "native source code differs from reviewed snapshot")
            admission.verify_native_context_tokens(row, root)
            oracle_name = next(name for name, digest in review["source_sha256"].items() if digest == review["oracle_source_sha256"])
            checked = admission._independent_oracle(source_id, evidence["task"], evidence["raw_projection"]["final"], evidence["artifacts"], evidence["kv"],
                                                   source_sha256=review["oracle_source_sha256"], source_path=root / evidence["source_paths"][oracle_name])
            admission._need(checked == evidence["oracle_audit"]["details"], "native independent oracle result changed")
            location = (relative, index)
            if not checked["passed"]:
                observed[source_id]["oracle_failed"] += 1
                admission._need(location not in wanted, "failed native observation selected for SFT")
            elif row["task_id"] in seen:
                observed[source_id]["duplicate_success"] += 1
                admission._need(location not in wanted, "duplicate native task selected")
            else:
                seen.add(row["task_id"])
                selection = wanted.pop(location, None)
                admission._need(selection is not None and selection["trace_sha256"] == content_hash(row) and selection["trace_id"] == row["trace_id"] and selection["task_id"] == row["task_id"] and selection["split"] == split, "native first-success selection mismatch")
                observed[source_id]["admitted"] += 1
                admitted[source_id][split] += 1
                # A lightweight training projection is not independently
                # admissible JSONL; only this verified manifest can supply it.
                light = {key: value for key, value in row.items() if key not in {"native_evidence", "tool_events", "effective_messages"}}
                light["native_evidence_reference"] = {"path": relative, "row_index": index, "trace_sha256": selection["trace_sha256"]}
                records[split].append(light)
    expected_raw = set()
    for relative, info in files.items():
        if relative.startswith("evidence/") and "/observations/" in relative:
            expected_raw.update((relative, index) for index in range(info["records"]))
    admission._need(covered == expected_raw and not wanted and total == manifest["all_observations"]["records"], "native shard observation coverage mismatch")
    admission._need(manifest["source_counts"] == {"observed": observed, "admitted": admitted}, "native shard source counts mismatch")
    from picoagent.training.data import _check_disjoint
    _check_disjoint(records)
    for split, rows in records.items():
        admission._need(len(rows) == manifest["splits"][split]["records"], "native admitted split count mismatch")
    return manifest, records
