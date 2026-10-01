"""Explicit admission of reviewed native teacher observations, never a runtime.

This module does not launch commands or enable local learner execution. It reads
original observation artifacts, binds their projections, and copies sealed
snapshots. Native evidence is separate from Docker/Podman verification.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from .audit import file_hash, read_jsonl
from .oracles import check_task_result
from .schema import DataValidationError, canonical_json, content_hash, safe_relative_path, validate_task

NATIVE_MANIFEST_SCHEMA = "picoagent.native_teacher.dataset.v1"
NATIVE_EVIDENCE_SCHEMA = "picoagent.native_teacher.evidence.v1"
NATIVE_REVIEW_SCHEMA = "picoagent.native_teacher.source_review.v1"
LEGACY_TEACHER_MODE = "luna_authored_program_deterministic_replay"
TEACHER_MODE = "reviewed_procedural_replay"
SOURCE_KINDS = frozenset({"luna_cli", "luna_python", "luna_recovery"})
# Reviewed recorder journals show these exact tool schemas. Source-hash mapping
# keeps old snapshots independent of later changes to installed descriptions.
SOURCE_TOOL_PROTOCOLS = {
    "72315a78fe82416e6471329955df8b7f28998727a4b7f7d84fa2906095bed0a8":
        "aea60579c4a93836e06073e839a566d95b2ead2f6c52223a4af0dd4ecf72d07b",
}
# Populated only after actual artifacts and their generator/recorder/oracle code
# have been independently reviewed. A generic caller-provided approved flag is
# not an admission capability.
_ORACLE_MODULE_CACHE: dict[tuple[str, str], Any] = {}

APPROVED_SOURCE_REVIEWS: dict[str, frozenset[str]] = {
    "luna_python": frozenset({"8778addde92a5549d21b6b9428d5ac2e7529017e4dc64551bd087211e98cc97f"}),
    "luna_cli": frozenset({
        "35e1df0499fa3411fa4607495dc1f127172999da3b8fa8412ab81692602be0cd",
        "42c8252510b67dfa3b735b4fdab9f63b3b9ee3cd63e4895ccd3a28fd41919127",
    }),
}


# Actual observation files reviewed alongside the source. A source approval is
# not permission to invent a new native record. Large batches should be frozen
# into a small number of aggregate JSONL files before this review step.
APPROVED_OBSERVATION_FILES: dict[str, frozenset[str]] = {
    "8778addde92a5549d21b6b9428d5ac2e7529017e4dc64551bd087211e98cc97f": frozenset({"b8a0f6e6197205cffc1960fdd0c51c2a27b68846421f504d066416ada3375cce"}),
    "35e1df0499fa3411fa4607495dc1f127172999da3b8fa8412ab81692602be0cd": frozenset({
        "571aa6ed68cafc5c55bb47ef21274563952065734bf914c233f3037279e9d6f0",
        "084c01a9f5f4505b09ba42300176227e15bdda41e78ff55dd0e08cb7cf8448f8",
    }),
    "42c8252510b67dfa3b735b4fdab9f63b3b9ee3cd63e4895ccd3a28fd41919127": frozenset({
        "532493cfb1f8fa0b08727cbea3caed6f712b84cabf755dd354d8f6ee542f7ea4",
        "ea100012f7ca690a2e1eb74b9f9b62e810a2207ee2a3db11fe073bda3e3a52ab",
        "1d913338f66ec2918cae242172720f4b5813235de66cfdd4c938fac5c9b79531",
        "97542d4514411ce0b525c1c0ab405b0941ac86eb28dfad0e872b46cea8718c1d",
        "17e74fb3c10d8d780b9dd85c5b492b0e2c43877b0933b9098a8bb286e2712bf1",
        "c9f97960ce1ca5b07d2f450e9aa711a2b71215975425ab2f6c3f1b73b79e26ca",
        "bca375b63d86c5616eafdec25d7273dced7c3ae9190bde78b18edb4b1a850be7",
        "feacff2158bfc5af30f1c8f3b698b0d912d361f9367c754090ffa335497bccad",
        "b6ee1c41862db525c1171a728e579996dca3eb721b0127e22138105f6dfc8f06",
    }),
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _bytes(receipt: dict[str, Any], name: str) -> bytes:
    _need(isinstance(receipt.get(name + "_b64"), str), f"native receipt requires {name} bytes")
    try:
        data = base64.b64decode(receipt[name + "_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("invalid native receipt base64") from exc
    _need(hashlib.sha256(data).hexdigest() == receipt.get(name + "_sha256"), f"native {name} byte hash mismatch")
    return data


def _validate_review(review: dict[str, Any]) -> None:
    _need(review.get("schema") == NATIVE_REVIEW_SCHEMA, "missing native source review")
    _need(review.get("source_id") in SOURCE_KINDS, "unknown native source category")
    for field in ("reviewer", "review_notes", "oracle_id", "command_policy"):
        _need(isinstance(review.get(field), str) and bool(review[field]), f"native source review requires {field}")
    _need(review.get("approved") is True, "native source is not reviewed and approved")
    sources = review.get("source_sha256", {})
    _need(isinstance(sources, dict) and bool(sources), "native review must freeze its source files")
    for name, digest in sources.items():
        _need(safe_relative_path(name) and _digest(digest), "invalid reviewed native source hash")
    _need(_digest(review.get("oracle_source_sha256")), "native review requires independently checked oracle source hash")
    _need(review.get("oracle_source_sha256") in sources.values(), "oracle source is not among reviewed frozen files")
    _need(review.get("teacher_mode") in {TEACHER_MODE, LEGACY_TEACHER_MODE}, "native replay cannot claim adaptive model decisions")
    _need(review.get("arbitrary_learner_execution_allowed") is False, "native review must not enable local learner execution")
    _need(content_hash(review) in APPROVED_SOURCE_REVIEWS.get(review["source_id"], frozenset()), "native source review has not been independently pinned for admission")
    _need(bool(APPROVED_OBSERVATION_FILES.get(content_hash(review))), "native review has no actual approved observation artifact")


def validate_native_evidence(trace: dict[str, Any]) -> None:
    """Structural validation plus hash/receipt/replay bindings, with no execution.

    Training must additionally use verify_native_snapshot, which validates the
    frozen external evidence tree and independent source reviews. This function
    alone is not a generic admission route for caller-supplied JSONL.
    """
    evidence = trace.get("native_evidence", {})
    _need(evidence.get("schema") == NATIVE_EVIDENCE_SCHEMA, "native trace lacks sealed evidence")
    review = evidence.get("review", {})
    _validate_review(review)
    _need(evidence.get("review_sha256") == content_hash(review), "native source review hash mismatch")
    _need(evidence.get("source_id") == review["source_id"], "native source identity mismatch")
    _need(evidence.get("source_sha256") == review["source_sha256"], "native execution source differs from reviewed files")
    provenance = trace["provenance"]
    _need(provenance.get("teacher_mode") in {TEACHER_MODE, LEGACY_TEACHER_MODE}, "native teacher mode must disclose deterministic replay")
    if provenance.get("teacher_mode") == TEACHER_MODE:
        _need(provenance.get("teacher_model") is None and provenance.get("teacher_decision_mode") == TEACHER_MODE, "procedural decisions must not be labeled as sampled model output")
    else:
        _need(provenance.get("teacher_model") == "gpt-6-luna", "legacy native author attribution changed; create a corrected new seal")
    _need(provenance.get("container_semantic_replay") == "not_verified", "native evidence cannot claim container parity")
    runtime = provenance["runtime"]
    for key in ("python_version", "platform", "locale"):
        _need(isinstance(runtime.get(key), str) and bool(runtime[key]), f"native runtime requires {key}")
    _need(isinstance(runtime.get("executables"), dict) and bool(runtime["executables"]), "native runtime must identify executables/versions")
    task = evidence.get("task", {})
    validate_task(task)
    _need(trace.get("task_sha256") == content_hash(task), "native frozen task hash mismatch")
    for field in ("task_id", "family", "template_id", "split"):
        _need(trace[field] == task[field], f"native task {field} mismatch")
    candidate = evidence.get("candidate")
    _need(isinstance(candidate, dict) and bool(candidate), "native trace must preserve its original authored candidate")
    _need(evidence.get("candidate_sha256") == content_hash(candidate), "native candidate hash mismatch")
    if "task_id" in candidate:
        _need(candidate["task_id"] == task["task_id"], "native candidate belongs to a different task")
    if "task_sha256" in candidate:
        _need(candidate["task_sha256"] == content_hash(task), "native candidate task hash mismatch")
    if "planned_actions" in candidate:
        _need(candidate["planned_actions"] == task["reference"]["plan"], "native original plan differs from frozen task plan")
    raw = evidence.get("raw_record")
    _need(isinstance(raw, dict) and bool(raw), "native trace must preserve its original observation record")
    _need(trace.get("raw_attempt_sha256") == content_hash(raw), "native raw observation hash mismatch")
    if raw.get("source_snapshot_manifest_sha256") is not None:
        _need(raw["source_snapshot_manifest_sha256"] == review.get("source_snapshot_sha256"), "native source snapshot differs from independently reviewed execution version")
    _need(evidence.get("raw_record_sha256") == content_hash(raw), "native raw record linkage mismatch")
    original_sources = raw.get("source_sha256", raw.get("native_evidence", {}).get("source_sha256", {}))
    _need(all(review["source_sha256"].get(name) == digest for name, digest in original_sources.items()), "native original helper source hashes differ from reviewed snapshot")
    _need(isinstance(evidence.get("raw_record_index"), int) and evidence["raw_record_index"] >= 0, "native source record index required")
    for field in ("raw_path", "review_path", "task_path", "candidate_path"):
        _need(isinstance(evidence.get(field), str) and safe_relative_path(evidence[field]), "native evidence must reference snapshot-local paths")
    receipts = evidence.get("receipts", [])
    events = trace["tool_events"]
    _need(isinstance(receipts, list) and bool(receipts) and len(receipts) == len(events), "native tool events need ordered raw receipts")
    for index, (receipt, event) in enumerate(zip(receipts, events)):
        _need(receipt.get("sequence") == index, "native receipt sequence mismatch")
        for field in ("tool_call_id", "name", "arguments"):
            _need(receipt.get(field) == event.get(field), f"native receipt {field} mismatch")
        _need(receipt.get("result") == event.get("result"), "native receipt result differs from observed tool reply")
        result = event["result"]
        _need(isinstance(result, dict), "native observed tool result must be an object")
        _need("container_id" not in result and result.get("backend") != "container", "native result cannot forge container receipts")
        kind = receipt.get("execution_kind", "subprocess")
        if kind == "subprocess":
            _need(event["name"] in {"bash", "python", "write_file"}, "unexpected native subprocess tool")
            argv = receipt.get("argv")
            _need(isinstance(argv, list) and bool(argv) and all(isinstance(v, str) for v in argv), "native receipt must preserve exact argv")
            _need(isinstance(receipt.get("stdin"), str) and isinstance(receipt.get("cwd"), str) and bool(receipt["cwd"]), "native receipt must preserve stdin and cwd")
            arguments = json.loads(event["arguments"])
            if event["name"] == "bash":
                _need(len(argv) >= 3 and Path(argv[0]).name == "bash" and argv[-2:] == ["-c", arguments.get("command")] and argv[1:-2] in ([], ["--noprofile", "--norc"]), "native Bash argv differs from requested command")
            elif event["name"] == "python":
                code = arguments.get("code")
                _need(isinstance(code, str) and len(argv) >= 3 and Path(argv[0]).name.startswith("python") and argv[-2] == "-c", "native Python invocation must use the reviewed interpreter")
                _need(all(flag in {"-I", "-B", "-u"} for flag in argv[1:-2]), "native Python invocation contains unreviewed flags")
                direct = argv[-1] == code
                runner = (receipt["stdin"] == code and hashlib.sha256(argv[-1].encode("utf-8")).hexdigest() == review.get("python_runner_sha256"))
                _need(direct or runner, "native Python receipt is not bound to requested source or reviewed fixed runner")
            for stream in ("stdout", "stderr"):
                data = _bytes(receipt, stream)
                _need(result.get(stream) == data.decode("utf-8", errors="replace"), f"native decoded {stream} differs from genuine captured bytes")
            _need(type(receipt.get("exit_code")) is int, "native receipt exit status required")
            for field in ("exit_code", "timed_out", "truncated"):
                _need(receipt.get(field) == result.get(field), f"native receipt {field} mismatch")
            _need(type(receipt.get("timed_out")) is bool and type(receipt.get("truncated")) is bool, "native timeout/truncation flags required")
            _need(isinstance(receipt.get("duration_seconds"), (int, float)) and receipt["duration_seconds"] >= 0, "native duration required")
        else:
            _need(kind == "host_function" and event["name"] in {"search", "knowledge"}, "unknown native host observation type")
            _need(isinstance(receipt.get("operation"), dict), "native host function receipt requires exact operation")
            _need(receipt["operation"] == json.loads(event["arguments"]), "native host operation differs from requested arguments")
            _need(_digest(receipt.get("state_before_sha256")) and _digest(receipt.get("state_after_sha256")), "native host state changes require hashes")
    audit = evidence.get("oracle_audit", {})
    _need(audit.get("independent_recomputation") is True and audit.get("passed") is trace["verification"]["passed"], "native independent oracle audit missing")
    _need(audit.get("task_sha256") == trace["task_sha256"] and audit.get("oracle_id") == review["oracle_id"], "native oracle audit source/input mismatch")
    _need(audit.get("source_sha256") == review["oracle_source_sha256"], "native oracle audit source hash mismatch")
    artifacts, kv = evidence.get("artifacts", {}), evidence.get("kv", {})
    _need(isinstance(artifacts, dict) and isinstance(kv, dict), "native post-state artifacts/KV must be preserved")
    _need(evidence.get("artifacts_sha256") == content_hash(artifacts) and evidence.get("kv_sha256") == content_hash(kv), "native post-state hash mismatch")
    actual = check_task_result(task, trace["messages"][-1].get("content") or "", artifacts=artifacts, kv=kv)
    _need(actual["passed"] is audit["passed"], "native recorded outcome disagrees with frozen task oracle")
    # Adapters additionally preserve the raw transcript/model events unchanged.
    projection = evidence.get("raw_projection", {})
    _need(projection == extract_native_observation(evidence["source_id"], raw, task), "native projection is not the exact reviewed raw-source projection")
    for field in ("messages", "effective_messages", "model_events", "tool_events"):
        _need(projection.get(field) == trace.get(field), f"native {field} was changed after observation")
    _need(projection.get("receipts") == receipts, "native receipts differ from source projection")
    _need(projection.get("runtime") == runtime, "native runtime identity differs from original receipt")
    _need(projection.get("artifacts") == artifacts and projection.get("kv") == kv, "native post-state differs from original observation")
    _need(projection.get("task_sha256") == trace["task_sha256"], "native raw observation task binding mismatch")
    _need(projection.get("source_module_sha256") in review["source_sha256"].values(), "native recorder source hash is not reviewed")
    _need(content_hash(trace.get("tools")) == projection.get("tool_schemas_sha256"), "native tool schemas differ from observed source")
    _need(evidence.get("raw_projection_sha256") == content_hash(projection), "native raw projection hash mismatch")


def _inside(root: Path, relative: str) -> Path:
    _need(safe_relative_path(relative), "snapshot path must be relative")
    current = root
    for component in Path(relative).parts:
        current = current / component
        _need(not current.is_symlink(), "snapshot evidence must not be a symlink")
    resolved = current.resolve()
    _need(resolved.is_relative_to(root.resolve()), "snapshot evidence escapes root")
    return current


def verify_native_snapshot(manifest_path: str | Path, *, allow_native_teacher: bool = False) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    _need(allow_native_teacher is True, "native teacher datasets require explicit opt-in")
    from .schema import validate_trace
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _need(manifest.get("schema") == NATIVE_MANIFEST_SCHEMA, "unsupported native teacher manifest")
    if manifest.get("storage") == "gzip_sharded_v1":
        from .native_storage import verify_sharded_snapshot
        return verify_sharded_snapshot(path, manifest)
    _need(manifest.get("admission") == "audited_native_teacher_observed_only", "invalid native admission policy")
    _need(manifest.get("teacher_mode") in {TEACHER_MODE, LEGACY_TEACHER_MODE} and manifest.get("arbitrary_learner_execution_allowed") is False, "native snapshot must disclose teacher replay and prohibit learner-local execution")
    _need(manifest.get("container_semantic_replay") == "not_verified", "native snapshot cannot claim container parity")
    _need(manifest.get("lockbox_used") is False and set(manifest.get("splits", {})) == {"train", "dev"}, "native SFT snapshot must contain only train/dev")
    files = manifest.get("files", {})
    _need(isinstance(files, dict) and bool(files), "native snapshot lacks evidence hashes")
    for relative, entry in files.items():
        file = _inside(path.parent, relative)
        _need(file.is_file() and file.stat().st_size == entry.get("bytes") and file_hash(file) == entry.get("sha256"), f"native snapshot integrity mismatch: {relative}")
    records: dict[str, list[dict[str, Any]]] = {}
    raw_cache: dict[str, list[dict[str, Any]]] = {}
    reviews: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, entry in manifest["splits"].items():
        relative = entry.get("path")
        _need(relative == f"{split}.jsonl" and relative in files, "native split path must be its hashed sibling JSONL")
        rows = read_jsonl(_inside(path.parent, relative))
        _need(bool(rows) and len(rows) == entry.get("records"), "native split record count mismatch")
        for trace in rows:
            validate_trace(trace, allow_native_teacher=True)
            _need(trace["split"] == split and trace["status"] == "success" and trace["provenance"]["execution"] == "native_teacher_observed", "native admission only accepts successful correctly split observations")
            evidence = trace["native_evidence"]
            raw_path, review_path = evidence["raw_path"], evidence["review_path"]
            _need(raw_path in files and review_path in files, "native source evidence is not frozen inside snapshot")
            _need(files[raw_path]["sha256"] in APPROVED_OBSERVATION_FILES[evidence["review_sha256"]], "native observation file was never independently approved")
            for object_name in ("task", "candidate"):
                original_path = evidence[object_name + "_path"]
                original_index = evidence.get(object_name + "_record_index")
                _need(original_path in files and type(original_index) is int and original_index >= 0, "native task/candidate original is not frozen")
                if original_path not in raw_cache:
                    raw_cache[original_path] = read_jsonl(_inside(path.parent, original_path))
                originals = raw_cache[original_path]
                _need(original_index < len(originals) and originals[original_index] == evidence[object_name], "native task/candidate differs from original source record")
            if raw_path not in raw_cache:
                raw_cache[raw_path] = read_jsonl(_inside(path.parent, raw_path))
            index = evidence["raw_record_index"]
            _need(index < len(raw_cache[raw_path]) and raw_cache[raw_path][index] == evidence["raw_record"], "native trace differs from frozen source observation")
            if review_path not in reviews:
                reviews[review_path] = json.loads(_inside(path.parent, review_path).read_text(encoding="utf-8"))
            _need(reviews[review_path] == evidence["review"], "native source review differs from frozen review")
            for name, digest in evidence["source_sha256"].items():
                source_path = evidence["source_paths"].get(name)
                _need(source_path in files and files[source_path]["sha256"] == digest, "native reviewed source code is not frozen in snapshot")
            source_id = evidence["source_id"]
            repeated = _independent_oracle(source_id, evidence["task"], trace["messages"][-1].get("content") or "",
                                           evidence["artifacts"], evidence["kv"], source_sha256=evidence["review"]["oracle_source_sha256"], source_path=_inside(path.parent, evidence["source_paths"][next(name for name, digest in evidence["source_sha256"].items() if digest == evidence["review"]["oracle_source_sha256"])]))
            _need(repeated.get("passed") is True and repeated == evidence["oracle_audit"]["details"], "native independent oracle no longer reproduces audited result")
            counts.setdefault(source_id, {"train": 0, "dev": 0})[split] += 1
        records[split] = rows
    # Reuse normalization-aware duplicate and whole-family holdout checks without
    # granting the generic training JSONL loader any native admission capability.
    from picoagent.training.data import _check_disjoint
    _check_disjoint(records)
    all_path = "all_projected_observations.jsonl"
    _need(all_path in files, "native snapshot must preserve every projected failure/retry")
    all_rows = read_jsonl(_inside(path.parent, all_path))
    observed: dict[str, dict[str, int]] = {}
    admitted: dict[str, dict[str, int]] = {}
    expected_splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    seen_task_ids: set[str] = set()
    covered: set[tuple[str, int]] = set()
    for row in all_rows:
        validate_trace(row, allow_native_teacher=True)
        evidence = row["native_evidence"]
        source_id = evidence["source_id"]
        observed.setdefault(source_id, {"observed": 0, "oracle_failed": 0, "duplicate_success": 0, "admitted": 0})
        admitted.setdefault(source_id, {"train": 0, "dev": 0})
        observed[source_id]["observed"] += 1
        raw_path, index = evidence["raw_path"], evidence["raw_record_index"]
        _need(raw_path in files and raw_path.startswith(f"evidence/{source_id}/observations/"), "native raw record is outside its source evidence tree")
        _need(files[raw_path]["sha256"] in APPROVED_OBSERVATION_FILES[evidence["review_sha256"]], "native observation file was never independently approved")
        if raw_path not in raw_cache:
            raw_cache[raw_path] = read_jsonl(_inside(path.parent, raw_path))
        _need(index < len(raw_cache[raw_path]) and raw_cache[raw_path][index] == evidence["raw_record"], "native projected failure/retry differs from raw observation")
        _need((raw_path, index) not in covered, "native raw observation projected more than once")
        covered.add((raw_path, index))
        for kind in ("task", "candidate"):
            original_path, original_index = evidence[kind + "_path"], evidence[kind + "_record_index"]
            _need(original_path in files, "native original task/candidate not frozen")
            if original_path not in raw_cache:
                raw_cache[original_path] = read_jsonl(_inside(path.parent, original_path))
            _need(0 <= original_index < len(raw_cache[original_path]) and raw_cache[original_path][original_index] == evidence[kind], "native projected task/candidate differs from frozen original")
        repeated = _independent_oracle(source_id, evidence["task"], row["messages"][-1].get("content") or "", evidence["artifacts"], evidence["kv"], source_sha256=evidence["review"]["oracle_source_sha256"], source_path=_inside(path.parent, evidence["source_paths"][next(name for name, digest in evidence["source_sha256"].items() if digest == evidence["review"]["oracle_source_sha256"])]))
        _need(repeated == evidence["oracle_audit"]["details"] and repeated.get("passed") is row["verification"]["passed"], "native failure/retry oracle outcome changed")
        _need(row["split"] in expected_splits, "test/lockbox record present in native training observations")
        if not repeated["passed"]:
            observed[source_id]["oracle_failed"] += 1
        elif row["task_id"] in seen_task_ids:
            observed[source_id]["duplicate_success"] += 1
        else:
            seen_task_ids.add(row["task_id"])
            expected_splits[row["split"]].append(row)
            observed[source_id]["admitted"] += 1
            admitted[source_id][row["split"]] += 1
    all_raw_keys: set[tuple[str, int]] = set()
    for relative in files:
        if relative.startswith("evidence/") and "/observations/" in relative and relative.endswith(".jsonl"):
            if relative not in raw_cache:
                raw_cache[relative] = read_jsonl(_inside(path.parent, relative))
            all_raw_keys.update((relative, index) for index in range(len(raw_cache[relative])))
    _need(covered == all_raw_keys, "native snapshot omitted an original observation/failure")
    _need(expected_splits == records, "native admission changed the declared first-success task selection")
    _need(manifest.get("source_counts") == {"observed": observed, "admitted": admitted}, "native per-source observed/admitted counts mismatch")
    return manifest, records


def copy_native_snapshot(manifest_path: str | Path, destination: str | Path) -> Path:
    """Copy an already sealed native snapshot with its entire evidence tree."""
    manifest, _ = verify_native_snapshot(manifest_path, allow_native_teacher=True)
    source = Path(manifest_path).resolve().parent
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    for relative in sorted(manifest["files"]):
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with _inside(source, relative).open("rb") as incoming, output.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        os.chmod(output, 0o444)
    manifest_target = target / "manifest.json"
    with Path(manifest_path).open("rb") as incoming, manifest_target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    os.chmod(manifest_target, 0o444)
    verify_native_snapshot(manifest_target, allow_native_teacher=True)
    return manifest_target


def extract_native_observation(source_id: str, raw: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Source-specific, reviewed, read-only adapters; no guessed field coercions."""
    if source_id == "luna_python":
        return _extract_python_observation(raw, task)
    if source_id != "luna_cli":
        raise DataValidationError(f"native source adapter has not yet been reviewed: {source_id}")
    _need(raw.get("schema") == "picoagent.cli_native_teacher_observed.v1" and raw.get("execution_kind") == "native_teacher_observed", "unexpected original CLI observation schema")
    _need(raw.get("not_a_trace") is True and raw.get("sft_admissible") is False, "original CLI diagnostic provenance must remain unchanged")
    _need(raw.get("model_identity") is None and raw.get("provider_generation") is None, "CLI replay cannot be relabeled as adaptive model generation")
    _need(raw.get("task_sha256") == content_hash(task), "CLI observation does not match frozen task")
    for key in ("task_id", "family", "template_id", "split"):
        _need(raw.get(key) == task[key], "CLI raw/task identity mismatch")
    _need(raw.get("candidate_plan_sha256") == content_hash(task["reference"]["plan"]), "CLI observed plan differs from frozen candidate")
    before = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in task["environment"]["files"].items()}
    _need(raw.get("fixture_sha256_before") == before and raw.get("fixture_sha256_after") == before and raw.get("fixture_unchanged") is True, "CLI task fixtures changed or differ from approved input")
    _need(raw.get("artifact_poststate") == {} and raw.get("kv_poststate") == {}, "CLI read-only source unexpectedly changed post-state")
    metadata = raw["native_execution_metadata"]
    _need(metadata.get("container_id") is None and metadata.get("runtime") is None, "CLI native metadata contains a container claim")
    runtime = {"backend": "native_teacher", "python_version": metadata["python_version"],
               "platform": canonical_json(metadata["platform"]), "locale": metadata["locale"],
               "executables": {"python": {"path": metadata["python_executable"], "sha256": metadata["python_executable_sha256"], "version": metadata["python_version"]},
                               "bash": {"path": metadata["bash_executable"], "sha256": metadata["bash_executable_sha256"], "version": metadata["bash_version"]}},
               "environment_policy": metadata["environment_policy"], "outer_exec_receipt_id": metadata.get("outer_exec_receipt_id"),
               "workspace_lifecycle": metadata["workspace_lifecycle"]}
    messages = copy.deepcopy(raw["transcript"])
    _need(content_hash(messages[0]["content"]) == raw["system_prompt_sha256"], "CLI system prompt hash mismatch")
    _need(messages[1] == {"role": "user", "content": task["prompt"]}, "CLI task prompt differs from actual teacher context")
    events, receipts = [], []
    for index, original in enumerate(raw["tool_events"]):
        arguments = original["arguments_json"]
        _need(json.loads(arguments) == original["arguments"], "CLI decoded tool arguments mismatch")
        plan = task["reference"]["plan"]
        _need(index < len(plan) and original["name"] == plan[index]["name"] and original["arguments"] == plan[index]["arguments"], "CLI executed command differs from independently reviewed frozen action plan")
        invocation = original["invocation"]
        _need(invocation == {"tool_name": original["name"], "tool_call_id": original["tool_call_id"], "arguments_json": arguments, "arguments": original["arguments"]}, "CLI invocation receipt differs from requested action")
        result = original["result"]
        event = {"type": "tool_execution", "tool_call_id": original["tool_call_id"], "name": original["name"], "arguments": arguments,
                 "result": copy.deepcopy(result), "verified": "error" not in result}
        receipt = {"sequence": index, "tool_call_id": event["tool_call_id"], "name": event["name"], "arguments": arguments, "result": copy.deepcopy(result)}
        execution = original["execution"]
        if original["name"] == "search":
            _need(execution["backend"] == "local_fixture_search", "CLI search receipt is not local fixture retrieval")
            from .collector import LocalCorpusSearch
            args = original["arguments"]
            _need(result == LocalCorpusSearch(task["environment"]["docs"]).search(args["query"], args.get("limit", 5)), "CLI fixture search result differs from frozen corpus")
            corpus_hash = content_hash(task["environment"]["docs"])
            receipt.update(execution_kind="host_function", operation=copy.deepcopy(args), state_before_sha256=corpus_hash,
                           state_after_sha256=corpus_hash, state_hash_basis="frozen_fixture_corpus_not_process_streams")
        else:
            _need(execution["backend"] == "platform_exec_command_per_task_temp_workspace", "unexpected CLI native backend")
            stdin = base64.b64decode(execution["stdin_base64"], validate=True).decode("utf-8")
            receipt.update(execution_kind="subprocess", argv=execution["argv"], stdin=stdin, cwd=execution["cwd"],
                           environment=execution["environment"], exit_code=execution["exit_code"], timed_out=execution["timed_out"],
                           truncated=execution["truncated"], duration_seconds=execution["duration_ns"] / 1_000_000_000)
            for stream in ("stdout", "stderr"):
                encoded = execution[stream + "_base64"]
                receipt[stream + "_b64"] = encoded
                captured = base64.b64decode(encoded, validate=True)
                receipt[stream + "_sha256"] = hashlib.sha256(captured).hexdigest()
                _need(result.get(stream) == captured.decode("utf-8", errors="replace"), "CLI decoded reply differs from captured raw stream")
        events.append(event)
        receipts.append(receipt)
    model_events = [{"type": "assistant", **copy.deepcopy(event)} for event in raw["teacher_events"]]
    return {"messages": messages, "effective_messages": copy.deepcopy(messages), "model_events": model_events,
            "tool_events": events, "receipts": receipts, "runtime": runtime, "artifacts": {}, "kv": {},
            "task_sha256": raw["task_sha256"], "source_module_sha256": raw["source_module_sha256"],
            "tool_schemas_sha256": raw["tool_schemas_sha256"], "final": raw["final_response"]}


def _independent_oracle(source_id: str, task: dict[str, Any], final: str,
                        artifacts: dict[str, str], kv: dict[str, Any], *, source_sha256: str, source_path: str | Path) -> dict[str, Any]:
    """Only explicit source-reviewed pure oracles; never a caller import string."""
    _need(source_id in {"luna_cli", "luna_python"}, "independent oracle source has not been reviewed")
    key = (source_id, source_sha256)
    if key not in _ORACLE_MODULE_CACHE:
        _need(file_hash(source_path) == source_sha256, "frozen independent oracle source differs from pinned review")
        # This exact source hash was independently pinned before this call.
        # Only pure checker functions are invoked; no recorded commands run.
        spec = importlib.util.spec_from_file_location("picoagent.data._reviewed_native_oracle_" + source_sha256, source_path)
        _need(spec is not None and spec.loader is not None, "cannot load reviewed frozen oracle source")
        module = importlib.util.module_from_spec(spec)
        previous_bytecode_policy = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode_policy
        _ORACLE_MODULE_CACHE[key] = module
    module = _ORACLE_MODULE_CACHE[key]
    if source_id == "luna_cli":
        return module.independent_oracle_check(task, final)
    return module.independent_oracle(task, final)


def seal_native_snapshot(sources: list[dict[str, Any]], destination: str | Path, *,
                         allow_native_teacher: bool = False, shard_bytes: int | None = None) -> Path:
    """Create a new audited dataset from actual, source-reviewed observations.

    A source specification contains a pinned `review`, observation_paths,
    task_paths, candidate_paths, and code_paths mapping review names to actual
    source files. All reads/copies are local. No tool commands are replayed.
    Unknown source review hashes and unsupported adapters fail closed.
    """
    _need(allow_native_teacher is True, "native sealing requires explicit opt-in")
    if shard_bytes is not None:
        from .native_storage import seal_sharded_snapshot
        return seal_sharded_snapshot(sources, destination, shard_bytes=shard_bytes)
    from .audit import write_new_json
    from .schema import validate_trace
    from picoagent.harness.tools import TOOL_SCHEMAS
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    admitted_counts: dict[str, dict[str, int]] = {}
    observed_counts: dict[str, dict[str, int]] = {}
    all_rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()

    def snapshot_file(source: str | Path, relative: str) -> str:
        output = _inside(target, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open("rb") as incoming, output.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        return relative

    for specification in sources:
        review = copy.deepcopy(specification["review"])
        _validate_review(review)
        source_id = review["source_id"]
        source_tool_schemas = specification.get("tool_schemas", TOOL_SCHEMAS)
        _need(source_id not in observed_counts, "supply one specification per reviewed source")
        prefix = f"evidence/{source_id}"
        source_paths: dict[str, str] = {}
        _need(set(specification["code_paths"]) == set(review["source_sha256"]), "native source file set differs from independent review")
        for name, source in specification["code_paths"].items():
            _need(file_hash(source) == review["source_sha256"][name], "reviewed native source bytes changed before sealing")
            source_paths[name] = snapshot_file(source, f"{prefix}/source/{name}")
        for extra_name, original in specification.get("extra_paths", {}).items():
            _need(safe_relative_path(extra_name), "invalid native auxiliary evidence path")
            snapshot_file(original, f"{prefix}/auxiliary/{extra_name}")
        review_path = f"{prefix}/review.json"
        write_new_json(target / review_path, review)
        indexes: dict[str, dict[str, tuple[dict[str, Any], str, int]]] = {"task": {}, "candidate": {}}
        for kind in ("task", "candidate"):
            for file_index, original in enumerate(specification[kind + "_paths"]):
                relative = snapshot_file(original, f"{prefix}/{kind}s/{file_index:03d}.jsonl")
                for row_index, item in enumerate(read_jsonl(target / relative)):
                    task_id = item["task_id"]
                    _need(task_id not in indexes[kind], f"duplicate original native {kind} identity")
                    indexes[kind][task_id] = (item, relative, row_index)
        counts = {"observed": 0, "oracle_failed": 0, "duplicate_success": 0, "admitted": 0}
        observed_counts[source_id] = counts
        admitted_counts[source_id] = {"train": 0, "dev": 0}
        for file_index, original in enumerate(specification["observation_paths"]):
            _need(file_hash(original) in APPROVED_OBSERVATION_FILES[content_hash(review)], "native observation artifact has not yet been reviewed")
            raw_path = snapshot_file(original, f"{prefix}/observations/{file_index:03d}.jsonl")
            for row_index, raw in enumerate(read_jsonl(target / raw_path)):
                task_id = raw.get("task_id") or raw.get("task", {}).get("task_id")
                _need(task_id in indexes["task"] and task_id in indexes["candidate"], "native observation lacks original task/candidate record")
                task, task_path, task_index = indexes["task"][task_id]
                candidate, candidate_path, candidate_index = indexes["candidate"][task_id]
                _need(task["split"] in rows_by_split, "test/lockbox observations cannot enter a native training source")
                projection = extract_native_observation(source_id, raw, task)
                _need(projection["source_module_sha256"] in review["source_sha256"].values(), "native observation came from unreviewed source version")
                _need(content_hash(source_tool_schemas) == projection["tool_schemas_sha256"], "native source's recorded tool schema is not the current reviewed schema")
                counts["observed"] += 1
                oracle = _independent_oracle(source_id, task, projection["final"], projection["artifacts"], projection["kv"], source_sha256=review["oracle_source_sha256"], source_path=_inside(target, source_paths[next(name for name, digest in review["source_sha256"].items() if digest == review["oracle_source_sha256"])]))
                _need(type(oracle.get("passed")) is bool, "independent native oracle did not return a boolean outcome")
                verification = check_task_result(task, projection["final"], artifacts=projection["artifacts"], kv=projection["kv"])
                _need(oracle["passed"] is verification["passed"], "independent recomputation disagrees with frozen task oracle")
                audit = {"independent_recomputation": True, "passed": oracle["passed"], "details": oracle,
                         "task_sha256": content_hash(task), "oracle_id": review["oracle_id"], "source_sha256": review["oracle_source_sha256"]}
                evidence = {"schema": NATIVE_EVIDENCE_SCHEMA, "source_id": source_id, "source_sha256": review["source_sha256"],
                            "source_paths": source_paths, "review": review, "review_sha256": content_hash(review), "review_path": review_path,
                            "raw_record": raw, "raw_record_sha256": content_hash(raw), "raw_path": raw_path, "raw_record_index": row_index,
                            "task": task, "task_path": task_path, "task_record_index": task_index,
                            "candidate": candidate, "candidate_path": candidate_path, "candidate_record_index": candidate_index,
                            "candidate_sha256": content_hash(candidate), "raw_projection": projection, "raw_projection_sha256": content_hash(projection),
                            "receipts": projection["receipts"], "artifacts": projection["artifacts"], "kv": projection["kv"],
                            "artifacts_sha256": content_hash(projection["artifacts"]), "kv_sha256": content_hash(projection["kv"]), "oracle_audit": audit}
                trace = {"schema_version": task["schema_version"], "trace_id": f"native:{source_id}:{content_hash(raw)}",
                         **{key: task[key] for key in ("task_id", "family", "template_id", "split")},
                         "task_sha256": content_hash(task), "raw_attempt_sha256": content_hash(raw),
                         "status": "success" if oracle["passed"] else "failed", "verification": verification,
                         "provenance": {**task["provenance"], "execution": "native_teacher_observed", "teacher_model": None,
                                        "teacher_decision_mode": TEACHER_MODE, "teacher_mode": TEACHER_MODE, "teacher": source_id + ":procedural_callback", "runtime": projection["runtime"],
                                        "container_semantic_replay": "not_verified", "context_compaction_enabled": any(event.get("type") == "compaction" for event in projection["model_events"]),
                                        "accepted_compactions": sum(event.get("type") == "compaction" and event.get("accepted") is True for event in projection["model_events"]), "source_review_sha256": content_hash(review)},
                         "tools": copy.deepcopy(source_tool_schemas), "native_evidence": evidence,
                         **{key: projection[key] for key in ("messages", "effective_messages", "model_events", "tool_events")}}
                validate_trace(trace, allow_native_teacher=True)
                all_rows.append(trace)
                if not oracle["passed"]:
                    counts["oracle_failed"] += 1
                elif task_id in seen_tasks:
                    counts["duplicate_success"] += 1
                else:
                    seen_tasks.add(task_id)
                    rows_by_split[task["split"]].append(trace)
                    counts["admitted"] += 1
                    admitted_counts[source_id][task["split"]] += 1
    _need(all(rows_by_split.values()), "native dataset requires nonempty train and development splits")
    from picoagent.training.data import _check_disjoint
    _check_disjoint(rows_by_split)
    for name, rows in (("train", rows_by_split["train"]), ("dev", rows_by_split["dev"]), ("all_projected_observations", all_rows)):
        with (target / f"{name}.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
    files = {str(path.relative_to(target)): {"sha256": file_hash(path), "bytes": path.stat().st_size}
             for path in sorted(target.rglob("*")) if path.is_file()}
    manifest = {"schema": NATIVE_MANIFEST_SCHEMA, "admission": "audited_native_teacher_observed_only", "lockbox_used": False,
                "teacher_mode": TEACHER_MODE, "container_semantic_replay": "not_verified", "arbitrary_learner_execution_allowed": False,
                "source_counts": {"observed": observed_counts, "admitted": admitted_counts},
                "splits": {split: {"path": f"{split}.jsonl", "records": len(rows), "families": sorted({row["family"] for row in rows}),
                                   "templates": sorted({row["template_id"] for row in rows})} for split, rows in rows_by_split.items()},
                "files": files, "selection": "first successful observed original variant per task in declared source order",
                "limitations": ["Luna-authored programs replayed deterministically; no adaptive Luna model decisions are claimed.",
                                "Native CPU observations do not establish Docker/Podman semantic parity.",
                                "Source reviews and hashes are integrity records, not cryptographic remote execution attestations."]}
    write_new_json(target / "manifest.json", manifest)
    verify_native_snapshot(target / "manifest.json", allow_native_teacher=True)
    for path in target.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    return target / "manifest.json"


def _extract_python_observation(raw: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    _need(raw.get("schema_version") == "picoagent.native_observation.v2" and raw.get("execution") == "native_teacher_observed", "unsupported original Python observation schema")
    _need(raw.get("sft_admissible") is False, "original Python diagnostic file must remain unadmitted")
    evidence = raw["native_evidence"]
    _need(evidence["task"] == task and raw["task_sha256"] == content_hash(task), "Python observation frozen task mismatch")
    _need(raw["raw_attempt_sha256"] == content_hash(evidence), "Python original evidence hash mismatch")
    _need(evidence["candidate_sha256"] == content_hash(evidence["candidate"]) == raw["candidate_sha256"], "Python original candidate linkage mismatch")
    _need(evidence["candidate"]["execution"] == "unexecuted" and evidence["candidate"]["training_eligible"] is False, "Python candidate must preserve pre-action status")
    fixtures = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in task["environment"]["files"].items()}
    _need(evidence["fixture_sha256"] == fixtures, "Python observed fixtures differ from frozen task")
    metadata = raw["provenance"]["runtime"]
    _need(metadata.get("container_id") is None and metadata.get("image") is None, "Python native source claims a container identity")
    runtime = {"backend": "native_teacher", "python_version": metadata["python_version"], "platform": metadata["platform"],
               "locale": canonical_json(metadata["locale"]),
               "executables": {"python": {"path": metadata["python_executable"], "sha256": metadata["python_executable_sha256"], "version": metadata["python_version"]},
                               "bash": {"path": metadata["bash_executable"], "sha256": metadata["bash_executable_sha256"], "version": None}},
               "environment_policy": evidence["environment_policy"], "executable_version_note": "Bash byte hash recorded; version string not recorded"}
    events = [{"type": "tool_execution", **copy.deepcopy(event)} for event in evidence["tool_events"]]
    receipts = []
    _need(len(events) == len(evidence["receipts"]), "Python raw receipt/event count mismatch")
    for original, event in zip(evidence["receipts"], events):
        receipt = copy.deepcopy(original)
        receipt["execution_kind"] = "subprocess"
        receipt["result"] = copy.deepcopy(event["result"])
        for stream in ("stdout", "stderr"):
            receipt[stream + "_b64"] = original[stream + "_bytes_b64"]
            _need(_bytes(receipt, stream).decode("utf-8", errors="replace") == event["result"][stream] == original[stream], "Python raw stream differs from observed reply")
        _need(hashlib.sha256(receipt["stdin"].encode()).hexdigest() == receipt["stdin_sha256"], "Python stdin hash mismatch")
        receipts.append(receipt)
    models = []
    for event in evidence["callback_events"]:
        _need(event["type"] == "deterministic_callback", "Python source cannot claim sampled model decisions")
        models.append({**copy.deepcopy(event), "type": "assistant", "decision_origin": "deterministic_callback"})
    artifacts = {}
    for name, artifact in evidence["artifacts"].items():
        _need(hashlib.sha256(artifact["text"].encode()).hexdigest() == artifact["sha256"], "Python post-artifact byte hash mismatch")
        artifacts[name] = artifact["text"]
    tool_source_hash = evidence["source_sha256"]["src/picoagent/harness/tools.py"]
    _need(tool_source_hash in SOURCE_TOOL_PROTOCOLS, "Python recorded tool schema source is not reviewed")
    return {"messages": copy.deepcopy(raw["messages"]), "effective_messages": copy.deepcopy(raw["effective_messages"]),
            "model_events": models, "tool_events": events, "receipts": receipts, "runtime": runtime,
            "artifacts": artifacts, "kv": evidence["kv"], "task_sha256": raw["task_sha256"],
            "source_module_sha256": evidence["source_sha256"]["src/picoagent/data/luna_python_curriculum.py"],
            "tool_schemas_sha256": SOURCE_TOOL_PROTOCOLS[tool_source_hash], "final": raw["messages"][-1].get("content") or ""}
