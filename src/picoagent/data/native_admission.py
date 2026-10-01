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
import inspect
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
COMPACTION_SOURCES = frozenset({"native_compaction", "native_compaction_retention"})
SOURCE_KINDS = frozenset({"luna_cli", "luna_python", "luna_recovery", "native_knowledge_search"}) | COMPACTION_SOURCES
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
    "native_knowledge_search": frozenset({"37c1f9bdc12137c323ca4eb28d9202baee903839efd8b63fe40dbe116a96222d"}),
    "native_compaction_retention": frozenset({"38e7c4ad2005d5d7891df2062ada32f5a5b3ce76998b4313bce251b1e7608540"}),
    "luna_python": frozenset({
        "8778addde92a5549d21b6b9428d5ac2e7529017e4dc64551bd087211e98cc97f",
        "713251b07d38814cd4a9a0afee7a53ea7206fb6b07e85ba8a20c3a3b64c03b37",
    }),
    "native_compaction": frozenset({"00e37e04fd0b82f94638e988632bbaea66dccfce95954a27c8add4d979da2d46", "1f0ba16b8182e54823a5132dc76c499c5f0900d6d9e99a5b8b2654d7653b41d7"}),
    "luna_recovery": frozenset({"13016ed73578cccfc95a5d6a1069e5a44ae05d7be378ac83383895f7e3add448"}),
    "luna_cli": frozenset({
        "35e1df0499fa3411fa4607495dc1f127172999da3b8fa8412ab81692602be0cd",
        "42c8252510b67dfa3b735b4fdab9f63b3b9ee3cd63e4895ccd3a28fd41919127",
    }),
}


# Actual observation files reviewed alongside the source. A source approval is
# not permission to invent a new native record. Large batches should be frozen
# into a small number of aggregate JSONL files before this review step.
APPROVED_OBSERVATION_FILES: dict[str, frozenset[str]] = {
    "37c1f9bdc12137c323ca4eb28d9202baee903839efd8b63fe40dbe116a96222d": frozenset({"74ee28219f5e81061c9c9dbe9f4fca0dd6ef355ab138cccc5a9ed497a19dc864", "84ba11b31c967df014cf9a1ebe6aa6d61f22226ce8032e61d1f6b185651e1069"}),
    '38e7c4ad2005d5d7891df2062ada32f5a5b3ce76998b4313bce251b1e7608540': frozenset({'c72690fd24a7550efc9f51db9a9e9c28987cf1055e45aaee82e51504397dd758'}),
    '1f0ba16b8182e54823a5132dc76c499c5f0900d6d9e99a5b8b2654d7653b41d7': frozenset({
        '9fc06e9c5796b8e53233dc11bf59739320fc25856f243d40e096abb2d7b5036a',
        '9ec3fa5129467126570a0e695c2fb35b606e5dde1579df0e611d18abc7ab45d3',
        'eca0306ceb02f464e96da38cfb11f28616fa6a424fd5e486e5dd55a00d3721f8',
        '91b21eab761d7312819e415d7838e67cda49cc3dae31b9add6767b14ecca9050',
        '51c4eea824cba3e6ea28df5286fa14a4f572c9d8de7fd989628b6d5d6629bfb7',
        '371f81044806c485f65adb010a32f03817d07b31128c33f0b844c529066fd6c6',
        '9993e9863f3087ca564c3f582e682a2f0608885c190bdbc3ee59c8e982ef6dd3',
        'fba48027ecd756c32e8efcf561661cff80124f51b628501628639274da0a6c00',
        'a33742e9011e905e56ab4d49fddf6343b5d318857cdc461b23c72ba54234a4f1',
        '253eebab3d0a870f8e0bfaaaccafe65b9a90c78e7d271947a61633bb1794dde9',
        '95dd77f3eb9074f27596e5c27deb17574eeb173e6dd1a70af869f08da1ae999b',
        '8fe34b2d7bdd4b71dff5c3df83fb1bf6d4d5d8eee73aebcb7eabc88e2777d1b5',
        'c92c7339116ae98c54cb046bf42f70548b83ffb8868678797828497e6830d73b',
        'a842c5017b579c75dbb3687501da11b91ff5eb5f33fad63a800f0f39034dc2fa',
        'e0ecdceb9b2e5d4dc7a99ba553d53a64928e86fa866760bcc0152d58e2da3384',
        '85f069dea11fb9e4f47f0dfc80b48e2cd979d5579f131076cb6a7fd4f8a9a92e',
        '1566413799689d44f3277d4d8c68493ea777010eba03664d668d4500bc2946ba',
        'ab50a91699cda6d83022eee0237bd7d42611a956b18d73b4dc8816ce97021764',
        '63191c2f3d04d2552219ae6f8caf3adebbdc530df2b1b8bf31bf14bf0b123bf8',
        '0e1e5ce96e8b63e51634902dfdb3f8ddc46c16e1ba46a71190e70bb4fdc9bf1b',
        '757c7a0f3b7b384e8faacaeac08ccc57b264b79a44cc568bf78bd40ce1a0f967',
        '7f104756de27ec145fbc3cc60e3f5ee35199728c1da22f3e77bf9bf3155052d5',
        '903c6d1c23138ea709bf673cfd3de12e02be7c1a27962bd59a7cca37d027ae41',
        '718101018219384382a01ea12adadcf9f1dcfd8112bca84a0a32241dc9ebe427',
        'ff4221a14c38557716815c3db0e95dc82045c7bdee41f52ac9b2ff230840f735',
        '3fc0f258599187e61d732f00b6d98506d716368804008a3fd8fd5e5b42032fc6',
        'ac832c4fe81efcbc4c156067de2a3877bfafa0168440c62cbdec59594af27098',
        '66500d81a720e1cb637be090d95236e54c2c2e0fab6d426be83381c599962ec8',
        'fcc5a4dd506c4da3ee225e6241252a7a71a6653d839c7ccd9cba61f9e358b29c',
        '7dad75679925053edc89353ae0006c2b8014d0f8ea8db574492dc1894ef4f3c6',
        'd00df4105643515d0d6afc6c0bd3380fd3e7dfb0d351bfc6e50170eda64b734b',
        '3c02543eaca0cfd6ae3cd03f765526e366b363c5a3991f4179e1e333731a9762',
        '9bac880013a89b3862ff3bd94068be6e63d8785e9978119652de043ab638a95e',
        '31724dfc63033ce12dfef5cf338ac3453d1de54bc59aad680ec63dfe89366510',
        '4b94a1c46628868d9a2f584c86d90e7cd4b21ca5be33be3cdb80cc858bb8348e',
        'e2fefe0b6f2220e88438254732e1e149170cc9f39a22ddaaa83054b2523b8d4e',
        '7fd5d33405116aeb38da7c8bc65e160222c3cfa139b76209539755a75992fddb',
        '7db46b90f96c2e5b28b89564135249856c60961afc88638546096d156c78fed8',
        '0aa3e9e90cc48368009468546215b24e143648b8473601ee729a37e2be7bd405',
        '86e4f324964e9b0068834d89a5b9bfc5082aafa64bf90311a2994d20c634ae30',
        '2c07090f5b2ad93b2e2ad51750e469938be71e9c9a18cea369df65acb0f5b400',
        '0ecf2da7ee4bb37f8dd05f6e8eb14da7ed964109645c0355a271826f74e4a4aa'
    }),
    "13016ed73578cccfc95a5d6a1069e5a44ae05d7be378ac83383895f7e3add448": frozenset({"20694652969c0c3b876421ed82f2823da2bc76d13f62d24d63424cdd5b75e43b"}),
    "713251b07d38814cd4a9a0afee7a53ea7206fb6b07e85ba8a20c3a3b64c03b37": frozenset({"00a68626882b3ec0a0c83526f16ec5a8523315299d80685b4f4ca6da5fc67ec8",
        'dcd448e748445252a32d57a61cd3ccd500d3e8f4b5825ed9749373b6f17293f9',
        'bb4bdbe8ff983558a2a6a345591df23a69fa74b6765ce61ebd01d802aca62ff9',
        '1165fb76684b301fb395ff88503d2066b9f12aa8217fe3ef81eac15cf17c6507',
        '57a4b95f22cea118436ca77c8fccf01f5735b4e02bd345fcb83615b70a0719ad',
        '264d1860fa1e147a6edf879c502532929c7ceee2809194cf8028dc6af9c4a6c2',
        '01604d434e8b36fc1676b56c4c03d65a66b8b2cf5380c2e21bf622483201764d',
        '4f0e82fe185ffc903a04ad5319d928ec52496f842a7cd345581cab5a4025c922',
        '1cae8e12d84af8a08cbee43246fe0176d3df7af875c45617dec72e3fea2ccc3d',
        '45c66139989f120c719757ffd08e18ea6d4dc50d1af0d78349b119558cf1fbe2',
        '19ca191eba9bc5501135d223873ce70048d540db26e9d18ce26791cbf0b5ed35',
        '9163a6e2a069a8ab2a0672fb0e16ec3023767f0012a2676671dbf9f4c81cb1a0',
        'ce4e5b9ad55e5691a9ef31c45164311b2c090029d8327192d6cce3e266ef1851',
        '4e59c662152087d249f381d954717b3adc913e17a5a8960c3cd38dfb58989a47'}),
    "00e37e04fd0b82f94638e988632bbaea66dccfce95954a27c8add4d979da2d46": frozenset({"2b50dcdd507f5c748249563e9058f65e8c4c6e04b2ea6de8b8b113417f69c08c"}),
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


def native_problem_identity(trace: dict[str, Any]) -> str:
    """Use only the origin link already checked against a pinned source review."""
    return trace.get("provenance", {}).get("origin_base_task_id", trace["task_id"])


def selected_compaction_modes(records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    return {split: {mode: sum(row.get("provenance", {}).get("context_mode") == mode for row in rows)
                    for mode in ("full", "half", "manual")} for split, rows in records.items()}


def _select_native_success(trace: dict[str, Any], problems: dict[str, str],
                           conversations: dict[str, str], *, deduplicate: bool) -> bool:
    """Select a validated successful view; never resolve cross-split leakage."""
    identity, split = native_problem_identity(trace), trace["split"]
    _need(identity not in problems or problems[identity] == split, "cross-split native canonical problem overlap")
    fingerprint = None
    if deduplicate:
        from picoagent.training.data import conversation_fingerprint
        fingerprint = conversation_fingerprint(trace["messages"])
        _need(fingerprint not in conversations or conversations[fingerprint] == split, "cross-split native conversation overlap")
    if identity in problems or (fingerprint is not None and fingerprint in conversations):
        return False
    problems[identity] = split
    if fingerprint is not None:
        conversations[fingerprint] = split
    return True


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
    origins = review.get("origin_base_task_ids", {})
    origin_hashes = review.get("origin_base_task_sha256", {})
    if origins or origin_hashes:
        _need(review["source_id"] == "native_compaction_retention" and set(origins) == set(origin_hashes), "native origin mapping requires its distinct reviewed retention source")
        _need(all(isinstance(key, str) and isinstance(value, str) and key and value and _digest(origin_hashes[key]) for key, value in origins.items()), "native origin mapping must hash-bind every base task")
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
    origin = review.get("origin_base_task_ids", {}).get(task["task_id"], task["task_id"])
    _need(native_problem_identity(trace) == origin, "native canonical problem identity differs from reviewed origin link")
    _need(provenance.get("origin_base_task_sha256") == review.get("origin_base_task_sha256", {}).get(task["task_id"]), "native origin task hash differs from reviewed derivation")
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
    if evidence["source_id"] in {"luna_python", "native_knowledge_search"}:
        _need(candidate == raw["native_evidence"]["candidate"], "native Python candidate differs from executed source linkage")
    elif evidence["source_id"] in COMPACTION_SOURCES:
        _need(candidate == raw["candidate"], "native compaction candidate differs from observed base plan")
    elif evidence["source_id"] == "luna_recovery":
        _need(candidate == {"task_id": task["task_id"], **raw["candidate"]}, "native recovery candidate differs from mechanically indexed original")
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
        elif kind == "host_function" and event["name"] == "write_file":
            _need(evidence["source_id"] == "luna_recovery", "native host file writes require their source-specific review")
            arguments = json.loads(event["arguments"])
            _need(receipt.get("operation") == arguments, "native host write operation differs from arguments")
            data = arguments["content"].encode("utf-8")
            effect = receipt.get("host_effect", {})
            _need(safe_relative_path(arguments["path"]) and arguments["path"].startswith(("scripts/", "output/")), "native host write escapes its allowed task paths")
            _need(effect == {"path": arguments["path"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "readback_verified": True}, "native file write lacks exact observed readback")
            _need(result.get("path") == arguments["path"] and result.get("bytes") == len(data), "native file write reply differs from observed bytes")
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
    if "context_mode" in provenance:
        _need(provenance["context_mode"] == projection.get("mode"), "native selected mode differs from observed mode")
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
    if manifest.get("storage") == "sealed_native_collection_v1":
        from .native_collection import verify_native_collection
        return verify_native_collection(path, manifest)
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
            verify_native_context_tokens(trace, path.parent)
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
    seen_task_ids: dict[str, str] = {}
    seen_conversations: dict[str, str] = {}
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
        verify_native_context_tokens(row, path.parent)
        repeated = _independent_oracle(source_id, evidence["task"], row["messages"][-1].get("content") or "", evidence["artifacts"], evidence["kv"], source_sha256=evidence["review"]["oracle_source_sha256"], source_path=_inside(path.parent, evidence["source_paths"][next(name for name, digest in evidence["source_sha256"].items() if digest == evidence["review"]["oracle_source_sha256"])]))
        _need(repeated == evidence["oracle_audit"]["details"] and repeated.get("passed") is row["verification"]["passed"], "native failure/retry oracle outcome changed")
        _need(row["split"] in expected_splits, "test/lockbox record present in native training observations")
        if not repeated["passed"]:
            observed[source_id]["oracle_failed"] += 1
        elif not _select_native_success(row, seen_task_ids, seen_conversations, deduplicate=manifest.get("conversation_deduplication") == "first_within_split"):
            observed[source_id]["duplicate_success"] += 1
        else:
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
    if "selected_compaction_modes" in manifest:
        _need(manifest["selected_compaction_modes"] == selected_compaction_modes(records), "native selected mode counts mismatch")
    return manifest, records


def copy_native_snapshot(manifest_path: str | Path, destination: str | Path) -> Path:
    """Copy an already sealed native snapshot with its entire evidence tree."""
    expected_hash = file_hash(manifest_path)
    manifest, _ = verify_native_snapshot(manifest_path, allow_native_teacher=True)
    manifest_target = _copy_verified_native_snapshot(manifest_path, destination, verified_manifest=manifest,
                                                      expected_manifest_sha256=expected_hash)
    verify_native_snapshot(manifest_target, allow_native_teacher=True)
    return manifest_target


def _copy_verified_native_snapshot(manifest_path: str | Path, destination: str | Path, *,
                                   verified_manifest: dict[str, Any], expected_manifest_sha256: str) -> Path:
    """Internal copy immediately after strict verification in the same process.

    The caller must retain the manifest returned by verify_native_snapshot (or
    verify_dataset) and its expected byte hash. This is not an admission API or
    an unverified JSONL loader. Every source/destination byte is checked, but
    already completed semantic, tokenizer, and oracle checks are not repeated.
    Standalone copy_native_snapshot keeps strict verification before and after.
    """
    _need(_digest(expected_manifest_sha256), "verified copy needs expected manifest byte hash")
    manifest_bytes = Path(manifest_path).read_bytes()
    _need(hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha256,
          "verified copy manifest changed since validation")
    _need(json.loads(manifest_bytes) == verified_manifest and verified_manifest.get("schema") == NATIVE_MANIFEST_SCHEMA,
          "verified copy manifest does not match the prior validated object")
    files = verified_manifest.get("files", {})
    _need(isinstance(files, dict) and bool(files) and "manifest.json" not in files, "verified copy requires an external evidence inventory")
    source = Path(manifest_path).resolve().parent
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    for relative in sorted(files):
        expected = files[relative]
        _need(type(expected.get("bytes")) is int and expected["bytes"] >= 0 and _digest(expected.get("sha256")), "verified copy has invalid file metadata")
        output = _inside(target, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        with _inside(source, relative).open("rb") as incoming, output.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                digest.update(block)
                size += len(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        _need(size == expected["bytes"] and digest.hexdigest() == expected["sha256"], f"verified copy source integrity mismatch: {relative}")
        _need(output.stat().st_size == size and file_hash(output) == expected["sha256"], f"verified copy destination integrity mismatch: {relative}")
        os.chmod(output, 0o444)
    manifest_target = target / "manifest.json"
    with manifest_target.open("xb") as outgoing:
        outgoing.write(manifest_bytes)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    _need(file_hash(manifest_target) == expected_manifest_sha256 and file_hash(manifest_path) == expected_manifest_sha256,
          "verified copy manifest byte integrity mismatch")
    os.chmod(manifest_target, 0o444)
    return manifest_target


def combine_native_snapshots(manifest_paths: list[str | Path], destination: str | Path, *,
                             allow_native_teacher: bool = False, shard_bytes: int = 16 * 1024 * 1024) -> Path:
    """Verify and reseal disjoint source snapshots with all their raw evidence.

    Sources cannot be repeated (including two versions of the same source).
    Original snapshots are never changed. The combined first-success selection
    and train/dev family disjointness are independently verified by the sealer.
    """
    _need(allow_native_teacher is True, "native combination requires explicit opt-in")
    from .native_storage import iter_rows
    sources, seen = [], set()
    for original in manifest_paths:
        manifest_path = Path(original).resolve()
        manifest, records = verify_native_snapshot(manifest_path, allow_native_teacher=True)
        del records
        root = manifest_path.parent
        expected = set(manifest["source_counts"]["observed"])
        _need(not expected.intersection(seen), "cannot combine repeated native sources or versions")
        seen.update(expected)
        paths = manifest.get("all_observations", {}).get("paths", ["all_projected_observations.jsonl"])
        examples = {}
        for relative in paths:
            for trace in iter_rows(root / relative):
                evidence = trace["native_evidence"]
                examples.setdefault(evidence["source_id"], (evidence, trace["tools"]))
                if set(examples) == expected:
                    break
            if set(examples) == expected:
                break
        _need(set(examples) == expected, "native source lacks preserved raw observations")
        for source_id, (evidence, source_tools) in examples.items():
            prefix = f"evidence/{source_id}/"
            specification = {"review": evidence["review"],
                             "code_paths": {name: root / path for name, path in evidence["source_paths"].items()},
                             "extra_paths": {path.removeprefix(prefix + "auxiliary/"): root / path
                                             for path in manifest["files"] if path.startswith(prefix + "auxiliary/")}}
            specification["extra_paths"][f"prior_sealed_dataset_manifest.{file_hash(manifest_path)}.json"] = manifest_path
            for kind, folder in (("task", "tasks"), ("candidate", "candidates"), ("observation", "observations")):
                specification[kind + "_paths"] = [root / path for path in sorted(manifest["files"])
                                                  if path.startswith(prefix + folder + "/")]
            # Preserve the exact recorded protocol even if installed schemas change.
            specification["tool_schemas"] = source_tools
            sources.append(specification)
    return seal_native_snapshot(sources, destination, allow_native_teacher=True, shard_bytes=shard_bytes)


def extract_native_observation(source_id: str, raw: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Source-specific, reviewed, read-only adapters; no guessed field coercions."""
    if source_id in COMPACTION_SOURCES:
        return _extract_compaction_observation(raw, task, source_id=source_id)
    if source_id == "luna_python":
        return _extract_python_observation(raw, task)
    if source_id == "luna_recovery":
        return _extract_recovery_observation(raw, task)
    if source_id == "native_knowledge_search":
        return _extract_knowledge_search_observation(raw, task)
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
    _need(source_id in SOURCE_KINDS, "independent oracle source has not been reviewed")
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
    if source_id == "luna_cli" or source_id in COMPACTION_SOURCES:
        return module.independent_oracle_check(task, final)
    if source_id == "luna_recovery":
        # The reviewed recovery checker consumes fixtures and the visible prompt;
        # remove expected-answer/reference fields as an executable leak check.
        inputs = copy.deepcopy(task)
        inputs.pop("oracle", None)
        inputs.pop("reference", None)
        expected, expected_artifacts, expected_kv = module.independent_recovery_oracle(inputs)
        return {"passed": final.strip() == expected and artifacts == expected_artifacts and kv == expected_kv,
                "expected_final": expected, "expected_artifacts_sha256": content_hash(expected_artifacts),
                "expected_kv_sha256": content_hash(expected_kv)}
    if source_id == "native_knowledge_search":
        inputs = copy.deepcopy(task)
        inputs.pop("oracle", None)
        inputs.pop("reference", None)
        _need(artifacts == {}, "in-process knowledge/search cannot claim subprocess artifacts")
        return module.independent_knowledge_search_oracle(inputs, final, kv)
    parameters = inspect.signature(module.independent_oracle).parameters
    if "artifacts" in parameters:
        return module.independent_oracle(task, final, artifacts=artifacts, kv=kv)
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
    seen_tasks: dict[str, str] = {}
    seen_conversations: dict[str, str] = {}

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
                if task_id in review.get("origin_base_task_ids", {}):
                    trace["provenance"]["origin_base_task_id"] = review["origin_base_task_ids"][task_id]
                    trace["provenance"]["origin_base_task_sha256"] = review["origin_base_task_sha256"][task_id]
                if source_id in COMPACTION_SOURCES:
                    trace["provenance"]["context_mode"] = projection["mode"]
                validate_trace(trace, allow_native_teacher=True)
                all_rows.append(trace)
                if not oracle["passed"]:
                    counts["oracle_failed"] += 1
                elif not _select_native_success(trace, seen_tasks, seen_conversations, deduplicate=True):
                    counts["duplicate_success"] += 1
                else:
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
                "selected_compaction_modes": selected_compaction_modes(rows_by_split),
                "conversation_deduplication": "first_within_split",
                "splits": {split: {"path": f"{split}.jsonl", "records": len(rows), "families": sorted({row["family"] for row in rows}),
                                   "templates": sorted({row["template_id"] for row in rows})} for split, rows in rows_by_split.items()},
                "files": files, "selection": "first successful observed variant per reviewed canonical problem identity in declared source order",
                "limitations": ["Reviewed procedural teacher replay; no sampled model decisions are claimed.",
                                "Native CPU observations do not establish Docker/Podman semantic parity.",
                                "Source reviews and hashes are integrity records, not cryptographic remote execution attestations."]}
    write_new_json(target / "manifest.json", manifest)
    verify_native_snapshot(target / "manifest.json", allow_native_teacher=True)
    for path in target.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    return target / "manifest.json"


def _extract_python_observation(raw: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    _need(raw.get("schema_version") in {"picoagent.native_observation.v2", "picoagent.native_observation.v3"} and raw.get("execution") == "native_teacher_observed", "unsupported original Python observation schema")
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
        if raw["schema_version"] == "picoagent.native_observation.v3":
            captured = evidence["artifact_bytes"][name]
            data = base64.b64decode(captured["bytes_b64"], validate=True)
            _need(hashlib.sha256(data).hexdigest() == captured["sha256"] == artifact["sha256"] and len(data) == captured["size_bytes"] and data.decode("utf-8", errors="replace") == artifact["text"], "Python raw artifact capture differs from post-state")
    if "tools" in evidence:
        _need(content_hash(evidence["tools"]) == evidence["tool_schemas_sha256"], "Python callback tool schema hash mismatch")
    tool_source_hash = evidence["source_sha256"]["src/picoagent/harness/tools.py"]
    _need(tool_source_hash in SOURCE_TOOL_PROTOCOLS, "Python recorded tool schema source is not reviewed")
    return {"messages": copy.deepcopy(raw["messages"]), "effective_messages": copy.deepcopy(raw["effective_messages"]),
            "model_events": models, "tool_events": events, "receipts": receipts, "runtime": runtime,
            "artifacts": artifacts, "kv": evidence["kv"], "task_sha256": raw["task_sha256"],
            "source_module_sha256": evidence["source_sha256"]["src/picoagent/data/luna_python_curriculum.py"],
            "tool_schemas_sha256": SOURCE_TOOL_PROTOCOLS[tool_source_hash], "final": raw["messages"][-1].get("content") or ""}


def _extract_compaction_observation(raw: dict[str, Any], task: dict[str, Any], *, source_id: str = "native_compaction") -> dict[str, Any]:
    _need(raw.get("schema") == "picoagent.native_observation.v1" and raw.get("source_id") == source_id and source_id in COMPACTION_SOURCES, "unknown native compaction schema")
    _need(raw.get("sft_admissible") is False and raw.get("teacher_model") is None and raw.get("model_identity") is None and raw.get("provider_generation") is None, "native compaction cannot claim model sampling")
    _need(raw.get("teacher_decision_mode") == TEACHER_MODE, "native compaction must disclose procedural replay")
    _need(raw.get("task") == task and raw.get("task_id") == raw.get("base_task_id") == task["task_id"] and raw.get("task_sha256") == content_hash(task), "compaction mode variants must share their exact base task")
    _need(raw.get("candidate_sha256") == content_hash(raw["candidate"]), "compaction candidate linkage mismatch")
    expected = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in task["environment"]["files"].items()}
    _need(raw.get("fixture_sha256_before") == expected == raw.get("fixture_sha256_after") and raw.get("fixture_unchanged") is True, "native compaction altered its read-only fixtures")
    _need(raw.get("mode") in {"full", "half", "manual"}, "invalid compaction mode variant")
    _need(raw.get("artifacts") == {} and raw.get("kv") == {}, "read-only compaction unexpectedly produced state")
    _need(content_hash(raw["tools"]) == raw["tool_schemas_sha256"], "native compaction tool schema mismatch")
    goal = json.loads(task["prompt"].split("\nPICO-COMPACTION-GOAL\n", 1)[1])
    path = goal["initial_path"]
    for index, receipt in enumerate(raw["receipts"]):
        _need(receipt["sequence"] == index and receipt["name"] == "bash" and json.loads(receipt["arguments"]) == {"command": "cat -- " + str(path)}, "compaction read did not follow the observed next path")
        stdout = _bytes(receipt, "stdout")
        if receipt["exit_code"] == 0 and not receipt["timed_out"]:
            _need(path in task["environment"]["files"] and stdout == task["environment"]["files"][path].encode(), "compaction read bytes differ from frozen fixture")
            packet = json.loads(stdout)
            _need(packet["index"] == index and packet["task_id"] == task["task_id"], "compaction observation chain order mismatch")
            path = packet["next"]
    if raw.get("status") == "observed_success":
        _need(path is None and not raw.get("validation_errors"), "successful compaction did not complete the observed chain")
    events = raw["all_harness_events"]
    models = [copy.deepcopy(event) for event in events if event["type"] in {"assistant", "compaction"}]
    _need(raw["model_events"] == [event for event in events if event["type"] in {"assistant", "compaction", "compaction_error"}], "native compaction model events differ from raw harness events")
    _need(raw["tool_events"] == [event for event in events if event["type"] == "tool_execution"], "native compaction tool events differ from raw harness events")
    budget = raw["context_budget"]
    trigger = budget["max_tokens"] - budget["reserve_tokens"] - budget["compaction_headroom_tokens"]
    _need(0 < trigger <= budget["max_tokens"] - budget["reserve_tokens"] and trigger == budget["trigger_budget"], "compaction trigger budget is not reproducible")
    for event in models:
        if event["type"] == "compaction":
            _need(event.get("mode", "half") == raw["mode"] and event.get("trigger_budget") == trigger == event.get("retained_context_budget"), "compaction event budget or mode differs from actual run configuration")
    if source_id == "native_compaction_retention":
        _need(raw["mode"] == "manual", "retention extension must preserve actual manual mode")
        coverage = {"nonempty": 0, "empty": 0, "retained_and_discarded": 0}
        for event in models:
            if event["type"] != "compaction":
                continue
            groups = json.loads(event["summary_request"][1]["content"])["groups"]
            keep = event["keep_group_indices"]
            coverage["nonempty" if keep else "empty"] += 1
            coverage["retained_and_discarded"] += bool(keep) and len(keep) < len(groups)
            _need(event["tokens_before"] > trigger, "manual retention was not genuinely budget-triggered")
        _need(coverage == raw.get("retention_coverage"), "retention coverage does not match actual decisions")
        if raw.get("status") == "observed_success":
            _need(coverage["retained_and_discarded"] > 0, "retention extension lacks actual keep-and-drop decisions")
    return {"messages": copy.deepcopy(raw["messages"]), "effective_messages": copy.deepcopy(raw["effective_messages"]),
            "model_events": models, "tool_events": copy.deepcopy(raw["tool_events"]), "receipts": copy.deepcopy(raw["receipts"]),
            "runtime": copy.deepcopy(raw["runtime"]), "artifacts": {}, "kv": {}, "task_sha256": raw["task_sha256"],
            "source_module_sha256": raw["source_module_sha256"], "tool_schemas_sha256": raw["tool_schemas_sha256"], "final": raw["final"],
            "mode": raw["mode"], "context_budget": budget, "tokenizer": raw["tokenizer"]}


_CONTEXT_TOKENIZERS: dict[str, Any] = {}
_CONTEXT_TOKEN_CHECKED: set[str] = set()


def verify_native_context_tokens(trace: dict[str, Any], root: Path) -> None:
    """Recompute actual compaction budgets with the frozen public tokenizer."""
    evidence = trace["native_evidence"]
    if evidence["source_id"] not in COMPACTION_SOURCES:
        return
    raw = evidence["raw_record"]
    identity = evidence["raw_record_sha256"]
    if identity in _CONTEXT_TOKEN_CHECKED:
        return
    tokenizer_info = raw["tokenizer"]
    cache_key = content_hash(tokenizer_info)
    for name, info in tokenizer_info["files"].items():
        _need(evidence["source_sha256"].get("tokenizer/" + name) == info["sha256"], "compaction tokenizer asset differs from frozen review")
    if cache_key not in _CONTEXT_TOKENIZERS:
        from transformers import AutoTokenizer
        tokenizer_file = evidence["source_paths"]["tokenizer/tokenizer.json"]
        tokenizer_root = _inside(root, tokenizer_file).parent
        _CONTEXT_TOKENIZERS[cache_key] = AutoTokenizer.from_pretrained(str(tokenizer_root), local_files_only=True, trust_remote_code=False)
    tokenizer = _CONTEXT_TOKENIZERS[cache_key]
    from picoagent.harness.protocol import render_messages
    def count(messages, tools):
        return len(tokenizer.encode(render_messages(messages, tools, add_generation_prompt=True), add_special_tokens=False))
    budget = raw["context_budget"]
    input_budget = budget["max_tokens"] - budget["reserve_tokens"]
    trigger = input_budget - budget["compaction_headroom_tokens"]
    for event in trace["model_events"]:
        if event["type"] == "compaction":
            _need(event["trigger_budget"] == trigger == event["retained_context_budget"], "compaction trigger differs from real configuration")
            _need(event["tokens_before"] == count(event["before_messages"], raw["tools"]) and event["tokens_after"] == count(event["result_messages"], raw["tools"]), "compaction recorded token count does not reproduce")
            _need(event["tokens_before"] > trigger, "native compaction was not triggered by its actual budget")
            _need(count(event["summary_request"], []) <= input_budget, "compactor request exceeds actual reserved context")
            _need(count(event["summary_request"] + [event["summary_response"]], []) <= budget["max_tokens"], "complete compaction example exceeds actual context")
            if evidence["source_id"] == "native_compaction_retention":
                from picoagent.harness.protocol import END_MESSAGE
                target = canonical_json(event["summary_response"]) + END_MESSAGE
                _need(len(tokenizer.encode(target, add_special_tokens=False)) <= budget["reserve_tokens"], "retention decision exceeds actual generation reserve")
        else:
            _need(count(event["input_messages"], raw["tools"]) <= input_budget, "actual action context exceeds reserved tokenizer budget")
    _CONTEXT_TOKEN_CHECKED.add(identity)


def _extract_recovery_observation(raw: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Adapt only the byte-complete, full-context reviewed recovery pilot."""
    _need(raw.get('schema') == 'picoagent.native_observation.v1' and raw.get('execution') == 'native_teacher_observed', 'unknown recovery observation schema')
    _need(raw.get('task') == task and raw.get('sft_admissible') is False, 'recovery task/provenance binding mismatch')
    _need(raw.get('teacher') == {'identity': 'RecoveryCandidateTeacher', 'mode': 'procedural_candidate_callback_deterministic_replay', 'model': 'not_sampled'}, 'recovery teacher cannot claim model sampling')
    _need(task['family'] in {'luna_recovery.file_missing_csv', 'luna_recovery.cli_cut_bad_option'} and task['seed'] == 0, 'recovery admission is restricted to the actually reviewed two-family pilot')
    candidate = raw['candidate']
    _need(candidate['status'] == 'unexecuted' and candidate['training_eligible'] is False and candidate['plan'] == task['reference']['plan'] and candidate['plan_sha256'] == content_hash(candidate['plan']), 'recovery original plan mismatch')
    _need(raw['messages'][:2] == [{'role': 'system', 'content': raw['system_prompt']}, {'role': 'user', 'content': task['prompt']}], 'recovery full callback prompt was not preserved')
    _need(hashlib.sha256(raw['system_prompt'].encode()).hexdigest() == raw['system_prompt_sha256'], 'recovery system prompt hash mismatch')
    _need(content_hash(raw['tool_schemas']) == raw['tool_schemas_sha256'], 'recovery tool schema hash mismatch')
    for event in raw['model_events']:
        _need(event['type'] == 'assistant' and event['tool_schemas'] == raw['tool_schemas'], 'recovery per-action schema/context not preserved')
    runtime = copy.deepcopy(raw['runtime'])
    _need(runtime.pop('backend') == 'native_teacher_observed' and 'container_id' not in runtime and 'image' not in runtime, 'recovery native runtime identity mismatch')
    runtime['backend'] = 'native_teacher'
    for executable in runtime['executables'].values():
        _need(isinstance(executable, dict) and _digest(executable['binary_sha256']), 'recovery executable identity missing')
        for stream in ('stdout', 'stderr'):
            _bytes(executable, 'version_' + stream)
        _need(executable['version_argv'] == [executable['path'], '--version'] and executable['version_exit_code'] == 0, 'recovery version receipt mismatch')
    receipts = copy.deepcopy(raw['receipts'])
    _need(len(receipts) == len(raw['tool_events']), 'recovery receipt count mismatch')
    for index, (receipt, event) in enumerate(zip(receipts, raw['tool_events'])):
        _need(receipt['sequence'] == index and all(receipt[key] == event[key] for key in ('tool_call_id', 'name', 'arguments', 'result')), 'recovery raw receipt differs from actual reply')
        _need(_bytes(receipt, 'stdin').decode('utf-8', errors='replace') == receipt['stdin'], 'recovery stdin bytes mismatch')
        _need(receipt['truncated'] is False, 'reviewed recovery pilot cannot contain unpreserved truncated streams')
        receipt['environment'] = copy.deepcopy(runtime['environment_values'])
        if receipt['execution_kind'] == 'host_function':
            _need(receipt['name'] == 'write_file' and receipt['operation'] == 'write_only_within_task_workspace', 'unreviewed recovery host operation')
            receipt['recorded_operation'] = receipt['operation']
            receipt['operation'] = json.loads(receipt['arguments'])
    _need(set(raw['artifact_byte_receipts']) == set(raw['artifacts']), 'recovery artifact capture incomplete')
    for name, value in {**raw['artifact_byte_receipts'], '__kv__': raw['kv_byte_receipt']}.items():
        data = base64.b64decode(value['content_b64'], validate=True)
        _need(len(data) == value['bytes'] and hashlib.sha256(data).hexdigest() == value['sha256'], 'recovery post-state byte hash mismatch')
        text = canonical_json(raw['kv']) if name == '__kv__' else raw['artifacts'][name]
        _need(data.decode('utf-8') == text, 'recovery post-state differs from captured bytes')
    _need(raw['final'] == raw['messages'][-1]['content'], 'recovery final differs from actual reply')
    return {'messages': copy.deepcopy(raw['messages']), 'effective_messages': copy.deepcopy(raw['effective_messages']),
            'model_events': copy.deepcopy(raw['model_events']), 'tool_events': copy.deepcopy(raw['tool_events']),
            'receipts': receipts, 'runtime': runtime, 'artifacts': copy.deepcopy(raw['artifacts']), 'kv': copy.deepcopy(raw['kv']),
            'task_sha256': content_hash(task), 'source_module_sha256': raw['source_sha256']['src/picoagent/data/luna_recovery_curriculum.py'],
            'tool_schemas_sha256': raw['tool_schemas_sha256'], 'final': raw['final']}


def _validate_store_bytes(snapshot: dict[str, Any], state: dict[str, Any]) -> None:
    if snapshot.get('exists') is False:
        _need(snapshot == {'exists': False} and state == {}, 'missing native store cannot claim file bytes or nonempty state')
        return
    _need(snapshot.get('exists') is True, 'native store snapshot must record existence')
    data = base64.b64decode(snapshot['content_b64'], validate=True)
    _need(len(data) == snapshot['bytes'] and hashlib.sha256(data).hexdigest() == snapshot['sha256'], 'native store file byte hash mismatch')
    _need(data == canonical_json(state).encode('utf-8'), 'native store file bytes differ from observed state')


def _extract_knowledge_search_observation(raw: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    _need(raw.get('schema') == 'picoagent.native_observation.v1' and raw.get('source_id') == 'native_knowledge_search', 'unknown native knowledge/search schema')
    _need(raw.get('execution') == 'native_teacher_observed' and raw.get('sft_admissible') is False and raw.get('teacher_model') is None and raw.get('teacher_mode') == TEACHER_MODE, 'knowledge/search teacher provenance mismatch')
    e = raw['native_evidence']
    _need(content_hash(e) == raw['raw_attempt_sha256'] and e['task'] == task and content_hash(task) == e['task_sha256'] == raw['task_sha256'], 'knowledge/search task or raw evidence hash mismatch')
    _need(content_hash(e['candidate']) == raw['candidate_sha256'] == e['candidate_sha256'] and e['candidate']['plan'] == task['reference']['plan'], 'knowledge/search original candidate mismatch')
    _need(e['candidate']['execution'] == 'unexecuted' and e['candidate']['training_eligible'] is False, 'knowledge/search original plan must stay unexecuted')
    _need(e['messages'][:2] == [{'role': 'system', 'content': e['system_prompt']}, {'role': 'user', 'content': task['prompt']}], 'knowledge/search full callback prompt mismatch')
    _need(hashlib.sha256(e['system_prompt'].encode()).hexdigest() == e['system_prompt_sha256'], 'knowledge/search system prompt hash mismatch')
    _need(content_hash(e['tool_schemas']) == e['tool_schemas_sha256'], 'knowledge/search protocol hash mismatch')
    _need(e['corpus_fixture'] == task['environment']['docs'] and content_hash(e['corpus_fixture']) == e['corpus_sha256'] and e['search_source'] == 'original_fixture_corpus', 'native search corpus must be the original task fixture')
    _need(task['environment']['files'] == {} and e['artifacts'] == {}, 'host-only source cannot contain executable task files or claim artifacts')
    state = copy.deepcopy(task['environment']['kv'])
    _need(e['initial_kv'] == state, 'knowledge store was not initialized from frozen fixtures')
    _validate_store_bytes(e['initial_knowledge_file'], state)
    file_state = e['initial_knowledge_file']
    receipts = copy.deepcopy(e['receipts'])
    _need(len(receipts) == len(e['tool_events']), 'host receipt/event count mismatch')
    for index, (receipt, event) in enumerate(zip(receipts, e['tool_events'])):
        _need(receipt['sequence'] == index and receipt['execution_kind'] == 'host_function' and receipt['name'] in {'knowledge', 'search'}, 'unreviewed host tool invocation')
        _need(all(receipt[k] == event[k] for k in ('name', 'tool_call_id', 'arguments', 'result')), 'host receipt differs from actual tool reply')
        args = json.loads(receipt['arguments'])
        _need(args == receipt['operation'] and isinstance(receipt['duration_seconds'], (int, float)) and receipt['duration_seconds'] >= 0, 'host operation/timing mismatch')
        _need(not any(k in receipt for k in ('argv', 'stdout', 'stderr', 'container_id')), 'host receipt cannot claim subprocess or container fields')
        if receipt['name'] == 'search':
            from .collector import LocalCorpusSearch
            corpus = task['environment']['docs']
            _need(receipt['corpus_snapshot'] == corpus and receipt['search_source'] == 'original_fixture_corpus', 'search receipt does not preserve its actual fixture corpus')
            for field in ('corpus_sha256', 'corpus_before_sha256', 'corpus_after_sha256'):
                _need(receipt[field] == content_hash(corpus), 'search corpus changed during retrieval')
            _need(receipt['result'] == LocalCorpusSearch(corpus).search(args['query'], args.get('limit', 5)), 'actual local search result differs from frozen corpus retrieval')
            receipt['state_before_sha256'] = receipt['corpus_before_sha256']
            receipt['state_after_sha256'] = receipt['corpus_after_sha256']
            receipt['state_hash_basis'] = 'original_fixture_corpus'
        else:
            _need(receipt['state_before'] == state and receipt['state_before_sha256'] == content_hash(state) and receipt['store_file_before'] == file_state, 'knowledge before-state or file continuity mismatch')
            _validate_store_bytes(receipt['store_file_before'], state)
            after = copy.deepcopy(state)
            op = args['operation']
            if op == 'list':
                expected = {'items': {k: v for k, v in sorted(state.items()) if k.startswith(args.get('prefix', ''))}, 'untrusted': True}
            elif op == 'get':
                expected = {'key': args['key'], 'value': state.get(args['key']), 'untrusted': True}
            elif op == 'set':
                after[args['key']] = copy.deepcopy(args['value'])
                expected = {'stored': args['key']}
            elif op == 'delete':
                expected = {'deleted': args['key'] in state}
                after.pop(args['key'], None)
            else:
                raise DataValidationError('unreviewed knowledge operation')
            _need(receipt['result'] == expected and receipt['state_after'] == after and receipt['state_after_sha256'] == content_hash(after), 'knowledge result or transition differs from observed inputs')
            _validate_store_bytes(receipt['store_file_after'], after)
            if op in {'get', 'list'}:
                _need(receipt['store_file_after'] == receipt['store_file_before'], 'knowledge read changed the store file')
            else:
                _need(receipt['store_file_after']['exists'] is True, 'knowledge mutation lacks an actual persisted store file')
            state, file_state = after, receipt['store_file_after']
    _need(e['kv'] == state and e['knowledge_file_final'] == file_state, 'knowledge final state/file does not follow actual operations')
    _validate_store_bytes(e['knowledge_file_final'], state)
    for model, callback in zip(e['model_events'], e['callback_events'], strict=True):
        _need(model['type'] == 'assistant' and callback['type'] == 'callback_decision' and all(model[k] == callback[k] for k in ('input_messages', 'message', 'tool_schemas')) and model['tool_schemas'] == e['tool_schemas'], 'host decision differs from actual callback context')
    runtime = copy.deepcopy(e['runtime'])
    _need(runtime['backend'] == 'native_teacher' and 'container_id' not in runtime, 'host source cannot claim container execution')
    runtime['executables'] = {'python': {'path': runtime['python_executable'], 'sha256': runtime['python_executable_sha256'], 'version': runtime['python_version']}}
    return {'messages': copy.deepcopy(e['messages']), 'effective_messages': copy.deepcopy(e['effective_messages']),
            'model_events': copy.deepcopy(e['model_events']), 'tool_events': copy.deepcopy(e['tool_events']), 'receipts': receipts,
            'runtime': runtime, 'artifacts': {}, 'kv': copy.deepcopy(e['kv']), 'task_sha256': content_hash(task),
            'source_module_sha256': e['source_sha256']['src/picoagent/data/luna_knowledge_search_curriculum.py'],
            'tool_schemas_sha256': e['tool_schemas_sha256'], 'final': e['final'] or ''}
