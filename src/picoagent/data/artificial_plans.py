"""Audited, explicitly synthetic action-plan views over observed tool traces.

Never label augmented model inputs as observed. Original tools, observations,
final answers and raw native evidence are unchanged. A compact view replaces a
normal view for selected base tasks; it does not multiply unique-task counts.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from .audit import file_hash, write_new_json
from .schema import canonical_json, content_hash

SCHEMA = "picoagent.artificial_action_plan.dataset.v1"
MAX_NOTE_BYTES = 256
MAX_NOTE_TOKENS = 24
APPROVED_TEMPLATE_REVIEWS: frozenset[str] = frozenset({'60e6ada7486014c5505192f0fdf0c20e4745594bcb1614e5e60ae2aa334273d1'})


def augment_record(record: dict, notes: list[dict]) -> dict:
    """Validate original visible prefixes and actions before changing content."""
    events = record.get("model_events")
    if not isinstance(events, list) or not events:
        raise ValueError("action-plan views require exact model events")
    if any(event.get("type") != "assistant" for event in events):
        raise ValueError("compaction views cannot be post-hoc annotated in this version")
    if record.get("provenance", {}).get("context_compaction_enabled"):
        raise ValueError("compacted traces need a separately validated augmentation")
    if not notes:
        raise ValueError("no action-plan annotations supplied")
    by_call = {}
    for note in notes:
        if note["task_id"] != record["task_id"] or note["tool_call_id"] in by_call:
            raise ValueError("annotation task mismatch or duplicate tool call")
        text = note["note"]
        if (not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_NOTE_BYTES
                or "\n" in text or "\r" in text or "END_MESSAGE" in text):
            raise ValueError("artificial plan must be short single-line text")
        by_call[note["tool_call_id"]] = note
    found = set()
    for event in events:
        message = event["message"]
        calls = message.get("tool_calls", [])
        if not calls:
            continue
        if len(calls) != 1 or (message.get("content") or "") != "":
            raise ValueError("initial augmentation supports one call with empty original content")
        call = calls[0]
        note = by_call.get(call["id"])
        if note is None:
            raise ValueError("every tool action in the selected trace needs an annotation")
        action = {"name": call["function"]["name"], "arguments_json": call["function"]["arguments"],
                  "tool_call_id": call["id"]}
        if (note["source_action_sha256"] != content_hash(action)
                or note["prefix_messages_sha256"] != content_hash(event["input_messages"])):
            raise ValueError("annotation does not bind the original visible prefix and action")
        found.add(call["id"])
    if found != set(by_call):
        raise ValueError("annotation refers to an unobserved tool action")
    result = copy.deepcopy(record)
    def rewrite(messages):
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                message["content"] = by_call[message["tool_calls"][0]["id"]]["note"]
    rewrite(result["messages"])
    if "effective_messages" in result:
        rewrite(result["effective_messages"])
    for event in result["model_events"]:
        rewrite(event["input_messages"])
        rewrite([event["message"]])
    result["artificial_action_plan_view"] = {
        "schema": "picoagent.artificial_action_plan.view.v1",
        "parent_trace_sha256": content_hash(record),
        "annotations_sha256": content_hash(notes),
        "model_context_origin": "synthetically_augmented_not_observed",
        "tool_execution_origin": "original_observations_unchanged",
        "note_origin": "model_authored_template_with_deterministic_reuse",
        "teacher_model": None,
    }
    result["provenance"]["model_context_origin"] = "synthetically_augmented_not_observed"
    return result



def template_body(template: dict) -> dict:
    body = {key: value for key, value in template.items() if key != "template_sha256"}
    if "template_sha256" in template and template["template_sha256"] != content_hash(body):
        raise ValueError("template self-hash does not match its canonical body")
    return body


def _review(templates: dict, review: dict, notes: list[dict]) -> str:
    digest = content_hash(review)
    if (digest not in APPROVED_TEMPLATE_REVIEWS or
            review.get("schema") != "picoagent.artificial_action_plan.review.v1" or
            review.get("template_bodies_sha256") != content_hash(templates) or
            review.get("annotation_rows_sha256") != content_hash(notes)):
        raise ValueError("literal templates and exact annotation corpus need an independently pinned review")
    identity = review.get("tokenizer_identity", {})
    if (not identity.get("model_id") or len(identity.get("revision", "")) != 40 or
            len(identity.get("tokenizer_json_sha256", "")) != 64):
        raise ValueError("review requires an immutable tokenizer identity")
    return digest


def validate_note_tokenizer(manifest_path: str | Path, tokenizer, *, model_id: str,
                            revision: str, tokenizer_json_path: str | Path) -> None:
    """Recount all unique literal notes with the actual training tokenizer."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    review = json.loads((path.parent / "template_review.json").read_text())
    identity = {"model_id": model_id, "revision": revision,
                "tokenizer_json_sha256": file_hash(tokenizer_json_path)}
    if identity != manifest["tokenizer_identity"] or identity != review["tokenizer_identity"]:
        raise ValueError("actual training tokenizer differs from the reviewed action-plan tokenizer")
    templates = json.loads((path.parent / "templates.json").read_text())
    for template in templates.values():
        count = len(tokenizer.encode(template["note"], add_special_tokens=False))
        if count != template["note_tokens"] or not 0 < count <= MAX_NOTE_TOKENS:
            raise ValueError("actual training tokenizer violates compact-note token evidence")


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("unsafe augmentation snapshot path")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("augmentation files must remain inside their snapshot")
    return target


def _notes(path: Path) -> list[dict]:
    output = []
    logical_bytes = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        while line := handle.readline(65537):
            logical_bytes += len(line.encode())
            if len(line) > 65536 or len(output) >= 100000 or logical_bytes > 64 * 1024 * 1024:
                raise ValueError("annotation corpus exceeds the reviewed bounds")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("annotation rows must be objects")
            output.append(value)
    return output


def _apply(rows: dict[str, list[dict]], notes: list[dict], templates: dict,
           selected: list[str], note_counter: Callable[[str], int] | None = None) -> dict:
    by_task = {}
    known_train = {row["task_id"]: row for row in rows["train"]}
    for note in notes:
        if note["task_id"] not in known_train:
            raise ValueError("this augmentation version accepts only training-task notes")
        raw_template = templates.get(note["template_id"])
        template = template_body(raw_template) if raw_template is not None else None
        if (template is None or note["template_sha256"] != content_hash(template)
                or note["note"] != template["note"]):
            raise ValueError("note differs from its reviewed literal template")
        if (type(note.get("note_tokens")) is not int or not 0 < note["note_tokens"] <= MAX_NOTE_TOKENS
                or note["note_tokens"] != template.get("note_tokens")):
            raise ValueError("annotation lacks bounded token-count evidence")
        record = known_train[note["task_id"]]
        actions = [(event, call) for event in record.get("model_events", [])
                   if event.get("type") == "assistant" for call in event["message"].get("tool_calls", [])]
        matches = [(index, event, call) for index, (event, call) in enumerate(actions)
                   if call["id"] == note["tool_call_id"]]
        if len(matches) != 1:
            raise ValueError("annotation does not identify exactly one original action")
        index, event, call = matches[0]
        if (template.get("family") != record["family"] or template.get("action_index") != index
                or template.get("tool_name") != call["function"]["name"]):
            raise ValueError("template family, action position or tool does not match")
        visible_user = "\n".join(message.get("content", "") for message in event["input_messages"]
                                  if message["role"] == "user").casefold()
        if any(term.casefold() not in visible_user for term in template.get("context_prerequisites", [])):
            raise ValueError("template prerequisites absent from visible user context")
        if note_counter is not None and note_counter(note["note"]) != note["note_tokens"]:
            raise ValueError("recorded note token count differs from pinned tokenizer")
        by_task.setdefault(note["task_id"], []).append(note)
    if len(selected) != len(set(selected)) or not set(selected) <= set(by_task):
        raise ValueError("selected annotation tasks must be unique and have notes")
    selected_set = set(selected)
    result = {"train": [], "dev": rows["dev"]}
    # Validate every supplied annotation, even if this run keeps its normal view.
    for record in rows["train"]:
        augmented = augment_record(record, by_task[record["task_id"]]) if record["task_id"] in by_task else None
        result["train"].append(augmented if record["task_id"] in selected_set else record)
    return result


def verify_plan_snapshot(manifest_path: str | Path, *, allow_native_teacher: bool = False,
                         allow_artificial_action_plans: bool = False) -> tuple[dict, dict]:
    if not allow_artificial_action_plans or not allow_native_teacher:
        raise ValueError("artificial plans require explicit augmentation and native-evidence opt-ins")
    from .native_admission import verify_native_snapshot
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("lockbox_used") is not False
            or manifest.get("selection") != "one_normal_or_annotated_view_per_base_task"):
        raise ValueError("invalid artificial-action-plan manifest")
    required = {"base/manifest.json", "templates.json", "template_review.json", "annotations.jsonl.gz"}
    if not required <= set(manifest["files"]):
        raise ValueError("augmentation manifest omits required hashed evidence")
    for relative, info in manifest["files"].items():
        target = _inside(path.parent, relative)
        if not target.is_file() or target.stat().st_size != info["bytes"] or file_hash(target) != info["sha256"]:
            raise ValueError("artificial-plan snapshot integrity mismatch")
    base_path = _inside(path.parent, "base/manifest.json")
    if file_hash(base_path) != manifest["base_manifest_sha256"]:
        raise ValueError("base native snapshot identity changed")
    base, rows = verify_native_snapshot(base_path, allow_native_teacher=True)
    if not {"base/" + name for name in base["files"]} <= set(manifest["files"]):
        raise ValueError("augmentation manifest omits base evidence files")
    templates = json.loads(_inside(path.parent, "templates.json").read_text())
    review = json.loads(_inside(path.parent, "template_review.json").read_text())
    notes = _notes(_inside(path.parent, "annotations.jsonl.gz"))
    if (_review(templates, review, notes) != manifest["template_review_sha256"] or
            review["tokenizer_identity"] != manifest["tokenizer_identity"]):
        raise ValueError("template review or tokenizer identity changed")
    augmented = _apply(rows, notes, templates, manifest["selected_task_ids"])
    counts = {split: len(values) for split, values in augmented.items()}
    if counts != manifest["counts"] or len(manifest["selected_task_ids"]) != manifest["annotated_train_tasks"]:
        raise ValueError("augmentation counts differ from unique base tasks")
    return manifest, augmented


def _copy_verified_plan_snapshot(manifest_path: str | Path, destination: str | Path, *,
                                 verified_manifest: dict, expected_manifest_sha256: str) -> Path:
    """Internal byte-checked copy after same-process strict semantic verification."""
    source = Path(manifest_path).resolve()
    raw = source.read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected_manifest_sha256
            or json.loads(raw) != verified_manifest or verified_manifest.get("schema") != SCHEMA):
        raise ValueError("augmentation manifest changed since strict verification")
    files = verified_manifest.get("files", {})
    if not isinstance(files, dict) or not files or "manifest.json" in files:
        raise ValueError("augmentation requires a separate evidence inventory")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    for relative, info in files.items():
        output = _inside(target, relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        digest, size = hashlib.sha256(), 0
        with _inside(source.parent, relative).open("rb") as incoming, output.open("xb") as outgoing:
            while block := incoming.read(1024 * 1024):
                digest.update(block)
                size += len(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if (size != info["bytes"] or digest.hexdigest() != info["sha256"]
                or output.stat().st_size != size or file_hash(output) != info["sha256"]):
            raise ValueError("augmentation evidence changed during snapshot copy")
        os.chmod(output, 0o444)
    with (target / "manifest.json").open("xb") as outgoing:
        outgoing.write(raw)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if file_hash(source) != expected_manifest_sha256 or file_hash(target / "manifest.json") != expected_manifest_sha256:
        raise ValueError("augmentation manifest changed during snapshot copy")
    os.chmod(target / "manifest.json", 0o444)
    return target / "manifest.json"


def copy_plan_snapshot(manifest_path: str | Path, destination: str | Path) -> Path:
    source = Path(manifest_path).resolve()
    expected = file_hash(source)
    manifest, _ = verify_plan_snapshot(source, allow_native_teacher=True, allow_artificial_action_plans=True)
    return _copy_verified_plan_snapshot(source, destination, verified_manifest=manifest,
                                        expected_manifest_sha256=expected)


def build_plan_snapshot(base_manifest: str | Path, notes: list[dict], templates: dict,
                        destination: str | Path, *, note_counter: Callable[[str], int],
                        tokenizer_identity: dict, template_review: dict) -> Path:
    """Call only after reviewing the literal templates; no automatic semantic approval."""
    from .native_admission import _copy_verified_native_snapshot, verify_native_snapshot
    templates = {name: template_body(template) for name, template in templates.items()}
    template_review_sha256 = _review(templates, template_review, notes)
    if tokenizer_identity != template_review["tokenizer_identity"]:
        raise ValueError("tokenizer identity differs from pinned template review")
    base_hash = file_hash(base_manifest)
    base, rows = verify_native_snapshot(base_manifest, allow_native_teacher=True)
    if base.get("storage") not in {"gzip_sharded_v1", "sealed_native_collection_v1"}:
        raise ValueError("augmented production views require bounded native source shards")
    by_family = {}
    noted = {note["task_id"] for note in notes}
    for record in rows["train"]:
        if record["task_id"] in noted:
            by_family.setdefault(record["family"], []).append(record["task_id"])
    # Half per family, one view per base. Original traces stay in the base snapshot.
    selected = sorted(task for tasks in by_family.values() for task in sorted(tasks)[::2])
    _apply(rows, notes, templates, selected, note_counter)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    copied = _copy_verified_native_snapshot(base_manifest, root / "base", verified_manifest=base,
                                            expected_manifest_sha256=base_hash)
    write_new_json(root / "templates.json", templates)
    write_new_json(root / "template_review.json", template_review)
    with (root / "annotations.jsonl.gz").open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for note in notes:
                stream.write((canonical_json(note) + "\n").encode())
    files = {p.relative_to(root).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
             for p in sorted(root.rglob("*")) if p.is_file()}
    manifest = {"schema": SCHEMA, "lockbox_used": False,
                "selection": "one_normal_or_annotated_view_per_base_task",
                "base_manifest_sha256": file_hash(copied), "files": files,
                "template_review_sha256": template_review_sha256,
                "tokenizer_identity": tokenizer_identity, "selected_task_ids": selected,
                "annotated_train_tasks": len(selected),
                "counts": {split: len(values) for split, values in rows.items()},
                "model_context_origin": "selected_contexts_synthetically_augmented_not_observed",
                "tool_execution_origin": "original_native_observations_unchanged"}
    write_new_json(root / "manifest.json", manifest)
    verify_plan_snapshot(root / "manifest.json", allow_native_teacher=True, allow_artificial_action_plans=True)
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    return root / "manifest.json"
