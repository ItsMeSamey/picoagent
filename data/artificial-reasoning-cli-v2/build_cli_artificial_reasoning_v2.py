#!/usr/bin/env python3
"""Build compact, prefix-only synthetic action notes from the CLI train packets.

This script reads only the frozen train prefix-packet file. It does not open any
observation outputs or dev/test packets and does not replay tools.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_REL = "data/luna-cli-v1/native_teacher_observed/batch_train_dev_10k_v1/annotation_packets/train.prefix_only_packets.jsonl"
SOURCE = ROOT / SOURCE_REL
SOURCE_SHA256 = "e387e5beae6423115779f884d71286a9ebf746400c6bf37840a7b1169da9ab7a"
TOKENIZER_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
TOKENIZER_DIR = Path("/workspace/scratch/443c242dcc32/.hf-cache/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots") / TOKENIZER_REVISION
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
MAX_NOTE_TOKENS = 24


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8"))


# Each literal note summarizes the proposed operation only. The action-shape
# hash is filled from the first train packet for the corresponding family/slot,
# then every packet is independently checked against that shape and context.
TEMPLATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "family": "cli.csv_average_price", "action_index": 0,
        "operation": "Read the catalog CSV, compute the arithmetic mean of price, and emit average_price JSON.",
        "note": "catalog.csv→mean(price); emit JSON.average_price",
        "tool_name": "python", "prompt_contains": ["input/catalog.csv", "arithmetic mean", "price"],
        "action_checks": ["CSV.DictReader", "price-field mean", "JSON key average_price"],
    },
    {
        "family": "cli.csv_below_stock", "action_index": 0,
        "operation": "Read catalog rows, filter units strictly below the prompt's limit, sort SKU values, and emit items JSON.",
        "note": "catalog.csv→units<prompt_limit; sort SKU; emit JSON.items",
        "tool_name": "python", "prompt_contains": ["input/catalog.csv", "units are strictly below", "SKU values", "alphabetically"],
        "action_checks": ["CSV.DictReader", "units below prompt limit", "sorted SKU", "JSON key items"],
    },
    {
        "family": "cli.csv_name_order", "action_index": 0,
        "operation": "Read catalog rows, alphabetize SKU values, and emit items JSON.",
        "note": "catalog.csv→sort SKU; emit JSON.items",
        "tool_name": "python", "prompt_contains": ["input/catalog.csv", "SKU values", "sorted alphabetically"],
        "action_checks": ["CSV.DictReader", "sorted SKU", "JSON key items"],
    },
    {
        "family": "cli.csv_revenue_total", "action_index": 0,
        "operation": "Read catalog rows, sum units multiplied by price, and emit integer revenue JSON.",
        "note": "catalog.csv→sum(units×price); emit JSON.revenue",
        "tool_name": "python", "prompt_contains": ["input/catalog.csv", "total revenue", "units", "price"],
        "action_checks": ["CSV.DictReader", "sum units times price", "JSON key revenue"],
    },
    {
        "family": "cli.json_group_totals", "action_index": 0,
        "operation": "Read JSON records, group by group, sum amount, and emit alphabetized totals.",
        "note": "records.json→group/sum(amount); sort groups; emit totals",
        "tool_name": "python", "prompt_contains": ["input/records.json", "sum amount by group"],
        "action_checks": ["JSON load records", "group amount accumulation", "sorted totals", "JSON key totals"],
    },
    {
        "family": "cli.json_sorted_keys", "action_index": 0,
        "operation": "Read settings JSON, sort top-level keys, and emit keys JSON.",
        "note": "settings.json→sort top-level keys; emit JSON.keys",
        "tool_name": "python", "prompt_contains": ["input/settings.json", "top-level keys", "alphabetical order"],
        "action_checks": ["JSON load settings", "sorted top-level keys", "JSON key keys"],
    },
    {
        "family": "cli.json_unique_tags", "action_index": 0,
        "operation": "Read JSON records, collect unique tags, alphabetize them, and emit tags JSON.",
        "note": "records.json→unique tags; sort; emit JSON.tags",
        "tool_name": "python", "prompt_contains": ["input/records.json", "unique tags", "alphabetical order"],
        "action_checks": ["JSON load records", "unique record tags", "sorted tags", "JSON key tags"],
    },
    {
        "family": "cli.log_level_counts", "action_index": 0,
        "operation": "Read the event log and count ERROR and WARN line prefixes.",
        "note": "events.log→count ERROR/WARN prefixes; emit JSON counts",
        "tool_name": "python", "prompt_contains": ["input/events.log", "count lines beginning ERROR and WARN"],
        "action_checks": ["read log lines", "ERROR prefix count", "WARN prefix count", "JSON errors/warnings"],
    },
    {
        "family": "cli.readonly_inventory", "action_index": 0,
        "operation": "Read only report.txt and labels.txt, count the named files and combined lines, and make no writes.",
        "note": "read report+labels; count files/lines; no writes",
        "tool_name": "python", "prompt_contains": ["input/report.txt", "input/labels.txt", "read only", "do not create, edit, or delete"],
        "action_checks": ["Path report and labels", "combined line count", "JSON files/lines", "no write calls"],
    },
    {
        "family": "cli.recover_missing_csv", "action_index": 0,
        "operation": "Perform the proposed read-only probe of the suggested legacy CSV path.",
        "note": "read-only probe: cat input/catalog-old.csv",
        "tool_name": "bash", "prompt_contains": ["catalog-old.csv", "may be missing", "do not change any files"],
        "action_checks": ["exact cat of suggested path", "read-only probe"],
    },
    {
        "family": "cli.recover_missing_csv", "action_index": 1,
        "operation": "After the visible missing-path error, find the sole input CSV, sum units times price, and emit revenue JSON.",
        "note": "old CSV missing; glob sole *.csv; sum units×price; emit revenue",
        "tool_name": "python", "prompt_contains": ["catalog-old.csv", "sole CSV", "revenue", "do not change any files"],
        "action_checks": ["prior FileNotFound result", "glob input CSV", "unique path check", "sum units times price", "JSON key revenue"],
    },
    {
        "family": "cli.text_headings", "action_index": 0,
        "operation": "Read Markdown notes, extract heading text in original order, and emit headings JSON.",
        "note": "notes.txt→Markdown heading text; keep order; emit JSON.headings",
        "tool_name": "python", "prompt_contains": ["input/notes.txt", "Markdown heading text", "original order"],
        "action_checks": ["read notes lines", "Markdown heading filter", "JSON key headings"],
    },
    {
        "family": "cli.text_word_counts", "action_index": 0,
        "operation": "Read labels, lowercase words, count them, and emit counts JSON.",
        "note": "labels.txt→lowercase words/count; emit JSON.counts",
        "tool_name": "python", "prompt_contains": ["input/labels.txt", "case-insensitive words", "count"],
        "action_checks": ["read labels", "lowercase word extraction", "word counter", "JSON key counts"],
    },
)


def packet_action_index(packet: dict[str, Any]) -> int:
    match = re.search(r"::action-(\d+)$", packet["packet_id"])
    if not match:
        raise ValueError("packet id has no action position")
    return int(match.group(1))


def action_signature(packet: dict[str, Any], family: str) -> str:
    action = packet["proposed_action"]
    args = json.loads(action["arguments_json"])
    if action["name"] == "bash":
        shape: Any = {"tool": "bash", "argv": shlex.split(args["command"])}
    elif action["name"] == "python":
        tree = ast.parse(args["code"])
        if family == "cli.csv_below_stock":
            class ThresholdNormalizer(ast.NodeTransformer):
                def visit_Compare(self, node: ast.Compare) -> ast.AST:
                    self.generic_visit(node)
                    if any(isinstance(op, ast.Lt) for op in node.ops):
                        node.comparators = [
                            ast.Constant(value="<PROMPT_LIMIT>") if isinstance(item, ast.Constant) and type(item.value) is int else item
                            for item in node.comparators
                        ]
                    return node
            tree = ThresholdNormalizer().visit(tree)
            ast.fix_missing_locations(tree)
        shape = {"tool": "python", "ast": ast.dump(tree, include_attributes=False)}
    else:
        raise ValueError(f"unexpected tool {action['name']}")
    return content_hash(shape)


def prompt_text(packet: dict[str, Any]) -> str:
    return "\n".join(m.get("content", "") for m in packet["prefix_messages"] if m.get("role") == "user").lower()


def validate_context(spec: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if packet.get("split") != "train":
        errors.append("not_train")
    if packet.get("family") != spec["family"] or packet_action_index(packet) != spec["action_index"]:
        errors.append("family_or_position_mismatch")
    if packet["proposed_action"].get("name") != spec["tool_name"]:
        errors.append("tool_name_mismatch")
    user = prompt_text(packet)
    for term in spec["context_prerequisites"]:
        if term.lower() not in user:
            errors.append(f"missing_prompt_context:{term}")
    roles = [m.get("role") for m in packet["prefix_messages"]]
    if not roles or roles[0] != "system" or "user" not in roles:
        errors.append("missing_system_or_user_prefix")
    if spec["family"] == "cli.csv_below_stock":
        threshold = re.search(r"units are strictly below\s+(\d+)", user)
        action_code = json.loads(packet["proposed_action"]["arguments_json"])["code"]
        code_threshold = re.search(r'int\(r\["units"\]\)\s*<\s*(\d+)', action_code)
        if not threshold or not code_threshold or threshold.group(1) != code_threshold.group(1):
            errors.append("prompt_action_threshold_mismatch")
    if spec["family"] == "cli.readonly_inventory":
        code = json.loads(packet["proposed_action"]["arguments_json"])["code"]
        tree = ast.parse(code)
        forbidden_calls = {"write_text", "write_bytes", "unlink", "mkdir", "rename", "replace", "touch", "rmdir"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                errors.append(f"readonly_action_has_mutating_call:{node.func.attr}")
    if spec["family"] == "cli.recover_missing_csv" and spec["action_index"] == 0:
        if any(m.get("role") == "tool" for m in packet["prefix_messages"]):
            errors.append("probe_prefix_has_unexpected_prior_tool_result")
    if spec["family"] == "cli.recover_missing_csv" and spec["action_index"] == 1:
        prior_tool = [m for m in packet["prefix_messages"] if m.get("role") == "tool" and m.get("name") == "bash"]
        prior_call = None
        for message in packet["prefix_messages"]:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls", []):
                if call.get("id") == (prior_tool[-1].get("tool_call_id") if prior_tool else None):
                    prior_call = call
        if not prior_call:
            errors.append("missing_prior_cat_call")
        else:
            try:
                command = json.loads(prior_call["function"]["arguments"])["command"]
                if prior_call["function"].get("name") != "bash" or shlex.split(command) != ["cat", "input/catalog-old.csv"]:
                    errors.append("prior_bash_call_not_expected_missing_path")
            except (KeyError, TypeError, json.JSONDecodeError):
                errors.append("invalid_prior_bash_call")
        if not prior_tool:
            errors.append("missing_prior_bash_receipt")
        else:
            try:
                result = json.loads(prior_tool[-1]["content"])
                if result.get("exit_code") != 1 or "No such file or directory" not in result.get("stderr", ""):
                    errors.append("prior_bash_receipt_not_missing_file_error")
            except (TypeError, json.JSONDecodeError):
                errors.append("invalid_prior_bash_receipt")
    return errors


def main() -> int:
    existing_outputs = [p for p in OUT.iterdir() if p.name != Path(__file__).name] if OUT.exists() else []
    if existing_outputs:
        raise SystemExit(f"refusing to overwrite non-empty output directory: {OUT}")
    if not SOURCE.is_file() or sha256(SOURCE.read_bytes()) != SOURCE_SHA256:
        raise SystemExit("train packet source missing or hash mismatch")
    tokenizer_json = TOKENIZER_DIR / "tokenizer.json"
    if not tokenizer_json.is_file() or sha256(tokenizer_json.read_bytes()) != TOKENIZER_SHA256:
        raise SystemExit("pinned local Smol tokenizer is unavailable or hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True, use_fast=True)

    # One in-memory train-only pass supplies a reference AST shape and then the
    # exact packet rows. No dev/test data are read.
    packets: list[tuple[int, bytes, dict[str, Any]]] = []
    first: dict[tuple[str, int], dict[str, Any]] = {}
    with SOURCE.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            packet = json.loads(raw)
            if packet.get("split") != "train":
                raise ValueError("train source unexpectedly contains a non-train packet")
            key = (packet["family"], packet_action_index(packet))
            first.setdefault(key, packet)
            packets.append((line_number, raw.rstrip(b"\n"), packet))

    specs_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    templates: list[dict[str, Any]] = []
    for original in TEMPLATE_SPECS:
        spec = copy.deepcopy(original)
        key = (spec["family"], spec["action_index"])
        if key not in first:
            continue
        spec["template_id"] = f"{spec['family']}.action-{spec['action_index']}.compact.v2"
        spec["action_signature_method"] = "canonical tool + Python AST without source positions; prompt-threshold numeric literal normalized only for below-stock family; bash argv exact"
        spec["action_signature_sha256"] = action_signature(first[key], spec["family"])
        spec["context_prerequisites"] = spec.pop("prompt_contains")
        spec["note_tokens"] = len(tokenizer.encode(spec["note"], add_special_tokens=False))
        if spec["note_tokens"] > MAX_NOTE_TOKENS:
            raise SystemExit(f"template exceeds pinned token budget: {spec['template_id']}")
        spec["author_type"] = "native_assistant_synthetic_annotation"
        spec["author_model"] = None
        spec["author_model_status"] = "unknown_not_exposed_to_this_worker"
        spec["authorship_method"] = "assistant-authored literal template; deterministic reuse per matching prefix packet"
        spec["template_sha256"] = content_hash(spec)
        specs_by_key[key] = spec
        templates.append(spec)

    template_keys = set(specs_by_key)
    seen_keys = {(p["family"], packet_action_index(p)) for _, _, p in packets}
    missing_keys = sorted(seen_keys - template_keys)
    unused_keys = sorted(template_keys - seen_keys)
    if missing_keys:
        raise SystemExit(f"missing template action positions: {missing_keys}")
    if unused_keys:
        raise SystemExit(f"templates without observed train positions: {unused_keys}")

    notes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts_by_family: Counter[str] = Counter()
    counts_by_template: Counter[str] = Counter()
    token_lengths: list[int] = []
    task_ids: set[str] = set()
    for line_number, raw_without_newline, packet in packets:
        key = (packet["family"], packet_action_index(packet))
        spec = specs_by_key[key]
        problems = validate_context(spec, packet)
        try:
            signature = action_signature(packet, spec["family"])
        except Exception as exc:  # preserve and report malformed/unmatched examples
            signature = None
            problems.append(f"action_parse_failure:{type(exc).__name__}")
        if signature != spec["action_signature_sha256"]:
            problems.append("proposed_action_operation_mismatch")
        note = spec["note"]
        note_tokens = len(tokenizer.encode(note, add_special_tokens=False))
        if note_tokens != spec["note_tokens"]:
            problems.append("template_note_token_count_mismatch")
        if note_tokens > MAX_NOTE_TOKENS:
            problems.append("note_exceeds_pinned_token_budget")
        if problems:
            rejected.append({
                "packet_id": packet.get("packet_id"), "task_id": packet.get("task_id"),
                "family": packet.get("family"), "action_index": packet_action_index(packet),
                "parent_packet_sha256": sha256(raw_without_newline), "problems": problems,
            })
            continue
        row = {
            "schema": "picoagent.cli_artificial_action_plan.v2",
            "task_id": packet["task_id"],
            "packet_id": packet["packet_id"],
            "family": packet["family"],
            "split": "train",
            "action_index": packet_action_index(packet),
            "tool_call_id": packet["proposed_action"]["tool_call_id"],
            "note": note,
            "template_id": spec["template_id"],
            "template_sha256": spec["template_sha256"],
            "parent_packet_sha256": sha256(raw_without_newline),
            "parent_packet_hash_method": "sha256(raw_jsonl_line_without_newline)",
            "prefix_messages_sha256": content_hash(packet["prefix_messages"]),
            "source_action_sha256": content_hash(packet["proposed_action"]),
            "packet_source_action_sha256": packet["source_action_sha256"],
            "source_locator": {"path": SOURCE_REL, "line_number": line_number},
            "proposed_action_signature_sha256": signature,
            "note_tokens": note_tokens,
            "tokenizer": "HuggingFaceTB/SmolLM2-360M",
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_json_sha256": TOKENIZER_SHA256,
            "author_type": "native_assistant_synthetic_annotation",
            "author_model": None,
            "author_model_status": "unknown_not_exposed_to_this_worker",
            "authorship_method": "model-authored templates with deterministic reuse; no per-packet model call",
            "training_eligible": False,
            "validation": {"passed": True, "scope": "prefix/context and proposed-action shape only"},
        }
        notes.append(row)
        counts_by_family[row["family"]] += 1
        counts_by_template[row["template_id"]] += 1
        token_lengths.append(note_tokens)
        task_ids.add(row["task_id"])

    if not OUT.exists():
        OUT.mkdir(parents=True)
    write_jsonl_exclusive(OUT / "templates.jsonl", templates)
    write_jsonl_exclusive(OUT / "train.notes.jsonl", notes)
    write_jsonl_exclusive(OUT / "rejected_matches.jsonl", rejected)

    script_sha = sha256(Path(__file__).read_bytes())
    template_path = OUT / "templates.jsonl"
    notes_path = OUT / "train.notes.jsonl"
    reject_path = OUT / "rejected_matches.jsonl"
    manifest = {
        "schema": "picoagent.cli_artificial_reasoning_manifest.v2",
        "status": "synthetic_template_reuse_pending_augmented_view_audit",
        "training_eligible": False,
        "source_split_read": "train_only",
        "source_packet_file": SOURCE_REL,
        "source_packet_file_sha256": SOURCE_SHA256,
        "source_packet_count": len(packets),
        "unique_source_task_count": len(task_ids),
        "unique_families": len({p["family"] for _, _, p in packets}),
        "unique_action_positions": len(seen_keys),
        "templates": {"path": template_path.name, "count": len(templates), "sha256": sha256(template_path.read_bytes()), "template_ids": [t["template_id"] for t in templates]},
        "notes": {"path": notes_path.name, "count": len(notes), "sha256": sha256(notes_path.read_bytes()), "count_by_family": dict(sorted(counts_by_family.items())), "count_by_template": dict(sorted(counts_by_template.items()))},
        "rejected_matches": {"path": reject_path.name, "count": len(rejected), "sha256": sha256(reject_path.read_bytes()), "reason_counts": dict(sorted(Counter(reason for row in rejected for reason in row["problems"]).items()))},
        "missing_template_action_positions": missing_keys,
        "templates_without_observed_action_positions": unused_keys,
        "all_instantiated_notes_match_literal_template": all(row["note"] == specs_by_key[(row["family"], row["action_index"])]["note"] for row in notes),
        "tokenizer": {"model": "HuggingFaceTB/SmolLM2-360M", "revision": TOKENIZER_REVISION, "tokenizer_json_sha256": TOKENIZER_SHA256, "max_note_tokens": MAX_NOTE_TOKENS, "measured_note_tokens_min": min(token_lengths) if token_lengths else None, "measured_note_tokens_max": max(token_lengths) if token_lengths else None, "measured_note_tokens_mean": (sum(token_lengths) / len(token_lengths)) if token_lengths else None, "counting": "AutoTokenizer fast tokenizer; encode(note, add_special_tokens=False)"},
        "authorship": {"author_type": "native_assistant_synthetic_annotation", "model": None, "model_status": "unknown_not_exposed_to_this_worker", "template_count": len(templates), "model_authored_templates_then_deterministic_reuse": True, "per_packet_model_calls": 0, "private_reasoning_used": False},
        "coverage": {"input_packets": len(packets), "accepted_notes": len(notes), "rejected_packets": len(rejected), "accepted_plus_rejected_equals_input": len(notes) + len(rejected) == len(packets), "unique_tasks_with_notes": len(task_ids), "packets_with_notes_by_family": dict(sorted(counts_by_family.items()))},
        "do_not_read_or_include": ["dev packets", "test packets", "future tool outputs", "final answers", "oracle values", "real traces were not modified"],
        "builder_script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": script_sha},
    }
    write_json_exclusive(OUT / "manifest.json", manifest)
    print(json.dumps({"accepted": len(notes), "rejected": len(rejected), "templates": len(templates), "unique_tasks": len(task_ids), "token_min": manifest["tokenizer"]["measured_note_tokens_min"], "token_max": manifest["tokenizer"]["measured_note_tokens_max"], "manifest": str(OUT / "manifest.json")}, sort_keys=True))
    return 0 if not rejected and len(notes) == len(packets) else 1


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def write_json_exclusive(path: Path, obj: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
