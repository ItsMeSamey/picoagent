"""Original, family-held-out CLI and safe file-work curriculum.

Candidate plans here are procedural authoring metadata, not tool transcripts.
They are marked unexecuted until collect_task runs them through the shared
ContainerSandbox/AgentHarness. The optional native replay runs only these
Luna-authored safe teacher fixtures and is separate from any learner rollout.
"""
from __future__ import annotations

import copy
import ast
import base64
import csv
import hashlib
import io
import json
import locale
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .audit import file_hash, read_jsonl, verify_curriculum, write_curriculum
from .generators import GENERATOR_VERSION, SYSTEM_PROMPT
from .oracles import check_task_result
from .schema import SCHEMA_VERSION, canonical_json, content_hash, validate_task

CURRICULUM_VERSION = "luna-cli-v1"
CANDIDATE_SCHEMA = "picoagent.cli_candidate_plan.v1"
EVALUATION_EXCLUSIONS = {
    "cli.english_threshold_count:00000000": {
        "task_id": "cli.english_threshold_count:00000000",
        "seed": 0,
        "reason": "Transient procedural-teacher generator smoke check ran before holdout execution gating was requested.",
        "scope": "Exclude this exact task instance from any sealed final test evaluation; keep its family otherwise held out.",
        "driver_invocation": "run_native_teacher_observed([generate_task('cli.english_threshold_count')], <TemporaryDirectory>/cli_english_threshold_count); default seed=0",
        "ephemeral_result_path": "/tmp/tmp05myvfzq/cli_english_threshold_count/observations.jsonl",
        "available_tool_output_summary": {"records": 1, "passed": 1, "failed": 0},
        "raw_stdout_stderr_tool_receipt": "not retained; do not reconstruct or treat the summary as a raw observation",
        "execution_context": "platform exec-command sandbox with a per-run TemporaryDirectory; it was removed when the smoke-check process exited; not ContainerSandbox",
        "teacher_identity": "luna_cli_procedural_fixture_teacher; no provider model involved",
        "model_training_or_pretraining_contamination": False,
    }
}

# Family membership, not seed, determines the split. Each split contains 12
# families; all seeds of any family remain in that one split.
FAMILY_SPLITS: dict[str, str] = {
    "cli.csv_revenue_total": "train",
    "cli.csv_average_price": "train",
    "cli.csv_below_stock": "train",
    "cli.csv_name_order": "train",
    "cli.text_word_counts": "train",
    "cli.text_headings": "train",
    "cli.json_sorted_keys": "train",
    "cli.json_unique_tags": "train",
    "cli.json_group_totals": "train",
    "cli.log_level_counts": "train",
    "cli.readonly_inventory": "train",
    "cli.recover_missing_csv": "train",
    "cli.csv_team_max": "dev",
    "cli.csv_top_value": "dev",
    "cli.csv_category_count": "dev",
    "cli.text_longest_line": "dev",
    "cli.text_unique_lines": "dev",
    "cli.json_active_ids": "dev",
    "cli.json_project_fields": "dev",
    "cli.json_defaults_overlay": "dev",
    "cli.json_nested_totals": "dev",
    "cli.log_first_error_context": "dev",
    "cli.english_readonly_report": "dev",
    "cli.help_flag_lookup": "dev",
    "cli.csv_median_price": "test",
    "cli.english_threshold_count": "test",
    "cli.csv_reverse_order": "test",
    "cli.text_occurrence_context": "test",
    "cli.english_unique_word_count": "test",
    "cli.json_score_order": "test",
    "cli.json_missing_required_fields": "test",
    "cli.english_boolean_tally": "test",
    "cli.log_first_error_line": "test",
    "cli.readonly_empty_files": "test",
    "cli.recover_bad_log_suffix": "test",
    "cli.help_recovery_recipe": "test",
}

_CSV_NAMES = ("amber-mug", "birch-tray", "cedar-lamp", "dune-cup", "elm-bowl", "fern-clock", "hazel-mat", "iris-pen")
_WORDS = ("cedar", "quiet", "river", "maple", "bright", "stone", "meadow", "paper", "silver", "garden", "thread", "morning")
_LEVELS = ("INFO", "WARN", "ERROR")


def _rng(family: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{CURRICULUM_VERSION}:{family}:{seed}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _csv_text(headers: list[str], rows: list[list[Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def _seeded_material(family: str, seed: int) -> dict[str, Any]:
    rng = _rng(family, seed)
    token = hashlib.sha256(f"{family}:{seed}".encode()).hexdigest()[:8]
    names = list(_CSV_NAMES)
    rng.shuffle(names)
    catalog = [
        {"sku": name, "team": rng.choice(("north", "south", "west")),
         "category": rng.choice(("home", "office", "garden")), "units": rng.randint(1, 14),
         "price": rng.randint(3, 31)}
        for name in names
    ]
    rows = [{"id": f"r{i}", "name": _CSV_NAMES[i], "score": rng.randint(1, 99),
             "active": rng.choice((True, False)), "group": rng.choice(("north", "south", "west")),
             "amount": rng.randint(2, 35), "tags": rng.sample(["blue", "green", "amber", "quiet", "rapid"], 2)}
            for i in range(7)]
    labels = [rng.choice(_WORDS[:8]) for _ in range(13)]
    text_lines = [
        "# Shift Notes",
        f"Keep {rng.choice(_WORDS)} records together",
        "## Checked Files",
        f"Read the {rng.choice(_WORDS)} list carefully",
        "Use small safe steps",
        f"Review {rng.choice(_WORDS)} before saving",
    ]
    severity_lines = [f"{rng.choice(_LEVELS)} batch_{i:02d} item={rng.randint(10, 99)}" for i in range(15)]
    # Ensure every log task has at least one error and a useful context line.
    severity_lines[4] = "INFO batch_04 item=41"
    severity_lines[5] = f"ERROR batch_05 item={rng.randint(10, 99)}"
    severity_lines[6] = "WARN batch_06 item=62"
    records_csv = _csv_text(["sku", "team", "category", "units", "price"],
                            [[r[k] for k in ("sku", "team", "category", "units", "price")] for r in catalog])
    config = {"zeta": rng.randint(1, 9), "alpha": "ready", "quota": rng.randint(5, 40), "beta": False}
    environment: dict[str, Any] = {"files": {}, "kv": {}, "docs": []}
    return {"token": token, "rng": rng, "catalog": catalog, "records": rows, "labels": labels,
            "text_lines": text_lines, "logs": severity_lines, "records_csv": records_csv,
            "config": config, "environment": environment}


def _script(op: str, params: dict[str, Any]) -> str:
    """Return a safe, file-reading solution program; never interpolate answers."""
    path = params.get("csv_path", "input/catalog.csv")
    if op == "revenue_total":
        return f'''import csv,json
rows=list(csv.DictReader(open({path!r}, newline="")))
print(json.dumps({{"revenue":sum(int(r["units"])*int(r["price"]) for r in rows)}}))'''
    if op == "average_price":
        return '''import csv,json
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
print(json.dumps({"average_price":sum(int(r["price"]) for r in rows)/len(rows)}))'''
    if op == "below_stock":
        return f'''import csv,json
rows=csv.DictReader(open("input/catalog.csv", newline=""))
print(json.dumps({{"items":sorted(r["sku"] for r in rows if int(r["units"]) < {params["threshold"]})}}))'''
    if op == "name_order":
        return '''import csv,json
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
print(json.dumps({"items":sorted(r["sku"] for r in rows)}))'''
    if op == "team_max":
        return '''import csv,json
best={}
for r in csv.DictReader(open("input/catalog.csv", newline="")):
 key=r["team"]; best[key]=max(best.get(key,0),int(r["units"]))
print(json.dumps({"max_units_by_team":dict(sorted(best.items()))}))'''
    if op == "top_value":
        return '''import csv,json
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
rows.sort(key=lambda r:(-int(r["units"])*int(r["price"]),r["sku"]))
print(json.dumps({"items":[r["sku"] for r in rows[:3]]}))'''
    if op == "category_count":
        return '''import csv,json
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
print(json.dumps({"categories":len({r["category"] for r in rows})}))'''
    if op == "median_price":
        return '''import csv,json,statistics
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
print(json.dumps({"median_price":statistics.median(int(r["price"]) for r in rows)}))'''
    if op == "threshold_count":
        return f'''import csv,json
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
print(json.dumps({{"count":sum(int(r["units"]) >= {params["threshold"]} for r in rows)}}))'''
    if op == "reverse_order":
        return '''import csv,json
rows=list(csv.DictReader(open("input/catalog.csv", newline="")))
print(json.dumps({"items":[r["sku"] for r in reversed(rows)]}))'''
    if op == "word_counts":
        return '''import collections,json,re
words=re.findall(r"[A-Za-z]+",open("input/labels.txt").read().lower())
print(json.dumps({"counts":dict(sorted(collections.Counter(words).items()))}))'''
    if op == "headings":
        return '''import json
lines=open("input/notes.txt").read().splitlines()
print(json.dumps({"headings":[line.lstrip("# ").strip() for line in lines if line.startswith("# ") or line.startswith("## ")]}))'''
    if op == "longest_line":
        return '''import json
lines=open("input/notes.txt").read().splitlines()
line=max(lines,key=lambda x:(len(x),-lines.index(x)))
print(json.dumps({"line":line,"length":len(line)}))'''
    if op == "unique_lines":
        return '''import json
lines=open("input/notes.txt").read().splitlines()
print(json.dumps({"unique_lines":len(set(lines))}))'''
    if op == "occurrence_context":
        return f'''import json
lines=open("input/notes.txt").read().splitlines()
target={params["target"]!r}
index=next(i for i,line in enumerate(lines) if target.lower() in line.lower())
print(json.dumps({{"line_number":index+1,"line":lines[index]}}))'''
    if op == "unique_word_count":
        return '''import json,re
words=re.findall(r"[A-Za-z]+",open("input/labels.txt").read().lower())
print(json.dumps({"count":len(set(words))}))'''
    if op == "sorted_keys":
        return '''import json
data=json.load(open("input/settings.json"))
print(json.dumps({"keys":sorted(data)}))'''
    if op == "unique_tags":
        return '''import json
rows=json.load(open("input/records.json"))
print(json.dumps({"tags":sorted({tag for r in rows for tag in r["tags"]})}))'''
    if op == "group_totals":
        return '''import json
rows=json.load(open("input/records.json")); totals={}
for row in rows: totals[row["group"]]=totals.get(row["group"],0)+row["amount"]
print(json.dumps({"totals":dict(sorted(totals.items()))}))'''
    if op == "active_ids":
        return '''import json
rows=json.load(open("input/records.json"))
print(json.dumps({"ids":sorted(r["id"] for r in rows if r["active"])}))'''
    if op == "project_fields":
        return '''import json
rows=json.load(open("input/records.json"))
print(json.dumps({"records":[{"id":r["id"],"name":r["name"]} for r in rows]}))'''
    if op == "defaults_overlay":
        return '''import json
data=json.load(open("input/config-pair.json"))
result=dict(data["defaults"]); result.update(data["overrides"])
print(json.dumps({"config":result}))'''
    if op == "nested_totals":
        return '''import json
data=json.load(open("input/groups.json"))
print(json.dumps({"totals":{g["name"]:sum(g["values"]) for g in data}}))'''
    if op == "score_order":
        return '''import json
rows=json.load(open("input/records.json")); rows.sort(key=lambda r:(-r["score"],r["id"]))
print(json.dumps({"ids":[r["id"] for r in rows]}))'''
    if op == "missing_fields":
        return '''import json
rows=json.load(open("input/incomplete.json")); required={"id","name","score"}
print(json.dumps({"missing":{r["id"]:sorted(required-set(r)) for r in rows if required-set(r)}}))'''
    if op == "boolean_tally":
        return '''import json
rows=json.load(open("input/records.json")); enabled=sum(bool(r["active"]) for r in rows)
print(json.dumps({"enabled":enabled,"disabled":len(rows)-enabled}))'''
    if op == "log_counts":
        return '''import json
lines=open("input/events.log").read().splitlines()
print(json.dumps({"errors":sum(x.startswith("ERROR ") for x in lines),"warnings":sum(x.startswith("WARN ") for x in lines)}))'''
    if op == "first_error_context":
        return '''import json
lines=open("input/events.log").read().splitlines(); i=next(i for i,x in enumerate(lines) if x.startswith("ERROR "))
print(json.dumps({"before":lines[i-1] if i else None,"error":lines[i],"after":lines[i+1] if i+1<len(lines) else None}))'''
    if op == "first_error_line":
        return '''import json
line=next(x for x in open("input/events.log").read().splitlines() if x.startswith("ERROR "))
print(json.dumps({"line":line}))'''
    if op == "readonly_inventory":
        return '''import json
from pathlib import Path
paths=[Path("input/report.txt"),Path("input/labels.txt")]
print(json.dumps({"files":len(paths),"lines":sum(len(p.read_text().splitlines()) for p in paths)}))'''
    if op == "readonly_report":
        return '''import json
from pathlib import Path
paths=[Path("input/report.txt"),Path("input/notes.txt")]
print(json.dumps({"files":len(paths),"lines":sum(len(p.read_text().splitlines()) for p in paths)}))'''
    if op == "readonly_empty":
        return '''import json
from pathlib import Path
paths=sorted(str(p) for p in Path("input").glob("*.txt") if p.is_file() and p.stat().st_size==0)
print(json.dumps({"empty_files":paths}))'''
    if op == "recover_csv_total":
        return '''import csv,json
from pathlib import Path
paths=sorted(Path("input").glob("*.csv"))
if len(paths)!=1: raise SystemExit("expected one CSV file in input/")
rows=list(csv.DictReader(paths[0].open(newline="")))
print(json.dumps({"revenue":sum(int(r["units"])*int(r["price"]) for r in rows)}))'''
    if op == "recover_log_counts":
        return '''import json
from pathlib import Path
paths=sorted(Path("input").glob("*.log"))
if len(paths)!=1: raise SystemExit("expected one log file in input/")
lines=paths[0].read_text().splitlines()
print(json.dumps({"errors":sum(x.startswith("ERROR ") for x in lines),"warnings":sum(x.startswith("WARN ") for x in lines)}))'''
    raise ValueError(f"unknown curriculum operation: {op}")


def _expected(op: str, material: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    catalog, records = material["catalog"], material["records"]
    if op in {"revenue_total", "recover_csv_total"}:
        return {"revenue": sum(r["units"] * r["price"] for r in catalog)}
    if op == "average_price":
        return {"average_price": sum(r["price"] for r in catalog) / len(catalog)}
    if op == "below_stock":
        return {"items": sorted(r["sku"] for r in catalog if r["units"] < params["threshold"])}
    if op == "name_order":
        return {"items": sorted(r["sku"] for r in catalog)}
    if op == "team_max":
        result = {}
        for r in catalog:
            result[r["team"]] = max(result.get(r["team"], 0), r["units"])
        return {"max_units_by_team": dict(sorted(result.items()))}
    if op == "top_value":
        return {"items": [r["sku"] for r in sorted(catalog, key=lambda r: (-r["units"] * r["price"], r["sku"]))[:3]]}
    if op == "category_count":
        return {"categories": len({r["category"] for r in catalog})}
    if op == "median_price":
        vals = sorted(r["price"] for r in catalog)
        middle = len(vals) // 2
        return {"median_price": vals[middle] if len(vals) % 2 else (vals[middle - 1] + vals[middle]) / 2}
    if op == "threshold_count":
        return {"count": sum(r["units"] >= params["threshold"] for r in catalog)}
    if op == "reverse_order":
        return {"items": [r["sku"] for r in reversed(catalog)]}
    if op == "word_counts":
        counts: dict[str, int] = {}
        for word in re.findall(r"[A-Za-z]+", " ".join(material["labels"]).lower()):
            counts[word] = counts.get(word, 0) + 1
        return {"counts": dict(sorted(counts.items()))}
    if op == "headings":
        return {"headings": [line.lstrip("# ") for line in material["text_lines"] if line.startswith("# ") or line.startswith("## ")]}
    if op == "longest_line":
        line = max(material["text_lines"], key=lambda x: (len(x), -material["text_lines"].index(x)))
        return {"line": line, "length": len(line)}
    if op == "unique_lines":
        return {"unique_lines": len(set(material["text_lines"]))}
    if op == "occurrence_context":
        i = next(i for i, line in enumerate(material["text_lines"]) if params["target"].lower() in line.lower())
        return {"line_number": i + 1, "line": material["text_lines"][i]}
    if op == "unique_word_count":
        return {"count": len(set(re.findall(r"[A-Za-z]+", " ".join(material["labels"]).lower())))}
    if op == "sorted_keys":
        return {"keys": sorted(material["config"])}
    if op == "unique_tags":
        return {"tags": sorted({tag for r in records for tag in r["tags"]})}
    if op == "group_totals":
        totals: dict[str, int] = {}
        for r in records:
            totals[r["group"]] = totals.get(r["group"], 0) + r["amount"]
        return {"totals": dict(sorted(totals.items()))}
    if op == "active_ids":
        return {"ids": sorted(r["id"] for r in records if r["active"])}
    if op == "project_fields":
        return {"records": [{"id": r["id"], "name": r["name"]} for r in records]}
    if op == "defaults_overlay":
        source = material["config_pair"]
        result = dict(source["defaults"])
        result.update(source["overrides"])
        return {"config": result}
    if op == "nested_totals":
        return {"totals": {g["name"]: sum(g["values"]) for g in material["groups"]}}
    if op == "score_order":
        return {"ids": [r["id"] for r in sorted(records, key=lambda r: (-r["score"], r["id"]))]}
    if op == "missing_fields":
        required = {"id", "name", "score"}
        return {"missing": {r["id"]: sorted(required - set(r)) for r in material["incomplete"] if required - set(r)}}
    if op == "boolean_tally":
        enabled = sum(bool(r["active"]) for r in records)
        return {"enabled": enabled, "disabled": len(records) - enabled}
    if op == "log_counts":
        return {"errors": sum(x.startswith("ERROR ") for x in material["logs"]), "warnings": sum(x.startswith("WARN ") for x in material["logs"])}
    if op == "first_error_context":
        i = next(i for i, x in enumerate(material["logs"]) if x.startswith("ERROR "))
        return {"before": material["logs"][i - 1] if i else None, "error": material["logs"][i], "after": material["logs"][i + 1] if i + 1 < len(material["logs"]) else None}
    if op == "first_error_line":
        return {"line": next(x for x in material["logs"] if x.startswith("ERROR "))}
    if op == "readonly_inventory":
        return {"files": 2, "lines": 4}
    if op == "readonly_report":
        return {"files": 2, "lines": 2 + len(material["text_lines"])}
    if op == "readonly_empty":
        return {"empty_files": ["input/empty.txt"]}
    if op == "recover_log_counts":
        return {"errors": sum(x.startswith("ERROR ") for x in material["logs"]), "warnings": sum(x.startswith("WARN ") for x in material["logs"])}
    if op == "help_flag_lookup":
        return {"flag": "--dry-run", "effect": "shows planned changes without modifying files"}
    if op == "help_recovery_recipe":
        return {"steps": ["inspect", "copy", "verify", "report"]}
    raise ValueError(f"unknown curriculum operation: {op}")


_FAMILY_OPS: dict[str, str] = {
    "cli.csv_revenue_total": "revenue_total", "cli.csv_average_price": "average_price",
    "cli.csv_below_stock": "below_stock", "cli.csv_name_order": "name_order",
    "cli.text_word_counts": "word_counts", "cli.text_headings": "headings",
    "cli.json_sorted_keys": "sorted_keys", "cli.json_unique_tags": "unique_tags",
    "cli.json_group_totals": "group_totals", "cli.log_level_counts": "log_counts",
    "cli.readonly_inventory": "readonly_inventory", "cli.recover_missing_csv": "recover_csv_total",
    "cli.csv_team_max": "team_max", "cli.csv_top_value": "top_value",
    "cli.csv_category_count": "category_count", "cli.text_longest_line": "longest_line",
    "cli.text_unique_lines": "unique_lines", "cli.json_active_ids": "active_ids",
    "cli.json_project_fields": "project_fields", "cli.json_defaults_overlay": "defaults_overlay",
    "cli.json_nested_totals": "nested_totals", "cli.log_first_error_context": "first_error_context",
    "cli.english_readonly_report": "readonly_report", "cli.help_flag_lookup": "help_flag_lookup",
    "cli.csv_median_price": "median_price", "cli.english_threshold_count": "threshold_count",
    "cli.csv_reverse_order": "reverse_order", "cli.text_occurrence_context": "occurrence_context",
    "cli.english_unique_word_count": "unique_word_count", "cli.json_score_order": "score_order",
    "cli.json_missing_required_fields": "missing_fields", "cli.english_boolean_tally": "boolean_tally",
    "cli.log_first_error_line": "first_error_line", "cli.readonly_empty_files": "readonly_empty",
    "cli.recover_bad_log_suffix": "recover_log_counts", "cli.help_recovery_recipe": "help_recovery_recipe",
}


def _add_operation_fixture(material: dict[str, Any], op: str) -> None:
    env = material["environment"]
    files = env["files"]
    if op in {"revenue_total", "average_price", "below_stock", "name_order", "team_max", "top_value", "category_count", "median_price", "threshold_count", "reverse_order"}:
        files["input/catalog.csv"] = material["records_csv"]
    if op in {"word_counts", "unique_word_count"}:
        files["input/labels.txt"] = " ".join(material["labels"]) + "\n"
    if op in {"headings", "longest_line", "unique_lines", "occurrence_context"}:
        files["input/notes.txt"] = "\n".join(material["text_lines"]) + "\n"
    if op in {"sorted_keys"}:
        files["input/settings.json"] = canonical_json(material["config"])
    if op in {"unique_tags", "group_totals", "active_ids", "project_fields", "score_order", "boolean_tally"}:
        files["input/records.json"] = canonical_json(material["records"])
    if op == "defaults_overlay":
        pair = {"defaults": {"format": "text", "workers": 2, "retries": material["rng"].randint(0, 5)},
                "overrides": {"format": "json", "workers": material["rng"].randint(3, 12)}}
        material["config_pair"] = pair
        files["input/config-pair.json"] = canonical_json(pair)
    if op == "nested_totals":
        groups = [{"name": f"zone_{name}", "values": [material["rng"].randint(1, 19) for _ in range(3)]} for name in ("a", "b", "c")]
        material["groups"] = groups
        files["input/groups.json"] = canonical_json(groups)
    if op == "missing_fields":
        incomplete = [{"id": "r0", "name": "first", "score": 9}, {"id": "r1", "score": 7},
                      {"id": "r2", "name": "third"}, {"id": "r3", "name": "fourth", "score": 4, "extra": True}]
        material["incomplete"] = incomplete
        files["input/incomplete.json"] = canonical_json(incomplete)
    if op in {"log_counts", "first_error_context", "first_error_line", "recover_log_counts"}:
        files["input/events.log"] = "\n".join(material["logs"]) + "\n"
    if op == "readonly_inventory":
        files["input/report.txt"] = "Summary\nRows reviewed\n"
        files["input/labels.txt"] = "first\nsecond\n"
    if op == "readonly_report":
        files["input/report.txt"] = "Daily report\nNo changes requested\n"
        files["input/notes.txt"] = "\n".join(material["text_lines"]) + "\n"
    if op == "readonly_empty":
        files["input/empty.txt"] = ""
        files["input/message.txt"] = "Nothing to update\n"
    if op == "recover_csv_total":
        files.pop("input/catalog.csv", None)
        files["input/sales_summary.csv"] = material["records_csv"]
    if op == "recover_log_counts":
        files["input/events.log"] = "\n".join(material["logs"]) + "\n"
    if op in {"help_flag_lookup", "help_recovery_recipe"}:
        doc_id = f"cli_guide_{material['token']}"
        if op == "help_flag_lookup":
            content = ("Original fixture help page: weave-clean --help\n"
                       "Usage: weave-clean [OPTIONS] PATH\n"
                       "  --dry-run  shows planned changes without modifying files\n"
                       "  --apply    writes the proposed cleanup\n")
        else:
            content = ("Original fixture recovery guide for weave-recover\n"
                       "Required steps in order: inspect, copy, verify, report.\n"
                       "Do not skip a step; keep source files unchanged until verification.\n")
        files[f"docs/{doc_id}.md"] = content
        env["docs"] = [{"id": doc_id, "title": "weave command help", "content": content}]
        material["doc_id"] = doc_id
        material["doc_content"] = content


def _prompt(op: str, params: dict[str, Any], token: str, doc_id: str | None = None) -> str:
    prompts = {
        "revenue_total": "Read input/catalog.csv and calculate total revenue as units times price for every row. Return only JSON with integer key revenue.",
        "average_price": "Read input/catalog.csv and calculate the arithmetic mean of price. Return only JSON with numeric key average_price.",
        "below_stock": "Read input/catalog.csv and list SKU values whose units are strictly below {threshold}. Sort the names alphabetically. Return only JSON with key items.",
        "name_order": "Read input/catalog.csv and return its SKU values sorted alphabetically, as JSON with key items.",
        "word_counts": "Read input/labels.txt, count case-insensitive words, and return a JSON object with key counts mapping each word to its count.",
        "headings": "Read input/notes.txt and return all Markdown heading text, without leading # marks, in original order under JSON key headings.",
        "sorted_keys": "Read input/settings.json and return its top-level keys in alphabetical order under JSON key keys.",
        "unique_tags": "Read input/records.json and return the unique tags in alphabetical order under JSON key tags.",
        "group_totals": "Read input/records.json and sum amount by group. Return JSON with key totals mapping group names to sums.",
        "log_counts": "Read input/events.log and count lines beginning ERROR and WARN. Return JSON with integer keys errors and warnings.",
        "readonly_inventory": "Read only input/report.txt and input/labels.txt. Do not create, edit, or delete anything. Return JSON with the number of named files and their combined line count, keys files and lines.",
        "recover_csv_total": "The suggested path input/catalog-old.csv may be missing. If it fails, inspect input/ and find its sole CSV. Do not change any files. Calculate revenue (units times price) from that CSV and return JSON with key revenue.",
        "team_max": "Read input/catalog.csv and report the maximum units value for each team. Return JSON with key max_units_by_team.",
        "top_value": "Read input/catalog.csv, rank rows by revenue (units times price) descending, break ties by SKU ascending, and return the top three SKU values under JSON key items.",
        "category_count": "Read input/catalog.csv and return the number of distinct categories as JSON integer key categories.",
        "longest_line": "Read input/notes.txt and return the first longest line and its character length in JSON keys line and length.",
        "unique_lines": "Read input/notes.txt and count distinct complete lines. Return JSON with integer key unique_lines.",
        "active_ids": "Read input/records.json and return the IDs of active records sorted alphabetically under JSON key ids.",
        "project_fields": "Read input/records.json and keep only each record's id and name, preserving input order. Return JSON with key records.",
        "defaults_overlay": "Read input/config-pair.json. Apply overrides over defaults and return the resulting object under JSON key config.",
        "nested_totals": "Read input/groups.json and sum each group's values. Return JSON with key totals mapping each group name to its sum.",
        "first_error_context": "Read input/events.log and find the first ERROR line. Return it plus its immediate previous and next lines, with null for a missing neighbor, under JSON keys before, error, after.",
        "readonly_report": "Read only input/report.txt and input/notes.txt. Do not create, edit, or delete anything. Return a plain English sentence in this form: The report has N lines across 2 named files.",
        "help_flag_lookup": "Use local help lookup for {doc_id}, then read the returned help page. Return JSON with flag --dry-run and its documented effect under keys flag and effect.",
        "median_price": "Read input/catalog.csv and find the median price. Return JSON with numeric key median_price.",
        "threshold_count": "Read input/catalog.csv and count rows with units at least {threshold}. Reply in one plain English sentence: There are N catalog rows with at least {threshold} units.",
        "reverse_order": "Read input/catalog.csv and return SKU values in reverse file order under JSON key items.",
        "occurrence_context": "Read input/notes.txt. Find the first line containing {target!r}, ignoring case. Return its 1-based line number and complete text under JSON keys line_number and line.",
        "unique_word_count": "Read input/labels.txt and count distinct case-insensitive words. Reply exactly: The note has N distinct words.",
        "score_order": "Read input/records.json, order IDs by score descending with ID ascending for ties, and return them under JSON key ids.",
        "missing_fields": "Read input/incomplete.json. For each record, report missing required fields from id, name, score. Return JSON with key missing mapping record IDs to sorted missing-field arrays; omit complete records.",
        "boolean_tally": "Read input/records.json and count true/false values of active. Reply in one plain English sentence: I found N enabled and M disabled records.",
        "first_error_line": "Read input/events.log and return the first line beginning ERROR as JSON string key line.",
        "readonly_empty": "Read only the text files in input/. Do not create, edit, or delete anything. Return the paths of empty text files under JSON key empty_files.",
        "recover_log_counts": "The suggested path input/events.lgo may be misspelled. If it fails, inspect input/, use the sole .log file, and count ERROR and WARN lines. Do not modify files. Return JSON integer keys errors and warnings.",
        "help_recovery_recipe": "Look up the local help/recovery guide {doc_id} and read its required sequence. Return JSON with key steps in the exact documented order.",
    }
    dynamic = {"threshold": params.get("threshold", 0), "target": params.get("target", ""), "doc_id": doc_id or ""}
    return f"CLI curriculum {token}. {prompts[op].format(**dynamic)}"


def generate_task(family: str, seed: int = 0) -> dict[str, Any]:
    """Build one deterministic original task and its unexecuted candidate plan."""
    if family not in FAMILY_SPLITS:
        raise ValueError(f"unknown luna CLI family: {family}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    op = _FAMILY_OPS[family]
    material = _seeded_material(family, seed)
    params: dict[str, Any] = {}
    if op == "below_stock":
        params["threshold"] = material["rng"].randint(4, 12)
    elif op == "threshold_count":
        params["threshold"] = material["rng"].randint(5, 12)
    elif op == "occurrence_context":
        params["target"] = next(word for line in material["text_lines"] for word in _WORDS if word.lower() in line.lower())
    _add_operation_fixture(material, op)
    expected = _expected(op, material, params)
    prompt = _prompt(op, params, material["token"], material.get("doc_id"))
    oracle: dict[str, Any]
    finalizer: dict[str, Any]
    if op in {"readonly_report", "threshold_count", "unique_word_count", "boolean_tally"}:
        oracle = {"kind": "text_exact", "expected": _english_expected(op, expected, params)}
        finalizer = {"kind": "english", "operation": op, "threshold": params.get("threshold")}
    else:
        oracle = {"kind": "json_exact", "expected": expected}
        if op == "help_flag_lookup":
            finalizer = {"kind": "help_flag", "flag": "--dry-run"}
        elif op == "help_recovery_recipe":
            finalizer = {"kind": "help_steps"}
        else:
            finalizer = {"kind": "json_stdout"}
    plan: list[dict[str, Any]] = []
    if op == "recover_csv_total":
        plan = [{"name": "bash", "arguments": {"command": "cat input/catalog-old.csv"}, "expect_error": True},
                {"name": "python", "arguments": {"code": _script(op, params)}}]
    elif op == "recover_log_counts":
        plan = [{"name": "bash", "arguments": {"command": "cat input/events.lgo"}, "expect_error": True},
                {"name": "python", "arguments": {"code": _script(op, params)}}]
    elif op in {"help_flag_lookup", "help_recovery_recipe"}:
        doc_id = material["doc_id"]
        plan = [{"name": "search", "arguments": {"query": doc_id}},
                {"name": "bash", "arguments": {"command": f"cat docs/{doc_id}.md"}}]
    else:
        plan = [{"name": "python", "arguments": {"code": _script(op, params)}}]
    task = {"schema_version": SCHEMA_VERSION, "task_id": f"{family}:{seed:08d}", "family": family,
            "template_id": family + ".v1", "domain": "cli", "split": FAMILY_SPLITS[family], "seed": seed,
            "prompt": prompt, "environment": material["environment"], "oracle": oracle,
            "reference": {"plan": plan, "finalizer": finalizer,
                          "final": _english_expected(op, expected, params) if oracle["kind"] == "text_exact" else canonical_json(expected)},
            "provenance": {"source": "original_procedural", "benchmark": False,
                           "generator_version": GENERATOR_VERSION,
                           "origin": "Hand-authored original CLI and file-work curriculum; no benchmark or external task text"}}
    task["input_sha256"] = content_hash({"prompt": task["prompt"], "environment": task["environment"]})
    validate_task(task)
    return task


def _english_expected(op: str, answer: dict[str, Any], params: dict[str, Any]) -> str:
    if op == "readonly_report":
        return f"The report has {answer['lines']} lines across 2 named files."
    if op == "threshold_count":
        return f"There are {answer['count']} catalog rows with at least {params['threshold']} units."
    if op == "unique_word_count":
        return f"The note has {answer['count']} distinct words."
    if op == "boolean_tally":
        return f"I found {answer['enabled']} enabled and {answer['disabled']} disabled records."
    raise ValueError(f"no English final for {op}")


def generate_tasks(*, seeds_per_family: int = 1, seed_start: int = 0,
                   seeds_by_split: dict[str, int] | None = None,
                   splits: tuple[str, ...] = ("train", "dev", "test")) -> list[dict[str, Any]]:
    """Materialize a modest dataset; use iter_tasks() for large batches."""
    return list(iter_tasks(seed_start=seed_start, seeds_per_family=seeds_per_family,
                           seeds_by_split=seeds_by_split, splits=splits))


class LunaCLITeacher:
    """Callback for collect_task: request planned tools, consume only observed outputs.

    It never reads the oracle or reference final. Tool calls are executed and
    receipts supplied by collect_task's shared AgentHarness + ContainerSandbox.
    """
    def __init__(self, task: dict[str, Any]):
        self.plan = copy.deepcopy(task["reference"]["plan"])
        self.finalizer = copy.deepcopy(task["reference"]["finalizer"])
        self.step = 0

    def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if messages and messages[-1].get("role") == "tool":
            observed = json.loads(messages[-1]["content"])
            previous = self.plan[self.step - 1]
            failed = ("error" in observed or observed.get("exit_code", 0) != 0
                      or observed.get("timed_out") or observed.get("truncated"))
            if previous.get("expect_error"):
                if not failed:
                    return {"role": "assistant", "content": "The expected file lookup did not fail, so I stopped rather than assuming the recovery path."}
            elif failed:
                return {"role": "assistant", "content": "I couldn’t complete the requested check because a tool step failed."}
        if self.step < len(self.plan):
            action = self.plan[self.step]
            self.step += 1
            args = copy.deepcopy(action["arguments"])
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"luna_cli_{self.step}", "type": "function",
                "function": {"name": action["name"], "arguments": canonical_json(args)}}]}
        observations = [json.loads(message["content"]) for message in messages if message.get("role") == "tool"]
        return {"role": "assistant", "content": _final_from_observations(self.finalizer, observations)}


def _final_from_observations(finalizer: dict[str, Any], observations: list[dict[str, Any]]) -> str:
    """Produce a final answer from harness-observed stdout/search results only."""
    if not observations:
        return "I couldn’t finish because there was no observed tool result."
    kind = finalizer["kind"]
    if kind == "json_stdout":
        stdout = observations[-1].get("stdout", "").strip()
        try:
            return canonical_json(json.loads(stdout))
        except (ValueError, TypeError):
            return stdout
    if kind == "english":
        values = json.loads(observations[-1].get("stdout", ""))
        op = finalizer["operation"]
        if op == "readonly_report":
            return f"The report has {values['lines']} lines across 2 named files."
        if op == "threshold_count":
            return f"There are {values['count']} catalog rows with at least {finalizer['threshold']} units."
        if op == "unique_word_count":
            return f"The note has {values['count']} distinct words."
        if op == "boolean_tally":
            return f"I found {values['enabled']} enabled and {values['disabled']} disabled records."
    if kind in {"help_flag", "help_steps"}:
        manual = next((row.get("stdout", "") for row in reversed(observations) if "stdout" in row), "")
        if kind == "help_flag":
            flag = finalizer["flag"]
            match = re.search(rf"^\s*{re.escape(flag)}\s+(.+)$", manual, re.MULTILINE)
            if not match:
                return "I couldn’t find that flag in the observed help text."
            return canonical_json({"flag": flag, "effect": match.group(1).strip()})
        match = re.search(r"Required steps in order:\s*([a-z, ]+)\.", manual)
        if not match:
            return "I couldn’t find an ordered recovery recipe in the observed guide."
        return canonical_json({"steps": [step.strip() for step in match.group(1).split(",")]})
    return "I couldn’t interpret the observed tool result."


def independent_expected_from_fixtures(task: dict[str, Any]) -> Any:
    """Recompute an answer from the frozen fixture files, never oracle.expected."""
    family = task["family"]
    op = _FAMILY_OPS[family]
    files = task["environment"]["files"]

    def text(path: str) -> str:
        return files[path]

    def csv_rows(path: str = "input/catalog.csv") -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(text(path))))

    def json_value(path: str) -> Any:
        return json.loads(text(path))

    if op in {"revenue_total", "average_price", "below_stock", "name_order", "team_max", "top_value", "category_count", "median_price", "threshold_count", "reverse_order"}:
        rows = csv_rows()
        if op == "revenue_total":
            return {"revenue": sum(int(r["units"]) * int(r["price"]) for r in rows)}
        if op == "average_price":
            return {"average_price": sum(int(r["price"]) for r in rows) / len(rows)}
        if op == "below_stock":
            threshold = int(re.search(r"strictly below ([0-9]+)", task["prompt"]).group(1))
            return {"items": sorted(r["sku"] for r in rows if int(r["units"]) < threshold)}
        if op == "name_order":
            return {"items": sorted(r["sku"] for r in rows)}
        if op == "team_max":
            result: dict[str, int] = {}
            for row in rows:
                result[row["team"]] = max(result.get(row["team"], 0), int(row["units"]))
            return {"max_units_by_team": dict(sorted(result.items()))}
        if op == "top_value":
            rows.sort(key=lambda r: (-int(r["units"]) * int(r["price"]), r["sku"]))
            return {"items": [r["sku"] for r in rows[:3]]}
        if op == "category_count":
            return {"categories": len({r["category"] for r in rows})}
        if op == "median_price":
            values = sorted(int(r["price"]) for r in rows)
            n = len(values)
            return {"median_price": values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2}
        if op == "threshold_count":
            threshold = int(re.search(r"units at least ([0-9]+)", task["prompt"]).group(1))
            return {"count": sum(int(r["units"]) >= threshold for r in rows)}
        return {"items": [r["sku"] for r in reversed(rows)]}

    if op in {"word_counts", "unique_word_count"}:
        words = re.findall(r"[A-Za-z]+", text("input/labels.txt").lower())
        if op == "unique_word_count":
            return {"count": len(set(words))}
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        return {"counts": dict(sorted(counts.items()))}
    if op in {"headings", "longest_line", "unique_lines", "occurrence_context"}:
        lines = text("input/notes.txt").splitlines()
        if op == "headings":
            return {"headings": [line.lstrip("# ").strip() for line in lines if line.startswith("# ") or line.startswith("## ")]}
        if op == "longest_line":
            line = max(lines, key=lambda x: (len(x), -lines.index(x)))
            return {"line": line, "length": len(line)}
        if op == "unique_lines":
            return {"unique_lines": len(set(lines))}
        target = re.search(r"first line containing '([^']+)'", task["prompt"]).group(1)
        index = next(i for i, line in enumerate(lines) if target.lower() in line.lower())
        return {"line_number": index + 1, "line": lines[index]}
    if op == "sorted_keys":
        return {"keys": sorted(json_value("input/settings.json"))}
    if op == "unique_tags":
        rows = json_value("input/records.json")
        return {"tags": sorted({tag for row in rows for tag in row["tags"]})}
    if op == "group_totals":
        totals: dict[str, int] = {}
        for row in json_value("input/records.json"):
            totals[row["group"]] = totals.get(row["group"], 0) + row["amount"]
        return {"totals": dict(sorted(totals.items()))}
    if op in {"active_ids", "project_fields", "score_order", "boolean_tally"}:
        rows = json_value("input/records.json")
        if op == "active_ids":
            return {"ids": sorted(r["id"] for r in rows if r["active"])}
        if op == "project_fields":
            return {"records": [{"id": r["id"], "name": r["name"]} for r in rows]}
        if op == "score_order":
            return {"ids": [r["id"] for r in sorted(rows, key=lambda r: (-r["score"], r["id"]))]}
        active = sum(bool(r["active"]) for r in rows)
        return {"enabled": active, "disabled": len(rows) - active}
    if op == "defaults_overlay":
        data = json_value("input/config-pair.json")
        merged = dict(data["defaults"])
        merged.update(data["overrides"])
        return {"config": merged}
    if op == "nested_totals":
        return {"totals": {row["name"]: sum(row["values"]) for row in json_value("input/groups.json")}}
    if op == "missing_fields":
        required = {"id", "name", "score"}
        return {"missing": {row["id"]: sorted(required - set(row)) for row in json_value("input/incomplete.json") if required - set(row)}}
    if op in {"log_counts", "first_error_context", "first_error_line", "recover_log_counts"}:
        path = "input/events.log" if "input/events.log" in files else next(p for p in files if p.startswith("input/") and p.endswith(".log"))
        lines = text(path).splitlines()
        if op in {"log_counts", "recover_log_counts"}:
            return {"errors": sum(x.startswith("ERROR ") for x in lines), "warnings": sum(x.startswith("WARN ") for x in lines)}
        index = next(i for i, line in enumerate(lines) if line.startswith("ERROR "))
        if op == "first_error_line":
            return {"line": lines[index]}
        return {"before": lines[index - 1] if index else None, "error": lines[index], "after": lines[index + 1] if index + 1 < len(lines) else None}
    if op == "readonly_inventory":
        paths = ("input/report.txt", "input/labels.txt")
        return {"files": len(paths), "lines": sum(len(text(path).splitlines()) for path in paths)}
    if op == "readonly_report":
        paths = ("input/report.txt", "input/notes.txt")
        return {"files": len(paths), "lines": sum(len(text(path).splitlines()) for path in paths)}
    if op == "readonly_empty":
        return {"empty_files": sorted(path for path, value in files.items() if path.startswith("input/") and path.endswith(".txt") and value == "")}
    if op == "recover_csv_total":
        paths = sorted(path for path in files if path.startswith("input/") and path.endswith(".csv"))
        if len(paths) != 1:
            raise ValueError("frozen recovery task must contain exactly one CSV")
        rows = csv_rows(paths[0])
        return {"revenue": sum(int(r["units"]) * int(r["price"]) for r in rows)}
    if op == "help_flag_lookup":
        manual = next(value for path, value in files.items() if path.startswith("docs/"))
        effect = re.search(r"^\s*--dry-run\s+(.+)$", manual, re.MULTILINE).group(1).strip()
        return {"flag": "--dry-run", "effect": effect}
    if op == "help_recovery_recipe":
        manual = next(value for path, value in files.items() if path.startswith("docs/"))
        steps = re.search(r"Required steps in order:\s*([a-z, ]+)\.", manual).group(1)
        return {"steps": [step.strip() for step in steps.split(",")]}
    raise ValueError(f"no independent oracle for {family}")


def independent_oracle_check(task: dict[str, Any], final_content: str) -> dict[str, Any]:
    """Pure check against fixture-recomputed expected output, independent of teacher."""
    expected = independent_expected_from_fixtures(task)
    if task["oracle"]["kind"] == "text_exact":
        op = _FAMILY_OPS[task["family"]]
        if op == "readonly_report":
            expected_text = f"The report has {expected['lines']} lines across 2 named files."
        elif op == "threshold_count":
            threshold = int(re.search(r"units at least ([0-9]+)", task["prompt"]).group(1))
            expected_text = f"There are {expected['count']} catalog rows with at least {threshold} units."
        elif op == "unique_word_count":
            expected_text = f"The note has {expected['count']} distinct words."
        elif op == "boolean_tally":
            expected_text = f"I found {expected['enabled']} enabled and {expected['disabled']} disabled records."
        else:
            raise ValueError("unknown text oracle")
        passed = final_content.strip() == expected_text
    else:
        try:
            passed = canonical_json(json.loads(final_content)) == canonical_json(expected)
        except (ValueError, TypeError):
            passed = False
    return {"passed": passed, "oracle": "fixture_recomputed_independent_v1",
            "check": task["oracle"]["kind"], "failure": None if passed else "observed final differs from fixture-derived result"}


def _validate_trusted_teacher_code(code: str) -> None:
    """Fail closed before running any reviewed procedural fixture code locally."""
    tree = ast.parse(code)
    allowed_imports = {"csv", "json", "statistics", "collections", "re", "pathlib"}
    forbidden_names = {"eval", "exec", "compile", "__import__", "breakpoint", "input", "globals", "locals"}
    mutators = {"write_text", "write_bytes", "unlink", "mkdir", "rmdir", "rename", "replace", "touch", "chmod", "symlink_to", "hardlink_to"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in allowed_imports for alias in node.names):
                raise ValueError("native teacher fixture attempted a non-allowlisted import")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in allowed_imports:
                raise ValueError("native teacher fixture attempted a non-allowlisted import")
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            raise ValueError("native teacher fixture contains a dynamic-code or interactive primitive")
        elif isinstance(node, ast.Attribute) and node.attr in mutators:
            raise ValueError("native teacher fixture attempted a file mutation")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                if len(node.args) > 1 or any(arg.arg in {"mode", "encoding"} and arg.arg == "mode" for arg in node.keywords):
                    raise ValueError("native teacher fixture may only open files read-only")
                if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise ValueError("native teacher fixture may only open a literal task-relative path")
                path = node.args[0].value
                if path.startswith("/") or ".." in Path(path).parts or "\\" in path:
                    raise ValueError("native teacher fixture path escapes its task workspace")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "open":
                if node.args or any(arg.arg == "mode" for arg in node.keywords):
                    raise ValueError("native teacher fixture may only open files read-only")


def _capture_source_snapshot(output_dir: str | Path) -> str:
    """Preserve exact source bytes for reproducible review of this batch."""
    from picoagent.harness.tools import TOOL_SCHEMAS, _PYTHON_RUNNER

    output = Path(output_dir)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    paths = (
        "src/picoagent/data/luna_cli_curriculum.py", "src/picoagent/data/generators.py",
        "src/picoagent/data/schema.py", "src/picoagent/data/oracles.py", "src/picoagent/data/audit.py",
        "src/picoagent/data/collector.py", "src/picoagent/harness/agent.py", "src/picoagent/harness/tools.py",
        "src/picoagent/harness/sandbox.py", "src/picoagent/harness/knowledge.py",
    )
    manifest_files = {}
    for relative in paths:
        source = repo_root / relative
        data = source.read_bytes()
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError("source snapshot is immutable and differs from current sources")
        else:
            target.write_bytes(data)
        manifest_files[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    protocol = {"tool_schemas": TOOL_SCHEMAS, "tool_schemas_sha256": content_hash(TOOL_SCHEMAS),
                "python_runner_source": _PYTHON_RUNNER,
                "python_runner_sha256": hashlib.sha256(_PYTHON_RUNNER.encode("utf-8")).hexdigest(),
                "python_runner_bytes": len(_PYTHON_RUNNER.encode("utf-8"))}
    protocol_path = snapshot / "tool_protocol.json"
    protocol_bytes = (canonical_json(protocol) + "\n").encode("utf-8")
    if protocol_path.exists():
        if protocol_path.read_bytes() != protocol_bytes:
            raise ValueError("tool protocol source snapshot is immutable and differs")
    else:
        protocol_path.write_bytes(protocol_bytes)
    manifest_files["tool_protocol.json"] = {"sha256": hashlib.sha256(protocol_bytes).hexdigest(), "bytes": len(protocol_bytes)}
    manifest = {"schema": "picoagent.cli_source_snapshot.v1", "generator_version": GENERATOR_VERSION,
                "luna_curriculum_version": CURRICULUM_VERSION, "files": manifest_files}
    manifest_path = snapshot / "manifest.json"
    serialized = (canonical_json(manifest) + "\n").encode("utf-8")
    if manifest_path.exists():
        if manifest_path.read_bytes() != serialized:
            raise ValueError("source snapshot manifest is immutable and differs")
    else:
        manifest_path.write_bytes(serialized)
    return hashlib.sha256(serialized).hexdigest()


def run_native_teacher_observed(tasks: list[dict[str, Any]], output_dir: str | Path, *,
                               permitted_splits: tuple[str, ...] = ("train", "dev"), event_sink=None,
                               capture_sources: bool = True,
                               source_snapshot_sha256: str | None = None) -> dict[str, Any]:
    """Run only reviewed authored teacher plans in isolated per-task exec dirs.

    This is diagnostic native-teacher evidence, never a ContainerSandbox rollout
    or training trace. It is deliberately opt-in and separately manifested.
    """
    from picoagent.data.collector import LocalCorpusSearch
    from picoagent.harness.tools import TOOL_SCHEMAS, _PYTHON_RUNNER

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    if capture_sources:
        source_snapshot_sha256 = _capture_source_snapshot(destination)
    records: list[dict[str, Any]] = []
    source_file = Path(__file__).resolve()
    source_sha = file_hash(source_file)
    bash_executable = shutil.which("bash")
    if not bash_executable:
        raise RuntimeError("bash is required for the reviewed local fixture observation pass")
    bash_probe = subprocess.run([bash_executable, "--version"], capture_output=True, timeout=5, check=False,
                                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C.UTF-8"})
    bash_version = bash_probe.stdout.decode("utf-8", errors="replace").splitlines()[0]
    bash_path = Path(bash_executable).resolve()
    python_path = Path(sys.executable).resolve()
    for task in tasks:
        validate_task(task)
        if task["split"] not in permitted_splits:
            raise ValueError(f"native observation is not authorized for split {task['split']}")
        if task["task_id"] in EVALUATION_EXCLUSIONS:
            raise ValueError(f"task instance is excluded from sealed evaluation: {task['task_id']}")
        if task["family"] not in FAMILY_SPLITS or canonical_json(task) != canonical_json(generate_task(task["family"], task["seed"])):
            raise ValueError("native teacher can run only this module's hand-authored fixtures")
        call_tag = hashlib.sha256(task["task_id"].encode()).hexdigest()[:12]
        start_time = time.time_ns()
        teacher = LunaCLITeacher(task)
        search = LocalCorpusSearch(task["environment"]["docs"])
        assistant_events: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        transcript = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task["prompt"]}]
        with tempfile.TemporaryDirectory(prefix="luna-cli-teacher-") as temp_dir:
            workspace = Path(temp_dir).resolve()
            for relative, contents in task["environment"]["files"].items():
                from picoagent.harness.sandbox import safe_relative_path
                rel = safe_relative_path(relative)
                target = workspace.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(contents, encoding="utf-8")
            before = {path: hashlib.sha256((workspace / path).read_bytes()).hexdigest()
                      for path in sorted(task["environment"]["files"])}
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": temp_dir,
                   "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}
            final = ""
            stop_reason = "max_turns"
            for turn in range(32):
                input_messages = copy.deepcopy(transcript)
                response = teacher(input_messages, TOOL_SCHEMAS)
                assistant_events.append({"event_id": f"{call_tag}_assistant_{turn}",
                                         "input_messages": input_messages, "message": copy.deepcopy(response),
                                         "producer": "luna_cli_procedural_fixture_teacher"})
                if event_sink is not None:
                    event_sink("teacher_response", assistant_events[-1])
                transcript.append(copy.deepcopy(response))
                calls = response.get("tool_calls", [])
                if not calls:
                    final = response.get("content") or ""
                    stop_reason = "final"
                    break
                if len(calls) != 1:
                    raise ValueError("trusted teacher must request one tool action per turn")
                call = calls[0]
                name = call["function"]["name"]
                args_json = call["function"]["arguments"]
                args = json.loads(args_json)
                invocation: dict[str, Any] = {"tool_name": name, "tool_call_id": call["id"],
                                              "arguments_json": args_json, "arguments": args}
                execution: dict[str, Any]
                if name == "search":
                    result = search.search(args["query"], limit=args.get("limit", 5))
                    execution = {"backend": "local_fixture_search", "argv": None, "stdin_base64": None,
                                 "cwd": None, "environment": {"network": "disabled; local fixture search only"},
                                 "stdout_base64": None, "stderr_base64": None, "exit_code": 0,
                                 "timed_out": False, "truncated": False, "duration_ns": 0}
                elif name == "bash":
                    command = args["command"]
                    if not re.fullmatch(r"cat (?:input/(?:catalog-old\.csv|events\.lgo)|docs/[a-z0-9_]+\.md)", command):
                        raise ValueError("native teacher fixture bash command is outside its reviewed cat-only allowlist")
                    started = time.monotonic_ns()
                    try:
                        proc = subprocess.run(["bash", "--noprofile", "--norc", "-c", command],
                                              cwd=workspace, env=env, capture_output=True, timeout=15, check=False)
                        out, err, code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
                    except subprocess.TimeoutExpired as exc:
                        out = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode()
                        err = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode()
                        code, timed_out = 124, True
                    execution = {"backend": "platform_exec_command_per_task_temp_workspace", "argv": ["bash", "--noprofile", "--norc", "-c", command],
                                 "stdin_base64": base64.b64encode(b"").decode("ascii"), "cwd": str(workspace),
                                 "environment": dict(env), "stdout_base64": base64.b64encode(out).decode("ascii"),
                                 "stderr_base64": base64.b64encode(err).decode("ascii"), "exit_code": code,
                                 "timed_out": timed_out, "truncated": False, "duration_ns": time.monotonic_ns() - started}
                    result = {"stdout": out.decode("utf-8", errors="replace"), "stderr": err.decode("utf-8", errors="replace"),
                              "exit_code": code, "timed_out": timed_out, "truncated": False,
                              "backend": "native_teacher_fixture"}
                elif name == "python":
                    code_text = args["code"]
                    _validate_trusted_teacher_code(code_text)
                    stdin_bytes = code_text.encode("utf-8")
                    command_argv = [sys.executable, "-I", "-c", _PYTHON_RUNNER]
                    started = time.monotonic_ns()
                    try:
                        proc = subprocess.run(command_argv, cwd=workspace, env=env, input=stdin_bytes,
                                              capture_output=True, timeout=15, check=False)
                        out, err, exit_code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
                    except subprocess.TimeoutExpired as exc:
                        out = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode()
                        err = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode()
                        exit_code, timed_out = 124, True
                    execution = {"backend": "platform_exec_command_per_task_temp_workspace", "argv": command_argv,
                                 "stdin_base64": base64.b64encode(stdin_bytes).decode("ascii"), "cwd": str(workspace),
                                 "environment": dict(env), "stdout_base64": base64.b64encode(out).decode("ascii"),
                                 "stderr_base64": base64.b64encode(err).decode("ascii"), "exit_code": exit_code,
                                 "timed_out": timed_out, "truncated": False, "duration_ns": time.monotonic_ns() - started}
                    result = {"stdout": out.decode("utf-8", errors="replace"), "stderr": err.decode("utf-8", errors="replace"),
                              "exit_code": exit_code, "timed_out": timed_out, "truncated": False,
                              "backend": "native_teacher_fixture"}
                else:
                    raise ValueError(f"unexpected tool in native teacher plan: {name}")
                if name != "search" and (len(execution["stdout_base64"]) > 131072 or len(execution["stderr_base64"]) > 131072):
                    raise ValueError("native teacher fixture output exceeded recording bound")
                observations.append(copy.deepcopy(result))
                tool_result_msg = {"role": "tool", "name": name, "tool_call_id": call["id"], "content": canonical_json(result)}
                transcript.append(tool_result_msg)
                tool_events.append({"tool_call_id": call["id"], "name": name, "arguments_json": args_json,
                                    "arguments": args, "result": copy.deepcopy(result), "invocation": invocation,
                                    "execution": execution})
                if event_sink is not None:
                    event_sink("tool_execution", tool_events[-1])
            after = {path: hashlib.sha256((workspace / path).read_bytes()).hexdigest()
                     for path in sorted(task["environment"]["files"])}
        task_sha = content_hash(task)
        pure_oracle = independent_oracle_check(task, final) if stop_reason == "final" else {"passed": False, "oracle": "fixture_recomputed_independent_v1", "check": task["oracle"]["kind"], "failure": "no final response"}
        shared_oracle = check_task_result(task, final, artifacts={}, kv={}) if stop_reason == "final" else {"passed": False, "checks": [], "failures": ["no final response"]}
        if pure_oracle["passed"] != shared_oracle["passed"]:
            raise ValueError(f"independent and shared oracles disagree for {task['task_id']}")
        record = {"schema": "picoagent.cli_native_teacher_observed.v1", "execution_kind": "native_teacher_observed",
                  "not_a_trace": True, "sft_admissible": False, "requires_container_replay": True,
                  "status": "teacher_fixture_observed" if pure_oracle["passed"] else "teacher_fixture_failed",
                  "task_id": task["task_id"], "task_sha256": task_sha,
                  "candidate_plan_sha256": content_hash(task["reference"]["plan"]), "source_module_sha256": source_sha,
                  "family": task["family"], "template_id": task["template_id"], "split": task["split"],
                  "teacher_identity": "luna_cli_procedural_fixture_teacher", "model_identity": None,
                  "provider_generation": None, "started_at_unix_ns": start_time,
                  "source_snapshot_manifest_sha256": source_snapshot_sha256,
                  "system_prompt_sha256": content_hash(SYSTEM_PROMPT), "tool_schemas_sha256": content_hash(TOOL_SCHEMAS),
                  "transcript": transcript, "teacher_events": assistant_events, "tool_events": tool_events,
                  "native_execution_metadata": {"backend": "platform_exec_command_per_task_temp_workspace",
                      "container_id": None, "runtime": None, "python_executable": sys.executable,
                      "python_executable_sha256": file_hash(python_path), "python_version": platform.python_version(),
                      "bash_executable": str(bash_path), "bash_executable_sha256": file_hash(bash_path),
                      "bash_version": bash_version,
                      "bash_version_probe": {"argv": [bash_executable, "--version"],
                          "stdout_base64": base64.b64encode(bash_probe.stdout).decode("ascii"),
                          "stderr_base64": base64.b64encode(bash_probe.stderr).decode("ascii"),
                          "exit_code": bash_probe.returncode, "timeout_seconds": 5},
                      "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
                      "locale": locale.setlocale(locale.LC_ALL, None), "outer_exec_receipt_id": None,
                      "environment_policy": {"PATH": "inherited executable search path only", "HOME": "per-task temporary workspace",
                          "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8",
                          "network": "not invoked; teacher code import/call allowlist excludes network modules"},
                      "workspace_lifecycle": "one TemporaryDirectory per task; removed on completion"},
                  "fixture_sha256_before": before, "fixture_sha256_after": after,
                  "fixture_unchanged": before == after, "artifact_poststate": {}, "kv_poststate": {},
                  "final_response": final, "stop_reason": stop_reason,
                  "oracle_checks": {"independent": pure_oracle, "shared": shared_oracle}}
        records.append(record)
        if event_sink is not None:
            event_sink("attempt_completed", {"task_id": task["task_id"], "task_sha256": task_sha,
                                               "status": record["status"], "final_response": final,
                                               "oracle_checks": record["oracle_checks"]})

    path = destination / "observations.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    manifest = {"schema": "picoagent.cli_native_teacher_manifest.v1", "execution_kind": "native_teacher_observed",
                "records": len(records), "file_sha256": file_hash(path), "bytes": path.stat().st_size,
                "task_hashes": {record["task_id"]: record["task_sha256"] for record in records},
                "admission": "not admitted; this artifact is a deterministic teacher fixture replay and every task still requires real ContainerSandbox collection"}
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {"observations": str(path), "manifest": str(manifest_path), "records": len(records),
            "passed": sum(record["oracle_checks"]["independent"]["passed"] for record in records),
            "failed": sum(not record["oracle_checks"]["independent"]["passed"] for record in records)}


def iter_tasks(*, seed_start: int = 0, seeds_per_family: int = 1,
               seeds_by_split: dict[str, int] | None = None,
               splits: tuple[str, ...] = ("train", "dev", "test")):
    """Yield deterministic seeded variants without materializing a large batch."""
    if seed_start < 0 or seeds_per_family < 1:
        raise ValueError("seed_start must be nonnegative and seeds_per_family positive")
    counts = {split: seeds_per_family for split in ("train", "dev", "test")}
    if seeds_by_split is not None:
        unknown = set(seeds_by_split) - set(counts)
        if unknown or any(type(value) is not int or value < 1 for value in seeds_by_split.values()):
            raise ValueError("seeds_by_split must contain positive counts for train/dev/test")
        counts.update(seeds_by_split)
    if not set(splits) <= set(counts):
        raise ValueError("unknown split requested")
    for family in sorted(FAMILY_SPLITS):
        split = FAMILY_SPLITS[family]
        if split in splits:
            for seed in range(seed_start, seed_start + counts[split]):
                yield generate_task(family, seed)


def run_native_teacher_observed_batch(tasks, output_dir: str | Path, *,
                                      permitted_splits: tuple[str, ...] = ("train", "dev"),
                                      retry_failed: bool = False) -> dict[str, Any]:
    """Append-only, resumable per-task replays; never accumulates the batch in RAM.

    Each attempt has a hash-chained incremental event journal. A restarted
    process skips successfully recorded tasks; failed completed tasks are
    retried only when retry_failed=True, and earlier attempt directories remain.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    attempts_root = output / "attempts"
    attempts_root.mkdir(exist_ok=True)
    index_path = output / "attempt_index.jsonl"
    existing = read_jsonl(index_path) if index_path.exists() and index_path.stat().st_size else []
    previous_index_hash = "0" * 64
    for number, row in enumerate(existing):
        claimed = row.get("sha256")
        unsigned = {key: value for key, value in row.items() if key != "sha256"}
        if row.get("sequence") != number or row.get("previous_sha256") != previous_index_hash or content_hash(unsigned) != claimed:
            raise ValueError("native attempt index hash chain is invalid")
        previous_index_hash = claimed
    source_snapshot_sha256 = _capture_source_snapshot(output)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        by_task.setdefault(row["task_id"], []).append(row)
    completed = skipped = passed = failed = 0
    for task in tasks:
        validate_task(task)
        if task["split"] not in permitted_splits:
            raise ValueError(f"native observation is not authorized for split {task['split']}")
        if task["task_id"] in EVALUATION_EXCLUSIONS:
            raise ValueError(f"task instance is excluded from sealed evaluation: {task['task_id']}")
        expected_sha = content_hash(task)
        prior = by_task.get(task["task_id"], [])
        if any(row["task_sha256"] != expected_sha for row in prior):
            raise ValueError("resume index task ID is bound to different task content")
        latest = prior[-1] if prior else None
        if latest and (latest["status"] == "teacher_fixture_observed" or not retry_failed):
            skipped += 1
            continue
        group_name = hashlib.sha256(task["task_id"].encode()).hexdigest()[:16]
        group = attempts_root / group_name
        group.mkdir(exist_ok=True)
        attempt_number = len(prior) + 1
        attempt_dir = group / f"attempt-{attempt_number:04d}"
        while attempt_dir.exists():
            attempt_number += 1
            attempt_dir = group / f"attempt-{attempt_number:04d}"
        attempt_dir.mkdir()
        journal_path = attempt_dir / "events.jsonl"
        previous = "0" * 64
        sequence = 0

        def journal(kind: str, payload: Any) -> None:
            nonlocal previous, sequence
            row = {"sequence": sequence, "kind": kind, "payload": payload, "previous_sha256": previous}
            row["sha256"] = content_hash(row)
            with journal_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            previous = row["sha256"]
            sequence += 1

        journal("attempt_started", {"task_id": task["task_id"], "task_sha256": expected_sha,
                                    "family": task["family"], "split": task["split"],
                                    "candidate_plan_sha256": content_hash(task["reference"]["plan"])})
        status = "error"
        result_manifest = None
        error = None
        try:
            result_manifest = run_native_teacher_observed([task], attempt_dir / "result",
                                                          permitted_splits=permitted_splits, event_sink=journal,
                                                          capture_sources=False,
                                                          source_snapshot_sha256=source_snapshot_sha256)
            observed_path = Path(result_manifest["observations"])
            record = read_jsonl(observed_path)[0]
            status = record["status"]
            passed += int(status == "teacher_fixture_observed")
            failed += int(status != "teacher_fixture_observed")
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            journal("runner_exception", error)
            failed += 1
        attempt_manifest = {"schema": "picoagent.cli_native_attempt_manifest.v1", "task_id": task["task_id"],
                            "task_sha256": expected_sha, "attempt_number": attempt_number,
                            "status": status, "event_count": sequence, "event_chain_sha256": previous,
                            "journal_sha256": file_hash(journal_path), "error": error,
                            "result_manifest": result_manifest}
        attempt_manifest_path = attempt_dir / "attempt_manifest.json"
        attempt_manifest_path.write_text(canonical_json(attempt_manifest) + "\n", encoding="utf-8")
        index_row = {"sequence": len(existing), "previous_sha256": previous_index_hash,
                     "task_id": task["task_id"], "task_sha256": expected_sha,
                     "family": task["family"], "split": task["split"], "attempt_number": attempt_number,
                     "attempt_dir": str(attempt_dir.relative_to(output)), "status": status,
                     "manifest_sha256": file_hash(attempt_manifest_path), "journal_sha256": file_hash(journal_path),
                     "source_snapshot_manifest_sha256": source_snapshot_sha256}
        index_row["sha256"] = content_hash(index_row)
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(index_row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        by_task.setdefault(task["task_id"], []).append(index_row)
        existing.append(index_row)
        previous_index_hash = index_row["sha256"]
        completed += 1
    return {"completed_attempts": completed, "skipped_completed": skipped, "passed": passed, "failed": failed,
            "attempt_index": str(index_path), "append_only": True,
            "recorded_tasks": len(by_task), "sft_admissible": False,
            "permitted_splits": list(permitted_splits)}


def candidate_plan_rows(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Package planned actions without pretending they have been executed."""
    rows = {split: [] for split in ("train", "dev", "test")}
    for task in tasks:
        rows[task["split"]].append({
            "schema": CANDIDATE_SCHEMA, "task_id": task["task_id"], "family": task["family"],
            "template_id": task["template_id"], "split": task["split"],
            "status": "unexecuted_candidate_plan", "execution": "unexecuted",
            "author": "luna_agent_authored_procedural_plan", "not_a_trace": True,
            "planned_actions": copy.deepcopy(task["reference"]["plan"]),
            "note": "No model call, tool stdout/stderr, execution receipt, or success claim is present. Run through collect_task to obtain observed evidence.",
        })
    return rows


def write_luna_cli_dataset(output_dir: str | Path, tasks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write standard tasks/authored views plus a hash-audited unexecuted-plan view."""
    output = Path(output_dir)
    rows = generate_tasks() if tasks is None else tasks
    curriculum_manifest = write_curriculum(output, rows,
        configuration={"dataset": CURRICULUM_VERSION, "families": len(FAMILY_SPLITS),
                       "records_by_split": {split: sum(task["split"] == split for task in rows) for split in ("train", "dev", "test")},
                       "evaluation_exclusions": EVALUATION_EXCLUSIONS},
        split_policy=FAMILY_SPLITS)
    plan_rows = candidate_plan_rows(rows)
    files = {}
    for split, split_rows in plan_rows.items():
        path = output / f"{split}.candidate_plans.unexecuted.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(canonical_json(row) + "\n")
        files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size, "records": len(split_rows), "status": "unexecuted_candidate_plan"}
    plans_manifest = {"schema": "picoagent.cli_candidate_manifest.v1", "curriculum_version": CURRICULUM_VERSION,
                      "files": files, "note": "Candidate action plans only; these are not tool traces or model-written rollouts."}
    plan_manifest_path = output / "candidate_plans.manifest.json"
    plan_manifest_path.write_text(canonical_json(plans_manifest) + "\n", encoding="utf-8")
    exclusions_path = output / "evaluation_exclusions.json"
    exclusions_path.write_text(canonical_json({"schema": "picoagent.evaluation_exclusions.v1",
                                               "exclusions": EVALUATION_EXCLUSIONS}) + "\n", encoding="utf-8")
    return {"curriculum_manifest": str(curriculum_manifest), "candidate_plan_manifest": str(plan_manifest_path),
            "tasks": len(rows), "families": len({task["family"] for task in rows}),
            "splits": {split: sum(task["split"] == split for task in rows) for split in ("train", "dev", "test")}}


def verify_luna_cli_dataset(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    curriculum = verify_curriculum(output / "manifest.json")
    manifest = json.loads((output / "candidate_plans.manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "picoagent.cli_candidate_manifest.v1":
        raise ValueError("unsupported candidate plan manifest")
    count = 0
    for name, info in manifest["files"].items():
        path = output / name
        if path.name != name or file_hash(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"candidate plan hash mismatch: {name}")
        rows = read_jsonl(path)
        if len(rows) != info["records"]:
            raise ValueError(f"candidate plan record count mismatch: {name}")
        for row in rows:
            if row.get("schema") != CANDIDATE_SCHEMA or row.get("status") != "unexecuted_candidate_plan" or row.get("execution") != "unexecuted" or row.get("not_a_trace") is not True:
                raise ValueError("candidate plan incorrectly claims execution")
            if any(key in row for key in ("stdout", "stderr", "tool_events", "container_id", "verification")):
                raise ValueError("unexecuted candidate plan contains fabricated execution evidence")
            count += 1
    return {"passed": True, "tasks": sum(curriculum["counts"].values()), "candidate_plans": count,
            "family_counts": {split: len(curriculum["families"][split]) for split in ("train", "dev", "test")}}


def build_annotation_packets(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Make prefix-only inputs for a later synthetic-note pass; no hidden rationale."""
    packets: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "dev", "test")}
    for record in records:
        if record.get("execution_kind") != "native_teacher_observed" or record.get("not_a_trace") is not True:
            raise ValueError("annotation packets can only be projected from native teacher observations")
        for index, event in enumerate(record["teacher_events"]):
            calls = event.get("message", {}).get("tool_calls", [])
            if not calls:
                continue
            if len(calls) != 1:
                raise ValueError("annotation packet projection expects one proposed action per turn")
            call = calls[0]
            packet = {
                "schema": "picoagent.cli_artificial_reasoning_packet.v1",
                "packet_id": f"{record['task_id']}::action-{index}",
                "task_id": record["task_id"], "family": record["family"], "template_id": record["template_id"],
                "split": record["split"], "variant": "artificial_reasoning_annotation",
                "annotation_status": "pending_new_Luna_pass",
                # Input is exactly what was visible before this action; later observations/final turns are absent.
                "prefix_messages": copy.deepcopy(event["input_messages"]),
                "proposed_action": {"tool_call_id": call["id"], "name": call["function"]["name"],
                                    "arguments_json": call["function"]["arguments"]},
                "style_options": ["compact_code", "pseudocode", "caveman_shorthand"],
                "token_hint": {"minimum": 8, "maximum": 24},
                "instruction": ("Using only the visible prefix and proposed action, write one synthetic task-state note in 8–24 tokens. "
                               "Use compact code, pseudocode, or caveman shorthand rather than normal prose. "
                               "Do not claim hidden reasoning, future observations, tool outcomes, or a final answer."),
                "source_execution_kind": "native_teacher_observed",
                "source_teacher_identity": record["teacher_identity"],
                "source_action_sha256": content_hash(call),
            }
            packets[record["split"]].append(packet)
    return packets


def write_annotation_packets(observations_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Export annotation inputs only; a separate annotator creates note variants."""
    source = Path(observations_path)
    records = read_jsonl(source)
    by_split = build_annotation_packets(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, rows in by_split.items():
        path = output / f"{split}.prefix_only_packets.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size,
                            "records": len(rows), "split": split}
    unique_tasks = len({record["task_id"] for record in records})
    manifest = {"schema": "picoagent.cli_annotation_packet_manifest.v1",
                "source_observations_sha256": file_hash(source), "unique_source_tasks": unique_tasks,
                "annotation_packets": sum(len(rows) for rows in by_split.values()),
                "note": "Input packets are prefix-only; no synthetic note has been generated yet.", "files": files}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "unique_source_tasks": unique_tasks,
            "annotation_packets": manifest["annotation_packets"],
            "split_counts": {split: len(rows) for split, rows in by_split.items()}}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    target = root / "data" / "luna-cli-v1"
    print(json.dumps(write_luna_cli_dataset(target), indent=2))
