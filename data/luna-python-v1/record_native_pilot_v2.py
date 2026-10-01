"""Corrected append-only native-observation pilot, separate from verified traces."""
from __future__ import annotations

import ast
import base64
import csv
import datetime
import hashlib
import io
import json
import locale
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from picoagent.data.generators import SYSTEM_PROMPT
from picoagent.data.luna_python_curriculum import (
    DOC_FAMILIES, LunaPythonCallback, _code, generate_luna_python_task,
)
from picoagent.data.schema import canonical_json, content_hash
from picoagent.harness.tools import _PYTHON_RUNNER, TOOL_SCHEMAS

REPO = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
OUTPUT = TRACK / "native_teacher_observed.v2"
ALLOWED = (("train", "py_math.invoice_total"), ("train", "py_docs.affine"),
           ("dev", "py_table.pivot_counts"))
SOURCE_PATHS = (
    "src/picoagent/data/luna_python_curriculum.py",
    "src/picoagent/data/oracles.py",
    "src/picoagent/data/schema.py",
    "src/picoagent/harness/agent.py",
    "src/picoagent/harness/tools.py",
    "src/picoagent/harness/sandbox.py",
    "data/luna-python-v1/record_native_pilot_v2.py",
)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_file_line(path: Path, value: dict) -> None:
    with path.open("ab") as handle:
        handle.write((canonical_json(value) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def load_split(split: str) -> dict[str, dict]:
    path = TRACK / f"{split}.tasks.jsonl"
    return {row["family"]: row for row in (json.loads(line) for line in path.read_text().splitlines() if line)}


def install_fixtures(task: dict, workspace: Path) -> dict[str, str]:
    for relative, text in task["environment"]["files"].items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    return {relative: sha_file(workspace / relative) for relative in sorted(task["environment"]["files"])}


def independent_expected(task: dict) -> dict:
    """Recompute from fixture bytes, never from task.oracle.expected."""
    family = task["family"]
    files = task["environment"]["files"]
    if family == "py_math.invoice_total":
        data = json.loads(files["input/task.json"])
        subtotal = sum((Decimal(row["unit_price"]) * row["units"] for row in data["lines"]), Decimal(0))
        amount = subtotal + Decimal(data["fee"])
        return {"total": str(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))}
    if family == "py_docs.affine":
        module_path, source = next((p, v) for p, v in files.items() if p.endswith(".py"))
        doc = source.split('"""', 2)[1]
        function_name = re.search(r"API callable: ([A-Za-z_][A-Za-z0-9_]*)", doc).group(1)
        if not function_name:
            raise ValueError("fixture API doc does not name its function")
        data = json.loads(files["input/task.json"])
        addition = re.search(r"Add (-?[0-9]+), then multiply by ([0-9]+)", doc)
        multiplication = re.search(r"Multiply by ([0-9]+), then add (-?[0-9]+)", doc)
        if addition:
            add, mult = map(int, addition.groups())
            result = [(value + add) * mult for value in data["values"]]
        elif multiplication:
            mult, add = map(int, multiplication.groups())
            result = [value * mult + add for value in data["values"]]
        else:
            raise ValueError(f"fixture documentation in {module_path} lacks a recognized transform")
        return {"values": result}
    if family == "py_table.pivot_counts":
        rows = list(csv.DictReader(io.StringIO(files["input/task.csv"])))
        regions = sorted({row["region"] for row in rows})
        states = sorted({row["state"] for row in rows})
        return {"counts": {region: {state: sum(row["region"] == region and row["state"] == state for row in rows)
                                    for state in states} for region in regions}}
    raise ValueError(f"no independent oracle for pilot family {family}")


def independent_oracle(task: dict, final: str) -> dict:
    expected = independent_expected(task)
    expected_text = "Computed result: " + canonical_json(expected) + "."
    return {"passed": final.strip() == expected_text,
            "method": "independent_fixture_recomputation",
            "expected_sha256": content_hash(expected),
            "failures": [] if final.strip() == expected_text else ["final answer differs from independently recomputed fixture result"]}


def module_name(task: dict) -> str:
    return next(Path(path).stem for path in task["environment"]["files"] if path.endswith(".py"))


def candidate_row(task: dict, callback_source_hash: str) -> dict:
    family = task["family"]
    if family in DOC_FAMILIES:
        module = module_name(task)
        planned = [
            {"tool": "bash", "arguments": {"command": f"python -m pydoc {module}"}},
            {"tool": "python", "arguments": {"code_source": "derive only from the observed pydoc response using LunaPythonCallback"}},
        ]
    else:
        code = _code(family)
        ast.parse(code)
        planned = [{"tool": "python", "arguments": {"code": code}}]
    return {"schema_version": "picoagent.native_candidate_plan.v1",
            "candidate_id": f"reviewed-deterministic-helper:{task['task_id']}",
            "task_id": task["task_id"], "task_sha256": content_hash(task),
            "family": family, "template_id": task["template_id"], "split": task["split"],
            "status": "unexecuted_candidate_plan", "execution": "unexecuted",
            "author": "reviewed_deterministic_callback_program", "model": "not_sampled",
            "source_path": "src/picoagent/data/luna_python_curriculum.py",
            "source_sha256": callback_source_hash, "planned_tool_calls": planned,
            "receipts": [], "tool_events": [], "has_final_response": False,
            "training_eligible": False,
            "note": "Pre-action deterministic helper plan only; the later observed record links its actual execution receipts."}


def capture_process(name: str, args: dict, workspace: Path, child_env: dict,
                    call_id: str) -> tuple[dict, dict]:
    if name == "bash":
        command = args["command"]
        if not command.startswith("python -m pydoc local_luna_api_"):
            raise ValueError("pilot Bash allowlist permits only local generated-module pydoc")
        argv = ["bash", "--noprofile", "--norc", "-c", command]
        stdin_text = ""
    elif name == "python":
        code = args["code"]
        if not isinstance(code, str) or not code:
            raise ValueError("Python action must contain reviewed nonempty code")
        argv = [sys.executable, "-I", "-c", _PYTHON_RUNNER]
        stdin_text = code
    else:
        raise ValueError("pilot tool allowlist permits only bash pydoc and python")
    stdin_bytes = stdin_text.encode("utf-8")
    started = time.perf_counter()
    timed_out = False
    proc = subprocess.Popen(argv, cwd=str(workspace), env=child_env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        stdout_raw, stderr_raw = proc.communicate(input=stdin_bytes, timeout=30)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout_raw, stderr_raw = proc.communicate()
        exit_code = 124
    duration = round(time.perf_counter() - started, 6)
    result = {"stdout": stdout_raw.decode("utf-8", errors="replace"),
              "stderr": stderr_raw.decode("utf-8", errors="replace"),
              "exit_code": exit_code, "timed_out": timed_out, "truncated": False,
              "duration_seconds": duration}
    receipt = {"sequence": None, "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "tool_call_id": call_id, "name": name,
               "arguments": canonical_json(args), "argv": argv, "stdin": stdin_text,
               "stdin_sha256": sha_bytes(stdin_bytes), "cwd": str(workspace),
               "environment": dict(sorted(child_env.items())),
               "stdout": result["stdout"], "stderr": result["stderr"],
               "stdout_bytes_b64": base64.b64encode(stdout_raw).decode("ascii"),
               "stderr_bytes_b64": base64.b64encode(stderr_raw).decode("ascii"),
               "stdout_sha256": sha_bytes(stdout_raw), "stderr_sha256": sha_bytes(stderr_raw),
               "exit_code": exit_code, "timed_out": timed_out, "truncated": False,
               "duration_seconds": duration}
    return result, receipt


def snapshot_sources(output: Path) -> dict[str, str]:
    mapping = {}
    for relative in SOURCE_PATHS:
        source = REPO / relative
        raw = source.read_bytes()
        digest = sha_bytes(raw)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        mapping[relative] = digest
    return mapping


def runtime_info(child_env: dict) -> dict:
    bash = shutil.which("bash", path=child_env["PATH"])
    info = {"backend": "ordinary_exec_command_sandbox",
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_executable_sha256": sha_file(Path(sys.executable)),
            "bash_executable": bash,
            "bash_executable_sha256": sha_file(Path(bash)) if bash else None,
            "platform": platform.platform(), "locale": locale.getlocale(),
            "container_id": None, "image": None}
    return info


def run_one(split: str, family: str, task: dict, sources: dict[str, str], child_env: dict,
            candidates_path: Path, event_dir: Path) -> dict:
    regenerated = generate_luna_python_task(family, task["seed"])
    if canonical_json(regenerated) != canonical_json(task):
        raise ValueError("frozen task does not match allowlisted generator family/seed")
    task_hash = content_hash(task)
    candidate = candidate_row(task, sources["src/picoagent/data/luna_python_curriculum.py"])
    candidate_hash = content_hash(candidate)
    canonical_file_line(candidates_path, candidate)
    attempt_id = uuid.uuid4().hex
    event_path = event_dir / f"{attempt_id}.events.jsonl"

    def journal(event):
        canonical_file_line(event_path, {"sequence": event_counter[0], "event": event})
        event_counter[0] += 1

    event_counter = [0]
    journal({"type": "attempt_started", "attempt_id": attempt_id,
             "task_id": task["task_id"], "task_sha256": task_hash,
             "candidate_id": candidate["candidate_id"], "candidate_sha256": candidate_hash})
    policy = LunaPythonCallback()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task["prompt"]}]
    callbacks, receipts, tool_events = [], [], []
    artifacts = {}
    status = "error"
    final = ""
    try:
        with tempfile.TemporaryDirectory(prefix="luna-native-v2-") as temporary:
            workspace = Path(temporary).resolve()
            fixture_hashes = install_fixtures(task, workspace)
            journal({"type": "fixtures_seeded", "cwd": str(workspace), "fixture_sha256": fixture_hashes})
            for _ in range(4):
                input_messages = json.loads(canonical_json(messages))
                journal({"type": "callback_input", "input_messages": input_messages,
                         "tool_schemas": TOOL_SCHEMAS})
                response = policy(input_messages, TOOL_SCHEMAS)
                assistant = json.loads(canonical_json(response))
                callback_event = {"type": "deterministic_callback", "input_messages": input_messages,
                                  "message": assistant}
                callbacks.append(callback_event)
                journal({"type": "callback_output", "message": assistant})
                messages.append(assistant)
                calls = assistant.get("tool_calls", [])
                if not calls:
                    final = assistant.get("content") or ""
                    journal({"type": "final_response", "content": final})
                    status = "success"
                    break
                if len(calls) != 1:
                    raise ValueError("pilot policy must issue one tool call per callback")
                call = calls[0]
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                if name == "python":
                    if family in DOC_FAMILIES and len(receipts) == 1:
                        docs = json.loads(input_messages[-1]["content"])["stdout"]
                        code = _code(family, module_name(task), docs)
                    else:
                        code = _code(family)
                    if args.get("code") != code:
                        raise ValueError("tool code differs from reviewed family helper code")
                journal({"type": "tool_invocation", "tool_call_id": call["id"], "name": name,
                         "arguments": call["function"]["arguments"], "workspace": str(workspace)})
                result, receipt = capture_process(name, args, workspace, child_env, call["id"])
                receipt["sequence"] = len(receipts)
                receipts.append(receipt)
                journal({"type": "tool_result", "receipt": receipt})
                tool_events.append({"step": len(callbacks), "name": name,
                                    "tool_call_id": call["id"],
                                    "arguments": call["function"]["arguments"],
                                    "result": result, "verified": result["exit_code"] == 0})
                tool_message = {"role": "tool", "name": name,
                                "tool_call_id": call["id"], "content": canonical_json(result)}
                messages.append(tool_message)
                if result["exit_code"] != 0 or result["timed_out"]:
                    status = "failed"
                    break
            if task["oracle"].get("artifact_path"):
                path = workspace / task["oracle"]["artifact_path"]
                if path.is_file():
                    artifacts[task["oracle"]["artifact_path"]] = path.read_text(encoding="utf-8")
        verify = independent_oracle(task, final)
        journal({"type": "independent_oracle", "verification": verify})
        if status != "success" or not verify["passed"]:
            status = "failed"
        else:
            status = "success"
        evidence = {"task": task, "candidate": candidate,
                    "candidate_sha256": candidate_hash, "source_sha256": sources,
                    "fixture_sha256": fixture_hashes, "receipts": receipts,
                    "callback_events": callbacks, "tool_events": tool_events,
                    "artifacts": {k: {"text": v, "sha256": sha_bytes(v.encode("utf-8"))}
                                  for k, v in artifacts.items()},
                    "kv": {}, "independent_verification": verify,
                    "environment_policy": {"backend": "ordinary_exec_command_sandbox",
                                           "network_used": False,
                                           "child_env": child_env, "timeout_seconds": 30,
                                           "tool_allowlist": ["generated local pydoc", "reviewed Python helper"]}}
        record = {"schema_version": "picoagent.native_observation.v2",
                  "trace_id": f"native-observed-v2:{task['task_id']}:{attempt_id}",
                  "task_id": task["task_id"], "family": family, "template_id": task["template_id"],
                  "split": split, "execution": "native_teacher_observed", "status": status,
                  "provenance": {"source": "original_procedural", "benchmark": False,
                                 "teacher_mode": "reviewed_deterministic_callback",
                                 "teacher_model": "not_sampled",
                                 "runtime": runtime_info(child_env)},
                  "candidate_id": candidate["candidate_id"], "candidate_sha256": candidate_hash,
                  "task_sha256": task_hash, "raw_attempt_sha256": content_hash(evidence),
                  "messages": messages, "effective_messages": messages,
                  "native_evidence": evidence, "verification": verify,
                  "sft_admissible": False}
        journal({"type": "attempt_finished", "status": status,
                 "task_sha256": task_hash, "candidate_sha256": candidate_hash})
        return record
    except Exception as exc:
        journal({"type": "attempt_exception", "error_type": type(exc).__name__, "message": str(exc)})
        journal({"type": "attempt_finished", "status": "error"})
        raise


def main():
    if OUTPUT.exists():
        raise FileExistsError("corrected v2 output directory already exists; preserve it and create a new version")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    candidates_path = OUTPUT / "candidate_plans.unexecuted.jsonl"
    candidates_path.touch(exist_ok=False)
    records_path = OUTPUT / "records.jsonl"
    records_path.touch(exist_ok=False)
    event_dir = OUTPUT / "events"
    event_dir.mkdir(exist_ok=False)
    sources = snapshot_sources(OUTPUT)
    child_env = {"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin",
                 "HOME": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    train, dev = load_split("train"), load_split("dev")
    tasks = []
    for split, family in ALLOWED:
        task = (train if split == "train" else dev)[family]
        if task["split"] != split or task["seed"] != 0:
            raise ValueError("pilot family allowlist has unexpected split or seed")
        tasks.append((split, family, task))
    progress_path = OUTPUT / "progress.jsonl"
    progress_path.touch(exist_ok=False)
    failures_path = OUTPUT / "failures.jsonl"
    failures_path.touch(exist_ok=False)
    manifest = {"schema": "picoagent.native_observation.manifest.v2",
                "track": "luna-python-v1", "execution": "native_teacher_observed",
                "sft_admissible": False, "record_file": records_path.name,
                "candidate_file": candidates_path.name, "source_sha256": sources,
                "runtime": runtime_info(child_env),
                "launcher": {"tool": "functions.exec_command",
                             "cmd": "PYTHONPATH=src python data/luna-python-v1/record_native_pilot_v2.py",
                             "argv": [sys.executable, str(Path(__file__).resolve())],
                             "cwd": str(REPO), "environment": {"PYTHONPATH": "src"}},
                "policy": {"network_used": False, "tool_allowlist": ["generated local pydoc", "reviewed Python helper"],
                           "child_env": child_env, "timeout_seconds": 30},
                "record_count": 0, "trace_ids": []}
    for split, family, task in tasks:
        try:
            record = run_one(split, family, task, sources, child_env, candidates_path, event_dir)
            canonical_file_line(records_path, record)
            manifest["record_count"] += 1
            manifest["trace_ids"].append(record["trace_id"])
            canonical_file_line(progress_path, {"event": "record_completed", "trace_id": record["trace_id"],
                                                "records_sha256": sha_file(records_path)})
        except Exception as exc:
            failure = {"task_id": task["task_id"], "family": family, "split": split,
                       "status": "error", "execution": "native_teacher_observed",
                       "sft_admissible": False, "error_type": type(exc).__name__, "message": str(exc)}
            canonical_file_line(failures_path, failure)
            canonical_file_line(progress_path, {"event": "attempt_failed", "task_id": task["task_id"],
                                                "error_type": type(exc).__name__})
            continue
    manifest["records_sha256"] = sha_file(records_path)
    manifest["candidate_sha256"] = sha_file(candidates_path)
    manifest["failures_sha256"] = sha_file(failures_path)
    manifest["progress_sha256"] = sha_file(progress_path)
    manifest["events"] = {p.name: sha_file(p) for p in sorted(event_dir.glob("*.events.jsonl"))}
    manifest["source_snapshots"] = {str(p.relative_to(OUTPUT)): sha_file(p)
                                    for p in sorted((OUTPUT / "source_snapshot").rglob("*")) if p.is_file()}
    manifest_path = OUTPUT / "manifest.json"
    with manifest_path.open("xb") as stream:
        stream.write((canonical_json(manifest) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    print(canonical_json({"output": str(OUTPUT), "records": manifest["record_count"],
                          "trace_ids": manifest["trace_ids"]}))


if __name__ == "__main__":
    main()

