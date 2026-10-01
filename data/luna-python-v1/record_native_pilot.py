"""Run a tiny allowlisted deterministic teacher pilot; never sample model code."""
from __future__ import annotations

import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from picoagent.data.generators import SYSTEM_PROMPT
from picoagent.data.luna_python_curriculum import (
    DOC_FAMILIES, LunaPythonCallback, _code,
    generate_luna_python_task,
)
from picoagent.data.oracles import check_task_result
from picoagent.data.schema import canonical_json, content_hash
from picoagent.harness.tools import _PYTHON_RUNNER, TOOL_SCHEMAS

ROOT = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
ALLOWED = (("train", "py_math.invoice_total"), ("train", "py_docs.affine"), ("dev", "py_table.pivot_counts"))


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def materialize(task, workspace):
    for relative, text in task["environment"]["files"].items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return {name: sha_bytes((workspace / name).read_bytes()) for name in sorted(task["environment"]["files"])}


def invoke(name, args, workspace, env):
    if name == "bash":
        command = args["command"]
        if not command.startswith("python -m pydoc local_luna_api_"):
            raise ValueError("pilot Bash allowlist permits only the generated local pydoc read")
        argv = ["bash", "--noprofile", "--norc", "-c", command]
        stdin = ""
    elif name == "python":
        code = args["code"]
        # Require byte-for-byte agreement with the reviewed family program. The
        # docs call is dynamically derived only from its actual captured pydoc.
        argv = [sys.executable, "-I", "-c", _PYTHON_RUNNER]
        stdin = code
    else:
        raise ValueError("pilot permits only Bash pydoc and reviewed Python calls")
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(argv, input=stdin, text=True, encoding="utf-8",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=str(workspace), env=env, timeout=30, check=False)
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        exit_code = 124
    duration = round(time.perf_counter() - started, 6)
    result = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code,
              "timed_out": timed_out, "truncated": False, "duration_seconds": duration}
    receipt = {"name": name, "arguments": canonical_json(args), "argv": argv,
               "stdin": stdin, "cwd": str(workspace), "environment": dict(sorted(env.items())),
               **result, "stdout_sha256": sha_bytes(stdout.encode("utf-8")),
               "stderr_sha256": sha_bytes(stderr.encode("utf-8"))}
    return result, receipt


def source_hashes():
    relative = [
        "src/picoagent/data/luna_python_curriculum.py",
        "src/picoagent/data/oracles.py",
        "src/picoagent/data/schema.py",
        "src/picoagent/harness/agent.py",
        "src/picoagent/harness/tools.py",
        "src/picoagent/harness/sandbox.py",
        "data/luna-python-v1/record_native_pilot.py",
    ]
    out = {}
    for name in relative:
        path = ROOT / name
        out[name] = sha_bytes(path.read_bytes())
    return out


def run_one(split, family, task):
    regenerated = generate_luna_python_task(family, task["seed"])
    if canonical_json(regenerated) != canonical_json(task):
        raise ValueError("pilot task differs from the approved deterministic generator output")
    trace_id = "native-observed:" + task["task_id"]
    policy = LunaPythonCallback()
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user = {"role": "user", "content": task["prompt"]}
    messages = [system, user]
    callback_events, receipts, tool_events = [], [], []
    env = {"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin",
           "HOME": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    artifacts = {}
    fixture_hashes = {}
    with tempfile.TemporaryDirectory(prefix="luna-native-pilot-") as temporary:
        workspace = Path(temporary).resolve()
        fixture_hashes = materialize(task, workspace)
        final = ""
        status = "failed"
        for _ in range(4):
            inputs = json.loads(canonical_json(messages))
            response = policy(inputs, TOOL_SCHEMAS)
            assistant = json.loads(canonical_json(response))
            callback_events.append({"type": "deterministic_callback", "input_messages": inputs, "message": assistant})
            messages.append(assistant)
            calls = assistant.get("tool_calls", [])
            if not calls:
                final = assistant.get("content") or ""
                status = "candidate_final"
                break
            if len(calls) != 1:
                raise ValueError("pilot policy may issue one tool call at a time")
            call = calls[0]
            args = json.loads(call["function"]["arguments"])
            if call["function"]["name"] == "python":
                if family in DOC_FAMILIES:
                    if len(receipts) == 1:
                        observed_docs = json.loads(inputs[-1]["content"])["stdout"]
                        module = next(path[:-3].split("/")[-1] for path in task["environment"]["files"] if path.endswith(".py"))
                        expected_code = _code(family, module, observed_docs)
                    else:
                        expected_code = _code(family)
                else:
                    expected_code = _code(family)
                if args.get("code") != expected_code:
                    raise ValueError("Python call differs from reviewed deterministic family program")
            result, receipt = invoke(call["function"]["name"], args, workspace, env)
            receipt["sequence"] = len(receipts)
            receipt["tool_call_id"] = call["id"]
            receipts.append(receipt)
            tool_events.append({"step": len(callback_events), "name": call["function"]["name"],
                                "tool_call_id": call["id"], "arguments": call["function"]["arguments"],
                                "result": result, "verified": result["exit_code"] == 0})
            tool_message = {"role": "tool", "name": call["function"]["name"],
                            "tool_call_id": call["id"], "content": canonical_json(result)}
            messages.append(tool_message)
            if result["exit_code"] != 0 or result["timed_out"]:
                break
        if "output/chart.svg" in task["oracle"].get("artifact_path", ""):
            path = workspace / task["oracle"]["artifact_path"]
            if path.is_file():
                artifacts[task["oracle"]["artifact_path"]] = path.read_text(encoding="utf-8")
    verification = check_task_result(task, final, artifacts=artifacts)
    if status != "candidate_final" or not verification["passed"]:
        status = "failed"
    else:
        status = "success"
    frozen_candidate = {
        "candidate_id": "reviewed-helper:" + family,
        "model": "not-sampled",
        "mode": "reviewed_deterministic_callback",
        "source": "src/picoagent/data/luna_python_curriculum.py",
        "source_sha256": source_hashes()["src/picoagent/data/luna_python_curriculum.py"],
        "execution": "native_teacher_observed",
        "sft_eligible": False,
        "note": "This is a deterministic helper program, not a GPT-6 Luna sampled response.",
    }
    evidence = {"task": task, "candidate": frozen_candidate, "fixture_sha256": fixture_hashes,
                "source_sha256": source_hashes(), "receipts": receipts,
                "artifacts": {k: {"text": v, "sha256": sha_bytes(v.encode("utf-8"))} for k, v in artifacts.items()},
                "kv": {}, "callback_events": callback_events, "tool_events": tool_events,
                "environment_policy": {"backend": "ordinary_exec_command_sandbox", "network_used": False,
                                       "allowlist": ["python -m pydoc for one generated local module",
                                                     "reviewed deterministic Python family program"],
                                       "child_env": env, "timeout_seconds": 30}}
    record = {
        "schema_version": "picoagent.native_observation.v1",
        "trace_id": trace_id, "task_id": task["task_id"], "family": family,
        "template_id": task["template_id"], "split": split,
        "execution": "native_teacher_observed", "status": status,
        "provenance": {"source": "original_procedural", "benchmark": False,
                       "teacher_mode": "reviewed_deterministic_callback",
                       "teacher_model": "not_sampled",
                       "runtime": {"backend": "ordinary_exec_command_sandbox",
                                   "python_version": platform.python_version(),
                                   "python_executable": sys.executable,
                                   "platform": platform.platform(),
                                   "locale": locale.getlocale(),
                                   "container_id": None, "image": None}},
        "messages": messages, "effective_messages": messages,
        "candidate": frozen_candidate, "native_evidence": evidence,
        "verification": {**verification, "passed": verification["passed"] and status == "success"},
        "sft_admissible": False,
        "task_sha256": content_hash(task),
        "raw_attempt_sha256": content_hash(evidence),
    }
    return record


def main():
    train = {row["family"]: row for row in read_rows(TRACK / "train.tasks.jsonl")}
    dev = {row["family"]: row for row in read_rows(TRACK / "dev.tasks.jsonl")}
    selected = []
    for split, family in ALLOWED:
        task = (train if split == "train" else dev)[family]
        if task["split"] != split or task["seed"] != 0:
            raise ValueError("pilot allowlist row has unexpected split/seed")
        selected.append(run_one(split, family, task))
    out = TRACK / "native_teacher_observed.jsonl"
    with out.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(canonical_json(row) + "\n")
    manifest = {"schema": "picoagent.native_observation.manifest.v1",
                "record_file": out.name, "record_sha256": sha_bytes(out.read_bytes()),
                "record_count": len(selected), "trace_ids": [r["trace_id"] for r in selected],
                "execution": "native_teacher_observed", "sft_admissible": False,
                "source_sha256": source_hashes()}
    (TRACK / "native_teacher_observed.manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(canonical_json({"path": str(out), "records": len(selected),
                          "statuses": {r["task_id"]: r["status"] for r in selected}}))


if __name__ == "__main__":
    main()

