"""Resumable CPU-only native observation batch for reviewed Python helpers.

This recorder never calls a live model and never executes Luna-authored candidate
code. It records deterministic reviewed callback replays as native_teacher_observed,
separate from ContainerSandbox/verified_environment traces.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import locale
import os
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
    DOC_FAMILIES, FAMILY_SPLITS, LunaPythonCallback, _code,
    generate_luna_python_task, verify_independent_result,
)
from picoagent.data.schema import canonical_json, content_hash
from picoagent.harness.tools import _PYTHON_RUNNER, TOOL_SCHEMAS

REPO = Path(__file__).resolve().parents[2]
TRACK = Path(__file__).resolve().parent
SOURCE_PATHS = (
    "src/picoagent/data/luna_python_curriculum.py",
    "src/picoagent/data/generators.py",
    "src/picoagent/data/schema.py",
    "src/picoagent/data/oracles.py",
    "src/picoagent/harness/agent.py",
    "src/picoagent/harness/tools.py",
    "src/picoagent/harness/sandbox.py",
    "data/luna-python-v1/record_native_batch.py",
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def append_fsynced(path: Path, value: dict) -> None:
    with path.open("ab") as stream:
        stream.write((canonical_json(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]



def runtime_metadata(env: dict[str, str]) -> dict:
    bash_path = shutil.which("bash", path=env["PATH"])
    return {
        "backend": "ordinary_exec_command_sandbox",
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_executable_sha256": digest_file(Path(sys.executable)),
        "bash_executable": bash_path,
        "bash_executable_sha256": digest_file(Path(bash_path)) if bash_path else None,
        "platform": platform.platform(),
        "locale": locale.getlocale(),
        "container_id": None,
        "image": None,
    }


def task_iterator(train_seeds: int, dev_seeds: int):
    if train_seeds < 1 or dev_seeds < 1:
        raise ValueError("seed counts must be positive")
    for family in sorted(FAMILY_SPLITS):
        split = FAMILY_SPLITS[family]
        if split == "test":
            continue
        count = train_seeds if split == "train" else dev_seeds
        for seed in range(count):
            yield split, generate_luna_python_task(family, seed)


def family_candidate(task: dict, source_sha: str) -> dict:
    family = task["family"]
    if family in DOC_FAMILIES:
        module = next(Path(name).stem for name in task["environment"]["files"] if name.endswith(".py"))
        actions = [
            {"tool": "bash", "arguments": {"command": f"python -m pydoc {module}"}},
            {"tool": "python", "arguments": {"code_source": "derive only from captured pydoc via reviewed callback"}},
        ]
    else:
        code = _code(family)
        actions = [{"tool": "python", "arguments": {"code": code}}]
    return {
        "schema_version": "picoagent.native_candidate_plan.v1",
        "candidate_id": f"reviewed-helper:{task['task_id']}",
        "task_id": task["task_id"], "task_sha256": content_hash(task),
        "family": family, "template_id": task["template_id"], "split": task["split"],
        "status": "unexecuted_candidate_plan", "execution": "unexecuted",
        "author": "reviewed_deterministic_callback_program", "model": None,
        "source_path": "src/picoagent/data/luna_python_curriculum.py",
        "source_sha256": source_sha, "planned_tool_calls": actions,
        "receipts": [], "tool_events": [], "has_final_response": False,
        "training_eligible": False, "luna_candidate_reused": False,
        "note": "Reviewed fixed helper plan; separate Luna-authored candidates remain unexecuted and are not attributed to this replay.",
    }


def materialize(task: dict, workspace: Path) -> dict[str, str]:
    for relative, content in task["environment"]["files"].items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    return {name: digest_file(workspace / name) for name in sorted(task["environment"]["files"])}


def run_process(tool_name: str, arguments: dict, call_id: str,
                workspace: Path, child_env: dict[str, str]) -> tuple[dict, dict]:
    if tool_name == "bash":
        command = arguments["command"]
        module = command.removeprefix("python -m pydoc ")
        if not command.startswith("python -m pydoc local_luna_api_") or not module.isidentifier():
            raise ValueError("reviewed Bash allowlist permits only the generated local pydoc module")
        argv = ["bash", "--noprofile", "--norc", "-c", command]
        stdin_text = ""
    elif tool_name == "python":
        code = arguments.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("reviewed Python invocation needs exact nonempty code")
        argv = [sys.executable, "-I", "-c", _PYTHON_RUNNER]
        stdin_text = code
    else:
        raise ValueError("reviewed callback uses only Bash pydoc and Python")
    stdin_bytes = stdin_text.encode("utf-8")
    start = time.perf_counter()
    timed_out = False
    process = subprocess.Popen(
        argv, cwd=str(workspace), env=child_env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout_raw, stderr_raw = process.communicate(input=stdin_bytes, timeout=30)
        exit_code = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout_raw, stderr_raw = process.communicate()
        exit_code = 124
    duration = round(time.perf_counter() - start, 6)
    result = {
        "stdout": stdout_raw.decode("utf-8", errors="replace"),
        "stderr": stderr_raw.decode("utf-8", errors="replace"),
        "exit_code": exit_code, "timed_out": timed_out, "truncated": False,
        "duration_seconds": duration,
    }
    receipt = {
        "sequence": None, "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool_call_id": call_id, "name": tool_name, "arguments": canonical_json(arguments),
        "argv": argv, "stdin": stdin_text, "stdin_sha256": digest_bytes(stdin_bytes),
        "cwd": str(workspace), "environment": dict(sorted(child_env.items())),
        "stdout": result["stdout"], "stderr": result["stderr"],
        "stdout_bytes_b64": base64.b64encode(stdout_raw).decode("ascii"),
        "stderr_bytes_b64": base64.b64encode(stderr_raw).decode("ascii"),
        "stdout_sha256": digest_bytes(stdout_raw), "stderr_sha256": digest_bytes(stderr_raw),
        "exit_code": exit_code, "timed_out": timed_out, "truncated": False,
        "duration_seconds": duration,
    }
    return result, receipt


def run_task(task: dict, candidate: dict, source_hashes: dict[str, str],
             child_env: dict[str, str], event_log: Path) -> tuple[dict, dict]:
    task_hash = content_hash(task)
    candidate_hash = content_hash(candidate)
    attempt_id = uuid.uuid4().hex
    seq = [0]

    def event(payload: dict) -> None:
        append_fsynced(event_log, {
            "attempt_id": attempt_id, "task_id": task["task_id"],
            "sequence": seq[0], "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event": payload,
        })
        seq[0] += 1

    event({"type": "attempt_started", "task_sha256": task_hash,
           "candidate_id": candidate["candidate_id"], "candidate_sha256": candidate_hash})
    policy = LunaPythonCallback()
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user = {"role": "user", "content": task["prompt"]}
    messages = [system, user]
    callback_events, receipts, tool_events, artifacts = [], [], [], {}
    artifact_bytes = {}
    fixture_hashes = {}
    final = ""
    status = "error"
    workspace_path = None
    try:
        with tempfile.TemporaryDirectory(prefix="luna-python-batch-") as directory:
            workspace = Path(directory).resolve()
            workspace_path = str(workspace)
            fixture_hashes = materialize(task, workspace)
            event({"type": "fixtures_materialized", "cwd": workspace_path,
                   "fixture_sha256": fixture_hashes})
            for _ in range(5):
                inputs = json.loads(canonical_json(messages))
                event({"type": "callback_input", "input_messages": inputs, "tools": TOOL_SCHEMAS})
                response = policy(inputs, TOOL_SCHEMAS)
                assistant = json.loads(canonical_json(response))
                callback_events.append({"type": "deterministic_callback",
                                        "input_messages": inputs, "message": assistant})
                event({"type": "callback_output", "message": assistant})
                messages.append(assistant)
                calls = assistant.get("tool_calls", [])
                if not calls:
                    final = assistant.get("content") or ""
                    status = "success"
                    event({"type": "final_response", "content": final})
                    break
                if len(calls) != 1:
                    raise ValueError("reviewed callback may issue one call at a time")
                call = calls[0]
                name = call["function"]["name"]
                arguments = json.loads(call["function"]["arguments"])
                if name == "python":
                    family = task["family"]
                    if family in DOC_FAMILIES and len(receipts) == 1:
                        docs = json.loads(inputs[-1]["content"])["stdout"]
                        module = next(Path(path).stem for path in task["environment"]["files"] if path.endswith(".py"))
                        reviewed_code = _code(family, module, docs)
                    else:
                        reviewed_code = _code(family)
                    if arguments.get("code") != reviewed_code:
                        raise ValueError("callback code differs from frozen reviewed family program")
                event({"type": "tool_invocation", "tool_call_id": call["id"],
                       "name": name, "arguments": call["function"]["arguments"]})
                result, receipt = run_process(name, arguments, call["id"], workspace, child_env)
                receipt["sequence"] = len(receipts)
                receipts.append(receipt)
                event({"type": "tool_result", "receipt": receipt})
                tool_events.append({
                    "step": len(callback_events), "name": name,
                    "tool_call_id": call["id"], "arguments": call["function"]["arguments"],
                    "result": result, "verified": result["exit_code"] == 0,
                })
                messages.append({"role": "tool", "name": name,
                                 "tool_call_id": call["id"], "content": canonical_json(result)})
                if result["exit_code"] != 0 or result["timed_out"]:
                    status = "failed"
                    break
            artifact_path = task["oracle"].get("artifact_path")
            if artifact_path and (workspace / artifact_path).is_file():
                raw_artifact = (workspace / artifact_path).read_bytes()
                artifact_text = raw_artifact.decode("utf-8", errors="replace")
                artifact_bytes[artifact_path] = {
                    "bytes_b64": base64.b64encode(raw_artifact).decode("ascii"),
                    "sha256": digest_bytes(raw_artifact), "size_bytes": len(raw_artifact),
                }
                artifacts[artifact_path] = artifact_text
                event({"type": "artifact_capture", "path": artifact_path,
                       **artifact_bytes[artifact_path]})
        verification = verify_independent_result(task, final, artifacts)
        event({"type": "independent_verification", "verification": verification})
        if not verification["passed"]:
            status = "failed"
        evidence = {
            "task": task, "candidate": candidate, "candidate_sha256": candidate_hash,
            "source_sha256": source_hashes, "fixture_sha256": fixture_hashes,
            "tools": TOOL_SCHEMAS, "tool_schemas_sha256": content_hash(TOOL_SCHEMAS),
            "receipts": receipts, "callback_events": callback_events, "tool_events": tool_events,
            "artifacts": {name: {"text": value, "sha256": digest_bytes(value.encode("utf-8"))}
                          for name, value in artifacts.items()},
            "artifact_bytes": artifact_bytes,
            "kv": {}, "independent_verification": verification,
            "event_attempt_id": attempt_id, "event_workspace": workspace_path,
            "environment_policy": {"backend": "ordinary_exec_command_sandbox",
                                   "network_used": False, "child_env": child_env,
                                   "timeout_seconds": 30,
                                   "tool_allowlist": ["python -m pydoc local fixture", "reviewed deterministic Python helper"]},
        }
        record = {
            "schema_version": "picoagent.native_observation.v3",
            "trace_id": f"native-v3:{task['task_id']}:{attempt_id}",
            "task_id": task["task_id"], "family": task["family"],
            "template_id": task["template_id"], "split": task["split"],
            "execution": "native_teacher_observed", "status": status,
            "provenance": {
                "source": "original_procedural", "benchmark": False,
                "teacher_model": None, "teacher_mode": "reviewed_procedural_replay",
                "teacher_decision_mode": "reviewed_procedural_replay",
                "candidate_author": "reviewed_deterministic_callback_program",
                "runtime": {
                    "backend": "ordinary_exec_command_sandbox",
                    "python_version": platform.python_version(), "python_executable": sys.executable,
                    "python_executable_sha256": digest_file(Path(sys.executable)),
                    "bash_executable": shutil.which("bash", path=child_env["PATH"]),
                    "bash_executable_sha256": digest_file(Path(shutil.which("bash", path=child_env["PATH"]))),
                    "platform": platform.platform(), "locale": locale.getlocale(),
                    "container_id": None, "image": None,
                },
                "container_semantic_replay": "not_verified",
            },
            "task_sha256": task_hash, "candidate_id": candidate["candidate_id"],
            "tools": TOOL_SCHEMAS, "tool_schemas_sha256": content_hash(TOOL_SCHEMAS),
            "candidate_sha256": candidate_hash, "raw_attempt_sha256": content_hash(evidence),
            "messages": messages, "effective_messages": messages,
            "native_evidence": evidence, "verification": verification,
            "sft_admissible": False,
        }
        event({"type": "attempt_finished", "status": status,
               "record_sha256": content_hash(record)})
        return record, {"attempt_id": attempt_id, "task_sha256": task_hash,
                        "candidate_sha256": candidate_hash, "status": status}
    except Exception as exc:
        event({"type": "attempt_exception", "error_type": type(exc).__name__,
               "message": str(exc)})
        event({"type": "attempt_finished", "status": "error"})
        raise


def write_initial_tasks(output: Path, train_seeds: int, dev_seeds: int,
                        source_hashes: dict[str, str]) -> list[tuple[dict, dict]]:
    tasks = list(task_iterator(train_seeds, dev_seeds))
    frozen = {split: [] for split in ("train", "dev")}
    for split, task in tasks:
        frozen[split].append(task)
    task_root = output / "tasks"
    task_root.mkdir(exist_ok=False)
    files = {}
    for split, rows in frozen.items():
        path = task_root / f"{split}.tasks.jsonl"
        with path.open("xb") as stream:
            for task in rows:
                stream.write((canonical_json(task) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        files[path.name] = {"sha256": digest_file(path), "bytes": path.stat().st_size,
                            "records": len(rows)}
    tasks_manifest = {
        "schema": "picoagent.curriculum.manifest.v1",
        "track": "luna-python-original-v1",
        "configuration": {"train_seeds_per_family": train_seeds,
                          "dev_seeds_per_family": dev_seeds,
                          "test_execution": False},
        "split_policy": FAMILY_SPLITS,
        "split_policy_sha256": content_hash(FAMILY_SPLITS),
        "source_sha256": source_hashes,
        "files": files,
        "counts": {split: len(rows) for split, rows in frozen.items()},
    }
    path = task_root / "manifest.json"
    with path.open("xb") as stream:
        stream.write((canonical_json(tasks_manifest) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    candidates_path = output / "candidate_plans.reviewed_unexecuted.jsonl"
    with candidates_path.open("xb") as stream:
        for _, task in tasks:
            candidate = family_candidate(task, source_hashes["src/picoagent/data/luna_python_curriculum.py"])
            stream.write((canonical_json(candidate) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return tasks


def source_snapshots(output: Path) -> dict[str, str]:
    hashes = {}
    for relative in SOURCE_PATHS:
        data = (REPO / relative).read_bytes()
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[relative] = digest_bytes(data)
    return hashes


def load_frozen_tasks(output: Path) -> list[tuple[str, dict]]:
    rows = []
    for split in ("train", "dev"):
        path = output / "tasks" / f"{split}.tasks.jsonl"
        for task in jsonl_rows(path):
            regenerated = generate_luna_python_task(task["family"], task["seed"])
            if canonical_json(regenerated) != canonical_json(task):
                raise ValueError("frozen task no longer matches the reviewed generator source")
            rows.append((split, task))
    return rows


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".partial.json")
    with temporary.open("wb") as stream:
        stream.write((canonical_json(manifest) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/luna-python-v1/native_teacher_observed.scale-v1")
    parser.add_argument("--train-seeds-per-family", type=int, default=625)
    parser.add_argument("--dev-seeds-per-family", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="resume an existing interrupted batch with identical frozen sources and tasks")
    args = parser.parse_args(argv)
    output = (REPO / args.output_dir).resolve()
    if not output.is_relative_to(REPO.resolve()):
        raise ValueError("output directory must stay inside the repository workspace")
    child_env = {"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin",
                 "HOME": "/tmp", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                 "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    current_sources = {relative: digest_bytes((REPO / relative).read_bytes()) for relative in SOURCE_PATHS}
    if output.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume only for the identical reviewed frozen batch")
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("cannot resume: frozen manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            raise ValueError("batch is already complete; preserve it and choose a new output directory")
        expected_config = {"train_seeds_per_family": args.train_seeds_per_family,
                           "dev_seeds_per_family": args.dev_seeds_per_family,
                           "test_execution": False}
        if manifest.get("source_sha256") != current_sources or manifest.get("configuration") != expected_config:
            raise ValueError("resume source/config differs from the frozen run")
        if manifest.get("task_manifest_sha256") != digest_file(output / "tasks/manifest.json"):
            raise ValueError("resume task manifest hash mismatch")
        if manifest.get("candidate_file_sha256") != digest_file(output / "candidate_plans.reviewed_unexecuted.jsonl"):
            raise ValueError("resume candidate sidecar hash mismatch")
        tasks = load_frozen_tasks(output)
        source_hashes = current_sources
    else:
        output.mkdir(parents=True, exist_ok=False)
        source_hashes = source_snapshots(output)
        tasks = write_initial_tasks(output, args.train_seeds_per_family,
                                    args.dev_seeds_per_family, source_hashes)
        manifest = {
            "schema": "picoagent.native_observation.manifest.v3",
            "track": "luna-python-original-v1", "status": "in_progress",
            "execution": "native_teacher_observed", "teacher_model": None,
            "teacher_mode": "reviewed_procedural_replay",
            "teacher_decision_mode": "reviewed_procedural_replay", "sft_admissible": False,
            "source_sha256": source_hashes,
            "task_manifest_sha256": digest_file(output / "tasks/manifest.json"),
            "candidate_file_sha256": digest_file(output / "candidate_plans.reviewed_unexecuted.jsonl"),
            "configuration": {"train_seeds_per_family": args.train_seeds_per_family,
                              "dev_seeds_per_family": args.dev_seeds_per_family,
                              "test_execution": False},
            "runtime": runtime_metadata(child_env), "child_env": child_env,
            "launcher": {"tool": "functions.exec_command", "argv": sys.argv,
                         "cwd": str(Path.cwd()), "python_executable": sys.executable},
            "records": 0, "failures": 0, "success_by_split": {"train": 0, "dev": 0},
        }
        for filename in ("events.jsonl", "records.jsonl", "failures.jsonl", "progress.jsonl"):
            (output / filename).touch(exist_ok=False)
        manifest_path = output / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write((canonical_json(manifest) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    manifest_path = output / "manifest.json"
    events_path, records_path = output / "events.jsonl", output / "records.jsonl"
    failures_path, progress_path = output / "failures.jsonl", output / "progress.jsonl"
    candidate_by_task = {row["task_id"]: row for row in jsonl_rows(output / "candidate_plans.reviewed_unexecuted.jsonl")}
    if set(candidate_by_task) != {task["task_id"] for _, task in tasks}:
        raise ValueError("frozen candidates do not cover the exact frozen task set")
    previous_records = jsonl_rows(records_path)
    previous_failures = jsonl_rows(failures_path)
    success_ids = {row["task_id"] for row in previous_records if row.get("status") == "success"}
    manifest["records"] = len(previous_records)
    manifest["failures"] = len(previous_failures)
    manifest["success_by_split"] = {split: sum(row.get("split") == split and row.get("status") == "success"
                                               for row in previous_records) for split in ("train", "dev")}
    manifest["status"] = "in_progress"
    write_manifest(manifest_path, manifest)
    try:
        for split, task in tasks:
            if task["task_id"] in success_ids:
                continue
            candidate = candidate_by_task[task["task_id"]]
            try:
                record, summary = run_task(task, candidate, source_hashes, child_env, events_path)
                append_fsynced(records_path, record)
                manifest["records"] += 1
                if record["status"] == "success":
                    manifest["success_by_split"][split] += 1
                    success_ids.add(task["task_id"])
                else:
                    append_fsynced(failures_path, record)
                    manifest["failures"] += 1
                append_fsynced(progress_path, {"event": "task_completed", **summary,
                                               "record_sha256": content_hash(record),
                                               "record_index": manifest["records"] - 1})
            except Exception as exc:
                failure = {"task_id": task["task_id"], "family": task["family"],
                           "split": split, "status": "error", "execution": "native_teacher_observed",
                           "sft_admissible": False, "error_type": type(exc).__name__,
                           "message": str(exc)}
                append_fsynced(failures_path, failure)
                manifest["failures"] += 1
                append_fsynced(progress_path, {"event": "task_failed", "task_id": task["task_id"],
                                               "error_type": type(exc).__name__})
            if manifest["records"] % 25 == 0 or manifest["failures"] > 0:
                write_manifest(manifest_path, manifest)
        manifest["status"] = "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        append_fsynced(progress_path, {"event": "batch_interrupted", "completed_records": manifest["records"]})
        raise
    finally:
        manifest.update(events_sha256=digest_file(events_path),
                        records_sha256=digest_file(records_path),
                        failures_sha256=digest_file(failures_path),
                        progress_sha256=digest_file(progress_path))
        manifest["source_snapshots"] = {
            str(path.relative_to(output)): digest_file(path)
            for path in sorted((output / "source_snapshot").rglob("*")) if path.is_file()
        }
        write_manifest(manifest_path, manifest)
    print(canonical_json({"output": str(output), "status": manifest["status"],
                          "records": manifest["records"], "failures": manifest["failures"],
                          "successful_train": manifest["success_by_split"]["train"],
                          "successful_dev": manifest["success_by_split"]["dev"]}))


if __name__ == "__main__":
    main()

