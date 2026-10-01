"""Paired, family-balanced evaluation of the three context modes."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
from typing import Callable

MODES = ("full", "half", "manual")


def summarize(records: list[dict]) -> dict:
    """Incomplete pairs fail closed; failures remain in every denominator.

    Macro-family success prevents thousands of seeded variants in one easy
    family from hiding weak coverage elsewhere. These are internal curriculum
    metrics, never public benchmark scores.
    """
    if not records:
        raise ValueError("evaluation requires paired records")
    keys, cases, families = set(), defaultdict(set), {}
    splits, checkpoints, evidence = set(), set(), set()
    for row in records:
        mode, task = row["mode"], row["task_id"]
        if mode not in MODES or (mode, task) in keys:
            raise ValueError("unknown mode or duplicate task/mode result")
        for field in ("task_success", "protocol_valid"):
            if type(row[field]) is not bool:
                raise ValueError(f"{field} must be boolean")
        if type(row["compactions"]) is not int or row["compactions"] < 0:
            raise ValueError("compaction count must be nonnegative")
        for field in ("compaction_attempts", "compaction_failures", "model_invocations",
                      "model_responses", "model_errors", "context_overflows", "attempt_errors"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if type(row["input_tokens"]) is not int or row["input_tokens"] < 0:
            raise ValueError("token count must be nonnegative")
        if row["split"] not in {"dev", "test"}:
            raise ValueError("evaluation cannot use training cases")
        if task in families and families[task] != row["family"]:
            raise ValueError("paired case changed family")
        families[task] = row["family"]
        keys.add((mode, task))
        cases[task].add(mode)
        splits.add(row["split"])
        checkpoints.add(row["checkpoint"])
        evidence.add(row["execution"])
    if any(modes != set(MODES) for modes in cases.values()):
        raise ValueError("every task must run in all three modes")
    if len(splits) != 1 or len(checkpoints) != 1:
        raise ValueError("paired report requires one split and one checkpoint")
    per_mode = {}
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        by_family = defaultdict(list)
        for row in rows:
            by_family[row["family"]].append(row["task_success"])
        family_scores = {family: mean(values) for family, values in sorted(by_family.items())}
        task_count = len(rows)
        compacted = sum(row["compactions"] > 0 for row in rows)
        invocation_tasks = sum(row["model_invocations"] > 0 for row in rows)
        per_mode[mode] = {
            "tasks": task_count, "successes": sum(row["task_success"] for row in rows),
            "task_failures": task_count - sum(row["task_success"] for row in rows),
            "task_success": mean(row["task_success"] for row in rows),
            "macro_family_success": mean(family_scores.values()), "families": family_scores,
            "protocol_valid": mean(row["protocol_valid"] for row in rows),
            "protocol_failures": task_count - sum(row["protocol_valid"] for row in rows),
            "compaction_exercised": compacted / task_count,
            "compaction_exercised_tasks": compacted,
            "compaction_attempt_tasks": sum(row["compaction_attempts"] > 0 for row in rows),
            "compaction_failure_tasks": sum(row["compaction_failures"] > 0 for row in rows),
            "model_invocation_tasks": invocation_tasks,
            "model_invocation_coverage": invocation_tasks / task_count,
            "model_invocations": sum(row["model_invocations"] for row in rows),
            "model_responses": sum(row["model_responses"] for row in rows),
            "model_error_tasks": sum(row["model_errors"] > 0 for row in rows),
            "context_overflow_tasks": sum(row["context_overflows"] > 0 for row in rows),
            "attempt_error_tasks": sum(row["attempt_errors"] > 0 for row in rows),
            "mean_compactions": mean(row["compactions"] for row in rows),
            "mean_input_tokens": mean(row["input_tokens"] for row in rows),
        }
    scores = [value["macro_family_success"] for value in per_mode.values()]
    learned_eligible = (evidence == {"verified_environment"} and
                        all(value["model_invocation_coverage"] == 1 for value in per_mode.values()))
    return {"schema": "picoagent.mode_evaluation.v1", "split": next(iter(splits)),
            "checkpoint": next(iter(checkpoints)), "paired_tasks": len(cases),
            "execution": sorted(evidence), "per_mode": per_mode,
            "worst_mode_macro_success": min(scores), "mean_mode_macro_success": mean(scores),
            "learned_model_score_verified": learned_eligible}


def selection_key(report: dict) -> tuple[float, float]:
    """Use only actual dev runs; optimize the weakest mode, then the mean."""
    if report["split"] != "dev" or not report["learned_model_score_verified"]:
        raise ValueError("checkpoint selection requires verified dev evaluation")
    if any(mode["compaction_exercised"] != 1 for mode in report["per_mode"].values()):
        raise ValueError("mode selection suite must exercise compaction on every task")
    if any(mode["model_invocation_coverage"] != 1 for mode in report["per_mode"].values()):
        raise ValueError("mode selection suite must invoke the policy on every task")
    return report["worst_mode_macro_success"], report["mean_mode_macro_success"]


def _attempt_call_counts(attempt_path) -> tuple[int, int, int, list[str], list[str], list[dict]]:
    """Return durable callback and compaction counts/errors from the attempt log.

    These come from the attempt's append-only callback journal, not inferred from
    successful assistant messages. A malformed reply and a generation exception
    therefore remain visible in the same per-task denominator.
    """
    path = Path(attempt_path) / "events.jsonl"
    if not path.is_file():
        return 0, 0, 0, [], [], []
    requests = responses = compaction_requests = 0
    errors, compaction_errors = [], []
    request_payloads = []
    pending_compaction = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind, payload = event.get("kind"), event.get("payload", {})
            if kind == "model_request":
                requests += 1
                request_payloads.append(payload)
                pending_compaction = payload.get("tools") == []
                if pending_compaction:
                    compaction_requests += 1
            elif kind == "model_response":
                responses += 1
                pending_compaction = False
            elif kind == "model_exception":
                message = str(payload.get("message", "model exception"))
                errors.append(message)
                if pending_compaction:
                    compaction_errors.append(message)
                pending_compaction = False
    return requests, responses, compaction_requests, errors, compaction_errors, request_payloads


def evaluate(tasks: list[dict], *, policy: Callable, checkpoint: str, archive_root,
             image: str, runtime: str | None = None, max_tokens: int = 4096,
             reserve_tokens: int = 768, max_steps: int = 32, unlock_test: bool = False) -> dict:
    """Actual isolated rollouts, same tasks and model in each mode.

    Policy must be a stateless deterministic learned-model callback with the
    shared count_tokens method. Private fixture oracles go only to the collector.
    No scripted teacher fallback, command substitution, or answer repair.
    """
    from pathlib import Path
    import json
    from picoagent.data.collector import collect_task
    from picoagent.harness.tools import TOOL_SCHEMAS
    if not tasks or not checkpoint.strip() or not callable(getattr(policy, "count_tokens", None)):
        raise ValueError("tasks, checkpoint identity and policy tokenizer are required")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("duplicate evaluation task")
    splits = {task["split"] for task in tasks}
    if splits not in ({"dev"}, {"test"}) or (splits == {"test"} and not unlock_test):
        raise ValueError("use dev only; final test requires explicit unlock_test")
    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in tasks:
        for mode in MODES:
            result = collect_task(task, root / mode, model=policy,
                                  teacher_name=f"learned_policy:{checkpoint}", image=image,
                                  runtime=runtime, max_steps=max_steps,
                                  context_mode=mode, context_max_tokens=max_tokens,
                                  context_reserve_tokens=reserve_tokens,
                                  token_counter=lambda messages: policy.count_tokens(messages, TOOL_SCHEMAS),
                                  request_token_counter=lambda messages: policy.count_tokens(messages, []))
            trace = result["trace"]
            raw = json.loads((Path(result["attempt_path"]) / "raw.json").read_text())
            rollout = raw.get("result") or {}
            rollout_events = rollout.get("events", [])
            (request_count, response_count, compaction_requests, callback_errors,
             callback_compaction_errors, request_payloads) = _attempt_call_counts(result["attempt_path"])
            model_errors = [event.get("message", "model protocol error") for event in rollout_events
                            if event.get("type") == "model_error"] + callback_errors
            compaction_errors = [event.get("error", "compaction error") for event in rollout_events
                                 if event.get("type") == "compaction_error"] + callback_compaction_errors
            stop_reason = rollout.get("stop_reason", "no_result")
            raw_error = raw.get("error") or {}
            compaction_failed = bool(compaction_errors) or stop_reason == "context_budget"
            failure_text = " ".join(str(value) for value in [*model_errors, *compaction_errors,
                                                              raw_error.get("message", "")])
            context_overflow = (stop_reason == "context_budget" or bool(re.search(
                r"context|position|token.{0,20}(?:limit|exceed)|exceed.{0,20}(?:context|token)",
                failure_text, re.IGNORECASE)))
            protocol_valid = (bool(rollout) and stop_reason != "context_budget" and
                              not model_errors and not compaction_errors)
            # Count every actual callback request, including requests that ended
            # in a parser/context exception before an assistant event was emitted.
            inputs = sum(policy.count_tokens(payload["messages"], payload.get("tools", []))
                         for payload in request_payloads if "messages" in payload)
            model_events = trace.get("model_events", [])
            row = {"task_id": task["task_id"], "family": task["family"], "split": task["split"],
                   "mode": mode, "checkpoint": checkpoint,
                   "execution": trace["provenance"]["execution"],
                   "task_success": trace["status"] == "success" and trace["verification"]["passed"],
                   "protocol_valid": protocol_valid,
                   "compactions": sum(event["type"] == "compaction" and event["accepted"] for event in model_events),
                   "compaction_attempts": compaction_requests,
                   "compaction_failures": int(compaction_failed),
                   "model_invocations": request_count,
                   "model_responses": response_count,
                   "model_errors": int(bool(model_errors)),
                   "context_overflows": int(context_overflow),
                   "attempt_errors": int(bool(raw_error)),
                   "input_tokens": inputs, "attempt": str(result["attempt_path"])}
            rows.append(row)
            from picoagent.harness.context import append_trace
            append_trace(root / "results.jsonl", row)
    report = summarize(rows)
    from picoagent.data.schema import content_hash
    report["conditions"] = {"task_specs_sha256": content_hash(tasks), "image": image,
                            "runtime": runtime, "context": max_tokens,
                            "generation_reserve": reserve_tokens, "max_steps": max_steps,
                            "policy_seed": getattr(policy, "seed", None)}
    from picoagent.data.audit import write_new_json
    write_new_json(root / "report.json", report)
    return report
