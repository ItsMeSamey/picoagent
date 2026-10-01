"""Paired, family-balanced evaluation of the three context modes."""
from __future__ import annotations

from collections import defaultdict
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
        per_mode[mode] = {
            "tasks": len(rows), "successes": sum(row["task_success"] for row in rows),
            "task_success": mean(row["task_success"] for row in rows),
            "macro_family_success": mean(family_scores.values()), "families": family_scores,
            "protocol_valid": mean(row["protocol_valid"] for row in rows),
            "compaction_exercised": mean(row["compactions"] > 0 for row in rows),
            "mean_compactions": mean(row["compactions"] for row in rows),
            "mean_input_tokens": mean(row["input_tokens"] for row in rows),
        }
    scores = [value["macro_family_success"] for value in per_mode.values()]
    return {"schema": "picoagent.mode_evaluation.v1", "split": next(iter(splits)),
            "checkpoint": next(iter(checkpoints)), "paired_tasks": len(cases),
            "execution": sorted(evidence), "per_mode": per_mode,
            "worst_mode_macro_success": min(scores), "mean_mode_macro_success": mean(scores),
            "learned_model_score_verified": evidence == {"verified_environment"}}


def selection_key(report: dict) -> tuple[float, float]:
    """Use only actual dev runs; optimize the weakest mode, then the mean."""
    if report["split"] != "dev" or not report["learned_model_score_verified"]:
        raise ValueError("checkpoint selection requires verified dev evaluation")
    if any(mode["compaction_exercised"] != 1 for mode in report["per_mode"].values()):
        raise ValueError("mode selection suite must exercise compaction on every task")
    return report["worst_mode_macro_success"], report["mean_mode_macro_success"]


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
            protocol_valid = bool(rollout) and not any(
                event["type"] in {"model_error", "compaction_error"}
                for event in rollout.get("events", []))
            events = trace.get("model_events", [])
            inputs = sum(policy.count_tokens(event["input_messages"], TOOL_SCHEMAS)
                         if event["type"] == "assistant" else
                         policy.count_tokens(event["summary_request"], []) for event in events)
            row = {"task_id": task["task_id"], "family": task["family"], "split": task["split"],
                   "mode": mode, "checkpoint": checkpoint,
                   "execution": trace["provenance"]["execution"],
                   "task_success": trace["status"] == "success" and trace["verification"]["passed"],
                   "protocol_valid": protocol_valid,
                   "compactions": sum(event["type"] == "compaction" and event["accepted"] for event in events),
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
