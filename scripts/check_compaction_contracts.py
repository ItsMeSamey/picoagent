#!/usr/bin/env python3
"""Offline tokenizer/controller checks, never learned-model performance.

Run from repo root: PYTHONPATH=src:tests python scripts/check_compaction_contracts.py
--tokenizer PATH_TO_PINNED_SNAPSHOT --output data/compaction-modes-v1/contracts.json
"""
import argparse
import json
from pathlib import Path

from picoagent.data.audit import file_hash, write_new_json
from picoagent.data.compaction_curriculum import SPLIT_POLICY, audit_compaction_trace, generate_compaction_task
from picoagent.data.oracles import check_task_result
from picoagent.harness.protocol import render_messages
from picoagent.harness.tools import TOOL_SCHEMAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from test_compaction_curriculum import unit_rollout
    path = Path(args.tokenizer)
    revision = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
    if path.name != revision or not path.is_dir():
        raise ValueError("use the pinned local SmolLM2 tokenizer snapshot")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    def count(messages, tools=TOOL_SCHEMAS):
        return len(tokenizer.encode(render_messages(messages, tools, add_generation_prompt=True),
                                     add_special_tokens=False))
    records = []
    for mode in ("full", "half", "manual"):
        for family, split in sorted(SPLIT_POLICY.items()):
            if split == "test":
                continue
            task = generate_compaction_task(family, 12)
            result, trace, manager = unit_rollout(
                task, mode=mode, max_tokens=4096, reserve_tokens=768,
                token_counter=count, request_token_counter=lambda messages: count(messages, []))
            audit = audit_compaction_trace(trace, token_counter=count)
            accepted = [event for event in manager.events if event["type"] == "compaction"]
            record = {"mode": mode, "family": family, "split": split,
                      "task_id": task["task_id"], "stop_reason": result.stop_reason,
                      "fixture_oracle_passed": check_task_result(task, result.final)["passed"],
                      "semantic_audit_passed": audit["passed"], "compactions": len(accepted),
                      "max_summary_request_tokens": max(count(event["summary_request"], []) for event in accepted),
                      "max_summary_output_tokens": max(count(event["summary_request"] + [event["summary_response"]], []) - count(event["summary_request"], []) for event in accepted),
                      "max_summary_sft_tokens": max(count(event["summary_request"] + [event["summary_response"]], []) for event in accepted),
                      "error": result.error}
            if not record["fixture_oracle_passed"] or not audit["passed"] or record["max_summary_request_tokens"] > 3328 or record["max_summary_sft_tokens"] > 4096:
                raise ValueError(f"mode contract failed: {record}")
            records.append(record)
    report = {"schema": "picoagent.compaction_contracts.v1",
              "execution": "unexecuted_unit_fixture", "learned_model_performance": "not_measured",
              "training_eligible": False, "test_families_used": False,
              "max_tokens": 4096, "generation_reserve": 768,
              "tokenizer_revision": revision, "tokenizer_sha256": file_hash(path / "tokenizer.json"),
              "records": records, "source_sha256": {name: file_hash(name) for name in [
                  "scripts/check_compaction_contracts.py", "tests/test_compaction_curriculum.py",
                  "src/picoagent/harness/context.py", "src/picoagent/data/compaction_curriculum.py"]}}
    write_new_json(Path(args.output), report)
    print(json.dumps({"mode_cases": len(records), "passed": True, "learned_scores": None}))


if __name__ == "__main__":
    main()
