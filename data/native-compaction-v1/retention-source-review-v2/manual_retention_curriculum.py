"""Separate mixed-length tasks and visible-prefix, tokenizer-aware retention.

The selector knows the fixed protocol/tokenizer configuration, but receives no
task object, oracle, future files or hidden conversation state. Its retention
policy is deliberately narrow: keep the latest complete tool group if it fits.
"""
from __future__ import annotations

import json

from picoagent.harness.context import MANUAL_INSTRUCTION, SUMMARY_PREFIX
from picoagent.harness.protocol import END_MESSAGE, render_messages
from picoagent.harness.tools import TOOL_SCHEMAS

from .compaction_curriculum import GOAL_MARKER, VisibleContextTeacher
from .compaction_curriculum import generate_compaction_task, visible_memory
from .generators import SYSTEM_PROMPT
from .schema import canonical_json, content_hash, validate_messages, validate_task

SOURCE_ID = "native_compaction_retention"
VERSION = "manual_retention_mixed_v2"
FAMILIES = ("compaction.running_balance", "compaction.latest_status",
            "compaction.threshold_count", "compaction.first_status")


def generate_retention_task(family: str, seed: int) -> dict:
    if family not in FAMILIES:
        raise ValueError("retention extension contains train/dev families only")
    task = generate_compaction_task(family, seed)
    old_id = task["task_id"]
    task["family"] = family.replace("compaction.", "retention.", 1)
    task["task_id"] = f"{task['family']}:{seed:08d}:h16:mixed-v2"
    task["base_task_id"] = task["task_id"]
    task["template_id"] = task["family"] + ".v2"
    prefix, serialized = task["prompt"].split(GOAL_MARKER, 1)
    goal = json.loads(serialized)
    goal["task_id"] = task["task_id"]
    task["prompt"] = prefix + GOAL_MARKER + canonical_json(goal)
    for path, text in task["environment"]["files"].items():
        packet = json.loads(text)
        if packet["task_id"] != old_id:
            raise ValueError("base fixture identity mismatch")
        packet["task_id"] = task["task_id"]
        # Long observations force real compaction. Two shorter observations
        # between them provide useful recent atomic groups that can fit intact.
        words = 96 if packet["index"] % 3 == 0 else 32
        packet["audit_detail"] = " ".join(packet["audit_detail"].split()[:words])
        task["environment"]["files"][path] = canonical_json(packet) + "\n"
    task["provenance"]["curriculum_track"] = VERSION
    task["provenance"]["origin"] = "Original mixed-length train/dev retention tasks; no benchmarks"
    task["compaction"].update(detail_words="mixed_96_32_32", manual_selection="tokenizer_aware_latest_group")
    task["reference"] = {"teacher": "TokenizerAwareRetentionTeacher", "source": "model_visible_context_only"}
    task["input_sha256"] = content_hash({"prompt": task["prompt"], "environment": task["environment"]})
    validate_task(task)
    return task


class TokenizerAwareRetentionTeacher(VisibleContextTeacher):
    """Select a fitting contiguous recent group, using only visible history.

    The known static system prompt and tool schemas are protocol overhead, not
    task-specific information. They match the fixed native collection harness.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        if not (not tools and len(messages) == 2
                and messages[0].get("content", "").startswith(MANUAL_INSTRUCTION)):
            return super().__call__(messages, tools)
        payload = json.loads(messages[1]["content"])
        groups = payload["groups"]
        selections = []
        if len(groups) > 1 and groups[-1]["messages"][-1].get("role") == "tool":
            selections.append([groups[-1]["id"]])
        selections.append([])
        for keep in selections:
            past = [message for group in groups if group["id"] not in keep for message in group["messages"]]
            retained = [message for group in groups if group["id"] in keep for message in group["messages"]]
            validate_messages(past)
            memory = canonical_json(visible_memory(past))
            response = {"role": "assistant", "content": canonical_json({"keep_groups": keep, "summary": memory})}
            result = [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": SUMMARY_PREFIX + memory}] + retained
            count = len(self.tokenizer.encode(render_messages(result, TOOL_SCHEMAS, add_generation_prompt=True),
                                              add_special_tokens=False))
            target_tokens = len(self.tokenizer.encode(canonical_json(response) + END_MESSAGE,
                                                      add_special_tokens=False))
            if count <= payload["retained_context_budget"] and target_tokens <= 768:
                return response
        raise ValueError("visible-prefix summary/retention cannot fit the fixed 4K/768 contract")
