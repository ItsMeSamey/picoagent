"""Assistant-only SFT labels for the shared inference wire protocol.

Token offsets are checked against exact rendered character spans. Tokens crossing
role boundaries are masked rather than accidentally supervising user/tool input.
No packed examples and no silent truncation are used in this baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncodingStats:
    examples: int
    total_tokens: int
    assistant_tokens: int
    maximum_length: int

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def mask_offsets(input_ids: list[int], offsets: list[tuple[int, int]], assistant_spans: list[tuple[int, int]]) -> list[int]:
    """Label only tokens wholly inside an assistant span; (0, 0) specials mask."""
    if len(input_ids) != len(offsets):
        raise ValueError("Token IDs and offset mappings differ in length")
    labels = []
    for token_id, (start, end) in zip(input_ids, offsets):
        labels.append(token_id if end > start and any(a <= start and end <= b for a, b in assistant_spans) else IGNORE_INDEX)
    return labels


def encode_trace(record: dict[str, Any], tokenizer: Any, max_seq_length: int, *, supervise_last_only: bool = False) -> dict[str, list[int]]:
    from picoagent.harness.protocol import render_segments
    from picoagent.harness.tools import TOOL_SCHEMAS

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Assistant-only offset masking requires a fast tokenizer")
    segments = render_segments(record["messages"], tools=record.get("tools", TOOL_SCHEMAS))
    parts, spans, cursor = [], [], 0
    for index, segment in enumerate(segments):
        parts.append(segment.text)
        if segment.trainable and (not supervise_last_only or index == len(segments) - 1):
            spans.append((cursor, cursor + len(segment.text)))
        cursor += len(segment.text)
    text = "".join(parts)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if len(ids) > max_seq_length:
        raise ValueError(f"Trace {record['trace_id']} has {len(ids)} tokens > max_seq_length={max_seq_length}; shorten upstream or increase the frozen config (no silent truncation)")
    labels = mask_offsets(ids, offsets, spans)
    # The causal loss shifts labels by one, so a supervised position at index 0
    # alone would not actually train anything.
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError(f"Trace {record['trace_id']} contains no trainable assistant tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def event_examples(record: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    """Expand exact model inputs, never infer old actions from a compacted tail."""
    from picoagent.harness.protocol import validate_conversation
    from picoagent.harness.tools import TOOL_SCHEMAS

    events = record.get("model_events")
    compacted = record.get("provenance", {}).get("context_compaction_enabled", False)
    if events is None:
        if compacted:
            raise ValueError("Compacted traces require exact model_events; final messages cannot reconstruct training inputs")
        return [(record, False)]
    if not isinstance(events, list) or not events:
        raise ValueError("model_events must be a nonempty array")
    examples = []
    for index, event in enumerate(events):
        if event.get("type") == "assistant":
            inputs, output = event.get("input_messages"), event.get("message")
            tools = record.get("tools", TOOL_SCHEMAS)
        elif event.get("type") == "compaction" and event.get("accepted") is True:
            inputs, output = event.get("summary_request"), event.get("summary_response")
            tools = []
            if not isinstance(output, dict) or output.get("tool_calls"):
                raise ValueError("Compaction supervision requires a plain assistant summary")
        else:
            continue
        if not isinstance(inputs, list) or not isinstance(output, dict) or output.get("role") != "assistant":
            raise ValueError("Model events must preserve exact input_messages and assistant output")
        messages = validate_conversation(inputs + [output], allow_pending=True)
        examples.append(({"trace_id": f"{record['trace_id']}:event-{index}", "messages": messages, "tools": tools}, True))
    if not examples:
        raise ValueError("No trainable assistant/accepted compaction events")
    return examples


def encode_records(records: Iterable[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> tuple[list[dict[str, list[int]]], EncodingStats]:
    encoded = []
    for record in records:
        examples = event_examples(record)
        for example, last_only in examples:
            encoded.append(encode_trace(example, tokenizer, max_seq_length, supervise_last_only=last_only))
    return encoded, EncodingStats(
        examples=len(encoded),
        total_tokens=sum(len(row["input_ids"]) for row in encoded),
        assistant_tokens=sum(sum(value != IGNORE_INDEX for value in row["labels"]) for row in encoded),
        maximum_length=max((len(row["input_ids"]) for row in encoded), default=0),
    )


class AssistantOnlyCollator:
    """Right-pad inputs and labels separately, keeping all padding out of loss."""

    def __init__(self, pad_token_id: int, *, pad_to_length: int | None = None):
        self.pad_token_id = pad_token_id
        self.pad_to_length = pad_to_length

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not examples:
            raise ValueError("Cannot collate an empty batch")
        length = max(len(row["input_ids"]) for row in examples)
        if self.pad_to_length is not None:
            if length > self.pad_to_length:
                raise ValueError("Example exceeds fixed padding length")
            length = self.pad_to_length
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in examples:
            padding = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [IGNORE_INDEX] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}
