"""Deterministic, tokenizer-independent protocol shared by SFT and inference."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from typing import Any, Mapping, Sequence

Message = dict[str, Any]
ROLES = frozenset({"system", "user", "assistant", "tool"})
END_MESSAGE = "\nEND_MESSAGE\n"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_message(message: Mapping[str, Any]) -> Message:
    """Validate message shape without silently dropping training information."""
    value = copy.deepcopy(dict(message))
    if value.get("role") not in ROLES:
        raise ValueError("message role must be system, user, assistant, or tool")
    if not isinstance(value.get("content", ""), (str, type(None))):
        raise ValueError("content must be a string or null")
    value.setdefault("content", "")
    calls = value.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    if calls:
        if value["role"] != "assistant":
            raise ValueError("tool_calls belong to assistant messages")
        ids: set[str] = set()
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                raise ValueError("tool call needs a nonempty string id")
            if call["id"] in ids:
                raise ValueError("duplicate tool call id")
            ids.add(call["id"])
            if call.get("type", "function") != "function":
                raise ValueError("only function tool calls are supported")
            fn = call.get("function", {})
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise ValueError("tool call needs function name")
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = canonical_json(fn["arguments"])
            if not isinstance(fn.get("arguments"), str):
                raise ValueError("tool arguments must be a JSON string or object")
            call.setdefault("type", "function")
    if value["role"] == "tool" and not isinstance(value.get("tool_call_id"), str):
        raise ValueError("tool message needs tool_call_id")
    canonical_json(value)  # Reject non-serializable values and NaN.
    return value


def validate_conversation(messages: Sequence[Mapping[str, Any]], *, allow_pending: bool = False) -> list[Message]:
    """Require each assistant tool group to be followed by exactly its replies."""
    result = [validate_message(message) for message in messages]
    pending: set[str] = set()
    seen: set[str] = set()
    for message in result:
        if message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending:
                raise ValueError("orphan, duplicate, or out-of-order tool result")
            pending.remove(call_id)
        else:
            if pending:
                raise ValueError("assistant tool calls must receive all results before another message")
            for call in message.get("tool_calls", []):
                if call["id"] in seen:
                    raise ValueError("tool call ids must be unique throughout a conversation")
                seen.add(call["id"])
                pending.add(call["id"])
    if pending and not allow_pending:
        raise ValueError("conversation ends with pending tool calls")
    return result


@dataclass(frozen=True)
class TextSegment:
    text: str
    trainable: bool
    role: str


def render_message(message: Mapping[str, Any]) -> str:
    value = validate_message(message)
    return value["role"].upper() + ":\n" + canonical_json(value) + END_MESSAGE


def render_segments(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None) -> list[TextSegment]:
    segments: list[TextSegment] = []
    if tools is not None:
        segments.append(TextSegment("TOOLS:\n" + canonical_json(list(tools)) + "\nEND_TOOLS\n", False, "system"))
    for message in messages:
        value = validate_message(message)
        role = value["role"]
        segments.append(TextSegment(role.upper() + ":\n", False, role))
        segments.append(TextSegment(canonical_json(value) + END_MESSAGE, role == "assistant", role))
    return segments


def render_messages(messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None, *, add_generation_prompt: bool = False) -> str:
    text = "".join(segment.text for segment in render_segments(messages, tools))
    return text + ("ASSISTANT:\n" if add_generation_prompt else "")


def parse_assistant(text: str) -> Message:
    """Parse exactly one generated JSON assistant message, never tool output."""
    text = text.strip()
    if text.startswith("ASSISTANT:\n"):
        text = text[len("ASSISTANT:\n"):]
    value, end = json.JSONDecoder().raw_decode(text)
    if text[end:].strip() not in {"", "END_MESSAGE"}:
        raise ValueError("unexpected text after assistant message")
    if not isinstance(value, dict):
        raise ValueError("assistant output must be a JSON object")
    result = validate_message(value)
    if result["role"] != "assistant":
        raise ValueError("model output must have assistant role")
    return result
