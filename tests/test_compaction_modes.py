"""Controller contract checks, not learned-model performance measurements."""
import json

import pytest

from picoagent.harness.context import ContextBudgetExceeded, ContextManager
from picoagent.harness.protocol import validate_conversation


def count(messages):
    return sum(len(message.get("content") or "") + 10 for message in messages)


def history():
    return [{"role": "system", "content": "fixed instructions"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"fact{i}=" + "x" * 250}
        for i in range(6)]


def test_full_summarizes_all_history_and_preserves_system():
    observed = []
    def model(messages, tools):
        observed.append(messages)
        return {"role": "assistant", "content": "goal retained; facts0..5 retained"}
    result = ContextManager(model, mode="full", max_tokens=4096,
                            token_counter=count).compact(history(), force=True)
    assert json.loads(observed[0][1]["content"]) == history()[1:]
    assert result.messages[0] == history()[0]
    assert result.event["retained_messages"] == []
    assert len(result.messages) == 2
    assert result.event["mode"] == "full"


def test_manual_model_can_choose_noncontiguous_groups():
    def model(messages, tools):
        groups = json.loads(messages[1]["content"])["groups"]
        assert [group["id"] for group in groups] == list(range(6))
        return {"role": "assistant", "content": json.dumps({
            "keep_groups": [5, 0], "summary": "facts1..4 retained"})}
    result = ContextManager(model, mode="manual", token_counter=count).compact(history(), force=True)
    assert result.event["keep_group_indices"] == [0, 5]
    assert result.event["retained_messages"] == [history()[1], history()[6]]
    assert result.messages[0] == history()[0]
    assert result.messages[2:] == [history()[1], history()[6]]
    validate_conversation(result.messages)


def test_manual_retains_tool_call_and_reply_as_one_unit():
    source = [{"role": "user", "content": "filler" * 300},
              {"role": "assistant", "content": "", "tool_calls": [
                  {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "/workspace"},
              {"role": "user", "content": "continue"}]
    def model(messages, tools):
        groups = json.loads(messages[1]["content"])["groups"]
        assert len(groups[1]["messages"]) == 2
        return {"role": "assistant", "content": '{"keep_groups":[1,2],"summary":"original goal"}'}
    result = ContextManager(model, mode="manual", token_counter=count).compact(source, force=True)
    assert result.messages[1:] == source[1:]
    validate_conversation(result.messages)


@pytest.mark.parametrize("ids", [[-1], [20], [True], [0, 0], "all"])
def test_invalid_manual_selection_is_preserved_and_rejected(ids):
    def model(messages, tools):
        return {"role": "assistant", "content": json.dumps({"keep_groups": ids, "summary": "x"})}
    manager = ContextManager(model, mode="manual", token_counter=count)
    with pytest.raises(ContextBudgetExceeded):
        manager.compact(history(), force=True)
    assert manager.events[-1]["type"] == "compaction_error"
    assert "raw_response_repr" in manager.events[-1]
    assert manager.events[-1]["before_messages"] == history()


@pytest.mark.parametrize("mode", ["full", "half", "manual"])
def test_compaction_request_budget_is_checked_before_model_call(mode):
    def never_called(messages, tools):
        raise AssertionError("model must not receive an over-budget compaction request")
    manager = ContextManager(never_called, mode=mode, token_counter=count,
                             request_token_counter=lambda messages: 100000)
    with pytest.raises(ContextBudgetExceeded, match="request exceeds"):
        manager.compact(history(), force=True)
    assert manager.events[-1]["type"] == "compaction_error"


def test_mode_must_be_explicit_valid_value():
    with pytest.raises(ValueError, match="mode"):
        ContextManager(lambda messages, tools: {}, mode="truncate")
