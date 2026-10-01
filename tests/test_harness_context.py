import json
from pathlib import Path
import tempfile
import unittest

from picoagent.harness import ContextBudgetExceeded, ContextManager, validate_conversation


def count(messages):
    return sum(len(message.get("content") or "") + 10 for message in messages)


class ContextTests(unittest.TestCase):
    def test_midpoint_uses_tokens_instead_of_message_count(self):
        source = [{"role": "system", "content": "rules"},
                  {"role": "user", "content": "a" * 800}] + [
                      {"role": "assistant" if i % 2 == 0 else "user", "content": "b" * 20}
                      for i in range(6)]
        manager = ContextManager(lambda m, t: {"role": "assistant", "content": "brief"},
                                 max_tokens=1000, reserve_tokens=100, token_counter=count)
        result = manager.compact(source, force=True)
        self.assertEqual(result.event["split_index"], 2)
        self.assertEqual(result.event["source_messages"], source[1:2])
        self.assertEqual(result.messages[2:], source[2:])

    def test_summary_replaces_first_half_and_suffix_is_untouched(self):
        source = [{"role": "system", "content": "rules"}] + [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i) + "x" * 200} for i in range(8)]
        observed = []
        def model(messages, tools):
            observed.append((messages, tools))
            return {"role": "assistant", "content": "remember exact facts"}
        with tempfile.TemporaryDirectory() as root:
            manager = ContextManager(model, max_tokens=1600, reserve_tokens=100, token_counter=count, trace_path=Path(root) / "trace.jsonl")
            result = manager.compact(source)
            self.assertIsNotNone(result.event)
            self.assertEqual(result.messages[0], source[0])
            self.assertEqual(result.messages[2:], source[5:])
            self.assertEqual(result.event["source_messages"], source[1:5])
            self.assertEqual(observed[0][1], [])
            trace = json.loads((Path(root) / "trace.jsonl").read_text())
            self.assertEqual(trace["summary_response"]["content"], "remember exact facts")
            self.assertEqual(source[1]["content"], "0" + "x" * 200)

    def test_split_never_orphans_a_tool_result(self):
        calls = [{"id": "c" + str(i), "type": "function", "function": {"name": "knowledge", "arguments": '{}'}} for i in range(3)]
        source = [{"role": "user", "content": "x" * 1000}, {"role": "assistant", "content": None, "tool_calls": calls}] + [{"role": "tool", "tool_call_id": call["id"], "content": "result"} for call in calls] + [{"role": "user", "content": "recent"}]
        manager = ContextManager(lambda m, t: {"role": "assistant", "content": "short"}, max_tokens=1000, reserve_tokens=100, token_counter=count)
        result = manager.compact(source, force=True)
        validate_conversation(result.messages)
        self.assertEqual(result.event["source_messages"], source[:1])
        self.assertEqual(result.event["retained_messages"], source[1:])

    def test_nonshrinking_summary_is_rejected_and_recorded(self):
        manager = ContextManager(lambda m, t: {"role": "assistant", "content": "x" * 5000}, max_tokens=100, reserve_tokens=20, token_counter=count)
        with self.assertRaises(ContextBudgetExceeded):
            manager.fit([{"role": "user", "content": "a" * 60}, {"role": "assistant", "content": "b" * 60}])
        self.assertFalse(manager.events[-1]["accepted"])

    def test_noop_when_under_budget_and_failure_for_single_giant_message(self):
        manager = ContextManager(lambda m, t: self.fail("model should not run"), max_tokens=1000, reserve_tokens=100, token_counter=count)
        self.assertIsNone(manager.compact([{"role": "user", "content": "hi"}]).event)
        with self.assertRaises(ContextBudgetExceeded):
            manager.fit([{"role": "user", "content": "x" * 2000}])
