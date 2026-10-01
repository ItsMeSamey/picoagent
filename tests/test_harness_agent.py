import json
from pathlib import Path
import tempfile
import unittest

from picoagent.harness import AgentHarness, KnowledgeStore, ToolRegistry, validate_conversation


class AgentTests(unittest.TestCase):
    def test_end_to_end_tools_without_host_code(self):
        responses = iter([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "k1", "function": {"name": "knowledge", "arguments": {"operation": "set", "key": "result", "value": 42}}}]},
            {"role": "assistant", "content": "42"},
        ])
        seen = []
        def model(messages, tools):
            seen.append((messages, tools))
            return next(responses)
        with tempfile.TemporaryDirectory() as root:
            tools = ToolRegistry(None, KnowledgeStore(Path(root) / "kv.json"))
            result = AgentHarness(model, tools, trace_path=Path(root) / "trace.jsonl").run("Remember 42")
            self.assertEqual(result.final, "42")
            self.assertEqual(result.stop_reason, "final")
            self.assertEqual(tools.knowledge.get("result"), 42)
            self.assertEqual(seen[1][0][-1]["role"], "tool")
            self.assertEqual(result.events[0]["input_messages"][-1]["content"], "Remember 42")
            self.assertTrue(result.events[1]["verified"])
            validate_conversation(result.messages)
            self.assertEqual(len((Path(root) / "trace.jsonl").read_text().splitlines()), 3)

    def test_invalid_calls_are_recoverable_tool_results(self):
        responses = iter([
            {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "unknown", "arguments": "{}"}}]},
            {"role": "assistant", "content": "Tool unavailable"},
        ])
        with tempfile.TemporaryDirectory() as root:
            result = AgentHarness(lambda m, t: next(responses), ToolRegistry(None, KnowledgeStore(Path(root) / "kv.json"))).run("task")
            self.assertEqual(result.stop_reason, "final")
            self.assertFalse(result.events[1]["verified"])
            self.assertEqual(json.loads(result.messages[-2]["content"])["error"], "ValueError")

    def test_bad_role_duplicate_calls_and_step_limit(self):
        with tempfile.TemporaryDirectory() as root:
            tools = ToolRegistry(None, KnowledgeStore(Path(root) / "kv.json"))
            result = AgentHarness(lambda m, t: {"role": "user", "content": "x"}, tools).run("task")
            self.assertEqual(result.stop_reason, "invalid_model_output")
            model = lambda m, t: {"role": "assistant", "tool_calls": [{"id": "same", "function": {"name": "knowledge", "arguments": '{"operation":"list"}'}}]}
            result = AgentHarness(model, tools, max_steps=1).run("task")
            self.assertEqual(result.stop_reason, "max_steps")
            validate_conversation(result.messages)
            result = AgentHarness(model, tools, max_steps=2).run("task")
            self.assertEqual(result.stop_reason, "invalid_model_output")
            self.assertEqual(len([e for e in result.events if e["type"] == "tool_execution"]), 1)
