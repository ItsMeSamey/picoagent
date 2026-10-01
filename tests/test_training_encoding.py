import unittest

from picoagent.training.encoding import IGNORE_INDEX, encode_trace, mask_offsets
from picoagent.harness.protocol import render_segments


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        return {"input_ids": [ord(char) for char in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}


class TrainingEncodingTests(unittest.TestCase):
    def trace(self):
        return {"trace_id": "test", "tools": [], "messages": [
            {"role": "system", "content": "SYSTEM PRIVATE"},
            {"role": "user", "content": "USER PRIVATE"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "python", "arguments": '{"code":"print(1)"}'}}]},
            {"role": "tool", "tool_call_id": "a", "content": "TOOL OBSERVATION"},
            {"role": "assistant", "content": "FINAL ANSWER"}]}

    def test_assistant_only_including_tool_calls(self):
        row = self.trace()
        encoded = encode_trace(row, CharacterTokenizer(), 10000)
        supervised = "".join(chr(value) for value in encoded["labels"] if value != IGNORE_INDEX)
        for expected in ("print(1)", "FINAL ANSWER", "END_MESSAGE"):
            self.assertIn(expected, supervised)
        for excluded in ("SYSTEM PRIVATE", "USER PRIVATE", "TOOL OBSERVATION", "ASSISTANT:\n", "TOOLS:\n"):
            self.assertNotIn(excluded, supervised)
        expected = "".join(segment.text for segment in render_segments(row["messages"], tools=[]) if segment.trainable)
        self.assertEqual(supervised, expected)

    def test_boundaries_and_special_tokens_are_masked(self):
        self.assertEqual(mask_offsets([1, 2, 3, 4], [(0, 0), (3, 6), (6, 9), (9, 11)], [(5, 10)]),
                         [IGNORE_INDEX, IGNORE_INDEX, 3, IGNORE_INDEX])

    def test_no_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            encode_trace(self.trace(), CharacterTokenizer(), 10)

    def test_requires_fast_tokenizer(self):
        tokenizer = CharacterTokenizer()
        tokenizer.is_fast = False
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            encode_trace(self.trace(), tokenizer, 10000)

    def test_no_assistant_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "no trainable"):
            encode_trace({"trace_id": "empty", "tools": [], "messages": [{"role": "user", "content": "hello"}]}, CharacterTokenizer(), 10000)


if __name__ == "__main__":
    unittest.main()

class TrainingEventEncodingTests(unittest.TestCase):
    def test_exact_event_inputs_and_only_current_output_supervised(self):
        from picoagent.training.encoding import encode_records
        row = {"trace_id": "events", "tools": [], "messages": [], "model_events": [
            {"type": "assistant", "input_messages": [{"role": "user", "content": "old user"}, {"role": "assistant", "content": "OLD ANSWER"}, {"role": "user", "content": "next"}], "message": {"role": "assistant", "content": "NEW ANSWER"}},
            {"type": "compaction", "accepted": True, "summary_request": [{"role": "user", "content": "summarize"}], "summary_response": {"role": "assistant", "content": "SUMMARY ANSWER"}},
        ]}
        rows, stats = encode_records([row], CharacterTokenizer(), 10000)
        self.assertEqual(stats.examples, 2)
        labels = ["".join(chr(value) for value in item["labels"] if value != IGNORE_INDEX) for item in rows]
        self.assertIn("NEW ANSWER", labels[0])
        self.assertNotIn("OLD ANSWER", labels[0])
        self.assertIn("SUMMARY ANSWER", labels[1])
        self.assertIn("OLD ANSWER", "".join(chr(value) for value in rows[0]["input_ids"]))

    def test_compacted_trace_cannot_fall_back_to_final_messages(self):
        from picoagent.training.encoding import encode_records
        with self.assertRaisesRegex(ValueError, "exact model_events"):
            encode_records([{"trace_id": "bad", "messages": [], "provenance": {"context_compaction_enabled": True}}], CharacterTokenizer(), 10000)
