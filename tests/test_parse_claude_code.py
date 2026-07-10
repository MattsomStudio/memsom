#!/usr/bin/env python3
"""Phase 3 — Claude Code transcript parser.

Run:  python -m unittest -v test_parse_claude_code
"""

import json
import tempfile
import unittest
from pathlib import Path

from memsom.bridge import chats as memsom_chats


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


class TestParseClaudeCode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "session.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_keeps_only_user_assistant(self):
        write_jsonl(self.path, [
            {"type": "queue-operation", "operation": "x"},
            {"type": "user", "message": {"role": "user", "content": "hello there"}},
            {"type": "ai-title", "title": "noise"},
            {"type": "attachment", "data": "noise"},
            {"type": "assistant", "message": {"role": "assistant",
                                              "content": [{"type": "text", "text": "hi back"}]}},
        ])
        recs = memsom_chats.parse_file("claude-code", self.path)
        self.assertEqual([r["role"] for r in recs], ["user", "assistant"])
        self.assertEqual([r["text"] for r in recs], ["hello there", "hi back"])

    def test_content_string_form(self):
        write_jsonl(self.path, [
            {"type": "user", "message": {"role": "user", "content": "a plain string message"}},
        ])
        recs = memsom_chats.parse_file("claude-code", self.path)
        self.assertEqual(recs[0]["text"], "a plain string message")

    def test_content_block_list_form(self):
        # text blocks concatenated; non-text blocks (tool_use) dropped.
        write_jsonl(self.path, [
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "text", "text": "first part"},
                {"type": "tool_use", "name": "bash", "input": {"cmd": "ls"}},
                {"type": "text", "text": "second part"},
            ]}},
        ])
        recs = memsom_chats.parse_file("claude-code", self.path)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["text"], "first part\nsecond part")

    def test_source_ref_has_line_number(self):
        write_jsonl(self.path, [
            {"type": "user", "message": {"role": "user", "content": "x"}},
        ])
        recs = memsom_chats.parse_file("claude-code", self.path)
        self.assertTrue(recs[0]["source_ref"].endswith("#L1"))


if __name__ == "__main__":
    unittest.main()
