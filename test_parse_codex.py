#!/usr/bin/env python3
"""Phase 3 — Codex rollout parser. Verified against a real rollout file.

Run:  python -m unittest -v test_parse_codex
"""

import json
import tempfile
import unittest
from pathlib import Path

import memsom_chats


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _user_msg(text):
    return {"type": "response_item",
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]}}


def _assistant_msg(text):
    return {"type": "response_item",
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}


def _event_mirror(role, text):
    kind = "user_message" if role == "user" else "agent_message"
    return {"type": "event_msg", "payload": {"type": kind, "message": text}}


class TestParseCodex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "rollout-x.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def test_keeps_response_item_messages(self):
        write_jsonl(self.path, [
            {"type": "session_meta", "payload": {"id": "abc"}},
            _user_msg("make an svg of a pelican"),
            _assistant_msg("done, saved it"),
        ])
        recs = memsom_chats.parse_file("codex", self.path)
        self.assertEqual([r["role"] for r in recs], ["user", "assistant"])
        self.assertEqual([r["text"] for r in recs],
                         ["make an svg of a pelican", "done, saved it"])

    def test_drops_event_msg_mirror(self):
        # Every message is written twice: response_item (canonical) + event_msg
        # (mirror). The parser must drop the mirror at the type gate so the
        # corpus is not doubled — proven by getting exactly ONE record back.
        write_jsonl(self.path, [
            _user_msg("only once please"),
            _event_mirror("user", "only once please"),
        ])
        recs = memsom_chats.parse_file("codex", self.path)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["text"], "only once please")

    def test_drops_tool_and_meta_lines(self):
        write_jsonl(self.path, [
            {"type": "session_meta", "payload": {"id": "abc"}},
            {"type": "turn_context", "payload": {"model": "x"}},
            {"type": "response_item",
             "payload": {"type": "function_call", "name": "exec", "arguments": "{}",
                         "call_id": "c1"}},
            {"type": "response_item",
             "payload": {"type": "function_call_output", "call_id": "c1", "output": "stuff"}},
            {"type": "event_msg", "payload": {"type": "token_count", "total": 42}},
        ])
        recs = memsom_chats.parse_file("codex", self.path)
        self.assertEqual(recs, [])

    def test_drops_developer_role(self):
        write_jsonl(self.path, [
            {"type": "response_item",
             "payload": {"type": "message", "role": "developer",
                         "content": [{"type": "input_text", "text": "system preamble"}]}},
            _user_msg("real user text"),
        ])
        recs = memsom_chats.parse_file("codex", self.path)
        self.assertEqual([r["role"] for r in recs], ["user"])


if __name__ == "__main__":
    unittest.main()
