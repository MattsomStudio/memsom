#!/usr/bin/env python3
"""Phase 3 — chat ingestion (DB side): stamps channel=user, dedups, dry-run.

Run:  python -m unittest -v test_chats
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import memdag
import memdag_chats


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


class TestIngestChats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "memdag.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        self.transcript = Path(self.tmp.name) / "session.jsonl"
        write_jsonl(self.transcript, [
            {"type": "user", "message": {"role": "user", "content": "first message"}},
            {"type": "assistant", "message": {"role": "assistant",
                                              "content": [{"type": "text", "text": "second message"}]}},
        ])

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def test_ingest_stamps_user_channel(self):
        summary = memdag_chats.ingest_chats(self.conn, "claude-code", files=[self.transcript])
        self.assertEqual(summary["messages"], 2)
        self.assertEqual(summary["new_nodes"], 2)
        channels = [r[0] for r in self.conn.execute("SELECT channel FROM nodes").fetchall()]
        self.assertTrue(channels and all(c == "user" for c in channels))

    def test_ingest_dedups_on_rerun(self):
        memdag_chats.ingest_chats(self.conn, "claude-code", files=[self.transcript])
        before = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        again = memdag_chats.ingest_chats(self.conn, "claude-code", files=[self.transcript])
        after = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(before, after, "re-ingest created duplicate nodes")
        self.assertEqual(again["new_nodes"], 0)

    def test_dry_run_inserts_nothing(self):
        summary = memdag_chats.ingest_chats(self.conn, "claude-code",
                                            files=[self.transcript], dry_run=True)
        self.assertEqual(summary["messages"], 2)
        self.assertEqual(summary["new_nodes"], 0)
        count = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
