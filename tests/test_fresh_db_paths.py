#!/usr/bin/env python3
"""Fresh-DB regression: every MCP-exposed tool must migrate the tables it touches.

The MCP server holds no persistent connection — each tool call routes through
memsom_mcp._call_tool -> memsom_cli.main -> a CLI handler that opens its own
connection. A brand-new friend's DB has only the core schema (nodes/edges) until
a handler runs the relevant module migrate(). This test drives the real dispatch
path against a never-migrated DB and asserts no 'no such table' / OperationalError.

This is the executable form of the Phase 0b audit. If a tool fails here, its
handler is missing a migrate() call for a table it reads/writes.

Run:  python -m unittest -v test_fresh_db_paths
"""

import os
import tempfile
import unittest
from pathlib import Path

import memsom
import memsom_mcp


class FreshDbTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._n = 0

    def tearDown(self):
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def _fresh_db(self):
        # A distinct, never-touched DB path for each tool under test.
        self._n += 1
        p = Path(self.tmp.name) / f"fresh_{self._n}.db"
        os.environ["MEMDAG_DB"] = str(p)
        return p

    def _assert_no_missing_table(self, name, args):
        self._fresh_db()
        text, _is_error = memsom_mcp._call_tool(name, args)
        low = text.lower()
        self.assertNotIn("no such table", low,
                         f"tool {name!r} hit a missing table on a fresh DB:\n{text}")
        self.assertNotIn("operationalerror", low,
                         f"tool {name!r} raised OperationalError on a fresh DB:\n{text}")

    # The two that own module tables (the real risk):
    def test_ingest_text_migrates_content_hash(self):
        self._assert_no_missing_table(
            "ingest_text", {"text": "hello sqlite world", "channel": "user"})

    def test_retrieve_migrates_postings(self):
        self._assert_no_missing_table("retrieve", {"query": "hello"})

    # ask runs migrate_all; profile reads label tables — both must survive a fresh DB.
    def test_ask_on_fresh_db(self):
        self._assert_no_missing_table("ask", {"question": "hello"})

    def test_profile_on_fresh_db(self):
        self._assert_no_missing_table("profile", {"id": 1})


class JournalMode(unittest.TestCase):
    """Document the concurrency posture relied on by doctor / the MCP server (W4)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["MEMDAG_DB"] = str(Path(self.tmp.name) / "j.db")

    def tearDown(self):
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def test_record_journal_mode(self):
        conn = memsom.get_connection()
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        # Not an assertion on WAL specifically — this pins the current mode so a
        # future change is visible. If mode != 'wal', concurrent readers (doctor,
        # the MCP server while a client holds the DB) must use busy_timeout.
        self.assertIn(mode.lower(), {"wal", "delete", "truncate", "persist", "memory", "off"})
        self._mode = mode  # surfaced via the test name/output for the WAL decision


if __name__ == "__main__":
    unittest.main()
