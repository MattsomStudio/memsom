#!/usr/bin/env python3
"""Phase 2 — `init`: creates the data dir + a fully-migrated DB, idempotently.

Run:  python -m unittest -v test_init
"""

import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import memsom
from memsom.interface import cli as memsom_cli


class TestInit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "dot-memsom"

    def tearDown(self):
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def _run_init(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            memsom_cli.cmd_init(Namespace(data_dir=str(self.data_dir)))
        return out.getvalue()

    def test_init_creates_migrated_db(self):
        stdout = self._run_init()
        db = self.data_dir / "memdag.db"
        # (a) dir + DB exist
        self.assertTrue(self.data_dir.is_dir())
        self.assertTrue(db.is_file())
        # (b) a MODULE table exists -> migrate_all ran, not just core schema
        conn = memsom.get_connection(str(db))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, "module table 'embeddings' missing — migrate_all did not run")
        # (c) stdout carries the resolved DB path for the bootstrap to capture
        self.assertIn(str(db), stdout.strip())

    def test_init_idempotent(self):
        self._run_init()
        # seed a node so we can prove a second init doesn't clobber data
        db = self.data_dir / "memdag.db"
        conn = memsom.get_connection(str(db))
        with conn:
            memsom.insert_node(conn, "a kept node", "user", memsom.RANK["user"])
        before = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        conn.close()
        # second init must not error and must not drop the node
        self._run_init()
        conn = memsom.get_connection(str(db))
        after = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        conn.close()
        self.assertEqual(before, after, "init clobbered existing nodes")


if __name__ == "__main__":
    unittest.main()
