#!/usr/bin/env python3
"""Tests for memsom_schema — idempotent migration helpers.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_schema.py \
    -t <repo> -v
"""

import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)  # 3.12 sqlite3 adapter regression = hard fail

import memsom
from memsom.storage import schema as memsom_schema


class Base(unittest.TestCase):
    """Temp-DB base — mirrors test_memsom.Base exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"  # missing parent: exercises mkdir
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAddColumnTwice(Base):
    def test_add_column_twice_one_column(self):
        """add_column returns True on first call, False on second; exactly one column."""
        result1 = memsom_schema.add_column(self.conn, "nodes", "zz_test", "INTEGER NOT NULL DEFAULT 0")
        self.assertTrue(result1)

        result2 = memsom_schema.add_column(self.conn, "nodes", "zz_test", "INTEGER NOT NULL DEFAULT 0")
        self.assertFalse(result2)

        # Exactly one column named 'zz_test' in PRAGMA output
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(nodes)")]
        self.assertEqual(cols.count("zz_test"), 1)


class TestColumnExistsTruthTable(Base):
    def test_column_exists_truth_table(self):
        """True for real column, False for missing column, False for missing table."""
        self.assertTrue(memsom_schema.column_exists(self.conn, "nodes", "content"))
        self.assertFalse(memsom_schema.column_exists(self.conn, "nodes", "nope"))
        self.assertFalse(memsom_schema.column_exists(self.conn, "no_such_table", "x"))


class TestTableExists(Base):
    def test_table_exists(self):
        """True for 'nodes' and 'edges'; False for a non-existent name."""
        self.assertTrue(memsom_schema.table_exists(self.conn, "nodes"))
        self.assertTrue(memsom_schema.table_exists(self.conn, "edges"))
        self.assertFalse(memsom_schema.table_exists(self.conn, "ghosts"))


class TestEnsureTable(Base):
    def test_ensure_table_idempotent(self):
        """ensure_table twice with same DDL; a row inserted between calls must survive."""
        ddl = "CREATE TABLE IF NOT EXISTS t9(a INTEGER PRIMARY KEY);"
        memsom_schema.ensure_table(self.conn, ddl)
        with self.conn:
            self.conn.execute("INSERT INTO t9(a) VALUES (42)")
        memsom_schema.ensure_table(self.conn, ddl)  # second call — must not drop/recreate
        row = self.conn.execute("SELECT a FROM t9 WHERE a = 42").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 42)

    def test_ensure_table_rejects_missing_guard(self):
        """ValueError when 'IF NOT EXISTS' is absent."""
        with self.assertRaises(ValueError):
            memsom_schema.ensure_table(self.conn, "CREATE TABLE t9(a INTEGER)")


class TestBadIdentifierRejected(Base):
    def test_bad_identifier_rejected(self):
        """ValueError from column_exists/add_column on names containing SQL injection chars."""
        bad_table = "nodes; DROP TABLE nodes"
        bad_col = "a b"

        with self.assertRaises(ValueError):
            memsom_schema.column_exists(self.conn, bad_table, "content")

        with self.assertRaises(ValueError):
            memsom_schema.column_exists(self.conn, "nodes", bad_col)

        with self.assertRaises(ValueError):
            memsom_schema.add_column(self.conn, bad_table, "content", "TEXT")

        with self.assertRaises(ValueError):
            memsom_schema.add_column(self.conn, "nodes", bad_col, "TEXT")


class TestDuplicateColumnRaceIsNoop(Base):
    def test_duplicate_column_race_is_noop(self):
        """Monkeypatched column_exists=False simulates a concurrent-caller race.

        add_column must catch the 'duplicate column' OperationalError and return
        False instead of raising.
        """
        # 'content' already exists in nodes.  Patch column_exists so the guard
        # returns False (simulating the race window), then call add_column for
        # that same column.  SQLite will raise "duplicate column name: content"
        # which add_column must swallow.
        with patch.object(memsom_schema, "column_exists", return_value=False):
            result = memsom_schema.add_column(self.conn, "nodes", "content", "TEXT")
        self.assertFalse(result)  # no raise, returns False


class TestReexports(Base):
    def test_reexports(self):
        """memsom.storage.schema.RANK and .NAME are the same objects as memsom.RANK / memsom.NAME."""
        self.assertIs(memsom_schema.RANK, memsom.RANK)
        self.assertIs(memsom_schema.NAME, memsom.NAME)


class TestVersionedMigrations(Base):
    def test_current_version_constant(self):
        """CURRENT_VERSION is a positive int constant."""
        self.assertIsInstance(memsom_schema.CURRENT_VERSION, int)
        self.assertGreaterEqual(memsom_schema.CURRENT_VERSION, 1)

    def test_run_on_fresh_db_with_no_status_is_noop(self):
        """A fresh DB has no status column yet; the v1 step is a no-op there.

        The base get_connection() only creates nodes/edges (no status column),
        so the rebuild must not fire and user_version stays at 0 because the
        baseline reconciler only stamps when the CHECK is actually present.
        """
        self.assertFalse(memsom_schema.column_exists(self.conn, "nodes", "status"))
        memsom_schema.run_versioned_migrations(self.conn)
        # No status column => CHECK not present => baseline stays 0, step no-ops.
        self.assertEqual(
            self.conn.execute("PRAGMA user_version").fetchone()[0], 0
        )

    def test_status_check_added_then_gated(self):
        """After the status column is added (no CHECK), the step adds the CHECK
        and bumps user_version; a second run is gated off (no-op)."""
        memsom_schema.add_column(
            self.conn, "nodes", "status", "TEXT NOT NULL DEFAULT 'live'"
        )
        self.assertFalse(memsom_schema._status_check_present(self.conn))

        memsom_schema.run_versioned_migrations(self.conn)
        self.assertTrue(memsom_schema._status_check_present(self.conn))
        self.assertEqual(
            self.conn.execute("PRAGMA user_version").fetchone()[0],
            memsom_schema.CURRENT_VERSION,
        )

        # invalid status now rejected
        with self.assertRaises(sqlite3.IntegrityError):
            with self.conn:
                self.conn.execute(
                    "INSERT INTO nodes(content,channel,label,source_ref,"
                    "created_at,status) VALUES('x','user',1,NULL,"
                    "'2026-01-01T00:00:00+00:00','bogus')"
                )

        # second run: gated, DDL unchanged
        ddl = memsom_schema._nodes_ddl(self.conn)
        memsom_schema.run_versioned_migrations(self.conn)
        self.assertEqual(memsom_schema._nodes_ddl(self.conn), ddl)


if __name__ == "__main__":
    unittest.main()
