#!/usr/bin/env python3
"""Tests for memdag_schema — idempotent migration helpers.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:\\Users\\you\\memdag -p test_memdag_schema.py \
    -t C:\\Users\\you\\memdag -v
"""

import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)  # 3.12 sqlite3 adapter regression = hard fail

import memdag
import memdag_schema


class Base(unittest.TestCase):
    """Temp-DB base — mirrors test_memdag.Base exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"  # missing parent: exercises mkdir
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()

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
        result1 = memdag_schema.add_column(self.conn, "nodes", "zz_test", "INTEGER NOT NULL DEFAULT 0")
        self.assertTrue(result1)

        result2 = memdag_schema.add_column(self.conn, "nodes", "zz_test", "INTEGER NOT NULL DEFAULT 0")
        self.assertFalse(result2)

        # Exactly one column named 'zz_test' in PRAGMA output
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(nodes)")]
        self.assertEqual(cols.count("zz_test"), 1)


class TestColumnExistsTruthTable(Base):
    def test_column_exists_truth_table(self):
        """True for real column, False for missing column, False for missing table."""
        self.assertTrue(memdag_schema.column_exists(self.conn, "nodes", "content"))
        self.assertFalse(memdag_schema.column_exists(self.conn, "nodes", "nope"))
        self.assertFalse(memdag_schema.column_exists(self.conn, "no_such_table", "x"))


class TestTableExists(Base):
    def test_table_exists(self):
        """True for 'nodes' and 'edges'; False for a non-existent name."""
        self.assertTrue(memdag_schema.table_exists(self.conn, "nodes"))
        self.assertTrue(memdag_schema.table_exists(self.conn, "edges"))
        self.assertFalse(memdag_schema.table_exists(self.conn, "ghosts"))


class TestEnsureTable(Base):
    def test_ensure_table_idempotent(self):
        """ensure_table twice with same DDL; a row inserted between calls must survive."""
        ddl = "CREATE TABLE IF NOT EXISTS t9(a INTEGER PRIMARY KEY);"
        memdag_schema.ensure_table(self.conn, ddl)
        with self.conn:
            self.conn.execute("INSERT INTO t9(a) VALUES (42)")
        memdag_schema.ensure_table(self.conn, ddl)  # second call — must not drop/recreate
        row = self.conn.execute("SELECT a FROM t9 WHERE a = 42").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 42)

    def test_ensure_table_rejects_missing_guard(self):
        """ValueError when 'IF NOT EXISTS' is absent."""
        with self.assertRaises(ValueError):
            memdag_schema.ensure_table(self.conn, "CREATE TABLE t9(a INTEGER)")


class TestBadIdentifierRejected(Base):
    def test_bad_identifier_rejected(self):
        """ValueError from column_exists/add_column on names containing SQL injection chars."""
        bad_table = "nodes; DROP TABLE nodes"
        bad_col = "a b"

        with self.assertRaises(ValueError):
            memdag_schema.column_exists(self.conn, bad_table, "content")

        with self.assertRaises(ValueError):
            memdag_schema.column_exists(self.conn, "nodes", bad_col)

        with self.assertRaises(ValueError):
            memdag_schema.add_column(self.conn, bad_table, "content", "TEXT")

        with self.assertRaises(ValueError):
            memdag_schema.add_column(self.conn, "nodes", bad_col, "TEXT")


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
        with patch.object(memdag_schema, "column_exists", return_value=False):
            result = memdag_schema.add_column(self.conn, "nodes", "content", "TEXT")
        self.assertFalse(result)  # no raise, returns False


class TestReexports(Base):
    def test_reexports(self):
        """memdag_schema.RANK and .NAME are the same objects as memdag.RANK / memdag.NAME."""
        self.assertIs(memdag_schema.RANK, memdag.RANK)
        self.assertIs(memdag_schema.NAME, memdag.NAME)


if __name__ == "__main__":
    unittest.main()
