#!/usr/bin/env python3
"""Tests for memdag_capgate — Gate #3 capability decision + audit.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_capgate.py -t <repo> -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_capgate as C
import memdag_policy as P


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()


class TestDecisionBoundary(Base):
    def test_allow_when_floor_meets_required(self):
        r = C.check_capability(self.conn, "sid", 2, "gmail.send", 2)
        self.assertEqual(r["decision"], "allow")
        self.assertEqual(r["floor_name"], "USER")
        self.assertEqual(r["required_name"], "USER")

    def test_allow_when_floor_above_required(self):
        r = C.check_capability(self.conn, "sid", 3, "x.y", 2)
        self.assertEqual(r["decision"], "allow")

    def test_deny_when_floor_below_required(self):
        # the headline case: external session, user-floor action
        r = C.check_capability(self.conn, "sid", 0, "gmail.send", 2)
        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["floor_name"], "EXTERNAL")
        self.assertIn("<", r["reason"])

    def test_external_required_always_allowed(self):
        # a read tool (required external=0) passes even at the lowest floor
        r = C.check_capability(self.conn, "sid", 0, "fetch.fetch", 0)
        self.assertEqual(r["decision"], "allow")

    def test_deny_sentinel_never_satisfied(self):
        # policy DENY (4) denies even a top-floor session
        r = C.check_capability(self.conn, "sid", 3, "unknown.tool", P.DENY)
        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["required_name"], "DENY")

    def test_each_rank_boundary(self):
        for floor in range(4):
            for required in range(5):
                r = C.check_capability(self.conn, "sid", floor, "t", required)
                self.assertEqual(r["decision"], "allow" if floor >= required else "deny")


class TestAudit(Base):
    def test_one_log_row_per_call(self):
        C.check_capability(self.conn, "sid", 2, "a", 2)
        C.check_capability(self.conn, "sid", 0, "b", 2)
        rows = C.recent_capability_log(self.conn)
        self.assertEqual(len(rows), 2)
        # most recent first
        self.assertEqual(rows[0]["tool"], "b")
        self.assertEqual(rows[0]["decision"], "deny")
        self.assertEqual(rows[1]["decision"], "allow")


class TestValidation(Base):
    def test_bad_floor_raises(self):
        for bad in (4, -1, True, "2"):
            with self.assertRaises(ValueError):
                C.check_capability(self.conn, "sid", bad, "t", 2)

    def test_bad_required_raises(self):
        for bad in (5, -1, True, "2"):
            with self.assertRaises(ValueError):
                C.check_capability(self.conn, "sid", 2, "t", bad)


if __name__ == "__main__":
    unittest.main()
