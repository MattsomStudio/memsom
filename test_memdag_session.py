#!/usr/bin/env python3
"""Tests for memdag_session — per-session integrity taint floor (Gate #3).

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_session.py -t <repo> -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_session


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


# ---------------------------------------------------------------------------
# begin_session
# ---------------------------------------------------------------------------

class TestBeginSession(Base):
    def test_default_start_floor_is_user(self):
        sid = memdag_session.begin_session(self.conn)
        self.assertEqual(memdag_session.current_floor(self.conn, sid), memdag.RANK["user"])

    def test_explicit_start_floor_by_name_and_int(self):
        s1 = memdag_session.begin_session(self.conn, "endorsed")
        s2 = memdag_session.begin_session(self.conn, 1)
        self.assertEqual(memdag_session.current_floor(self.conn, s1), 3)
        self.assertEqual(memdag_session.current_floor(self.conn, s2), 1)

    def test_sessions_have_distinct_ids(self):
        a = memdag_session.begin_session(self.conn)
        b = memdag_session.begin_session(self.conn)
        self.assertNotEqual(a, b)

    def test_clean_session_has_no_events(self):
        sid = memdag_session.begin_session(self.conn)
        self.assertEqual(memdag_session.session_log(self.conn, sid), [])


# ---------------------------------------------------------------------------
# current_floor
# ---------------------------------------------------------------------------

class TestCurrentFloor(Base):
    def test_unknown_session_raises(self):
        with self.assertRaises(ValueError):
            memdag_session.current_floor(self.conn, "nope")


# ---------------------------------------------------------------------------
# lower_floor — the watermark
# ---------------------------------------------------------------------------

class TestLowerFloor(Base):
    def test_drop_lowers_and_logs_event(self):
        sid = memdag_session.begin_session(self.conn)  # user=2
        new = memdag_session.lower_floor(self.conn, sid, "external", "fetch.fetch", "web_fetch")
        self.assertEqual(new, 0)
        self.assertEqual(memdag_session.current_floor(self.conn, sid), 0)
        ev = memdag_session.session_log(self.conn, sid)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["from_floor"], 2)
        self.assertEqual(ev[0]["to_floor"], 0)
        self.assertEqual(ev[0]["reason"], "web_fetch")
        self.assertEqual(ev[0]["tool"], "fetch.fetch")

    def test_floor_never_rises(self):
        # The core invariant: a higher incoming label is a safe no-op, never a raise.
        sid = memdag_session.begin_session(self.conn, "external")  # floor 0
        new = memdag_session.lower_floor(self.conn, sid, "endorsed", "x.tool", "attempt_raise")
        self.assertEqual(new, 0)
        self.assertEqual(memdag_session.current_floor(self.conn, sid), 0)

    def test_noop_writes_no_event(self):
        sid = memdag_session.begin_session(self.conn, "user")  # 2
        # incoming == current is a no-op
        memdag_session.lower_floor(self.conn, sid, "user", "x.tool", "same")
        self.assertEqual(memdag_session.session_log(self.conn, sid), [])

    def test_min_of_successive_drops(self):
        sid = memdag_session.begin_session(self.conn, "endorsed")  # 3
        self.assertEqual(memdag_session.lower_floor(self.conn, sid, 2, "a", "r"), 2)
        self.assertEqual(memdag_session.lower_floor(self.conn, sid, 0, "b", "r"), 0)
        # a later higher incoming cannot pull it back up
        self.assertEqual(memdag_session.lower_floor(self.conn, sid, 3, "c", "r"), 0)
        # two real drops -> two events (the no-op raise attempt logged nothing)
        self.assertEqual(len(memdag_session.session_log(self.conn, sid)), 2)

    def test_unknown_session_raises(self):
        with self.assertRaises(ValueError):
            memdag_session.lower_floor(self.conn, "nope", "external", "t", "r")

    def test_bad_incoming_label_raises(self):
        sid = memdag_session.begin_session(self.conn)
        for bad in (4, -1, "junk", True):
            with self.assertRaises(ValueError):
                memdag_session.lower_floor(self.conn, sid, bad, "t", "r")


# ---------------------------------------------------------------------------
# session_log ordering
# ---------------------------------------------------------------------------

class TestSessionLog(Base):
    def test_most_recent_first(self):
        sid = memdag_session.begin_session(self.conn, "endorsed")  # 3
        memdag_session.lower_floor(self.conn, sid, 2, "first", "r")
        memdag_session.lower_floor(self.conn, sid, 1, "second", "r")
        log = memdag_session.session_log(self.conn, sid)
        self.assertEqual(log[0]["tool"], "second")
        self.assertEqual(log[1]["tool"], "first")


if __name__ == "__main__":
    unittest.main()
