#!/usr/bin/env python3
"""Tests for memsom_gate — action-time integrity floor enforcement.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_gate.py \
    -t <repo> -v
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
import warnings
from datetime import datetime
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.integrity import gate as memsom_gate
from memsom.storage import schema as memsom_schema


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])


# ---------------------------------------------------------------------------
# Test 1 — allow when floor meets required
# ---------------------------------------------------------------------------

class TestAllowWhenFloorMeetsRequired(Base):
    def test_allow_when_floor_meets_required(self):
        e = self.add("endorsed source content here", "endorsed")   # label 3
        u = self.add("user source content here", "user")           # label 2
        d, _ = memsom.derive_node(self.conn, "derived from endorsed and user", [e, u])
        # floor = min(3, 2) = 2 = USER

        r = memsom_gate.check_action(self.conn, d, "user")

        self.assertEqual(r["decision"], "allow")
        self.assertEqual(r["floor"], 2)
        self.assertEqual(r["required"], 2)
        self.assertIsNone(r["culprit"])


# ---------------------------------------------------------------------------
# Test 2 — deny names weakest live leaf
# ---------------------------------------------------------------------------

class TestDenyNamesWeakestLiveLeaf(Base):
    def test_deny_names_weakest_live_leaf(self):
        e = self.add("endorsed source content here", "endorsed")   # label 3
        x = self.add("external source content here", "external")   # label 0
        d, _ = memsom.derive_node(self.conn, "derived from endorsed and external", [e, x])
        # floor = min(3, 0) = 0

        r = memsom_gate.check_action(self.conn, d, "user")

        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["culprit"], x)
        self.assertIn("weakest live leaf", r["reason"])


# ---------------------------------------------------------------------------
# Test 3 — tiebreak: lowest id wins
# ---------------------------------------------------------------------------

class TestDenyCulpritTiebreakLowestId(Base):
    def test_deny_culprit_tiebreak_lowest_id(self):
        x1 = self.add("external one content here", "external")    # label 0, lower id
        x2 = self.add("external two content here", "external")    # label 0, higher id
        self.assertLess(x1, x2)
        d, _ = memsom.derive_node(
            self.conn, "derived from two externals", [x1, x2]
        )
        # floor = 0

        r = memsom_gate.check_action(self.conn, d, "user")

        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["culprit"], x1)


# ---------------------------------------------------------------------------
# Test 4 — check_action never mutates labels
# ---------------------------------------------------------------------------

class TestCheckActionNeverMutatesLabels(Base):
    def test_check_action_never_mutates_labels(self):
        e = self.add("endorsed source content here", "endorsed")
        u = self.add("user source content here", "user")
        x = self.add("external source content here", "external")
        d_allow, _ = memsom.derive_node(self.conn, "allow derived node text", [e, u])
        d_deny, _ = memsom.derive_node(self.conn, "deny derived node text", [e, x])

        # Snapshot before
        before = self.conn.execute(
            "SELECT id, label, tombstoned FROM nodes ORDER BY id"
        ).fetchall()
        before_nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        before_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        # Run one allow and one deny
        memsom_gate.check_action(self.conn, d_allow, "user")
        memsom_gate.check_action(self.conn, d_deny, "user")

        # Snapshot after
        after = self.conn.execute(
            "SELECT id, label, tombstoned FROM nodes ORDER BY id"
        ).fetchall()
        after_nodes = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        after_edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        self.assertEqual(before, after)
        self.assertEqual(before_nodes, after_nodes)
        self.assertEqual(before_edges, after_edges)


# ---------------------------------------------------------------------------
# Test 5 — gate_log row per call + row shape
# ---------------------------------------------------------------------------

class TestGateLogRowPerCall(Base):
    def test_gate_log_row_per_call(self):
        e = self.add("endorsed source content here", "endorsed")
        x = self.add("external source content here", "external")
        u = self.add("user source content here", "user")
        d, _ = memsom.derive_node(self.conn, "derived content text here", [e, x])

        memsom_gate.check_action(self.conn, d, "external")   # allow
        memsom_gate.check_action(self.conn, d, "user")       # deny
        memsom_gate.check_action(self.conn, u, "user")       # allow (source node)

        count = self.conn.execute("SELECT COUNT(*) FROM gate_log").fetchone()[0]
        self.assertEqual(count, 3)

        # Verify last row shape
        row = self.conn.execute(
            "SELECT node, required, floor, decision, culprit, ts"
            " FROM gate_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        node_val, required_val, floor_val, decision_val, culprit_val, ts_val = row

        self.assertEqual(node_val, u)
        self.assertEqual(required_val, "USER")
        self.assertEqual(floor_val, 2)
        self.assertEqual(decision_val, "allow")
        self.assertIsNone(culprit_val)

        # ts must be ISO-8601 parseable
        parsed = datetime.fromisoformat(ts_val)
        self.assertIsNotNone(parsed)


# ---------------------------------------------------------------------------
# Test 6 — unknown node raises; no log row written
# ---------------------------------------------------------------------------

class TestUnknownNodeRaisesAndNoLog(Base):
    def test_unknown_node_raises_and_no_log(self):
        memsom_gate.migrate(self.conn)
        before = self.conn.execute("SELECT COUNT(*) FROM gate_log").fetchone()[0]
        self.assertEqual(before, 0)

        with self.assertRaises(ValueError):
            memsom_gate.check_action(self.conn, 9999, "user")

        after = self.conn.execute("SELECT COUNT(*) FROM gate_log").fetchone()[0]
        self.assertEqual(after, 0)


# ---------------------------------------------------------------------------
# Test 7 — bad required name raises
# ---------------------------------------------------------------------------

class TestBadRequiredNameRaises(Base):
    def test_bad_required_name_raises(self):
        d, _ = memsom.derive_node(
            self.conn,
            "some derived content text",
            [self.add("user src content", "user")],
        )
        with self.assertRaises(ValueError):
            memsom_gate.check_action(self.conn, d, "banana")


# ---------------------------------------------------------------------------
# Test 8 — READ paths are free (no gate_log rows from reads)
# ---------------------------------------------------------------------------

class TestReadPathsAreFree(Base):
    def test_read_paths_are_free(self):
        e = self.add("endorsed source content here", "endorsed")
        x = self.add("external source content here", "external")
        d, _ = memsom.derive_node(self.conn, "derived from endorsed and external", [e, x])
        # d has floor=0 — a "low integrity" node

        # Perform all READ operations
        sources = memsom.live_sources(self.conn)
        memsom.compose("a question about external content", sources)
        memsom.get_node(self.conn, d)
        memsom.parents_of(self.conn, d)

        # Ensure gate_log table exists (so COUNT is valid)
        memsom_gate.migrate(self.conn)

        count = self.conn.execute("SELECT COUNT(*) FROM gate_log").fetchone()[0]
        self.assertEqual(count, 0, "READ operations must never write to gate_log")


# ---------------------------------------------------------------------------
# Test 9 — source node is its own culprit
# ---------------------------------------------------------------------------

class TestSourceNodeIsOwnCulprit(Base):
    def test_source_node_is_own_culprit(self):
        x = self.add("external source content here", "external")   # label 0

        r = memsom_gate.check_action(self.conn, x, "endorsed")

        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["culprit"], x)


# ---------------------------------------------------------------------------
# Test 10 — deny after culprit tombstoned falls to next live leaf
# ---------------------------------------------------------------------------

class TestDenyAfterCulpritTombstonedFallsToNext(Base):
    def test_deny_after_culprit_tombstoned_falls_to_next(self):
        e = self.add("endorsed source content here", "endorsed")   # label 3
        x = self.add("external source content here", "external")   # label 0
        d, _ = memsom.derive_node(self.conn, "derived content text", [e, x])
        # floor stored = 0

        # Tombstone x directly via raw SQL (d's stored floor stays 0)
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1,"
                " tombstoned_at='2026-06-11T00:00:00+00:00'"
                " WHERE id=?",
                (x,),
            )

        # d's stored label is still 0 but x is now dead
        r = memsom_gate.check_action(self.conn, d, "endorsed")

        self.assertEqual(r["decision"], "deny")
        # culprit must now be e (the weakest LIVE leaf)
        self.assertEqual(r["culprit"], e)


# ---------------------------------------------------------------------------
# Test 11 — CLI exit codes
# ---------------------------------------------------------------------------

class TestCliCheckActionExitCodes(Base):
    def test_cli_check_action_exit_codes(self):
        e = self.add("endorsed source content here", "endorsed")
        x = self.add("external source content here", "external")
        d, _ = memsom.derive_node(self.conn, "derived content text", [e, x])
        # floor = 0; require user(2) -> deny -> exit 2

        # Capture stdout; deny path
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(buf):
                memsom_gate.main(["check-action", str(d), "--require", "user"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("DENY", buf.getvalue())

        # allow path — no SystemExit or exit code 0
        u = self.add("user source content here", "user")
        d_allow, _ = memsom.derive_node(
            self.conn, "derived from endorsed and user", [e, u]
        )
        # floor = min(3,2) = 2; require user(2) -> allow
        buf2 = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf2):
                memsom_gate.main(["check-action", str(d_allow), "--require", "user"])
            exit_code = 0
        except SystemExit as exc:
            exit_code = exc.code
        self.assertIn(exit_code, (0, None))
        self.assertIn("ALLOW", buf2.getvalue())


if __name__ == "__main__":
    unittest.main()
