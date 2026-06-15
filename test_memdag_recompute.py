#!/usr/bin/env python3
"""Tests for memdag_recompute — multi-hop integrity recompute.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_recompute.py \
    -t <repo> -v
"""

import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_recompute
import memdag_schema


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

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])


class TestChainRefloors(Base):
    """test_chain_refloors: after a source re-stamp, descendants refloor."""

    def test_chain_refloors(self):
        a = self.add("root external", "external")          # label 0
        b, _ = memdag.derive_node(self.conn, "mid", [a])   # label 0
        c, _ = memdag.derive_node(self.conn, "tip", [b])   # label 0

        # Simulate a source re-stamp: external -> USER (2)
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=2 WHERE id=?", (a,))

        changes = memdag_recompute.recompute_all(self.conn)
        changed_ids = {r[0] for r in changes}
        self.assertIn(b, changed_ids)
        self.assertIn(c, changed_ids)

        # Check (old, new) for b and c
        b_change = next(r for r in changes if r[0] == b)
        c_change = next(r for r in changes if r[0] == c)
        self.assertEqual(b_change, (b, 0, 2))
        self.assertEqual(c_change, (c, 0, 2))

        # Verify stored labels are now 2
        self.assertEqual(memdag.get_node(self.conn, b)["label"], 2)
        self.assertEqual(memdag.get_node(self.conn, c)["label"], 2)


class TestDiamondTrueMin(Base):
    """test_diamond_true_min: diamond finds the true min after a re-stamp."""

    def test_diamond_true_min(self):
        e = self.add("endorsed", "endorsed")   # label 3
        x = self.add("external", "external")   # label 0
        b, _ = memdag.derive_node(self.conn, "from endorsed", [e])   # label 3
        c, _ = memdag.derive_node(self.conn, "from external", [x])   # label 0
        d, _ = memdag.derive_node(self.conn, "join", [b, c])          # label 0

        # Re-stamp x from 0 to 2
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=2 WHERE id=?", (x,))

        changes = memdag_recompute.recompute_all(self.conn)

        changed_map = {r[0]: r for r in changes}

        # b is derived from endorsed(3) only; after x re-stamp b is still 3 (unchanged)
        self.assertNotIn(b, changed_map)

        # c was 0, now x is 2: c should become 2
        self.assertIn(c, changed_map)
        self.assertEqual(changed_map[c], (c, 0, 2))

        # d was 0, now min(b=3, c=2)=2
        self.assertIn(d, changed_map)
        self.assertEqual(changed_map[d], (d, 0, 2))

        # Check stored
        self.assertEqual(memdag.get_node(self.conn, c)["label"], 2)
        self.assertEqual(memdag.get_node(self.conn, d)["label"], 2)
        self.assertEqual(memdag.get_node(self.conn, b)["label"], 3)


class TestElevatedNodeIsFixedPoint(Base):
    """test_elevated_node_is_fixed_point_and_raises_child:
    a manually elevated node is a fixed point; its children still refloor off it.
    """

    def _create_elevations_table(self):
        memdag_schema.ensure_table(
            self.conn,
            "CREATE TABLE IF NOT EXISTS elevations ("
            "  id           INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  node         INTEGER NOT NULL,"
            "  from_label   INTEGER NOT NULL,"
            "  to_label     INTEGER NOT NULL,"
            "  reason       TEXT    NOT NULL,"
            "  elevated_by  TEXT    NOT NULL,"
            "  forced       INTEGER NOT NULL DEFAULT 0,"
            "  ts           TEXT    NOT NULL"
            ")"
        )

    def test_elevated_fixed_point_and_child_rises(self):
        a = self.add("external source", "external")   # label 0
        b, _ = memdag.derive_node(self.conn, "derived b", [a])  # label 0
        c, _ = memdag.derive_node(self.conn, "derived c", [b])  # label 0

        # Create elevations table and elevate b from 0 to 3
        self._create_elevations_table()
        with self.conn:
            self.conn.execute(
                "INSERT INTO elevations(node, from_label, to_label, reason, elevated_by, ts)"
                " VALUES (?, 0, 3, 'manual trust', 'test', '2026-06-11T00:00:00+00:00')",
                (b,)
            )
            self.conn.execute("UPDATE nodes SET label=3 WHERE id=?", (b,))

        changes = memdag_recompute.recompute_all(self.conn)
        changed_map = {r[0]: r for r in changes}

        # b is a fixed point — must NOT be in changes (not clobbered back to 0)
        self.assertNotIn(b, changed_map,
                         "elevated node b should be a fixed point and NOT recomputed")

        # c is derived from b (now fixed at 3): c should rise 0->3
        self.assertIn(c, changed_map)
        self.assertEqual(changed_map[c], (c, 0, 3))

        # Verify stored
        self.assertEqual(memdag.get_node(self.conn, b)["label"], 3)
        self.assertEqual(memdag.get_node(self.conn, c)["label"], 3)


class TestSourceLabelsNeverTouched(Base):
    """test_source_labels_never_touched: non-agent-derived nodes always keep their channel label."""

    def test_source_labels_never_touched(self):
        e = self.add("endorsed content", "endorsed")  # label 3
        u = self.add("user content", "user")           # label 2
        x = self.add("external content", "external")  # label 0

        # recompute_node on a source must be a no-op
        old, new = memdag_recompute.recompute_node(self.conn, e)
        self.assertEqual((old, new), (3, 3))

        old, new = memdag_recompute.recompute_node(self.conn, u)
        self.assertEqual((old, new), (2, 2))

        old, new = memdag_recompute.recompute_node(self.conn, x)
        self.assertEqual((old, new), (0, 0))

        # recompute_all must not touch any source node
        memdag_recompute.recompute_all(self.conn)

        for nid, channel in [(e, "endorsed"), (u, "user"), (x, "external")]:
            node = memdag.get_node(self.conn, nid)
            self.assertEqual(node["label"], memdag.RANK[channel],
                             f"source node {nid} ({channel}) label was mutated")


class TestIdempotent(Base):
    """test_idempotent: a second recompute_all is a strict no-op."""

    def test_idempotent(self):
        a = self.add("root", "external")   # label 0
        b, _ = memdag.derive_node(self.conn, "mid", [a])
        c, _ = memdag.derive_node(self.conn, "tip", [b])

        # Re-stamp a to 2
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=2 WHERE id=?", (a,))

        first = memdag_recompute.recompute_all(self.conn)
        self.assertTrue(first, "first call should have changes")

        second = memdag_recompute.recompute_all(self.conn)
        self.assertEqual(second, [], "second call should return no changes (idempotent)")


class TestTombstonedParentsExcluded(Base):
    """test_tombstoned_parents_excluded_and_zero_live_parent_fallback"""

    def test_tombstoned_excluded_and_zero_live_fallback(self):
        a = self.add("user src", "user")       # label 2
        x = self.add("external src", "external")  # label 0
        d, _ = memdag.derive_node(self.conn, "derived d", [a, x])  # label 0

        # Manually tombstone x (raw UPDATE to keep d live)
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1, tombstoned_at='2026-06-11T00:00:00+00:00'"
                " WHERE id=?", (x,)
            )

        # With x dead, d's only live parent is a(2): should rise to 2
        old, new = memdag_recompute.recompute_node(self.conn, d)
        self.assertEqual((old, new), (0, 2))
        self.assertEqual(memdag.get_node(self.conn, d)["label"], 2)

        # Now tombstone a as well — d has zero live parents
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1, tombstoned_at='2026-06-11T00:00:00+00:00'"
                " WHERE id=?", (a,)
            )

        # Zero live parents: stored label kept (no crash), returns (2, 2)
        old2, new2 = memdag_recompute.recompute_node(self.conn, d)
        self.assertEqual((old2, new2), (2, 2))


class TestUnknownIdRaises(Base):
    """test_unknown_id_raises: recompute_label raises ValueError for unknown id."""

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            memdag_recompute.recompute_label(self.conn, 999)


class TestDeepChainNoRecursionError(Base):
    """test_deep_chain_no_recursion_error: 1500-node chain completes without hitting recursion limit."""

    def test_deep_chain_no_recursion_error(self):
        # Build a 1500-node derived chain with raw INSERTs (like test_explain_deep_chain)
        with self.conn:
            self.conn.execute(
                "INSERT INTO nodes(content, channel, label, created_at)"
                " VALUES ('root', 'external', 0, '2026-06-11T00:00:00+00:00')"
            )
            for i in range(2, 1502):  # nodes 2..1501 inclusive = 1500 derived nodes
                self.conn.execute(
                    "INSERT INTO nodes(content, channel, label, created_at)"
                    " VALUES ('n', 'agent-derived', 0, '2026-06-11T00:00:00+00:00')"
                )
                self.conn.execute(
                    "INSERT INTO edges(child, parent) VALUES (?,?)", (i, i - 1)
                )

        # Re-stamp root (node 1) to label 2
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=2 WHERE id=1")

        # Should complete without RecursionError
        changes = memdag_recompute.recompute_all(self.conn)

        # Every derived node (2..1501) should refloor from 0 to 2
        self.assertEqual(len(changes), 1500,
                         f"expected 1500 changes, got {len(changes)}")

        # Spot-check tip
        tip = memdag.get_node(self.conn, 1501)
        self.assertEqual(tip["label"], 2)


if __name__ == "__main__":
    unittest.main()
