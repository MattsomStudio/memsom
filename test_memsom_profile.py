#!/usr/bin/env python3
"""Tests for memsom_profile — leaf-origin PROFILE (display-only).

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_profile.py \
    -t <repo> -v
"""

import contextlib
import io
import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_profile
import memsom_schema


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


class TestChainHistogram(Base):
    """test_chain_histogram: endorsed+user+external chain -> correct histogram."""

    def test_chain_histogram(self):
        e = self.add("endorsed content", "endorsed")    # label 3
        u = self.add("user content", "user")            # label 2
        x = self.add("external content", "external")   # label 0
        a, _ = memsom.derive_node(self.conn, "a", [e, u, x])
        b, _ = memsom.derive_node(self.conn, "b", [a])

        p = memsom_profile.profile(self.conn, b)

        self.assertEqual(p["hist"], {3: 1, 2: 1, 0: 1})
        self.assertEqual(p["leaf_total"], 3)
        self.assertEqual(p["floor"], 0)
        self.assertEqual(p["external_leaf_ids"], [x])


class TestDiamondLeafCountedOnce(Base):
    """test_diamond_leaf_counted_once: UNION dedup prevents double-counting."""

    def test_diamond_leaf_counted_once(self):
        s = self.add("user source", "user")          # label 2
        l, _ = memsom.derive_node(self.conn, "left", [s])
        r, _ = memsom.derive_node(self.conn, "right", [s])
        j, _ = memsom.derive_node(self.conn, "join", [l, r])

        p = memsom_profile.profile(self.conn, j)

        self.assertEqual(p["leaf_total"], 1, "diamond leaf must be counted exactly once")
        self.assertEqual(p["hist"], {2: 1})


class TestFloorEqualsGetNodeLabel(Base):
    """test_floor_equals_get_node_label: profile floor matches stored node label."""

    def test_floor_equals_get_node_label(self):
        e = self.add("endorsed content", "endorsed")
        u = self.add("user content", "user")
        x = self.add("external content", "external")
        a, _ = memsom.derive_node(self.conn, "a", [e, u, x])
        b, _ = memsom.derive_node(self.conn, "b", [a])

        for nid in (e, u, x, a, b):
            p = memsom_profile.profile(self.conn, nid)
            stored_label = memsom.get_node(self.conn, nid)["label"]
            self.assertEqual(
                p["floor"], stored_label,
                f"node {nid}: profile floor {p['floor']} != stored label {stored_label}"
            )


class TestExcludesTombstonedAndRedactedLeaves(Base):
    """test_excludes_tombstoned_and_redacted_leaves: dead/redacted leaves excluded."""

    def test_excludes_tombstoned_and_redacted_leaves(self):
        s1 = self.add("source one", "external")   # label 0
        s2 = self.add("source two", "external")   # label 0
        s3 = self.add("source three", "external") # label 0
        d, _ = memsom.derive_node(self.conn, "derived", [s1, s2, s3])

        # Add the redacted column
        memsom_schema.add_column(self.conn, "nodes", "redacted", "INTEGER NOT NULL DEFAULT 0")

        # Tombstone s1 directly (raw SQL — history surgery ok in tests)
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1 WHERE id=?", (s1,)
            )

        # Redact s2
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET redacted=1 WHERE id=?", (s2,)
            )

        p = memsom_profile.profile(self.conn, d)

        # Only s3 should survive
        self.assertEqual(p["leaf_total"], 1)
        self.assertEqual(p["hist"], {0: 1})
        self.assertNotIn(s1, p["external_leaf_ids"])
        self.assertNotIn(s2, p["external_leaf_ids"])
        self.assertIn(s3, p["external_leaf_ids"])


class TestPaddingChangesHistogramNeverFloor(Base):
    """test_padding_changes_histogram_never_floor: Biba-fatigue point in one test."""

    def test_padding_changes_histogram_never_floor(self):
        x = self.add("external content", "external")    # label 0
        e1 = self.add("endorsed one", "endorsed")       # label 3

        d1, _ = memsom.derive_node(self.conn, "d1", [x, e1])

        e2 = self.add("endorsed two", "endorsed")
        e3 = self.add("endorsed three", "endorsed")
        e4 = self.add("endorsed four", "endorsed")
        e5 = self.add("endorsed five", "endorsed")

        d2, _ = memsom.derive_node(self.conn, "d2", [x, e1, e2, e3, e4, e5])

        p1 = memsom_profile.profile(self.conn, d1)
        p2 = memsom_profile.profile(self.conn, d2)

        # Floor must be identical despite padding
        self.assertEqual(p1["floor"], 0)
        self.assertEqual(p2["floor"], 0)
        self.assertEqual(p1["floor"], p2["floor"])

        # Histogram must differ (more endorsed in d2)
        self.assertNotEqual(p1["hist"], p2["hist"])
        self.assertEqual(p1["hist"], {3: 1, 0: 1})
        self.assertEqual(p2["hist"], {3: 5, 0: 1})


class TestBareSourceIsItsOwnLeaf(Base):
    """test_bare_source_is_its_own_leaf: non-derived nodes are self-leaf."""

    def test_external_source_is_own_leaf(self):
        x = self.add("external content", "external")
        p = memsom_profile.profile(self.conn, x)

        self.assertEqual(p["leaf_total"], 1)
        self.assertEqual(p["hist"], {0: 1})
        self.assertEqual(p["external_leaf_ids"], [x])
        self.assertEqual(p["floor"], 0)

    def test_endorsed_source_no_external_leaf(self):
        e = self.add("endorsed content", "endorsed")
        p = memsom_profile.profile(self.conn, e)

        self.assertEqual(p["hist"], {3: 1})
        self.assertEqual(p["external_leaf_ids"], [])
        self.assertEqual(p["floor"], 3)


class TestUnknownIdRaises(Base):
    """test_unknown_id_raises: ValueError for nonexistent node."""

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            memsom_profile.profile(self.conn, 999)


class TestFormatProfileExact(Base):
    """test_format_profile_exact: exact summary string matching."""

    def test_format_with_external(self):
        e = self.add("endorsed content", "endorsed")
        u = self.add("user content", "user")
        x = self.add("external content", "external")
        a, _ = memsom.derive_node(self.conn, "a", [e, u, x])
        b, _ = memsom.derive_node(self.conn, "b", [a])

        p = memsom_profile.profile(self.conn, b)
        expected = (
            f"floor: EXTERNAL (gates)"
            f" | provenance: 2 of 3 leaves endorsed/user"
            f", 1 external [mem:{x}] - inspect"
        )
        self.assertEqual(p["summary"], expected)

    def test_format_without_external(self):
        e = self.add("endorsed content", "endorsed")
        u = self.add("user content", "user")
        a, _ = memsom.derive_node(self.conn, "a", [e, u])

        p = memsom_profile.profile(self.conn, a)
        # No external suffix
        self.assertNotIn("external", p["summary"])
        self.assertIn("floor: USER (gates)", p["summary"])
        self.assertIn("provenance:", p["summary"])


class TestProfileNeverWrites(Base):
    """test_profile_never_writes: profile() is display-only, never mutates DB."""

    def test_profile_never_writes(self):
        e = self.add("endorsed content", "endorsed")
        x = self.add("external content", "external")
        d, _ = memsom.derive_node(self.conn, "derived", [e, x])

        # Snapshot before
        before = self.conn.execute(
            "SELECT id, label, tombstoned FROM nodes ORDER BY id"
        ).fetchall()

        memsom_profile.profile(self.conn, d)

        # Snapshot after
        after = self.conn.execute(
            "SELECT id, label, tombstoned FROM nodes ORDER BY id"
        ).fetchall()

        self.assertEqual(before, after, "profile() must not mutate any node rows")


class TestCliProfilePrints(Base):
    """test_cli_profile_prints: CLI smoke test for profile subcommand."""

    def test_cli_prints_floor(self):
        e = self.add("endorsed content", "endorsed")
        u = self.add("user content", "user")
        x = self.add("external content", "external")
        a, _ = memsom.derive_node(self.conn, "a", [e, u, x])
        j, _ = memsom.derive_node(self.conn, "b", [a])

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memsom_profile.main(["profile", str(j)])

        output = buf.getvalue()
        self.assertIn("floor:", output)

    def test_cli_unknown_id_exits_1(self):
        with self.assertRaises(SystemExit) as cm:
            memsom_profile.main(["profile", "999"])
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
