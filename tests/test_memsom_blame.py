#!/usr/bin/env python3
"""Tests for memsom_blame — derivation-DAG blame ("git blame") module.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_blame.py \
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
from memsom.interface import blame as memsom_blame
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_blame.migrate(self.conn)  # ensure redacted + status columns

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])

    def derive(self, content, parents):
        nid, _ = memsom.derive_node(self.conn, content, parents)
        return nid


class TestBlameListsExactRoots(Base):
    """blame(d) lists exactly its direct source roots, ordered label DESC."""

    def test_blame_lists_exact_roots(self):
        e = self.add("endorsed source content here", "endorsed")   # label 3
        u = self.add("user source content here", "user")           # label 2
        d = self.derive("derived answer", [e, u])

        result = memsom_blame.blame(self.conn, d)

        # Exactly 2 entries: e and u (no derived nodes)
        self.assertEqual(len(result), 2)
        ids = [r["id"] for r in result]
        self.assertEqual(ids, [e, u])  # label DESC: endorsed(3) before user(2)

        # Check channels and labels
        self.assertEqual(result[0]["channel"], "endorsed")
        self.assertEqual(result[0]["label"], 3)
        self.assertEqual(result[0]["label_name"], "ENDORSED")
        self.assertEqual(result[1]["channel"], "user")
        self.assertEqual(result[1]["label"], 2)
        self.assertEqual(result[1]["label_name"], "USER")

        # No derived nodes in list
        for r in result:
            self.assertNotEqual(r["channel"], "agent-derived")

        # States should be live
        for r in result:
            self.assertEqual(r["state"], "live")


class TestMultihopReachesRoots(Base):
    """blame through a multi-hop chain reaches only the root."""

    def test_multihop_reaches_roots(self):
        e = self.add("endorsed root source", "endorsed")
        d1 = self.derive("first derived", [e])
        d2 = self.derive("second derived", [d1])
        d3 = self.derive("third derived", [d2])

        result = memsom_blame.blame(self.conn, d3)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], e)
        self.assertEqual(result[0]["channel"], "endorsed")


class TestDiamondRootOnce(Base):
    """Diamond topology: a -> b, a -> c, (b, c) -> d; blame(d) lists a exactly once."""

    def test_diamond_root_once(self):
        a = self.add("shared root content here", "endorsed")
        b = self.derive("left branch", [a])
        c = self.derive("right branch", [a])
        d = self.derive("join node", [b, c])

        result = memsom_blame.blame(self.conn, d)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], a)


class TestRedactedAncestorShownWithState(Base):
    """Redacted root is included in blame with state 'redacted', line == '[REDACTED]'."""

    def test_redacted_ancestor_shown_with_state(self):
        e = self.add("original secret endorsed content here", "endorsed")
        d = self.derive("derived from secret", [e])

        # Redact the root
        memsom_redact.redact_node(self.conn, e, reason="test redaction")

        result = memsom_blame.blame(self.conn, d)
        self.assertEqual(len(result), 1)
        root = result[0]

        # State must contain 'redacted'
        self.assertIn("redacted", root["state"])

        # line must be '[REDACTED]', not the original text
        self.assertEqual(root["line"], "[REDACTED]")

        # Original text must not appear anywhere in the result lines
        for line in memsom_blame.format_blame(self.conn, d):
            self.assertNotIn("original secret", line)


class TestTombstonedAncestorShownNotSkipped(Base):
    """Tombstoned root is included in blame with state 'tombstoned'."""

    def test_tombstoned_ancestor_shown_not_skipped(self):
        e = self.add("user source that gets tombstoned", "user")
        # Use raw UPDATE to tombstone without cascading to derived children
        d = self.derive("derived from tombstoned root", [e])

        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1,"
                " tombstoned_at='2026-06-11T00:00:00+00:00'"
                " WHERE id=?",
                (e,)
            )

        result = memsom_blame.blame(self.conn, d)
        self.assertEqual(len(result), 1)
        root = result[0]
        self.assertEqual(root["id"], e)
        self.assertEqual(root["state"], "tombstoned")


class TestBlameOfSourceIsItself(Base):
    """blame(source_node) returns a single-entry list for itself."""

    def test_blame_of_source_is_itself(self):
        e = self.add("endorsed source content", "endorsed")

        result = memsom_blame.blame(self.conn, e)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], e)
        self.assertEqual(result[0]["channel"], "endorsed")
        self.assertEqual(result[0]["state"], "live")


class TestUnknownIdRaises(Base):
    """blame(unknown_id) raises ValueError."""

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            memsom_blame.blame(self.conn, 9999)


class TestCliOutput(Base):
    """CLI main(['blame', str(d)]) outputs expected lines."""

    def test_cli_output(self):
        e = self.add("endorsed source for cli test", "endorsed")
        u = self.add("user source for cli test", "user")
        d = self.derive("derived answer for cli", [e, u])

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memsom_blame.main(["blame", str(d)])
        out = buf.getvalue()

        # Header
        self.assertIn("came from:", out)

        # Both root ids appear
        self.assertIn(f"[{e}]", out)
        self.assertIn(f"[{u}]", out)

        # Channel and integrity name appear
        self.assertIn("endorsed", out)
        self.assertIn("user", out)
        self.assertIn("ENDORSED", out)
        self.assertIn("USER", out)


class TestBlameClearanceGate(Base):
    def test_above_clearance_content_suppressed(self):
        from memsom.integrity import confid as memsom_confid
        memsom_confid.migrate(self.conn)
        secret = self.add("TOPSECRET nuclear codes 0000", "endorsed")
        with self.conn:
            self.conn.execute("UPDATE nodes SET conf_label=3 WHERE id=?", (secret,))
        d = self.derive("derived answer", [secret])

        # No clearance (admin/history) -> content visible.
        entries = memsom_blame.blame(self.conn, d)
        self.assertIn("nuclear codes", entries[0]["line"])

        # PUBLIC clearance -> content of the SECRET root suppressed, metadata kept.
        gated = memsom_blame.blame(self.conn, d, clearance=0)
        self.assertEqual(gated[0]["line"], "[ABOVE CLEARANCE]")
        self.assertEqual(gated[0]["id"], secret, "metadata still visible for audit")
        self.assertEqual(gated[0]["channel"], "endorsed")


if __name__ == "__main__":
    unittest.main()
