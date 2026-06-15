#!/usr/bin/env python3
"""Tests for memdag_quarantine.

Run:
  python -W error::DeprecationWarning -m unittest discover \
      -s <repo> -p test_memdag_quarantine.py \
      -t <repo> -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_quarantine

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"  # missing parent: exercises mkdir
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_quarantine.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])

    def derive(self, content, parent_ids):
        nid, _label = memdag.derive_node(self.conn, content, parent_ids)
        return nid


class TestConsolidateQuarantinesExternalDerived(Base):
    """consolidate() quarantines a node whose label is EXTERNAL (0)."""

    def test_consolidate_quarantines_external_derived(self):
        e = self.add("Endorsed knowledge about nebula mesh configuration.", "endorsed")
        x = self.add("External source about nebula mesh docs.", "external")
        # derive from both: min(3, 0) = 0 => EXTERNAL label
        d = self.derive("Derived answer from endorsed and external.", [e, x])

        results = memdag_quarantine.consolidate(self.conn)

        # d must appear in results
        ids_quarantined = [r[0] for r in results]
        self.assertIn(d, ids_quarantined)
        cause = next(r[1] for r in results if r[0] == d)
        self.assertIn("EXTERNAL", cause)

        # status column flipped
        row = self.conn.execute("SELECT status FROM nodes WHERE id=?", (d,)).fetchone()
        self.assertEqual(row[0], "quarantined")

        # Demo-shape guard: memdag.live_sources still returns BOTH sources (frozen code
        # knows nothing about the status column)
        sources = memdag.live_sources(self.conn)
        source_ids = [s[0] for s in sources]
        self.assertIn(e, source_ids)
        self.assertIn(x, source_ids)


class TestConsolidateCatchesElevatedButTainted(Base):
    """consolidate() quarantines a node that was synthetically elevated to label=2
    but still has a live external ancestor."""

    def test_consolidate_catches_elevated_but_tainted(self):
        x = self.add("External source content for taint test.", "external")
        d = self.derive("Derived from external source only.", [x])

        # Simulate label elevation — the taint is still there via live external ancestor
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=2 WHERE id=?", (d,))

        results = memdag_quarantine.consolidate(self.conn)

        ids_quarantined = [r[0] for r in results]
        self.assertIn(d, ids_quarantined)
        cause = next(r[1] for r in results if r[0] == d)
        self.assertIn("live external ancestor", cause)


class TestQuarantinedExcludedFromPool(Base):
    """A quarantined SOURCE disappears from live_source_ids/live_unquarantined_sources
    but stays in memdag.live_sources."""

    def test_quarantined_excluded_from_pool(self):
        endorsed = self.add("Endorsed source for exclusion test.", "endorsed")
        external = self.add("External source for exclusion test.", "external")

        # Quarantine the external source node directly
        memdag_quarantine.quarantine_node(self.conn, external, "manual test quarantine")

        # live_source_ids excludes quarantined
        ids = memdag_quarantine.live_source_ids(self.conn)
        self.assertIn(endorsed, ids)
        self.assertNotIn(external, ids)

        # live_unquarantined_sources also excludes it
        unq = memdag_quarantine.live_unquarantined_sources(self.conn)
        unq_ids = [r[0] for r in unq]
        self.assertIn(endorsed, unq_ids)
        self.assertNotIn(external, unq_ids)

        # memdag.live_sources (frozen) still sees both
        all_sources = memdag.live_sources(self.conn)
        all_ids = [s[0] for s in all_sources]
        self.assertIn(endorsed, all_ids)
        self.assertIn(external, all_ids)


class TestPromoteRefusesThenSucceeds(Base):
    """promote() refuses when gates fail, then succeeds when they pass."""

    def test_promote_refuses_then_succeeds(self):
        e = self.add("Endorsed source for promote test.", "endorsed")
        x = self.add("External source for promote test.", "external")

        # d derived from [endorsed, external] -> consolidate quarantines d
        d = self.derive("Derived from endorsed and external.", [e, x])
        results = memdag_quarantine.consolidate(self.conn)
        self.assertTrue(any(r[0] == d for r in results), "d should have been quarantined")

        # promote(d) should FAIL: live external ancestor x is still present
        with self.assertRaises(ValueError) as cm:
            memdag_quarantine.promote(self.conn, d, "matt")
        self.assertIn("external", cm.exception.args[0].lower())

        # Now build d2 from [endorsed] only and manually quarantine it
        d2 = self.derive("Derived from endorsed only.", [e])
        flipped = memdag_quarantine.quarantine_node(self.conn, d2, "test manual quarantine")
        self.assertTrue(flipped)

        # promote(d2) should SUCCEED: has endorsed ancestor, no external ancestor
        memdag_quarantine.promote(self.conn, d2, "matt")

        row = self.conn.execute("SELECT status FROM nodes WHERE id=?", (d2,)).fetchone()
        self.assertEqual(row[0], "live")

        # A node with ONLY user ancestors: promote should raise (no endorsed ancestor)
        u = self.add("User source for promote test.", "user")
        d3 = self.derive("Derived from user only.", [u])
        memdag_quarantine.quarantine_node(self.conn, d3, "test no-endorsed quarantine")
        with self.assertRaises(ValueError) as cm2:
            memdag_quarantine.promote(self.conn, d3, "matt")
        self.assertIn("endorsed", cm2.exception.args[0].lower())


class TestConsolidateIdempotent(Base):
    """Second call to consolidate() returns []."""

    def test_consolidate_idempotent(self):
        x = self.add("External source for idempotent test.", "external")
        d = self.derive("Derived from external.", [x])

        first = memdag_quarantine.consolidate(self.conn)
        self.assertTrue(any(r[0] == d for r in first))

        second = memdag_quarantine.consolidate(self.conn)
        self.assertEqual(second, [])


class TestQuarantineNodeValidation(Base):
    """quarantine_node() raises ValueError on unknown id; ValueError on tombstoned;
    returns False on already-quarantined."""

    def test_quarantine_node_validation(self):
        # ValueError on unknown id
        with self.assertRaises(ValueError):
            memdag_quarantine.quarantine_node(self.conn, 9999, "bad id")

        # ValueError on tombstoned node
        u = self.add("User source to be tombstoned.", "user")
        memdag.revoke_cascade(self.conn, u, "gone")
        with self.assertRaises(ValueError):
            memdag_quarantine.quarantine_node(self.conn, u, "already dead")

        # False on already-quarantined
        good = self.add("Live node for double-quarantine test.", "user")
        first = memdag_quarantine.quarantine_node(self.conn, good, "first time")
        self.assertTrue(first)
        second = memdag_quarantine.quarantine_node(self.conn, good, "second time")
        self.assertFalse(second)


class TestReleaseAuditBreadcrumb(Base):
    """After promote(), quarantine_reason starts with 'promoted by matt'."""

    def test_release_audit_breadcrumb(self):
        e = self.add("Endorsed source for breadcrumb test.", "endorsed")
        d = self.derive("Derived from endorsed only.", [e])

        # Manually quarantine so we have a clean node with only endorsed ancestry
        memdag_quarantine.quarantine_node(self.conn, d, "pre-promote test")

        memdag_quarantine.promote(self.conn, d, "matt")

        row = self.conn.execute(
            "SELECT quarantine_reason FROM nodes WHERE id=?", (d,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertTrue(
            row[0].startswith("promoted by matt"),
            f"expected breadcrumb starting with 'promoted by matt', got: {row[0]!r}",
        )


class TestCliSubcommands(Base):
    """Smoke-test CLI subcommands via main()."""

    import io
    import contextlib

    def run_main(self, *argv):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memdag_quarantine.main(list(argv))
        return buf.getvalue()

    def run_main_stderr(self, *argv):
        import contextlib
        import io
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                memdag_quarantine.main(list(argv))
            except SystemExit:
                pass
        return buf_out.getvalue(), buf_err.getvalue()

    def test_cli_consolidate_nothing(self):
        # No agent-derived nodes: nothing to quarantine
        out = self.run_main("consolidate")
        self.assertIn("nothing to quarantine", out)

    def test_cli_consolidate_something(self):
        x = self.add("External source for CLI consolidate test.", "external")
        self.derive("Derived from external for CLI test.", [x])
        out = self.run_main("consolidate")
        self.assertIn("quarantined", out)

    def test_cli_quarantine_and_list(self):
        u = self.add("User source for CLI quarantine test.", "user")
        out = self.run_main("quarantine", str(u), "--reason", "cli test")
        self.assertIn(str(u), out)
        self.assertIn("quarantined", out)

        list_out = self.run_main("quarantine-list")
        self.assertIn(str(u), list_out)

    def test_cli_promote_gate_failure(self):
        x = self.add("External source for CLI promote test.", "external")
        d = self.derive("Derived from external for promote CLI test.", [x])
        memdag_quarantine.consolidate(self.conn)

        _out, err = self.run_main_stderr("promote", str(d), "--by", "matt")
        self.assertIn("endorsement required", err)

    def test_cli_quarantine_unknown_id(self):
        _out, err = self.run_main_stderr("quarantine", "9999", "--reason", "bad")
        self.assertIn("unknown", err.lower())


if __name__ == "__main__":
    unittest.main()
