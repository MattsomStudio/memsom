#!/usr/bin/env python3
"""Regression tests for the 2026-06-11 architecture audit changes (ARCH-AUDIT.md).

C-1: memsom_recompute.effective_labels — the shared-memo bulk pass must agree
     with the per-node recompute_label() oracle AND with what recompute_all()
     writes, including elevation fixed points.
C-2: memsom_schema.taint_filter_clauses — the three full-pool read paths
     (memsom_cli._build_pool, memsom_retrieve._build_retrieve_pool,
     memsom_anticipatory.untainted_sources) must agree on the same poisoned
     fixture: tombstoned / quarantined / redacted / archived / above-clearance
     nodes are excluded by ALL of them; clean nodes are included by ALL.

Run:
  python -W error::DeprecationWarning -m unittest test_memsom_archaudit -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_anticipatory
import memsom_cli
import memsom_confid
import memsom_quarantine
import memsom_recompute
import memsom_redact
import memsom_retrieve
import memsom_trust


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
            return memsom.insert_node(self.conn, content, channel,
                                      memsom.RANK[channel])


class TestEffectiveLabelsBulkAgreesWithOracle(Base):
    """C-1: effective_labels == per-node recompute_label == recompute_all writes."""

    def _build_graph(self):
        """Diamond + chain + elevation fixed point, with a stale stored label."""
        ext = self.add("external root", "external")        # 0
        usr = self.add("user root", "user")                # 2
        end = self.add("endorsed root", "endorsed")        # 3
        d1, _ = memsom.derive_node(self.conn, "d1", [usr, end])   # min=2
        d2, _ = memsom.derive_node(self.conn, "d2", [d1, ext])    # min=0
        d3, _ = memsom.derive_node(self.conn, "d3", [d2])         # 0
        # Manual elevation: d2 becomes a fixed point at USER(2)
        memsom_trust.elevate(self.conn, d2, 2, "test", "tester")
        # Simulate drift: stamp a wrong stored label on d3 (true eff = 2 now)
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=3 WHERE id=?", (d3,))
        return ext, usr, end, d1, d2, d3

    def test_bulk_matches_per_node_oracle(self):
        _ext, _usr, _end, d1, d2, d3 = self._build_graph()
        bulk = memsom_recompute.effective_labels(self.conn)
        bulk_by_id = {nid: eff for nid, _stored, eff in bulk}
        # Elevated fixed point d2 is skipped by the bulk pass (never mismatches)
        self.assertNotIn(d2, bulk_by_id)
        # Every bulk result equals the per-node oracle
        for nid, _stored, eff in bulk:
            self.assertEqual(eff, memsom_recompute.recompute_label(self.conn, nid),
                             f"bulk eff for [{nid}] diverges from recompute_label")
        # The drifted node is detected with the right effective value
        self.assertEqual(bulk_by_id[d3], 2)
        self.assertEqual(bulk_by_id[d1], 2)

    def test_bulk_matches_recompute_all_writes(self):
        _ext, _usr, _end, _d1, d2, d3 = self._build_graph()
        expected_changes = [(nid, stored, eff)
                            for nid, stored, eff
                            in memsom_recompute.effective_labels(self.conn)
                            if eff != stored]
        changes = memsom_recompute.recompute_all(self.conn)
        self.assertEqual(changes, expected_changes)
        self.assertEqual(changes, [(d3, 3, 2)])
        # Fixed point untouched; second pass is a strict no-op
        self.assertEqual(memsom.get_node(self.conn, d2)["label"], 2)
        self.assertEqual(memsom_recompute.recompute_all(self.conn), [])


class TestThreePoolsAgreeOnPoisonedFixture(Base):
    """C-2: cli pool, retrieve pool, and anticipatory pool exclude/include
    exactly the same nodes on a fixture carrying every taint dimension."""

    def setUp(self):
        super().setUp()
        memsom_cli.migrate_all(self.conn)
        self.clean = self.add("clean user fact", "user")
        self.tomb = self.add("tombstoned fact", "user")
        self.quar = self.add("quarantined fact", "user")
        self.reda = self.add("redacted secret fact", "user")
        self.arch = self.add("archived fact", "user")
        self.secret = self.add("secret-cleared fact", "user")
        memsom.revoke_cascade(self.conn, self.tomb, "test")
        memsom_quarantine.quarantine_node(self.conn, self.quar, "test")
        memsom_redact.redact_node(self.conn, self.reda, "test", cascade=True)
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET archived=1, archived_at=? WHERE id=?",
                (memsom.now_iso(), self.arch))
        memsom_confid.classify(self.conn, self.secret, "secret")
        self.tainted = {self.tomb, self.quar, self.reda, self.arch, self.secret}

    def _pools_at_internal_clearance(self):
        cli_pool = {r[0] for r in memsom_cli._build_pool(self.conn, "internal")}
        retr_pool = memsom_retrieve._build_retrieve_pool(
            self.conn, memsom_confid.parse_conf("internal"),
            min_integrity=None, exclude_quarantined=True, exclude_redacted=True)
        antic_pool = {r[0] for r in
                      memsom_anticipatory.untainted_sources(self.conn, "internal")}
        return cli_pool, retr_pool, antic_pool

    def test_all_pools_identical_and_taint_free(self):
        for name, pool in zip(("cli", "retrieve", "anticipatory"),
                              self._pools_at_internal_clearance()):
            self.assertIn(self.clean, pool, f"{name} pool lost the clean node")
            self.assertFalse(pool & self.tainted,
                             f"{name} pool leaked tainted node(s): {pool & self.tainted}")
        cli_pool, retr_pool, antic_pool = self._pools_at_internal_clearance()
        self.assertEqual(cli_pool, retr_pool)
        self.assertEqual(cli_pool, antic_pool)

    def test_retrieve_widening_flag_only_widens_quarantine(self):
        """F-15 fail-safe: exclude_quarantined=False re-admits ONLY the
        quarantined node — never redacted/archived/tombstoned/above-clearance."""
        widened = memsom_retrieve._build_retrieve_pool(
            self.conn, memsom_confid.parse_conf("internal"),
            min_integrity=None, exclude_quarantined=False, exclude_redacted=False)
        self.assertEqual(widened, {self.clean, self.quar})


if __name__ == "__main__":
    unittest.main()
