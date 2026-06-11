#!/usr/bin/env python3
"""Tests for memdag_heal — self-healing: detect-and-report + deterministic rebuild.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:\\Users\\you\\memdag -p test_memdag_heal.py \
    -t C:\\Users\\you\\memdag -v
"""

import os
import sqlite3
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_heal
import memdag_recompute
import memdag_schema


class Base(unittest.TestCase):
    """Temp-DB base — mirrors test_memdag.Base exactly."""

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


# ---------------------------------------------------------------------------
# Test 1: integrity violation found and fixed
# ---------------------------------------------------------------------------

class TestIntegrityViolationFoundAndFixed(Base):
    """test_integrity_violation_found_and_fixed:
    raw UPDATE sets wrong label; check finds it; rebuild fixes it; source untouched.
    """

    def test_integrity_violation_found_and_fixed(self):
        u = self.add("user source", "user")          # label 2
        d, _ = memdag.derive_node(self.conn, "derived", [u])  # label 2

        # Corrupt: raw UPDATE to wrong label
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=3 WHERE id=?", (d,))
        self.assertEqual(memdag.get_node(self.conn, d)["label"], 3)

        # check must find the mismatch
        violations = memdag_heal.check(self.conn)
        kinds = [v["kind"] for v in violations]
        self.assertIn("integrity-mismatch", kinds)
        mismatch = next(v for v in violations if v["kind"] == "integrity-mismatch")
        self.assertEqual(mismatch["node"], d)
        self.assertEqual(mismatch["expected"], 2)
        self.assertEqual(mismatch["actual"], 3)

        # rebuild fixes it
        summary = memdag_heal.rebuild_derived(self.conn)
        self.assertGreaterEqual(summary["integrity_fixed"], 1)

        # check now clean for (b)
        after = memdag_heal.check(self.conn)
        self.assertFalse(any(v["kind"] == "integrity-mismatch" for v in after))

        # label is corrected
        self.assertEqual(memdag.get_node(self.conn, d)["label"], 2)

        # source node never touched
        self.assertEqual(memdag.get_node(self.conn, u)["label"], 2)
        self.assertEqual(memdag.get_node(self.conn, u)["channel"], "user")


# ---------------------------------------------------------------------------
# Test 2: redacted-with-content found and fixed
# ---------------------------------------------------------------------------

class TestRedactedWithContentFoundAndFixed(Base):
    """test_redacted_with_content_found_and_fixed:
    raw UPDATE sets redacted=1 leaving content; check finds it; rebuild wipes it.
    """

    def test_redacted_with_content_found_and_fixed(self):
        # Add the redacted column manually (memdag_redact module is absent in this install)
        memdag_schema.add_column(self.conn, "nodes", "redacted",
                                 "INTEGER NOT NULL DEFAULT 0")

        nid = self.add("sensitive content", "user")

        # Raw UPDATE: set redacted=1 but leave content intact
        with self.conn:
            self.conn.execute("UPDATE nodes SET redacted=1 WHERE id=?", (nid,))

        # check must find the violation
        violations = memdag_heal.check(self.conn)
        kinds = [v["kind"] for v in violations]
        self.assertIn("redacted-with-content", kinds)
        v = next(vv for vv in violations if vv["kind"] == "redacted-with-content")
        self.assertEqual(v["node"], nid)

        # rebuild must wipe the content
        summary = memdag_heal.rebuild_derived(self.conn)
        self.assertEqual(summary["content_wiped"], 1)

        # content is now empty
        self.assertEqual(memdag.get_node(self.conn, nid)["content"], "")

        # no more redacted-with-content violations
        after = memdag_heal.check(self.conn)
        self.assertFalse(any(v["kind"] == "redacted-with-content" for v in after))


# ---------------------------------------------------------------------------
# Test 3: live child of tombstoned found and fixed
# ---------------------------------------------------------------------------

class TestLiveChildOfTombstonedFoundAndFixed(Base):
    """test_live_child_of_tombstoned_found_and_fixed:
    raw tombstone a parent without cascading; check flags the live child;
    rebuild tombstones the child with 'cascade from node {a}'.
    """

    def test_live_child_of_tombstoned_found_and_fixed(self):
        a = self.add("source a", "user")
        d, _ = memdag.derive_node(self.conn, "derived d", [a])

        # Raw tombstone of a WITHOUT cascade (simulate broken tombstone)
        now = memdag.now_iso()
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1, tombstoned_at=?, revoke_reason='raw kill'"
                " WHERE id=?",
                (now, a)
            )

        # d should still be live
        self.assertEqual(memdag.get_node(self.conn, d)["tombstoned"], 0)

        # check must flag d as live child of tombstoned parent
        violations = memdag_heal.check(self.conn)
        kinds = [v["kind"] for v in violations]
        self.assertIn("live-child-of-tombstoned", kinds)
        v = next(vv for vv in violations if vv["kind"] == "live-child-of-tombstoned")
        self.assertEqual(v["node"], d)

        # rebuild must tombstone d
        summary = memdag_heal.rebuild_derived(self.conn)
        self.assertGreaterEqual(summary["cascades_repaired"], 1)

        # d is now tombstoned
        d_node = memdag.get_node(self.conn, d)
        self.assertEqual(d_node["tombstoned"], 1)

        # reason must include 'cascade from node {a}'
        self.assertIn(f"cascade from node {a}", d_node["revoke_reason"])

        # a's own death record is untouched (first-death-wins)
        a_node = memdag.get_node(self.conn, a)
        self.assertEqual(a_node["revoke_reason"], "raw kill")
        self.assertEqual(a_node["tombstoned_at"], now)


# ---------------------------------------------------------------------------
# Test 4: dangling edge reported but NOT deleted
# ---------------------------------------------------------------------------

class TestDanglingEdgeReportedNotDeleted(Base):
    """test_dangling_edge_reported_not_deleted:
    open a raw connection without FK pragma, insert orphan edge;
    check reports it; rebuild leaves edge count unchanged.
    """

    def test_dangling_edge_reported_not_deleted(self):
        edge_count_before = self.conn.execute(
            "SELECT COUNT(*) FROM edges").fetchone()[0]

        # Open a SECOND raw connection WITHOUT the FK pragma to insert a dangling edge
        raw = sqlite3.connect(str(self.db))
        try:
            raw.execute("INSERT INTO edges(child, parent) VALUES (998, 999)")
            raw.commit()
        finally:
            raw.close()

        # Verify the edge was actually inserted
        edge_count_after = self.conn.execute(
            "SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(edge_count_after, edge_count_before + 1)

        # check must report the dangling edge
        violations = memdag_heal.check(self.conn)
        dangling = [v for v in violations if v["kind"] == "dangling-edge"]
        self.assertGreaterEqual(len(dangling), 1)
        edge_tuples = [v["edge"] for v in dangling]
        self.assertIn((998, 999), edge_tuples)

        # rebuild_derived must NOT delete the edge — report only
        summary = memdag_heal.rebuild_derived(self.conn)
        self.assertGreaterEqual(summary["dangling_edges_reported"], 1)

        # edge count unchanged
        edge_count_final = self.conn.execute(
            "SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(edge_count_final, edge_count_after)


# ---------------------------------------------------------------------------
# Test 5: conf mismatch found and fixed (requires memdag_confid)
# ---------------------------------------------------------------------------

class TestConfMismatchFoundAndFixed(Base):
    """test_conf_mismatch_found_and_fixed:
    classify a parent 'secret'; derive child; raw-set child conf to 0 (stale);
    check flags conf-mismatch; rebuild sets child conf to the correct value.

    Skipped when memdag_confid is not installed.
    """

    def setUp(self):
        super().setUp()
        # Skip if memdag_confid is not available
        if memdag_heal.memdag_confid is None:
            self.skipTest("memdag_confid not installed")

    def test_conf_mismatch_found_and_fixed(self):
        mc = memdag_heal.memdag_confid
        mc.migrate(self.conn)

        a = self.add("parent source", "user")
        # Classify the parent
        mc.classify(self.conn, a, 2)  # conf_label=2

        d, _ = memdag.derive_node(self.conn, "child derived", [a])

        # Raw-set child conf to 0 (stale — simulates recompute_conf was never run)
        with self.conn:
            self.conn.execute("UPDATE nodes SET conf_label=0 WHERE id=?", (d,))

        # check must find the mismatch
        violations = memdag_heal.check(self.conn)
        conf_violations = [v for v in violations if v["kind"] == "conf-mismatch"]
        self.assertGreaterEqual(len(conf_violations), 1)
        cv = next(v for v in conf_violations if v["node"] == d)
        self.assertEqual(cv["actual"], 0)
        self.assertEqual(cv["expected"], 2)

        # rebuild sets child conf to 2
        summary = memdag_heal.rebuild_derived(self.conn)
        self.assertGreaterEqual(summary["conf_fixed"], 1)

        # no more conf-mismatch violations
        after = memdag_heal.check(self.conn)
        self.assertFalse(any(v["kind"] == "conf-mismatch" for v in after))


# ---------------------------------------------------------------------------
# Test 6: sources never modified
# ---------------------------------------------------------------------------

class TestSourcesNeverModified(Base):
    """test_sources_never_modified:
    snapshot all non-derived rows before rebuild over a corrupted DB;
    they must be byte-identical after.
    """

    def test_sources_never_modified(self):
        u = self.add("user source text", "user")
        e = self.add("endorsed source text", "endorsed")
        x = self.add("external source text", "external")
        d, _ = memdag.derive_node(self.conn, "derived", [u, e])

        # Corrupt the derived node
        with self.conn:
            self.conn.execute("UPDATE nodes SET label=3 WHERE id=?", (d,))

        # Snapshot all source (non-derived) rows
        def snapshot_sources():
            return self.conn.execute(
                "SELECT id, content, channel, label, source_ref, created_at,"
                " tombstoned, tombstoned_at, revoke_reason"
                " FROM nodes WHERE channel != 'agent-derived'"
                " ORDER BY id"
            ).fetchall()

        before = snapshot_sources()

        memdag_heal.rebuild_derived(self.conn)

        after = snapshot_sources()
        self.assertEqual(before, after, "source rows were mutated by rebuild_derived")


# ---------------------------------------------------------------------------
# Test 7: clean DB checks clean
# ---------------------------------------------------------------------------

class TestCleanDbChecksClean(Base):
    """test_clean_db_checks_clean:
    fresh seeded-by-hand graph -> check()==[] and a second rebuild is all-zero.
    """

    def test_clean_db_checks_clean(self):
        u = self.add("user fact", "user")
        e = self.add("endorsed doc", "endorsed")
        d, _ = memdag.derive_node(self.conn, "derived answer", [u, e])

        # Everything is consistent — check must return empty list
        violations = memdag_heal.check(self.conn)
        self.assertEqual(violations, [])

        # rebuild on a clean DB returns all-zero summary
        summary = memdag_heal.rebuild_derived(self.conn)
        self.assertEqual(summary["integrity_fixed"], 0)
        self.assertEqual(summary["conf_fixed"], 0)
        self.assertEqual(summary["cascades_repaired"], 0)
        self.assertEqual(summary["content_wiped"], 0)
        self.assertEqual(summary["dangling_edges_reported"], 0)

        # Second rebuild is still all-zero (idempotent)
        summary2 = memdag_heal.rebuild_derived(self.conn)
        self.assertEqual(summary2["integrity_fixed"], 0)
        self.assertEqual(summary2["conf_fixed"], 0)
        self.assertEqual(summary2["cascades_repaired"], 0)
        self.assertEqual(summary2["content_wiped"], 0)
        self.assertEqual(summary2["dangling_edges_reported"], 0)


# ---------------------------------------------------------------------------
# Test 8: defensive import — memdag_confid = None
# ---------------------------------------------------------------------------

class TestDefensiveImport(Base):
    """test_defensive_import:
    monkeypatch memdag_heal.memdag_confid = None; check and rebuild still run
    (conf checks skipped, no crash).
    """

    def test_defensive_import(self):
        original_confid = memdag_heal.memdag_confid
        try:
            # Simulate memdag_confid being absent
            memdag_heal.memdag_confid = None

            u = self.add("user source", "user")
            d, _ = memdag.derive_node(self.conn, "derived", [u])

            # Corrupt the derived node
            with self.conn:
                self.conn.execute("UPDATE nodes SET label=3 WHERE id=?", (d,))

            # check must still work (no conf-mismatch checks attempted)
            violations = memdag_heal.check(self.conn)
            self.assertFalse(any(v["kind"] == "conf-mismatch" for v in violations))
            # but integrity-mismatch is still reported
            self.assertTrue(any(v["kind"] == "integrity-mismatch" for v in violations))

            # rebuild must still work
            summary = memdag_heal.rebuild_derived(self.conn)
            self.assertEqual(summary["conf_fixed"], 0)  # conf skipped
            self.assertGreaterEqual(summary["integrity_fixed"], 1)  # integrity still fixed

            # no crash — we made it here
        finally:
            memdag_heal.memdag_confid = original_confid


if __name__ == "__main__":
    unittest.main()
