#!/usr/bin/env python3
"""Tests for memdag_relate — GraphRAG-safe associative edges.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:\\Users\\you\\memdag -p test_memdag_relate.py \
    -t C:\\Users\\you\\memdag -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_confid
import memdag_quarantine
import memdag_redact
import memdag_relate


class Base(unittest.TestCase):
    """Temp-DB base — mirrors test_memdag.Base exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_relate.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        """Insert a node and return its id."""
        with self.conn:
            return memdag.insert_node(self.conn, content, channel,
                                      memdag.RANK[channel])

    def relate(self, a, b, kind="relates-to"):
        memdag_relate.relate(self.conn, a, b, kind)


# ---------------------------------------------------------------------------
# Test 1: star topology — neighborhood returns all direct neighbors
# ---------------------------------------------------------------------------

class TestNeighborhoodReturnRelated(Base):
    def test_neighborhood_returns_related(self):
        """Star u--(a,b,c): neighborhood(u, hops=1) returns exactly {a,b,c} with hops==1."""
        u = self.add("hub content long enough to pass snippet filter", "user")
        a = self.add("neighbor a content long enough", "user")
        b = self.add("neighbor b content long enough", "user")
        c = self.add("neighbor c content long enough", "user")
        self.relate(u, a)
        self.relate(u, b)
        self.relate(u, c)

        results = memdag_relate.neighborhood(self.conn, u, hops=1)
        ids = {d["id"] for d in results}
        self.assertEqual(ids, {a, b, c})
        for d in results:
            self.assertEqual(d["hops"], 1)


# ---------------------------------------------------------------------------
# Test 2: poisoned neighbor cannot ride along
# ---------------------------------------------------------------------------

class TestPoisonedNeighborCannotRide(Base):
    def test_poisoned_neighbor_cannot_ride(self):
        """chain good(endorsed=3) -- bad(external=0) -- far(endorsed=3)

        With min_integrity=1: bad excluded (path_min=0) AND far excluded
        (only path runs through bad).
        With min_integrity=0: both appear and far.path_min==0 (floor propagated).
        """
        good = self.add("Good endorsed content long enough to appear here.", "endorsed")
        bad = self.add("Bad external content long enough to appear here.", "external")
        far = self.add("Far endorsed content long enough to appear here.", "endorsed")
        self.relate(good, bad)
        self.relate(bad, far)

        # With min_integrity=1: bad excluded (path_min=0), far also excluded
        results = memdag_relate.neighborhood(self.conn, good, hops=3, min_integrity=1)
        ids = {d["id"] for d in results}
        self.assertNotIn(bad, ids, "bad should be excluded (path_min=0 < min_integrity=1)")
        self.assertNotIn(far, ids, "far should be excluded (only path through bad, path_min=0)")

        # With min_integrity=0: both appear; far.path_min == 0
        results0 = memdag_relate.neighborhood(self.conn, good, hops=3, min_integrity=0)
        ids0 = {d["id"] for d in results0}
        self.assertIn(bad, ids0, "bad should appear with min_integrity=0")
        self.assertIn(far, ids0, "far should appear with min_integrity=0")
        far_entry = next(d for d in results0 if d["id"] == far)
        self.assertEqual(far_entry["path_min"], 0,
                         "far.path_min should be 0 (floor flows through bad)")


# ---------------------------------------------------------------------------
# Test 3: path_min value propagation
# ---------------------------------------------------------------------------

class TestPathMinValue(Base):
    def test_path_min_value(self):
        """chain e(3) -- u(2) -- e2(3)

        neighborhood(e, hops=2)[e2].path_min == 2 (min along path).
        u.path_min == 2 (min(3, 2)).
        """
        e = self.add("Endorsed content long enough to pass snippet.", "endorsed")
        u = self.add("User content long enough to pass snippet filter.", "user")
        e2 = self.add("Endorsed2 content long enough to pass snippet.", "endorsed")
        self.relate(e, u)
        self.relate(u, e2)

        results = memdag_relate.neighborhood(self.conn, e, hops=2, min_integrity=0)
        by_id = {d["id"]: d for d in results}

        self.assertIn(u, by_id, "u should be in neighborhood")
        self.assertIn(e2, by_id, "e2 should be in neighborhood")

        self.assertEqual(by_id[u]["path_min"], 2,
                         "u.path_min = min(3, 2) = 2")
        self.assertEqual(by_id[e2]["path_min"], 2,
                         "e2.path_min = min(3, 2, 3) = 2 (floor set by u)")


# ---------------------------------------------------------------------------
# Test 4: best (widest) path wins in diamond topology
# ---------------------------------------------------------------------------

class TestBestPathWins(Base):
    def test_best_path_wins(self):
        """Diamond: start -- x(external=0) -- target
                    start -- y(user=2)     -- target

        target's path_min should be min(start_label, 2, target_label) via y,
        NOT 0 from the x path.  Relaxation picks the widest path.
        """
        start = self.add("Start endorsed content long enough here.", "endorsed")
        x = self.add("External low-integrity node content here.", "external")
        y = self.add("User medium-integrity node content here.", "user")
        target = self.add("Target endorsed content long enough here.", "endorsed")
        self.relate(start, x)
        self.relate(x, target)
        self.relate(start, y)
        self.relate(y, target)

        # start.label=3, y.label=2, target.label=3 => path_min via y = min(3,2,3) = 2
        # path_min via x = min(3,0,3) = 0
        # widest path should win: path_min for target = 2
        results = memdag_relate.neighborhood(self.conn, start, hops=2, min_integrity=0)
        by_id = {d["id"]: d for d in results}

        self.assertIn(target, by_id, "target should be reachable")
        self.assertEqual(by_id[target]["path_min"], 2,
                         "relaxation should pick widest path (via y, path_min=2) over x path (path_min=0)")


# ---------------------------------------------------------------------------
# Test 5: clearance filters by conf_label (results only, not conductance)
# ---------------------------------------------------------------------------

class TestClearanceFiltersConf(Base):
    def test_clearance_filters_conf(self):
        """A neighbor classified secret(2) is hidden at clearance='internal'(1)
        but visible at clearance='topsecret'(3).
        """
        hub = self.add("Hub endorsed content long enough here.", "endorsed")
        secret_node = self.add("Secret content long enough to appear here.", "user")
        self.relate(hub, secret_node)
        # Classify the neighbor as secret (conf=2)
        memdag_confid.classify(self.conn, secret_node, "secret")

        # clearance=internal(1): secret node hidden
        results_low = memdag_relate.neighborhood(
            self.conn, hub, hops=1, clearance="internal"
        )
        ids_low = {d["id"] for d in results_low}
        self.assertNotIn(secret_node, ids_low,
                         "secret node should be hidden at internal clearance")

        # clearance=topsecret(3): secret node visible
        results_high = memdag_relate.neighborhood(
            self.conn, hub, hops=1, clearance="topsecret"
        )
        ids_high = {d["id"] for d in results_high}
        self.assertIn(secret_node, ids_high,
                      "secret node should appear at topsecret clearance")


# ---------------------------------------------------------------------------
# Test 6: undirected — relate(a,b) means neighborhood(b) finds a
# ---------------------------------------------------------------------------

class TestUndirected(Base):
    def test_undirected(self):
        """relate(a, b) only; neighborhood(b) should find a (edge is undirected)."""
        a = self.add("Node a endorsed content long enough here.", "endorsed")
        b = self.add("Node b user content long enough here.", "user")
        self.relate(a, b)

        results = memdag_relate.neighborhood(self.conn, b, hops=1)
        ids = {d["id"] for d in results}
        self.assertIn(a, ids, "a should appear in neighborhood of b (undirected)")


# ---------------------------------------------------------------------------
# Test 7: tombstoned and quarantined nodes excluded AND non-conducting
# ---------------------------------------------------------------------------

class TestTombstonedAndQuarantinedExcludedAndNonconducting(Base):
    def test_tombstoned_excluded_and_nonconducting(self):
        """start -- middle(tombstoned) -- far: both middle and far should vanish."""
        start = self.add("Start endorsed content long enough here.", "endorsed")
        middle = self.add("Middle user content long enough here.", "user")
        far = self.add("Far endorsed content long enough here.", "endorsed")
        self.relate(start, middle)
        self.relate(middle, far)

        # Raw tombstone UPDATE (direct, like the spec says)
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET tombstoned=1, tombstoned_at=?, revoke_reason=? WHERE id=?",
                (memdag.now_iso(), "test tombstone", middle)
            )

        results = memdag_relate.neighborhood(self.conn, start, hops=3, min_integrity=0)
        ids = {d["id"] for d in results}
        self.assertNotIn(middle, ids, "tombstoned middle should not appear")
        self.assertNotIn(far, ids, "far should not appear (only path through tombstoned middle)")

    def test_quarantined_excluded_and_nonconducting(self):
        """start -- middle(quarantined) -- far: both middle and far should vanish."""
        start = self.add("Start endorsed content long enough here.", "endorsed")
        middle = self.add("Middle user content long enough here.", "user")
        far = self.add("Far endorsed content long enough here.", "endorsed")
        self.relate(start, middle)
        self.relate(middle, far)

        memdag_quarantine.quarantine_node(self.conn, middle, "suspected")

        results = memdag_relate.neighborhood(self.conn, start, hops=3, min_integrity=0)
        ids = {d["id"] for d in results}
        self.assertNotIn(middle, ids, "quarantined middle should not appear")
        self.assertNotIn(far, ids, "far should not appear (only path through quarantined middle)")


# ---------------------------------------------------------------------------
# Test 8: rel_edges distinct from provenance — revoke does NOT cascade
# ---------------------------------------------------------------------------

class TestRelEdgesDistinctFromProvenance(Base):
    def test_rel_edges_distinct_from_provenance(self):
        """relate(x, y); revoke_cascade(x) does NOT tombstone y.
        parents_of(y) is empty (y has no provenance edges).
        edges table count is unchanged.
        """
        x = self.add("Node x user content long enough here.", "user")
        y = self.add("Node y user content long enough here.", "user")
        self.relate(x, y)

        edges_before = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

        # Revoke x — this must NOT cascade to y via rel_edges
        memdag.revoke_cascade(self.conn, x, "test revoke")

        # y must still be alive
        node_y = memdag.get_node(self.conn, y)
        self.assertIsNotNone(node_y)
        self.assertEqual(node_y["tombstoned"], 0, "y should not be tombstoned")

        # y has no provenance parents
        parents = memdag.parents_of(self.conn, y)
        self.assertEqual(len(parents), 0, "y has no provenance edges")

        # provenance edges table unchanged
        edges_after = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(edges_before, edges_after, "provenance edges table should not change")


# ---------------------------------------------------------------------------
# Test 9: relate validation and idempotence
# ---------------------------------------------------------------------------

class TestRelateValidationAndIdempotence(Base):
    def test_relate_unknown_id_raises(self):
        """ValueError if either node id is unknown."""
        a = self.add("Node a content long enough here.", "user")
        with self.assertRaises(ValueError):
            memdag_relate.relate(self.conn, a, 99999)
        with self.assertRaises(ValueError):
            memdag_relate.relate(self.conn, 99999, a)

    def test_relate_self_raises(self):
        """ValueError if a == b."""
        a = self.add("Node a content long enough here.", "user")
        with self.assertRaises(ValueError):
            memdag_relate.relate(self.conn, a, a)

    def test_relate_idempotent(self):
        """Double relate leaves exactly one row in rel_edges."""
        a = self.add("Node a content long enough here.", "user")
        b = self.add("Node b content long enough here.", "user")
        memdag_relate.relate(self.conn, a, b)
        memdag_relate.relate(self.conn, a, b)  # idempotent: INSERT OR IGNORE
        count = self.conn.execute(
            "SELECT COUNT(*) FROM rel_edges WHERE a=? AND b=?", (a, b)
        ).fetchone()[0]
        self.assertEqual(count, 1, "double relate should leave exactly one row")


if __name__ == "__main__":
    unittest.main()
