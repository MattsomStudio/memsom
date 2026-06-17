#!/usr/bin/env python3
"""Tests for memdag_confid — Bell-LaPadula confidentiality axis.

Run:
    python -W error::DeprecationWarning -m unittest discover \
        -s <repo> -p test_memdag_confid.py \
        -t <repo> -v
"""

import contextlib, io, os, tempfile, unittest, warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_confid
import memdag_compact
import memdag_retrieve

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    """Temp-DB base — identical pattern to test_memdag.py Base."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "confid_test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_confid.migrate(self.conn)   # add conf_label column

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])

    # helpers

    def get_conf(self, nid):
        row = self.conn.execute("SELECT conf_label FROM nodes WHERE id = ?", (nid,)).fetchone()
        return row[0]

    def get_label(self, nid):
        """Biba integrity label."""
        return memdag.get_node(self.conn, nid)["label"]


# ---------------------------------------------------------------------------
# 1. Derived conf = MAX(parents), NOT min
# ---------------------------------------------------------------------------

class TestDerivedConfIsMaxNotMin(Base):

    def test_derived_conf_is_max_not_min(self):
        """Derived node conf = MAX(parent confs) — high-water-mark, not Biba floor."""
        p = self.add("public source prose line", "endorsed")
        s = self.add("secret source prose line", "user")

        memdag_confid.classify(self.conn, p, "public")   # conf 0
        memdag_confid.classify(self.conn, s, "secret")   # conf 2

        d, _ = memdag.derive_node(self.conn, "derived answer", [p, s])

        old, new = memdag_confid.recompute_conf(self.conn, d)
        self.assertEqual(old, 0)  # born public (default)
        self.assertEqual(new, 2)  # rose to MAX(0, 2)
        self.assertEqual(self.get_conf(d), 2)

        # Contrast: integrity (Biba) label must still be MIN of parents
        # parent labels: endorsed=3, user=2 => min=2
        self.assertEqual(self.get_label(d), min(memdag.RANK["endorsed"], memdag.RANK["user"]))


# ---------------------------------------------------------------------------
# 2. The two axes are independent
# ---------------------------------------------------------------------------

class TestAxesIndependent(Base):

    def test_axes_independent(self):
        """Low-integrity source can be topsecret; high-integrity source can be public.
        A node derived from both gets integrity=MIN=0 AND conf=MAX=3.
        """
        # external (integrity 0) classified topsecret (conf 3)
        ext = self.add("external topsecret source prose", "external")
        memdag_confid.classify(self.conn, ext, "topsecret")
        self.assertEqual(self.get_label(ext), 0)   # integrity stays 0
        self.assertEqual(self.get_conf(ext), 3)    # conf is 3

        # endorsed (integrity 3) stays public (conf 0)
        end = self.add("endorsed public source prose", "endorsed")
        # no classify call — conf stays at default 0
        self.assertEqual(self.get_label(end), 3)   # integrity 3
        self.assertEqual(self.get_conf(end), 0)    # conf 0

        # Derive from both
        d, d_label = memdag.derive_node(self.conn, "derived answer text", [ext, end])

        # Biba integrity: min(0, 3) = 0
        self.assertEqual(d_label, 0)

        # Confidentiality: max(3, 0) = 3 (after recompute)
        old, new = memdag_confid.recompute_conf(self.conn, d)
        self.assertEqual(new, 3)

        # Axes pull in opposite directions in this node
        self.assertEqual(self.get_label(d), 0)   # integrity: floor
        self.assertEqual(self.get_conf(d), 3)    # confidentiality: ceiling


# ---------------------------------------------------------------------------
# 3. Clearance filter hides above-clearance sources
# ---------------------------------------------------------------------------

class TestClearanceFilterHidesAbove(Base):

    def test_clearance_filter_hides_above(self):
        """sources_for_clearance(internal) must hide secret; allow public+internal."""
        n0 = self.add("public source prose line one", "external")
        n1 = self.add("internal source prose line two", "user")
        n2 = self.add("secret source prose line three", "endorsed")

        memdag_confid.classify(self.conn, n0, "public")    # 0
        memdag_confid.classify(self.conn, n1, "internal")  # 1
        memdag_confid.classify(self.conn, n2, "secret")    # 2

        visible = memdag_confid.sources_for_clearance(self.conn, "internal")
        self.assertIn(n0, visible)
        self.assertIn(n1, visible)
        self.assertNotIn(n2, visible)
        self.assertEqual(len(visible), 2)

        all_ids = memdag_confid.sources_for_clearance(self.conn, 3)  # topsecret
        self.assertIn(n0, all_ids)
        self.assertIn(n1, all_ids)
        self.assertIn(n2, all_ids)
        self.assertEqual(len(all_ids), 3)


# ---------------------------------------------------------------------------
# 4. Multi-hop high-water propagation in one ordered pass
# ---------------------------------------------------------------------------

class TestMultihopHighWater(Base):

    def test_multihop_high_water(self):
        """s(secret) -> d1 -> d2: recompute_conf_all raises both to 2 in one pass."""
        s = self.add("secret source prose line long enough", "user")
        memdag_confid.classify(self.conn, s, "secret")  # conf 2

        d1, _ = memdag.derive_node(self.conn, "first derived node answer", [s])
        d2, _ = memdag.derive_node(self.conn, "second derived node answer", [d1])

        # Both born at conf 0 (default)
        self.assertEqual(self.get_conf(d1), 0)
        self.assertEqual(self.get_conf(d2), 0)

        changed = memdag_confid.recompute_conf_all(self.conn)

        # Both must appear in changed list
        changed_ids = {r[0]: (r[1], r[2]) for r in changed}
        self.assertIn(d1, changed_ids)
        self.assertIn(d2, changed_ids)
        self.assertEqual(changed_ids[d1], (0, 2))
        self.assertEqual(changed_ids[d2], (0, 2))

        self.assertEqual(self.get_conf(d1), 2)
        self.assertEqual(self.get_conf(d2), 2)

        # Second call: everything already correct — no changes
        second = memdag_confid.recompute_conf_all(self.conn)
        self.assertEqual(second, [])


# ---------------------------------------------------------------------------
# 5. Sources are untouched by recompute
# ---------------------------------------------------------------------------

class TestSourcesUntouched(Base):

    def test_sources_untouched(self):
        """recompute_conf on a source returns (c, c); recompute_conf_all never lists sources."""
        src = self.add("some source node prose line", "endorsed")
        memdag_confid.classify(self.conn, src, "secret")  # conf 2

        old, new = memdag_confid.recompute_conf(self.conn, src)
        self.assertEqual(old, 2)
        self.assertEqual(new, 2)  # no change, no write

        # Derived node alongside the source — only the derived node should appear
        d, _ = memdag.derive_node(self.conn, "derived node answer text", [src])
        # Classify derived node manually to a wrong value so recompute_conf_all has something to fix
        memdag_confid.classify(self.conn, d, "public")

        changed = memdag_confid.recompute_conf_all(self.conn)
        changed_ids = [r[0] for r in changed]
        self.assertNotIn(src, changed_ids)   # source never listed
        self.assertIn(d, changed_ids)        # derived node was corrected


# ---------------------------------------------------------------------------
# 6. parse_conf validation
# ---------------------------------------------------------------------------

class TestParseConfValidation(Base):

    def test_parse_conf_validation(self):
        """'SECRET', 'secret', 2 -> 2; ValueError on 'ultra' and 5."""
        self.assertEqual(memdag_confid.parse_conf("SECRET"), 2)
        self.assertEqual(memdag_confid.parse_conf("secret"), 2)
        self.assertEqual(memdag_confid.parse_conf(2), 2)

        with self.assertRaises(ValueError):
            memdag_confid.parse_conf("ultra")
        with self.assertRaises(ValueError):
            memdag_confid.parse_conf(5)


# ---------------------------------------------------------------------------
# 7. Tombstoned parent excluded from recompute
# ---------------------------------------------------------------------------

class TestTombstonedParentExcluded(Base):

    def test_tombstoned_parent_excluded(self):
        """d derived from [public p, secret s]; tombstone s; recompute_conf(d) uses only p -> 0."""
        p = self.add("public parent source prose line", "user")
        s = self.add("secret parent source prose line", "endorsed")

        memdag_confid.classify(self.conn, p, "public")   # 0
        memdag_confid.classify(self.conn, s, "secret")   # 2

        d, _ = memdag.derive_node(self.conn, "derived answer node", [p, s])

        # First recompute: should see both parents -> conf 2
        old, new = memdag_confid.recompute_conf(self.conn, d)
        self.assertEqual(new, 2)

        # Raw-UPDATE s to tombstoned (bypass revoke_cascade to keep d live)
        with self.conn:
            self.conn.execute("UPDATE nodes SET tombstoned = 1 WHERE id = ?", (s,))

        # Recompute again: only p (conf 0) is live
        old2, new2 = memdag_confid.recompute_conf(self.conn, d)
        self.assertEqual(old2, 2)  # was 2
        self.assertEqual(new2, 0)  # drops to public — secret parent is gone
        self.assertEqual(self.get_conf(d), 0)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCli(Base):

    def run_main(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memdag_confid.main(list(argv))
        return buf.getvalue()

    def test_cli_classify_and_recompute(self):
        n = self.add("some source node", "user")
        out = self.run_main("classify", str(n), "--level", "secret")
        self.assertIn(f"[{n}] conf=SECRET", out)

        # Derived node
        d, _ = memdag.derive_node(self.conn, "derived answer text", [n])
        out2 = self.run_main("conf-recompute", str(d))
        self.assertIn(f"[{d}]", out2)
        self.assertIn("SECRET", out2)

    def test_cli_recompute_all(self):
        n = self.add("internal source node", "user")
        memdag_confid.classify(self.conn, n, "internal")
        d, _ = memdag.derive_node(self.conn, "derived node answer", [n])
        out = self.run_main("conf-recompute", "--all")
        self.assertIn(f"[{d}]", out)
        self.assertIn("INTERNAL", out)

    def test_cli_recompute_unchanged(self):
        n = self.add("public source node text", "user")
        d, _ = memdag.derive_node(self.conn, "derived text answer node", [n])
        # conf 0 for both — no change expected
        out = self.run_main("conf-recompute", str(d))
        self.assertIn("unchanged", out)

    def test_cli_recompute_all_nothing_changed(self):
        n = self.add("public source node text", "user")
        memdag.derive_node(self.conn, "derived answer node text", [n])
        out = self.run_main("conf-recompute", "--all")
        self.assertIn("nothing changed", out)


# ---------------------------------------------------------------------------
# 8. Bypass-2G/2H: archived-parent conf laundering is blocked
# ---------------------------------------------------------------------------

class TestArchivedConfLaunderingBlocked(Base):
    """Operator-class regression for Bypass-2G/2H (AUDIT 2026-06-11).

    Compact two SECRET episodes into a summary, then try to declassify the
    now-archived source parents to PUBLIC and re-run recompute_conf. The
    summary must stay SECRET and no SECRET content may surface at public
    clearance.
    """

    def setUp(self):
        super().setUp()
        memdag_compact.migrate(self.conn)
        memdag_retrieve.migrate(self.conn)

    def test_classify_archived_parents_cannot_launder_summary(self):
        s1 = memdag.insert_node(
            self.conn, "Secret eta: master-crypto-key-ZZ never distribute", "user")
        s2 = memdag.insert_node(
            self.conn, "Secret theta: master-crypto-key-WW never distribute backup", "user")
        memdag_confid.classify(self.conn, s1, "secret")
        memdag_confid.classify(self.conn, s2, "secret")

        minted = memdag_compact.compact(
            self.conn, group_by="similarity", min_group=2, sim_threshold=0.0)
        self.assertTrue(minted)
        nid = minted[0]
        self.assertEqual(self.get_conf(nid), 2)  # summary born SECRET (high-water)

        # Both sources are now archived. Downgrading them must be refused.
        for sid in (s1, s2):
            self.assertEqual(
                self.conn.execute(
                    "SELECT archived FROM nodes WHERE id=?", (sid,)).fetchone()[0], 1)
            with self.assertRaises(ValueError):
                memdag_confid.classify(self.conn, sid, "public")
            self.assertEqual(self.get_conf(sid), 2)  # unchanged

        # Even after a recompute pass the summary stays SECRET (parents still 2).
        memdag_confid.recompute_conf(self.conn, nid)
        self.assertEqual(self.get_conf(nid), 2)

        # sources_for_clearance(public) must not surface the archived sources.
        pub = memdag_confid.sources_for_clearance(self.conn, "public")
        self.assertNotIn(s1, pub)
        self.assertNotIn(s2, pub)

    def test_archived_node_can_still_be_raised(self):
        """The guard only blocks DOWNGRADES — raising an archived node is allowed."""
        s = memdag.insert_node(self.conn, "internal alpha note one two three", "user")
        memdag_confid.classify(self.conn, s, "internal")  # 1
        memdag.insert_node(self.conn, "internal alpha note one two four", "user")
        # second source so a group of >=2 forms on shared tokens
        for extra in self.conn.execute(
                "SELECT id FROM nodes WHERE channel='user'").fetchall():
            memdag_confid.classify(self.conn, extra[0], "internal")
        memdag_compact.compact(self.conn, group_by="similarity",
                               min_group=2, sim_threshold=0.0)
        # s is archived now; raising it to secret is fine
        memdag_confid.classify(self.conn, s, "secret")
        self.assertEqual(self.get_conf(s), 2)

    def test_retrieve_public_never_surfaces_secret_after_launder_attempt(self):
        s1 = memdag.insert_node(
            self.conn, "Secret iota final-key-alpha-99 never distribute anywhere", "user")
        s2 = memdag.insert_node(
            self.conn, "Secret kappa final-key-beta-88 never distribute anywhere", "user")
        memdag_confid.classify(self.conn, s1, "secret")
        memdag_confid.classify(self.conn, s2, "secret")
        minted = memdag_compact.compact(
            self.conn, group_by="similarity", min_group=2, sim_threshold=0.0)
        nid = minted[0]
        memdag_retrieve.index_node(self.conn, s1)
        memdag_retrieve.index_node(self.conn, s2)
        memdag_retrieve.index_node(self.conn, nid)

        # Launder attempt (each classify raises, swallow it) + recompute.
        for sid in (s1, s2):
            with contextlib.suppress(ValueError):
                memdag_confid.classify(self.conn, sid, "public")
        memdag_confid.recompute_conf(self.conn, nid)

        results = memdag_retrieve.retrieve(
            self.conn, "final-key distribute", clearance="public", k=10)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# CONFID-1 / CONFID-2: recompute_conf_all must be ORDER-INDEPENDENT.
# Federation assigns local ids in arrival order, so a derived child can land
# with a LOWER id than its derived parent. The old single ORDER BY id pass
# computed the child against its parent's stale conf and never re-raised it.
# ---------------------------------------------------------------------------

class TestRecomputeConfOrderIndependent(Base):

    def _out_of_order_dag(self):
        """Build child(id<parent) <- parent(derived) <- secret source.

        Returns (child_id, parent_id, source_id) with child_id < parent_id.
        """
        with self.conn:
            dc = memdag.insert_node(self.conn, "child", "agent-derived", 0)   # id 1
        with self.conn:
            s = memdag.insert_node(self.conn, "secret source", "endorsed", 3)  # id 2
        with self.conn:
            dp = memdag.insert_node(self.conn, "mid", "agent-derived", 0)      # id 3
        with self.conn:
            self.conn.execute("UPDATE nodes SET conf_label=3 WHERE id=?", (s,))  # SECRET source
            self.conn.execute("INSERT INTO edges(child,parent) VALUES (?,?)", (dp, s))
            self.conn.execute("INSERT INTO edges(child,parent) VALUES (?,?)", (dc, dp))
        self.assertLess(dc, dp, "child must have a lower id than its derived parent")
        return dc, dp, s

    def test_high_water_propagates_despite_low_child_id(self):
        dc, dp, s = self._out_of_order_dag()
        # Parent's stored conf is stale (0) when the low-id child is first visited.
        memdag_confid.recompute_conf_all(self.conn)
        self.assertEqual(self.get_conf(dp), 3, "derived parent must rise to SECRET")
        self.assertEqual(self.get_conf(dc), 3,
                         "low-id child must ALSO rise — not be stranded at PUBLIC")

    def test_idempotent_on_correct_db(self):
        self._out_of_order_dag()
        memdag_confid.recompute_conf_all(self.conn)         # converge
        again = memdag_confid.recompute_conf_all(self.conn)  # CONFID-2
        self.assertEqual(again, [], "second call on a correct DB must return []")


if __name__ == "__main__":
    unittest.main()
