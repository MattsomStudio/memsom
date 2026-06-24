#!/usr/bin/env python3
"""Tests for memdag_stale — the staleness cascade.

Run:
  python -W error::DeprecationWarning -m unittest discover -s . -p test_memdag_stale.py -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_schema
import memdag_ingest
import memdag_rederive
import memdag_stale


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_rederive.migrate(self.conn)
        memdag_stale.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    # helpers -----------------------------------------------------------------
    def src(self, content, channel="user", ref=None):
        with self.conn:
            return memdag.insert_node(
                self.conn, content, channel, memdag.RANK[channel], source_ref=ref)

    def der(self, content, parents):
        nid, _ = memdag.derive_node(self.conn, content, parents)
        return nid

    def edge(self, child, parent):
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO edges(child, parent) VALUES (?,?)", (child, parent))

    def is_stale(self, nid):
        return bool(self.conn.execute(
            "SELECT stale FROM nodes WHERE id=?", (nid,)).fetchone()[0])

    def derive_compose(self, question, parent_ids):
        """Build a derived node via compose + a recorded compose recipe."""
        qmarks = ",".join("?" * len(parent_ids))
        rows = self.conn.execute(
            f"SELECT id, content, channel, label, source_ref FROM nodes"
            f" WHERE id IN ({qmarks}) ORDER BY label DESC, id ASC",
            tuple(parent_ids)).fetchall()
        text, used = memdag.compose(question, rows)
        nid, _ = memdag.derive_node(self.conn, text, used)
        with self.conn:
            memdag_rederive.record_recipe(self.conn, nid, "compose", question=question)
        return nid, text


class TestMigrateIdempotent(Base):
    def test_migrate_twice(self):
        memdag_stale.migrate(self.conn)  # second call
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(nodes)")}
        self.assertTrue({"stale", "stale_at", "stale_reason"} <= cols)
        tabs = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"source_supersedes", "stale_log"} <= tabs)


class TestCascade(Base):
    def test_transitive_chain(self):
        a = self.src("alpha")
        b = self.der("derived b", [a])
        c = self.der("derived c", [b])
        n = memdag_stale.mark_stale_cascade(self.conn, a, "source changed")
        self.assertEqual(n, 3)
        for x in (a, b, c):
            self.assertTrue(self.is_stale(x))

    def test_cycle_terminates_marks_once(self):
        a = self.src("alpha")
        b = self.der("b", [a])
        c = self.der("c", [b])
        self.edge(a, c)  # c -> a back-edge => cycle a->b->c->a
        n = memdag_stale.mark_stale_cascade(self.conn, a, "x")
        self.assertEqual(n, 3)  # each marked exactly once despite the cycle

    def test_diamond_marked_once(self):
        a = self.src("alpha")
        b = self.der("b", [a])
        c = self.der("c", [a])
        d = self.der("d", [b, c])
        n = memdag_stale.mark_stale_cascade(self.conn, a, "x")
        self.assertEqual(n, 4)  # a,b,c,d each once

    def test_first_staleness_wins(self):
        a = self.src("alpha")
        b = self.der("b", [a])
        memdag_stale.mark_stale_cascade(self.conn, a, "first reason")
        at1 = self.conn.execute("SELECT stale_at, stale_reason FROM nodes WHERE id=?", (a,)).fetchone()
        # second cascade returns 0 newly-stale and preserves the original record
        n2 = memdag_stale.mark_stale_cascade(self.conn, a, "second reason")
        self.assertEqual(n2, 0)
        at2 = self.conn.execute("SELECT stale_at, stale_reason FROM nodes WHERE id=?", (a,)).fetchone()
        self.assertEqual(at1, at2)
        self.assertEqual(at2[1], "first reason")

    def test_liveness_and_edges_untouched(self):
        a = self.src("alpha")
        b = self.der("b", [a])
        edges_before = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        memdag_stale.mark_stale_cascade(self.conn, a, "x")
        # tombstoned stays 0 (stale != dead)
        for x in (a, b):
            self.assertEqual(
                self.conn.execute("SELECT tombstoned FROM nodes WHERE id=?", (x,)).fetchone()[0], 0)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0], edges_before)


class TestExcludeAndAnnotations(Base):
    def test_stale_not_in_taint_filter(self):
        clauses, _ = memdag_schema.taint_filter_clauses(self.conn)
        self.assertIn("tombstoned = 0", clauses)
        self.assertFalse(any("stale" in c for c in clauses))

    def test_exclude_clauses_optin(self):
        clauses, params = memdag_stale.stale_exclude_clauses(self.conn)
        self.assertEqual(clauses, ["stale = 0"])
        self.assertEqual(params, [])  # no bound param -> positional-safe

    def test_exclude_clauses_empty_premigrate(self):
        # fresh DB without the stale column
        db2 = Path(self.tmp.name) / "fresh" / "x.db"
        os.environ["MEMDAG_DB"] = str(db2)
        c2 = memdag.get_connection()
        try:
            clauses, params = memdag_stale.stale_exclude_clauses(c2)
            self.assertEqual((clauses, params), ([], []))
        finally:
            c2.close()
            os.environ["MEMDAG_DB"] = str(self.db)

    def test_annotations(self):
        a = self.src("alpha", ref="note.md")
        b = self.der("b", [a])
        memdag_stale.mark_stale_cascade(self.conn, a, "source changed")
        ann = memdag_stale.stale_annotations(self.conn, [a, b])
        self.assertEqual(set(ann), {a, b})
        self.assertEqual(ann[a]["reason"], "source changed")

    def test_annotations_empty_for_fresh(self):
        a = self.src("alpha")
        self.assertEqual(memdag_stale.stale_annotations(self.conn, [a]), {})


class TestSupersession(Base):
    def test_record_and_resolve(self):
        a = self.src("v1", ref="note.md")
        b = self.src("v2", ref="note.md")
        memdag_stale.record_source_supersession(self.conn, a, b, "note.md")
        self.assertEqual(memdag_stale.superseding_version(self.conn, a), b)
        self.assertEqual(memdag_stale.fresh_version_for(self.conn, a), b)

    def test_chain_walks_to_head(self):
        a = self.src("v1", ref="n")
        b = self.src("v2", ref="n")
        c = self.src("v3", ref="n")
        memdag_stale.record_source_supersession(self.conn, a, b, "n")
        memdag_stale.record_source_supersession(self.conn, b, c, "n")
        self.assertEqual(memdag_stale.fresh_version_for(self.conn, a), c)

    def test_once_forward_idempotent(self):
        a = self.src("v1", ref="n")
        b = self.src("v2", ref="n")
        memdag_stale.record_source_supersession(self.conn, a, b, "n")
        memdag_stale.record_source_supersession(self.conn, a, 999, "n")  # ignored (PK)
        self.assertEqual(memdag_stale.superseding_version(self.conn, a), b)

    def test_no_successor_returns_none(self):
        a = self.src("v1", ref="n")
        self.assertIsNone(memdag_stale.fresh_version_for(self.conn, a))


class TestIngestAutoDetect(Base):
    def test_reingest_changed_supersedes_and_stales(self):
        a = memdag_ingest.ingest_text(self.conn, "deadline is march", "user", source_ref="note.md")[0]
        d = self.der("derived from a", [a])
        b = memdag_ingest.ingest_text(self.conn, "deadline is april", "user", source_ref="note.md")[0]
        self.assertNotEqual(a, b)
        self.assertEqual(memdag_stale.superseding_version(self.conn, a), b)
        self.assertTrue(self.is_stale(a))
        self.assertTrue(self.is_stale(d))   # descendant cascaded
        self.assertFalse(self.is_stale(b))  # the fresh version is not stale

    def test_identical_reingest_dedups_no_trigger(self):
        a = memdag_ingest.ingest_text(self.conn, "same text", "user", source_ref="note.md")[0]
        again = memdag_ingest.ingest_text(self.conn, "same text", "user", source_ref="note.md")[0]
        self.assertEqual(a, again)  # dedup
        self.assertFalse(self.is_stale(a))
        self.assertIsNone(memdag_stale.superseding_version(self.conn, a))

    def test_different_ref_no_trigger(self):
        a = memdag_ingest.ingest_text(self.conn, "deadline march", "user", source_ref="a.md")[0]
        memdag_ingest.ingest_text(self.conn, "deadline april", "user", source_ref="b.md")[0]
        self.assertFalse(self.is_stale(a))
        self.assertIsNone(memdag_stale.superseding_version(self.conn, a))

    def test_dead_old_no_trigger(self):
        a = memdag_ingest.ingest_text(self.conn, "deadline march", "user", source_ref="note.md")[0]
        memdag.revoke_cascade(self.conn, a, "killed")  # old version tombstoned
        b = memdag_ingest.ingest_text(self.conn, "deadline april", "user", source_ref="note.md")[0]
        # taint-filtered predecessor lookup skips the dead old version
        self.assertIsNone(memdag_stale.superseding_version(self.conn, a))

    def test_no_source_ref_no_trigger(self):
        a = memdag_ingest.ingest_text(self.conn, "deadline march", "user")[0]
        b = memdag_ingest.ingest_text(self.conn, "deadline april", "user")[0]
        self.assertIsNone(memdag_stale.superseding_version(self.conn, a))

    def test_new_node_persists_if_trigger_raises(self):
        memdag_ingest.ingest_text(self.conn, "deadline march", "user", source_ref="note.md")
        orig = memdag_stale.on_reingest_supersede

        def boom(*a, **k):
            raise RuntimeError("boom")

        memdag_stale.on_reingest_supersede = boom
        try:
            ids = memdag_ingest.ingest_text(self.conn, "deadline april", "user", source_ref="note.md")
        finally:
            memdag_stale.on_reingest_supersede = orig
        self.assertTrue(ids and memdag.get_node(self.conn, ids[0]) is not None)


class TestFreshen(Base):
    def test_freshen_rewires_regenerates_archives_chains(self):
        a1 = memdag_ingest.ingest_text(self.conn, "the deadline is march fifteenth.", "user", source_ref="note.md")[0]
        d, old_text = self.derive_compose("deadline", [a1])
        a2 = memdag_ingest.ingest_text(self.conn, "the deadline is april twentieth.", "user", source_ref="note.md")[0]
        self.assertTrue(self.is_stale(d))
        res = memdag_stale.freshen(self.conn, d)
        new_id = res["regenerated"]
        self.assertIsNotNone(new_id)
        # old derived archived; new derives from the fresh source a2, not a1
        self.assertEqual(self.conn.execute("SELECT archived FROM nodes WHERE id=?", (d,)).fetchone()[0], 1)
        parents = [r[0] for r in self.conn.execute("SELECT parent FROM edges WHERE child=?", (new_id,)).fetchall()]
        self.assertIn(a2, parents)
        self.assertNotIn(a1, parents)
        # supersedes chained old->new
        self.assertEqual(
            memdag_rederive.get_recipe(self.conn, new_id)["supersedes"], d)
        # content changed
        new_text = memdag.get_node(self.conn, new_id)["content"]
        self.assertNotEqual(new_text, old_text)

    def test_freshen_noop_when_nothing_stale(self):
        a = memdag_ingest.ingest_text(self.conn, "deadline march", "user", source_ref="note.md")[0]
        d, _ = self.derive_compose("deadline", [a])
        res = memdag_stale.freshen(self.conn, d)
        self.assertEqual(res["rewired"], [])
        self.assertIsNone(res["regenerated"])

    def test_freshen_re_floors_label_down(self):
        # Manually supersede an endorsed source with a LOWER-integrity fresh
        # version and confirm freshen floors the derived label DOWN (regenerate's
        # min()). Proves freshen carries no label write of its own — it can only
        # re-derive, never inflate.
        a_hi = self.src("the deadline is march fifteenth.", channel="endorsed", ref="note.md")
        d, _ = self.derive_compose("deadline", [a_hi])
        self.assertEqual(memdag.get_node(self.conn, d)["label"], memdag.RANK["endorsed"])
        a_lo = self.src("the deadline is april twentieth.", channel="external", ref="note.md")
        memdag_stale.record_source_supersession(self.conn, a_hi, a_lo, "note.md")
        memdag_stale.mark_stale_cascade(self.conn, a_hi, "x")
        new_id = memdag_stale.freshen(self.conn, d)["regenerated"]
        self.assertIsNotNone(new_id)
        self.assertEqual(memdag.get_node(self.conn, new_id)["label"], memdag.RANK["external"])

    def test_freshen_never_rewires_onto_dead_version(self):
        # T3: if the fresh head is tombstoned, freshen must not repoint onto it.
        a1 = self.src("the deadline is march fifteenth.", ref="note.md")
        d, _ = self.derive_compose("deadline", [a1])
        a2 = self.src("the deadline is april twentieth.", ref="note.md")
        memdag_stale.record_source_supersession(self.conn, a1, a2, "note.md")
        memdag_stale.mark_stale_cascade(self.conn, a1, "x")
        memdag.revoke_cascade(self.conn, a2, "fresh version killed")  # head now dead
        res = memdag_stale.freshen(self.conn, d)
        self.assertEqual(res["rewired"], [])  # nothing to repoint onto
        # original edge intact
        parents = [r[0] for r in self.conn.execute("SELECT parent FROM edges WHERE child=?", (d,)).fetchall()]
        self.assertIn(a1, parents)


class TestUnstale(Base):
    def test_unstale_clears_one(self):
        a = self.src("alpha")
        b = self.der("b", [a])
        memdag_stale.mark_stale_cascade(self.conn, a, "x")
        n = memdag_stale.unstale(self.conn, a)
        self.assertEqual(n, 1)
        self.assertFalse(self.is_stale(a))
        self.assertTrue(self.is_stale(b))  # descendant unaffected (single-node)

    def test_unstale_noop_when_fresh(self):
        a = self.src("alpha")
        self.assertEqual(memdag_stale.unstale(self.conn, a), 0)


if __name__ == "__main__":
    unittest.main()
