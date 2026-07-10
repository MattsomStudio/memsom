#!/usr/bin/env python3
"""Tests for memsom_rederive — run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_rederive.py \
    -t <repo> -v
"""

import os, tempfile, unittest, warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.retrieval import rederive as memsom_rederive


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_rederive.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])


class TestMigrateIdempotent(Base):
    """test 1: migrate twice -> derivation_recipe exists exactly once, no error."""

    def test_migrate_twice(self):
        memsom_rederive.migrate(self.conn)  # second call (setUp ran the first)
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='derivation_recipe'"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        # columns are as declared
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(derivation_recipe)")}
        self.assertEqual(cols, {"node_id", "engine", "recipe_json", "supersedes"})


class TestRecipeCapture(Base):
    """test 2: record + read back a compose recipe and an extractive (NULL) recipe."""

    def test_compose_recipe_roundtrip(self):
        a = self.add("endorsed source content text here", "endorsed")
        c, _ = memsom.derive_node(self.conn, "derived from a", [a])
        with self.conn:
            memsom_rederive.record_recipe(self.conn, c, "compose", question="what is a?")

        got = memsom_rederive.get_recipe(self.conn, c)
        self.assertEqual(got["engine"], "compose")
        self.assertEqual(got["recipe"], {"question": "what is a?"})
        self.assertIsNone(got["supersedes"])

    def test_extractive_recipe_is_null_blob(self):
        a = self.add("endorsed source content text here", "endorsed")
        c, _ = memsom.derive_node(self.conn, "derived from a", [a])
        with self.conn:
            memsom_rederive.record_recipe(self.conn, c, "extractive")

        got = memsom_rederive.get_recipe(self.conn, c)
        self.assertEqual(got["engine"], "extractive")
        self.assertEqual(got["recipe"], {})
        row = self.conn.execute(
            "SELECT recipe_json FROM derivation_recipe WHERE node_id=?", (c,)).fetchone()
        self.assertIsNone(row[0])  # empty recipe -> NULL, not '{}'

    def test_unrecorded_node_returns_none(self):
        a = self.add("endorsed source content text here", "endorsed")
        c, _ = memsom.derive_node(self.conn, "derived from a", [a])
        self.assertIsNone(memsom_rederive.get_recipe(self.conn, c))  # legacy = None = 'unknown'

    def test_replace_on_conflict(self):
        a = self.add("endorsed source content text here", "endorsed")
        c, _ = memsom.derive_node(self.conn, "derived from a", [a])
        with self.conn:
            memsom_rederive.record_recipe(self.conn, c, "llm", model="x")
            memsom_rederive.record_recipe(self.conn, c, "compose", question="q")  # INSERT OR REPLACE
        got = memsom_rederive.get_recipe(self.conn, c)
        self.assertEqual(got["engine"], "compose")
        self.assertEqual(got["recipe"], {"question": "q"})


class RegenBase(Base):
    """Helpers for building real deterministic derived nodes."""

    def src_rows(self, ids):
        qs = ",".join("?" * len(ids))
        return self.conn.execute(
            f"SELECT id, content, channel, label, source_ref FROM nodes WHERE id IN ({qs})"
            " ORDER BY label DESC, id ASC", ids).fetchall()

    def make_compose(self, question, parent_ids):
        rows = self.src_rows(parent_ids)
        text, used = memsom.compose(question, rows)
        self.assertTrue(text, "test fixture: compose yielded nothing")
        with self.conn:
            c, _ = memsom.derive_node(self.conn, text, used)
            memsom_rederive.record_recipe(self.conn, c, "compose", question=question)
        return c, text

    def make_extractive(self, parent_ids, k=5):
        from memsom.lifecycle import compact as memsom_compact
        rows = sorted(((i, self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (i,)).fetchone()[0]) for i in parent_ids),
            key=lambda r: r[0])
        text = memsom_compact.extractive_summary(rows, k)
        with self.conn:
            c, _ = memsom.derive_node(self.conn, text, parent_ids)
            memsom_rederive.record_recipe(self.conn, c, "extractive", k=k)
        return c, text

    def n_nodes(self):
        return self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def archived(self, nid):
        return self.conn.execute(
            "SELECT COALESCE(archived,0) FROM nodes WHERE id=?", (nid,)).fetchone()[0]

    def parents(self, nid):
        return sorted(r[0] for r in self.conn.execute(
            "SELECT parent FROM edges WHERE child=?", (nid,)))

    def kill(self, nid):  # tombstone ONE node directly (no cascade) to isolate regenerate
        with self.conn:
            self.conn.execute("UPDATE nodes SET tombstoned=1 WHERE id=?", (nid,))


class TestRegenerateCompose(RegenBase):
    """test 3: a dead parent -> regenerate mints fresh from survivors, archives old."""

    def test_changed_mints_and_archives(self):
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        b = self.add("Apples can be pressed into cider during the autumn apple harvest.", "endorsed")
        c, c_text = self.make_compose("apples", [a, b])

        self.kill(b)
        new_id = memsom_rederive.regenerate(self.conn, c)

        self.assertIsNotNone(new_id)
        self.assertEqual(self.archived(c), 1)          # old retired
        self.assertEqual(self.archived(new_id), 0)     # new is live
        self.assertEqual(self.parents(new_id), [a])    # edges to LIVE parent only
        exp_text, _ = memsom.compose("apples", self.src_rows([a]))
        new_content = self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (new_id,)).fetchone()[0]
        self.assertEqual(new_content, exp_text)
        self.assertNotEqual(new_content, c_text)
        rec = memsom_rederive.get_recipe(self.conn, new_id)
        self.assertEqual(rec["engine"], "compose")
        self.assertEqual(rec["supersedes"], c)

    def test_no_churn_when_nothing_changed(self):
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        b = self.add("Apples can be pressed into cider during the autumn apple harvest.", "endorsed")
        c, _ = self.make_compose("apples", [a, b])

        before = self.n_nodes()
        self.assertIsNone(memsom_rederive.regenerate(self.conn, c))  # no parent died
        self.assertEqual(self.n_nodes(), before)       # nothing minted
        self.assertEqual(self.archived(c), 0)          # not retired

    def test_all_parents_gone_retire_no_replace(self):
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        b = self.add("Apples can be pressed into cider during the autumn apple harvest.", "endorsed")
        c, _ = self.make_compose("apples", [a, b])
        self.kill(a)
        self.kill(b)
        before = self.n_nodes()
        self.assertIsNone(memsom_rederive.regenerate(self.conn, c))
        self.assertEqual(self.n_nodes(), before)
        self.assertEqual(self.archived(c), 0)          # left as-is (cascade owns the tombstone)

    def test_idempotent_retry(self):
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        b = self.add("Apples can be pressed into cider during the autumn apple harvest.", "endorsed")
        c, _ = self.make_compose("apples", [a, b])
        self.kill(b)

        first = memsom_rederive.regenerate(self.conn, c)
        n_after_first = self.n_nodes()
        self.assertIsNone(memsom_rederive.regenerate(self.conn, c))     # c now archived -> no-op
        self.assertIsNone(memsom_rederive.regenerate(self.conn, first)) # fresh -> unchanged -> no-op
        self.assertEqual(self.n_nodes(), n_after_first)                 # no extra mint


class TestRegenerateGuards(RegenBase):
    """test 5: non-deterministic / legacy engines are flagged, never auto-rolled."""

    def test_llm_engine_not_rolled(self):
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        b = self.add("Apples can be pressed into cider during the autumn apple harvest.", "endorsed")
        with self.conn:
            c, _ = memsom.derive_node(self.conn, "llm prose summary", [a, b])
            memsom_rederive.record_recipe(self.conn, c, "llm", model="x", prompt="p")
        self.kill(b)
        self.assertIsNone(memsom_rederive.regenerate(self.conn, c))

    def test_unknown_legacy_node_not_rolled(self):
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        with self.conn:
            c, _ = memsom.derive_node(self.conn, "legacy derived", [a])  # no recipe recorded
        self.kill(a)
        self.assertIsNone(memsom_rederive.regenerate(self.conn, c))


class TestRegenerateExtractive(RegenBase):
    """test 6: extractive replays at the stored k from the surviving episodes."""

    def test_extractive_regenerates(self):
        from memsom.lifecycle import compact as memsom_compact
        a = self.add("First episode about networking and TCP handshakes in detail here.", "user")
        b = self.add("Second episode about TCP congestion control and window scaling here.", "user")
        c, c_text = self.make_extractive([a, b], k=5)
        self.kill(b)

        new_id = memsom_rederive.regenerate(self.conn, c)
        self.assertIsNotNone(new_id)
        self.assertEqual(self.parents(new_id), [a])
        exp = memsom_compact.extractive_summary(
            [(a, self.conn.execute("SELECT content FROM nodes WHERE id=?", (a,)).fetchone()[0])], 5)
        new_content = self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (new_id,)).fetchone()[0]
        self.assertEqual(new_content, exp)
        self.assertNotEqual(new_content, c_text)


class TestErase(RegenBase):
    """G5: erase scrubs the whole lineage — descendants AND superseded copies."""

    def content(self, nid):
        return self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (nid,)).fetchone()[0]

    def test_erase_source_scrubs_derived_keeps_edges(self):
        a = self.add("Secret token ABC123 lives in this source sentence here now.", "user")
        b = self.add("Another unrelated note about apples and oranges for context here.", "user")
        c, _ = self.make_compose("token", [a, b])

        erased = memsom_rederive.erase(self.conn, a, "PII scrub")

        self.assertIn(a, erased)
        self.assertIn(c, erased)                 # derived summary scrubbed
        self.assertEqual(self.content(a), "")
        self.assertEqual(self.content(c), "")
        self.assertEqual(self.parents(c), sorted([a, b]))  # edges intact -> blame still walks
        # a sibling source NOT in a's lineage is untouched
        self.assertNotEqual(self.content(b), "")

    def test_erase_follows_supersedes_chain(self):
        """The load-bearing case: erasing the live version must also scrub the
        archived predecessor that redact_node's cascade would miss."""
        a = self.add("Apples are red fruits that grow on apple trees in orchards.", "endorsed")
        b = self.add("Apples can be pressed into cider during the autumn apple harvest.", "endorsed")
        c, _ = self.make_compose("apples", [a, b])
        self.kill(b)
        cprime = memsom_rederive.regenerate(self.conn, c)   # c archived, cprime.supersedes = c
        self.assertIsNotNone(cprime)
        self.assertEqual(self.archived(c), 1)

        erased = memsom_rederive.erase(self.conn, cprime, "scrub the secret")

        self.assertIn(cprime, erased)
        self.assertIn(c, erased)                  # SUPERSEDED archived copy also scrubbed
        self.assertEqual(self.content(cprime), "")
        self.assertEqual(self.content(c), "")     # the copy holding the secret is gone
        # live parent (not in cprime's version lineage) is untouched
        self.assertEqual(self.content(a),
                         "Apples are red fruits that grow on apple trees in orchards.")

    def test_erase_is_idempotent(self):
        a = self.add("Secret token ABC123 lives in this source sentence here now.", "user")
        c, _ = self.make_compose("token", [a])
        first = memsom_rederive.erase(self.conn, a, "scrub")
        second = memsom_rederive.erase(self.conn, a, "scrub again")  # already-redacted skipped
        self.assertEqual(second, [])              # nothing newly destroyed
        self.assertIn(a, first)
        self.assertIn(c, first)


if __name__ == "__main__":
    unittest.main()
