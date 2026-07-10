#!/usr/bin/env python3
"""Tests for memsom_contradict — cross-source contradiction detector.

The adjudicator (anchored per-sentence NLI) is injected as a deterministic stub in
most tests, so no embedder / NLI weights are needed. The anchored-adjudicator tests
monkeypatch memsom_embed with controlled vectors to exercise the anchor gate. Real
models are validated separately by bench/contradict_eval.py against the backup brain.

Run:
  python -W error::DeprecationWarning -m unittest discover \
      -s <repo> -p test_memsom_contradict.py -t <repo> -v
"""

import os
import tempfile
import unittest
import unittest.mock
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.interface import ingest as memsom_ingest
from memsom.integrity import stale as memsom_stale
from memsom.integrity import contradict as memsom_contradict


def flag_when(pred, score=0.99):
    """Adjudicator stub: verdict when pred(new_text, cand_text) is true, else None."""
    return lambda new, cand: ("nli", "stub", score) if pred(new, cand) else None


# the canonical contradiction fixture: WAF vendor disagreement (either direction)
WAF_ADJ = flag_when(lambda n, c: {"sucuri", "cloudflare"} <= {w.lower() for w in (n + " " + c).split()}
                    or ("sucuri" in (n + c).lower() and "cloudflare" in (n + c).lower()))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_ingest.migrate(self.conn)
        memsom_stale.migrate(self.conn)
        memsom_contradict.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        for k in ("MEMDAG_DB", "MEMDAG_CONTRADICT", "MEMDAG_CONTRADICT_NLI",
                  "MEMDAG_CONTRADICT_ENFORCE", "MEMDAG_CONTRADICT_NLI_THRESHOLD",
                  "MEMDAG_CONTRADICT_ANCHOR"):
            os.environ.pop(k, None)
        self.tmp.cleanup()

    def add(self, content, channel="user", source_ref=None):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, source_ref=source_ref)

    def state(self, nid):
        return self.conn.execute(
            "SELECT stale, stale_reason FROM nodes WHERE id=?", (nid,)).fetchone()


class TestDetectCore(Base):
    def test_enforce_stales_older_and_records(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")],
            adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(marked, [(old, "nli")])
        self.assertEqual(self.state(old)[0], 1)
        self.assertTrue(self.state(old)[1].startswith(memsom_contradict.REASON_PREFIX))

    def test_directionality_older_loses(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")   # lower id
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        memsom_contradict.detect(self.conn, new,
                                 candidates=[(old, "Sucuri is the active WAF.")],
                                 adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(self.state(old)[0], 1)   # older loses
        self.assertEqual(self.state(new)[0], 0)   # newer survives

    def test_no_verdict_no_action(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("The mesh uses UDP on every host.", source_ref="b")
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")],
            adjudicate=WAF_ADJ, enforce=True)         # stub won't fire (no cloudflare)
        self.assertEqual(marked, [])
        self.assertEqual(self.state(old)[0], 0)

    def test_never_writes_source_supersedes(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        memsom_contradict.detect(self.conn, new,
                                 candidates=[(old, "Sucuri is the active WAF.")],
                                 adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM source_supersedes").fetchone()[0], 0)

    def test_idempotent_no_double_record(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        for _ in range(2):
            memsom_contradict.detect(self.conn, new,
                                     candidates=[(old, "Sucuri is the active WAF.")],
                                     adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(len(memsom_contradict.list_contradictions(self.conn)), 1)

    def test_agent_derived_probe_skipped(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", channel="agent-derived")
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")],
            adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(marked, [])

    def test_no_adjudicator_is_noop(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        # adjudicate not injected, env off -> _default_adjudicator() is None -> no-op
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")])
        self.assertEqual(marked, [])


class TestObserveOnly(Base):
    def test_observe_records_but_does_not_stale(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")],
            adjudicate=WAF_ADJ, enforce=False)
        self.assertEqual(marked, [(old, "nli")])          # detected + returned
        self.assertEqual(self.state(old)[0], 0)           # but NOT staled
        rows = memsom_contradict.list_contradictions(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["enforced"])

    def test_enforce_env_flips_default(self):
        os.environ["MEMDAG_CONTRADICT_ENFORCE"] = "1"
        self.assertTrue(memsom_contradict._enforce_default())
        os.environ.pop("MEMDAG_CONTRADICT_ENFORCE")
        self.assertFalse(memsom_contradict._enforce_default())

    def test_observed_only_filter(self):
        a1 = self.add("Sucuri is the active WAF.", source_ref="a")
        b1 = self.add("Cloudflare is the active WAF.", source_ref="b")
        memsom_contradict.detect(self.conn, b1,
                                 candidates=[(a1, "Sucuri is the active WAF.")],
                                 adjudicate=WAF_ADJ, enforce=False)
        c1 = self.add("Sucuri is the active WAF here.", source_ref="c")
        d1 = self.add("Cloudflare is the active WAF here.", source_ref="d")
        memsom_contradict.detect(self.conn, d1,
                                 candidates=[(c1, "Sucuri is the active WAF here.")],
                                 adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(len(memsom_contradict.list_contradictions(self.conn)), 2)
        obs = memsom_contradict.list_contradictions(self.conn, observed_only=True)
        self.assertEqual(len(obs), 1)
        self.assertFalse(obs[0]["enforced"])

    def test_reason_namespace_survives_verify_stale(self):
        from memsom.integrity import verify_stale as memsom_verify_stale
        old = self.add("Sucuri is the active WAF.", source_ref="memory:acme_waf")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        memsom_contradict.detect(self.conn, new,
                                 candidates=[(old, "Sucuri is the active WAF.")],
                                 adjudicate=WAF_ADJ, enforce=True)
        self.assertEqual(self.state(old)[0], 1)
        memsom_verify_stale.recompute_verify_stale(self.conn)
        self.assertEqual(self.state(old)[0], 1)   # not our namespace to clear


class TestAdjudicatorResolution(Base):
    def test_off_unless_opted_in(self):
        self.assertIsNone(memsom_contradict._default_adjudicator())

    def test_none_when_nli_unavailable(self):
        os.environ["MEMDAG_CONTRADICT_NLI"] = "1"
        with unittest.mock.patch.object(memsom_contradict, "nli_available",
                                        return_value=False):
            self.assertIsNone(memsom_contradict._default_adjudicator())

    def test_built_when_opted_in_and_available(self):
        os.environ["MEMDAG_CONTRADICT_NLI"] = "1"
        from memsom.retrieval import embed as memsom_embed
        with unittest.mock.patch.object(memsom_contradict, "nli_available",
                                        return_value=True), \
             unittest.mock.patch.object(memsom_embed, "bge_available",
                                        return_value=True):
            adj = memsom_contradict._default_adjudicator()
            self.assertTrue(callable(adj))


class TestSentenceSplit(Base):
    def test_strips_frontmatter_and_code_and_short_frags(self):
        text = ("---\nname: x\nsection: y\n---\n"
                "Sucuri is the active WAF for the site.\n"
                "```\nignore=this code fence\n```\n"
                "- a short one\n"
                "Cloudflare took over the WAF role last quarter.")
        sents = memsom_contradict._sentences(text)
        joined = " ".join(sents)
        self.assertIn("Sucuri is the active WAF", joined)
        self.assertIn("Cloudflare took over", joined)
        self.assertNotIn("ignore=this", joined)     # code fence dropped
        self.assertNotIn("name: x", joined)         # frontmatter dropped


class TestAnchoredAdjudicator(Base):
    """Exercise the anchor gate with controlled sentence embeddings: sentences are
    mapped to orthogonal topic vectors, so same-topic pairs clear the anchor and
    cross-topic pairs don't — no real embedder / NLI needed."""

    def _topic_vec(self, text):
        t = text.lower()
        if "waf" in t:
            return {"dense": [1.0, 0.0, 0.0]}
        if "port" in t:
            return {"dense": [0.0, 1.0, 0.0]}
        return {"dense": [0.0, 0.0, 1.0]}

    def _patched(self, nli_fn, anchor=0.75, threshold=0.85):
        from memsom.retrieval import embed as memsom_embed
        p1 = unittest.mock.patch.object(memsom_embed, "bge_available", return_value=True)
        p2 = unittest.mock.patch.object(memsom_embed, "encode_doc", side_effect=self._topic_vec)
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)
        return memsom_contradict._make_anchored_adjudicator(
            nli_fn, anchor=anchor, threshold=threshold)

    def test_same_topic_high_nli_flags(self):
        adj = self._patched(lambda prem, hyp: 0.97)
        v = adj("Cloudflare is the active WAF.", "Sucuri is the active WAF.")
        self.assertIsNotNone(v)
        self.assertEqual(v[0], "nli")
        self.assertAlmostEqual(v[2], 0.97)

    def test_cross_topic_never_reaches_nli(self):
        called = []
        adj = self._patched(lambda prem, hyp: called.append(1) or 0.99)
        v = adj("Cloudflare is the active WAF.", "The service listens on port 8080.")
        self.assertIsNone(v)              # anchor gate rejects: WAF vs port -> cosine 0
        self.assertEqual(called, [])      # NLI never invoked

    def test_same_topic_low_nli_ignored(self):
        adj = self._patched(lambda prem, hyp: 0.10)
        v = adj("Cloudflare is the active WAF.", "Sucuri is the active WAF.")
        self.assertIsNone(v)              # anchored but NLI below threshold


class TestSweep(Base):
    def _all_others(self, pid, _ptext):
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT id, content FROM nodes WHERE id != ? AND tombstoned = 0", (pid,))]

    def test_backfill_older_loses_converges(self):
        old = self.add("Sucuri is the active WAF.", source_ref="memory:acme")
        new = self.add("Cloudflare is the active WAF.", source_ref="session:x")
        stats = memsom_contradict.sweep(
            self.conn, backfill=True, enforce=True, adjudicate=WAF_ADJ,
            candidate_fn=self._all_others)
        self.assertEqual(stats["probed"], 2)
        self.assertEqual(stats["contradictions"], 1)   # both probed, one record
        self.assertEqual(self.state(old)[0], 1)
        self.assertEqual(self.state(new)[0], 0)

    def test_defaults_to_observe(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        self.add("Cloudflare is the active WAF.", source_ref="b")
        stats = memsom_contradict.sweep(
            self.conn, backfill=True, adjudicate=WAF_ADJ, candidate_fn=self._all_others)
        self.assertFalse(stats["enforced"])
        self.assertEqual(stats["contradictions"], 1)   # recorded
        self.assertEqual(self.state(old)[0], 0)        # nothing staled

    def test_cursor_advances_and_incremental_skips(self):
        self.add("Sucuri is the active WAF.", source_ref="a")
        self.add("Cloudflare is the active WAF.", source_ref="b")
        memsom_contradict.sweep(self.conn, backfill=True, adjudicate=WAF_ADJ,
                                candidate_fn=self._all_others)
        s2 = memsom_contradict.sweep(self.conn, adjudicate=WAF_ADJ,
                                     candidate_fn=self._all_others)
        self.assertEqual(s2["probed"], 0)
        self.add("A third unrelated note about lifting.", source_ref="c")
        s3 = memsom_contradict.sweep(self.conn, adjudicate=WAF_ADJ,
                                     candidate_fn=self._all_others)
        self.assertEqual(s3["probed"], 1)

    def test_limit_defers_and_resumes(self):
        for i in range(4):
            self.add(f"Note number {i} about topic {i}.", source_ref=str(i))
        s1 = memsom_contradict.sweep(self.conn, backfill=True, limit=2,
                                     adjudicate=WAF_ADJ, candidate_fn=self._all_others)
        self.assertEqual(s1["probed"], 2)
        self.assertEqual(s1["deferred"], 2)
        s2 = memsom_contradict.sweep(self.conn, adjudicate=WAF_ADJ,
                                     candidate_fn=self._all_others)
        self.assertEqual(s2["probed"], 2)
        self.assertEqual(s2["deferred"], 0)


class TestIngestHook(Base):
    def test_off_by_default(self):
        self.assertFalse(memsom_contradict.enabled())
        memsom_ingest.ingest_text(self.conn, "Sucuri is the active WAF.", "user", source_ref="a")
        memsom_ingest.ingest_text(self.conn, "Cloudflare is the active WAF.", "user", source_ref="b")
        self.assertEqual(len(memsom_contradict.list_contradictions(self.conn)), 0)

    def test_runs_when_enabled(self):
        os.environ["MEMDAG_CONTRADICT"] = "1"
        # inject the adjudicator via _default_adjudicator so the hook picks it up
        with unittest.mock.patch.object(memsom_contradict, "_default_adjudicator",
                                        return_value=WAF_ADJ):
            memsom_ingest.ingest_text(self.conn, "Sucuri is the active WAF.", "user", source_ref="a")
            memsom_ingest.ingest_text(self.conn, "Cloudflare is the active WAF.", "user", source_ref="b")
        rows = memsom_contradict.list_contradictions(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["enforced"])   # observe by default


if __name__ == "__main__":
    unittest.main()
