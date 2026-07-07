#!/usr/bin/env python3
"""Tests for memsom_contradict — the cross-source contradiction staleness trigger.

Run:
  python -W error::DeprecationWarning -m unittest discover \
      -s <repo> -p test_memsom_contradict.py -t <repo> -v

No embedder / NLI weights needed: candidates are injected, and the NLI tier is a
deterministic stub. The structured tier exercises the real extract_claim path.
"""

import os
import tempfile
import unittest
import unittest.mock
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_ingest
import memsom_stale
import memsom_contradict


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_ingest.migrate(self.conn)
        memsom_stale.migrate(self.conn)      # ensure the stale columns exist for negative cases
        memsom_contradict.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_CONTRADICT", None)
        self.tmp.cleanup()

    def add(self, content, channel="user", source_ref=None):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, source_ref=source_ref)

    def state(self, nid):
        return self.conn.execute(
            "SELECT stale, stale_reason FROM nodes WHERE id=?", (nid,)).fetchone()


class TestStructuredTier(Base):
    def test_structured_conflict_marks_old_stale(self):
        # same key, different value (topical scoping is provided here by injecting
        # the candidate — in prod the retrieval gate does it).
        old = self.add("waf=sucuri", source_ref="memory:acme_waf")
        new = self.add("waf=cloudflare", source_ref="session:2026-07-07")
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "waf=sucuri")])
        self.assertEqual(marked, [(old, "structured")])
        stale, reason = self.state(old)
        self.assertEqual(stale, 1)
        self.assertTrue(reason.startswith(memsom_contradict.REASON_PREFIX))
        rows = memsom_contradict.list_contradictions(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "structured")

    def test_directionality_new_never_staled(self):
        old = self.add("port=8080", source_ref="a")
        new = self.add("port=9090", source_ref="b")
        memsom_contradict.detect(self.conn, new, candidates=[(old, "port=8080")])
        self.assertEqual(self.state(old)[0], 1)   # old staled
        self.assertEqual(self.state(new)[0], 0)   # new untouched

    def test_same_value_is_not_a_contradiction(self):
        old = self.add("waf=sucuri", source_ref="a")
        new = self.add("waf=sucuri", source_ref="b")
        marked = memsom_contradict.detect(self.conn, new, candidates=[(old, "waf=sucuri")])
        self.assertEqual(marked, [])
        self.assertEqual(self.state(old)[0], 0)

    def test_never_writes_source_supersedes(self):
        # regression against the landmine: a contradiction must NOT create a
        # source_supersedes link (freshen/substitute_fresh would misfire on it).
        import memsom_stale
        memsom_stale.migrate(self.conn)
        old = self.add("waf=sucuri", source_ref="a")
        new = self.add("waf=cloudflare", source_ref="b")
        memsom_contradict.detect(self.conn, new, candidates=[(old, "waf=sucuri")])
        n = self.conn.execute("SELECT COUNT(*) FROM source_supersedes").fetchone()[0]
        self.assertEqual(n, 0)

    def test_idempotent_no_double_record(self):
        old = self.add("waf=sucuri", source_ref="a")
        new = self.add("waf=cloudflare", source_ref="b")
        memsom_contradict.detect(self.conn, new, candidates=[(old, "waf=sucuri")])
        memsom_contradict.detect(self.conn, new, candidates=[(old, "waf=sucuri")])
        self.assertEqual(len(memsom_contradict.list_contradictions(self.conn)), 1)


class TestNliTier(Base):
    def test_nli_stub_marks_prose_contradiction(self):
        # prose the structured tier can't parse (extract_claim -> None); the NLI
        # stub confirms the contradiction.
        old = self.add("Sucuri is the active WAF for the corporate apex.", source_ref="a")
        new = self.add("Cloudflare is the active WAF for the corporate apex.", source_ref="b")
        nli = lambda premise, hypothesis: 0.95
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF for the corporate apex.")],
            nli=nli)
        self.assertEqual(marked, [(old, "nli")])
        row = memsom_contradict.list_contradictions(self.conn)[0]
        self.assertEqual(row["verdict"], "nli")
        self.assertAlmostEqual(row["score"], 0.95)

    def test_nli_below_threshold_ignored(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("The mesh uses UDP port on every host.", source_ref="b")
        nli = lambda p, h: 0.30
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")], nli=nli)
        self.assertEqual(marked, [])

    def test_nli_absent_falls_back_to_structured_only(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        # no nli passed -> structured tier returns None on prose -> nothing marked
        marked = memsom_contradict.detect(
            self.conn, new, candidates=[(old, "Sucuri is the active WAF.")])
        self.assertEqual(marked, [])


class TestGuards(Base):
    def test_agent_derived_new_node_skipped(self):
        old = self.add("waf=sucuri", source_ref="a")
        new = self.add("waf=cloudflare", channel="agent-derived")
        marked = memsom_contradict.detect(self.conn, new, candidates=[(old, "waf=sucuri")])
        self.assertEqual(marked, [])

    def test_reason_namespace_survives_verify_stale(self):
        # a contradiction flag must NOT be cleared by the verify_stale sweep
        # (which only clears its own "unverified since"/"overdue" reasons).
        import memsom_verify_stale
        old = self.add("waf=sucuri", source_ref="memory:acme_waf")
        new = self.add("waf=cloudflare", source_ref="b")
        memsom_contradict.detect(self.conn, new, candidates=[(old, "waf=sucuri")])
        self.assertEqual(self.state(old)[0], 1)
        memsom_verify_stale.recompute_verify_stale(self.conn)
        self.assertEqual(self.state(old)[0], 1)   # still stale — not our namespace to clear

    def test_ingest_hook_off_by_default(self):
        # with the env unset, ingest must not run the detector at all.
        self.assertFalse(memsom_contradict.enabled())
        memsom_ingest.ingest_text(self.conn, "waf=sucuri", "user", source_ref="a")
        memsom_ingest.ingest_text(self.conn, "waf=cloudflare", "user", source_ref="b")
        self.assertEqual(len(memsom_contradict.list_contradictions(self.conn)), 0)


class TestNliTierWiring(Base):
    """Phase 2 plumbing — no model download: the real scorer is monkeypatched, and
    the availability probe is forced. Proves detect() resolves the default NLI
    scorer from the env and that the ingest hook drives it end-to-end."""

    def test_default_nli_off_unless_opted_in(self):
        # detector on, but semantic tier NOT opted in -> _default_nli() is None
        self.assertIsNone(memsom_contradict._default_nli())

    def test_default_nli_none_when_unavailable(self):
        os.environ["MEMDAG_CONTRADICT_NLI"] = "1"
        try:
            with unittest.mock.patch.object(memsom_contradict, "nli_available",
                                            return_value=False):
                self.assertIsNone(memsom_contradict._default_nli())
        finally:
            os.environ.pop("MEMDAG_CONTRADICT_NLI", None)

    def test_default_nli_used_when_opted_in_and_available(self):
        os.environ["MEMDAG_CONTRADICT_NLI"] = "1"
        try:
            with unittest.mock.patch.object(memsom_contradict, "nli_available",
                                            return_value=True):
                fn = memsom_contradict._default_nli()
                self.assertIs(fn, memsom_contradict.nli_score)
        finally:
            os.environ.pop("MEMDAG_CONTRADICT_NLI", None)

    def test_detect_uses_resolved_default_nli_on_prose(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        os.environ["MEMDAG_CONTRADICT_NLI"] = "1"
        try:
            with unittest.mock.patch.object(memsom_contradict, "nli_available",
                                            return_value=True), \
                 unittest.mock.patch.object(memsom_contradict, "nli_score",
                                            return_value=0.97):
                marked = memsom_contradict.detect(
                    self.conn, new, candidates=[(old, "Sucuri is the active WAF.")])
            self.assertEqual(marked, [(old, "nli")])
        finally:
            os.environ.pop("MEMDAG_CONTRADICT_NLI", None)

    def test_threshold_env_respected(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        os.environ["MEMDAG_CONTRADICT_NLI_THRESHOLD"] = "0.99"
        try:
            # score 0.90 is below the raised 0.99 bar -> not flagged
            marked = memsom_contradict.detect(
                self.conn, new, candidates=[(old, "Sucuri is the active WAF.")],
                nli=lambda p, h: 0.90)
            self.assertEqual(marked, [])
        finally:
            os.environ.pop("MEMDAG_CONTRADICT_NLI_THRESHOLD", None)

    def test_ingest_hook_runs_nli_when_both_flags_on(self):
        os.environ["MEMDAG_CONTRADICT"] = "1"
        os.environ["MEMDAG_CONTRADICT_NLI"] = "1"
        try:
            with unittest.mock.patch.object(memsom_contradict, "nli_available",
                                            return_value=True), \
                 unittest.mock.patch.object(memsom_contradict, "nli_score",
                                            return_value=0.96):
                memsom_ingest.ingest_text(self.conn, "Sucuri is the active WAF.",
                                          "user", source_ref="a")
                memsom_ingest.ingest_text(self.conn, "Cloudflare is the active WAF.",
                                          "user", source_ref="b")
            rows = memsom_contradict.list_contradictions(self.conn)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["verdict"], "nli")
        finally:
            os.environ.pop("MEMDAG_CONTRADICT", None)
            os.environ.pop("MEMDAG_CONTRADICT_NLI", None)


class TestSweep(Base):
    """Batch sweep — covers flat-file bridge memories the ingest hook doesn't. Uses
    an injected candidate_fn (all other live nodes) so no embedder/model is needed;
    the structured tier does the adjudication."""

    def _all_others(self, pid, _ptext):
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT id, content FROM nodes WHERE id != ? AND tombstoned = 0", (pid,))]

    def test_backfill_finds_preexisting_contradiction_older_loses(self):
        old = self.add("waf=sucuri", source_ref="memory:acme_waf")   # lower id = older
        new = self.add("waf=cloudflare", source_ref="session:x")     # higher id = newer
        stats = memsom_contradict.sweep(
            self.conn, backfill=True, use_nli=False, candidate_fn=self._all_others)
        self.assertEqual(stats["probed"], 2)          # both nodes probed
        self.assertEqual(stats["contradictions"], 1)  # but pair converges to ONE
        self.assertEqual(self.state(old)[0], 1)       # older loses
        self.assertEqual(self.state(new)[0], 0)       # newer survives
        rows = memsom_contradict.list_contradictions(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["old_id"], rows[0]["new_id"]), (old, new))

    def test_cursor_advances_and_incremental_skips(self):
        old = self.add("waf=sucuri", source_ref="a")
        new = self.add("waf=cloudflare", source_ref="b")
        memsom_contradict.sweep(self.conn, backfill=True, use_nli=False,
                                candidate_fn=self._all_others)
        # a fresh incremental sweep with nothing new probes zero nodes
        stats2 = memsom_contradict.sweep(self.conn, use_nli=False,
                                         candidate_fn=self._all_others)
        self.assertEqual(stats2["probed"], 0)
        # add a third node; only it is probed next time
        self.add("port=443", source_ref="c")
        stats3 = memsom_contradict.sweep(self.conn, use_nli=False,
                                         candidate_fn=self._all_others)
        self.assertEqual(stats3["probed"], 1)

    def test_limit_defers_and_resumes(self):
        for i in range(4):
            self.add(f"k{i}=v{i}", source_ref=str(i))
        s1 = memsom_contradict.sweep(self.conn, backfill=True, use_nli=False,
                                     limit=2, candidate_fn=self._all_others)
        self.assertEqual(s1["probed"], 2)
        self.assertEqual(s1["deferred"], 2)
        s2 = memsom_contradict.sweep(self.conn, use_nli=False,
                                     candidate_fn=self._all_others)
        self.assertEqual(s2["probed"], 2)     # resumes from the watermark
        self.assertEqual(s2["deferred"], 0)

    def test_sweep_idempotent(self):
        self.add("waf=sucuri", source_ref="a")
        self.add("waf=cloudflare", source_ref="b")
        memsom_contradict.sweep(self.conn, backfill=True, use_nli=False,
                                candidate_fn=self._all_others)
        memsom_contradict.sweep(self.conn, backfill=True, use_nli=False,
                                candidate_fn=self._all_others)
        self.assertEqual(len(memsom_contradict.list_contradictions(self.conn)), 1)

    def test_sweep_nli_tier_on_prose(self):
        old = self.add("Sucuri is the active WAF.", source_ref="a")
        new = self.add("Cloudflare is the active WAF.", source_ref="b")
        stats = memsom_contradict.sweep(
            self.conn, backfill=True, candidate_fn=self._all_others,
            nli=lambda p, h: 0.97)                     # inject the scorer
        self.assertEqual(stats["contradictions"], 1)
        self.assertEqual(self.state(old)[0], 1)
        self.assertEqual(memsom_contradict.list_contradictions(self.conn)[0]["verdict"], "nli")


if __name__ == "__main__":
    unittest.main()
