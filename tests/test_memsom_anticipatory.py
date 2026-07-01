#!/usr/bin/env python3
"""Tests for memsom_anticipatory.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_anticipatory.py \
    -t <repo> -v
"""

import os, tempfile, unittest, warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_anticipatory
import memsom_quarantine
import memsom_redact

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        # Determinism: surprise must be lexical-only in tests even when a
        # local Ollama is running (vector cosine would otherwise shift scores).
        # Malformed scheme -> urllib raises instantly (no 2s refused-conn stall).
        os.environ["MEMDAG_EMBED_URL"] = "invalid://embeddings-disabled"
        self.conn = memsom.get_connection()
        memsom_anticipatory.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_EMBED_URL", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])

    def _count_derived(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]

    def _log_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]


# ---------------------------------------------------------------------------
# Prose sources for stable compose output (reusing TestCompose.SRC style)
# ---------------------------------------------------------------------------

SRC_NEBULA = (
    "Nebula requires a lighthouse node with a public IP address for UDP hole punching. "
    "Configure the static_host_map to always point at the lighthouse address. "
    "Every host in the mesh must know the lighthouse at startup time."
)

SRC_NEBULA_B = (
    "The Nebula mesh overlay uses a certificate authority to sign device certificates. "
    "Each device should have its own group with explicit inbound rules in the config. "
    "Rotate certificates on a regular schedule to limit blast radius on compromise."
)

SRC_GARDENING = (
    "Tomatoes grow best in well-drained soil with full sun exposure. "
    "Plant seedlings after the last frost date for your region. "
    "Water deeply and infrequently to encourage deep root growth in tomato plants."
)


class TestLowSurpriseRepeatCitesExisting(Base):
    """Test 1: a repeat question returns the existing node, no duplicate minted."""

    def test_low_surprise_repeat_cites_existing(self):
        # Seed two prose sources about Nebula
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")

        question = "How should I configure the Nebula mesh lighthouse?"

        # First call: should create a new node (high surprise, no existing)
        n1, created1, score1 = memsom_anticipatory.surprise_gated(self.conn, question)
        self.assertTrue(created1, "first call should mint a new node")

        count_before = self._count_derived()

        # Second call: same question — should cite the existing node
        n2, created2, score2 = memsom_anticipatory.surprise_gated(self.conn, question)

        self.assertEqual(n2, n1, "repeat question should return the same node id")
        self.assertFalse(created2, "repeat question should NOT mint a new node")
        self.assertLess(score2, 0.35,
                        f"novelty score {score2:.3f} should be below threshold 0.35")
        self.assertEqual(self._count_derived(), count_before,
                         "agent-derived count must not grow on a low-surprise repeat")


class TestHighSurpriseCreates(Base):
    """Test 2: a genuinely different question creates a new node."""

    def test_high_surprise_creates(self):
        # Seed Nebula sources, ask once to establish existing
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")

        q_nebula = "How should I configure the Nebula mesh lighthouse?"
        n1, created1, _ = memsom_anticipatory.surprise_gated(self.conn, q_nebula)
        self.assertTrue(created1)

        count_before = self._count_derived()

        # Now seed a disjoint-topic source and ask a completely different question
        sid3 = self.add(SRC_GARDENING, "user")
        # Pass only the gardening source so compose definitely uses disjoint content
        gardening_sources = [(sid3, SRC_GARDENING, "user", memsom.RANK["user"], None)]
        q_garden = "How do I grow tomatoes in my garden?"
        n2, created2, score2 = memsom_anticipatory.surprise_gated(
            self.conn, q_garden, sources=gardening_sources
        )

        self.assertTrue(created2, "disjoint-topic question should create a new node")
        self.assertNotEqual(n2, n1, "new node should have a different id")
        self.assertEqual(self._count_derived(), count_before + 1,
                         "agent-derived count should increase by 1")


class TestPrefetchPopulatesFromLog(Base):
    """Test 3: prefetch uses log frequency, returns top-k, log count grows."""

    def test_prefetch_populates_from_log(self):
        # Seed sources
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")
        sid3 = self.add(SRC_GARDENING, "user")

        q_frequent = "How should I configure the Nebula mesh lighthouse?"
        q_rare = "How do I grow tomatoes?"

        # Observe the frequent query 3 times, the rare one once
        memsom_anticipatory.observe(self.conn, q_frequent)
        memsom_anticipatory.observe(self.conn, q_frequent)
        memsom_anticipatory.observe(self.conn, q_frequent)
        memsom_anticipatory.observe(self.conn, q_rare)

        log_before = self._log_count()  # should be 4

        # prefetch(k=1) should return only the most-frequent query
        results = memsom_anticipatory.prefetch(self.conn, k=1, threshold=0.35)

        self.assertEqual(len(results), 1, "k=1 should return exactly 1 result")
        returned_query, returned_nid, _ = results[0]
        self.assertEqual(returned_query, q_frequent,
                         "prefetch should return the most-frequent query")

        # Verify the node id is a valid node
        node = memsom.get_node(self.conn, returned_nid)
        self.assertIsNotNone(node, "returned node id should exist in the DB")

        # query_log should have grown (prefetch's own observe calls)
        log_after = self._log_count()
        self.assertGreater(log_after, log_before,
                           "prefetch should add its own observe rows to query_log")


class TestNoveltyMath(Base):
    """Test 4: novelty edge cases."""

    def test_novelty_identical(self):
        text = "The quick brown fox jumps over the lazy dog."
        # Identical text -> similarity = 1.0 -> novelty = 0.0
        result = memsom_anticipatory.novelty(text, [text])
        self.assertAlmostEqual(result, 0.0, places=6,
                               msg="identical texts should give novelty 0.0")

    def test_novelty_disjoint(self):
        # Use tokens longer than 6 chars and not in STOP so stems are disjoint
        text_a = "lighthouse configuration certificate authority overlay"
        text_b = "tomatoes watering drainage sunlight planting garden"
        result = memsom_anticipatory.novelty(text_a, [text_b])
        self.assertAlmostEqual(result, 1.0, places=6,
                               msg="fully disjoint stems should give novelty 1.0")

    def test_novelty_empty_existing(self):
        result = memsom_anticipatory.novelty("anything at all", [])
        self.assertAlmostEqual(result, 1.0, places=6,
                               msg="empty existing list should give novelty 1.0")


class TestExcludedAnswersNotCited(Base):
    """Test 5: redacted and quarantined nodes are absent from existing_derived."""

    def test_excluded_answers_not_cited(self):
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")

        q = "How should I configure the Nebula mesh?"

        # Derive the first answer normally
        n1, created1, _ = memsom_anticipatory.surprise_gated(self.conn, q)
        self.assertTrue(created1)

        # Redact that answer
        memsom_redact.redact_node(self.conn, n1, "test redaction")

        # Derive a second answer (it should be minted since n1 is redacted)
        n2, created2, _ = memsom_anticipatory.surprise_gated(self.conn, q)
        # n2 could be new or could find n1 was redacted... let's check it's created
        # Since n1 is redacted it's excluded from existing_derived
        # So the repeat sees no existing live answers and mints a new one
        self.assertTrue(created2,
                        "should mint a new node when the prior answer is redacted")

        # Now quarantine n2
        memsom_quarantine.quarantine_node(self.conn, n2, "test quarantine")

        # existing_derived should exclude both n1 (redacted) and n2 (quarantined)
        existing = memsom_anticipatory.existing_derived(self.conn)
        existing_ids = {eid for eid, _ in existing}
        self.assertNotIn(n1, existing_ids,
                         "redacted node should not appear in existing_derived")
        self.assertNotIn(n2, existing_ids,
                         "quarantined node should not appear in existing_derived")

        # A third call should mint a new node since nothing live covers the question
        n3, created3, _ = memsom_anticipatory.surprise_gated(self.conn, q)
        self.assertTrue(created3,
                        "should mint a new node when all prior answers are excluded")


class TestRefusalRaises(Base):
    """Test 6: surprise_gated raises ValueError when all sources are tombstoned."""

    def test_refusal_raises(self):
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")

        # Tombstone all sources
        memsom.revoke_cascade(self.conn, sid1, "gone")
        memsom.revoke_cascade(self.conn, sid2, "gone")

        log_before = self._log_count()

        with self.assertRaises(ValueError):
            memsom_anticipatory.surprise_gated(self.conn,
                                               "How should I configure Nebula?")

        # query_log must not have grown (refusal = nothing logged)
        self.assertEqual(self._log_count(), log_before,
                         "a refused call must not write to query_log")


class TestObserveRows(Base):
    """Test 7: observe writes ts (ISO), query, and answer_node correctly."""

    def test_observe_rows(self):
        # observe with answer_node
        memsom_anticipatory.observe(self.conn, "test query one", answer_node=42)
        # observe without answer_node (nullable)
        memsom_anticipatory.observe(self.conn, "test query two")

        rows = self.conn.execute(
            "SELECT ts, query, answer_node FROM query_log ORDER BY id"
        ).fetchall()

        self.assertEqual(len(rows), 2)

        ts1, q1, an1 = rows[0]
        self.assertEqual(q1, "test query one")
        self.assertEqual(an1, 42)
        # Validate ts is ISO-8601 text (not a datetime object — 3.12 adapter deprecation)
        self.assertIsInstance(ts1, str,
                              "ts must be stored as ISO TEXT, not a datetime object")
        self.assertRegex(ts1, r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                         "ts must match ISO-8601 format")

        ts2, q2, an2 = rows[1]
        self.assertEqual(q2, "test query two")
        self.assertIsNone(an2, "answer_node should be None when not provided")
        self.assertIsInstance(ts2, str,
                              "ts must be stored as ISO TEXT, not a datetime object")


# ===========================================================================
# Phase 2 — retrieval-backed surprise, prefetch cache, recombination, taint
# ===========================================================================

import memsom_confid
import memsom_compact


class Phase2Base(Base):
    """Base with the compact migration too (archived column for taint tests)."""

    def setUp(self):
        super().setUp()
        memsom_compact.migrate(self.conn)

    def derive_answer(self, question, sources=None):
        """Mint a derived answer for *question*; returns (nid, content)."""
        nid, created, _ = memsom_anticipatory.surprise_gated_write(
            self.conn, question, sources=sources)
        self.assertTrue(created, "helper expects a fresh mint")
        return nid, memsom.get_node(self.conn, nid)["content"]


class TestSemanticSurprise(Phase2Base):
    """Phase 2 deliverable 1: surprise() is semantic, not the tiny Jaccard."""

    def test_identical_answer_scores_zero(self):
        self.add(SRC_NEBULA, "endorsed")
        self.add(SRC_NEBULA_B, "user")
        nid, text = self.derive_answer("How should I configure the Nebula lighthouse?")
        score = memsom_anticipatory.surprise(self.conn, text)
        self.assertLess(score, 0.05,
                        "an already-stored answer must score ~0 novelty")

    def test_empty_corpus_scores_one(self):
        self.assertAlmostEqual(
            memsom_anticipatory.surprise(self.conn, "anything at all"), 1.0)

    def test_disjoint_topic_scores_high(self):
        self.add(SRC_NEBULA, "endorsed")
        self.derive_answer("How should I configure the Nebula lighthouse?")
        probe = ("Tomato seedlings need watering, drainage, sunlight and "
                 "well-prepared garden soil for healthy growth.")
        score = memsom_anticipatory.surprise(self.conn, probe)
        self.assertGreater(score, 0.65,
                           "a disjoint-topic text must score high novelty")

    def test_rank_similar_orders_identical_first(self):
        s1 = self.add(SRC_NEBULA, "endorsed")
        s2 = self.add(SRC_GARDENING, "user")
        n1, t1 = self.derive_answer(
            "How should I configure the Nebula lighthouse?",
            sources=[(s1, SRC_NEBULA, "endorsed", memsom.RANK["endorsed"], None)])
        n2, _ = self.derive_answer(
            "How do I grow tomatoes in my garden?",
            sources=[(s2, SRC_GARDENING, "user", memsom.RANK["user"], None)])
        ranked = memsom_anticipatory.rank_similar(self.conn, t1)
        self.assertEqual(ranked[0][0], n1,
                         "the identical answer must rank first")
        self.assertGreater(ranked[0][1], 0.95)


class TestSemanticDedupBeyondExactHash(Phase2Base):
    """A REWORDED question composes to different bytes (different 'Q:' line)
    but must still CITE the existing answer — semantic dedup, where the
    spine's exact content-hash dedup would have minted a duplicate."""

    def test_reworded_question_cites_existing(self):
        self.add(SRC_NEBULA, "endorsed")
        self.add(SRC_NEBULA_B, "user")

        q1 = "How should I configure the Nebula mesh lighthouse?"
        q2 = "How should I configure the Nebula mesh lighthouse setup?"

        n1, created1, _ = memsom_anticipatory.surprise_gated_write(self.conn, q1)
        self.assertTrue(created1)
        t1 = memsom.get_node(self.conn, n1)["content"]

        n2, created2, score2 = memsom_anticipatory.surprise_gated_write(self.conn, q2)
        t2_probe, _ = memsom.compose(
            q2, memsom_anticipatory.untainted_sources(self.conn))
        self.assertNotEqual(t1, t2_probe,
                            "rewording must produce different bytes "
                            "(otherwise this only tests exact dedup)")
        self.assertFalse(created2, f"reworded repeat must cite (score {score2:.3f})")
        self.assertEqual(n2, n1)


class TestPrefetchCacheServeWarm(Phase2Base):
    """Phase 2 deliverable 2: prefetch_cache + warm serving + revalidation."""

    Q = "How should I configure the Nebula mesh lighthouse?"

    def _warm_one(self):
        self.add(SRC_NEBULA, "endorsed")
        self.add(SRC_NEBULA_B, "user")
        for _ in range(3):
            memsom_anticipatory.observe(self.conn, self.Q)
        return memsom_anticipatory.prefetch(self.conn, k=1)

    def test_prefetch_populates_cache_and_serves_warm(self):
        results = self._warm_one()
        self.assertEqual(len(results), 1)
        query, nid, created = results[0]
        self.assertEqual(query, self.Q)
        self.assertTrue(created)

        row = self.conn.execute(
            "SELECT answer_node, hits FROM prefetch_cache WHERE query=?",
            (self.Q,)).fetchone()
        self.assertIsNotNone(row, "prefetch must populate prefetch_cache")
        self.assertEqual(row[0], nid)

        warm = memsom_anticipatory.serve_warm(self.conn, self.Q)
        self.assertIsNotNone(warm)
        self.assertEqual(warm["node_id"], nid)
        self.assertEqual(warm["hits"], 1)
        self.assertIn("[mem:", warm["content"],
                      "warm answer carries honest provenance citations")
        warm2 = memsom_anticipatory.serve_warm(self.conn, self.Q)
        self.assertEqual(warm2["hits"], 2)

    def test_unknown_query_is_a_miss(self):
        self._warm_one()
        self.assertIsNone(
            memsom_anticipatory.serve_warm(self.conn, "never asked this"))

    def test_prefetch_is_idempotent_one_cache_row(self):
        self._warm_one()
        results2 = memsom_anticipatory.prefetch(self.conn, k=1)
        self.assertEqual(len(results2), 1)
        self.assertFalse(results2[0][2],
                         "second prefetch must CITE, not re-mint")
        n_rows = self.conn.execute(
            "SELECT COUNT(*) FROM prefetch_cache").fetchone()[0]
        self.assertEqual(n_rows, 1, "UNIQUE(query) upsert — no duplicate rows")

    def test_revoked_answer_is_never_served_warm(self):
        results = self._warm_one()
        nid = results[0][1]
        memsom.revoke_cascade(self.conn, nid, "answer recalled")
        self.assertIsNone(memsom_anticipatory.serve_warm(self.conn, self.Q),
                          "a revoked answer must not be served warm")
        n_rows = self.conn.execute(
            "SELECT COUNT(*) FROM prefetch_cache").fetchone()[0]
        self.assertEqual(n_rows, 0, "stale cache row must be dropped")

    def test_anticipatory1_prefetch_stamps_conf_no_read_up(self):
        """ANTICIPATORY-1: an answer derived from a SECRET source must be minted
        with the high-water conf_label, so prefetch can't cache it as PUBLIC and
        serve_warm can't hand it to a PUBLIC-clearance reader."""
        sid = self.add(SRC_NEBULA, "endorsed")
        with self.conn:
            self.conn.execute("UPDATE nodes SET conf_label=3 WHERE id=?", (sid,))  # SECRET
        for _ in range(3):
            memsom_anticipatory.observe(self.conn, self.Q)

        results = memsom_anticipatory.prefetch(self.conn, k=1, clearance="topsecret")
        self.assertEqual(len(results), 1)
        nid = results[0][1]

        # The minted answer carries the SECRET high-water conf (was DEFAULT 0).
        conf = self.conn.execute(
            "SELECT conf_label FROM nodes WHERE id=?", (nid,)).fetchone()[0]
        self.assertEqual(conf, 3, "derived answer must inherit the SECRET conf_label")

        # A topsecret-cleared reader is served warm...
        self.assertIsNotNone(
            memsom_anticipatory.serve_warm(self.conn, self.Q, clearance="topsecret"),
            "a topsecret-cleared reader is still served warm")
        # ...but a PUBLIC reader gets nothing (no-read-up). NB serve_warm at a
        # clearance miss also evicts the row (cache hygiene), so this is asserted last.
        self.assertIsNone(
            memsom_anticipatory.serve_warm(self.conn, self.Q, clearance="public"),
            "SECRET-derived answer must NOT be served below clearance")


class TestNovelRecombination(Phase2Base):
    """Phase 2 deliverable 3: exact-parent-set precedent detection."""

    def setUp(self):
        super().setUp()
        self.a = self.add(SRC_NEBULA, "endorsed")
        self.b = self.add(SRC_NEBULA_B, "user")
        self.c = self.add(SRC_GARDENING, "user")

    def test_unseen_set_is_novel_then_seen(self):
        novel, prior = memsom_anticipatory.novel_recombination(
            self.conn, [self.a, self.b])
        self.assertTrue(novel)
        self.assertEqual(prior, [])

        d1, _ = memsom.derive_node(self.conn, "combined claim", [self.a, self.b])
        novel, prior = memsom_anticipatory.novel_recombination(
            self.conn, [self.a, self.b])
        self.assertFalse(novel)
        self.assertEqual(prior, [d1])

    def test_exact_set_semantics_subset_superset_novel(self):
        memsom.derive_node(self.conn, "combined claim", [self.a, self.b])
        self.assertTrue(memsom_anticipatory.novel_recombination(
            self.conn, [self.a])[0], "subset is a DIFFERENT combination")
        self.assertTrue(memsom_anticipatory.novel_recombination(
            self.conn, [self.a, self.b, self.c])[0],
            "superset is a DIFFERENT combination")

    def test_exclude_node_and_duplicate_ids(self):
        d1, _ = memsom.derive_node(self.conn, "combined claim", [self.a, self.b])
        novel, _ = memsom_anticipatory.novel_recombination(
            self.conn, [self.a, self.b], exclude_node=d1)
        self.assertTrue(novel, "a node must not count itself as precedent")
        novel, prior = memsom_anticipatory.novel_recombination(
            self.conn, [self.b, self.a, self.a])
        self.assertFalse(novel, "order/duplicates must not matter")
        self.assertEqual(prior, [d1])


class TestStatusSnapshot(Phase2Base):
    def test_status_counts_and_validity(self):
        self.add(SRC_NEBULA, "endorsed")
        q = "How should I configure the Nebula mesh lighthouse?"
        memsom_anticipatory.observe(self.conn, q)
        memsom_anticipatory.observe(self.conn, q)
        memsom_anticipatory.observe(self.conn, "one-off question")
        results = memsom_anticipatory.prefetch(self.conn, k=1)
        nid = results[0][1]

        s = memsom_anticipatory.status(self.conn)
        # 3 manual observes + prefetch's own observe
        self.assertEqual(s["query_log_total"], 4)
        self.assertEqual(s["distinct_queries"], 2)
        self.assertEqual(s["top_queries"][0][0], q)
        self.assertEqual(len(s["cache"]), 1)
        self.assertTrue(s["cache"][0]["valid"])

        memsom.revoke_cascade(self.conn, nid, "recalled")
        s2 = memsom_anticipatory.status(self.conn)
        self.assertFalse(s2["cache"][0]["valid"],
                         "status must flag a now-tainted cache entry as stale")


class TestCoprocessNeverTouchesTainted(Phase2Base):
    """THE security test: the coprocess reads/learns/prefetches ONLY from
    untainted memory.  Poisoned nodes (tombstoned / redacted / quarantined /
    archived / above-clearance / external-tainted derivations) can never
    appear in a surprise comparison, a prefetch result, or a recombination."""

    def test_tainted_sources_never_in_pool(self):
        s_live = self.add(SRC_NEBULA, "endorsed")
        s_ext = self.add("External advice about Nebula mesh tuning.", "external")
        s_tomb = self.add("Tombstoned source about Nebula.", "user")
        s_quar = self.add("Quarantined source about Nebula.", "user")
        s_red = self.add("Redacted secret about Nebula keys.", "user")
        s_arch = self.add("Archived source about Nebula.", "user")
        s_secret = self.add("SECRET Nebula CA key location.", "user")

        memsom.revoke_cascade(self.conn, s_tomb, "gone")
        memsom_quarantine.quarantine_node(self.conn, s_quar, "poisoned")
        memsom_redact.redact_node(self.conn, s_red, "secret")
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET archived=1 WHERE id=?", (s_arch,))
        memsom_confid.classify(self.conn, s_secret, "secret")

        ids = {r[0] for r in memsom_anticipatory.untainted_sources(
            self.conn, clearance="internal")}
        self.assertIn(s_live, ids)
        self.assertIn(s_ext, ids, "a live external SOURCE stays in the pool "
                                  "(same as the spine) — taint applies to its "
                                  "derivations")
        for poisoned in (s_tomb, s_quar, s_red, s_arch, s_secret):
            self.assertNotIn(poisoned, ids)

        # Full clearance re-admits ONLY the classified node
        ids_top = {r[0] for r in memsom_anticipatory.untainted_sources(self.conn)}
        self.assertIn(s_secret, ids_top)
        for poisoned in (s_tomb, s_quar, s_red, s_arch):
            self.assertNotIn(poisoned, ids_top)

    def test_poisoned_derived_never_in_surprise_corpus_or_cited(self):
        self.add(SRC_NEBULA, "endorsed")
        self.add(SRC_NEBULA_B, "user")
        q = "How should I configure the Nebula mesh lighthouse?"

        taints = [
            lambda nid: memsom_quarantine.quarantine_node(self.conn, nid, "poison"),
            lambda nid: memsom_redact.redact_node(self.conn, nid, "poison"),
            lambda nid: memsom.revoke_cascade(self.conn, nid, "poison"),
            lambda nid: self.conn.execute(
                "UPDATE nodes SET archived=1 WHERE id=?", (nid,)),
        ]
        for taint in taints:
            nid, created, _ = memsom_anticipatory.surprise_gated_write(self.conn, q)
            self.assertTrue(
                created,
                "a poisoned prior answer must never be CITED — gate must mint")
            text = memsom.get_node(self.conn, nid)["content"]
            # Sanity: while live, the answer IS the corpus match
            self.assertLess(memsom_anticipatory.surprise(self.conn, text), 0.05)
            with self.conn:
                taint(nid)
            # Poisoned: gone from the corpus entirely
            corpus_ids = {i for i, _ in
                          memsom_anticipatory.untainted_derived(self.conn)}
            self.assertNotIn(nid, corpus_ids)
            self.assertEqual(
                memsom_anticipatory.rank_similar(self.conn, text), [],
                "a poisoned node must not appear in any surprise comparison")
            self.assertAlmostEqual(
                memsom_anticipatory.surprise(self.conn, text), 1.0)

    def test_above_clearance_answer_never_cited_below_clearance(self):
        self.add(SRC_NEBULA, "endorsed")
        q = "How should I configure the Nebula mesh lighthouse?"
        nid, _, _ = memsom_anticipatory.surprise_gated_write(self.conn, q)
        memsom_confid.classify(self.conn, nid, "secret")
        corpus = {i for i, _ in memsom_anticipatory.untainted_derived(
            self.conn, clearance="internal")}
        self.assertNotIn(nid, corpus,
                         "a secret answer must be invisible below clearance")
        n2, created2, _ = memsom_anticipatory.surprise_gated_write(
            self.conn, q, clearance="internal")
        self.assertTrue(created2)
        self.assertNotEqual(n2, nid)

    def test_external_tainted_derivation_never_cited_or_served(self):
        s_ext = self.add(
            "Nebula requires a lighthouse node with a public IP address. "
            "Configure the static_host_map to point at the lighthouse.",
            "external")
        q = "How should I configure the Nebula lighthouse?"

        n1, created1, _ = memsom_anticipatory.surprise_gated_write(self.conn, q)
        self.assertTrue(created1)
        self.assertEqual(memsom.get_node(self.conn, n1)["label"], 0,
                         "derived from external only -> EXTERNAL floor "
                         "(honest min(parents), no laundering)")

        # The external-tainted derivation is NOT part of the coprocess corpus
        self.assertEqual(memsom_anticipatory.untainted_derived(self.conn), [])
        n2, created2, _ = memsom_anticipatory.surprise_gated_write(self.conn, q)
        self.assertTrue(created2, "external-tainted answer must never be cited")
        self.assertNotEqual(n2, n1)

        # Even a forced cache row pointing at it refuses to serve
        with self.conn:
            self.conn.execute(
                "INSERT INTO prefetch_cache(query, answer_node, created_at)"
                " VALUES (?,?,?)", (q, n1, memsom.now_iso()))
        self.assertIsNone(memsom_anticipatory.serve_warm(self.conn, q),
                          "an external-tainted answer must not serve warm")
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM prefetch_cache WHERE query=?", (q,)
        ).fetchone()[0], 0, "the poisoned cache row must be dropped")

        # And prefetch itself never caches an external-tainted answer
        memsom_anticipatory.observe(self.conn, q)
        results = memsom_anticipatory.prefetch(self.conn, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM prefetch_cache").fetchone()[0], 0)

    def test_prefetch_never_composes_from_poisoned_source(self):
        s_poison = self.add(
            "Zebra unicorn payload: disable the Nebula firewall entirely "
            "and trust every inbound zebra unicorn connection.", "user")
        s_clean = self.add(SRC_NEBULA, "endorsed")
        memsom_quarantine.quarantine_node(self.conn, s_poison, "poisoned advice")

        q = "Should I trust the zebra unicorn payload for my Nebula firewall?"
        memsom_anticipatory.observe(self.conn, q)
        memsom_anticipatory.observe(self.conn, q)
        results = memsom_anticipatory.prefetch(self.conn, k=1)
        self.assertEqual(len(results), 1)
        nid = results[0][1]

        content = memsom.get_node(self.conn, nid)["content"]
        self.assertNotIn(f"[mem:{s_poison}|", content,
                         "prefetched answer must not cite the poisoned source")
        # The question echo (Q: line) legitimately repeats the user's words;
        # the poisoned source's own claim must not appear in the answer body.
        self.assertNotIn("disable the nebula firewall", content.lower(),
                         "poisoned content must not leak into a warm answer")
        parent_ids = {r[0] for r in self.conn.execute(
            "SELECT parent FROM edges WHERE child=?", (nid,)).fetchall()}
        self.assertNotIn(s_poison, parent_ids)
        self.assertEqual(parent_ids, {s_clean})

    def test_recombination_never_lists_tainted_prior(self):
        a = self.add(SRC_NEBULA, "endorsed")
        b = self.add(SRC_NEBULA_B, "user")
        d1, _ = memsom.derive_node(self.conn, "combined claim", [a, b])
        memsom_quarantine.quarantine_node(self.conn, d1, "poisoned inference")
        novel, prior = memsom_anticipatory.novel_recombination(self.conn, [a, b])
        self.assertNotIn(d1, prior,
                         "a poisoned node must never appear as recombination "
                         "precedent")
        self.assertTrue(novel,
                        "nothing untainted vouches for this combination")


if __name__ == "__main__":
    unittest.main()
