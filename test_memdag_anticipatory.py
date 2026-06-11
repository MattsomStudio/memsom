#!/usr/bin/env python3
"""Tests for memdag_anticipatory.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:\\Users\\you\\memdag -p test_memdag_anticipatory.py \
    -t C:\\Users\\you\\memdag -v
"""

import os, tempfile, unittest, warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_anticipatory
import memdag_quarantine
import memdag_redact

HERE = Path(__file__).resolve().parent


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_anticipatory.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])

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
        n1, created1, score1 = memdag_anticipatory.surprise_gated(self.conn, question)
        self.assertTrue(created1, "first call should mint a new node")

        count_before = self._count_derived()

        # Second call: same question — should cite the existing node
        n2, created2, score2 = memdag_anticipatory.surprise_gated(self.conn, question)

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
        n1, created1, _ = memdag_anticipatory.surprise_gated(self.conn, q_nebula)
        self.assertTrue(created1)

        count_before = self._count_derived()

        # Now seed a disjoint-topic source and ask a completely different question
        sid3 = self.add(SRC_GARDENING, "user")
        # Pass only the gardening source so compose definitely uses disjoint content
        gardening_sources = [(sid3, SRC_GARDENING, "user", memdag.RANK["user"], None)]
        q_garden = "How do I grow tomatoes in my garden?"
        n2, created2, score2 = memdag_anticipatory.surprise_gated(
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
        memdag_anticipatory.observe(self.conn, q_frequent)
        memdag_anticipatory.observe(self.conn, q_frequent)
        memdag_anticipatory.observe(self.conn, q_frequent)
        memdag_anticipatory.observe(self.conn, q_rare)

        log_before = self._log_count()  # should be 4

        # prefetch(k=1) should return only the most-frequent query
        results = memdag_anticipatory.prefetch(self.conn, k=1, threshold=0.35)

        self.assertEqual(len(results), 1, "k=1 should return exactly 1 result")
        returned_query, returned_nid, _ = results[0]
        self.assertEqual(returned_query, q_frequent,
                         "prefetch should return the most-frequent query")

        # Verify the node id is a valid node
        node = memdag.get_node(self.conn, returned_nid)
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
        result = memdag_anticipatory.novelty(text, [text])
        self.assertAlmostEqual(result, 0.0, places=6,
                               msg="identical texts should give novelty 0.0")

    def test_novelty_disjoint(self):
        # Use tokens longer than 6 chars and not in STOP so stems are disjoint
        text_a = "lighthouse configuration certificate authority overlay"
        text_b = "tomatoes watering drainage sunlight planting garden"
        result = memdag_anticipatory.novelty(text_a, [text_b])
        self.assertAlmostEqual(result, 1.0, places=6,
                               msg="fully disjoint stems should give novelty 1.0")

    def test_novelty_empty_existing(self):
        result = memdag_anticipatory.novelty("anything at all", [])
        self.assertAlmostEqual(result, 1.0, places=6,
                               msg="empty existing list should give novelty 1.0")


class TestExcludedAnswersNotCited(Base):
    """Test 5: redacted and quarantined nodes are absent from existing_derived."""

    def test_excluded_answers_not_cited(self):
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")

        q = "How should I configure the Nebula mesh?"

        # Derive the first answer normally
        n1, created1, _ = memdag_anticipatory.surprise_gated(self.conn, q)
        self.assertTrue(created1)

        # Redact that answer
        memdag_redact.redact_node(self.conn, n1, "test redaction")

        # Derive a second answer (it should be minted since n1 is redacted)
        n2, created2, _ = memdag_anticipatory.surprise_gated(self.conn, q)
        # n2 could be new or could find n1 was redacted... let's check it's created
        # Since n1 is redacted it's excluded from existing_derived
        # So the repeat sees no existing live answers and mints a new one
        self.assertTrue(created2,
                        "should mint a new node when the prior answer is redacted")

        # Now quarantine n2
        memdag_quarantine.quarantine_node(self.conn, n2, "test quarantine")

        # existing_derived should exclude both n1 (redacted) and n2 (quarantined)
        existing = memdag_anticipatory.existing_derived(self.conn)
        existing_ids = {eid for eid, _ in existing}
        self.assertNotIn(n1, existing_ids,
                         "redacted node should not appear in existing_derived")
        self.assertNotIn(n2, existing_ids,
                         "quarantined node should not appear in existing_derived")

        # A third call should mint a new node since nothing live covers the question
        n3, created3, _ = memdag_anticipatory.surprise_gated(self.conn, q)
        self.assertTrue(created3,
                        "should mint a new node when all prior answers are excluded")


class TestRefusalRaises(Base):
    """Test 6: surprise_gated raises ValueError when all sources are tombstoned."""

    def test_refusal_raises(self):
        sid1 = self.add(SRC_NEBULA, "endorsed")
        sid2 = self.add(SRC_NEBULA_B, "user")

        # Tombstone all sources
        memdag.revoke_cascade(self.conn, sid1, "gone")
        memdag.revoke_cascade(self.conn, sid2, "gone")

        log_before = self._log_count()

        with self.assertRaises(ValueError):
            memdag_anticipatory.surprise_gated(self.conn,
                                               "How should I configure Nebula?")

        # query_log must not have grown (refusal = nothing logged)
        self.assertEqual(self._log_count(), log_before,
                         "a refused call must not write to query_log")


class TestObserveRows(Base):
    """Test 7: observe writes ts (ISO), query, and answer_node correctly."""

    def test_observe_rows(self):
        # observe with answer_node
        memdag_anticipatory.observe(self.conn, "test query one", answer_node=42)
        # observe without answer_node (nullable)
        memdag_anticipatory.observe(self.conn, "test query two")

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


if __name__ == "__main__":
    unittest.main()
