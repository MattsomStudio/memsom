#!/usr/bin/env python3
"""Tests for memdag_retrieve.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:/Users/you/memdag -p test_memdag_retrieve.py \
    -t C:/Users/you/memdag -v
"""

import json
import os
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_confid
import memdag_quarantine
import memdag_redact
import memdag_retrieve
import memdag_schema


# ---------------------------------------------------------------------------
# Base test setup (mirrors test_memdag.py Base)
# ---------------------------------------------------------------------------

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        # Run all relevant migrations
        memdag_retrieve.migrate(self.conn)
        memdag_confid.migrate(self.conn)
        memdag_quarantine.migrate(self.conn)
        memdag_redact.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_embed(vec):
    """Return a fake _call_ollama_embed that always returns vec."""
    def _fake(text, timeout=10):
        return list(vec)
    return _fake


def _pack_vec(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


# ---------------------------------------------------------------------------
# TestTokenize
# ---------------------------------------------------------------------------

class TestTokenize(unittest.TestCase):
    def test_lowercase_alnum(self):
        tokens = memdag_retrieve.tokenize("Hello, World! 123")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("123", tokens)

    def test_stopwords_removed(self):
        tokens = memdag_retrieve.tokenize("how should I configure a network")
        # "how", "should", "i", "a" are stopwords
        for sw in ("how", "shoul", "i", "a"):
            self.assertNotIn(sw, tokens)
        self.assertIn("config", tokens)  # "configure" -> stem "config"

    def test_stem_length_6(self):
        tokens = memdag_retrieve.tokenize("configure")
        self.assertIn("config", tokens)

    def test_empty_text(self):
        self.assertEqual(memdag_retrieve.tokenize(""), [])

    def test_repeats_preserved(self):
        tokens = memdag_retrieve.tokenize("nebula nebula")
        self.assertEqual(tokens.count("nebula"), 2)


# ---------------------------------------------------------------------------
# TestMigrate
# ---------------------------------------------------------------------------

class TestMigrate(Base):
    def test_tables_created(self):
        for table in ("postings", "docstats", "embeddings"):
            self.assertTrue(
                memdag_schema_table_exists(self.conn, table),
                f"table {table!r} missing after migrate()",
            )

    def test_migrate_idempotent(self):
        # Should not raise on second call
        memdag_retrieve.migrate(self.conn)
        memdag_retrieve.migrate(self.conn)


def memdag_schema_table_exists(conn, name):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# TestIndexNode
# ---------------------------------------------------------------------------

class TestIndexNode(Base):
    def test_index_node_builds_postings(self):
        nid = self.add("nebula lighthouse configuration network", "user")
        memdag_retrieve.index_node(self.conn, nid)
        rows = self.conn.execute(
            "SELECT term, tf FROM postings WHERE node_id=?", (nid,)
        ).fetchall()
        self.assertGreater(len(rows), 0)

    def test_index_node_builds_docstats(self):
        nid = self.add("nebula lighthouse configuration", "endorsed")
        memdag_retrieve.index_node(self.conn, nid)
        row = self.conn.execute(
            "SELECT length FROM docstats WHERE node_id=?", (nid,)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertGreater(row[0], 0)

    def test_index_node_skips_agent_derived(self):
        src = self.add("nebula network source", "user")
        nid, _ = memdag.derive_node(self.conn, "answer about nebula", [src])
        memdag_retrieve.index_node(self.conn, nid)
        row = self.conn.execute(
            "SELECT 1 FROM docstats WHERE node_id=?", (nid,)
        ).fetchone()
        self.assertIsNone(row)

    def test_index_node_skips_tombstoned(self):
        nid = self.add("nebula lighthouse", "user")
        memdag.revoke_cascade(self.conn, nid, "gone")
        memdag_retrieve.index_node(self.conn, nid)
        row = self.conn.execute(
            "SELECT 1 FROM docstats WHERE node_id=?", (nid,)
        ).fetchone()
        self.assertIsNone(row)

    def test_index_node_rebuild_replaces_old(self):
        """index_node called twice on same node -> no duplicate postings."""
        nid = self.add("nebula lighthouse network configuration", "user")
        memdag_retrieve.index_node(self.conn, nid)
        memdag_retrieve.index_node(self.conn, nid)
        count = self.conn.execute(
            "SELECT COUNT(*) FROM postings WHERE node_id=?", (nid,)
        ).fetchone()[0]
        # Should equal the number of distinct terms, not doubled
        distinct = self.conn.execute(
            "SELECT COUNT(DISTINCT term) FROM postings WHERE node_id=?", (nid,)
        ).fetchone()[0]
        self.assertEqual(count, distinct)

    def test_index_node_stores_embedding_when_ollama_up(self):
        nid = self.add("nebula configuration", "user")
        fake_vec = [0.1, 0.2, 0.3, 0.4]
        with patch.object(memdag_retrieve, "_call_ollama_embed", _make_fake_embed(fake_vec)):
            memdag_retrieve.index_node(self.conn, nid)
        row = self.conn.execute(
            "SELECT vec FROM embeddings WHERE node_id=?", (nid,)
        ).fetchone()
        self.assertIsNotNone(row)
        recovered = memdag_retrieve._blob_to_vec(row[0])
        self.assertEqual(len(recovered), 4)
        for a, b in zip(recovered, fake_vec):
            self.assertAlmostEqual(a, b, places=5)

    def test_index_node_skips_embedding_when_ollama_down(self):
        nid = self.add("nebula configuration", "user")
        def _failing(*args, **kwargs):
            raise OSError("connection refused")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _failing):
            # Must not raise
            memdag_retrieve.index_node(self.conn, nid)
        # BM25 index should still be built
        row = self.conn.execute(
            "SELECT length FROM docstats WHERE node_id=?", (nid,)
        ).fetchone()
        self.assertIsNotNone(row)
        # No embedding stored
        emb = self.conn.execute(
            "SELECT 1 FROM embeddings WHERE node_id=?", (nid,)
        ).fetchone()
        self.assertIsNone(emb)


# ---------------------------------------------------------------------------
# TestIndexAll
# ---------------------------------------------------------------------------

class TestIndexAll(Base):
    def test_reindex_builds_postings_for_all_sources(self):
        ids = []
        for i in range(5):
            ids.append(self.add(f"nebula configuration node number {i} lighthouse", "user"))
        # Also add an agent-derived node that should NOT be indexed
        derived_id, _ = memdag.derive_node(self.conn, "derived answer", [ids[0]])

        def _failing(*args, **kwargs):
            raise OSError("no ollama")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _failing):
            n = memdag_retrieve.index_all(self.conn)

        self.assertEqual(n, 5)  # only source nodes counted
        # derived node not in docstats
        row = self.conn.execute(
            "SELECT 1 FROM docstats WHERE node_id=?", (derived_id,)
        ).fetchone()
        self.assertIsNone(row)


# ---------------------------------------------------------------------------
# TestBM25
# ---------------------------------------------------------------------------

class TestBM25(Base):
    def _plant_docs(self):
        """Plant 20+ docs: one highly relevant, rest noise."""
        relevant_id = self.add(
            "nebula overlay network lighthouse configuration static host map "
            "nebula nebula overlay overlay network configuration", "endorsed"
        )
        noise_ids = []
        for i in range(22):
            noise_ids.append(self.add(
                f"unrelated document about gardening tomatoes cucumbers {i}", "user"
            ))
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
        return relevant_id, noise_ids

    def test_bm25_ranks_relevant_above_noise(self):
        relevant_id, noise_ids = self._plant_docs()
        results = memdag_retrieve.bm25(self.conn, "nebula overlay network lighthouse", k=5)
        self.assertGreater(len(results), 0)
        top_nid = results[0][0]
        self.assertEqual(top_nid, relevant_id,
                         f"Expected relevant doc [{relevant_id}] at rank 1, got [{top_nid}]")

    def test_bm25_returns_empty_when_no_postings(self):
        results = memdag_retrieve.bm25(self.conn, "anything", k=5)
        self.assertEqual(results, [])

    def test_bm25_returns_empty_for_stopword_only_query(self):
        self.add("nebula configuration network", "user")
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
        results = memdag_retrieve.bm25(self.conn, "how should the", k=5)
        self.assertEqual(results, [])

    def test_bm25_respects_k(self):
        for i in range(10):
            self.add(f"nebula network configuration test document {i}", "user")
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
        results = memdag_retrieve.bm25(self.conn, "nebula network configuration", k=3)
        self.assertLessEqual(len(results), 3)

    def test_bm25_scores_positive(self):
        self.add("nebula network configuration", "user")
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
        results = memdag_retrieve.bm25(self.conn, "nebula network", k=5)
        for _, score in results:
            self.assertGreater(score, 0.0)


# ---------------------------------------------------------------------------
# TestVectorSearch
# ---------------------------------------------------------------------------

class TestVectorSearch(Base):
    def test_vector_search_returns_empty_when_ollama_down(self):
        def _failing(*a, **kw):
            raise OSError("connection refused")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _failing):
            results = memdag_retrieve.vector_search(self.conn, "nebula config", k=5)
        self.assertEqual(results, [])

    def test_vector_search_returns_empty_when_no_embeddings(self):
        self.add("nebula configuration", "user")
        fake_vec = [1.0, 0.0, 0.0]
        with patch.object(memdag_retrieve, "_call_ollama_embed", _make_fake_embed(fake_vec)):
            results = memdag_retrieve.vector_search(self.conn, "nebula", k=5)
        # No embeddings stored yet -> still empty
        self.assertEqual(results, [])

    def test_vector_search_cosine_order(self):
        """Monkeypatched Ollama: vectors inserted directly; query close to doc B."""
        # Plant two docs with known embeddings
        nid_a = self.add("nebula overlay", "user")
        nid_b = self.add("lighthouse network", "user")

        # Manually insert embeddings
        vec_a = [1.0, 0.0, 0.0, 0.0]  # points in x direction
        vec_b = [0.0, 1.0, 0.0, 0.0]  # points in y direction
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec) VALUES (?,?,?,?)",
                (nid_a, "test", 4, _pack_vec(vec_a)),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec) VALUES (?,?,?,?)",
                (nid_b, "test", 4, _pack_vec(vec_b)),
            )

        # Query vector close to vec_b (y direction)
        query_vec = [0.0, 0.9, 0.1, 0.0]
        with patch.object(memdag_retrieve, "_call_ollama_embed", _make_fake_embed(query_vec)):
            results = memdag_retrieve.vector_search(self.conn, "dummy query", k=5)

        self.assertGreater(len(results), 0)
        top_nid = results[0][0]
        self.assertEqual(top_nid, nid_b,
                         f"Expected doc B [{nid_b}] closest to query, got [{top_nid}]")

    def test_vector_search_respects_k(self):
        for i in range(5):
            nid = self.add(f"content {i}", "user")
            vec = [float(i), 0.0, 0.0, 0.0]
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec) VALUES (?,?,?,?)",
                    (nid, "test", 4, _pack_vec(vec)),
                )
        query_vec = [1.0, 0.0, 0.0, 0.0]
        with patch.object(memdag_retrieve, "_call_ollama_embed", _make_fake_embed(query_vec)):
            results = memdag_retrieve.vector_search(self.conn, "test", k=2)
        self.assertLessEqual(len(results), 2)


# ---------------------------------------------------------------------------
# TestRRF
# ---------------------------------------------------------------------------

class TestRRF(unittest.TestCase):
    def test_rrf_deterministic(self):
        bm25_list = [(1, 3.0), (2, 2.0), (3, 1.0)]
        vec_list = [(2, 0.9), (1, 0.8), (4, 0.5)]
        r1 = memdag_retrieve._rrf_fuse(bm25_list, vec_list)
        r2 = memdag_retrieve._rrf_fuse(bm25_list, vec_list)
        self.assertEqual(r1, r2)

    def test_rrf_boosts_overlap(self):
        """A doc appearing in BOTH lists should outscore one appearing in only one."""
        bm25_list = [(1, 5.0), (2, 2.0)]
        vec_list = [(1, 0.9), (3, 0.8)]
        fused = memdag_retrieve._rrf_fuse(bm25_list, vec_list)
        nids = [nid for nid, _ in fused]
        # Doc 1 is in both -> should be rank 1
        self.assertEqual(nids[0], 1)

    def test_rrf_disjoint_lists(self):
        bm25_list = [(1, 5.0)]
        vec_list = [(2, 0.9)]
        fused = memdag_retrieve._rrf_fuse(bm25_list, vec_list)
        nids = {nid for nid, _ in fused}
        self.assertIn(1, nids)
        self.assertIn(2, nids)

    def test_rrf_empty_inputs(self):
        self.assertEqual(memdag_retrieve._rrf_fuse([], []), [])
        r = memdag_retrieve._rrf_fuse([(1, 2.0)], [])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0][0], 1)

    def test_rrf_scores_positive(self):
        bm25_list = [(1, 3.0), (2, 1.5)]
        vec_list = [(2, 0.8), (3, 0.4)]
        fused = memdag_retrieve._rrf_fuse(bm25_list, vec_list)
        for _, score in fused:
            self.assertGreater(score, 0.0)


# ---------------------------------------------------------------------------
# TestRetrieve
# ---------------------------------------------------------------------------

class TestRetrieve(Base):
    def _seed_and_index(self, docs):
        """Insert docs [(content, channel)] and index with fake embed disabled."""
        ids = []
        for content, channel in docs:
            ids.append(self.add(content, channel))
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
        return ids

    def test_retrieve_bm25_fallback_when_ollama_down(self):
        """retrieve() must not crash when Ollama is unreachable."""
        nid = self.add("nebula lighthouse network overlay configuration", "endorsed")
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(self.conn, "nebula lighthouse", k=5)
        # Must return results (BM25-only) without crashing
        self.assertGreater(len(results), 0)
        nids = [r[0] for r in results]
        self.assertIn(nid, nids)

    def test_retrieve_respects_clearance(self):
        """Nodes with conf_label above clearance must be excluded."""
        pub_id = self.add("nebula public network configuration overlay", "user")
        sec_id = self.add("nebula secret network configuration overlay", "user")
        # Classify sec_id as secret (2)
        memdag_confid.classify(self.conn, sec_id, "secret")

        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(
                self.conn, "nebula network configuration", k=10, clearance="public"
            )
        nids = [r[0] for r in results]
        self.assertIn(pub_id, nids)
        self.assertNotIn(sec_id, nids)

    def test_retrieve_excludes_quarantined(self):
        nid_good = self.add("nebula overlay lighthouse configuration network", "user")
        nid_quar = self.add("nebula overlay lighthouse quarantine network", "user")
        memdag_quarantine.quarantine_node(self.conn, nid_quar, "test quarantine")

        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(
                self.conn, "nebula overlay lighthouse", k=10,
                exclude_quarantined=True,
            )
        nids = [r[0] for r in results]
        self.assertIn(nid_good, nids)
        self.assertNotIn(nid_quar, nids)

    def test_retrieve_excludes_redacted(self):
        nid_good = self.add("nebula overlay lighthouse configuration network", "endorsed")
        nid_red = self.add("nebula overlay lighthouse redacted network", "user")
        memdag_redact.redact_node(self.conn, nid_red, "test redact")

        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(
                self.conn, "nebula overlay lighthouse", k=10,
                exclude_redacted=True,
            )
        nids = [r[0] for r in results]
        self.assertIn(nid_good, nids)
        self.assertNotIn(nid_red, nids)

    def test_retrieve_excludes_tombstoned(self):
        nid_live = self.add("nebula overlay lighthouse configuration", "endorsed")
        nid_dead = self.add("nebula overlay lighthouse tombstoned", "user")
        memdag.revoke_cascade(self.conn, nid_dead, "killed")

        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(
                self.conn, "nebula overlay lighthouse", k=10,
            )
        nids = [r[0] for r in results]
        self.assertIn(nid_live, nids)
        self.assertNotIn(nid_dead, nids)

    def test_retrieve_empty_db(self):
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            results = memdag_retrieve.retrieve(self.conn, "anything", k=5)
        self.assertEqual(results, [])

    # F-15: redaction must purge the retrieval index and never surface the node,
    # even via the non-default exclude_redacted=False path (fail-safe).
    def test_redaction_deindexes_and_never_surfaces(self):
        nid = self.add("ultrasecret missile launch code foxtrot niner", "user")

        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_node(self.conn, nid)
            before = self.conn.execute(
                "SELECT COUNT(*) FROM postings WHERE node_id=?", (nid,)).fetchone()[0]
            self.assertGreater(before, 0)

            # Redact -> the redact cascade calls deindex_node
            memdag_redact.redact_node(self.conn, nid, "test redact")

            after = self.conn.execute(
                "SELECT COUNT(*) FROM postings WHERE node_id=?", (nid,)).fetchone()[0]
            self.assertEqual(after, 0, "postings must be purged on redaction")
            doc = self.conn.execute(
                "SELECT COUNT(*) FROM docstats WHERE node_id=?", (nid,)).fetchone()[0]
            self.assertEqual(doc, 0)

            # Even the widened flag must not leak the redacted node id.
            leaky = memdag_retrieve.retrieve(
                self.conn, "missile launch code foxtrot", k=5, exclude_redacted=False)
        self.assertNotIn(nid, [r[0] for r in leaky])

    def test_deindex_node_no_op_without_schema(self):
        """deindex_node must never create retrieval tables when they don't exist."""
        tmp = tempfile.TemporaryDirectory()
        try:
            db = Path(tmp.name) / "bare.db"
            os.environ["MEMDAG_DB"] = str(db)
            conn = memdag.get_connection()
            try:
                nid = memdag.insert_node(conn, "x", "user", memdag.RANK["user"])
                memdag_retrieve.deindex_node(conn, nid)  # must not raise / create tables
                self.assertFalse(
                    memdag_schema.table_exists(conn, "postings"))
            finally:
                conn.close()
        finally:
            os.environ.pop("MEMDAG_DB", None)
            tmp.cleanup()

    def test_retrieve_result_shape(self):
        """Each result row must be (id, content, channel, label, source_ref)."""
        nid = self.add("nebula lighthouse network configuration overlay", "endorsed")
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(self.conn, "nebula lighthouse", k=5)
        self.assertGreater(len(results), 0)
        row = results[0]
        self.assertEqual(len(row), 5, "row must be (id, content, channel, label, source_ref)")
        rid, content, channel, label, source_ref = row
        self.assertIsInstance(rid, int)
        self.assertIsInstance(content, str)
        self.assertIn(channel, ("endorsed", "user", "agent-derived", "external"))
        self.assertIsInstance(label, int)

    def test_retrieve_with_vector_cosine_ordering(self):
        """With monkeypatched Ollama, RRF should surface the vectorly-closer doc."""
        nid_a = self.add("nebula overlay network lighthouse config static map", "endorsed")
        nid_b = self.add("tomato cucumber gardening soil water unrelated unrelated", "user")

        # Give nid_a a vec close to query; nid_b a vec far from query
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 0.0, 1.0]
        query_vec = [0.95, 0.1, 0.0]

        call_count = [0]

        def _fake_embed(text, timeout=10):
            call_count[0] += 1
            # First two calls are index_node (content), rest are queries
            if "nebula" in text.lower():
                return list(vec_a)
            elif "tomato" in text.lower():
                return list(vec_b)
            else:
                return list(query_vec)

        with patch.object(memdag_retrieve, "_call_ollama_embed", _fake_embed):
            memdag_retrieve.index_all(self.conn)
            results = memdag_retrieve.retrieve(self.conn, "nebula overlay lighthouse", k=5)

        nids = [r[0] for r in results]
        self.assertIn(nid_a, nids)
        # nid_a should rank above nid_b (both BM25 and vector agree)
        if nid_b in nids:
            self.assertLess(nids.index(nid_a), nids.index(nid_b))

    def test_retrieve_min_integrity_filter(self):
        """min_integrity parameter must exclude nodes below threshold."""
        nid_end = self.add("nebula endorsed overlay configuration lighthouse network", "endorsed")
        nid_ext = self.add("nebula external overlay configuration lighthouse network", "external")

        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
            # min_integrity=2 (user) -> external (0) excluded
            results = memdag_retrieve.retrieve(
                self.conn, "nebula overlay configuration", k=10,
                min_integrity=2,
            )
        nids = [r[0] for r in results]
        self.assertIn(nid_end, nids)
        self.assertNotIn(nid_ext, nids)


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------

class TestCLI(Base):
    def test_reindex_cli(self):
        import io, contextlib
        self.add("nebula overlay lighthouse configuration network", "user")
        def _no_embed(*a, **kw):
            raise OSError("down")
        buf = io.StringIO()
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            with contextlib.redirect_stdout(buf):
                memdag_retrieve.main(["reindex"])
        out = buf.getvalue()
        self.assertIn("indexed", out)
        self.assertIn("1", out)

    def test_retrieve_cli_no_results(self):
        import io, contextlib
        buf = io.StringIO()
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            with contextlib.redirect_stdout(buf):
                memdag_retrieve.main(["retrieve", "nebula"])
        out = buf.getvalue()
        self.assertIn("no results", out)

    def test_retrieve_cli_with_results(self):
        import io, contextlib
        self.add("nebula overlay lighthouse configuration network", "endorsed")
        def _no_embed(*a, **kw):
            raise OSError("down")
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            memdag_retrieve.index_all(self.conn)
        buf = io.StringIO()
        with patch.object(memdag_retrieve, "_call_ollama_embed", _no_embed):
            with contextlib.redirect_stdout(buf):
                memdag_retrieve.main(["retrieve", "nebula overlay lighthouse", "--k", "5"])
        out = buf.getvalue()
        self.assertIn("endorsed", out)


# ---------------------------------------------------------------------------
# TestCosineHelper
# ---------------------------------------------------------------------------

class TestCosineHelper(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(memdag_retrieve._cosine(v, v), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(memdag_retrieve._cosine(a, b), 0.0, places=5)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        self.assertEqual(memdag_retrieve._cosine(a, b), 0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(memdag_retrieve._cosine(a, b), -1.0, places=5)

    def test_mismatched_lengths(self):
        self.assertEqual(memdag_retrieve._cosine([1.0], [1.0, 2.0]), 0.0)


# ---------------------------------------------------------------------------
# TestVecPackUnpack
# ---------------------------------------------------------------------------

class TestVecPackUnpack(unittest.TestCase):
    def test_roundtrip(self):
        vec = [0.1, -0.5, 3.14, 0.0]
        blob = memdag_retrieve._vec_to_blob(vec)
        recovered = memdag_retrieve._blob_to_vec(blob)
        self.assertEqual(len(recovered), len(vec))
        for a, b in zip(recovered, vec):
            self.assertAlmostEqual(a, b, places=5)

    def test_empty_vector(self):
        blob = memdag_retrieve._vec_to_blob([])
        recovered = memdag_retrieve._blob_to_vec(blob)
        self.assertEqual(recovered, [])


if __name__ == "__main__":
    unittest.main()
