#!/usr/bin/env python3
"""Tests for memsom_embed (BGE-M3 triple-fusion backend) and its wiring into
memsom_retrieve.

CI-SAFE BY CONSTRUCTION: the BGE encoder (FlagEmbedding/torch/numpy) is never
imported here. Every test that exercises the bge path patches
`memsom_embed.bge_available` -> True and `encode_doc`/`encode_query` -> canned
dicts. The serialization + scoring primitives are pure stdlib and tested
directly. The deps are absent on the CI box; these tests must still pass.

Run:
  python -W error::DeprecationWarning -m unittest test_memsom_embed -v
"""

import os
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_confid
import memsom_embed
import memsom_quarantine
import memsom_redact
import memsom_retrieve
import memsom_schema


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)  # clean per test
        self.conn = memsom.get_connection()
        memsom_retrieve.migrate(self.conn)
        memsom_embed.migrate(self.conn)
        memsom_confid.migrate(self.conn)
        memsom_quarantine.migrate(self.conn)
        memsom_redact.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])

    def count(self, table, nid):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE node_id = ?", (nid,)
        ).fetchone()[0]


def _enc(dense, sparse, colbert):
    """Build a canned encode result dict."""
    return {"dense": [float(x) for x in dense],
            "sparse": {str(k): float(v) for k, v in sparse.items()},
            "colbert": [[float(x) for x in row] for row in colbert]}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestEmbedDispatch(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)

    def test_default_is_ollama(self):
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        self.assertEqual(memsom_embed.backend(), "ollama")

    def test_honors_env(self):
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        self.assertEqual(memsom_embed.backend(), "bge-m3")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        self.assertEqual(memsom_embed.backend(), "bm25")

    def test_unknown_falls_back_to_ollama(self):
        os.environ["MEMDAG_EMBED_BACKEND"] = "nonsense"
        self.assertEqual(memsom_embed.backend(), "ollama")

    def test_active_model_name(self):
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        self.assertEqual(memsom_embed.active_model_name(), "bge-m3")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        self.assertEqual(memsom_embed.active_model_name(), "")
        os.environ["MEMDAG_EMBED_BACKEND"] = "ollama"
        self.assertEqual(memsom_embed.active_model_name(),
                         memsom_retrieve._embed_model())

    def test_colbert_candidates_default_and_override(self):
        os.environ.pop("MEMDAG_COLBERT_CANDIDATES", None)
        self.assertEqual(memsom_embed.colbert_candidates(), 100)
        os.environ["MEMDAG_COLBERT_CANDIDATES"] = "7"
        try:
            self.assertEqual(memsom_embed.colbert_candidates(), 7)
        finally:
            os.environ.pop("MEMDAG_COLBERT_CANDIDATES", None)


# ---------------------------------------------------------------------------
# Pure scoring + serialization primitives (no mocks needed; stdlib only)
# ---------------------------------------------------------------------------

class TestPrimitives(unittest.TestCase):
    def test_colbert_blob_roundtrip(self):
        mat = [[0.5, -0.25], [1.0, 0.0], [-0.75, 0.125]]
        blob = memsom_embed.colbert_to_blob(mat)
        # 3 tokens * 2 dim * 2 bytes (fp16) = 12 bytes
        self.assertEqual(len(blob), 12)
        back = memsom_embed.blob_to_colbert(blob, 3, 2)
        self.assertEqual(back, mat)  # all values exactly fp16-representable

    def test_colbert_blob_empty(self):
        self.assertEqual(memsom_embed.blob_to_colbert(b"", 0, 1024), [])

    def test_sparse_dot_shared_keys(self):
        self.assertEqual(memsom_embed.sparse_dot({"a": 2.0, "b": 3.0},
                                                 {"b": 4.0, "c": 5.0}), 12.0)

    def test_sparse_dot_disjoint_is_zero(self):
        self.assertEqual(memsom_embed.sparse_dot({"a": 1.0}, {"b": 1.0}), 0.0)

    def test_sparse_dot_empty(self):
        self.assertEqual(memsom_embed.sparse_dot({}, {"a": 1.0}), 0.0)

    def test_colbert_maxsim_known(self):
        # Each query token's best doc match is 1.0; summed over 2 tokens = 2.0.
        q = [[1.0, 0.0], [0.0, 1.0]]
        d = [[1.0, 0.0], [0.0, 1.0]]
        self.assertAlmostEqual(memsom_embed.colbert_maxsim(q, d), 2.0, places=5)

    def test_colbert_maxsim_empty(self):
        self.assertEqual(memsom_embed.colbert_maxsim([], [[1.0]]), 0.0)
        self.assertEqual(memsom_embed.colbert_maxsim([[1.0]], []), 0.0)


# ---------------------------------------------------------------------------
# Degradation: bge unavailable / encoder fails -> Ollama -> BM25
# ---------------------------------------------------------------------------

class TestBgeDegradation(Base):
    def _fake_ollama(self, vec):
        def _f(text, timeout=10):
            return list(vec)
        return _f

    def test_unavailable_falls_back_to_ollama(self):
        """backend=bge-m3 but FlagEmbedding absent: index_node stores the Ollama
        dense embedding (model=nomic), and NO sparse/colbert rows."""
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        nid = self.add("nebula overlay mesh", "user")
        with patch.object(memsom_embed, "bge_available", lambda: False), \
             patch.object(memsom_retrieve, "_call_ollama_embed",
                          self._fake_ollama([1.0, 0.0, 0.0])):
            memsom_retrieve.index_node(self.conn, nid)
        # BM25 present, ollama dense present, bge tables empty.
        self.assertGreater(self.count("docstats", nid), 0)
        row = self.conn.execute(
            "SELECT model FROM embeddings WHERE node_id = ?", (nid,)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], memsom_retrieve._embed_model())
        self.assertEqual(self.count("sparse_vecs", nid), 0)
        self.assertEqual(self.count("colbert_vecs", nid), 0)

    def test_encoder_failure_falls_back_to_ollama(self):
        """bge available but encode_doc returns None -> Ollama path taken."""
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        nid = self.add("lighthouse network", "user")
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", lambda t: None), \
             patch.object(memsom_retrieve, "_call_ollama_embed",
                          self._fake_ollama([0.0, 1.0, 0.0])):
            memsom_retrieve.index_node(self.conn, nid)
        row = self.conn.execute(
            "SELECT model FROM embeddings WHERE node_id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], memsom_retrieve._embed_model())
        self.assertEqual(self.count("colbert_vecs", nid), 0)

    def test_bm25_backend_stores_no_vectors(self):
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        nid = self.add("nebula overlay", "user")
        with patch.object(memsom_retrieve, "_call_ollama_embed",
                          self._fake_ollama([1.0, 0.0])):
            memsom_retrieve.index_node(self.conn, nid)
        self.assertGreater(self.count("docstats", nid), 0)  # BM25 still built
        self.assertEqual(self.count("embeddings", nid), 0)  # no vectors at all
        self.assertEqual(self.count("sparse_vecs", nid), 0)


# ---------------------------------------------------------------------------
# Collision fix: vector_search reads only the active backend's model rows
# ---------------------------------------------------------------------------

class TestModelCollisionFix(Base):
    def _plant(self, nid, model, vec):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec)"
                " VALUES (?,?,?,?)",
                (nid, model, len(vec), struct.pack(f"<{len(vec)}f", *vec)))

    def test_ollama_query_ignores_bge_rows(self):
        nid_nomic = self.add("alpha", "user")
        nid_bge = self.add("beta", "user")
        self._plant(nid_nomic, memsom_retrieve._embed_model(), [1.0, 0.0, 0.0, 0.0])
        self._plant(nid_bge, "bge-m3", [1.0, 0.0, 0.0, 0.0])
        os.environ["MEMDAG_EMBED_BACKEND"] = "ollama"
        with patch.object(memsom_retrieve, "_call_ollama_embed",
                          lambda t, timeout=10: [1.0, 0.0, 0.0, 0.0]):
            res = memsom_retrieve.vector_search(self.conn, "q", k=5)
        ids = [r[0] for r in res]
        self.assertIn(nid_nomic, ids)
        self.assertNotIn(nid_bge, ids)  # bge row filtered out

    def test_bge_query_ignores_ollama_rows(self):
        nid_nomic = self.add("alpha", "user")
        nid_bge = self.add("beta", "user")
        self._plant(nid_nomic, memsom_retrieve._embed_model(), [1.0, 0.0])
        self._plant(nid_bge, "bge-m3", [1.0, 0.0])
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        enc_q = lambda t: _enc([1.0, 0.0], {}, [[1.0, 0.0]])
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", enc_q):
            res = memsom_retrieve.vector_search(self.conn, "q", k=5)
        ids = [r[0] for r in res]
        self.assertIn(nid_bge, ids)
        self.assertNotIn(nid_nomic, ids)


# ---------------------------------------------------------------------------
# Triple-fusion storage + ColBERT re-rank
# ---------------------------------------------------------------------------

class TestTripleFusion(Base):
    def test_store_bge_writes_all_three_tables(self):
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        nid = self.add("nebula mesh overlay network", "user")
        enc = lambda t: _enc([1.0, 0.0, 0.0, 0.0], {"42": 0.7, "7": 0.3},
                             [[1.0, 0.0], [0.0, 1.0]])
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", enc):
            memsom_retrieve.index_node(self.conn, nid)
        self.assertEqual(self.count("docstats", nid), 1)
        emb = self.conn.execute(
            "SELECT model FROM embeddings WHERE node_id=?", (nid,)).fetchone()
        self.assertEqual(emb[0], "bge-m3")
        self.assertEqual(self.count("sparse_vecs", nid), 1)
        cb = self.conn.execute(
            "SELECT n_tokens, dim FROM colbert_vecs WHERE node_id=?", (nid,)).fetchone()
        self.assertEqual(cb, (2, 2))

    def test_colbert_rerank_reorders_within_candidates(self):
        """Two pool-gated candidates; ColBERT MaxSim strongly prefers B ->
        B ranks first after rerank even if dense/sparse were a near-tie."""
        ids = [self.add("doc alpha shared term", "user"),
               self.add("doc beta shared term", "user")]
        nid_a, nid_b = ids
        # Index both under bge with identical dense/sparse but B-favoring colbert.
        def enc_doc(text):
            if "alpha" in text:
                return _enc([1.0, 0.0], {"1": 1.0}, [[0.1, 0.0]])
            return _enc([1.0, 0.0], {"1": 1.0}, [[1.0, 0.0]])  # beta: strong colbert
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", enc_doc):
            for nid in ids:
                memsom_retrieve.index_node(self.conn, nid)
        # Query colbert vector aligns with B's strong token.
        enc_q = lambda t: _enc([1.0, 0.0], {"1": 1.0}, [[1.0, 0.0]])
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", enc_q):
            res = memsom_retrieve.retrieve(self.conn, "shared term", k=2)
        ids_out = [r[0] for r in res]
        self.assertEqual(ids_out[0], nid_b, f"colbert should rank B first, got {ids_out}")

    def test_colbert_rerank_is_pool_only(self):
        """colbert_rerank reorders exactly the ids it is given; never adds."""
        cand = [101, 202]
        # No colbert rows for these ids, bge available -> all score 0, order kept.
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query",
                          lambda t: _enc([1.0], {}, [[1.0]])):
            os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
            out = memsom_retrieve.colbert_rerank(self.conn, "q", cand)
        self.assertEqual([nid for nid, _ in out], cand)


# ---------------------------------------------------------------------------
# Deindex purges all three tables (redact cascade + direct)
# ---------------------------------------------------------------------------

class TestDeindexPurgesAllThree(Base):
    def _index_bge(self, nid, text="content here"):
        enc = lambda t: _enc([1.0, 0.0], {"3": 1.0}, [[1.0, 0.0]])
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", enc):
            os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
            memsom_retrieve.index_node(self.conn, nid)

    def test_deindex_node_purges_bge_tables(self):
        nid = self.add("nebula overlay", "user")
        self._index_bge(nid)
        self.assertEqual(self.count("sparse_vecs", nid), 1)
        self.assertEqual(self.count("colbert_vecs", nid), 1)
        memsom_retrieve.deindex_node(self.conn, nid)
        self.assertEqual(self.count("sparse_vecs", nid), 0)
        self.assertEqual(self.count("colbert_vecs", nid), 0)
        self.assertEqual(self.count("embeddings", nid), 0)

    def test_redact_cascade_purges_bge_tables(self):
        nid = self.add("secret nebula key material", "user")
        self._index_bge(nid)
        with self.conn:
            memsom_redact.redact_node(self.conn, nid, "test redaction", cascade=True)
        self.assertEqual(self.count("sparse_vecs", nid), 0)
        self.assertEqual(self.count("colbert_vecs", nid), 0)

    def test_deindex_bge_noop_without_tables(self):
        """deindex_bge must be a safe no-op when the bge schema is absent."""
        import sqlite3
        bare = sqlite3.connect(":memory:")  # no migrations -> no bge tables
        try:
            memsom_embed.deindex_bge(bare, 1)  # must not raise
        finally:
            bare.close()


# ---------------------------------------------------------------------------
# SECURITY: the re-ranker has no membership power
# ---------------------------------------------------------------------------

class TestSecurityInvariant(Base):
    def _set_conf(self, nid, level):
        with self.conn:
            self.conn.execute(
                "UPDATE nodes SET conf_label = ? WHERE id = ?", (level, nid))

    def test_above_clearance_node_excluded_despite_strong_colbert(self):
        """A SECRET node with a maximal-MaxSim ColBERT vector is excluded at the
        pool gate and never enters the re-ranker — clearance, not ranking,
        decides membership."""
        nid_secret = self.add("classified launch codes shared", "user")
        nid_low = self.add("public shared notes", "user")
        # Both indexed under bge; secret gets the STRONGER colbert match.
        def enc_doc(text):
            if "classified" in text:
                return _enc([1.0, 0.0], {"9": 1.0}, [[1.0, 0.0]])   # dominant
            return _enc([0.0, 1.0], {"5": 1.0}, [[0.0, 0.2]])       # weak
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", enc_doc):
            for nid in (nid_secret, nid_low):
                memsom_retrieve.index_node(self.conn, nid)
        self._set_conf(nid_secret, 3)  # SECRET
        self._set_conf(nid_low, 0)     # PUBLIC

        enc_q = lambda t: _enc([1.0, 0.0], {"9": 1.0}, [[1.0, 0.0]])  # favors secret
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", enc_q):
            # Low clearance: secret must be excluded despite superior colbert.
            res_low = memsom_retrieve.retrieve(self.conn, "shared", k=5, clearance="public")
            ids_low = [r[0] for r in res_low]
            self.assertNotIn(nid_secret, ids_low)
            self.assertIn(nid_low, ids_low)
            # High clearance: secret is now eligible (proves the GATE excluded it,
            # not a bug).
            res_hi = memsom_retrieve.retrieve(self.conn, "shared", k=5, clearance="topsecret")
            self.assertIn(nid_secret, [r[0] for r in res_hi])

    def test_returned_low_row_keeps_real_label(self):
        nid = self.add("ordinary user note", "user")
        enc = lambda t: _enc([1.0], {"1": 1.0}, [[1.0]])
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", enc):
            memsom_retrieve.index_node(self.conn, nid)
        enc_q = lambda t: _enc([1.0], {"1": 1.0}, [[1.0]])
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", enc_q):
            res = memsom_retrieve.retrieve(self.conn, "ordinary note", k=5)
        row = [r for r in res if r[0] == nid][0]
        # tuple = (id, content, channel, label, source_ref); label is the real
        # user rank, untouched by ranking.
        self.assertEqual(row[3], memsom.RANK["user"])


# ---------------------------------------------------------------------------
# N-ary RRF: two-arg golden unchanged, three-arg fuses
# ---------------------------------------------------------------------------

class TestRrfNary(unittest.TestCase):
    def test_two_arg_matches_manual(self):
        a = [(1, 5.0), (2, 4.0)]
        b = [(2, 0.9), (3, 0.8)]
        fused = dict(memsom_retrieve._rrf_fuse(a, b))
        # node 2 appears in both at rank 2 (a) and rank 1 (b)
        self.assertAlmostEqual(fused[2], 1.0 / (60 + 2) + 1.0 / (60 + 1), places=9)
        self.assertAlmostEqual(fused[1], 1.0 / (60 + 1), places=9)

    def test_three_arg_fuses(self):
        a = [(1, 1.0)]
        b = [(2, 1.0)]
        c = [(1, 1.0), (2, 1.0)]
        fused = dict(memsom_retrieve._rrf_fuse(a, b, c))
        self.assertEqual(set(fused.keys()), {1, 2})

    def test_rrf_c_keyword(self):
        a = [(1, 1.0)]
        fused = dict(memsom_retrieve._rrf_fuse(a, rrf_c=10))
        self.assertAlmostEqual(fused[1], 1.0 / (10 + 1), places=9)


if __name__ == "__main__":
    unittest.main()
