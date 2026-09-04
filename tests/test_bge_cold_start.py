#!/usr/bin/env python3
"""Cold-start-on-demand for the BGE-M3 query encoder (option 2, 2026-09-04).

WHAT SHIPPED. The bge-m3 encoder is a ~5 GB torch process that is lazy and
idle-killed, so most of the time it is DOWN. Before this change, a retrieve
whose backend was bge-m3 fell STRAIGHT to BM25 the instant a cheap probe said
the encoder was not already up (`vector_search` bailed on `bge_usable()`), and
recorded a "query encoder unreachable" degraded signal — so the tool/hook path
was keyword-only almost always, and "idle" masqueraded as "degraded".

Now `vector_search` calls `embed.cold_start_encode_query`, which ATTEMPTS the
encode and WAITS (bounded) for the encoder to come up (in-process model load,
or the local supervisor spawning its backend on demand). Consequences under
test here:

  1. an idle/down encoder that CAN come up -> DENSE results, NO degraded line
     (a keyword-disjoint query still lands its semantic hit);
  2. the in-process model is evicted after `retrieval.bge_idle_ttl` idle seconds
     (default 60) so VRAM is not pinned; the knob drives the window;
  3. a GENUINE cold-start failure (encoder unimportable / erroring / will not
     come up in the bounded window) STILL degrades to BM25 and records the
     signal;
  4. the cold start is bounded — it never hangs a prompt forever.

RED-BEFORE / GREEN-AFTER: on the pre-change tree
`test_cold_encoder_cold_starts_and_returns_dense` FAILS (vector_search bailed to
[] on bge_usable()==False, so the dense-only hit never surfaced and a degraded
trail was written); `test_idle_ttl_*` / `cold_start_encode_query` /
`bge_idle_ttl` FAIL by AttributeError (the symbols did not exist).

CI-SAFE: torch / FlagEmbedding / the supervisor are never reached — every
encoder is patched to a canned vector, to None, or to a slow stub.
"""

import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import memsom
from memsom import tuning as memsom_tuning
from memsom.interface import cli as memsom_cli
from memsom.retrieval import bge_client
from memsom.retrieval import embed as memsom_embed
from memsom.retrieval import retrieve as memsom_retrieve

BGE = memsom_embed.BGE_MODEL_NAME


def _blob(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


class Base(unittest.TestCase):
    """Throwaway bge-m3 store per test; embed module globals reset so an idle
    evictor / cached model from another test can never leak in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "store" / "test.db"
        self._env = {k: os.environ.get(k) for k in
                     ("MEMDAG_DB", "MEMDAG_EMBED_BACKEND", "MEMDAG_BGE_IDLE_TTL",
                      "MEMDAG_BGE_ENCODE_VIA")}
        os.environ["MEMDAG_DB"] = str(self.db)
        # Do NOT pin bge-m3 before migrate_all: it runs the feature-status
        # recorder, which under bge-m3 calls bge_available() -> `import
        # FlagEmbedding`. Tests must reach zero heavy deps, so build the store
        # under the suite default and switch to bge-m3 per test (set_backend).
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        memsom_tuning.clear_override("embed.backend")
        memsom_tuning.clear_override("retrieval.bge_idle_ttl")
        memsom_retrieve._LAST_QUERY_FALLBACK = None
        self._reset_embed_globals()
        self.conn = memsom.get_connection()
        memsom_cli.migrate_all(self.conn)

    def set_backend(self, name):
        os.environ["MEMDAG_EMBED_BACKEND"] = name
        memsom_tuning.clear_override("embed.backend")

    def tearDown(self):
        self.conn.close()
        self._reset_embed_globals()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        memsom_tuning.clear_override("embed.backend")
        memsom_tuning.clear_override("retrieval.bge_idle_ttl")
        self.tmp.cleanup()

    @staticmethod
    def _reset_embed_globals():
        memsom_embed._MODEL = None
        memsom_embed._LAST_USE = 0.0
        memsom_embed._EVICTOR = None
        memsom_embed._BGE_AVAILABLE = None

    def add(self, content, channel="user"):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])

    def put_vec(self, nid, vec, model=BGE):
        """Store a canned bge dense vector AND build the BM25 postings directly
        (never index_node -- that would fire a real network embed to the live
        supervisor on this box and clobber the canned vector)."""
        content = self.conn.execute("SELECT content FROM nodes WHERE id = ?",
                                    (nid,)).fetchone()[0]
        tf = {}
        for t in memsom_retrieve.tokenize(content):
            tf[t] = tf.get(t, 0) + 1
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec) VALUES (?,?,?,?)",
                (nid, model, len(vec), _blob(list(vec))))
            self.conn.execute("DELETE FROM postings WHERE node_id = ?", (nid,))
            if tf:
                self.conn.executemany(
                    "INSERT INTO postings(term, node_id, tf) VALUES (?,?,?)",
                    [(term, nid, n) for term, n in tf.items()])
            self.conn.execute(
                "INSERT OR REPLACE INTO docstats(node_id, length) VALUES (?, ?)",
                (nid, sum(tf.values())))


# ---------------------------------------------------------------------------
# 1. cold-start-on-demand -> dense, no degraded
# ---------------------------------------------------------------------------

class TestColdStartReturnsDense(Base):
    def test_cold_encoder_cold_starts_and_returns_dense(self):
        """The encoder is DOWN at call time (bge_usable() would be False), yet a
        keyword-disjoint query lands its semantic hit via a cold start, and NO
        degraded line is emitted."""
        self.set_backend("bge-m3")
        # Target: lexically DISJOINT from the query, so BM25 cannot reach it — it
        # can only surface through the dense vector. Distractors are disjoint
        # from the query too, so the whole fused result is dense-driven: if the
        # cold start did NOT run, retrieve() would fall to BM25 and return
        # nothing relevant, and the semantic hit would never appear.
        target = self.add("primary display adapter model in the tower unit")
        self.put_vec(target, [1.0, 0.0, 0.0, 0.0])
        for i in range(3):
            d = self.add(f"unrelated note {i} about kitchen recipes and travel plans")
            self.put_vec(d, [0.0, 1.0, 0.0, 0.0])

        query = "which graphics card is in the desktop"          # disjoint from every node
        canned = {"dense": [1.0, 0.0, 0.0, 0.0], "sparse": {}, "colbert": []}

        # Encoder cold: torch not importable AND supervisor /health down, so the
        # OLD bge_usable() gate would have bailed to [] before encoding. The cold
        # start (encode_query) nonetheless produces a vector.
        with patch.object(memsom_embed, "bge_available", lambda: False), \
             patch.object(bge_client, "bge_http_available", lambda force=False: False), \
             patch.object(memsom_embed, "encode_query", lambda t: dict(canned)):
            vec_hits = memsom_retrieve.vector_search(self.conn, query, k=5)
            results = memsom_retrieve.retrieve(self.conn, query, k=3)

        top_ids = [nid for nid, _ in vec_hits]
        self.assertEqual(top_ids[0], target,
                         f"dense cold start did not rank the semantic hit first: {vec_hits}")
        self.assertTrue(results and results[0][0] == target,
                        f"the keyword-disjoint semantic hit was not the top fused "
                        f"result (dense cold start did not drive retrieval): "
                        f"{[r[0] for r in results]}")
        # A successful cold start is NOT a degradation.
        self.assertIsNone(memsom_retrieve.last_query_fallback_reason())
        self.assertIsNone(memsom_retrieve.last_query_fallback(self.conn))
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_cold_start_clears_a_previous_degraded_trail(self):
        """A genuine failure writes the trail; a later successful cold start
        clears it so no stale warning survives the warm-up."""
        self.set_backend("bge-m3")
        self.put_vec(self.add("some vector node"), [0.5, 0.5, 0.0, 0.0])
        with patch.object(memsom_embed, "encode_query", lambda t: None):
            memsom_retrieve.vector_search(self.conn, "q")
        self.assertIsNotNone(memsom_retrieve.last_query_fallback(self.conn))
        canned = {"dense": [0.5, 0.5, 0.0, 0.0], "sparse": {}, "colbert": []}
        with patch.object(memsom_embed, "encode_query", lambda t: dict(canned)):
            memsom_retrieve.vector_search(self.conn, "q")
        self.assertIsNone(memsom_retrieve.last_query_fallback(self.conn),
                          "a successful cold start must clear the degraded trail")
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])


# ---------------------------------------------------------------------------
# 2. idle keep-alive knob evicts the in-process model
# ---------------------------------------------------------------------------

class TestIdleTtlEviction(Base):
    def test_knob_default_and_resolution(self):
        self.assertEqual(memsom_embed.bge_idle_ttl(), 60)
        os.environ["MEMDAG_BGE_IDLE_TTL"] = "5"
        self.assertEqual(memsom_embed.bge_idle_ttl(), 5)
        os.environ["MEMDAG_BGE_IDLE_TTL"] = "0"          # 0 disables eviction
        self.assertEqual(memsom_embed.bge_idle_ttl(), 0)

    def test_evicts_after_the_window(self):
        """The knob controls the keep-alive: with a tiny TTL the model goes cold
        after the window. Time is injected, not slept."""
        os.environ["MEMDAG_BGE_IDLE_TTL"] = "2"
        memsom_embed._MODEL = object()                    # pretend a model is resident
        memsom_embed._LAST_USE = 100.0
        # 1.5 s idle < 2 s TTL -> stays warm
        self.assertFalse(memsom_embed._evict_if_idle(now=101.5))
        self.assertIsNotNone(memsom_embed._MODEL)
        # 3 s idle >= 2 s TTL -> evicted, VRAM handed back
        self.assertTrue(memsom_embed._evict_if_idle(now=103.0))
        self.assertIsNone(memsom_embed._MODEL)

    def test_changing_the_knob_changes_the_behavior(self):
        """Same idle gap, different TTL -> different verdict."""
        memsom_embed._MODEL = object()
        memsom_embed._LAST_USE = 0.0
        os.environ["MEMDAG_BGE_IDLE_TTL"] = "3600"        # long window -> keep warm
        self.assertFalse(memsom_embed._evict_if_idle(now=10.0))
        self.assertIsNotNone(memsom_embed._MODEL)
        os.environ["MEMDAG_BGE_IDLE_TTL"] = "1"           # short window -> evict
        self.assertTrue(memsom_embed._evict_if_idle(now=10.0))
        self.assertIsNone(memsom_embed._MODEL)

    def test_ttl_zero_never_evicts(self):
        os.environ["MEMDAG_BGE_IDLE_TTL"] = "0"
        memsom_embed._MODEL = object()
        memsom_embed._LAST_USE = 0.0
        self.assertFalse(memsom_embed._evict_if_idle(now=1_000_000.0))
        self.assertIsNotNone(memsom_embed._MODEL)


# ---------------------------------------------------------------------------
# 3. a GENUINE cold-start failure still degrades
# ---------------------------------------------------------------------------

class TestGenuineFailureStillDegrades(Base):
    def test_unimportable_erroring_encoder_degrades_and_signals(self):
        self.set_backend("bge-m3")
        self.put_vec(self.add("a bge vector"), [0.1, 0.2, 0.3, 0.4])
        # Encoder cannot produce a vector by ANY path.
        with patch.object(memsom_embed, "encode_query", lambda t: None):
            self.assertEqual(memsom_retrieve.vector_search(self.conn, "q"), [])
        trail = memsom_retrieve.last_query_fallback(self.conn)
        self.assertIsNotNone(trail)
        self.assertIn("bge-m3", trail["reason"])
        self.assertTrue(any("query encoder unreachable" in l
                            for l in memsom_retrieve.retrieval_warnings(self.conn)))

    def test_cold_start_is_bounded_never_hangs(self):
        """A cold start that would take forever returns None inside the bound,
        so the caller degrades instead of hanging the prompt."""
        def _slow(_t):
            time.sleep(30)   # far longer than the bound below
            return {"dense": [1, 0, 0, 0], "sparse": {}, "colbert": []}

        with patch.object(memsom_embed, "encode_query", _slow):
            t0 = time.monotonic()
            enc = memsom_embed.cold_start_encode_query("q", timeout_s=0.3)
            elapsed = time.monotonic() - t0
        self.assertIsNone(enc)
        self.assertLess(elapsed, 5.0, "cold start did not respect its bound")


# ---------------------------------------------------------------------------
# 4. the knob is a first-class tunable (panel-readable, mspanel-readable)
# ---------------------------------------------------------------------------

class TestKnobRegistered(unittest.TestCase):
    def test_registered_with_default_and_bounds(self):
        self.assertIn("retrieval.bge_idle_ttl", memsom_tuning.REGISTRY)
        knob = memsom_tuning.REGISTRY["retrieval.bge_idle_ttl"]
        self.assertEqual(knob.default, 60)
        self.assertEqual(knob.type, int)
        self.assertEqual(knob.bounds, (0, 86400))
        self.assertEqual(knob.feature, "retrieval.bge")

    def test_visible_in_as_json_for_the_panel(self):
        j = memsom_tuning.as_json()
        self.assertIn("retrieval.bge_idle_ttl", j)
        self.assertEqual(j["retrieval.bge_idle_ttl"]["default"], 60)

    def test_out_of_bounds_env_falls_back_to_default(self):
        prev = os.environ.get("MEMDAG_BGE_IDLE_TTL")
        try:
            os.environ["MEMDAG_BGE_IDLE_TTL"] = "999999999"   # > 86400
            memsom_tuning._clear_warned("retrieval.bge_idle_ttl")
            self.assertEqual(memsom_embed.bge_idle_ttl(), 60)
        finally:
            if prev is None:
                os.environ.pop("MEMDAG_BGE_IDLE_TTL", None)
            else:
                os.environ["MEMDAG_BGE_IDLE_TTL"] = prev


if __name__ == "__main__":
    unittest.main()
