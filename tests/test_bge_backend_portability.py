#!/usr/bin/env python3
"""Portability of the bge-m3 backend: a fresh clone with NO supervisor and NO
torch must degrade cleanly, never hang, never crash.

Covers the encode-path dispatch (embed._dispatch_encode / bge_usable), the
LOCAL supervisor HTTP client (bge_client), and the dense/sparse/colbert signal
toggles. CI-SAFE: torch/FlagEmbedding are never imported — bge_available is
patched and the HTTP layer (memsom.effects.net.fetch) is mocked.

Run:
  python -W error::DeprecationWarning -m unittest test_bge_backend_portability -v
"""
import base64
import contextlib
import io
import os
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.retrieval import embed as memsom_embed
from memsom.retrieval import bge_client
from memsom.retrieval import retrieve as memsom_retrieve
from memsom.effects import net as memsom_net


def _embed_response(dense, sparse, colbert):
    """Build a bge_service /embed JSON reply (colbert as fp16 LE base64)."""
    import json
    n = len(colbert)
    dim = len(colbert[0]) if n else 0
    flat = [float(x) for row in colbert for x in row]
    b64 = base64.b64encode(struct.pack(f"<{len(flat)}e", *flat)).decode("ascii")
    return json.dumps({
        "dense": [[float(x) for x in dense]],
        "sparse": [{str(k): float(v) for k, v in sparse.items()}],
        "colbert_b64": [b64],
        "colbert_shape": [[n, dim]],
    }).encode("utf-8")


class EnvBase(unittest.TestCase):
    """Isolate every bge env knob and reset the module-level health cache."""

    _KEYS = ("MEMDAG_EMBED_BACKEND", "MEMDAG_BGE_URL", "MEMDAG_BGE_ENCODE_VIA",
             "MEMDAG_BGE_DENSE", "MEMDAG_BGE_SPARSE", "MEMDAG_BGE_COLBERT")

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self._KEYS}
        bge_client._reset_probe()
        memsom_embed._WARNED_FALLBACK = False

    def tearDown(self):
        for k in self._KEYS:
            os.environ.pop(k, None)
            if self._saved.get(k) is not None:
                os.environ[k] = self._saved[k]
        bge_client._reset_probe()
        memsom_embed._WARNED_FALLBACK = False


# ---------------------------------------------------------------------------
# bge_client: parse + fail-fast health
# ---------------------------------------------------------------------------

class TestBgeClient(EnvBase):
    def test_health_unreachable_is_false_not_hang(self):
        """A refused/erroring supervisor -> available False, bounded by the probe
        timeout (fetch is called with the short HEALTH_TIMEOUT)."""
        captured = {}

        def fake_fetch(url, **kw):
            captured["timeout"] = kw.get("timeout")
            raise memsom_net.NetworkError("connection refused")

        with patch("memsom.effects.net.fetch", side_effect=fake_fetch):
            self.assertFalse(bge_client.bge_http_available(force=True))
        self.assertEqual(captured["timeout"], bge_client.HEALTH_TIMEOUT)

    def test_unset_url_is_unavailable_without_network(self):
        os.environ["MEMDAG_BGE_URL"] = ""
        with patch("memsom.effects.net.fetch",
                   side_effect=AssertionError("must not touch the network")):
            self.assertFalse(bge_client.configured())
            self.assertFalse(bge_client.bge_http_available(force=True))

    def test_encode_http_parses_all_three_signals(self):
        body = _embed_response([0.1, 0.2], {"7": 0.5, "9": 0.25},
                               [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])

        def fake_fetch(url, **kw):
            return body

        with patch("memsom.effects.net.fetch", side_effect=fake_fetch):
            enc = bge_client.encode_http("hello")
        self.assertEqual(enc["dense"], [0.1, 0.2])
        self.assertEqual(enc["sparse"], {"7": 0.5, "9": 0.25})
        self.assertEqual(enc["colbert"], [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])

    def test_encode_http_none_on_error(self):
        with patch("memsom.effects.net.fetch",
                   side_effect=memsom_net.NetworkError("boom")):
            self.assertIsNone(bge_client.encode_http("hello"))


# ---------------------------------------------------------------------------
# Encode-path dispatch: supervisor preferred, in-process fallback, BM25 degrade
# ---------------------------------------------------------------------------

class TestEncodeDispatch(EnvBase):
    def test_no_supervisor_uses_inprocess(self):
        """/health down -> the supervisor HTTP path is skipped and in-process
        torch is used; encode_http is never called."""
        os.environ["MEMDAG_BGE_ENCODE_VIA"] = "auto"
        http = MagicMock()
        with patch.object(bge_client, "bge_http_available", lambda force=False: False), \
             patch.object(bge_client, "encode_http", http), \
             patch.object(memsom_embed, "_encode",
                          lambda t: {"dense": [1.0], "sparse": {}, "colbert": [[1.0]]}):
            enc = memsom_embed.encode_doc("x")
        self.assertEqual(enc["dense"], [1.0])
        http.assert_not_called()

    def test_supervisor_preferred_when_healthy(self):
        os.environ["MEMDAG_BGE_ENCODE_VIA"] = "auto"
        inproc = MagicMock(side_effect=AssertionError("must not use torch when supervisor is up"))
        with patch.object(bge_client, "bge_http_available", lambda force=False: True), \
             patch.object(bge_client, "encode_http",
                          lambda t: {"dense": [9.0], "sparse": {}, "colbert": []}), \
             patch.object(memsom_embed, "_encode", inproc):
            enc = memsom_embed.encode_query("q")
        self.assertEqual(enc["dense"], [9.0])

    def test_inprocess_mode_never_probes_supervisor(self):
        os.environ["MEMDAG_BGE_ENCODE_VIA"] = "inprocess"
        with patch("memsom.effects.net.fetch",
                   side_effect=AssertionError("inprocess must not touch the network")), \
             patch.object(memsom_embed, "_encode",
                          lambda t: {"dense": [2.0], "sparse": {}, "colbert": []}):
            enc = memsom_embed.encode_doc("x")
        self.assertEqual(enc["dense"], [2.0])

    def test_no_path_at_all_degrades_to_none_with_one_warning(self):
        """No supervisor AND no torch (import error) -> None + exactly one warning."""
        os.environ["MEMDAG_BGE_ENCODE_VIA"] = "auto"

        def no_torch(_t):
            raise ModuleNotFoundError("No module named 'FlagEmbedding'")

        buf = io.StringIO()
        with patch.object(bge_client, "bge_http_available", lambda force=False: False), \
             patch.object(memsom_embed, "_encode", no_torch), \
             contextlib.redirect_stderr(buf):
            self.assertIsNone(memsom_embed.encode_doc("x"))
            self.assertIsNone(memsom_embed.encode_query("y"))
        self.assertEqual(buf.getvalue().count("BGE-M3 backend requested"), 1)


# ---------------------------------------------------------------------------
# bge_usable: torch OR supervisor, torch-first (no network probe when torch present)
# ---------------------------------------------------------------------------

class TestBgeUsable(EnvBase):
    def test_torch_present_no_network(self):
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch("memsom.effects.net.fetch",
                   side_effect=AssertionError("must not probe when torch is present")):
            self.assertTrue(memsom_embed.bge_usable())

    def test_torch_absent_supervisor_up(self):
        with patch.object(memsom_embed, "bge_available", lambda: False), \
             patch.object(bge_client, "bge_http_available", lambda force=False: True):
            self.assertTrue(memsom_embed.bge_usable())

    def test_both_absent_is_false(self):
        with patch.object(memsom_embed, "bge_available", lambda: False), \
             patch.object(bge_client, "bge_http_available", lambda force=False: False):
            self.assertFalse(memsom_embed.bge_usable())

    def test_inprocess_mode_ignores_supervisor(self):
        os.environ["MEMDAG_BGE_ENCODE_VIA"] = "inprocess"
        with patch.object(memsom_embed, "bge_available", lambda: False), \
             patch.object(bge_client, "bge_http_available", lambda force=False: True):
            self.assertFalse(memsom_embed.bge_usable())


# ---------------------------------------------------------------------------
# Signal toggles: store writes only enabled types + purges the rest; query gates
# ---------------------------------------------------------------------------

class TestSignalToggles(EnvBase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["MEMDAG_DB"] = str(Path(self.tmp.name) / "sub" / "t.db")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        self.conn = memsom.get_connection()
        memsom_retrieve.migrate(self.conn)
        memsom_embed.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()
        super().tearDown()

    def _count(self, table, nid):
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE node_id = ?", (nid,)).fetchone()[0]

    def _index(self, nid):
        enc = {"dense": [1.0, 0.0], "sparse": {"1": 1.0}, "colbert": [[1.0, 0.0]]}
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_doc", lambda t: dict(enc)):
            memsom_retrieve.index_node(self.conn, nid)

    def test_all_on_writes_three(self):
        nid = memsom.insert_node(self.conn, "alpha bravo charlie", "user")
        self.conn.commit()
        self._index(nid)
        self.assertEqual(self._count("embeddings", nid), 1)
        self.assertEqual(self._count("sparse_vecs", nid), 1)
        self.assertEqual(self._count("colbert_vecs", nid), 1)

    def test_sparse_off_skips_and_purges(self):
        nid = memsom.insert_node(self.conn, "alpha bravo charlie", "user")
        self.conn.commit()
        self._index(nid)                       # start with all three present
        self.assertEqual(self._count("sparse_vecs", nid), 1)
        os.environ["MEMDAG_BGE_SPARSE"] = "0"   # toggle sparse off, reindex
        self._index(nid)
        self.assertEqual(self._count("sparse_vecs", nid), 0)   # purged
        self.assertEqual(self._count("embeddings", nid), 1)    # others intact
        self.assertEqual(self._count("colbert_vecs", nid), 1)

    def test_sparse_search_gated_off(self):
        nid = memsom.insert_node(self.conn, "alpha bravo charlie", "user")
        self.conn.commit()
        self._index(nid)
        enc_q = {"dense": [1.0, 0.0], "sparse": {"1": 1.0}, "colbert": [[1.0, 0.0]]}
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", lambda t: dict(enc_q)):
            on = memsom_retrieve.sparse_search(self.conn, "alpha", k=5)
            os.environ["MEMDAG_BGE_SPARSE"] = "0"
            off = memsom_retrieve.sparse_search(self.conn, "alpha", k=5)
        self.assertTrue(on)          # sparse contributes when on
        self.assertEqual(off, [])    # and nothing when off

    def test_vector_search_gated_off(self):
        """Dense knob off -> vector_search returns [] on the bge path (and never
        even encodes the query)."""
        nid = memsom.insert_node(self.conn, "alpha bravo charlie", "user")
        self.conn.commit()
        self._index(nid)
        enc_q = {"dense": [1.0, 0.0], "sparse": {"1": 1.0}, "colbert": [[1.0, 0.0]]}
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", lambda t: dict(enc_q)):
            on = memsom_retrieve.vector_search(self.conn, "alpha", k=5)
            os.environ["MEMDAG_BGE_DENSE"] = "0"
            with patch.object(memsom_embed, "encode_query",
                              side_effect=AssertionError("must not encode when dense off")):
                off = memsom_retrieve.vector_search(self.conn, "alpha", k=5)
        self.assertTrue(on)          # dense contributes when on
        self.assertEqual(off, [])    # short-circuits (no encode) when off

    def test_colbert_rerank_gated_off(self):
        """ColBERT knob off -> colbert_rerank passes the candidates through with
        0.0 scores (no re-ranking) and never encodes the query."""
        nid = memsom.insert_node(self.conn, "alpha bravo charlie", "user")
        self.conn.commit()
        self._index(nid)
        enc_q = {"dense": [1.0, 0.0], "sparse": {"1": 1.0}, "colbert": [[1.0, 0.0]]}
        with patch.object(memsom_embed, "bge_available", lambda: True), \
             patch.object(memsom_embed, "encode_query", lambda t: dict(enc_q)):
            on = memsom_retrieve.colbert_rerank(self.conn, "alpha", [nid])
            os.environ["MEMDAG_BGE_COLBERT"] = "0"
            with patch.object(memsom_embed, "encode_query",
                              side_effect=AssertionError("must not encode when colbert off")):
                off = memsom_retrieve.colbert_rerank(self.conn, "alpha", [nid])
        self.assertTrue(any(score != 0.0 for _, score in on))  # real MaxSim when on
        self.assertEqual(off, [(nid, 0.0)])                     # pass-through when off


if __name__ == "__main__":
    unittest.main()
