#!/usr/bin/env python3
"""Regression tests for the split-backend / dense-dark bug (2026-08-31..09-04).

WHAT HAPPENED (measured on the live PC store, 2026-09-04): 643 embeddable
live nodes carried vectors under TWO models -- 524 bge-m3 (1024-dim) and 119
nomic-embed-text (768-dim). The MCP servers ran with MEMDAG_EMBED_BACKEND=
bge-m3; the Stop-hook importer ran with no such var and so wrote under
embed.py's compiled-in default (ollama/nomic). vector_search reads
`WHERE model = <active>` (the dim-collision fix), so every node the hook had
re-embedded since 08-31 -- the freshest memories -- was invisible to dense
retrieval, and nothing anywhere said so: retrieval_degraded was empty (the
embeds SUCCEEDED, just under the wrong model), query_log was empty, and the
prompt hook's log showed BM25 for 1,913/1,913 prompts with no line flagging it.

THE FIX under test:
  1. the store PINS its backend (retrieval_meta) on the first vector write and
     on every full reindex; embed.backend(conn) resolves override > env > pin
     > default, so an env-less process adopts the store's backend;
  2. embedding_coverage / retrieval_warnings measure dense reach over live
     nodes and produce the one-line AI-facing signal;
  3. a query-side encode fallback leaves a trail (sentinel row) and a warning;
  4. the prompt hook injects the signal (and flags a warm-endpoint miss on a
     dense store) even when nothing else matched;
  5. a surviving warm server re-adopts the endpoint file a sibling removed.

RED-BEFORE / GREEN-AFTER: every test here fails on the pre-fix tree either by
AttributeError (the functions did not exist) or, for test_env_unset_process_
writes_under_the_pinned_backend, by asserting model == 'bge-m3' where the old
code wrote 'nomic-embed-text'. test_live_store_is_single_backend_and_fully_
covered is the operator gate: pointed at the 2026-09-04 backup it fails
(524/643 dense-dark); at the reindexed live store it passes.

CI-SAFE: no FlagEmbedding/torch/Ollama is ever reached -- every encoder is
patched to a canned vector or to a failure.

Run:
  python -m pytest tests/test_retrieval_coverage.py -q
"""

import io
import json
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import memsom
from memsom import tuning as memsom_tuning
from memsom.interface import cli as memsom_cli
from memsom.interface import features as memsom_features
from memsom.interface import prompt_hook as ph
from memsom.retrieval import embed as memsom_embed
from memsom.retrieval import retrieve as memsom_retrieve
from memsom.retrieval import warm

BGE = memsom_embed.BGE_MODEL_NAME
NOMIC = memsom_retrieve.DEFAULT_EMBED_MODEL

_CANNED_BGE = {"dense": [0.3, 0.4, 0.5, 0.6], "sparse": {"1": 0.5},
               "colbert": [[0.1, 0.2, 0.3, 0.4]]}


def _blob(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


class Base(unittest.TestCase):
    """Throwaway store per test; MEMDAG_EMBED_BACKEND managed here (the
    suite-wide bm25 pin is restored on teardown); no in-process override."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "store" / "test.db"
        self._env = {k: os.environ.get(k) for k in ("MEMDAG_DB", "MEMDAG_EMBED_BACKEND")}
        os.environ["MEMDAG_DB"] = str(self.db)
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        memsom_tuning.clear_override("embed.backend")
        memsom_retrieve._LAST_QUERY_FALLBACK = None
        self.conn = memsom.get_connection()
        memsom_cli.migrate_all(self.conn)

    def tearDown(self):
        self.conn.close()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        memsom_tuning.clear_override("embed.backend")
        self.tmp.cleanup()

    def add(self, content, channel="user"):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])

    def put_vec(self, nid, model, vec=(0.1, 0.2, 0.3, 0.4)):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec) VALUES (?,?,?,?)",
                (nid, model, len(vec), _blob(list(vec))))

    def set_backend(self, name):
        os.environ["MEMDAG_EMBED_BACKEND"] = name


# ---------------------------------------------------------------------------
# 1. coverage + the split gate
# ---------------------------------------------------------------------------

class TestCoverage(Base):
    def test_single_backend_store_is_fully_covered_and_silent(self):
        self.set_backend("bge-m3")
        for i in range(3):
            self.put_vec(self.add(f"node {i}"), BGE)
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertEqual((cov["total"], cov["covered"], cov["dark"], cov["split"]),
                         (3, 3, 0, False))
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_more_than_one_model_over_live_nodes_is_a_split_and_dark(self):
        """THE gate: live embeddable nodes under >1 model = the bug shape."""
        self.set_backend("bge-m3")
        self.put_vec(self.add("old bge node"), BGE)
        self.put_vec(self.add("another bge node"), BGE)
        self.put_vec(self.add("fresh node re-embedded by an env-less process"), NOMIC)
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertTrue(cov["split"], cov)
        self.assertEqual(cov["by_model"], {BGE: 2, NOMIC: 1})
        self.assertEqual((cov["covered"], cov["dark"]), (2, 1))
        lines = memsom_retrieve.retrieval_warnings(self.conn)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("RETRIEVAL DEGRADED", lines[0])
        self.assertIn("1/3", lines[0])
        self.assertIn("dense-dark", lines[0])
        self.assertIn("memsom reindex", lines[0])
        self.assertIn(f"{NOMIC}=1", lines[0])

    def test_dark_means_the_active_reader_cannot_see_it(self):
        """The same split read from the OTHER side: active=ollama makes the
        bge rows the dark ones (this is what `memsom features` showed live
        in an env-less shell: 524/643 dark)."""
        self.set_backend("ollama")
        self.put_vec(self.add("bge a"), BGE)
        self.put_vec(self.add("bge b"), BGE)
        self.put_vec(self.add("nomic c"), NOMIC)
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertEqual((cov["covered"], cov["dark"]), (1, 2))

    def test_node_with_no_vector_at_all_is_dark(self):
        self.set_backend("ollama")
        self.put_vec(self.add("has a vector"), NOMIC)
        self.add("never embedded")
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertEqual((cov["total"], cov["covered"], cov["dark"], cov["split"]),
                         (2, 1, 1, False))
        self.assertTrue(memsom_retrieve.retrieval_warnings(self.conn))

    def test_only_live_embeddable_nodes_count(self):
        """A stale foreign-model row on a tombstoned node, and an agent-derived
        node, never make a healthy store look split."""
        self.set_backend("bge-m3")
        self.put_vec(self.add("live"), BGE)
        dead = self.add("dead")
        self.put_vec(dead, NOMIC)
        with self.conn:
            self.conn.execute("UPDATE nodes SET tombstoned = 1 WHERE id = ?", (dead,))
        self.add("derived", channel="agent-derived")
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertEqual((cov["total"], cov["covered"], cov["dark"], cov["split"]),
                         (1, 1, 0, False))
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_bm25_store_expects_no_vectors(self):
        self.set_backend("bm25")
        self.put_vec(self.add("leftover a"), BGE)
        self.put_vec(self.add("leftover b"), NOMIC)
        self.add("no vector")
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertEqual(cov["dark"], 0)
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_coverage_is_read_only(self):
        self.set_backend("bge-m3")
        self.put_vec(self.add("x"), BGE)
        ro = memsom.get_connection(self.db, read_only=True)
        try:
            cov = memsom_retrieve.embedding_coverage(ro)
            self.assertEqual(cov["covered"], 1)
            self.assertEqual(memsom_retrieve.retrieval_warnings(ro), [])
            self.assertIsNone(memsom_retrieve.pinned_backend(ro))
        finally:
            ro.close()

    def test_features_report_the_split_as_degraded(self):
        self.set_backend("bge-m3")
        self.put_vec(self.add("bge"), BGE)
        self.put_vec(self.add("nomic"), NOMIC)
        with patch.object(memsom_embed, "bge_available", return_value=True):
            st = memsom_features._retrieval_bge(self.conn)
        self.assertEqual(st["state"], "degraded", st)
        self.assertIn("dense-dark", st["detail"])
        self.set_backend("ollama")
        st = memsom_features._retrieval_dense(self.conn)
        self.assertEqual(st["state"], "degraded", st)
        self.assertIn("1/2", st["detail"])

    @unittest.skipUnless(os.environ.get("MEMSOM_LIVE_STORE_CHECK"),
                         "operator gate: set MEMSOM_LIVE_STORE_CHECK=<path to a store>")
    def test_live_store_is_single_backend_and_fully_covered(self):
        """Operator gate against a REAL store, read-only. Red on the
        2026-09-04 backup (split, 524/643 dark); green after `memsom reindex`."""
        path = os.environ["MEMSOM_LIVE_STORE_CHECK"]
        conn = memsom.get_connection(path, read_only=True)
        try:
            cov = memsom_retrieve.embedding_coverage(conn)
            pinned = memsom_retrieve.pinned_backend(conn)
        finally:
            conn.close()
        self.assertFalse(cov["split"], f"live nodes span >1 embedding model: {cov}")
        self.assertEqual(cov["dark"], 0,
                         f"{cov['dark']}/{cov['total']} live nodes are dense-dark: {cov}")
        self.assertIsNotNone(pinned, "store has no pinned backend")
        self.assertEqual(cov["backend"], pinned, cov)


# ---------------------------------------------------------------------------
# 2. the query-encoder fallback trail
# ---------------------------------------------------------------------------

class TestQueryFallbackTrail(Base):
    def test_ollama_query_encoder_unreachable_leaves_a_trail(self):
        self.set_backend("ollama")
        self.put_vec(self.add("a stored vector"), NOMIC)

        def boom(text, timeout=None):
            raise RuntimeError("connection refused")

        with patch.object(memsom_retrieve, "_call_ollama_embed", boom):
            self.assertEqual(memsom_retrieve.vector_search(self.conn, "anything"), [])
        trail = memsom_retrieve.last_query_fallback(self.conn)
        self.assertIsNotNone(trail)
        self.assertIn("connection refused", trail["reason"])
        self.assertIn("ollama", trail["reason"])
        self.assertLessEqual(trail["age_s"], 60)
        self.assertEqual(memsom_retrieve.last_query_fallback_reason(), trail["reason"])
        # the sentinel is a trail, never a node in the re-index queue
        self.assertEqual(memsom_retrieve.degraded_nodes(self.conn), [])
        self.assertEqual(memsom_features._degraded_count(self.conn), 0)
        lines = memsom_retrieve.retrieval_warnings(self.conn)
        self.assertTrue(any("query encoder unreachable" in l for l in lines), lines)
        st = memsom_features._retrieval_dense(self.conn)
        self.assertEqual(st["state"], "degraded", st)

    def test_bge_without_an_encode_path_leaves_a_trail(self):
        self.set_backend("bge-m3")
        self.put_vec(self.add("a bge vector"), BGE)
        with patch.object(memsom_embed, "bge_usable", return_value=False):
            self.assertEqual(memsom_retrieve.vector_search(self.conn, "q"), [])
        trail = memsom_retrieve.last_query_fallback(self.conn)
        self.assertIsNotNone(trail)
        self.assertIn("bge-m3", trail["reason"])

    def test_healthy_encode_leaves_no_trail(self):
        """CONTROL: a store whose encoder answers has no sentinel, no
        in-process reason and no warning."""
        self.set_backend("ollama")
        nid = self.add("the mesh firewall must be opened in nebula and ufw")
        with patch.object(memsom_retrieve, "_call_ollama_embed",
                          lambda text, timeout=None: [0.1, 0.2, 0.3, 0.4]):
            memsom_retrieve.index_node(self.conn, nid)
            hits = memsom_retrieve.vector_search(self.conn, "firewall")
        self.assertEqual([h[0] for h in hits], [nid])
        self.assertIsNone(memsom_retrieve.last_query_fallback(self.conn))
        self.assertIsNone(memsom_retrieve.last_query_fallback_reason())
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_recovery_clears_the_in_process_reason_but_keeps_the_trail(self):
        self.set_backend("ollama")
        self.put_vec(self.add("v"), NOMIC, (0.1, 0.2, 0.3, 0.4))
        calls = {"n": 0}

        def flaky(text, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("down")
            return [0.1, 0.2, 0.3, 0.4]

        with patch.object(memsom_retrieve, "_call_ollama_embed", flaky):
            memsom_retrieve.vector_search(self.conn, "q")
            self.assertIsNotNone(memsom_retrieve.last_query_fallback_reason())
            memsom_retrieve.vector_search(self.conn, "q")
        self.assertIsNone(memsom_retrieve.last_query_fallback_reason())
        self.assertIsNotNone(memsom_retrieve.last_query_fallback(self.conn),
                             "the stored trail must survive recovery")

    def test_readonly_connection_never_raises(self):
        self.set_backend("ollama")
        self.put_vec(self.add("v"), NOMIC)
        ro = memsom.get_connection(self.db, read_only=True)
        try:
            with patch.object(memsom_retrieve, "_call_ollama_embed",
                              side_effect=RuntimeError("down")):
                self.assertEqual(memsom_retrieve.vector_search(ro, "q"), [])
            self.assertIsNotNone(memsom_retrieve.last_query_fallback_reason())
            self.assertIsNone(memsom_retrieve.last_query_fallback(ro))
        finally:
            ro.close()

    def test_retrieve_cli_prints_the_signal_on_stdout(self):
        """The MCP `retrieve` tool returns this command's stdout: the warning
        has to be THERE, not on stderr."""
        self.set_backend("ollama")
        nid = self.add("the nebula firewall rule for the mesh")
        memsom_retrieve.index_node(self.conn, nid)  # BM25 half; embed fails -> queued
        out = io.StringIO()
        with patch.object(memsom_retrieve, "_call_ollama_embed",
                          side_effect=RuntimeError("refused")), \
                redirect_stdout(out), redirect_stderr(io.StringIO()):
            memsom_retrieve.main(["retrieve", "nebula firewall"])
        self.assertIn("RETRIEVAL DEGRADED", out.getvalue())
        self.assertIn("query encoder unreachable", out.getvalue())


# ---------------------------------------------------------------------------
# 3. the pin: an env-less process adopts the store's backend
# ---------------------------------------------------------------------------

class TestPin(Base):
    def _index_under_bge(self, nid):
        with patch.object(memsom_embed, "bge_usable", return_value=True), \
                patch.object(memsom_embed, "encode_doc", return_value=_CANNED_BGE):
            self.assertTrue(memsom_retrieve.index_node(self.conn, nid))

    def test_first_vector_write_pins_the_store(self):
        self.assertIsNone(memsom_retrieve.pinned_backend(self.conn))
        self.set_backend("bge-m3")
        self._index_under_bge(self.add("first"))
        self.assertEqual(memsom_retrieve.pinned_backend(self.conn), "bge-m3")

    def test_env_unset_process_writes_under_the_pinned_backend(self):
        """THE regression. Pre-fix: env unset -> DEFAULT_BACKEND ('ollama') ->
        a nomic row lands in a bge-m3 store and the bge reader never sees it."""
        self.set_backend("bge-m3")
        self._index_under_bge(self.add("indexed by the MCP server (env set)"))
        os.environ.pop("MEMDAG_EMBED_BACKEND", None)   # the Stop-hook importer
        self.assertEqual(memsom_embed.backend(self.conn), "bge-m3")
        self.assertEqual(memsom_embed.active_model_name(self.conn), BGE)
        self.assertEqual(memsom_embed.backend(), memsom_embed.DEFAULT_BACKEND,
                         "without a conn the documented env-or-default answer stands")
        nid = self.add("re-imported by the Stop hook (env unset)")
        with patch.object(memsom_embed, "bge_usable", return_value=True), \
                patch.object(memsom_embed, "encode_doc", return_value=_CANNED_BGE), \
                patch.object(memsom_retrieve, "_call_ollama_embed",
                             side_effect=AssertionError("nomic path taken")):
            memsom_retrieve.index_node(self.conn, nid)
        model = self.conn.execute("SELECT model FROM embeddings WHERE node_id = ?",
                                  (nid,)).fetchone()[0]
        self.assertEqual(model, BGE)
        cov = memsom_retrieve.embedding_coverage(self.conn)
        self.assertEqual((cov["split"], cov["dark"]), (False, 0), cov)

    def test_explicit_env_beats_the_pin_but_warns_once(self):
        memsom_retrieve.pin_backend(self.conn, "bge-m3")
        self.set_backend("ollama")
        memsom_embed._WARNED_PIN_MISMATCH = False
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(memsom_embed.backend(self.conn), "ollama")
            memsom_embed.backend(self.conn)
        self.assertEqual(err.getvalue().count("[memsom] embed.backend="), 1, err.getvalue())

    def test_override_beats_everything(self):
        memsom_retrieve.pin_backend(self.conn, "bge-m3")
        self.set_backend("ollama")
        memsom_tuning.override("embed.backend", "bm25")
        self.assertEqual(memsom_embed.backend(self.conn), "bm25")

    def test_bm25_never_warns_about_the_pin(self):
        """The prompt hook pins itself to bm25 (no model load) on every prompt;
        bm25 writes nothing, so a pinned dense store is not a mismatch."""
        memsom_retrieve.pin_backend(self.conn, "bge-m3")
        memsom_embed._WARNED_PIN_MISMATCH = False
        err = io.StringIO()
        with redirect_stderr(err):
            memsom_tuning.override("embed.backend", "bm25")
            self.assertEqual(memsom_embed.backend(self.conn), "bm25")
            memsom_tuning.clear_override("embed.backend")
            self.set_backend("bm25")
            self.assertEqual(memsom_embed.backend(self.conn), "bm25")
        self.assertEqual(err.getvalue(), "")

    def test_a_bad_pin_value_is_ignored(self):
        memsom_retrieve.pin_backend(self.conn, "not-a-backend")
        self.assertEqual(memsom_embed.backend(self.conn), memsom_embed.DEFAULT_BACKEND)

    def test_reindex_cli_pins_and_reports_coverage(self):
        self.set_backend("ollama")
        self.add("one node to index")
        out = io.StringIO()
        with patch.object(memsom_retrieve, "_call_ollama_embed",
                          lambda text, timeout=None: [0.1, 0.2, 0.3, 0.4]), \
                redirect_stdout(out):
            memsom_retrieve.main(["reindex"])
        self.assertIn("backend pinned: ollama", out.getvalue())
        self.assertIn("dense coverage 1/1", out.getvalue())
        self.assertNotIn("RETRIEVAL DEGRADED", out.getvalue())
        self.assertEqual(memsom_retrieve.pinned_backend(self.conn), "ollama")


# ---------------------------------------------------------------------------
# 4. the prompt hook injects the signal
# ---------------------------------------------------------------------------

_SPLIT_LINE = "⚠️ RETRIEVAL DEGRADED: 1/3 memory nodes are dense-dark (test)"


class TestHookSignal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = Path(self.tmp.name)
        self.params = {"mode": "inject", "floor": 0.35, "deadline_ms": 800, "log_max_mb": 20}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, health, source, hits=()):
        return ph.run_prompt_hook(
            {"prompt": "what did we decide about the mesh firewall"},
            memory_dir=self.mem, params=self.params,
            query_fn=lambda prompt, k=3, clearance="topsecret", deadline_ms=800: (list(hits), source),
            health_fn=lambda: health)

    def _log(self):
        p = ph.log_path(self.mem)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]

    def test_split_store_and_warm_miss_are_injected_even_with_no_hits(self):
        out = self._run(("bge-m3", [_SPLIT_LINE]), "bm25")
        self.assertIsNotNone(out, "a degraded store must not be silent")
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn(_SPLIT_LINE, ctx)
        self.assertIn("warm endpoint down", ctx)
        self.assertIn("BM25-only", ctx)
        rec = self._log()[-1]
        self.assertEqual(len(rec["degraded"]), 2, rec)
        self.assertTrue(rec["injected"])

    def test_healthy_store_served_warm_is_silent(self):
        self.assertIsNone(self._run(("bge-m3", []), "warm"))
        self.assertEqual(self._log()[-1]["degraded"], [])

    def test_bm25_store_never_flags_the_warm_miss(self):
        self.assertIsNone(self._run(("bm25", []), "bm25"))
        self.assertIsNone(self._run(("", []), "timeout"))

    def test_degraded_lines_table(self):
        self.assertEqual(ph.degraded_lines("warm", "bge-m3", []), [])
        self.assertEqual(len(ph.degraded_lines("timeout", "ollama", [])), 1)
        self.assertEqual(len(ph.degraded_lines("error", "bge-m3", ["x"])), 2)
        self.assertEqual(ph.degraded_lines("bm25", "bm25", []), [])

    def test_warning_rides_after_the_memory_block(self):
        hits = [{"id": 1, "stem": "reference_nebula_mesh_firewall",
                 "label": "reference_nebula_mesh_firewall",
                 "hook": "mesh service = two firewalls", "score": 0.9}]
        out = self._run(("bge-m3", [_SPLIT_LINE]), "bm25", hits)
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertLess(ctx.index("reference_nebula_mesh_firewall"), ctx.index(_SPLIT_LINE))


class TestHookStoreHealth(Base):
    """store_health() reads the REAL store, read-only, before any override."""

    def test_reads_the_split_and_the_configured_backend(self):
        self.set_backend("bge-m3")
        self.put_vec(self.add("bge"), BGE)
        self.put_vec(self.add("nomic"), NOMIC)
        backend, lines = ph.store_health(self.db)
        self.assertEqual(backend, "bge-m3")
        self.assertEqual(len(lines), 1)
        self.assertIn("1/2", lines[0])

    def test_env_less_hook_process_sees_the_pinned_backend(self):
        memsom_retrieve.pin_backend(self.conn, "bge-m3")
        self.put_vec(self.add("bge"), BGE)
        backend, lines = ph.store_health(self.db)
        self.assertEqual((backend, lines), ("bge-m3", []))

    def test_missing_store_is_unknown_not_an_error(self):
        self.assertEqual(ph.store_health(Path(self.tmp.name) / "nope.db"), ("", []))

    def test_bm25_override_from_the_fallback_path_does_not_hide_the_signal(self):
        """_bm25_hits pins the process to bm25 for the query; the health read
        happens first and must report what the store is configured for."""
        self.set_backend("bge-m3")
        self.put_vec(self.add("bge"), BGE)
        self.put_vec(self.add("nomic"), NOMIC)
        self.addCleanup(memsom_tuning.clear_override, "embed.backend")
        seen = {}

        def q(prompt, k=3, clearance="topsecret", deadline_ms=800):
            memsom_tuning.override("embed.backend", "bm25")   # what _bm25_hits does
            return [], "bm25"

        mem = Path(self.tmp.name) / "mem"
        mem.mkdir()
        out = ph.run_prompt_hook({"prompt": "a long enough prompt for retrieval"},
                                 memory_dir=mem,
                                 params={"mode": "inject", "floor": 0.35,
                                         "deadline_ms": 800, "log_max_mb": 20},
                                 query_fn=q)
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertIn("dense-dark", ctx)
        self.assertIn("warm endpoint down", ctx)


# ---------------------------------------------------------------------------
# 5. a surviving warm server re-adopts the endpoint file
# ---------------------------------------------------------------------------

class TestWarmEndpointReadoption(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "memdag.db"
        self._env = os.environ.get("MEMDAG_DB")
        os.environ["MEMDAG_DB"] = str(self.db)
        conn = memsom.get_connection(self.db)
        memsom_cli.migrate_all(conn)
        conn.close()
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.stop()
        if self._env is None:
            os.environ.pop("MEMDAG_DB", None)
        else:
            os.environ["MEMDAG_DB"] = self._env
        self.tmp.cleanup()

    def _start(self):
        s = warm.WarmServer(db_path=self.db,
                            open_conn=lambda: memsom.get_connection(self.db)).start()
        self.servers.append(s)
        return s

    def test_survivor_readopts_after_the_file_owner_exits(self):
        a = self._start()
        b = self._start()
        self.assertEqual(warm.read_endpoint(self.db)["port"], b.port)
        b.stop()
        self.assertIsNone(warm.read_endpoint(self.db), "b removed its own file")
        self.assertTrue(a.ensure_endpoint_file())
        self.assertEqual(warm.read_endpoint(self.db)["port"], a.port)
        self.assertFalse(a.ensure_endpoint_file(), "own intact file: nothing to do")
        # the hook can reach it again
        hits = warm.warm_query("anything at all", k=1, deadline_s=1.0, db_path=self.db)
        self.assertEqual(hits, [])

    def test_a_live_sibling_is_left_alone(self):
        a = self._start()
        c = self._start()
        self.assertFalse(a.ensure_endpoint_file())
        self.assertEqual(warm.read_endpoint(self.db)["port"], c.port)

    def test_a_dead_endpoint_file_is_replaced(self):
        a = self._start()
        stale = {"host": "127.0.0.1", "port": a.port, "token": "not-the-token",
                 "pid": 999999999, "db": str(self.db), "version": 1}
        warm._write_private(warm.endpoint_file(self.db), json.dumps(stale))
        self.assertTrue(a.ensure_endpoint_file())
        self.assertEqual(warm.read_endpoint(self.db)["token"], a.token)

    def test_watchdog_tick_readopts(self):
        a = self._start()
        b = self._start()
        b.stop()
        wd = warm.WarmWatchdog(a, interval_s=60)
        self.assertTrue(wd.check_once())
        self.assertEqual(warm.read_endpoint(self.db)["port"], a.port)

    def test_stopped_server_never_writes(self):
        a = self._start()
        a.stop()
        self.assertFalse(a.ensure_endpoint_file())
        self.assertIsNone(warm.read_endpoint(self.db))


if __name__ == "__main__":
    unittest.main()
