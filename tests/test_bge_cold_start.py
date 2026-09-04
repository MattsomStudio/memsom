#!/usr/bin/env python3
"""Cold-start-on-demand of the DETACHED bge-m3 supervisor (2026-09-04).

WHAT SHIPPED. The bge-m3 encoder is a ~5 GB torch process that must never live
inside memsom's own process on a GPU box (the MCP server is a Claude Code
child; the in-process cold start, commit 50a5bac, segfaulted). It lives in a
detached supervisor that memsom talks to over loopback HTTP. Before this
change an idle/down supervisor was an INSTANT BM25 fall plus a "degraded"
signal. Now:

  1. `bge_client.ensure_supervisor()` spawns `retrieval.bge_spawn_cmd` DETACHED
     when /health is down, waits up to `retrieval.bge_spawn_timeout`, and is
     single-flight (one launch under concurrent callers; a launch that never
     got healthy is not retried inside a cooldown);
  2. an empty spawn_cmd never spawns — a fresh clone behaves exactly as before;
  3. every /embed carries `idle_ttl` = `retrieval.bge_idle_ttl`, and a spawned
     supervisor gets BGE_PROC_IDLE_SEC in its env;
  4. `vector_search` records degraded ONLY on a genuine failure (spawn never
     healthy / embed error); a served query clears the trail;
  5. with bge_encode_via=supervisor, torch / FlagEmbedding are NEVER imported
     into the querying process.

CI-SAFE: the "supervisor" is tests/_fake_bge_supervisor.py (stdlib), in a
thread or spawned as a real detached child of the test; no torch, no numpy
needed (colbert_maxsim degrades to pure Python), no GPU.
"""

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
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
from memsom.effects import proc as memsom_proc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fake_bge_supervisor as fake  # noqa: E402

BGE = memsom_embed.BGE_MODEL_NAME
HELPER = Path(__file__).resolve().parent / "_fake_bge_supervisor.py"
ENV_KEYS = ("MEMDAG_DB", "MEMDAG_EMBED_BACKEND", "MEMDAG_BGE_ENCODE_VIA",
            "MEMDAG_BGE_URL", "MEMDAG_BGE_SPAWN_CMD", "MEMDAG_BGE_SPAWN_TIMEOUT",
            "MEMDAG_BGE_IDLE_TTL")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _blob(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


def _quote(p) -> str:
    return '"' + str(p) + '"'


class Base(unittest.TestCase):
    """Throwaway store per test; every bge_client / embed / tuning global that
    could leak (probe cache, spawn cooldown, query memo, overrides) reset."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "store" / "test.db"
        self._env = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ["MEMDAG_DB"] = str(self.db)
        for key in ("embed.backend", "retrieval.bge_encode_via", "retrieval.bge_url",
                    "retrieval.bge_spawn_cmd", "retrieval.bge_spawn_timeout",
                    "retrieval.bge_idle_ttl"):
            memsom_tuning.clear_override(key)
        memsom_tuning._invalidate_persisted()
        self._reset_globals()
        # Build the store under the suite default backend, switch per test.
        self.conn = memsom.get_connection()
        memsom_cli.migrate_all(self.conn)
        self._servers = []

    def tearDown(self):
        self.conn.close()
        for srv in self._servers:
            try:
                srv.shutdown()
                srv.server_close()
            # FAILOPEN-in-tests: a server that already stopped is fine.
            except Exception:
                pass
        self._reset_globals()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for key in ("embed.backend", "retrieval.bge_encode_via", "retrieval.bge_url",
                    "retrieval.bge_spawn_cmd", "retrieval.bge_spawn_timeout",
                    "retrieval.bge_idle_ttl"):
            memsom_tuning.clear_override(key)
        memsom_tuning._invalidate_persisted()
        self.tmp.cleanup()

    @staticmethod
    def _reset_globals():
        bge_client._reset_probe()
        # Never drop a live child's handle (ResourceWarning at GC); reap the
        # ones that have exited, wait briefly for the rest (the fake exits on
        # /quit or its own --ttl).
        for c in list(bge_client._CHILDREN):
            try:
                c.wait(timeout=5)
            # FAILOPEN-in-tests: a child that outlives the wait is reaped by its --ttl.
            except Exception:
                pass
        bge_client._CHILDREN[:] = [c for c in bge_client._CHILDREN if c.poll() is None]
        memsom_embed._reset_query_memo()
        memsom_embed._WARNED_SUPERVISOR = False
        memsom_embed._WARNED_FALLBACK = False
        memsom_embed._MODEL = None
        memsom_retrieve._LAST_QUERY_FALLBACK = None

    # -- helpers ----------------------------------------------------------
    def env(self, **kv):
        for k, v in kv.items():
            os.environ[k] = str(v)

    def fake_up(self, **state):
        port = free_port()
        srv, st = fake.serve_in_thread(port, **state)
        self._servers.append(srv)
        self.env(MEMDAG_BGE_URL=f"http://127.0.0.1:{port}/embed")
        return port, st

    def closed_url(self):
        port = free_port()   # nothing listens here
        self.env(MEMDAG_BGE_URL=f"http://127.0.0.1:{port}/embed")
        return port

    def add(self, content, channel="user"):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])

    def put_vec(self, nid, vec, model=BGE):
        """Canned dense vector + BM25 postings directly (never index_node, which
        would embed for real)."""
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
            for t, n in tf.items():
                self.conn.execute(
                    "INSERT INTO postings(term, node_id, tf) VALUES (?,?,?)", (t, nid, n))
            self.conn.execute(
                "INSERT OR REPLACE INTO docstats(node_id, length) VALUES (?,?)",
                (nid, sum(tf.values())))


# ---------------------------------------------------------------------------
# (vi) the knobs
# ---------------------------------------------------------------------------

class TestKnobs(unittest.TestCase):
    def test_idle_ttl_registration(self):
        k = memsom_tuning.REGISTRY["retrieval.bge_idle_ttl"]
        self.assertEqual((k.type, k.default, k.bounds), (int, 60, (0, 86400)))
        self.assertEqual(k.source, "env:MEMDAG_BGE_IDLE_TTL")
        self.assertEqual(k.feature, "retrieval.bge")

    def test_spawn_cmd_registration(self):
        k = memsom_tuning.REGISTRY["retrieval.bge_spawn_cmd"]
        self.assertEqual((k.type, k.default), (str, ""))
        self.assertEqual(k.source, "env:MEMDAG_BGE_SPAWN_CMD")
        self.assertEqual(k.feature, "retrieval.bge")

    def test_spawn_timeout_registration(self):
        k = memsom_tuning.REGISTRY["retrieval.bge_spawn_timeout"]
        self.assertEqual((k.type, k.default, k.bounds), (int, 30, (1, 300)))
        self.assertEqual(k.source, "env:MEMDAG_BGE_SPAWN_TIMEOUT")

    def test_accessors_coerce_and_clamp(self):
        saved = {k: os.environ.get(k) for k in ("MEMDAG_BGE_IDLE_TTL", "MEMDAG_BGE_SPAWN_TIMEOUT")}
        try:
            os.environ.pop("MEMDAG_BGE_IDLE_TTL", None)
            os.environ.pop("MEMDAG_BGE_SPAWN_TIMEOUT", None)
            self.assertEqual(bge_client.idle_ttl(), 60)
            self.assertEqual(bge_client.spawn_timeout(), 30)
            os.environ["MEMDAG_BGE_IDLE_TTL"] = "0"
            self.assertEqual(bge_client.idle_ttl(), 0)
            os.environ["MEMDAG_BGE_IDLE_TTL"] = "999999"      # out of bounds -> default
            memsom_tuning._clear_warned("retrieval.bge_idle_ttl")
            self.assertEqual(bge_client.idle_ttl(), 60)
            os.environ["MEMDAG_BGE_SPAWN_TIMEOUT"] = "5"
            self.assertEqual(bge_client.spawn_timeout(), 5)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# (i) empty spawn_cmd -> never spawns (the fresh-clone contract)
# ---------------------------------------------------------------------------

class TestNoSpawnCmd(Base):
    def test_down_supervisor_without_spawn_cmd_never_launches(self):
        self.closed_url()
        with patch.object(bge_client.memsom_proc, "popen",
                          side_effect=AssertionError("must not spawn")):
            self.assertFalse(bge_client.ensure_supervisor())
        self.assertEqual(bge_client._SPAWN_COUNT, 0)

    def test_auto_mode_still_falls_to_inprocess(self):
        """Existing behaviour: auto + down supervisor + no spawn_cmd -> the
        in-process encoder is used, encode_http never called."""
        self.closed_url()
        self.env(MEMDAG_BGE_ENCODE_VIA="auto")
        with patch.object(bge_client.memsom_proc, "popen",
                          side_effect=AssertionError("must not spawn")), \
             patch.object(bge_client, "encode_http",
                          side_effect=AssertionError("supervisor is down")), \
             patch.object(memsom_embed, "_encode",
                          lambda t: {"dense": [1.0], "sparse": {}, "colbert": []}):
            self.assertEqual(memsom_embed.encode_doc("x")["dense"], [1.0])

    def test_supervisor_mode_never_falls_to_inprocess(self):
        self.closed_url()
        self.env(MEMDAG_BGE_ENCODE_VIA="supervisor")
        with patch.object(memsom_embed, "_encode",
                          side_effect=AssertionError("torch must not be used")):
            self.assertIsNone(memsom_embed.encode_query("q"))
        self.assertFalse(memsom_embed.bge_usable())


# ---------------------------------------------------------------------------
# (ii) spawn once under concurrency, then a real encode over the wire
# ---------------------------------------------------------------------------

class TestSpawnOnce(Base):
    def test_eight_concurrent_callers_spawn_one_supervisor(self):
        port = free_port()
        count = Path(self.tmp.name) / "count.txt"
        self.env(MEMDAG_BGE_URL=f"http://127.0.0.1:{port}/embed",
                 MEMDAG_BGE_SPAWN_CMD=f"{_quote(sys.executable)} {_quote(HELPER)} "
                                      f"--port {port} --count-file {_quote(count)} --ttl 40",
                 MEMDAG_BGE_SPAWN_TIMEOUT="20", MEMDAG_BGE_ENCODE_VIA="supervisor")
        results = []
        lock = threading.Lock()

        def go():
            ok = bge_client.ensure_supervisor()
            with lock:
                results.append(ok)

        try:
            t0 = time.monotonic()
            threads = [threading.Thread(target=go) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            elapsed = time.monotonic() - t0
            self.assertEqual(results, [True] * 8, results)
            self.assertLess(elapsed, 25.0)
            time.sleep(0.5)
            self.assertEqual(count.read_text().count("started"), 1,
                             "single-flight: exactly one launch")
            self.assertEqual(bge_client._SPAWN_COUNT, 1)
            # the spawned fake serves a real encode with all three signals
            enc = memsom_embed.encode_query("hello")
            self.assertEqual(enc["dense"], fake.DENSE)
            self.assertEqual(enc["sparse"], fake.SPARSE)
            self.assertEqual(enc["colbert"], fake.COLBERT)
        finally:
            try:
                from memsom.effects import net as memsom_net
                memsom_net.fetch(f"http://127.0.0.1:{port}/quit", data=b"{}", timeout=2)
            # FAILOPEN-in-tests: the child also exits on its own --ttl.
            except Exception:
                pass


# ---------------------------------------------------------------------------
# (iii) a spawn that dies -> bounded False, degraded recorded, no respawn storm
# ---------------------------------------------------------------------------

class TestSpawnFailure(Base):
    def _configure(self):
        self.closed_url()
        self.env(MEMDAG_BGE_SPAWN_CMD=f"{_quote(sys.executable)} -c \"import sys; sys.exit(3)\"",
                 MEMDAG_BGE_SPAWN_TIMEOUT="1", MEMDAG_BGE_ENCODE_VIA="supervisor",
                 MEMDAG_EMBED_BACKEND="bge-m3")

    def test_ensure_supervisor_is_false_within_timeout(self):
        self._configure()
        t0 = time.monotonic()
        self.assertFalse(bge_client.ensure_supervisor())
        # bound = spawn_timeout (1 s) + at most two full-length closed-port
        # probes (MEASURED 2 s each on Windows loopback) + slack; the point is
        # "bounded", not "instant".
        self.assertLess(time.monotonic() - t0, 1 + 2 * bge_client.HEALTH_TIMEOUT + 3)
        self.assertEqual(bge_client._SPAWN_COUNT, 1)
        # cooldown: a second call inside the window does not launch again
        t1 = time.monotonic()
        self.assertFalse(bge_client.ensure_supervisor())
        self.assertEqual(bge_client._SPAWN_COUNT, 1)
        self.assertLess(time.monotonic() - t1, bge_client.HEALTH_TIMEOUT + 1,
                        "inside the cooldown a call costs at most one probe")

    def test_vector_search_records_degraded_on_a_genuine_failure(self):
        self._configure()
        self.put_vec(self.add("a bge vector"), (1.0, 0.0, 0.0, 0.0))
        self.assertTrue(memsom_embed.bge_usable(), "spawn_cmd set -> a path exists")
        self.assertEqual(memsom_retrieve.vector_search(self.conn, "q"), [])
        trail = memsom_retrieve.last_query_fallback(self.conn)
        self.assertIsNotNone(trail)
        self.assertIn("bge-m3 query encode failed", trail["reason"])
        self.assertEqual(memsom_retrieve.last_query_fallback_reason(), trail["reason"])
        lines = memsom_retrieve.retrieval_warnings(self.conn)
        self.assertTrue(any("query encoder unreachable" in l for l in lines), lines)
        self.assertEqual(bge_client._SPAWN_COUNT, 1, "one launch per outage, even across signals")


# ---------------------------------------------------------------------------
# (iv) supervisor up -> dense rows, trail cleared, no warning
# ---------------------------------------------------------------------------

class TestServed(Base):
    def test_live_supervisor_returns_dense_and_clears_the_trail(self):
        self.fake_up()
        self.env(MEMDAG_BGE_ENCODE_VIA="supervisor", MEMDAG_EMBED_BACKEND="bge-m3")
        nid = self.add("the nebula firewall rule for the mesh")
        self.put_vec(nid, (1.0, 0.0, 0.0, 0.0))
        other = self.add("unrelated note about coffee")
        self.put_vec(other, (0.0, 1.0, 0.0, 0.0))
        # a stale trail from an earlier outage must not survive a served query
        memsom_retrieve._record_query_degraded(self.conn, "earlier outage")
        self.assertIsNotNone(memsom_retrieve.last_query_fallback(self.conn))

        hits = memsom_retrieve.vector_search(self.conn, "anything", k=2)
        self.assertEqual([h[0] for h in hits], [nid, other])
        self.assertAlmostEqual(hits[0][1], 1.0, places=5)
        self.assertIsNone(memsom_retrieve.last_query_fallback(self.conn))
        self.assertIsNone(memsom_retrieve.last_query_fallback_reason())
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_full_retrieve_path_over_the_wire(self):
        _port, st = self.fake_up()
        self.env(MEMDAG_BGE_ENCODE_VIA="supervisor", MEMDAG_EMBED_BACKEND="bge-m3")
        nid = self.add("the nebula firewall rule for the mesh")
        self.put_vec(nid, (1.0, 0.0, 0.0, 0.0))
        rows = memsom_retrieve.retrieve(self.conn, "firewall", k=3)
        self.assertEqual([r[0] for r in rows], [nid])
        # dense + sparse + colbert all read the SAME encode: one round trip
        self.assertEqual(st.get("embed_calls"), 1, st)
        self.assertEqual(memsom_retrieve.retrieval_warnings(self.conn), [])

    def test_features_report_active_over_the_supervisor_without_torch(self):
        from memsom.interface import features as memsom_features
        self.fake_up()
        self.env(MEMDAG_BGE_ENCODE_VIA="supervisor", MEMDAG_EMBED_BACKEND="bge-m3")
        with patch.object(memsom_embed, "bge_available",
                          side_effect=AssertionError("supervisor mode must not import torch")):
            st = memsom_features._retrieval_bge(self.conn)
        self.assertEqual(st["state"], "active", st)
        self.assertIn("supervisor reachable", st["detail"])

    def test_features_report_idle_spawnable_as_active_and_unspawnable_as_degraded(self):
        from memsom.interface import features as memsom_features
        self.closed_url()
        self.env(MEMDAG_BGE_ENCODE_VIA="supervisor", MEMDAG_EMBED_BACKEND="bge-m3")
        st = memsom_features._retrieval_bge(self.conn)
        self.assertEqual(st["state"], "degraded", st)
        self.env(MEMDAG_BGE_SPAWN_CMD="python -c pass")
        st = memsom_features._retrieval_bge(self.conn)
        self.assertEqual(st["state"], "active", st)
        self.assertIn("spawned on demand", st["detail"])


# ---------------------------------------------------------------------------
# (v) idle_ttl on the wire + BGE_PROC_IDLE_SEC in the spawn env, detached flags
# ---------------------------------------------------------------------------

class TestIdleTtl(Base):
    def test_idle_ttl_rides_on_every_embed(self):
        _port, st = self.fake_up()
        self.env(MEMDAG_BGE_IDLE_TTL="7", MEMDAG_BGE_ENCODE_VIA="supervisor")
        enc = memsom_embed.encode_query("hello")
        self.assertEqual(enc["dense"], fake.DENSE)
        self.assertEqual(st["last_body"]["idle_ttl"], 7)
        self.assertEqual(st["last_body"]["input"], "hello")

    def test_spawn_env_and_detachment(self):
        self.closed_url()
        self.env(MEMDAG_BGE_IDLE_TTL="7", MEMDAG_BGE_SPAWN_CMD='launcher --flag "a b"',
                 MEMDAG_BGE_SPAWN_TIMEOUT="5")
        captured = {}
        state = {"launched": False}

        def fake_popen(argv, *, env=None, keep=(), **kwargs):
            captured["argv"] = argv
            captured["env"] = env
            captured["kwargs"] = kwargs
            state["launched"] = True

            class _P:
                pid = 4242

                def poll(self):
                    return None
            return _P()

        def fake_health(force=False, timeout=None):
            return state["launched"]

        with patch.object(bge_client.memsom_proc, "popen", fake_popen), \
             patch.object(bge_client, "bge_http_available", fake_health):
            self.assertTrue(bge_client.ensure_supervisor())
        self.assertEqual(captured["argv"], ["launcher", "--flag", "a b"])
        self.assertEqual(captured["env"], {"BGE_PROC_IDLE_SEC": "7"})
        kw = captured["kwargs"]
        self.assertEqual(kw["stdin"], memsom_proc.DEVNULL)
        self.assertEqual(kw["stdout"], memsom_proc.DEVNULL)
        self.assertEqual(kw["stderr"], memsom_proc.DEVNULL)
        self.assertTrue(kw["close_fds"])
        if sys.platform == "win32":
            flags = kw["creationflags"]
            for f in (memsom_proc.DETACHED_PROCESS, memsom_proc.CREATE_NEW_PROCESS_GROUP,
                      memsom_proc.CREATE_NO_WINDOW):
                self.assertEqual(flags & f, f)
        else:
            self.assertTrue(kw["start_new_session"])

    def test_windows_paths_survive_the_split(self):
        cmd = r'"C:\Program Files\Python312\python.exe" "C:\Users\me\.claude\x.py" --port 1'
        with patch.object(bge_client.sys, "platform", "win32"):
            argv = bge_client._split_cmd(cmd)
        self.assertEqual(argv, [r"C:\Program Files\Python312\python.exe",
                                r"C:\Users\me\.claude\x.py", "--port", "1"])


# ---------------------------------------------------------------------------
# (vii) supervisor mode never imports torch / FlagEmbedding into the querier
# ---------------------------------------------------------------------------

_CHILD = r"""
import json, sys, struct
import memsom
from memsom.interface import cli
from memsom.retrieval import retrieve as r
conn = memsom.get_connection()
cli.migrate_all(conn)
with conn:
    nid = memsom.insert_node(conn, "the nebula firewall rule for the mesh", "user", memsom.RANK["user"])
    conn.execute("INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec) VALUES (?,?,?,?)",
                 (nid, "bge-m3", 4, struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)))
rows = r.retrieve(conn, "firewall", k=3)
print(json.dumps({"torch": "torch" in sys.modules,
                  "flag": "FlagEmbedding" in sys.modules,
                  "hits": [row[0] for row in rows], "nid": nid,
                  "warnings": r.retrieval_warnings(conn)}))
"""


class TestNoTorchInProcess(Base):
    def test_fresh_process_query_in_supervisor_mode_loads_no_torch(self):
        self.fake_up()
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMDAG_")}
        env.update({"MEMDAG_DB": str(self.db), "MEMDAG_EMBED_BACKEND": "bge-m3",
                    "MEMDAG_BGE_ENCODE_VIA": "supervisor",
                    "MEMDAG_BGE_URL": os.environ["MEMDAG_BGE_URL"]})
        res = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True,
                             text=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stderr)
        out = json.loads(res.stdout.strip().splitlines()[-1])
        self.assertFalse(out["torch"], out)
        self.assertFalse(out["flag"], out)
        self.assertEqual(out["hits"], [out["nid"]], out)
        self.assertEqual(out["warnings"], [], out)


if __name__ == "__main__":
    unittest.main()
