"""Tests for the UserPromptSubmit retrieval hook, its warm endpoint, the log
+ rotation, hook-stats, and the plugin packaging files.

Run:  python -m pytest tests/test_prompt_hook.py -q
"""
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import memsom
from memsom.bridge import bridge_import as bi
from memsom.bridge import wire_claude as wc
from memsom.interface import cli as memsom_cli
from memsom.interface import prompt_hook as ph
from memsom.lifecycle import forget
from memsom.retrieval import retrieve as memsom_retrieve
from memsom.retrieval import warm
from memsom import tuning

HERE = Path(__file__).resolve().parent.parent

MEMORIES = {
    "feedback_piped_exit_codes": (
        "description: never trust a piped exit code; cmd | tail reports tail's 0\n",
        "A pipeline's exit status is the LAST command's. Use pipefail or check PIPESTATUS "
        "when a piped command's failure matters."),
    "reference_nebula_mesh_firewall": (
        "description: mesh service = two firewalls, the Nebula firewall AND host ufw\n",
        "A service on the Nebula mesh must be opened in the Nebula firewall config and "
        "in the host firewall (ufw / iptables). Nebula filters ICMP, so probe with TCP."),
    "user_fitness_physique": (
        "description: heavy lifter, classic physique aim\n",
        "Trains heavy compound lifts four days a week; classic physique target."),
}


def _make_store(tmp):
    """A memory dir with three memories imported + indexed into a throwaway DB."""
    tmp = Path(tmp)
    mem = tmp / "memory"
    mem.mkdir()
    lines = ["# Memory", "", "## Feedback"]
    for stem, (desc, body) in MEMORIES.items():
        (mem / f"{stem}.md").write_text(
            f"---\nname: {stem}\n{desc}type: {stem.split('_')[0]}\n---\n\n{body}\n",
            encoding="utf-8")
        lines.append(f"- [{stem}]({stem}.md) — {desc.split(':', 1)[1].strip()}")
    (mem / "MEMORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    db = tmp / "store" / "memdag.db"
    os.environ["MEMDAG_DB"] = str(db)
    os.environ["MEMDAG_BRIDGE_MEMORY_DIR"] = str(mem)
    conn = memsom.get_connection()
    memsom_cli.migrate_all(conn)
    bi.migrate(conn)
    bi.import_all(conn, mem, dry_run=False)
    memsom_retrieve.index_all(conn)
    conn.close()
    return mem, db


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in
                     ("MEMDAG_DB", "MEMDAG_BRIDGE_MEMORY_DIR", "MEMDAG_EMBED_BACKEND")}
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        self.tmp = tempfile.TemporaryDirectory()
        self.mem, self.db = _make_store(self.tmp.name)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()


# ---------------------------------------------------------------------------
# hits + coverage
# ---------------------------------------------------------------------------

class TestHits(_StoreCase):
    def test_hits_rank_the_right_memory_and_carry_stem_hook_score(self):
        conn = memsom.get_connection()
        try:
            hits = warm.hits_for(conn, "piped command exit code pipefail", k=3)
        finally:
            conn.close()
        self.assertTrue(hits)
        self.assertEqual(hits[0]["stem"], "feedback_piped_exit_codes")
        self.assertIn("piped exit code", hits[0]["hook"])
        self.assertGreater(hits[0]["score"], 0.3)
        for h in hits:
            self.assertGreaterEqual(h["score"], 0.0)
            self.assertLessEqual(h["score"], 1.0)

    def test_unrelated_prompt_scores_below_the_floor(self):
        conn = memsom.get_connection()
        try:
            hits = warm.hits_for(conn, "quarterly marketing budget spreadsheet colour palette", k=3)
        finally:
            conn.close()
        floor = forget.PANEL_PARAM_DEFAULTS["prompt_hook_floor"]
        self.assertEqual(ph.apply_floor(hits, floor), [])

    def test_clearance_filters_the_pool(self):
        conn = memsom.get_connection()
        try:
            nid = conn.execute("SELECT id FROM nodes WHERE source_ref = ?",
                               ("memory:feedback_piped_exit_codes",)).fetchone()[0]
            conn.execute("UPDATE nodes SET conf_label = 3 WHERE id = ?", (nid,))
            conn.commit()
            hits = warm.hits_for(conn, "piped command exit code pipefail", k=3,
                                 clearance="public")
        finally:
            conn.close()
        self.assertNotIn("feedback_piped_exit_codes", [h["stem"] for h in hits])


# ---------------------------------------------------------------------------
# warm endpoint
# ---------------------------------------------------------------------------

class TestWarmEndpoint(_StoreCase):
    def setUp(self):
        super().setUp()
        self.srv = warm.WarmServer(self.db).start()

    def tearDown(self):
        self.srv.stop()
        super().tearDown()

    def test_endpoint_file_written_and_removed(self):
        f = warm.endpoint_file(self.db)
        self.assertTrue(f.exists())
        data = json.loads(f.read_text())
        self.assertEqual(data["port"], self.srv.port)
        self.assertEqual(data["host"], "127.0.0.1")
        self.srv.stop()
        self.assertFalse(f.exists())

    def test_warm_query_returns_hits(self):
        hits = warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
        self.assertEqual(hits[0]["stem"], "reference_nebula_mesh_firewall")

    def test_bad_token_refused(self):
        ep = warm.read_endpoint(self.db)
        with socket.create_connection(("127.0.0.1", ep["port"]), timeout=2) as s:
            s.sendall((json.dumps({"token": "nope", "method": "retrieve",
                                   "query": "nebula"}) + "\n").encode())
            resp = json.loads(s.makefile().readline())
        self.assertEqual(resp, {"error": "unauthorized"})

    def test_non_loopback_peer_refused_before_any_work(self):
        opened = []

        def open_conn():
            opened.append(1)
            return memsom.get_connection()
        raw = json.dumps({"token": self.srv.token, "method": "retrieve",
                          "query": "nebula"}).encode()
        resp = warm.handle_request(raw, "10.0.0.7", self.srv.token, open_conn)
        self.assertEqual(resp["error"], "forbidden")
        self.assertEqual(opened, [])
        ok = warm.handle_request(raw, "127.0.0.1", self.srv.token, open_conn)
        self.assertIn("hits", ok)
        self.assertEqual(opened, [1])

    def test_only_retrieve_is_served(self):
        raw = json.dumps({"token": self.srv.token, "method": "revoke", "id": 1}).encode()
        resp = warm.handle_request(raw, "127.0.0.1", self.srv.token, memsom.get_connection)
        self.assertEqual(resp["error"], "unknown-method")

    def test_refuses_to_bind_off_loopback(self):
        with self.assertRaises(ValueError):
            warm.WarmServer(self.db, host="0.0.0.0").start()

    def test_endpoint_file_off_loopback_is_ignored(self):
        f = warm.endpoint_file(self.db)
        f.write_text(json.dumps({"host": "10.0.0.9", "port": 1, "token": "x"}))
        self.assertIsNone(warm.read_endpoint(self.db))

    def test_query_hits_prefers_warm(self):
        hits, source = ph.query_hits("nebula mesh firewall ufw", k=3, deadline_ms=2000)
        self.assertEqual(source, "warm")
        self.assertEqual(hits[0]["stem"], "reference_nebula_mesh_firewall")


class TestWarmResilience(_StoreCase):
    """The 2026-08-20 wedge: a listener that accepts but never serves. Every
    layer that bounds it is exercised here through real sockets."""

    def tearDown(self):
        srv = getattr(self, "srv", None)
        if srv is not None:
            srv.stop()
        super().tearDown()

    def test_ping_answers_without_touching_db(self):
        opened = []

        def open_conn():
            opened.append(1)
            return memsom.get_connection()
        self.srv = warm.WarmServer(self.db, open_conn=open_conn).start()
        self.assertTrue(self.srv.ping(timeout_s=2))
        raw = json.dumps({"token": self.srv.token, "method": "ping"}).encode()
        self.assertEqual(warm.handle_request(raw, "127.0.0.1", self.srv.token, open_conn),
                         {"pong": True, "pid": os.getpid()})
        self.assertEqual(opened, [])
        # ping still needs the token
        raw = json.dumps({"token": "nope", "method": "ping"}).encode()
        self.assertEqual(warm.handle_request(raw, "127.0.0.1", self.srv.token, open_conn),
                         {"error": "unauthorized"})

    def test_hanging_handler_does_not_block_second_client(self):
        release = threading.Event()
        calls = []

        def open_conn():
            calls.append(1)
            if len(calls) == 1:
                release.wait(10)          # first request hangs inside the handler
            return memsom.get_connection()
        self.srv = warm.WarmServer(self.db, open_conn=open_conn).start()
        ep = warm.read_endpoint(self.db)
        hung = socket.create_connection(("127.0.0.1", ep["port"]), timeout=5)
        hung.sendall((json.dumps({"token": ep["token"], "method": "retrieve",
                                  "query": "nebula"}) + "\n").encode())
        for _ in range(200):              # wait until the handler is inside open_conn
            if calls:
                break
            time.sleep(0.01)
        self.assertEqual(calls, [1])
        t0 = time.monotonic()
        hits = warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(hits[0]["stem"], "reference_nebula_mesh_firewall")
        self.assertTrue(self.srv.ping(timeout_s=2))
        release.set()
        hung.close()

    def test_silent_client_does_not_block_others(self):
        self.srv = warm.WarmServer(self.db).start()
        ep = warm.read_endpoint(self.db)
        idle = [socket.create_connection(("127.0.0.1", ep["port"]), timeout=5)
                for _ in range(4)]        # connect, never send
        try:
            hits = warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
            self.assertEqual(hits[0]["stem"], "reference_nebula_mesh_firewall")
            # the server hangs up on a silent client after CONN_TIMEOUT_S
            idle[0].settimeout(2)
            self.assertEqual(idle[0].recv(16), b"")
        finally:
            for s in idle:
                s.close()

    def test_handler_exception_does_not_kill_server(self):
        calls = []

        def open_conn():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return memsom.get_connection()
        self.srv = warm.WarmServer(self.db, open_conn=open_conn).start()
        with self.assertRaises(warm.WarmUnavailable) as cm:
            warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
        self.assertIn("internal", str(cm.exception))
        self.assertTrue(self.srv.alive())
        hits = warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
        self.assertEqual(hits[0]["stem"], "reference_nebula_mesh_firewall")

    def test_slow_endpoint_is_unavailable_not_waited_for(self):
        def open_conn():
            time.sleep(1.5)
            return memsom.get_connection()
        self.srv = warm.WarmServer(self.db, open_conn=open_conn).start()
        t0 = time.monotonic()
        hits, source = ph.query_hits("piped exit code pipefail", k=3, deadline_ms=2000)
        self.assertLess(time.monotonic() - t0, 1.2)   # ~250 ms warm + bm25
        self.assertEqual(source, "bm25")
        self.assertTrue(hits)

    def test_backoff_engages_after_two_failures_and_clears(self):
        gate = {"hang": True}

        def open_conn():
            if gate["hang"]:
                time.sleep(1.0)
            return memsom.get_connection()
        self.srv = warm.WarmServer(self.db, open_conn=open_conn).start()
        ep = warm.read_endpoint(self.db)
        with self.assertRaises(warm.WarmUnavailable):
            warm.warm_query("nebula", db_path=self.db)
        self.assertEqual(warm.read_backoff(self.db)["failures"], 1)
        self.assertFalse(warm.in_backoff(ep, self.db))
        with self.assertRaises(warm.WarmUnavailable):
            warm.warm_query("nebula", db_path=self.db)
        self.assertEqual(warm.read_backoff(self.db)["failures"], 2)
        self.assertTrue(warm.in_backoff(ep, self.db))
        # while backed off the warm path is skipped outright (no socket, no wait)
        gate["hang"] = False
        t0 = time.monotonic()
        with self.assertRaises(warm.WarmUnavailable) as cm:
            warm.warm_query("nebula", db_path=self.db)
        self.assertEqual(str(cm.exception), "backoff")
        self.assertLess(time.monotonic() - t0, 0.05)
        # the window expires on its own ...
        self.assertFalse(warm.in_backoff(ep, self.db, now=time.time() + warm.BACKOFF_S + 1))
        # ... and a successful call clears the sidecar
        warm.clear_backoff(self.db)
        hits = warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
        self.assertTrue(hits)
        self.assertIsNone(warm.read_backoff(self.db))

    def test_no_backoff_when_server_pid_is_dead(self):
        ep = {"port": 1, "pid": 4242}
        warm.note_warm_failure(ep, self.db, alive=lambda pid: False)
        n = warm.note_warm_failure(ep, self.db, alive=lambda pid: False)
        self.assertEqual(n, 2)
        self.assertFalse(warm.in_backoff(ep, self.db))
        # and a counter never carries across listeners (different port)
        warm.note_warm_failure({"port": 2, "pid": 4242}, self.db, alive=lambda pid: True)
        self.assertEqual(warm.read_backoff(self.db)["failures"], 1)

    def test_refused_connect_does_not_count_as_failure(self):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        warm.endpoint_file(self.db).write_text(
            json.dumps({"host": "127.0.0.1", "port": port, "token": "dead", "pid": 1}))
        with self.assertRaises(warm.WarmUnavailable):
            warm.warm_query("nebula", db_path=self.db)
        self.assertIsNone(warm.read_backoff(self.db))

    def test_restart_clears_backoff_and_rewrites_endpoint(self):
        self.srv = warm.WarmServer(self.db).start()
        ep = warm.read_endpoint(self.db)
        warm.note_warm_failure(ep, self.db, alive=lambda pid: True)
        warm.note_warm_failure(ep, self.db, alive=lambda pid: True)
        self.assertTrue(warm.in_backoff(ep, self.db))
        self.srv.restart()
        ep2 = warm.read_endpoint(self.db)
        self.assertNotEqual(ep2["token"], ep["token"])
        self.assertIsNone(warm.read_backoff(self.db))
        self.assertTrue(self.srv.ping(timeout_s=2))

    def test_watchdog_restarts_a_wedged_listener(self):
        self.srv = warm.WarmServer(self.db).start()
        # wedge it: the accept loop stops but the socket stays bound + listening
        self.srv._server.shutdown()
        self.assertFalse(self.srv.ping(timeout_s=0.5))
        wd = warm.WarmWatchdog(self.srv, interval_s=60, ping_timeout_s=0.5)
        self.assertFalse(wd.check_once())
        self.assertEqual(self.srv.restarts, 1)
        self.assertTrue(self.srv.ping(timeout_s=2))
        self.assertTrue(wd.check_once())
        ep = warm.read_endpoint(self.db)
        self.assertEqual(ep["port"], self.srv.port)
        hits = warm.warm_query("nebula mesh firewall ufw", k=3, db_path=self.db)
        self.assertTrue(hits)

    def test_mcp_shutdown_removes_endpoint_file(self):
        from memsom.interface import mcp as memsom_mcp
        srv = memsom_mcp._start_warm_endpoint()
        self.assertIsNotNone(srv)
        f = warm.endpoint_file(self.db)
        self.assertTrue(f.exists())
        self.assertIsNotNone(getattr(srv, "watchdog", None))
        memsom_mcp._stop_warm_endpoint(srv)
        self.assertFalse(f.exists())
        self.assertTrue(srv.watchdog._stop.is_set())

    def test_serve_stdio_removes_endpoint_file_on_exit(self):
        from memsom.interface import mcp as memsom_mcp
        f = warm.endpoint_file(self.db)
        with mock.patch.object(memsom_mcp, "_serve_lines",
                               side_effect=lambda stream: None):
            memsom_mcp.serve_stdio()
        self.assertFalse(f.exists())


class TestFallbackAndDeadline(_StoreCase):
    def test_bm25_fallback_when_endpoint_down(self):
        self.assertFalse(warm.endpoint_file(self.db).exists())
        hits, source = ph.query_hits("piped exit code pipefail", k=3, deadline_ms=2000)
        self.assertEqual(source, "bm25")
        self.assertEqual(hits[0]["stem"], "feedback_piped_exit_codes")

    def test_stale_endpoint_file_falls_back(self):
        # a dead server's file: nothing listens on that port -> refused -> bm25
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        warm.endpoint_file(self.db).write_text(
            json.dumps({"host": "127.0.0.1", "port": port, "token": "dead"}))
        hits, source = ph.query_hits("piped exit code pipefail", k=3, deadline_ms=2000)
        self.assertEqual(source, "bm25")
        self.assertTrue(hits)

    def test_slow_fallback_times_out_silently(self):
        def slow(*a, **k):
            time.sleep(0.6)
            return []
        with mock.patch.object(ph, "_bm25_hits", slow):
            hits, source = ph.query_hits("piped exit code pipefail", k=3, deadline_ms=100)
        self.assertEqual((hits, source), ([], "timeout"))

    def test_hook_query_cli_timeout_prints_nothing_exit_0(self):
        def slow(*a, **k):
            time.sleep(0.6)
            return []
        buf = io.StringIO()
        with mock.patch.object(ph, "_bm25_hits", slow), \
                mock.patch.object(ph, "_exit_now_if_worker_stuck", lambda: None), \
                redirect_stdout(buf):
            rc = memsom_cli.main(["hook-query", "piped exit code", "--deadline-ms", "100"])
        self.assertIn(rc, (None, 0))
        self.assertEqual(buf.getvalue(), "")

    def test_hook_query_cli_emits_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            memsom_cli.main(["hook-query", "piped exit code pipefail", "--deadline-ms", "2000"])
        out = json.loads(buf.getvalue())
        self.assertEqual(out["source"], "bm25")
        self.assertEqual(out["hits"][0]["stem"], "feedback_piped_exit_codes")

    def test_fallback_forces_bm25_backend(self):
        # in-process pin, not an env write (memsom.tuning.override) -- reset
        # it after the test so it cannot leak into a later test in this process.
        os.environ["MEMDAG_EMBED_BACKEND"] = "bge-m3"
        self.addCleanup(tuning.clear_override, "embed.backend")
        with mock.patch("memsom.retrieval.embed.bge_available",
                        side_effect=AssertionError("cold load attempted")):
            hits, source = ph.query_hits("piped exit code pipefail", k=3, deadline_ms=2000)
        self.assertEqual(source, "bm25")
        self.assertEqual(tuning.resolve("embed.backend"), "bm25")


# ---------------------------------------------------------------------------
# hook-prompt
# ---------------------------------------------------------------------------

def _fake_query(hits):
    def q(prompt, k=3, clearance="topsecret", deadline_ms=800):
        return hits, "fake"
    return q


HITS = [
    {"id": 1, "stem": "feedback_piped_exit_codes", "label": "feedback_piped_exit_codes",
     "hook": "never trust a piped exit code", "score": 0.8},
    {"id": 2, "stem": "reference_nebula_mesh_firewall", "label": "reference_nebula_mesh_firewall",
     "hook": "mesh service = two firewalls", "score": 0.4},
    {"id": 3, "stem": "user_fitness_physique", "label": "user_fitness_physique",
     "hook": "heavy lifter", "score": 0.1},
]


class TestHookPrompt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = Path(self.tmp.name)
        self.params = {"mode": "inject", "floor": 0.35, "deadline_ms": 800, "log_max_mb": 20}

    def tearDown(self):
        self.tmp.cleanup()

    def _log(self):
        p = ph.log_path(self.mem)
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]

    def test_parses_stdin_json_and_emits_block(self):
        out = ph.run_prompt_hook({"prompt": "why does my piped command report success?",
                                  "hook_event_name": "UserPromptSubmit"},
                                 memory_dir=self.mem, params=self.params,
                                 query_fn=_fake_query(HITS))
        doc = json.loads(out)
        ctx = doc["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(doc["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertTrue(ctx.startswith("Relevant memories:\n- [feedback_piped_exit_codes] "))
        self.assertIn("- [reference_nebula_mesh_firewall]", ctx)
        self.assertNotIn("user_fitness_physique", ctx)          # below the floor
        self.assertLessEqual(len(ctx.encode()), ph.MAX_BLOCK_BYTES)
        rec = self._log()[0]
        self.assertTrue(rec["injected"] and rec["would_inject"])
        self.assertEqual([h["stem"] for h in rec["hits"]],
                         [h["stem"] for h in HITS])

    def test_short_prompt_and_slash_command_are_silent_and_unlogged(self):
        for prompt in ("hi there", "/recall nebula firewall config please"):
            out = ph.run_prompt_hook({"prompt": prompt}, memory_dir=self.mem,
                                     params=self.params, query_fn=_fake_query(HITS))
            self.assertIsNone(out)
        self.assertEqual(self._log(), [])

    def test_floor_respected_nothing_above_means_no_output_but_logged(self):
        out = ph.run_prompt_hook({"prompt": "tell me about quarterly marketing budgets"},
                                 memory_dir=self.mem, params={**self.params, "floor": 0.9},
                                 query_fn=_fake_query(HITS))
        self.assertIsNone(out)
        rec = self._log()[0]
        self.assertFalse(rec["would_inject"])
        self.assertFalse(rec["injected"])

    def test_log_mode_writes_jsonl_and_emits_nothing(self):
        out = ph.run_prompt_hook({"prompt": "why does my piped command report success?"},
                                 memory_dir=self.mem, params={**self.params, "mode": "log"},
                                 query_fn=_fake_query(HITS))
        self.assertIsNone(out)
        rec = self._log()[0]
        self.assertEqual(rec["mode"], "log")
        self.assertTrue(rec["would_inject"])
        self.assertFalse(rec["injected"])
        self.assertEqual(rec["hits"][0]["score"], 0.8)

    def test_off_mode_logs_nothing(self):
        out = ph.run_prompt_hook({"prompt": "why does my piped command report success?"},
                                 memory_dir=self.mem, params={**self.params, "mode": "off"},
                                 query_fn=_fake_query(HITS))
        self.assertIsNone(out)
        self.assertEqual(self._log(), [])

    def test_timeout_emits_nothing(self):
        def q(prompt, k=3, clearance="topsecret", deadline_ms=800):
            return [], "timeout"
        out = ph.run_prompt_hook({"prompt": "why does my piped command report success?"},
                                 memory_dir=self.mem, params=self.params, query_fn=q)
        self.assertIsNone(out)
        self.assertEqual(self._log()[0]["source"], "timeout")

    def test_block_respects_byte_cap_on_line_boundary(self):
        big = [{"id": i, "label": f"stem_{i}", "hook": "x" * 80, "score": 0.9}
               for i in range(20)]
        block = ph.render_block(big)
        self.assertLessEqual(len(block.encode()), ph.MAX_BLOCK_BYTES)
        for line in block.splitlines()[1:]:
            self.assertTrue(line.startswith("- [stem_"))
            self.assertTrue(line.endswith("x"))

    def test_cli_reads_stdin_and_prints_json(self):
        payload = json.dumps({"prompt": "why does my piped command report success?"})
        buf = io.StringIO()
        with mock.patch.object(ph, "find_memory_dir", lambda: self.mem), \
                mock.patch.object(ph, "query_hits", _fake_query(HITS)), \
                mock.patch("sys.stdin", io.StringIO(payload)), redirect_stdout(buf):
            rc = memsom_cli.main(["hook-prompt"])
        self.assertIn(rc, (None, 0))
        doc = json.loads(buf.getvalue())
        self.assertIn("feedback_piped_exit_codes", doc["hookSpecificOutput"]["additionalContext"])

    def test_cli_garbage_stdin_is_silent_exit_0(self):
        buf = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO("not json")), redirect_stdout(buf):
            rc = memsom_cli.main(["hook-prompt"])
        self.assertIn(rc, (None, 0))
        self.assertEqual(buf.getvalue(), "")

    def test_params_come_from_canonical_json(self):
        weights = self.mem / ".weights"
        weights.mkdir()
        (weights / "canonical.json").write_text(json.dumps(
            {"params": {"prompt_hook_mode": "log", "prompt_hook_floor": 0.5,
                        "prompt_hook_deadline_ms": 300, "prompt_hook_log_max_mb": 2}}))
        p = ph.load_hook_params(self.mem)
        self.assertEqual(p, {"mode": "log", "floor": 0.5, "deadline_ms": 300,
                             "log_max_mb": 2.0, "project_bytes": 1024, "project_max": 2})

    def test_bad_params_fall_back_to_defaults(self):
        weights = self.mem / ".weights"
        weights.mkdir()
        (weights / "canonical.json").write_text(json.dumps(
            {"params": {"prompt_hook_mode": "loud", "prompt_hook_floor": 7,
                        "prompt_hook_deadline_ms": 1, "prompt_hook_log_max_mb": 0}}))
        p = ph.load_hook_params(self.mem)
        d = forget.PANEL_PARAM_DEFAULTS
        self.assertEqual(p["mode"], d["prompt_hook_mode"])
        self.assertEqual(p["floor"], d["prompt_hook_floor"])
        self.assertEqual(p["deadline_ms"], d["prompt_hook_deadline_ms"])
        self.assertEqual(p["log_max_mb"], d["prompt_hook_log_max_mb"])

    def test_scaffold_seeds_the_hook_params(self):
        bi.scaffold_memory_dir(self.mem)
        canon = json.loads((self.mem / ".weights" / "canonical.json").read_text())
        self.assertEqual(canon["params"]["prompt_hook_mode"], "inject")
        self.assertIn("prompt_hook_floor", canon["params"])


# ---------------------------------------------------------------------------
# log rotation + stats
# ---------------------------------------------------------------------------

class TestProjectMatch(unittest.TestCase):
    """P2 auto-load: the alias matcher (pure) and the hook's project block."""

    from memsom.bridge import project as _proj

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = Path(self.tmp.name)
        (self.mem / ".weights").mkdir()
        self.params = {"mode": "inject", "floor": 0.35, "deadline_ms": 800,
                       "log_max_mb": 20, "project_bytes": 1024, "project_max": 2}

    def tearDown(self):
        self.tmp.cleanup()

    def _cache(self, projects):
        cache = {"version": 1, "built_at": "t", "projects": projects}
        (self.mem / ".weights" / "project_aliases.json").write_text(
            json.dumps(cache), encoding="utf-8")
        return cache

    def _entry(self, aliases=(), block="## Status\n### Done", status="active"):
        return {"aliases": list(aliases), "status": status, "headline": "h",
                "last_verified": "2026-09-02", "features": "1 implemented",
                "path": "projects/x/project_x.md", "block": block}

    def _log(self):
        p = ph.log_path(self.mem)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l] \
            if p.exists() else []

    # --- pure matcher ---
    def test_boundary_phrase_and_case(self):
        cache = {"projects": {"mspanel": self._entry(aliases=["ms panel", "brain platform"])}}
        self.assertEqual(self._proj.match_projects("How is MSPANEL doing", cache, 2), (["mspanel"], []))
        self.assertEqual(self._proj.match_projects("the ms panel status", cache, 2), (["mspanel"], []))
        self.assertEqual(self._proj.match_projects("brain PLATFORM health", cache, 2), (["mspanel"], []))
        # substring inside a longer word does NOT match (word boundary)
        self.assertEqual(self._proj.match_projects("mspanelization theory", cache, 2), ([], []))
        self.assertEqual(self._proj.match_projects("nothing here", cache, 2), ([], []))

    def test_two_matches_ordered_by_position(self):
        cache = {"projects": {"memsom": self._entry(), "ondemand": self._entry()}}
        self.assertEqual(
            self._proj.match_projects("first ondemand then memsom", cache, 2),
            (["ondemand", "memsom"], []))

    def test_third_match_becomes_also_trailer(self):
        cache = {"projects": {"a1x": self._entry(), "b2x": self._entry(), "c3x": self._entry()}}
        primary, also = self._proj.match_projects("a1x then b2x then c3x", cache, 2)
        self.assertEqual(primary, ["a1x", "b2x"])
        self.assertEqual(also, ["c3x"])

    # --- hook integration ---
    def test_project_block_precedes_retrieval_block(self):
        self._cache({"mspanel": self._entry(aliases=["ms panel"])})
        out = ph.run_prompt_hook({"prompt": "how is the ms panel doing lately"},
                                 memory_dir=self.mem, params=self.params,
                                 query_fn=_fake_query(HITS))
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertTrue(ctx.startswith("[memsom project: mspanel |"))
        self.assertIn("\n\nRelevant memories:", ctx)          # retrieval block after
        self.assertLess(ctx.index("mspanel"), ctx.index("Relevant memories:"))

    def test_matched_node_dropped_from_hits_subnotes_stay(self):
        self._cache({"mspanel": self._entry(aliases=["ms panel"])})
        hits = [{"id": 1, "label": "project_mspanel", "hook": "the node", "score": 0.9},
                {"id": 2, "label": "project_mspanel_gotchas", "hook": "a gotcha", "score": 0.9}]
        out = ph.run_prompt_hook({"prompt": "tell me about the ms panel please"},
                                 memory_dir=self.mem, params=self.params,
                                 query_fn=_fake_query(hits))
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertNotIn("[project_mspanel]", ctx)            # node hit dropped
        self.assertIn("[project_mspanel_gotchas]", ctx)       # sub-note hit stays

    def test_three_char_alias_prompt_gets_block(self):
        self._cache({"ondemand": self._entry(aliases=["ond"])})
        out = ph.run_prompt_hook({"prompt": "ond"}, memory_dir=self.mem,
                                 params=self.params, query_fn=_fake_query(HITS))
        self.assertIsNotNone(out)                             # short, but alias matched
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertTrue(ctx.startswith("[memsom project: ondemand |"))
        self.assertNotIn("Relevant memories:", ctx)          # BM25 skipped under 12 chars

    def test_slash_still_skipped_even_with_alias(self):
        self._cache({"mspanel": self._entry(aliases=["ms panel"])})
        out = ph.run_prompt_hook({"prompt": "/status mspanel"}, memory_dir=self.mem,
                                 params=self.params, query_fn=_fake_query(HITS))
        self.assertIsNone(out)
        self.assertEqual(self._log(), [])

    def test_absent_cache_matches_prior_output(self):
        # no project_aliases.json → identical output to the pre-P2 retrieval-only path
        out = ph.run_prompt_hook({"prompt": "why does my piped command report success?"},
                                 memory_dir=self.mem, params=self.params,
                                 query_fn=_fake_query(HITS))
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        self.assertTrue(ctx.startswith("Relevant memories:"))

    def test_log_carries_projects(self):
        self._cache({"mspanel": self._entry(aliases=["ms panel"])})
        ph.run_prompt_hook({"prompt": "how is the ms panel doing today"},
                           memory_dir=self.mem, params=self.params,
                           query_fn=_fake_query(HITS))
        rec = self._log()[0]
        self.assertEqual(rec["projects"], ["mspanel"])
        self.assertGreater(rec["project_bytes"], 0)


class TestLogRotationAndStats(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rotation_keeps_three_generations(self):
        base = ph.log_path(self.mem)
        base.parent.mkdir()
        max_mb = 0.001   # ~1 KB cap so a few records trip it
        for gen in range(5):
            base.write_text(f"gen{gen}\n" * 400, encoding="utf-8")   # > 1 KB
            ph.append_log(self.mem, {"n": gen}, max_mb)
        names = sorted(p.name for p in base.parent.iterdir())
        self.assertEqual(names, ["hook_log.1.jsonl", "hook_log.2.jsonl",
                                 "hook_log.3.jsonl", "hook_log.jsonl"])
        self.assertTrue(base.read_text().startswith('{"n": 4}'))
        self.assertIn("gen4", (base.parent / "hook_log.1.jsonl").read_text())
        self.assertIn("gen2", (base.parent / "hook_log.3.jsonl").read_text())
        self.assertEqual(ph.rotated_paths(self.mem)[0], base)

    def test_no_rotation_under_cap(self):
        ph.append_log(self.mem, {"n": 1}, 20)
        ph.append_log(self.mem, {"n": 2}, 20)
        self.assertFalse(ph.rotate_if_needed(self.mem, 20))
        self.assertEqual(len(ph.rotated_paths(self.mem)), 1)

    def _seed_log(self):
        recs = [
            {"mode": "inject", "source": "warm", "ms": 12, "would_inject": True, "injected": True,
             "hits": [{"stem": "a", "score": 0.9}, {"stem": "b", "score": 0.5}]},
            {"mode": "inject", "source": "bm25", "ms": 40, "would_inject": False, "injected": False,
             "hits": [{"stem": "c", "score": 0.2}]},
            {"mode": "log", "source": "warm", "ms": 9, "would_inject": True, "injected": False,
             "hits": [{"stem": "a", "score": 0.7}]},
            {"mode": "log", "source": "timeout", "ms": 800, "would_inject": False,
             "injected": False, "hits": []},
        ]
        for r in recs:
            ph.append_log(self.mem, r, 20)
        # a rotated generation counts too
        (self.mem / ".weights" / "hook_log.1.jsonl").write_text(
            json.dumps({"mode": "inject", "source": "warm", "would_inject": True,
                        "injected": True, "hits": [{"stem": "a", "score": 0.95}]}) + "\n")

    def test_summary_counts(self):
        self._seed_log()
        s = ph.summarize_log(ph.iter_log_records(self.mem))
        self.assertEqual(s["queries"], 5)
        self.assertEqual(s["injected"], 2)
        self.assertEqual(s["would_inject"], 3)
        self.assertEqual(s["inject_rate"], 0.4)
        self.assertEqual(s["by_mode"], {"inject": 3, "log": 2})
        self.assertEqual(s["by_source"]["timeout"], 1)
        self.assertEqual(s["top_stems"][0], ("a", 3))
        hist = {row["bin"]: row["n"] for row in s["top1_histogram"]}
        self.assertEqual(hist["0.9-1.0"], 2)
        self.assertEqual(hist["0.2-0.3"], 1)
        sweep = {r["floor"]: r["would_inject_rate"] for r in s["floor_sweep"]}
        self.assertEqual(sweep[0.0], 0.8)      # 4 of 5 had a top-1 score at all
        self.assertEqual(sweep[0.8], 0.4)

    def test_hook_stats_cli_json(self):
        self._seed_log()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = memsom_cli.main(["hook-stats", "--memory-dir", str(self.mem), "--json"])
        self.assertIn(rc, (None, 0))
        out = json.loads(buf.getvalue())
        self.assertEqual(out["queries"], 5)
        self.assertEqual(len(out["log_files"]), 2)

    def test_hook_stats_cli_text(self):
        self._seed_log()
        buf = io.StringIO()
        with redirect_stdout(buf):
            memsom_cli.main(["hook-stats", "--memory-dir", str(self.mem)])
        text = buf.getvalue()
        self.assertIn("queries        : 5", text)
        self.assertIn("top surfaced stems", text)
        self.assertIn("would-inject rate by floor", text)


# ---------------------------------------------------------------------------
# wire-claude parity
# ---------------------------------------------------------------------------

class TestWireClaudePromptHook(unittest.TestCase):
    EXE = "/abs/memsom"

    def test_adds_prompt_hook_once(self):
        data = {}
        wc.merge_hooks(data, self.EXE)
        self.assertEqual(wc.merge_hooks(data, self.EXE), [])
        ups = data["hooks"]["UserPromptSubmit"]
        cmds = [h["command"] for g in ups for h in g["hooks"]]
        self.assertEqual(cmds, ['"/abs/memsom" hook-prompt'])
        self.assertEqual(ups[0]["hooks"][0]["timeout"], wc.PROMPT_HOOK_TIMEOUT_S)

    def test_upgrades_an_old_entry_in_place_and_drops_duplicates(self):
        data = {"hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "other"},
                       {"type": "command", "command": '"/old/memsom" hook-prompt'}]},
            {"hooks": [{"type": "command", "command": '"/older/memsom" hook-prompt'}]},
        ]}}
        changed = wc.merge_hooks(data, self.EXE)
        self.assertIn("UserPromptSubmit", changed)
        ups = data["hooks"]["UserPromptSubmit"]
        cmds = [h["command"] for g in ups for h in g["hooks"]]
        self.assertEqual(cmds, ["other", '"/abs/memsom" hook-prompt'])
        self.assertEqual(ups[0]["hooks"][1], wc.prompt_hook_entry(self.EXE))
        self.assertEqual(wc.merge_hooks(data, self.EXE), [])

    def test_wire_settings_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "settings.json"
            path.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": '"/old/memsom" hook-prompt'}]}]}}))
            res = wc.wire_settings(path, self.EXE)
            self.assertEqual(res["action"], "merged")
            self.assertEqual(sorted(res["events"]), ["Stop", "UserPromptSubmit"])
            self.assertEqual(wc.wire_settings(path, self.EXE)["action"], "unchanged")
            data = json.loads(path.read_text())
            self.assertEqual(data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"],
                             '"/abs/memsom" hook-prompt')

    def test_managed_block_mentions_retrieve_and_the_hook(self):
        from memsom.bridge import claude as mc
        self.assertIn("memsom retrieve", mc.CANONICAL)
        self.assertIn("prompt hook", mc.CANONICAL)
        self.assertIn("/recall", mc.CANONICAL)


# ---------------------------------------------------------------------------
# plugin packaging
# ---------------------------------------------------------------------------

class TestPluginPackaging(unittest.TestCase):
    def _load(self, rel):
        return json.loads((HERE / rel).read_text(encoding="utf-8"))

    def test_plugin_manifest(self):
        m = self._load(".claude-plugin/plugin.json")
        for key in ("name", "version", "description"):
            self.assertIn(key, m)
        self.assertEqual(m["name"], "memsom")
        for key in ("skills", "hooks", "mcpServers"):
            self.assertTrue(m[key].startswith("./"), key)
            self.assertTrue((HERE / m[key]).exists(), m[key])
        self.assertTrue(any((HERE / m["skills"]).glob("*/SKILL.md")))

    def test_hooks_json_schema(self):
        h = self._load("hooks/hooks.json")
        self.assertEqual(set(h), {"hooks"})
        events = h["hooks"]
        self.assertEqual(set(events), {"Stop", "UserPromptSubmit"})
        for event, groups in events.items():
            self.assertIsInstance(groups, list)
            for g in groups:
                self.assertNotIn("matcher", g)        # neither event supports matchers
                for hk in g["hooks"]:
                    self.assertEqual(hk["type"], "command")
                    self.assertEqual(hk["command"], "memsom")
                    self.assertIsInstance(hk["args"], list)
                    self.assertIsInstance(hk["timeout"], int)
        self.assertEqual(events["Stop"][0]["hooks"][0]["args"], ["bridge-render"])
        ups = events["UserPromptSubmit"][0]["hooks"][0]
        self.assertEqual(ups["args"], ["hook-prompt"])
        self.assertLessEqual(ups["timeout"], 30)

    def test_mcp_json(self):
        m = self._load(".mcp.json")
        srv = m["mcpServers"]["memsom"]
        self.assertEqual(srv["command"], "memsom-mcp")
        self.assertEqual(srv.get("args", []), [])

    def test_marketplace(self):
        m = self._load(".claude-plugin/marketplace.json")
        self.assertEqual(m["name"], "memsom")
        self.assertIn("name", m["owner"])
        entry = m["plugins"][0]
        self.assertEqual(entry["name"], "memsom")
        self.assertEqual(entry["source"], "./")
        plugin = self._load(".claude-plugin/plugin.json")
        self.assertEqual(entry["version"], plugin["version"])

    def test_no_machine_specific_paths_in_shipped_files(self):
        shipped = [".claude-plugin/plugin.json", ".claude-plugin/marketplace.json",
                   "hooks/hooks.json", ".mcp.json"]
        shipped += [str(p.relative_to(HERE)) for p in (HERE / "claude").rglob("*") if p.is_file()]
        for rel in shipped:
            text = (HERE / rel).read_text(encoding="utf-8")
            for bad in ("C:\\Users", "/Users/", "/home/", "192.168."):
                self.assertNotIn(bad, text, f"{rel} contains {bad!r}")

    def test_hook_subcommands_exist(self):
        for sub in ("hook-prompt", "hook-query", "hook-stats", "bridge-render"):
            with self.assertRaises(SystemExit) as cm, redirect_stdout(io.StringIO()):
                memsom_cli.main([sub, "--help"])
            self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
