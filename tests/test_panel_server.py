"""Tests for the panel HTTP server: bind policy, Host/Origin/Content-Type
write-protection, CSP hash exactness, the memory-cache TTL, and end-to-end
knob GET/POST round-trips against a real ThreadingHTTPServer on an ephemeral
port. Never touches live memsom state — synthetic profiles in temp dirs only.

Run:  python -m pytest tests/test_panel_server.py -q
"""

import http.client
import json
import re
import subprocess
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memsom.interface import panel


def _write_profile(dirpath, *, knobs=None, tasks=None, contracts=None):
    profile = {
        "knobs": knobs or [],
        "tasks": tasks or [],
        "contracts": contracts or [],
        "telemetry": {},
        "audit_log": str(Path(dirpath) / "audit.jsonl"),
    }
    path = Path(dirpath) / "panel_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def _fake_memory_builder():
    return {
        "generated": "2026-01-01 00:00 UTC",
        "totals": {"total": 3, "hot": 2, "cold": 1, "pinned": 1},
        "tier": {"hot": 2, "cold": 1},
        "types": {"user": 3},
        "hist": {"labels": ["0.0-0.1"], "data": [1]},
        "top_access": [],
        "scatter": [],
        "growth": [],
        "stale": [],
        "budget": {"bytes": 100, "cap": 16384, "pct": 0.6},
        "sessions": None,
        "thresholds": {"demote_below": 0.2, "promote_at": 0.5},
        "graph": {"nodes": [], "links": [], "sections": []},
    }


class LiveServer:
    """Starts a real ThreadingHTTPServer on an ephemeral port in a background
    thread; tears it down cleanly."""

    def __init__(self, profile_path, **build_kwargs):
        build_kwargs.setdefault("memory_builder", _fake_memory_builder)
        self.config = panel.build_config(profile_path, host="127.0.0.1", port=0, **build_kwargs)
        self.httpd = panel.build_server(self.config)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def get(self, path, host_header=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            if host_header is not None:
                conn.putrequest("GET", path, skip_host=True)
                conn.putheader("Host", host_header)
                conn.endheaders()
            else:
                conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read()
            return resp, body
        finally:
            conn.close()

    def post(self, path, payload_bytes, *, content_type="application/json",
             origin=None, host_header=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            headers = {}
            if content_type is not None:
                headers["Content-Type"] = content_type
            if origin is not None:
                headers["Origin"] = origin
            headers["Content-Length"] = str(len(payload_bytes))
            if host_header is not None:
                conn.putrequest("POST", path, skip_host=True)
                conn.putheader("Host", host_header)
                for k, v in headers.items():
                    conn.putheader(k, v)
                conn.endheaders()
                conn.send(payload_bytes)
            else:
                conn.request("POST", path, body=payload_bytes, headers=headers)
            resp = conn.getresponse()
            body = resp.read()
            return resp, body
        finally:
            conn.close()


class BindPolicyTests(unittest.TestCase):
    def test_non_loopback_host_refused_at_build_time(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            with self.assertRaises(ValueError):
                panel.build_config(profile_path, host="0.0.0.0", port=0)

    def test_wildcard_ipv6_refused(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            with self.assertRaises(ValueError):
                panel.build_config(profile_path, host="::", port=0)

    def test_loopback_spellings_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            for host in ("127.0.0.1", "localhost", "::1"):
                panel.build_config(profile_path, host=host, port=0)  # no raise


class HealthTests(unittest.TestCase):
    def test_health_shape(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/health")
                self.assertEqual(resp.status, 200)
                data = json.loads(body)
                self.assertEqual(data["app"], "memsom-panel")
                self.assertIn("version", data)
                self.assertTrue(data["ok"])
            finally:
                srv.close()


class HostHeaderTests(unittest.TestCase):
    def test_foreign_host_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/health", host_header="evil.example")
                self.assertEqual(resp.status, 403)
            finally:
                srv.close()

    def test_host_header_wrong_port_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/health", host_header=f"127.0.0.1:1")
                self.assertEqual(resp.status, 403)
            finally:
                srv.close()

    def test_matching_host_header_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/health", host_header=f"127.0.0.1:{srv.port}")
                self.assertEqual(resp.status, 200)
            finally:
                srv.close()

    def test_bracketed_ipv6_host_header_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/health", host_header=f"[::1]:{srv.port}")
                self.assertEqual(resp.status, 200)
            finally:
                srv.close()


class PostWriteProtectionTests(unittest.TestCase):
    def test_post_without_json_content_type_is_403(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.post("/api/knobs", b"{}", content_type="text/plain")
                self.assertEqual(resp.status, 403)
            finally:
                srv.close()

    def test_post_with_foreign_origin_is_403(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.post("/api/knobs", b"{}", origin="http://evil.example")
                self.assertEqual(resp.status, 403)
            finally:
                srv.close()

    def test_post_with_self_origin_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "tunables.json"
            knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                      "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 100}, "default": 1}]
            profile_path = _write_profile(d, knobs=knobs)
            srv = LiveServer(profile_path)
            try:
                origin = f"http://127.0.0.1:{srv.port}"
                payload = json.dumps({"id": "k1", "value": 7}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload, origin=origin)
                self.assertEqual(resp.status, 200)
                result = json.loads(body)
                self.assertTrue(result["ok"])
                self.assertEqual(result["current"], 7)
            finally:
                srv.close()

    def test_post_with_no_origin_header_succeeds(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "tunables.json"
            knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                      "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 100}, "default": 1}]
            profile_path = _write_profile(d, knobs=knobs)
            srv = LiveServer(profile_path)
            try:
                payload = json.dumps({"id": "k1", "value": 3}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 200)
            finally:
                srv.close()

    def test_post_host_header_rejection_applies_too(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.post("/api/knobs", b"{}", host_header="evil.example")
                self.assertEqual(resp.status, 403)
            finally:
                srv.close()


class CspHashTests(unittest.TestCase):
    def test_csp_hash_matches_served_script_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/")
                self.assertEqual(resp.status, 200)
                csp = resp.getheader("Content-Security-Policy")
                self.assertIsNotNone(csp)
                m = re.search(r"script-src 'sha256-([^']+)'", csp)
                self.assertIsNotNone(m)
                claimed_hash = m.group(1)

                html_text = body.decode("utf-8")
                script_m = re.search(r"<script>(.*?)</script>", html_text, re.DOTALL)
                self.assertIsNotNone(script_m)
                script_body = script_m.group(1)

                import base64
                import hashlib
                actual_hash = base64.b64encode(
                    hashlib.sha256(script_body.encode("utf-8")).digest()).decode("ascii")
                self.assertEqual(claimed_hash, actual_hash)

                # other required security headers present
                self.assertEqual(resp.getheader("X-Content-Type-Options"), "nosniff")
                self.assertEqual(resp.getheader("Referrer-Policy"), "no-referrer")
                self.assertEqual(resp.getheader("X-Frame-Options"), "DENY")
                self.assertEqual(resp.getheader("Cache-Control"), "no-store")
                self.assertIn("default-src 'none'", csp)

                # no inline event handlers - wired via addEventListener only
                self.assertNotIn("onclick=", html_text)
                self.assertNotIn("onchange=", html_text)
            finally:
                srv.close()


class KnobRoundTripTests(unittest.TestCase):
    def test_get_then_post_then_get_reflects_new_value(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "tunables.json"
            knobs = [{"id": "k1", "tier": 1, "label": "K1", "provider": "json-file",
                      "target": str(target), "key": "a.b", "type": "int",
                      "bounds": {"min": 0, "max": 100}, "default": 10}]
            profile_path = _write_profile(d, knobs=knobs)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/api/knobs")
                self.assertEqual(resp.status, 200)
                data = json.loads(body)
                row = next(k for k in data["knobs"] if k["id"] == "k1")
                self.assertEqual(row["current"], 10)  # absent file -> default
                self.assertFalse(row["dirty"])

                payload = json.dumps({"id": "k1", "value": 55}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 200)
                self.assertTrue(json.loads(body)["ok"])

                resp, body = srv.get("/api/knobs")
                data = json.loads(body)
                row = next(k for k in data["knobs"] if k["id"] == "k1")
                self.assertEqual(row["current"], 55)
                self.assertTrue(row["dirty"])
            finally:
                srv.close()

    def test_reset_restores_default(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "tunables.json"
            target.write_text(json.dumps({"a": {"b": 999}}), encoding="utf-8")
            knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                      "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 1000}, "default": 42}]
            profile_path = _write_profile(d, knobs=knobs)
            srv = LiveServer(profile_path)
            try:
                payload = json.dumps({"id": "k1", "reset": True}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 200)
                self.assertEqual(json.loads(body)["current"], 42)
            finally:
                srv.close()

    def test_out_of_bounds_post_returns_400_and_error_body(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "tunables.json"
            knobs = [{"id": "k1", "tier": 1, "provider": "json-file", "target": str(target),
                      "key": "a.b", "type": "int", "bounds": {"min": 0, "max": 10}, "default": 1}]
            profile_path = _write_profile(d, knobs=knobs)
            srv = LiveServer(profile_path)
            try:
                payload = json.dumps({"id": "k1", "value": 999}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 400)
                data = json.loads(body)
                self.assertFalse(data["ok"])
                self.assertIn("maximum", data["error"])
            finally:
                srv.close()


class MemoryTtlTests(unittest.TestCase):
    def test_two_calls_share_cached_at_then_refresh_bumps_it(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            # Real wall-clock resolution can repeat within a single test's
            # runtime; inject a strictly-incrementing stamp so two distinct
            # builds are always distinguishable.
            counter = {"n": 0}

            def fake_now():
                counter["n"] += 1
                return f"stamp-{counter['n']}"

            srv = LiveServer(profile_path, memory_now=fake_now)
            try:
                resp1, body1 = srv.get("/api/memory")
                self.assertEqual(resp1.status, 200)
                data1 = json.loads(body1)

                resp2, body2 = srv.get("/api/memory")
                data2 = json.loads(body2)
                self.assertEqual(data1["cached_at"], data2["cached_at"])

                resp3, body3 = srv.get("/api/memory?refresh=1")
                data3 = json.loads(body3)
                self.assertNotEqual(data1["cached_at"], data3["cached_at"])
            finally:
                srv.close()


class SystemRouteSmokeTest(unittest.TestCase):
    def test_api_system_returns_shape(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path, run=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
            try:
                resp, body = srv.get("/api/system")
                self.assertEqual(resp.status, 200)
                data = json.loads(body)
                for key in ("ts", "ram", "top_procs", "gpu", "disks", "probes", "syncthing"):
                    self.assertIn(key, data)
            finally:
                srv.close()


class SchtasksThroughServerTests(unittest.TestCase):
    def _fake_run(self, *, read_ok=True, write_ok=True):
        def run(cmd, **kwargs):
            script = cmd[-1]
            if "Get-ScheduledTask" in script:
                if read_ok:
                    return subprocess.CompletedProcess(
                        cmd, 0,
                        stdout='{"trigger":"Daily","enabled":true,"lastRunTime":"a","nextRunTime":"b"}',
                        stderr="")
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="denied")
            # Set-ScheduledTask (write)
            if write_ok:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="access is denied")
        return run

    def test_read_via_api_knobs(self):
        with tempfile.TemporaryDirectory() as d:
            knobs = [{"id": "t1", "tier": 2, "label": "Writable Task", "provider": "schtasks",
                      "key": "WritableTask", "type": "int", "bounds": {"min": 1, "max": 60},
                      "default": 5}]
            tasks = [{"name": "WritableTask", "writable": True}]
            profile_path = _write_profile(d, knobs=knobs, tasks=tasks)
            srv = LiveServer(profile_path, run=self._fake_run())
            try:
                resp, body = srv.get("/api/knobs")
                self.assertEqual(resp.status, 200)
                data = json.loads(body)
                row = next(k for k in data["knobs"] if k["id"] == "t1")
                self.assertEqual(row["task"]["trigger"], "Daily")
                self.assertTrue(row["task"]["enabled"])
            finally:
                srv.close()

    def test_degraded_write_returns_200_with_elevated_command(self):
        with tempfile.TemporaryDirectory() as d:
            knobs = [{"id": "t1", "tier": 2, "provider": "schtasks", "key": "WritableTask",
                      "type": "int", "bounds": {"min": 1, "max": 60}, "default": 5}]
            tasks = [{"name": "WritableTask", "writable": True}]
            profile_path = _write_profile(d, knobs=knobs, tasks=tasks)
            srv = LiveServer(profile_path, run=self._fake_run(write_ok=False))
            try:
                payload = json.dumps({"id": "t1", "value": 10}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 200)
                data = json.loads(body)
                self.assertFalse(data["ok"])
                self.assertTrue(data["degraded"])
                self.assertIn("elevated_command", data)
            finally:
                srv.close()

    def test_readonly_task_write_refused(self):
        with tempfile.TemporaryDirectory() as d:
            knobs = [{"id": "t1", "tier": 2, "provider": "schtasks", "key": "LockedTask",
                      "type": "int", "bounds": {"min": 1, "max": 60}, "default": 5}]
            tasks = [{"name": "LockedTask", "writable": False}]
            profile_path = _write_profile(d, knobs=knobs, tasks=tasks)
            srv = LiveServer(profile_path, run=self._fake_run())
            try:
                payload = json.dumps({"id": "t1", "value": 10}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 403)
                data = json.loads(body)
                self.assertFalse(data["ok"])
                self.assertIn("read-only", data["error"])
            finally:
                srv.close()


class Tier3ContractTests(unittest.TestCase):
    def test_post_to_contract_is_403_via_http(self):
        with tempfile.TemporaryDirectory() as d:
            contracts = [{"id": "dense_dim", "label": "dim", "value": 1024, "note": "fixed"}]
            profile_path = _write_profile(d, contracts=contracts)
            srv = LiveServer(profile_path)
            try:
                payload = json.dumps({"id": "dense_dim", "value": 2048}).encode("utf-8")
                resp, body = srv.post("/api/knobs", payload)
                self.assertEqual(resp.status, 403)
            finally:
                srv.close()

    def test_get_api_knobs_includes_contracts_list(self):
        with tempfile.TemporaryDirectory() as d:
            contracts = [{"id": "dense_dim", "label": "dim", "value": 1024, "note": "fixed"}]
            profile_path = _write_profile(d, contracts=contracts)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/api/knobs")
                data = json.loads(body)
                self.assertEqual(len(data["contracts"]), 1)
                self.assertEqual(data["contracts"][0]["id"], "dense_dim")
                self.assertEqual(data["contracts"][0]["tier"], 3)
            finally:
                srv.close()


class MemoryTouchTests(unittest.TestCase):
    """_read_memory_touches: the live feed behind the DASHBOARD mesh glow.

    The contract that matters is NEGATIVE — every heuristic in there scans free
    text, so the tests exist mainly to prove a touch can never name a node the
    memory dir doesn't actually contain."""

    SESSION = "aaaaaaaa-1111-2222-3333-444444444444"

    def _fixture(self, d, records, *, stems=("project_alpha", "user_beta"),
                 index="", activity=()):
        """Build a fake ~/.claude + memory dir. Records are written with a
        timestamp of `now` unless they carry their own."""
        root = Path(d)
        claude = root / ".claude"
        mem = root / "memory"
        (claude / "projects" / "proj").mkdir(parents=True)
        (claude / "episodic").mkdir(parents=True)
        mem.mkdir()
        for stem in stems:
            (mem / f"{stem}.md").write_text("x", encoding="utf-8")
        (mem / "MEMORY.md").write_text(index, encoding="utf-8")

        now = datetime.now(timezone.utc)
        iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = []
        for rec in records:
            rec.setdefault("timestamp", iso)
            rec.setdefault("sessionId", self.SESSION)
            lines.append(json.dumps(rec))
        (claude / "projects" / "proj" / f"{self.SESSION}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        if activity:
            (claude / "episodic" / "memory_activity.jsonl").write_text(
                "\n".join(json.dumps(dict(e, ts=e.get("ts", iso)))
                          for e in activity) + "\n", encoding="utf-8")
        return claude, mem, now

    @staticmethod
    def _assistant(*blocks):
        return {"message": {"role": "assistant", "content": list(blocks)}}

    def _stems(self, payload):
        return sorted(t["stem"] for t in payload["touches"] if t["stem"])

    def test_read_tool_on_memory_file_is_a_touch(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(mem / "project_alpha.md")}})])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(self._stems(out), ["project_alpha"])
            self.assertEqual(out["touches"][0]["source"], "read")
            self.assertEqual(out["chats"][0]["short"], self.SESSION[:8])

    def test_bash_cat_of_a_memory_file_is_a_touch(self):
        # Memories get read through Bash at least as often as through Read.
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": "cat /c/x/memory/user_beta.md | head -5"}})])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(self._stems(out), ["user_beta"])
            self.assertEqual(out["touches"][0]["source"], "bash")

    def test_unknown_stem_is_never_emitted(self):
        # The whole safety property: a path/token that looks like a memory but
        # has no file behind it must not reach the graph as a node id.
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(mem / "not_a_real_memory.md")}},
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": "cat memory/also_fake.md"}})])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(out["touches"], [])

    def test_memory_index_itself_is_not_a_node(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(mem / "MEMORY.md")}})])
            self.assertEqual(panel._read_memory_touches(claude, mem, now=now)["touches"], [])

    def test_file_outside_the_memory_dir_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(Path(d) / "elsewhere" / "project_alpha.md")}})])
            self.assertEqual(panel._read_memory_touches(claude, mem, now=now)["touches"], [])

    def test_memsom_id_arg_names_the_node_directly(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "mcp__memsom__explain",
                 "input": {"id": "user_beta"}})])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(self._stems(out), ["user_beta"])

    def test_memsom_query_emits_a_stemless_marker_then_result_stems(self):
        # retrieve/ask can't name a memory up front — the marker keeps the mesh
        # reacting, and the RESULT is what resolves to actual nodes.
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(d, [
                self._assistant({"type": "tool_use", "id": "toolu_1",
                                 "name": "mcp__memsom__retrieve",
                                 "input": {"query": "what does he lift"}}),
                {"message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": "1. project_alpha (0.81)\n2. nonexistent_thing (0.4)"}]}},
            ])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(self._stems(out), ["project_alpha"])
            self.assertTrue(any(t["stem"] is None and t["source"] == "memsom"
                                for t in out["touches"]))

    def test_result_of_a_non_memsom_tool_is_not_scanned(self):
        # A Read of some unrelated file that merely MENTIONS a stem is not a
        # touch of that memory — and scanning every result would be the
        # expensive path this feed deliberately avoids.
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(d, [
                self._assistant({"type": "tool_use", "id": "toolu_9", "name": "Read",
                                 "input": {"file_path": str(Path(d) / "notes.txt")}}),
                {"message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_9",
                     "content": "see project_alpha for details"}]}},
            ])
            self.assertEqual(panel._read_memory_touches(claude, mem, now=now)["touches"], [])

    def test_records_older_than_the_window_are_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            old = (datetime.now(timezone.utc) - timedelta(seconds=600)
                   ).strftime("%Y-%m-%dT%H:%M:%SZ")
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [dict(self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(mem / "project_alpha.md")}}), timestamp=old)])
            self.assertEqual(
                panel._read_memory_touches(claude, mem, window_s=90, now=now)["touches"], [])
            # …and the same record IS found with a window that covers it.
            self.assertEqual(
                len(panel._read_memory_touches(claude, mem, window_s=900, now=now)["touches"]), 1)

    def test_recall_event_titles_resolve_through_the_memory_index(self):
        # memory_activity.jsonl records a MIX of stems and human titles; only
        # MEMORY.md can map the titled half back to a node id.
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(
                d, [],
                index="- [Beta Facts](user_beta.md) - hook\n",
                activity=[{"session_id": "s1", "source": "askq_recall",
                           "memories": ["project_alpha", "Beta Facts",
                                        "session 68d319d4"]}])
            out = panel._read_memory_touches(claude, mem, now=now)
            # the episodic session title has no node and must simply vanish
            self.assertEqual(self._stems(out), ["project_alpha", "user_beta"])
            self.assertTrue(all(t["source"] == "recall" for t in out["touches"]))

    def test_voice_recall_is_its_own_chat(self):
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(
                d, [], activity=[{"session_id": "voice", "source": "voice_recall",
                                  "memories": ["project_alpha"]}])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(out["chats"][0]["kind"], "voice")
            self.assertEqual(out["chats"][0]["short"], "VOICE")
            self.assertEqual(out["touches"][0]["source"], "voice")

    def test_repeat_touches_collapse_to_the_newest_per_chat_and_stem(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(mem / "project_alpha.md")}})
                for _ in range(5)])
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(len(out["touches"]), 1)

    def test_malformed_lines_never_raise(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory"
            claude, _, now = self._fixture(d, [self._assistant(
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": str(mem / "project_alpha.md")}})])
            path = claude / "projects" / "proj" / f"{self.SESSION}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write("{not json at all\n[]\n\n")
            self.assertEqual(
                self._stems(panel._read_memory_touches(claude, mem, now=now)),
                ["project_alpha"])

    def test_empty_memory_dir_returns_an_empty_feed(self):
        with tempfile.TemporaryDirectory() as d:
            claude, mem, now = self._fixture(d, [], stems=())
            out = panel._read_memory_touches(claude, mem, now=now)
            self.assertEqual(out["touches"], [])
            self.assertEqual(out["chats"], [])

    def test_route_serves_the_feed(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/api/memory/touches")
                self.assertEqual(resp.status, 200)
                payload = json.loads(body)
                for key in ("now", "window_s", "chats", "touches"):
                    self.assertIn(key, payload)
            finally:
                srv.close()


class NotFoundTests(unittest.TestCase):
    def test_unknown_get_route_404(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.get("/nope")
                self.assertEqual(resp.status, 404)
            finally:
                srv.close()

    def test_unknown_post_route_404(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            srv = LiveServer(profile_path)
            try:
                resp, body = srv.post("/nope", b"{}")
                self.assertEqual(resp.status, 404)
            finally:
                srv.close()


if __name__ == "__main__":
    unittest.main()
