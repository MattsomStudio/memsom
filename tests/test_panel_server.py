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
