"""Kernel handler contract tests (+ the two in-passing fixes: the krn- filter
on the inference sessions list and the /api/agents/runs missing-send
regression, exercised over a live loopback server)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from memsom.providers import kernel_handlers as H
from memsom.providers.handlers import handle_inference_sessions
from memsom.providers.kernels import KernelRunner, KernelStore

from tests.test_panel_server import LiveServer, _write_profile


def _deps(d: Path):
    store = KernelStore(d / "kernels")
    runner = KernelRunner(store, d / "sessions", d / "claude")
    audit = d / "audit.jsonl"
    return store, runner, audit


class CreateTests(unittest.TestCase):
    def test_create_ok_records_cli_path(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store, runner, audit = _deps(d)
            # python is guaranteed present — a stand-in cli the which() passes.
            profile = {"providers": {"claude": {"cli_path": sys.executable}}}
            st, body = H.handle_kernel_create(
                store, profile, audit, {"name": "k", "engine": "claude",
                                        "cwd": str(d)})
            self.assertEqual(st, 201)
            self.assertEqual(body["kernel"]["cli_path"], sys.executable)
            self.assertIsNone(body["kernel"]["session_ptr"])

    def test_unknown_engine_400_missing_cli_501_bad_cwd_400(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store, runner, audit = _deps(d)
            st, _ = H.handle_kernel_create(store, {}, audit,
                                           {"engine": "gemini"})
            self.assertEqual(st, 400)
            st, body = H.handle_kernel_create(
                store, {"providers": {"claude": {
                    "cli_path": "definitely-not-a-real-cli-xyz"}}},
                audit, {"engine": "claude", "cwd": str(d)})
            self.assertEqual(st, 501)
            st, _ = H.handle_kernel_create(
                store, {"providers": {"claude": {"cli_path": sys.executable}}},
                audit, {"engine": "claude", "cwd": str(d / "nope")})
            self.assertEqual(st, 400)

    def test_audit_gate_refusal_503(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store, runner, _ = _deps(d)
            bad_audit = d / "as-a-dir"
            bad_audit.mkdir()
            profile = {"providers": {"claude": {"cli_path": sys.executable}}}
            st, body = H.handle_kernel_create(
                store, profile, bad_audit, {"engine": "claude", "cwd": str(d)})
            self.assertEqual(st, 503)
            self.assertIn("refused", body["error"])


class PromptAndLifecycleTests(unittest.TestCase):
    def test_prompt_404_and_busy_409(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store, runner, audit = _deps(d)
            st, _ = H.handle_kernel_prompt(store, runner, audit,
                                           "aaaabbbbcccc", {"prompt": "x"})
            self.assertEqual(st, 404)
            kernel = store.create("t", "claude", None, None, str(d),
                                  "claude")
            lock = runner._lock_for(kernel["kernel_id"])
            lock.acquire()
            try:
                st, body = H.handle_kernel_prompt(
                    store, runner, audit, kernel["kernel_id"],
                    {"prompt": "x"})
                self.assertEqual(st, 409)
                self.assertIn("busy", body["error"])
            finally:
                lock.release()

    def test_archive_hides_from_default_list(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store, runner, audit = _deps(d)
            kernel = store.create("t", "claude", None, None, str(d), "claude")
            st, _ = H.handle_kernel_archive(store, audit, kernel["kernel_id"])
            self.assertEqual(st, 200)
            st, body = H.handle_kernels_list(store, runner)
            self.assertEqual(body["kernels"], [])
            st, body = H.handle_kernels_list(store, runner,
                                             include_archived=True)
            self.assertEqual(len(body["kernels"]), 1)
            self.assertFalse(body["kernels"][0]["busy"])

    def test_kill_no_kernel_404(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            store, runner, audit = _deps(d)
            st, _ = H.handle_kernel_kill(store, runner, audit, "aaaabbbbcccc")
            self.assertEqual(st, 404)


class InferenceListFilterTests(unittest.TestCase):
    def test_krn_feeds_hidden_from_picker(self):
        class FakeRunner:
            def list_sessions(self):
                return [{"session_id": "abc123", "status": "done"},
                        {"session_id": "krn-aaaabbbbcccc-12345678",
                         "status": "done"}]

        st, body = handle_inference_sessions(FakeRunner())
        self.assertEqual(st, 200)
        ids = [s["session_id"] for s in body["sessions"]]
        self.assertEqual(ids, ["abc123"])


class RouteTests(unittest.TestCase):
    """Over a live loopback server: the kernel routes exist and the
    /api/agents/runs regression (route computed a body but never sent it,
    hanging the connection) stays fixed."""

    def test_agents_runs_sends_and_kernels_list_empty(self):
        with tempfile.TemporaryDirectory() as d:
            profile_path = _write_profile(d)
            server = LiveServer(profile_path)
            try:
                resp, body = server.get("/api/agents/runs")
                self.assertEqual(resp.status, 200)
                body = json.loads(body)
                self.assertTrue(body.get("ok"))
                self.assertIn("runs", body)
                resp, body = server.get("/api/kernels")
                self.assertEqual(resp.status, 200)
                body = json.loads(body)
                self.assertTrue(body.get("ok"))
                self.assertEqual(body["kernels"], [])
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
