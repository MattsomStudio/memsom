#!/usr/bin/env python3
"""Tests for memsom.interface.remote -- Sec3.5 point 7, fail closed.

No token / unknown token / revoked token all resolve to ZERO rows of store
content -- not a "degrade to local mode" fallback (Sec3.5 point 7 explicitly
rules that out: a remote client whose auth fails "refuses; it does not
quietly answer from a store it is not supposed to have").
"""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import memsom
from memsom.interface import cli as memsom_cli
from memsom.interface import remote as R


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_cli.migrate_all(self.conn)
        R.reset_device_sessions()
        self.nid = memsom.insert_node(self.conn, "the sky is blue today", "user")

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()


class TestFailClosed(Base):
    def test_no_token_denies_and_returns_no_content(self):
        r = R.handle_request(self.conn, "", "ask", {"question": "sky"})
        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["text"], "")
        self.assertTrue(r["is_error"])

    def test_none_token_denies(self):
        r = R.handle_request(self.conn, None, "ask", {"question": "sky"})
        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["text"], "")

    def test_unknown_token_denies(self):
        r = R.handle_request(self.conn, "not-a-real-token", "ask", {"question": "sky"})
        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["text"], "")

    def test_malformed_token_denies(self):
        for bad in ("   ", "a", "\x00\x01"):
            r = R.handle_request(self.conn, bad, "ask", {"question": "sky"})
            self.assertEqual(r["decision"], "deny", bad)

    def test_revoked_token_denies(self):
        d = R.add_device(self.conn, "laptop", "topsecret", ["ask"])
        ok = R.revoke_device(self.conn, d["device_id"])
        self.assertTrue(ok)
        r = R.handle_request(self.conn, d["token"], "ask", {"question": "sky"})
        self.assertEqual(r["decision"], "deny")
        self.assertEqual(r["text"], "")

    def test_auth_failure_never_falls_back_to_local_serving(self):
        """No branch in handle_request may reach _call_tool without a device --
        proven here by never seeing the seeded node's content leak through on
        ANY of the three failure modes."""
        for token in (None, "", "unknown", "revoked-later"):
            r = R.handle_request(self.conn, token, "ask", {"question": "sky"})
            self.assertNotIn("sky is blue", r["text"])

    def test_valid_token_allows_and_returns_content(self):
        d = R.add_device(self.conn, "laptop", "topsecret", [])
        r = R.handle_request(self.conn, d["token"], "ask", {"question": "sky"})
        self.assertEqual(r["decision"], "allow")
        self.assertFalse(r["is_error"])

    def test_unknown_tool_denies(self):
        d = R.add_device(self.conn, "laptop", "topsecret", [])
        r = R.handle_request(self.conn, d["token"], "not_a_real_tool", {})
        self.assertEqual(r["decision"], "deny")


class TestAudit(Base):
    def test_every_call_is_audited(self):
        R.handle_request(self.conn, "unknown", "ask", {"question": "x"})
        d = R.add_device(self.conn, "laptop", "public", [])
        R.handle_request(self.conn, d["token"], "ask", {"question": "x"})
        rows = R.recent_audit(self.conn, limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["decision"], "allow")  # most recent first
        self.assertEqual(rows[1]["decision"], "deny")

    def test_audit_row_on_unauthenticated_call_has_no_device_id(self):
        R.handle_request(self.conn, "unknown", "ask", {"question": "x"})
        rows = R.recent_audit(self.conn, limit=1)
        self.assertIsNone(rows[0]["device_id"])


class TestDeviceCrud(Base):
    def test_add_list_revoke_roundtrip(self):
        d = R.add_device(self.conn, "phone", "internal", ["revoke"])
        devices = R.list_devices(self.conn)
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["device_id"], d["device_id"])
        self.assertEqual(devices[0]["capabilities"], ["revoke"])
        self.assertIsNone(devices[0]["revoked_at"])

        self.assertTrue(R.revoke_device(self.conn, d["device_id"]))
        self.assertFalse(R.revoke_device(self.conn, d["device_id"]))  # already revoked
        self.assertFalse(R.revoke_device(self.conn, "nonexistent"))

        devices = R.list_devices(self.conn)
        self.assertIsNotNone(devices[0]["revoked_at"])

    def test_list_devices_never_exposes_token(self):
        R.add_device(self.conn, "phone", "internal", [])
        devices = R.list_devices(self.conn)
        self.assertNotIn("token", devices[0])
        self.assertNotIn("token_hash", devices[0])


class TestFeaturesShowsBothBlocks(Base):
    """Sec3.6: `memsom features --json` in client mode shows BOTH this
    machine's own features AND the remote server's -- a real HTTP round trip
    over 127.0.0.1, not a mock."""

    def test_client_mode_fetches_server_features_block(self):
        from memsom.interface import features as memsom_features
        from memsom.interface import serve as memsom_serve
        from memsom.storage import settings as memsom_settings

        d = R.add_device(self.conn, "client-under-test", "public", [])

        server = memsom_serve.build_server("127.0.0.1", 0)
        ip, port = server.server_address[:2]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            time.sleep(0.2)
            data_dir = Path(os.environ["MEMDAG_DB"]).parent
            memsom_settings.save_settings(data_dir, {
                "mode": "client",
                "remote_server_url": f"http://{ip}:{port}",
                "remote_device_token": d["token"],
            })
            old_home = os.environ.get("MEMDAG_HOME")
            os.environ["MEMDAG_HOME"] = str(data_dir)
            try:
                statuses = memsom_features.all_statuses(self.conn)
                self.assertEqual(statuses["remote.client"]["state"], "active")
                block = memsom_features._remote_server_features_block(statuses)
                self.assertIsNotNone(block)
                self.assertNotIn("error", block)
                self.assertIn("retrieval.dense", block)
            finally:
                if old_home is None:
                    os.environ.pop("MEMDAG_HOME", None)
                else:
                    os.environ["MEMDAG_HOME"] = old_home
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
