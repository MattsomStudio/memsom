#!/usr/bin/env python3
"""Tests for memsom.interface.remote -- Sec3.5 point 4a, the capability table.

"Each tool name is classified read or mutate... a device without the
capability gets 403 + an audit row before anything else runs." A read-only
device (empty capabilities set) must never be able to ingest/revoke/redact,
regardless of its clearance ceiling.
"""

import os
import tempfile
import unittest
from pathlib import Path

import memsom
from memsom.interface import cli as memsom_cli
from memsom.interface import remote as R


class TestToolClassification(unittest.TestCase):
    def test_mutate_by_definition_tools(self):
        for tool in ("revoke", "redact", "ingest_text", "obsidian_export", "export"):
            self.assertEqual(R.tool_class(tool), "mutate", tool)

    def test_read_tools(self):
        for tool in ("ask", "explain", "blame", "retrieve", "profile", "check_action"):
            self.assertEqual(R.tool_class(tool), "read", tool)

    def test_unknown_tool(self):
        self.assertEqual(R.tool_class("not_a_real_tool"), "unknown")

    def test_every_mcp_tool_is_classified(self):
        from memsom.interface import mcp as memsom_mcp
        for tool in memsom_mcp.TOOL_NAMES:
            self.assertIn(R.tool_class(tool), ("mutate", "read"), tool)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_cli.migrate_all(self.conn)
        R.reset_device_sessions()
        self.nid = memsom.insert_node(self.conn, "some source content", "user")

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()


class TestReadOnlyDevice(Base):
    """Read-only = topsecret clearance but ZERO capabilities -- proves the
    capability gate is independent of the clearance gate."""

    def _device(self):
        return R.add_device(self.conn, "readonly-laptop", "topsecret", [])

    def test_cannot_ingest_text(self):
        d = self._device()
        r = R.handle_request(self.conn, d["token"], "ingest_text",
                             {"text": "malicious", "channel": "user"})
        self.assertEqual(r["decision"], "deny")

    def test_cannot_revoke(self):
        d = self._device()
        r = R.handle_request(self.conn, d["token"], "revoke", {"id": self.nid})
        self.assertEqual(r["decision"], "deny")

    def test_cannot_redact(self):
        d = self._device()
        r = R.handle_request(self.conn, d["token"], "redact",
                             {"id": self.nid, "reason": "test"})
        self.assertEqual(r["decision"], "deny")

    def test_read_tools_still_work(self):
        d = self._device()
        r = R.handle_request(self.conn, d["token"], "ask", {"question": "source"})
        self.assertEqual(r["decision"], "allow")

    def test_denied_mutate_does_not_mutate_the_store(self):
        d = self._device()
        R.handle_request(self.conn, d["token"], "revoke", {"id": self.nid, "apply": True})
        node = memsom.get_node(self.conn, self.nid)
        self.assertFalse(node["tombstoned"])


class TestGrantedCapability(Base):
    def test_device_with_ingest_capability_can_ingest(self):
        d = R.add_device(self.conn, "ingest-bot", "public", ["ingest_text"])
        r = R.handle_request(self.conn, d["token"], "ingest_text",
                             {"text": "new fact", "channel": "user"})
        self.assertEqual(r["decision"], "allow")
        self.assertFalse(r["is_error"])

    def test_capability_is_per_tool_not_all_or_nothing(self):
        # granted ingest_text but NOT revoke
        d = R.add_device(self.conn, "ingest-bot", "public", ["ingest_text"])
        r_ingest = R.handle_request(self.conn, d["token"], "ingest_text",
                                    {"text": "new fact", "channel": "user"})
        r_revoke = R.handle_request(self.conn, d["token"], "revoke", {"id": self.nid})
        self.assertEqual(r_ingest["decision"], "allow")
        self.assertEqual(r_revoke["decision"], "deny")


class TestActionGateShadowMode(Base):
    """Sec3.5 point 4b: the action gate ships in shadow by default -- a
    granted capability call is always LOGGED to capgate's capability_log,
    but never blocked by (b) alone while the knob is 'shadow'."""

    def test_action_gate_decision_is_logged_but_shadow_never_blocks(self):
        from memsom.integrity import capgate as memsom_capgate
        d = R.add_device(self.conn, "ingest-bot", "public", ["ingest_text"])
        r = R.handle_request(self.conn, d["token"], "ingest_text",
                             {"text": "new fact", "channel": "user"})
        self.assertEqual(r["decision"], "allow")
        self.assertIsNotNone(r["action_gate"])
        rows = memsom_capgate.recent_capability_log(self.conn, limit=1)
        self.assertEqual(rows[0]["tool"], "ingest_text")


if __name__ == "__main__":
    unittest.main()
