#!/usr/bin/env python3
"""Tests for memdag_broker — Gate #3 MCP broker (config, pins, gate forward).

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_broker.py -t <repo> -v
"""

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_broker as B
import memdag_policy
import memdag_session


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_BROKER_CONFIG", None)
        self.tmp.cleanup()

    def write(self, name, data):
        p = Path(self.tmp.name) / name
        p.write_text(json.dumps(data), encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

class TestConfig(Base):
    def test_load_good(self):
        p = self.write("broker.json", {
            "upstreams": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}},
            "web_fetch_tools": ["fetch.fetch"],
            "session_start": "user",
        })
        cfg = B.load_config(p)
        self.assertIn("fetch", cfg["upstreams"])
        self.assertEqual(cfg["web_fetch_tools"], ["fetch.fetch"])
        self.assertFalse(cfg["ingest_external"])

    def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            B.load_config(Path(self.tmp.name) / "nope.json")

    def test_malformed_raises(self):
        p = Path(self.tmp.name) / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            B.load_config(p)

    def test_upstream_name_with_dot_rejected(self):
        p = self.write("broker.json", {"upstreams": {"a.b": {"command": "x"}}})
        with self.assertRaises(ValueError):
            B.load_config(p)

    def test_upstream_without_command_rejected(self):
        p = self.write("broker.json", {"upstreams": {"fetch": {"args": []}}})
        with self.assertRaises(ValueError):
            B.load_config(p)


# ---------------------------------------------------------------------------
# rug-pull pins
# ---------------------------------------------------------------------------

class TestPins(Base):
    def test_first_sight_pins_then_change_blocks(self):
        tools = [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]
        # first sight: pinned, nothing blocked
        self.assertEqual(B.pin_check(self.conn, "up", tools), set())
        # unchanged: still nothing blocked
        self.assertEqual(B.pin_check(self.conn, "up", tools), set())
        # schema changes -> blocked (rug pull)
        changed = [{"name": "t1", "description": "NOW EVIL", "inputSchema": {"type": "object"}}]
        self.assertEqual(B.pin_check(self.conn, "up", changed), {"t1"})


# ---------------------------------------------------------------------------
# decide_and_forward — the gate, with an injected caller (no subprocess)
# ---------------------------------------------------------------------------

class FakeUp:
    def __init__(self, tools):
        self.tools = tools


class TestDecideAndForward(Base):
    def setUp(self):
        super().setUp()
        self.policy = memdag_policy._normalize({
            "default": "deny",
            "rules": [
                {"tool": "fetch.fetch", "required": "external", "taints": "external"},
                {"tool": "mail.send", "required": "user"},
            ],
        })
        self.cfg = {"web_fetch_tools": ["fetch.fetch"], "ingest_external": False}
        self.sid = memdag_session.begin_session(self.conn, "user")
        self.calls = []

    def caller(self, u, t, a):
        self.calls.append(f"{u}.{t}")
        return B._text_result("ok")

    def test_clean_session_allows_consequential(self):
        r = B.decide_and_forward(self.conn, self.cfg, self.policy, self.sid,
                                 "mail.send", {}, self.caller)
        self.assertFalse(r["isError"])
        self.assertEqual(self.calls, ["mail.send"])

    def test_fetch_taints_then_send_denied_and_not_forwarded(self):
        B.decide_and_forward(self.conn, self.cfg, self.policy, self.sid,
                             "fetch.fetch", {}, self.caller)
        self.assertEqual(memdag_session.current_floor(self.conn, self.sid),
                         memdag.RANK["external"])
        before = list(self.calls)
        r = B.decide_and_forward(self.conn, self.cfg, self.policy, self.sid,
                                 "mail.send", {}, self.caller)
        self.assertTrue(r["isError"])
        self.assertIn("DENIED", r["content"][0]["text"])
        self.assertEqual(self.calls, before)  # denied call NOT forwarded

    def test_read_tool_never_denied_even_when_tainted(self):
        memdag_session.lower_floor(self.conn, self.sid, "external", "x", "test")
        r = B.decide_and_forward(self.conn, self.cfg, self.policy, self.sid,
                                 "fetch.fetch", {}, self.caller)
        self.assertFalse(r["isError"])

    def test_aggregated_tools_namespaced(self):
        ups = {"fetch": FakeUp([{"name": "fetch", "description": "d", "inputSchema": {}}])}
        names = {t["name"] for t in B.aggregated_tools(self.conn, ups)}
        self.assertIn("fetch.fetch", names)
        self.assertTrue(any(n.startswith("memdag.") for n in names))


if __name__ == "__main__":
    unittest.main()
