#!/usr/bin/env python3
"""Phase 5 — JSON config wiring (Claude Desktop / Claude Code fallback).

Run:  python -m unittest -v test_config_json
"""

import json
import tempfile
import unittest
from pathlib import Path

from memsom.storage import config as memsom_config

EXE = "/home/u/.memdag/venv/bin/memsom-mcp"
DB = "/home/u/.memdag/memdag.db"


class TestWireJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "claude_desktop_config.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_when_missing(self):
        res = memsom_config.wire_json(self.path, EXE, DB)
        self.assertEqual(res["action"], "created")
        data = json.loads(self.path.read_text())
        entry = data["mcpServers"]["memsom"]
        self.assertEqual(entry["command"], EXE)              # absolute path, not bare name
        self.assertEqual(entry["env"]["MEMDAG_DB"], DB)

    def test_preserves_existing_servers(self):
        self.path.write_text(json.dumps(
            {"mcpServers": {"other": {"command": "x", "args": []}}}), encoding="utf-8")
        memsom_config.wire_json(self.path, EXE, DB)
        data = json.loads(self.path.read_text())
        self.assertIn("other", data["mcpServers"])            # untouched
        self.assertIn("memsom", data["mcpServers"])           # added

    def test_idempotent(self):
        memsom_config.wire_json(self.path, EXE, DB)
        first = self.path.read_bytes()
        res2 = memsom_config.wire_json(self.path, EXE, DB)
        self.assertEqual(res2["action"], "unchanged")
        self.assertEqual(self.path.read_bytes(), first)       # byte-identical

    def test_writes_backup(self):
        original = json.dumps({"mcpServers": {"other": {"command": "x"}}})
        self.path.write_text(original, encoding="utf-8")
        memsom_config.wire_json(self.path, EXE, DB)
        bak = self.path.with_name(self.path.name + ".bak")
        self.assertTrue(bak.exists())
        self.assertEqual(bak.read_text(), original)           # pre-write contents

    def test_malformed_prints_not_writes(self):
        broken = "{ this is not valid json"
        self.path.write_text(broken, encoding="utf-8")
        res = memsom_config.wire_json(self.path, EXE, DB)
        self.assertEqual(res["action"], "malformed")
        self.assertIn("snippet", res)
        self.assertEqual(self.path.read_text(), broken)       # untouched

    def test_print_only_never_writes(self):
        self.path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
        before = self.path.read_bytes()
        res = memsom_config.wire_json(self.path, EXE, DB, print_only=True)
        self.assertEqual(res["action"], "print")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
