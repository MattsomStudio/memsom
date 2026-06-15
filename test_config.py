#!/usr/bin/env python3
"""Phase 5 — config path resolution + Claude Code CLI-vs-fallback routing.

Run:  python -m unittest -v test_config
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import memdag_config


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.home = Path("/home/u")

    def test_codex_path(self):
        self.assertEqual(memdag_config.client_config_path("codex", home=self.home),
                         self.home / ".codex" / "config.toml")

    def test_claude_code_path(self):
        self.assertEqual(memdag_config.client_config_path("claude-code", home=self.home),
                         self.home / ".claude.json")

    def test_desktop_paths_per_os(self):
        mac = memdag_config.client_config_path("claude-desktop", os_name="Darwin", home=self.home)
        win = memdag_config.client_config_path("claude-desktop", os_name="Windows", home=self.home)
        lin = memdag_config.client_config_path("claude-desktop", os_name="Linux", home=self.home)
        self.assertEqual(mac, self.home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json")
        self.assertEqual(win, self.home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json")
        self.assertEqual(lin, self.home / ".config" / "Claude" / "claude_desktop_config.json")


class TestClaudeCodeRouting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prefers_cli_when_present(self):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(memdag_config.shutil, "which", return_value="/usr/bin/claude"), \
             mock.patch.object(memdag_config.subprocess, "run", side_effect=fake_run):
            res = memdag_config.wire_claude_code("/abs/memdag-mcp", "/abs/db", home=self.home)
        self.assertEqual(res["action"], "claude-cli")
        self.assertEqual(calls[0][:4], ["claude", "mcp", "add", "memdag"])
        self.assertIn("--scope", calls[0])

    def test_falls_back_to_json_when_cli_absent(self):
        with mock.patch.object(memdag_config.shutil, "which", return_value=None):
            res = memdag_config.wire_claude_code("/abs/memdag-mcp", "/abs/db", home=self.home)
        self.assertEqual(res["action"], "created")
        cfg = self.home / ".claude.json"
        data = json.loads(cfg.read_text())
        self.assertEqual(data["mcpServers"]["memdag"]["command"], "/abs/memdag-mcp")


if __name__ == "__main__":
    unittest.main()
