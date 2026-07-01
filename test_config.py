#!/usr/bin/env python3
"""Phase 5 — config path resolution + Claude Code CLI-vs-fallback routing.

Run:  python -m unittest -v test_config
"""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import memsom_config


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.home = Path("/home/u")

    def test_codex_path(self):
        self.assertEqual(memsom_config.client_config_path("codex", home=self.home),
                         self.home / ".codex" / "config.toml")

    def test_claude_code_path(self):
        self.assertEqual(memsom_config.client_config_path("claude-code", home=self.home),
                         self.home / ".claude.json")

    def test_desktop_paths_per_os(self):
        mac = memsom_config.client_config_path("claude-desktop", os_name="Darwin", home=self.home)
        win = memsom_config.client_config_path("claude-desktop", os_name="Windows", home=self.home)
        lin = memsom_config.client_config_path("claude-desktop", os_name="Linux", home=self.home)
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

        with mock.patch.object(memsom_config.shutil, "which", return_value="/usr/bin/claude"), \
             mock.patch.object(memsom_config.subprocess, "run", side_effect=fake_run):
            res = memsom_config.wire_claude_code("/abs/memsom-mcp", "/abs/db", home=self.home)
        self.assertEqual(res["action"], "claude-cli")
        self.assertEqual(calls[0][:4], ["claude", "mcp", "add", "memsom"])
        self.assertIn("--scope", calls[0])

    def test_falls_back_to_json_when_cli_absent(self):
        with mock.patch.object(memsom_config.shutil, "which", return_value=None):
            res = memsom_config.wire_claude_code("/abs/memsom-mcp", "/abs/db", home=self.home)
        self.assertEqual(res["action"], "created")
        cfg = self.home / ".claude.json"
        data = json.loads(cfg.read_text())
        self.assertEqual(data["mcpServers"]["memsom"]["command"], "/abs/memsom-mcp")


def _args(**kw):
    """Build a namespace matching what argparse hands cmd_wire_config."""
    base = {"client": "codex", "exe": None, "db": None, "print_only": False}
    base.update(kw)
    return types.SimpleNamespace(**base)


# A path whose apostrophe closes the single-quoted TOML literal early, so the
# generated block fails to round-trip and wire_toml returns action="print".
APOS_EXE = "/home/o'brien/memsom-mcp"


class TestWireTomlRefusesApostrophe(unittest.TestCase):
    """The mechanism CFG-PRINT-SOFTFAIL-1 rides on: a real write that refuses."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_apostrophe_in_exe_refuses_to_write(self):
        res = memsom_config.wire_toml(self.path, APOS_EXE, "/tmp/x.db", print_only=False)
        self.assertEqual(res["action"], "print")
        self.assertFalse(self.path.exists())  # genuinely wrote nothing

    def test_apostrophe_in_db_refuses_to_write(self):
        res = memsom_config.wire_toml(self.path, "/abs/memsom-mcp", "/tmp/o'brien.db",
                                      print_only=False)
        self.assertEqual(res["action"], "print")
        self.assertFalse(self.path.exists())


class TestCmdWireConfigExitCode(unittest.TestCase):
    """cmd_wire_config exit-code contract — the CFG-PRINT-SOFTFAIL-1 regression.

    Previously only {malformed, exists_differs} were counted as failures, so a
    real-run action='print' (write-refusal) returned 0 and the client was left
    unconfigured while bootstrap reported success.
    """

    def test_real_run_print_refusal_returns_1(self):
        # Real run (print_only=False) where wiring soft-fails with action='print'.
        with mock.patch.object(memsom_config, "wire",
                               return_value={"action": "print", "path": "/p", "snippet": "x"}):
            rc = memsom_config.cmd_wire_config(_args(exe=APOS_EXE, db="/tmp/x.db"))
        self.assertEqual(rc, 1)

    def test_print_only_run_returns_0(self):
        # print-only mode: action='print' is the EXPECTED success, not a failure.
        with mock.patch.object(memsom_config, "wire",
                               return_value={"action": "print", "path": "/p", "snippet": "x"}):
            rc = memsom_config.cmd_wire_config(_args(print_only=True, exe=APOS_EXE, db="/tmp/x.db"))
        self.assertEqual(rc, 0)

    def test_real_run_print_via_real_wire_toml_returns_1(self):
        # End-to-end through the real wire/wire_toml path (no action mock): an
        # apostrophe path makes wire_toml refuse -> cmd_wire_config must return 1.
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            rc = memsom_config.cmd_wire_config(
                _args(client="codex", exe=APOS_EXE, db=str(home / "x.db")))
        self.assertEqual(rc, 1)

    def test_per_action_exit_matrix_real_run(self):
        # Lock the SUCCESS-WHITELIST: a real run maps confirmed writes -> 0 and
        # every refusal/unknown action -> 1. Guards against a future denylist
        # regression and against a new action value silently mapping to success.
        success = ("created", "merged", "unchanged", "claude-cli")
        failure = ("print", "malformed", "exists_differs", "some-future-action")
        for action in success:
            with mock.patch.object(memsom_config, "wire",
                                   return_value={"action": action, "path": "/p"}):
                rc = memsom_config.cmd_wire_config(_args(exe="/abs/x", db="/abs/d"))
            self.assertEqual(rc, 0, f"{action!r} should be a success (rc 0)")
        for action in failure:
            with mock.patch.object(memsom_config, "wire",
                                   return_value={"action": action, "path": "/p", "snippet": "x"}):
                rc = memsom_config.cmd_wire_config(_args(exe="/abs/x", db="/abs/d"))
            self.assertEqual(rc, 1, f"{action!r} should be a failure (rc 1)")


class TestEntryPointExitCodes(unittest.TestCase):
    """CFG-MAIN-EXITDROP-1 + dispatcher propagation: the exit code must survive
    BOTH the standalone __main__ and the memsom_cli dispatcher entry points."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "x.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        # client=codex so wiring routes to wire_toml (an apostrophe exe refuses);
        # avoids the claude-code CLI path / real $HOME config writes.
        return subprocess.run(argv, capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parent))

    def test_standalone_main_real_run_exits_1(self):
        # CFG-MAIN-EXITDROP-1: direct `python memsom_config.py` must not drop the code.
        r = self._run([sys.executable, "memsom_config.py", "--client", "codex",
                       "--exe", APOS_EXE, "--db", self.db])
        self.assertEqual(r.returncode, 1, r.stderr)

    def test_dispatcher_real_run_exits_1(self):
        r = self._run([sys.executable, "-m", "memsom_cli", "wire-config", "--client", "codex",
                       "--exe", APOS_EXE, "--db", self.db])
        self.assertEqual(r.returncode, 1, r.stderr)

    def test_dispatcher_print_only_exits_0(self):
        r = self._run([sys.executable, "-m", "memsom_cli", "wire-config", "--client", "codex",
                       "--exe", APOS_EXE, "--db", self.db, "--print-only"])
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
