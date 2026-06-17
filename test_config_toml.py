#!/usr/bin/env python3
"""Phase 5 — Codex TOML config wiring (append-only, literal-string paths).

Run:  python -m unittest -v test_config_toml
"""

import tempfile
import tomllib
import unittest
from pathlib import Path

import memdag_config

EXE = "/home/u/.memdag/venv/bin/memdag-mcp"
DB = "/home/u/.memdag/memdag.db"


class TestWireToml(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def _parsed(self):
        return tomllib.loads(self.path.read_text(encoding="utf-8"))

    def test_appends_block_when_absent(self):
        self.path.write_text('model = "gpt-5"\n', encoding="utf-8")
        res = memdag_config.wire_toml(self.path, EXE, DB)
        self.assertEqual(res["action"], "merged")
        p = self._parsed()
        self.assertEqual(p["mcp_servers"]["memdag"]["command"], EXE)
        # env lives in the SUB-TABLE, not an env_vars array
        self.assertEqual(p["mcp_servers"]["memdag"]["env"]["MEMDAG_DB"], DB)

    def test_preserves_existing_toml(self):
        self.path.write_text('model = "gpt-5"\n\n[mcp_servers.other]\ncommand = "x"\n',
                             encoding="utf-8")
        memdag_config.wire_toml(self.path, EXE, DB)
        p = self._parsed()
        self.assertEqual(p["model"], "gpt-5")
        self.assertIn("other", p["mcp_servers"])
        self.assertIn("memdag", p["mcp_servers"])

    def test_idempotent_toml(self):
        self.path.write_text('model = "gpt-5"\n', encoding="utf-8")
        memdag_config.wire_toml(self.path, EXE, DB)
        first = self.path.read_bytes()
        res2 = memdag_config.wire_toml(self.path, EXE, DB)
        self.assertEqual(res2["action"], "unchanged")
        self.assertEqual(self.path.read_bytes(), first)

    def test_writes_backup_toml(self):
        original = 'model = "gpt-5"\n'
        self.path.write_text(original, encoding="utf-8")
        memdag_config.wire_toml(self.path, EXE, DB)
        bak = self.path.with_name(self.path.name + ".bak")
        self.assertTrue(bak.exists())
        self.assertEqual(bak.read_text(), original)

    def test_malformed_toml_prints(self):
        broken = "this = = not valid toml\n"
        self.path.write_text(broken, encoding="utf-8")
        res = memdag_config.wire_toml(self.path, EXE, DB)
        self.assertEqual(res["action"], "malformed")
        self.assertEqual(self.path.read_text(), broken)

    def test_db_path_with_special_chars(self):
        # The load-bearing one: a real Windows backslash path must round-trip via
        # the single-quoted literal string (basic strings would choke on \U etc.),
        # and a macOS space-in-path must too.
        cases = [
            (r"C:\Users\foo\.memdag\venv\Scripts\memdag-mcp.exe",
             r"C:\Users\foo\.memdag\memdag.db"),
            ("/Users/foo/Library/Application Support/x/bin/memdag-mcp",
             "/Users/foo/.memdag/memdag.db"),
        ]
        for exe, db in cases:
            with self.subTest(exe=exe):
                p = Path(self.tmp.name) / f"cfg_{abs(hash(exe))}.toml"
                res = memdag_config.wire_toml(p, exe, db)
                self.assertEqual(res["action"], "created")
                parsed = tomllib.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(parsed["mcp_servers"]["memdag"]["command"], exe)
                self.assertEqual(parsed["mcp_servers"]["memdag"]["env"]["MEMDAG_DB"], db)


class TestWireTomlInlineTable(unittest.TestCase):
    """CONFIG-MERGE-INLINE-1: appending a [mcp_servers.memdag] header to a file
    that defines mcp_servers as an INLINE table produces invalid TOML — wire_toml
    must refuse and leave the file intact, not corrupt it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_inline_table_not_corrupted(self):
        original = 'mcp_servers = { other = { command = "x" } }\n'
        self.path.write_text(original, encoding="utf-8")
        res = memdag_config.wire_toml(self.path, EXE, DB)
        self.assertEqual(res["action"], "exists_differs",
                         "must refuse rather than corrupt an inline-table config")
        self.assertEqual(self.path.read_text(encoding="utf-8"), original,
                         "file must be left untouched")
        tomllib.loads(self.path.read_text(encoding="utf-8"))  # still parses


if __name__ == "__main__":
    unittest.main()
