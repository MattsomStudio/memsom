"""Tests for memdag_bridge_render — the shippable MEMORY.md regenerator.

Run:  python -m unittest discover -s . -p test_memdag_bridge_render.py
"""

import os
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_digest as digest
import memdag_bridge_render as br


# Generic fixtures — no author-identifying content (the scrub gate scans this file).
FILES = {
    "user_editor.md": "---\nname: Editor\ndescription: prefers tabs\ntype: user\n---\nbody\n",
    "feedback_tests.md": "---\nname: Run tests\ndescription: always run tests\ntype: feedback\n---\nr\n",
    "project_widget.md": "---\nname: Widget\ndescription: status\ntype: project\n---\ns\n",
}
INDEX = """# Memory

## About the User
- [Editor](user_editor.md) — prefers tabs

## Personal projects
- [Widget](project_widget.md) — status

## Feedback
- [Run tests](feedback_tests.md) — always run tests
"""


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["MEMDAG_DB"] = str(self.root / "t.db")
        # NEVER let claude-sync touch the real ~/.claude/CLAUDE.md during tests.
        os.environ["CLAUDE_MD_PATH"] = str(self.root / "CLAUDE.md")
        self.mem = self.root / "memory"
        self.mem.mkdir()
        for n, t in FILES.items():
            (self.mem / n).write_text(t, encoding="utf-8")
        self.memory_md = self.mem / "MEMORY.md"
        self.memory_md.write_text(INDEX, encoding="utf-8")
        self.conn = memdag.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        os.environ.pop("MEMDAG_DIGEST_TITLE", None)
        os.environ.pop("CLAUDE_MD_PATH", None)
        self.tmp.cleanup()


class TestRender(Base):
    def test_regenerates_memory_md(self):
        result = br.bridge_render(self.conn, self.mem)
        self.assertTrue(result["rendered"])
        self.assertTrue(result["ok"], result)
        out = self.memory_md.read_text(encoding="utf-8")
        self.assertTrue(out.startswith("# Memory"))
        self.assertIn("- [Editor](user_editor.md) — prefers tabs", out)
        self.assertIn("- [Run tests](feedback_tests.md) — always run tests", out)

    def test_title_overridable_via_env(self):
        os.environ["MEMDAG_DIGEST_TITLE"] = "# Memory - Test User"
        br.bridge_render(self.conn, self.mem)
        self.assertTrue(
            self.memory_md.read_text(encoding="utf-8").startswith("# Memory - Test User"))

    def test_verify_stale_disabled_when_threshold_nonpositive(self):
        # MEMDAG_VERIFY_STALE_DAYS <= 0 turns the pass off — render still succeeds.
        os.environ["MEMDAG_VERIFY_STALE_DAYS"] = "0"
        try:
            result = br.bridge_render(self.conn, self.mem)
            self.assertEqual(result["stale_marked"], 0)
            self.assertTrue(result["ok"])
        finally:
            os.environ.pop("MEMDAG_VERIFY_STALE_DAYS", None)


class TestNonAuthor(Base):
    def test_mirror_only_does_not_render(self):
        sentinel = "# Memory\n\n(original, untouched)\n"
        self.memory_md.write_text(sentinel, encoding="utf-8")
        result = br.bridge_render(self.conn, self.mem, render=False)
        self.assertFalse(result["rendered"])
        # mirror imported, but MEMORY.md left exactly as it was
        self.assertEqual(self.memory_md.read_text(encoding="utf-8"), sentinel)


class TestFailSafe(Base):
    def test_rejected_render_leaves_existing_file(self):
        sentinel = "# Memory\n\n(good existing brain)\n"
        self.memory_md.write_text(sentinel, encoding="utf-8")
        orig = digest.write_live
        digest.write_live = lambda *a, **k: (False, [{"detail": "simulated reject"}])
        try:
            result = br.bridge_render(self.conn, self.mem)
        finally:
            digest.write_live = orig
        self.assertFalse(result["ok"])
        self.assertEqual(self.memory_md.read_text(encoding="utf-8"), sentinel)

    def test_cli_never_raises_into_hook_chain(self):
        # The Stop-hook boundary must swallow any internal error and exit clean.
        orig = br.bridge_render
        br.bridge_render = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            # Should not raise despite the core blowing up.
            br._cmd_bridge_render_safe(Namespace(memory_dir=str(self.mem)))
        finally:
            br.bridge_render = orig


if __name__ == "__main__":
    unittest.main()
