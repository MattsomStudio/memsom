"""Tests for memsom_bridge_render — the shippable MEMORY.md regenerator.

Run:  python -m unittest discover -s . -p test_memsom_bridge_render.py
"""

import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.distill import digest as digest
from memsom.bridge import bridge_render as br


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
        # the bridge now indexes every imported node; keep the tests off the
        # network (an Ollama embed is ~1s/node) -> BM25-only for the suite
        self._embed_prev = os.environ.get("MEMDAG_EMBED_BACKEND")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        # NEVER let claude-sync touch the real ~/.claude/CLAUDE.md during tests.
        # Restore (not pop) in tearDown: tests/_isolation.py pins this for the
        # whole process and a pop would drop that fence for every later test.
        self._claude_md_prev = os.environ.get("CLAUDE_MD_PATH")
        os.environ["CLAUDE_MD_PATH"] = str(self.root / "CLAUDE.md")
        self.mem = self.root / "memory"
        self.mem.mkdir()
        for n, t in FILES.items():
            (self.mem / n).write_text(t, encoding="utf-8")
        self.memory_md = self.mem / "MEMORY.md"
        self.memory_md.write_text(INDEX, encoding="utf-8")
        self.conn = memsom.get_connection()

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        if self._embed_prev is None:
            os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        else:
            os.environ["MEMDAG_EMBED_BACKEND"] = self._embed_prev
        os.environ.pop("MEMDAG_DIGEST_TITLE", None)
        if self._claude_md_prev is None:
            os.environ.pop("CLAUDE_MD_PATH", None)
        else:
            os.environ["CLAUDE_MD_PATH"] = self._claude_md_prev
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


class TestProjectsIndex(Base):
    def test_writes_projects_index_and_pointer(self):
        import json
        (self.mem / "projects").mkdir()
        (self.mem / "projects" / "project_gadget.md").write_text(
            "---\nname: Gadget\ndescription: parked for now\ntype: project\n"
            "status: parked\n---\nx\n", encoding="utf-8")
        result = br.bridge_render(self.conn, self.mem)
        self.assertTrue(result["ok"], result)
        out = self.memory_md.read_text(encoding="utf-8")
        # project_ memories leave MEMORY.md; the pointer line replaces them
        self.assertNotIn("project_widget.md", out)
        self.assertNotIn("project_gadget.md", out)
        self.assertIn("## Personal projects\n" + digest.PROJECTS_POINTER_LINE, out)
        idx = self.mem / "projects" / "INDEX.md"
        self.assertTrue(idx.exists())
        self.assertEqual(result["info"]["projects_index"], str(idx))
        text = idx.read_text(encoding="utf-8")
        self.assertIn("## Standalone\n- [Widget](../project_widget.md) — status\n"
                      "- [Gadget](project_gadget.md) — parked for now [Parked]", text)
        self.assertFalse((self.mem / "projects" / "INDEX.md.tmp").exists())
        # shed manifest carries the line accounting + the projects reason
        shed = json.loads((self.mem / ".weights" / "shed.json").read_text(encoding="utf-8"))
        self.assertEqual(shed["max_lines"], digest.MAX_LINES)
        self.assertEqual(shed["lines"], out.count("\n"))
        self.assertEqual(shed["by_reason"].get("projects"), 2)

    def test_second_render_is_stable(self):
        # the rendered pointer is re-imported as a literal; the next render must
        # be byte-identical (no duplicate pointer, no churn)
        br.bridge_render(self.conn, self.mem)
        first = self.memory_md.read_text(encoding="utf-8")
        br.bridge_render(self.conn, self.mem)
        self.assertEqual(self.memory_md.read_text(encoding="utf-8"), first)
        self.assertEqual(first.count(digest.PROJECTS_POINTER_LINE), 1)


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


class TestFirstRunScaffold(Base):
    def test_render_scaffolds_canonical_and_projects_index(self):
        import json
        self.assertFalse((self.mem / ".weights" / "canonical.json").exists())
        result = br.bridge_render(self.conn, self.mem)
        self.assertTrue(result["ok"], result)
        params = json.loads((self.mem / ".weights" / "canonical.json")
                            .read_text(encoding="utf-8"))["params"]
        self.assertEqual(params["memory_max_lines"], digest.MAX_LINES)
        self.assertEqual(params["memory_budget"], digest.BUDGET)
        self.assertTrue((self.mem / "projects" / "INDEX.md").exists())

    def test_existing_canonical_is_never_overwritten(self):
        import json
        w = self.mem / ".weights"
        w.mkdir()
        (w / "canonical.json").write_text(
            json.dumps({"version": 1, "params": {"memory_budget": 4096}}), encoding="utf-8")
        br.bridge_render(self.conn, self.mem)
        data = json.loads((w / "canonical.json").read_text(encoding="utf-8"))
        self.assertEqual(data["params"], {"memory_budget": 4096})


# R1 regression: importing bridge_render must, as a side effect, register the
# "resolve_fact_refs" kernel.events subscriber that digest.render_digest relies
# on -- so a direct library caller of digest.write_live (never going through
# cli.py, which imports memsom.bridge.facts itself for CLI-command reasons)
# still gets [[fact_*]] refs resolved. Runs in a SUBPROCESS: the in-process
# `import ... as br` earlier in this module would already have pulled in
# memsom.bridge.facts via br's own import, making an in-process check pass
# regardless of whether bridge_render.py itself imports facts.
_SUBPROCESS_SCRIPT = """
import os, sys
import memsom
from memsom.bridge import bridge_import as bi
from memsom.bridge import bridge_render as br  # noqa: F401 -- side effect under test
from memsom.distill import digest

assert "memsom.interface.cli" not in sys.modules

mem = os.environ["_TEST_MEM_DIR"]
conn = memsom.get_connection()
bi.migrate(conn)
bi.import_memory_dir(conn, mem, dry_run=False)
digest.write_live(conn, mem)
sys.stdout.write(open(os.path.join(mem, "MEMORY.md"), encoding="utf-8").read())
"""


class TestFactResolutionImportSideEffect(unittest.TestCase):
    """F-4: this test spawns a subprocess and asserts on its full stdout
    (the rendered MEMORY.md, title line included), so it must pin
    MEMDAG_DIGEST_TITLE -- otherwise it renders under whatever the *shell
    running the test suite* happens to export, and the test's own pass/fail
    is not reproducible across environments."""

    _PINNED_TITLE = "# Memory"

    def setUp(self):
        self._saved_digest_title = os.environ.get("MEMDAG_DIGEST_TITLE")
        os.environ["MEMDAG_DIGEST_TITLE"] = self._PINNED_TITLE

    def tearDown(self):
        # save/restore, never a bare pop: put back exactly what was there
        # before this test ran (absent -> stays absent, set -> restored).
        if self._saved_digest_title is None:
            os.environ.pop("MEMDAG_DIGEST_TITLE", None)
        else:
            os.environ["MEMDAG_DIGEST_TITLE"] = self._saved_digest_title

    def test_write_live_resolves_facts_without_cli_import(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            mem = root / "memory"
            mem.mkdir()
            (mem / "fact_x.md").write_text(
                "---\nname: fact-x\ndescription: box RAM\ntype: fact\n"
                "value: 42\nunit: GB\nlast-verified: 2026-09-01\n"
                "section: Facts\n---\n\nmeasured\n", encoding="utf-8")
            (mem / "user_a.md").write_text(
                "---\nname: User A\ndescription: has [[fact_x]] of RAM\ntype: user\n"
                "section: About the User\n---\nbody\n", encoding="utf-8")

            env = dict(os.environ)
            env["MEMDAG_HOME"] = str(root)
            env["MEMDAG_DB"] = str(root / "t.db")
            env["MEMDAG_EMBED_BACKEND"] = "bm25"
            env["CLAUDE_MD_PATH"] = str(root / "CLAUDE.md")
            env["_TEST_MEM_DIR"] = str(mem)
            env["MEMDAG_DIGEST_TITLE"] = self._PINNED_TITLE  # belt-and-suspenders: explicit, not just inherited

            result = subprocess.run(
                [sys.executable, "-c", _SUBPROCESS_SCRIPT],
                env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            out = result.stdout
            self.assertNotIn("[[fact_", out)
            self.assertIn("42 GB", out)
            # the rendered title must be the pinned one, not whatever the
            # ambient shell exports (e.g. MEMDAG_DIGEST_TITLE=ZZZ would show up here)
            self.assertIn(self._PINNED_TITLE, out)


if __name__ == "__main__":
    unittest.main()
