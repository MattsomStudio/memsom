"""Tests for memsom_claude — the CLAUDE.md managed-block manager.

Run:  python -m unittest discover -s . -p test_memsom_claude.py
"""

import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom_claude as mc

HERE = Path(__file__).resolve().parent.parent


class TestPureUpsert(unittest.TestCase):
    def test_append_when_no_block(self):
        new, action = mc.upsert("# My CLAUDE\n\nmy rules\n", mc.render_block())
        self.assertEqual(action, "append")
        self.assertIn("my rules", new)                 # user content kept
        self.assertIn(mc.START, new)
        self.assertIn(mc.END, new)

    def test_replace_when_block_present(self):
        text = f"top\n\n{mc.START}\nstale body\n{mc.END}\n\nbottom\n"
        new, action = mc.upsert(text, mc.render_block())
        self.assertEqual(action, "replace")
        self.assertIn("top", new)
        self.assertIn("bottom", new)                   # content around block intact
        self.assertNotIn("stale body", new)
        self.assertIn("Memory protocol", new)

    def test_migrates_legacy_memdag_block_in_place(self):
        # A tester wired before the 2026-07-01 rename carries the legacy
        # memdag:managed markers. upsert() must REPLACE that block (migrating it
        # to the new memsom markers), not append a duplicate.
        text = (f"top\n\n{mc.LEGACY_START}\nold memdag body\n{mc.LEGACY_END}\n\n"
                "bottom\n")
        new, action = mc.upsert(text, mc.render_block())
        self.assertEqual(action, "replace")
        self.assertNotIn("old memdag body", new)
        self.assertNotIn(mc.LEGACY_START, new)          # legacy markers gone
        self.assertIn(mc.START, new)                     # new markers present
        self.assertEqual(new.count(mc.START), 1)         # exactly one block, no dupe
        self.assertIn("top", new)
        self.assertIn("bottom", new)

    def test_noop_when_already_current(self):
        text = f"top\n\n{mc.render_block()}\n\nbottom\n"
        new, action = mc.upsert(text, mc.render_block())
        self.assertEqual(action, "noop")
        self.assertEqual(new, text)


class TestSyncIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "CLAUDE.md"

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_when_absent(self):
        res = mc.sync(path=self.path)
        self.assertEqual(res["action"], "seed")
        self.assertTrue(self.path.exists())
        self.assertIn(mc.START, self.path.read_text(encoding="utf-8"))

    def test_append_backs_up_and_preserves_user_content(self):
        self.path.write_text("# Mine\n\nkeep this line\n", encoding="utf-8")
        res = mc.sync(path=self.path)
        self.assertEqual(res["action"], "append")
        body = self.path.read_text(encoding="utf-8")
        self.assertIn("keep this line", body)          # user content preserved
        self.assertIn(mc.START, body)
        bak = self.path.with_suffix(self.path.suffix + ".bak")
        self.assertTrue(bak.exists())                  # backup made before touching
        self.assertEqual(bak.read_text(encoding="utf-8"), "# Mine\n\nkeep this line\n")

    def test_idempotent_second_run_is_noop(self):
        mc.sync(path=self.path)                         # seed
        before = self.path.read_text(encoding="utf-8")
        res = mc.sync(path=self.path)                   # again
        self.assertEqual(res["action"], "noop")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_refresh_updates_only_the_block(self):
        self.path.write_text(
            f"# Mine\n\nuser top\n\n{mc.START}\nOLD\n{mc.END}\n\nuser bottom\n",
            encoding="utf-8")
        res = mc.sync(path=self.path)
        self.assertEqual(res["action"], "replace")
        body = self.path.read_text(encoding="utf-8")
        self.assertIn("user top", body)
        self.assertIn("user bottom", body)
        self.assertNotIn("OLD", body)
        self.assertIn("Memory protocol", body)

    def test_dry_run_writes_nothing(self):
        res = mc.sync(path=self.path, dry_run=True)
        self.assertEqual(res["action"], "seed")
        self.assertTrue(res["changed"])
        self.assertFalse(self.path.exists())


class TestShippedTemplateInSync(unittest.TestCase):
    def test_template_file_matches_code(self):
        # The shipped browsable template must never drift from the canonical source.
        shipped = (HERE / "claude" / "CLAUDE.md.template").read_text(encoding="utf-8")
        self.assertEqual(shipped, mc.render_template())


if __name__ == "__main__":
    unittest.main()
