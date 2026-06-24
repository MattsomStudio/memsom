"""Tests for memdag_digest — render MEMORY.md from memdag (Phase 3).

Run:  python -m unittest discover -s . -p test_memdag_digest.py
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_bridge_import as bi
import memdag_forget as forget
import memdag_digest as digest


FILES = {
    "user_adhd.md": "---\nname: ADHD\ndescription: has ADHD\ntype: user\n---\nbody\n",
    "feedback_debug.md": "---\nname: Debug loop\ndescription: use the loop\ntype: feedback\n---\nr\n",
    "project_kali.md": "---\nname: Kali VM\ndescription: status\ntype: project\n---\ns\n",
    "reference_vault.md": "---\nname: Vault\ndescription: where\ntype: reference\n---\np\n",
}
INDEX = """# Memory - Matthew

## About the User
- **Matthew** — goal: cybersecurity
- [ADHD](user_adhd.md) — has ADHD

## Current Setup & Learning
- [Kali VM](project_kali.md) — status

## References
- [Vault](reference_vault.md) — where

## Feedback
- [Debug loop](feedback_debug.md) — use the loop
"""


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["MEMDAG_DB"] = str(self.root / "t.db")
        self.mem = self.root / "memory"
        self.mem.mkdir()
        for n, t in FILES.items():
            (self.mem / n).write_text(t, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")
        self.conn = memdag.get_connection()
        bi.migrate(self.conn)
        forget.migrate(self.conn)
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def demote(self, stem):
        self.conn.execute(
            "UPDATE nodes SET forget_tier = 'cold' WHERE source_ref = ?",
            (f"memory:{stem}",))
        self.conn.commit()


class TestRender(Base):
    def test_has_title_and_sections(self):
        out = digest.render_digest(self.conn)
        self.assertTrue(out.startswith("# Memory - Matthew Somanlall"))
        self.assertIn("## About the User", out)
        self.assertIn("## Feedback", out)

    def test_file_links_and_literals_rendered(self):
        out = digest.render_digest(self.conn)
        self.assertIn("- [ADHD](user_adhd.md) — has ADHD", out)
        self.assertIn("- **Matthew** — goal: cybersecurity", out)  # literal verbatim

    def test_equivalent_to_source_index(self):
        # rendering the freshly-imported store reproduces the same per-section
        # file sets as the original MEMORY.md (the cutover GO criterion)
        out = digest.render_digest(self.conn)
        diffs = digest.compare_index(INDEX, out)
        self.assertEqual(diffs, {}, f"not equivalent: {diffs}")

    def test_section_order_matches_taxonomy(self):
        out = digest.render_digest(self.conn)
        i_about = out.index("## About the User")
        i_setup = out.index("## Current Setup & Learning")
        i_feedback = out.index("## Feedback")
        self.assertLess(i_about, i_setup)
        self.assertLess(i_setup, i_feedback)

    def test_cold_user_node_dropped(self):
        self.demote("project_kali")
        out = digest.render_digest(self.conn)
        self.assertNotIn("project_kali.md", out)
        # and the diff now reports it missing vs the real index
        diffs = digest.compare_index(INDEX, out)
        self.assertIn("Current Setup & Learning", diffs)

    def test_pinned_endorsed_never_dropped_even_if_cold(self):
        # force an endorsed node to 'cold' — it must still render (pinned wins)
        self.demote("user_adhd")
        out = digest.render_digest(self.conn)
        self.assertIn("user_adhd.md", out)

    def test_uncategorized_file_excluded(self):
        # a file on disk but never in MEMORY.md has no section -> not in digest
        (self.mem / "project_orphan.md").write_text(
            "---\nname: Orphan\ntype: project\n---\nx\n", encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        out = digest.render_digest(self.conn)
        self.assertNotIn("project_orphan.md", out)


class TestBudget(Base):
    def test_drops_lowest_rs_user_first_under_tight_budget(self):
        # set distinct RS so the drop order is deterministic
        self.conn.execute("UPDATE nodes SET forget_rs = 0.9 WHERE source_ref = 'memory:project_kali'")
        self.conn.execute("UPDATE nodes SET forget_rs = 0.1 WHERE source_ref = 'memory:reference_vault'")
        self.conn.commit()
        full = digest.render_digest(self.conn)
        # budget just below full size forces dropping the lowest-RS user line
        tight = len(full.encode("utf-8")) - 5
        out = digest.render_digest(self.conn, budget=tight)
        self.assertNotIn("reference_vault.md", out)   # rs 0.1 dropped first
        self.assertIn("user_adhd.md", out)            # pinned kept

    def test_raises_when_pinned_exceed_budget(self):
        with self.assertRaises(digest.DigestTooLarge):
            digest.render_digest(self.conn, budget=10)  # can't fit pinned+literal


if __name__ == "__main__":
    unittest.main()
