"""Tests for memsom_bridge_import — flat-file memory -> memsom nodes (Phase 1).

Run:  python -m unittest discover -s . -p test_memsom_bridge_import.py
"""

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.bridge import bridge_import as bi


SAMPLE = {
    "user_adhd.md": "---\nname: ADHD\ndescription: has ADHD\ntype: user\n---\n\nbody one\n",
    "feedback_debug.md": "---\nname: Debug loop\ndescription: use the loop\ntype: feedback\n---\n\nrule\n",
    "personal_sam.md": "---\nname: Sam\ndescription: context\ntype: personal\n---\n\nnote\n",
    "project_kali.md": "---\nname: Kali VM\ndescription: status\ntype: project\nsalience: 0.30\n---\n\nstate\n",
    "reference_vault.md": "---\nname: Vault path\ndescription: where the vault is\ntype: reference\n---\n\npath\n",
}

INDEX = """# Memory - Alex

## About the User
- **Alex** — goal: cybersecurity
- [ADHD](user_adhd.md) — has ADHD

## Personal context
- [Sam](personal_sam.md) — context
⏰ **Progress check DUE 2026-06-30** — raise it proactively

## Current Setup & Learning
- [Kali VM](project_kali.md) — status

## References
- [Vault path](reference_vault.md) — where the vault is

## Feedback
- [Debug loop](feedback_debug.md) — use the loop
"""


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.mem = self.root / "memory"
        self.mem.mkdir()
        for name, text in SAMPLE.items():
            (self.mem / name).write_text(text, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")
        self.conn = memsom.get_connection()
        bi.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def live_node(self, rel):
        row = bi._live_node_for_path(self.conn, rel)
        return memsom.get_node(self.conn, row[0]) if row else None


# --- migrate now also lays the staleness columns (Mac-safe path) --------------

class TestStaleMigrate(Base):
    def test_migrate_adds_stale_columns_and_tables(self):
        from memsom.storage import schema as memsom_schema
        for col in ("stale", "stale_at", "stale_reason"):
            self.assertTrue(memsom_schema.column_exists(self.conn, "nodes", col),
                            f"missing nodes.{col}")
        tables = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertIn("source_supersedes", tables)
        self.assertIn("stale_log", tables)


# --- pure helpers -------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    def test_section_map(self):
        m = bi.section_map(INDEX)
        self.assertEqual(m["user_adhd.md"], "About the User")
        self.assertEqual(m["project_kali.md"], "Current Setup & Learning")
        self.assertEqual(m["feedback_debug.md"], "Feedback")
        # the H1 must not be treated as a section
        self.assertNotIn(None, m.values())

    def test_memory_type_from_frontmatter_then_prefix(self):
        self.assertEqual(bi.memory_type("anything", {"type": "feedback"}), "feedback")
        self.assertEqual(bi.memory_type("project_foo", {}), "project")
        self.assertEqual(bi.memory_type("nounderscores", {}), "nounderscores")

    def test_channel_mapping(self):
        self.assertEqual(bi.CHANNEL_BY_TYPE["user"], "endorsed")
        self.assertEqual(bi.CHANNEL_BY_TYPE["personal"], "endorsed")
        self.assertEqual(bi.CHANNEL_BY_TYPE["feedback"], "endorsed")
        self.assertEqual(bi.CHANNEL_BY_TYPE["project"], "user")
        self.assertEqual(bi.CHANNEL_BY_TYPE["reference"], "user")

    def test_stamp_section_idempotent(self):
        text = SAMPLE["project_kali.md"]
        once = bi.stamp_section(text, "Current Setup & Learning")
        twice = bi.stamp_section(once, "Current Setup & Learning")
        self.assertEqual(once, twice)              # re-stamping is stable
        self.assertIn("section: Current Setup & Learning", once)
        # body + original keys survive
        self.assertIn("state", once)
        self.assertIn("salience: 0.30", once)
        # exactly one section line
        self.assertEqual(once.count("\nsection: "), 1)

    def test_stamp_section_none_no_frontmatter_noop(self):
        self.assertEqual(bi.stamp_section("plain body", None), "plain body")

    def test_stamp_section_replaces_existing(self):
        text = "---\nname: x\nsection: Old\n---\nbody\n"
        out = bi.stamp_section(text, "New")
        self.assertIn("section: New", out)
        self.assertNotIn("Old", out)
        self.assertEqual(out.count("\nsection: "), 1)


# --- import behaviour ---------------------------------------------------------

class TestImport(Base):
    def test_dry_run_writes_nothing(self):
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=True)
        self.assertEqual(stats["created"], len(SAMPLE))
        self.assertEqual(stats["total_files"], len(SAMPLE))
        # nothing actually persisted
        n = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        self.assertEqual(n, 0)

    def test_apply_creates_one_node_per_file(self):
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["created"], len(SAMPLE))
        live = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tombstoned = 0").fetchone()[0]
        self.assertEqual(live, len(SAMPLE))

    def test_channels_assigned_by_type(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(self.live_node("user_adhd.md")["channel"], "endorsed")
        self.assertEqual(self.live_node("personal_sam.md")["channel"], "endorsed")
        self.assertEqual(self.live_node("feedback_debug.md")["channel"], "endorsed")
        self.assertEqual(self.live_node("project_kali.md")["channel"], "user")
        self.assertEqual(self.live_node("reference_vault.md")["channel"], "user")

    def test_section_stamped_into_content(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        node = self.live_node("project_kali.md")
        self.assertIn("section: Current Setup & Learning", node["content"])
        # original body + frontmatter preserved
        self.assertIn("state", node["content"])
        self.assertIn("salience: 0.30", node["content"])

    def test_reimport_is_idempotent(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["created"], 0)
        self.assertEqual(stats["updated"], 0)
        self.assertEqual(stats["skipped"], len(SAMPLE))
        live = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tombstoned = 0").fetchone()[0]
        self.assertEqual(live, len(SAMPLE))

    def test_changed_file_tombstones_old_inserts_new(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        old = bi._live_node_for_path(self.conn, "project_kali.md")[0]
        (self.mem / "project_kali.md").write_text(
            SAMPLE["project_kali.md"].replace("state", "NEW state"), encoding="utf-8")
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["tombstoned"], 1)
        self.assertEqual(stats["skipped"], len(SAMPLE) - 1)
        new_node = self.live_node("project_kali.md")
        self.assertIn("NEW state", new_node["content"])
        self.assertNotEqual(new_node["id"], old)
        # old node is tombstoned, not deleted (history preserved)
        self.assertEqual(memsom.get_node(self.conn, old)["tombstoned"], 1)
        # still exactly one LIVE node for the path
        live_for_path = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE bridge_path = ? AND tombstoned = 0",
            ("project_kali.md",)).fetchone()[0]
        self.assertEqual(live_for_path, 1)

    def test_cold_file_not_in_index_still_imports(self):
        (self.mem / "project_orphan.md").write_text(
            "---\nname: Orphan\ntype: project\n---\nbody\n", encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        node = self.live_node("project_orphan.md")
        self.assertIsNotNone(node)                       # imported anyway
        self.assertNotIn("\nsection: ", node["content"])  # section-less (cold)

    def test_memory_md_itself_not_imported(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertIsNone(bi._live_node_for_path(self.conn, "MEMORY.md"))


class TestLiterals(Base):
    def _literal_nodes(self):
        return self.conn.execute(
            "SELECT content FROM nodes WHERE source_ref LIKE 'memory:literal:%' "
            "AND tombstoned = 0").fetchall()

    def test_parse_index_entries_classifies(self):
        entries = list(bi.parse_index_entries(INDEX))
        files = [p for s, k, p in entries if k == "file"]
        lits = [p for s, k, p in entries if k == "literal"]
        self.assertIn("user_adhd.md", files)
        # the two file-less lines are literals
        self.assertTrue(any("Alex" in t for t in lits))
        self.assertTrue(any("Progress check" in t for t in lits))
        # a backtick'd .md inside a literal must NOT be misread as a file link
        self.assertNotIn("progress-check-2026-05-31.md", files)

    def test_literals_imported_as_endorsed(self):
        bi.import_literals(self.conn, self.mem, dry_run=False)
        rows = self._literal_nodes()
        self.assertEqual(len(rows), 2)  # Alex lead + progress-check
        ch = self.conn.execute(
            "SELECT DISTINCT channel FROM nodes WHERE source_ref LIKE 'memory:literal:%'"
        ).fetchall()
        self.assertEqual(ch, [("endorsed",)])
        # the verbatim line is preserved in the body
        self.assertTrue(any("goal: cybersecurity" in r[0] for r in rows))

    def test_literals_idempotent(self):
        bi.import_literals(self.conn, self.mem, dry_run=False)
        stats = bi.import_literals(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["created"], 0)
        self.assertEqual(stats["skipped"], 2)
        self.assertEqual(len(self._literal_nodes()), 2)

    def test_removed_literal_is_tombstoned(self):
        bi.import_literals(self.conn, self.mem, dry_run=False)
        # drop the progress-check line from the index
        idx = (self.mem / "MEMORY.md").read_text(encoding="utf-8")
        idx = "\n".join(l for l in idx.split("\n") if "Progress check" not in l)
        (self.mem / "MEMORY.md").write_text(idx, encoding="utf-8")
        stats = bi.import_literals(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["tombstoned"], 1)
        self.assertEqual(len(self._literal_nodes()), 1)  # only the lead remains

    def test_import_all_combines(self):
        stats = bi.import_all(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["files"]["created"], len(SAMPLE))
        self.assertEqual(stats["literals"]["created"], 2)


class TestSweep(Base):
    """Reconcile deletions: a removed source file tombstones its node."""

    def test_deleted_file_node_is_swept(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        (self.mem / "project_kali.md").unlink()
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["swept"], 1)
        self.assertIsNone(bi._live_node_for_path(self.conn, "project_kali.md"))
        live = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE tombstoned = 0").fetchone()[0]
        self.assertEqual(live, len(SAMPLE) - 1)

    def test_sweep_dry_run_counts_but_keeps_node(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        (self.mem / "project_kali.md").unlink()
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=True)
        self.assertEqual(stats["swept"], 1)
        self.assertIsNotNone(bi._live_node_for_path(self.conn, "project_kali.md"))

    def test_clean_reimport_sweeps_nothing(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["swept"], 0)

    def test_sweep_spares_literal_nodes(self):
        bi.import_all(self.conn, self.mem, dry_run=False)
        lits_before = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE source_ref LIKE 'memory:literal:%' "
            "AND tombstoned = 0").fetchone()[0]
        self.assertGreater(lits_before, 0)
        (self.mem / "project_kali.md").unlink()
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        lits_after = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE source_ref LIKE 'memory:literal:%' "
            "AND tombstoned = 0").fetchone()[0]
        self.assertEqual(lits_after, lits_before)  # file sweep never touches literals


# --- Pass 2: [[wikilinks]] in bodies become associative rel_edges -------------

class TestRelateWikilinks(Base):
    def _wire_bodies(self):
        # user_adhd links to two real siblings, one not-yet-written target, and
        # itself; the code-fenced link must NOT become an edge.
        (self.mem / "user_adhd.md").write_text(
            "---\nname: ADHD\ntype: user\n---\n\n"
            "Links to [[personal_sam]] and [[feedback_debug]]. "
            "Future note [[reference_not_written_yet]]. Self [[user_adhd]].\n"
            "```\nfenced [[project_kali]] must be ignored\n```\n",
            encoding="utf-8")

    def test_wikilinks_create_edges_and_traverse(self):
        from memsom.retrieval import relate as memsom_relate
        self._wire_bodies()
        stats = bi.import_all(self.conn, self.mem, dry_run=False)["edges"]
        self.assertEqual(stats["edges"], 2)          # sam + debug
        self.assertEqual(stats["resolved"], 2)
        self.assertGreaterEqual(stats["unresolved"], 1)  # the not-written target
        self.assertEqual(stats["skipped_self"], 1)   # [[user_adhd]] self-link

        src = bi._live_node_for_path(self.conn, "user_adhd.md")[0]
        sam = bi._live_node_for_path(self.conn, "personal_sam.md")[0]
        dbg = bi._live_node_for_path(self.conn, "feedback_debug.md")[0]
        kali = bi._live_node_for_path(self.conn, "project_kali.md")[0]
        nbrs = {d["id"] for d in memsom_relate.neighborhood(self.conn, src, hops=1)}
        self.assertIn(sam, nbrs)
        self.assertIn(dbg, nbrs)
        self.assertNotIn(kali, nbrs)                 # fenced link excluded

    def test_relate_pass_is_idempotent(self):
        self._wire_bodies()
        bi.import_all(self.conn, self.mem, dry_run=False)
        n1 = self.conn.execute("SELECT COUNT(*) FROM rel_edges").fetchone()[0]
        bi.import_all(self.conn, self.mem, dry_run=False)  # re-run, no file change
        n2 = self.conn.execute("SELECT COUNT(*) FROM rel_edges").fetchone()[0]
        self.assertEqual(n1, n2)                      # INSERT OR IGNORE, no dupes

    def test_dry_run_writes_no_edges(self):
        self._wire_bodies()
        stats = bi.import_all(self.conn, self.mem, dry_run=True)["edges"]
        self.assertEqual(stats["edges"], 0)           # nothing written in dry-run
        # rel_edges table may not even exist in a pure dry-run; count defensively.
        try:
            n = self.conn.execute("SELECT COUNT(*) FROM rel_edges").fetchone()[0]
        except Exception:
            n = 0
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
