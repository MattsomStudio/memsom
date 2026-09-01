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
        # the bridge now indexes every imported node; keep the tests off the
        # network (an Ollama embed is ~1s/node) -> BM25-only for the suite
        self._embed_prev = os.environ.get("MEMDAG_EMBED_BACKEND")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
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
        if self._embed_prev is None:
            os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        else:
            os.environ["MEMDAG_EMBED_BACKEND"] = self._embed_prev
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
        self.assertEqual(bi.CHANNEL_BY_TYPE["fact"], "user")  # fact layer Phase 0

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

    def test_section_move_updates_literal(self):
        """Moving a literal line under a different ## heading must stick.

        Regression: the sref keys on the LINE TEXT hash only, and the skip path
        never compared the stored section — so a hand-moved literal was silently
        reverted to its old section on the next render.
        """
        bi.import_literals(self.conn, self.mem, dry_run=False)
        # same two literal lines, but the progress-check moved to a new section
        (self.mem / "MEMORY.md").write_text(
            "# Memory - Alex\n\n"
            "## About the User\n- **Alex** — goal: cybersecurity\n\n"
            "## Reminders\n⏰ **Progress check DUE 2026-06-30** — raise it proactively\n",
            encoding="utf-8")

        stats = bi.import_literals(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["updated"], 1)      # the moved line
        self.assertEqual(stats["skipped"], 1)      # the unmoved line
        rows = self._literal_nodes()
        self.assertEqual(len(rows), 2)             # still exactly two live literals
        prog = next(r[0] for r in rows if "Progress check" in r[0])
        self.assertIn("section: Reminders", prog)

    def test_section_move_dry_run_writes_nothing(self):
        bi.import_literals(self.conn, self.mem, dry_run=False)
        idx = (self.mem / "MEMORY.md").read_text(encoding="utf-8")
        idx = idx.replace("## Personal context", "## Renamed context")
        (self.mem / "MEMORY.md").write_text(idx, encoding="utf-8")
        stats = bi.import_literals(self.conn, self.mem, dry_run=True)
        self.assertEqual(stats["updated"], 1)
        prog = next(r[0] for r in self._literal_nodes() if "Progress check" in r[0])
        self.assertIn("section: Personal context", prog)  # untouched


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


# --- Fact layer Phase 0 (docs/facts-design.md) --------------------------------
#
# A fact is a normal memory file distinguished by type: fact. Two things to
# prove here: (1) it imports through the ordinary path with no special-casing
# (channel from CHANNEL_BY_TYPE, section self-filed via the frontmatter
# fallback since facts are never in MEMORY.md's curated index), and (2) the
# `[[...]]` reference form other memories must use to link it, because
# relate_wikilinks resolves against the FILENAME STEM (via _build_resolver /
# _resolve_target in memsom.bridge.obsidian), not the frontmatter `name:`
# slug. The spec's own example uses a kebab name (`fact-5070-toksps`) as the
# reference target; the filename convention (`fact_<snake_name>.md`) is
# underscored, so a kebab-form wikilink silently fails to resolve. See the
# fix documented in docs/facts-design.md.

FACT_FILE = (
    "---\n"
    "name: fact-5070-toksps\n"
    "description: RTX 5070 local LLM throughput\n"
    "type: fact\n"
    "value: 61\n"
    "unit: tok/s\n"
    "last-verified: 2026-07-14\n"
    "section: Facts\n"
    "---\n"
    "Optional context: how it was measured, on what workload.\n"
)


class TestFactLayerPhase0(Base):
    def _write_fact(self):
        (self.mem / "fact_5070_toksps.md").write_text(FACT_FILE, encoding="utf-8")

    def test_fact_file_imports_with_user_channel(self):
        self._write_fact()
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        node = self.live_node("fact_5070_toksps.md")
        self.assertIsNotNone(node)
        self.assertEqual(node["channel"], "user")

    def test_fact_section_self_files_from_frontmatter(self):
        """A fact is never in MEMORY.md's curated index — it must file itself
        via the `section:` frontmatter fallback (same path a brand-new,
        never-indexed memory takes; see TestIndexMetaFallback)."""
        self._write_fact()
        self.assertNotIn("fact_5070_toksps.md", INDEX)  # sanity: not curated
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        node = self.live_node("fact_5070_toksps.md")
        fm = bi.fm_top_level(bi.split_frontmatter(node["content"])[0])
        self.assertEqual(fm.get("section"), "Facts")

    def test_fact_referenced_by_filename_stem_resolves(self):
        """The reference form that actually works: the filename stem
        (underscored, per the `fact_<snake_name>.md` convention) — NOT the
        kebab `name:` slug from the frontmatter."""
        from memsom.retrieval import relate as memsom_relate
        self._write_fact()
        (self.mem / "user_adhd.md").write_text(
            "---\nname: ADHD\ntype: user\n---\n\n"
            "Throughput is [[fact_5070_toksps]].\n", encoding="utf-8")
        edges = bi.import_all(self.conn, self.mem, dry_run=False)["edges"]
        self.assertEqual(edges["resolved"], 1)
        self.assertEqual(edges["unresolved"], 0)

        src = bi._live_node_for_path(self.conn, "user_adhd.md")[0]
        fact = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]
        nbrs = {d["id"] for d in memsom_relate.neighborhood(self.conn, src, hops=1)}
        self.assertIn(fact, nbrs)

    def test_fact_referenced_by_kebab_name_does_not_resolve(self):
        """Documents the spec-vs-code mismatch: the frontmatter `name:` slug
        (kebab, matching the design doc's own example) is NOT what
        relate_wikilinks resolves against, so a link written that way is
        silently inert."""
        self._write_fact()
        (self.mem / "user_adhd.md").write_text(
            "---\nname: ADHD\ntype: user\n---\n\n"
            "Throughput is [[fact-5070-toksps]].\n", encoding="utf-8")
        edges = bi.import_all(self.conn, self.mem, dry_run=False)["edges"]
        self.assertEqual(edges["resolved"], 0)
        self.assertEqual(edges["unresolved"], 1)


# --- Fact layer Phase 1: dependencies and cascade (docs/facts-design.md) ------
#
# `depends_on:` is derivation, not association — it lands in the `edges` table
# CASCADE_CTE walks (memsom.derive_node/revoke_cascade/mark_stale_cascade),
# NOT rel_edges. parent = the depended-on fact's live node, child = the
# dependent fact's live node. Five behaviors, each regression-tested below.

GPU_FACT = (
    "---\n"
    "name: fact-pc-gpu\n"
    "type: fact\n"
    "value: RTX 5070\n"
    "last-verified: 2026-07-14\n"
    "section: Facts\n"
    "---\n"
    "GPU note.\n"
)

TOKSPS_FACT = (
    "---\n"
    "name: fact-5070-toksps\n"
    "type: fact\n"
    "value: 61\n"
    "unit: tok/s\n"
    "last-verified: 2026-07-14\n"
    "depends_on: fact_pc_gpu\n"
    "section: Facts\n"
    "---\n"
    "Throughput note.\n"
)


class TestFactDependencyCascade(Base):
    def _write_gpu_toksps(self):
        (self.mem / "fact_pc_gpu.md").write_text(GPU_FACT, encoding="utf-8")
        (self.mem / "fact_5070_toksps.md").write_text(TOKSPS_FACT, encoding="utf-8")

    def _edge_exists(self, child, parent):
        return self.conn.execute(
            "SELECT 1 FROM edges WHERE child = ? AND parent = ?", (child, parent)
        ).fetchone() is not None

    # --- Behavior 1: depends_on creates the edge; idempotent ------------------

    def test_depends_on_creates_edge_and_is_idempotent(self):
        self._write_gpu_toksps()
        stats = bi.import_all(self.conn, self.mem, dry_run=False)["fact_edges"]
        self.assertEqual(stats["resolved"], 1)
        self.assertEqual(stats["unresolved"], 0)

        gpu = bi._live_node_for_path(self.conn, "fact_pc_gpu.md")[0]
        toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]
        self.assertTrue(self._edge_exists(toksps, gpu))

        n_before = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        stats2 = bi.import_all(self.conn, self.mem, dry_run=False)["fact_edges"]
        n_after = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        self.assertEqual(n_before, n_after)   # re-import, no change -> creates nothing
        self.assertEqual(stats2["resolved"], 1)  # still resolves; just no new row

    # --- Behavior 2: supersede rewires edges, both as parent and as child -----

    def test_supersede_of_depended_on_fact_rewires_parent_edge(self):
        """The depended-on fact (GPU) changes value -> new live node; the
        dependent must be reachable from the NEW parent via cascade_set (the
        same CASCADE_CTE walk mark_stale_cascade/revoke_cascade use)."""
        self._write_gpu_toksps()
        bi.import_all(self.conn, self.mem, dry_run=False)
        old_gpu = bi._live_node_for_path(self.conn, "fact_pc_gpu.md")[0]

        (self.mem / "fact_pc_gpu.md").write_text(
            GPU_FACT.replace("RTX 5070", "RTX 5090"), encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)

        new_gpu = bi._live_node_for_path(self.conn, "fact_pc_gpu.md")[0]
        toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]
        self.assertNotEqual(new_gpu, old_gpu)
        self.assertEqual(memsom.get_node(self.conn, old_gpu)["tombstoned"], 1)

        descendants = {r[0] for r in memsom.cascade_set(self.conn, new_gpu)}
        self.assertIn(toksps, descendants)

    def test_supersede_of_dependent_fact_rewires_child_edge(self):
        """The dependent fact (TPS reading) itself gets a fresh value -> new
        live node; the (unchanged) parent must reach the NEW child, not the
        stale predecessor."""
        self._write_gpu_toksps()
        bi.import_all(self.conn, self.mem, dry_run=False)
        gpu = bi._live_node_for_path(self.conn, "fact_pc_gpu.md")[0]
        old_toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]

        (self.mem / "fact_5070_toksps.md").write_text(
            TOKSPS_FACT.replace("value: 61", "value: 68"), encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)

        new_toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]
        self.assertNotEqual(new_toksps, old_toksps)

        descendants = {r[0] for r in memsom.cascade_set(self.conn, gpu)}
        self.assertIn(new_toksps, descendants)

    # --- Behavior 3: missing target defers, resolves once it arrives ----------

    def test_missing_dependency_target_defers_then_resolves(self):
        (self.mem / "fact_5070_toksps.md").write_text(TOKSPS_FACT, encoding="utf-8")
        # fact_pc_gpu.md does not exist yet -> depends_on target is unresolved,
        # never a crash.
        stats = bi.import_all(self.conn, self.mem, dry_run=False)["fact_edges"]
        self.assertEqual(stats["unresolved"], 1)
        self.assertEqual(stats["resolved"], 0)
        toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM edges WHERE child = ?", (toksps,)).fetchone()[0], 0)

        # the target arrives on a later run -> picked up automatically
        (self.mem / "fact_pc_gpu.md").write_text(GPU_FACT, encoding="utf-8")
        stats2 = bi.import_all(self.conn, self.mem, dry_run=False)["fact_edges"]
        self.assertEqual(stats2["resolved"], 1)
        self.assertEqual(stats2["unresolved"], 0)
        gpu = bi._live_node_for_path(self.conn, "fact_pc_gpu.md")[0]
        self.assertTrue(self._edge_exists(toksps, gpu))

    # --- Behavior 4: deleting the fact FILE stales dependents, never tombstones them

    def test_deleted_fact_file_marks_dependent_stale_not_tombstoned(self):
        self._write_gpu_toksps()
        bi.import_all(self.conn, self.mem, dry_run=False)
        gpu = bi._live_node_for_path(self.conn, "fact_pc_gpu.md")[0]
        toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]

        (self.mem / "fact_pc_gpu.md").unlink()
        stats = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(stats["swept"], 1)
        self.assertGreater(stats["stale_cascaded"], 0)

        gpu_row = self.conn.execute(
            "SELECT tombstoned, stale FROM nodes WHERE id = ?", (gpu,)).fetchone()
        self.assertEqual(gpu_row[0], 1)          # GPU fact retired (tombstoned)

        toksps_row = self.conn.execute(
            "SELECT tombstoned, stale FROM nodes WHERE id = ?", (toksps,)).fetchone()
        self.assertEqual(toksps_row[0], 0)       # dependent stays LIVE
        self.assertEqual(toksps_row[1], 1)       # ...flagged stale for reverification

    def test_sweep_of_unrelated_file_does_not_stale_unrelated_facts(self):
        """A deleted memory file nothing depends_on cascades no further than
        itself — no unrelated fact goes stale as a side effect."""
        self._write_gpu_toksps()
        bi.import_all(self.conn, self.mem, dry_run=False)
        toksps = bi._live_node_for_path(self.conn, "fact_5070_toksps.md")[0]

        (self.mem / "project_kali.md").unlink()   # unrelated; nothing depends on it
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)

        toksps_row = self.conn.execute(
            "SELECT stale FROM nodes WHERE id = ?", (toksps,)).fetchone()
        self.assertEqual(toksps_row[0], 0)

    # --- Behavior 5: a depends_on cycle must not hang or crash -----------------

    def test_dependency_cycle_does_not_hang(self):
        (self.mem / "fact_a.md").write_text(
            "---\nname: fact-a\ntype: fact\nvalue: 1\ndepends_on: fact_b\n"
            "section: Facts\n---\nA.\n", encoding="utf-8")
        (self.mem / "fact_b.md").write_text(
            "---\nname: fact-b\ntype: fact\nvalue: 2\ndepends_on: fact_a\n"
            "section: Facts\n---\nB.\n", encoding="utf-8")

        stats = bi.import_all(self.conn, self.mem, dry_run=False)["fact_edges"]
        self.assertEqual(stats["resolved"], 2)
        self.assertEqual(stats["unresolved"], 0)

        a = bi._live_node_for_path(self.conn, "fact_a.md")[0]
        b = bi._live_node_for_path(self.conn, "fact_b.md")[0]

        # CASCADE_CTE already dedupes cycles (UNION, not UNION ALL) - exercised
        # here through the depends_on-populated edges table specifically.
        descendants = {r[0] for r in memsom.cascade_set(self.conn, a)}
        self.assertEqual(descendants, {a, b})

        from memsom.lifecycle import stale as memsom_stale
        n = memsom_stale.mark_stale_cascade(self.conn, a, "cycle test")
        self.assertEqual(n, 2)   # both a and b marked stale; no infinite loop


if __name__ == "__main__":
    unittest.main()


# --- index metadata must survive absence from MEMORY.md -----------------------
#
# digest._select_hot requires a truthy `section`, and section was sourced ONLY
# from the curated MEMORY.md index. So any file absent from the index resolved
# section=None and became permanently unselectable — even while hot. Two live
# failures (2026-07-13): a brand-new memory could never enter MEMORY.md at all,
# and a memory the digest's byte-budget evicted lost its section on the next
# import, turning a transient RS-ordered eviction into a permanent unfiling.

class TestIndexMetaFallback(Base):
    def _section_of(self, rel):
        node = self.live_node(rel)
        fm = bi.fm_top_level(bi.split_frontmatter(node["content"])[0])
        return fm.get("section")

    def test_new_memory_not_in_index_keeps_frontmatter_section(self):
        """A brand-new memory files itself via its own frontmatter."""
        (self.mem / "reference_new.md").write_text(
            "---\nname: New\ndescription: d\ntype: reference\n"
            "section: References\nindex_title: New thing\nindex_hook: the hook\n"
            "---\n\nbody\n", encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)

        self.assertEqual(self._section_of("reference_new.md"), "References")
        fm = bi.fm_top_level(
            bi.split_frontmatter(self.live_node("reference_new.md")["content"])[0])
        self.assertEqual(fm.get("index_title"), "New thing")
        self.assertEqual(fm.get("index_hook"), "the hook")

    def test_eviction_from_index_does_not_wipe_section(self):
        """A memory dropped from MEMORY.md keeps the section already stamped on it.

        Regression: eviction is a transient, RS-ordered budget decision made by the
        digest — it must never destroy the memory's filing metadata, or the entry
        can never come back.
        """
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(self._section_of("user_adhd.md"), "About the User")

        # the digest evicts it: its line disappears from MEMORY.md, file untouched
        index = (self.mem / "MEMORY.md").read_text(encoding="utf-8")
        evicted = "\n".join(l for l in index.splitlines()
                            if "user_adhd.md" not in l)
        (self.mem / "MEMORY.md").write_text(evicted, encoding="utf-8")
        # touch the file so the importer re-reads it rather than hash-skipping
        (self.mem / "user_adhd.md").write_text(
            SAMPLE["user_adhd.md"] + "\nmore\n", encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)

        self.assertEqual(self._section_of("user_adhd.md"), "About the User")

    def test_curated_index_still_wins_over_frontmatter(self):
        """MEMORY.md remains the curated source when it HAS a line for the file."""
        (self.mem / "user_adhd.md").write_text(
            "---\nname: ADHD\ndescription: has ADHD\ntype: user\n"
            "section: Wrong Section\n---\n\nbody\n", encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(self._section_of("user_adhd.md"), "About the User")


class TestProjectsSubdir(Base):
    """memory/projects/*.md imports exactly like a flat file (basename-keyed)."""

    def _write_project(self, name="project_widget.md", body="state\n"):
        sub = self.mem / "projects"
        sub.mkdir(exist_ok=True)
        path = sub / name
        path.write_text(
            f"---\nname: Widget\ndescription: d\ntype: project\n"
            f"section: Personal projects\n---\n\n{body}", encoding="utf-8")
        return path

    def test_iter_memory_files_walks_depth_two_and_skips_indexes(self):
        self._write_project()
        (self.mem / "projects" / "INDEX.md").write_text("# Projects\n", encoding="utf-8")
        (self.mem / "projects" / "acme").mkdir()
        (self.mem / "projects" / "acme" / "project_acme.md").write_text("x", encoding="utf-8")
        (self.mem / "projects" / "acme" / "project_acme_api.md").write_text("x", encoding="utf-8")
        (self.mem / "projects" / "acme" / "INDEX.md").write_text("x", encoding="utf-8")
        (self.mem / "projects" / "acme" / "deeper").mkdir()
        (self.mem / "projects" / "acme" / "deeper" / "project_nested.md").write_text("x", encoding="utf-8")
        files = bi.iter_memory_files(self.mem)
        names = [p.name for p in files]
        self.assertIn("project_widget.md", names)          # projects/
        self.assertIn("project_acme.md", names)            # projects/<slug>/
        self.assertIn("project_acme_api.md", names)
        self.assertNotIn("MEMORY.md", names)
        self.assertEqual(names.count("INDEX.md"), 0)
        self.assertNotIn("project_nested.md", names)       # depth 3 ignored
        self.assertEqual(len(names), len(SAMPLE) + 3)
        subdirs = {p.name: bi.memory_subdir(self.mem, p) for p in files}
        self.assertIsNone(subdirs["user_adhd.md"])
        self.assertEqual(subdirs["project_widget.md"], "projects")
        self.assertEqual(subdirs["project_acme_api.md"], "projects/acme")

    def test_roundtrip_import_reimport_tombstone(self):
        path = self._write_project()
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["created"], len(SAMPLE) + 1)
        node = self.live_node("project_widget.md")
        self.assertIsNotNone(node)
        self.assertEqual(node["source_ref"], "memory:project_widget")
        fm = bi.fm_top_level(bi.split_frontmatter(node["content"])[0])
        self.assertEqual(fm.get("memory_subdir"), "projects")
        self.assertEqual(fm.get("section"), "Personal projects")
        # unchanged re-import: skipped, nothing created
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["created"], 0)
        self.assertEqual(st["skipped"], len(SAMPLE) + 1)
        # edit: supersede like a flat file
        path.write_text(path.read_text(encoding="utf-8") + "more\n", encoding="utf-8")
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["updated"], 1)
        # delete: swept like a flat file
        path.unlink()
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["swept"], 1)
        self.assertIsNone(self.live_node("project_widget.md"))

    def test_move_between_levels_is_an_edit_not_a_new_memory(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        old = self.live_node("project_kali.md")
        (self.mem / "projects").mkdir()
        (self.mem / "project_kali.md").rename(self.mem / "projects" / "project_kali.md")
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual((st["created"], st["updated"], st["swept"]), (0, 1, 0))
        new = self.live_node("project_kali.md")
        self.assertNotEqual(new["id"], old["id"])
        fm = bi.fm_top_level(bi.split_frontmatter(new["content"])[0])
        self.assertEqual(fm.get("memory_subdir"), "projects")
        self.assertEqual(fm.get("section"), "Current Setup & Learning")  # curated line kept

    def test_wikilink_resolves_across_levels_by_stem(self):
        self._write_project(body="see [[user_adhd]] and [[project_acme_api]]\n")
        (self.mem / "projects" / "acme").mkdir()
        (self.mem / "projects" / "acme" / "project_acme_api.md").write_text(
            "---\nname: API\ntype: project\n---\nsee [[reference_vault]]\n", encoding="utf-8")
        (self.mem / "reference_vault.md").write_text(
            SAMPLE["reference_vault.md"] + "\nsee [[project_widget]]\n", encoding="utf-8")
        st = bi.import_all(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["edges"]["unresolved"], 0)
        # flat->projects/, projects/->flat, projects/->projects/<slug>/, <slug>/->flat
        self.assertEqual(st["edges"]["resolved"], 4)
        fm = bi.fm_top_level(bi.split_frontmatter(
            self.live_node("project_acme_api.md")["content"])[0])
        self.assertEqual(fm.get("memory_subdir"), "projects/acme")

    def test_move_into_project_dir_is_an_edit(self):
        self._write_project()
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        old = self.live_node("project_widget.md")
        (self.mem / "projects" / "widget").mkdir()
        (self.mem / "projects" / "project_widget.md").rename(
            self.mem / "projects" / "widget" / "project_widget.md")
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual((st["created"], st["updated"], st["swept"]), (0, 1, 0))
        new = self.live_node("project_widget.md")
        self.assertNotEqual(new["id"], old["id"])
        fm = bi.fm_top_level(bi.split_frontmatter(new["content"])[0])
        self.assertEqual(fm.get("memory_subdir"), "projects/widget")

    # --- duplicate stems: self-heal, never freeze (additive-sync leftover) ----

    def test_identical_duplicate_keeps_deepest_and_deletes_shallow(self):
        """robocopy /E / rsync --update leave the OLD flat copy behind after a
        MOVE into projects/: byte-identical -> the deep copy is canonical, the
        shallow one is deleted, the import continues and logs `dedup`."""
        flat = self.mem / "project_kali.md"
        deep_dir = self.mem / "projects" / "kali"
        deep_dir.mkdir(parents=True)
        deep = deep_dir / "project_kali.md"
        deep.write_bytes(flat.read_bytes())
        canonical, dups = bi.resolve_duplicates(self.mem)
        self.assertEqual(canonical["project_kali.md"], deep)
        self.assertEqual(dups[0]["action"], "delete")
        # iter never raises and returns exactly one copy
        names = [p.name for p in bi.iter_memory_files(self.mem)]
        self.assertEqual(names.count("project_kali.md"), 1)
        # dry-run: counts only, disk untouched
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=True)
        self.assertEqual(st["dedup"], 1)
        self.assertTrue(flat.exists())
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual((st["dedup"], st["quarantined"]), (1, 0))
        self.assertFalse(flat.exists())
        self.assertTrue(deep.exists())
        node = self.live_node("project_kali.md")
        fm = bi.fm_top_level(bi.split_frontmatter(node["content"])[0])
        self.assertEqual(fm.get("memory_subdir"), "projects/kali")

    def test_differing_duplicate_keeps_newer_and_quarantines_older(self):
        import os as _os
        import time as _time
        flat = self.mem / "project_kali.md"
        deep_dir = self.mem / "projects" / "kali"
        deep_dir.mkdir(parents=True)
        deep = deep_dir / "project_kali.md"
        deep.write_text(flat.read_text(encoding="utf-8") + "\nedited elsewhere\n",
                        encoding="utf-8")
        # make the FLAT copy the newer one so "newest wins" is what is tested,
        # not "deepest wins"
        old = _time.time() - 3600
        _os.utime(deep, (old, old))
        canonical, dups = bi.resolve_duplicates(self.mem)
        self.assertEqual(canonical["project_kali.md"], flat)
        self.assertEqual(dups[0]["action"], "quarantine")
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual((st["dedup"], st["quarantined"]), (0, 1))
        self.assertTrue(flat.exists())
        self.assertFalse(deep.exists())
        qdir = self.mem / ".weights" / "dup_quarantine"
        moved = list(qdir.glob("project_kali.*.md"))
        self.assertEqual(len(moved), 1)
        self.assertIn("edited elsewhere", moved[0].read_text(encoding="utf-8"))
        self.assertEqual(st["duplicates"][0]["quarantined_to"], str(moved[0]))
        # the quarantine dir is under .weights, so it is never walked as memory
        self.assertEqual([p.name for p in bi.iter_memory_files(self.mem)].count(
            "project_kali.md"), 1)

    def test_duplicate_raises_only_when_quarantine_write_fails(self):
        from unittest import mock
        flat = self.mem / "project_kali.md"
        deep_dir = self.mem / "projects" / "kali"
        deep_dir.mkdir(parents=True)
        (deep_dir / "project_kali.md").write_text("different", encoding="utf-8")
        with mock.patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(bi.DuplicateMemoryStem) as cm:
                bi.heal_duplicates(self.mem, dry_run=False)
        self.assertIn("project_kali.md", str(cm.exception))
        self.assertTrue(flat.exists())


class TestExplicitUnsection(Base):
    """`section: none` / `index: false` in the file withdraws it from the index."""

    def _section_of(self, rel):
        node = self.live_node(rel)
        fm = bi.fm_top_level(bi.split_frontmatter(node["content"])[0])
        return fm.get("section")

    def test_section_none_clears_stored_section_and_digest_excludes(self):
        from memsom.lifecycle import forget
        from memsom.distill import digest
        forget.migrate(self.conn)
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(self._section_of("reference_vault.md"), "References")
        # the file withdraws itself; the curated MEMORY.md line still exists
        (self.mem / "reference_vault.md").write_text(
            "---\nname: Vault path\ndescription: d\ntype: reference\n"
            "section: None\n---\n\npath\n", encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertIsNone(self._section_of("reference_vault.md"))
        forget.recompute_forget(self.conn)
        excluded = []
        out = digest.render_digest(self.conn, excluded_out=excluded)
        self.assertNotIn("reference_vault.md", out)
        self.assertIn({"stem": "reference_vault", "reason": "unsectioned"},
                      [{k: e[k] for k in ("stem", "reason")} for e in excluded])

    def test_index_false_also_withdraws(self):
        (self.mem / "reference_vault.md").write_text(
            "---\nname: Vault path\ntype: reference\nindex: false\n---\n\npath\n",
            encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertIsNone(self._section_of("reference_vault.md"))


class TestRetrievalIndexUpkeep(Base):
    """The bridge keeps postings/docstats current: insert_node never indexes,
    so without this every bridge-imported node was invisible to retrieve."""

    def _indexed(self, nid):
        n_post = self.conn.execute(
            "SELECT COUNT(*) FROM postings WHERE node_id = ?", (nid,)).fetchone()[0]
        n_doc = self.conn.execute(
            "SELECT COUNT(*) FROM docstats WHERE node_id = ?", (nid,)).fetchone()[0]
        return n_post, n_doc

    def test_import_indexes_new_nodes(self):
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["indexed"], len(SAMPLE))
        self.assertEqual(st["deindexed"], 0)
        nid = bi._live_node_for_path(self.conn, "user_adhd.md")[0]
        n_post, n_doc = self._indexed(nid)
        self.assertGreater(n_post, 0)
        self.assertEqual(n_doc, 1)

    def test_bm25_finds_a_term_from_the_file(self):
        from memsom.retrieval import retrieve as rt
        bi.import_all(self.conn, self.mem, dry_run=False)
        hits = rt.bm25(self.conn, "debug loop rule")
        target = bi._live_node_for_path(self.conn, "feedback_debug.md")[0]
        self.assertIn(target, [h[0] if isinstance(h, (tuple, list)) else h["id"]
                               for h in hits])

    def test_reimport_deindexes_superseded_and_indexes_new(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        old = bi._live_node_for_path(self.conn, "user_adhd.md")[0]
        (self.mem / "user_adhd.md").write_text(
            SAMPLE["user_adhd.md"] + "\nzebraquark\n", encoding="utf-8")
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual((st["indexed"], st["deindexed"]), (1, 1))
        new = bi._live_node_for_path(self.conn, "user_adhd.md")[0]
        self.assertEqual(self._indexed(old), (0, 0))
        self.assertGreater(self._indexed(new)[0], 0)
        self.assertTrue(self.conn.execute(
            "SELECT 1 FROM postings WHERE node_id = ? AND term LIKE 'zebraq%'",
            (new,)).fetchone())

    def test_sweep_deindexes_tombstoned_node(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        nid = bi._live_node_for_path(self.conn, "reference_vault.md")[0]
        self.assertGreater(self._indexed(nid)[0], 0)
        (self.mem / "reference_vault.md").unlink()
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["swept"], 1)
        self.assertEqual(st["deindexed"], 1)
        self.assertEqual(self._indexed(nid), (0, 0))

    def test_literals_are_indexed_too(self):
        st = bi.import_literals(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["indexed"], st["created"])
        self.assertGreater(st["indexed"], 0)

    def test_kill_switch_restores_write_only_behaviour(self):
        os.environ["MEMDAG_BRIDGE_INDEX"] = "0"
        try:
            st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        finally:
            os.environ.pop("MEMDAG_BRIDGE_INDEX", None)
        self.assertEqual((st["indexed"], st["deindexed"]), (0, 0))
        from memsom.storage import schema as memsom_schema
        if memsom_schema.table_exists(self.conn, "postings"):
            self.assertEqual(self.conn.execute(
                "SELECT COUNT(*) FROM postings").fetchone()[0], 0)

    def test_dry_run_indexes_nothing(self):
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=True)
        self.assertEqual((st["indexed"], st["deindexed"]), (0, 0))
