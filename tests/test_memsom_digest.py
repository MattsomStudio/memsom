"""Tests for memsom_digest — render MEMORY.md from memsom (Phase 3).

Run:  python -m unittest discover -s . -p test_memsom_digest.py
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.bridge import bridge_import as bi
from memsom.lifecycle import forget as forget
from memsom.distill import digest as digest
from memsom.lifecycle import stale as memsom_stale


FILES = {
    "user_adhd.md": "---\nname: ADHD\ndescription: has ADHD\ntype: user\n---\nbody\n",
    "feedback_debug.md": "---\nname: Debug loop\ndescription: use the loop\ntype: feedback\n---\nr\n",
    "reference_kali.md": "---\nname: Kali VM\ndescription: status\ntype: reference\n---\ns\n",
    "reference_vault.md": "---\nname: Vault\ndescription: where\ntype: reference\n---\np\n",
}
INDEX = """# Memory - Alex

## About the User
- **Alex** — goal: cybersecurity
- [ADHD](user_adhd.md) — has ADHD

## Current Setup & Learning
- [Kali VM](reference_kali.md) — status

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
        # the bridge now indexes every imported node; keep the tests off the
        # network (an Ollama embed is ~1s/node) -> BM25-only for the suite
        self._embed_prev = os.environ.get("MEMDAG_EMBED_BACKEND")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
        self.mem = self.root / "memory"
        self.mem.mkdir()
        for n, t in FILES.items():
            (self.mem / n).write_text(t, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")
        self.conn = memsom.get_connection()
        bi.migrate(self.conn)
        forget.migrate(self.conn)
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        if self._embed_prev is None:
            os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        else:
            os.environ["MEMDAG_EMBED_BACKEND"] = self._embed_prev
        self.tmp.cleanup()

    def demote(self, stem):
        self.conn.execute(
            "UPDATE nodes SET forget_tier = 'cold' WHERE source_ref = ?",
            (f"memory:{stem}",))
        self.conn.commit()


class TestRender(Base):
    def test_has_title_and_sections(self):
        out = digest.render_digest(self.conn)
        self.assertTrue(out.startswith("# Memory"))  # generic default title
        self.assertIn("## About the User", out)
        self.assertIn("## Feedback", out)

    def test_title_overridable_via_env(self):
        os.environ["MEMDAG_DIGEST_TITLE"] = "# Memory - Test User"
        try:
            out = digest.render_digest(self.conn)
            self.assertTrue(out.startswith("# Memory - Test User"))
        finally:
            os.environ.pop("MEMDAG_DIGEST_TITLE", None)

    def test_file_links_and_literals_rendered(self):
        out = digest.render_digest(self.conn)
        self.assertIn("- [ADHD](user_adhd.md) — has ADHD", out)
        self.assertIn("- **Alex** — goal: cybersecurity", out)  # literal verbatim

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
        self.demote("reference_kali")
        out = digest.render_digest(self.conn)
        self.assertNotIn("reference_kali.md", out)
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


class TestValidate(Base):
    def test_valid_store_passes(self):
        self.assertEqual(digest.validate(self.conn), [])

    def test_over_budget_blocks(self):
        problems = digest.validate(self.conn, budget=10)
        self.assertTrue(problems)
        self.assertEqual(problems[0]["kind"], "export-boundary")


class TestWriteLive(Base):
    def test_writes_real_when_valid(self):
        target = self.root / "out"
        target.mkdir()
        ok, info = digest.write_live(self.conn, target)
        self.assertTrue(ok)
        self.assertTrue((target / "MEMORY.md").exists())
        self.assertIn("## About the User", (target / "MEMORY.md").read_text(encoding="utf-8"))

    def test_failsafe_leaves_existing_file_when_invalid(self):
        target = self.root / "out"
        target.mkdir()
        good = "# existing good brain\n"
        (target / "MEMORY.md").write_text(good, encoding="utf-8")
        ok, problems = digest.write_live(self.conn, target, budget=10)  # forces failure
        self.assertFalse(ok)
        self.assertTrue(problems)
        # the existing file is untouched (fail-safe, not fail-open)
        self.assertEqual((target / "MEMORY.md").read_text(encoding="utf-8"), good)


class TestBudget(Base):
    def test_drops_lowest_rs_user_first_under_tight_budget(self):
        # set distinct RS so the drop order is deterministic
        self.conn.execute("UPDATE nodes SET forget_rs = 0.9 WHERE source_ref = 'memory:reference_kali'")
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


class TestStaleRender(Base):
    """Phase-2 render: inline ⚠ marker + the synthetic Needs Reverification block."""

    def _mark(self, stem, reason="unverified since 2026-05"):
        nid = self.conn.execute(
            "SELECT id FROM nodes WHERE source_ref = ?", (f"memory:{stem}",)
        ).fetchone()[0]
        memsom_stale.mark_stale_cascade(self.conn, nid, reason)
        return nid

    def test_nothing_stale_is_byte_identical(self):
        # Phase-1 no-op guard: with no stale flags the render is unchanged from the
        # pre-feature behaviour (no markers, no Needs Reverification section).
        out = digest.render_digest(self.conn)
        self.assertNotIn("Needs Reverification", out)
        self.assertNotIn("⚠", out)
        # and it still matches the source index exactly
        self.assertEqual(digest.compare_index(INDEX, out), {})

    def test_stale_marker_inline(self):
        self._mark("reference_kali", "unverified since 2026-04")
        out = digest.render_digest(self.conn)
        # the inline body line carries a BARE glyph (reason lives in the section)
        line = next(ln for ln in out.splitlines()
                    if "reference_kali.md" in ln and "⚠" in ln)
        self.assertIn("⚠", line)
        self.assertNotIn("unverified since", line)        # reason NOT inline (cheap)
        self.assertIn("unverified since 2026-04", out)     # reason IS in the section

    def test_needs_reverification_section_first(self):
        self._mark("reference_kali")
        out = digest.render_digest(self.conn)
        self.assertIn("## Needs Reverification", out)
        # it is the FIRST section under the H1
        self.assertLess(out.index("## Needs Reverification"),
                        out.index("## About the User"))

    def test_compare_index_ignores_reverify_section(self):
        # the synthetic section carries no real file entries, so the GO criterion
        # (per-section file-set equivalence) is unaffected by staleness
        self._mark("reference_kali")
        out = digest.render_digest(self.conn)
        self.assertEqual(digest.compare_index(INDEX, out), {})

    def test_reverify_section_dropped_first_under_budget(self):
        self._mark("reference_kali")
        full = digest.render_digest(self.conn)
        self.assertIn("## Needs Reverification", full)
        # a budget just under full forces the worklist section to shed FIRST,
        # while the inline marker on the note itself is retained
        tight = len(full.encode("utf-8")) - 5
        out = digest.render_digest(self.conn, budget=tight)
        self.assertNotIn("## Needs Reverification", out)
        self.assertIn("⚠", out)                       # inline marker still present


class TestLineCap(Base):
    """The shed loop must satisfy the LINE cap too (the consumer's ~200-line read)."""

    def _set_rs(self, stem, rs):
        self.conn.execute("UPDATE nodes SET forget_rs = ? WHERE source_ref = ?",
                          (rs, f"memory:{stem}"))
        self.conn.commit()

    def test_line_cap_sheds_with_reason_lines(self):
        self._set_rs("reference_kali", 0.9)
        self._set_rs("reference_vault", 0.1)
        full = digest.render_digest(self.conn)
        nlines = full.count("\n")
        excluded = []
        out = digest.render_digest(self.conn, max_lines=nlines - 1,
                                   excluded_out=excluded)
        self.assertLessEqual(out.count("\n"), nlines - 1)
        self.assertNotIn("reference_vault.md", out)   # lowest RS shed first
        self.assertIn("reference_kali.md", out)
        self.assertEqual([(e["stem"], e["reason"]) for e in excluded],
                         [("reference_vault", "lines")])

    def test_byte_cap_wins_the_label_when_both_are_over(self):
        self._set_rs("reference_vault", 0.1)
        full = digest.render_digest(self.conn)
        excluded = []
        digest.render_digest(self.conn, budget=len(full.encode("utf-8")) - 5,
                             max_lines=full.count("\n") - 1, excluded_out=excluded)
        self.assertEqual(excluded[0]["reason"], "budget")

    def test_raises_when_pinned_exceed_line_cap(self):
        with self.assertRaises(digest.DigestTooLarge):
            digest.render_digest(self.conn, max_lines=2)

    def test_validate_reports_line_cap(self):
        problems = digest.validate(self.conn, max_lines=2)
        self.assertTrue(problems)
        self.assertEqual(problems[0]["kind"], "export-boundary")

    def test_resolve_max_lines_from_canonical(self):
        import json
        (self.mem / ".weights").mkdir()
        (self.mem / ".weights" / "canonical.json").write_text(
            json.dumps({"params": {"memory_max_lines": 77}}), encoding="utf-8")
        self.assertEqual(digest.resolve_max_lines(self.mem), 77)
        self.assertEqual(digest.resolve_max_lines(self.root / "nowhere"), digest.MAX_LINES)


class TestLiveStatePartition(Base):
    """Live state (## Live state / type: fact) is shed LAST among droppables."""

    def setUp(self):
        super().setUp()
        (self.mem / "reference_gpu.md").write_text(
            "---\nname: GPU driver\ndescription: 580.x\ntype: reference\n"
            "section: Live state\n---\nx\n", encoding="utf-8")
        (self.mem / "fact_tps.md").write_text(
            "---\nname: tps\ndescription: tokens per second\ntype: fact\n"
            "value: 42\nsection: References\n---\nx\n", encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        # live-state entries get the LOWEST RS: naive RS order would drop them first
        for stem, rs in (("reference_gpu", 0.01), ("fact_tps", 0.02),
                         ("reference_vault", 0.5), ("reference_kali", 0.9)):
            self.conn.execute("UPDATE nodes SET forget_rs = ? WHERE source_ref = ?",
                              (rs, f"memory:{stem}"))
        self.conn.commit()

    def test_cold_fact_in_live_state_still_renders(self):
        # BUG (2026-08-20): fact_* files are channel `user`, so a cold tier
        # dropped them from "## Live state" as `cold` before the shed-last
        # ordering ever ran. Live state is exempt from TIER, not from budget.
        (self.mem / "fact_gpu_vram.md").write_text(
            "---\nname: gpu vram\ndescription: vram\ntype: fact\nvalue: 12\n"
            "unit: GB\nsection: Live state\n---\nx\n", encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        self.conn.execute("UPDATE nodes SET forget_rs = 0.0 WHERE source_ref = ?",
                          ("memory:fact_gpu_vram",))
        self.conn.commit()
        for stem in ("fact_gpu_vram", "fact_tps", "reference_gpu"):
            self.demote(stem)
        excluded = []
        out = digest.render_digest(self.conn, excluded_out=excluded)
        self.assertIn("fact_gpu_vram.md", out)      # cold fact, Live state section
        self.assertIn("fact_tps.md", out)           # cold fact, other section
        self.assertIn("reference_gpu.md", out)      # cold non-fact, Live state section
        self.assertEqual([e for e in excluded if e["reason"] == "cold"], [])

    def test_cold_non_fact_reference_outside_live_state_is_dropped(self):
        self.demote("reference_vault")              # plain reference, ## References
        excluded = []
        out = digest.render_digest(self.conn, excluded_out=excluded)
        self.assertNotIn("reference_vault.md", out)
        self.assertIn({"stem": "reference_vault", "reason": "cold", "rs": 0.5}, excluded)

    def test_live_state_section_renders_after_hardware_slot(self):
        out = digest.render_digest(self.conn)
        self.assertIn("## Live state", out)
        self.assertLess(out.index("## Live state"), out.index("## References"))

    def test_other_droppables_go_before_live_state(self):
        full = digest.render_digest(self.conn)
        excluded = []
        # cap tight enough to force exactly two drops
        two_lines_less = full.count("\n") - 2
        out = digest.render_digest(self.conn, max_lines=two_lines_less,
                                   excluded_out=excluded)
        dropped = [e["stem"] for e in excluded if e["reason"] in ("budget", "lines")]
        self.assertEqual(dropped[:2], ["reference_vault", "reference_kali"])
        self.assertIn("reference_gpu.md", out)
        self.assertIn("fact_tps.md", out)

    def test_live_state_is_still_droppable_when_nothing_else_is_left(self):
        # the smallest line cap that still renders = pinned + literals only
        for cap in range(3, 200):
            excluded = []
            try:
                out = digest.render_digest(self.conn, max_lines=cap,
                                           excluded_out=excluded)
                break
            except digest.DigestTooLarge:
                continue
        dropped = [e["stem"] for e in excluded if e["reason"] == "lines"]
        # all four user files gone, non-live first, then live-state by RS
        self.assertEqual(dropped, ["reference_vault", "reference_kali",
                                   "reference_gpu", "fact_tps"])
        self.assertIn("user_adhd.md", out)            # pinned survives


class TestProjectsSplit(Base):
    """project_ stems leave MEMORY.md for projects/INDEX.md."""

    def setUp(self):
        super().setUp()
        sub = self.mem / "projects"
        sub.mkdir()
        (sub / "project_widget.md").write_text(
            "---\nname: Widget\ndescription: building it\ntype: project\n"
            "section: Personal projects\n---\nx\n", encoding="utf-8")
        (sub / "project_old.md").write_text(
            "---\nname: Old thing\ndescription: shipped\ntype: project\n"
            "status: Closed\nsection: Personal projects\n---\nx\n", encoding="utf-8")
        (sub / "project_nosection.md").write_text(
            "---\nname: Unsectioned\ndescription: still indexed\ntype: project\n"
            "status: parked\n---\nx\n", encoding="utf-8")
        # a legacy flat project memory, cold -> Closed by tier
        (self.mem / "project_legacy.md").write_text(
            "---\nname: Legacy\ndescription: flat file\ntype: project\n"
            "section: Work\n---\nx\n", encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        self.demote("project_legacy")
        self.conn.execute("UPDATE nodes SET forget_rs = 0.9 WHERE source_ref = 'memory:project_widget'")
        self.conn.commit()

    def test_projects_excluded_from_main_digest_with_pointer(self):
        excluded = []
        out = digest.render_digest(self.conn, excluded_out=excluded)
        for stem in ("project_widget", "project_old", "project_nosection", "project_legacy"):
            self.assertNotIn(f"{stem}.md", out)
        self.assertIn("## Personal projects\n" + digest.PROJECTS_POINTER_LINE, out)
        self.assertEqual(out.count(digest.PROJECTS_POINTER_LINE), 1)
        reasons = {e["stem"]: e["reason"] for e in excluded}
        self.assertEqual(reasons["project_widget"], "projects")
        self.assertEqual(reasons["project_legacy"], "projects")

    def test_pointer_not_duplicated_after_reimport_of_rendered_index(self):
        # the importer mirrors the rendered pointer back as a literal node; the
        # next render must still carry exactly one copy
        out = digest.render_digest(self.conn)
        (self.mem / "MEMORY.md").write_text(out, encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        out2 = digest.render_digest(self.conn)
        self.assertEqual(out2.count(digest.PROJECTS_POINTER_LINE), 1)

    def test_no_pointer_when_no_projects(self):
        for p in list((self.mem / "projects").glob("*.md")) + [self.mem / "project_legacy.md"]:
            p.unlink()
        bi.import_all(self.conn, self.mem, dry_run=False)
        out = digest.render_digest(self.conn)
        self.assertNotIn("projects/INDEX.md", out)

    def test_standalone_projects_tagged_and_ordered(self):
        self.conn.execute("UPDATE nodes SET forget_rs = 0.2 WHERE source_ref = 'memory:project_legacy'")
        self.conn.commit()
        text = digest.render_projects_index(digest.project_entries(self.conn))
        self.assertTrue(text.startswith("# Projects\n"))
        self.assertIn("## Standalone", text)
        body = text[text.index("## Standalone"):].splitlines()[1:]
        # Active first (untagged), then Parked, then Closed; links relative to projects/
        self.assertEqual(body, [
            "- [Widget](project_widget.md) — building it",
            "- [Unsectioned](project_nosection.md) — still indexed [Parked]",
            "- [Old thing](project_old.md) — shipped [Closed]",
            "- [Legacy](../project_legacy.md) — flat file [Closed]",
        ])

    def test_projects_index_sorted_by_rs_desc_within_group(self):
        (self.mem / "projects" / "project_second.md").write_text(
            "---\nname: Second\ndescription: also active\ntype: project\n---\nx\n",
            encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        self.conn.execute("UPDATE nodes SET forget_rs = 0.3 WHERE source_ref = 'memory:project_second'")
        self.conn.execute("UPDATE nodes SET forget_rs = 0.9 WHERE source_ref = 'memory:project_widget'")
        self.conn.commit()
        text = digest.render_projects_index(digest.project_entries(self.conn))
        self.assertLess(text.index("project_widget.md"), text.index("project_second.md"))


class TestProjectsHierarchy(Base):
    """projects/<slug>/ dirs render as one group: parent headline + nested subs."""

    def _proj(self, slug, stem, name, desc, **fm):
        d = self.mem / "projects" / slug
        d.mkdir(parents=True, exist_ok=True)
        extra = "".join(f"{k}: {v}\n" for k, v in fm.items())
        (d / f"{stem}.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\ntype: project\n{extra}---\nx\n",
            encoding="utf-8")

    def setUp(self):
        super().setUp()
        self._proj("acme", "project_acme", "Acme", "the parent")
        self._proj("acme", "project_acme_api", "Acme API", "rest layer")
        self._proj("acme", "project_acme_ui", "Acme UI", "frontend", status="closed")
        self._proj("acme", "project_acme_db", "Acme DB", "schema", status="Parked")
        # a parked parent whose subs inherit unless they say otherwise
        self._proj("zed", "project_zed", "Zed", "shelved", status="parked")
        self._proj("zed", "project_zed_one", "Zed one", "inherits parked")
        self._proj("zed", "project_zed_two", "Zed two", "explicit wins", status="active")
        # a dir with no parent overview
        self._proj("orphan", "project_orphan_bit", "Orphan bit", "no parent here")
        (self.mem / "projects" / "project_loose.md").write_text(
            "---\nname: Loose\ndescription: standalone\ntype: project\n---\nx\n",
            encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        for stem, rs in (("project_acme_api", 0.9), ("project_acme_db", 0.95),
                         ("project_acme_ui", 0.99), ("project_acme", 0.8)):
            self.conn.execute("UPDATE nodes SET forget_rs = ? WHERE source_ref = ?",
                              (rs, f"memory:{stem}"))
        self.conn.commit()
        self.text = digest.render_projects_index(digest.project_entries(self.conn))

    def test_group_headline_is_parent_line_with_nested_subs(self):
        t = self.text
        i = t.index("### [Acme](acme/project_acme.md) — the parent")
        block = t[i:].split("\n\n", 1)[0].splitlines()
        # Active sub first (untagged) even though it has the lowest RS of the three,
        # then Parked, then Closed — links relative to projects/
        self.assertEqual(block[1:], [
            "  - [Acme API](acme/project_acme_api.md) — rest layer",
            "  - [Acme DB](acme/project_acme_db.md) — schema [Parked]",
            "  - [Acme UI](acme/project_acme_ui.md) — frontend [Closed]",
        ])

    def test_missing_parent_headline_visible(self):
        self.assertIn("### orphan (no parent overview)\n"
                      "  - [Orphan bit](orphan/project_orphan_bit.md) — no parent here",
                      self.text)

    def test_status_inherits_from_parent_unless_explicit(self):
        t = self.text
        self.assertIn("### [Zed](zed/project_zed.md) — shelved [Parked]", t)
        self.assertIn("  - [Zed two](zed/project_zed_two.md) — explicit wins\n", t)   # active
        self.assertIn("  - [Zed one](zed/project_zed_one.md) — inherits parked [Parked]", t)
        self.assertLess(t.index("Zed two"), t.index("Zed one"))   # Active first

    def test_standalone_after_groups_and_group_order(self):
        t = self.text
        self.assertLess(t.index("### [Acme]"), t.index("### [Zed]"))   # Active parent first
        self.assertLess(t.rindex("###"), t.index("## Standalone"))
        self.assertIn("## Standalone\n- [Loose](project_loose.md) — standalone", t)

    def test_pointer_literal_text(self):
        out = digest.render_digest(self.conn)
        self.assertIn("- Project memory lives in projects/ — read projects/INDEX.md "
                      "(one group per project, subprojects nested, Active/Parked/Closed) "
                      "when a task touches ongoing work.", out)


if __name__ == "__main__":
    unittest.main()
