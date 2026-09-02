"""Anti-creep batch: born-unindexed feedback, per-section budgets, the
consolidate proposers, index-stats, and the wire-claude absolute-path fix.

Run:  python -m pytest tests/test_anti_creep.py -q
"""
import io
import json
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.bridge import bridge_import as bi
from memsom.bridge import bridge_render as br
from memsom.bridge import wire_claude as wc
from memsom.distill import digest
from memsom.interface import audit as memsom_audit
from memsom.interface import index_stats
from memsom.bridge import consolidate
from memsom.lifecycle import forget

# Generic fixtures — no author-identifying content (the scrub gate scans tests).
CLUSTER = ("---\nname: Verification cluster\ndescription: verify before asserting\n"
           "type: feedback\nsection: Feedback\n---\n\nVerify things before you "
           "assert them.\n\n- [[feedback_old_check]] — check output before claiming\n")
OLD = ("---\nname: Old check\ndescription: check output before claiming\n"
       "type: feedback\nsection: Feedback\n---\n\nalways check the output\n")
INDEX = """# Memory

## About the User
- [Editor](user_editor.md) — prefers tabs

## Feedback
- [Verification cluster](feedback_cluster_verify.md) — verify before asserting
- [Old check](feedback_old_check.md) — check output before claiming
"""
USER = "---\nname: Editor\ndescription: prefers tabs\ntype: user\n---\nbody\n"


def _fm(content):
    return bi.fm_top_level(bi.split_frontmatter(content)[0])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        os.environ["MEMDAG_DB"] = str(self.root / "t.db")
        # restore (not pop) in tearDown: tests/_isolation.py pins this for the
        # whole process so no later test can sync the real ~/.claude/CLAUDE.md
        self._claude_md_prev = os.environ.get("CLAUDE_MD_PATH")
        os.environ["CLAUDE_MD_PATH"] = str(self.root / "CLAUDE.md")
        self.mem = self.root / "memory"
        self.mem.mkdir()
        (self.mem / "user_editor.md").write_text(USER, encoding="utf-8")
        (self.mem / "feedback_cluster_verify.md").write_text(CLUSTER, encoding="utf-8")
        (self.mem / "feedback_old_check.md").write_text(OLD, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")
        self.conn = memsom.get_connection()
        bi.migrate(self.conn)
        forget.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        if self._claude_md_prev is None:
            os.environ.pop("CLAUDE_MD_PATH", None)
        else:
            os.environ["CLAUDE_MD_PATH"] = self._claude_md_prev
        self.tmp.cleanup()

    def node_fm(self, rel):
        row = bi._live_node_for_path(self.conn, rel)
        return _fm(memsom.get_node(self.conn, row[0])["content"]) if row else None

    def write(self, name, text):
        (self.mem / name).write_text(text, encoding="utf-8")

    def params(self, **over):
        return {**forget.DEFAULTS, **forget.PANEL_PARAM_DEFAULTS, **over}


# --- 1. born-unindexed feedback -----------------------------------------------

class TestBornUnindexed(Base):
    NEW = ("---\nname: New lesson\ndescription: a fresh rule\ntype: feedback\n"
           "section: Feedback\n---\n\nthe rule\n")

    def test_new_feedback_without_why_own_line_is_unfiled_and_marked(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md", self.NEW)
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["born_unindexed"], 1)
        fm = self.node_fm("feedback_new_lesson.md")
        self.assertIsNone(fm.get("section"))
        self.assertEqual(fm.get("index_pending"), "needs_cluster")
        # the render records WHY it is absent
        excluded = []
        text = digest.render_digest(self.conn, excluded_out=excluded)
        self.assertNotIn("feedback_new_lesson.md", text)
        self.assertIn({"stem": "feedback_new_lesson", "reason": "needs_cluster", "rs": None},
                      [{k: e[k] for k in ("stem", "reason", "rs")} for e in excluded])

    def test_hold_persists_across_unchanged_reimport(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md", self.NEW)
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        # second import: the node now EXISTS, but the pending mark keeps it held
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["skipped"], 4)
        self.assertIsNone(self.node_fm("feedback_new_lesson.md").get("section"))

    def test_why_own_line_indexes_normally(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md",
                   self.NEW.replace("section: Feedback\n",
                                    "section: Feedback\nwhy_own_line: fires daily, no cluster fits\n"))
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["born_unindexed"], 0)
        fm = self.node_fm("feedback_new_lesson.md")
        self.assertEqual(fm.get("section"), "Feedback")
        self.assertIsNone(fm.get("index_pending"))
        self.assertIn("feedback_new_lesson.md", digest.render_digest(self.conn))

    def test_adding_why_own_line_later_releases_the_hold(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md", self.NEW)
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md",
                   self.NEW.replace("section: Feedback\n",
                                    "section: Feedback\nwhy_own_line: earned it\n"))
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(self.node_fm("feedback_new_lesson.md").get("section"), "Feedback")

    def test_empty_why_own_line_does_not_count(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md",
                   self.NEW.replace("section: Feedback\n", "section: Feedback\nwhy_own_line:\n"))
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["born_unindexed"], 1)

    def test_cluster_files_are_exempt(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_cluster_new.md",
                   CLUSTER.replace("Verification cluster", "New cluster"))
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["born_unindexed"], 0)
        self.assertEqual(self.node_fm("feedback_cluster_new.md").get("section"), "Feedback")

    def test_curated_index_line_is_exempt(self):
        """A line the user put in MEMORY.md is a deliberate act (and the
        fresh-install path for an existing hand-curated brain): never unfiled."""
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)   # first import
        self.assertEqual(st["born_unindexed"], 0)
        self.assertEqual(self.node_fm("feedback_old_check.md").get("section"), "Feedback")

    def test_existing_stored_nodes_are_not_retroactively_unfiled(self):
        # store the file under the OLD rules (param off), then turn the rule on
        bi.import_memory_dir(self.conn, self.mem, dry_run=False,
                             params=self.params(feedback_born_unindexed=False))
        self.write("feedback_new_lesson.md", self.NEW)
        bi.import_memory_dir(self.conn, self.mem, dry_run=False,
                             params=self.params(feedback_born_unindexed=False))
        self.assertEqual(self.node_fm("feedback_new_lesson.md").get("section"), "Feedback")
        self.write("feedback_new_lesson.md", self.NEW + "edit\n")
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)   # rule on
        self.assertEqual(st["born_unindexed"], 0)
        self.assertEqual(self.node_fm("feedback_new_lesson.md").get("section"), "Feedback")

    def test_param_off_disables_the_rule(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md", self.NEW)
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False,
                                  params=self.params(feedback_born_unindexed=False))
        self.assertEqual(st["born_unindexed"], 0)
        self.assertEqual(self.node_fm("feedback_new_lesson.md").get("section"), "Feedback")

    def test_param_is_read_from_canonical_json(self):
        weights = self.mem / ".weights"
        weights.mkdir()
        (weights / "canonical.json").write_text(json.dumps(
            {"version": 1, "memories": {}, "params": {"feedback_born_unindexed": False}}),
            encoding="utf-8")
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md", self.NEW)
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["born_unindexed"], 0)

    def test_non_feedback_sections_untouched(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_odd.md", self.NEW.replace("section: Feedback", "section: Work"))
        st = bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.assertEqual(st["born_unindexed"], 0)
        self.assertEqual(self.node_fm("feedback_odd.md").get("section"), "Work")

    def test_audit_reports_needs_cluster_as_info_not_orphan(self):
        bi.import_memory_dir(self.conn, self.mem, dry_run=False)
        self.write("feedback_new_lesson.md", self.NEW)
        ok, info = br.bridge_render(self.conn, self.mem, sync_claude=False)["ok"], None
        self.assertTrue(ok)
        findings, _files, _ = memsom_audit.run_audit(self.mem)
        names = {(f["name"], f["target"], f["sev"]) for f in findings}
        self.assertIn(("needs-cluster", "feedback_new_lesson.md", "INFO"), names)
        self.assertNotIn(("orphan-file", "feedback_new_lesson.md", "ERROR"), names)


# --- 2. per-section budget ----------------------------------------------------

def _mk(stem, born, *, rs=1.0, pinned=True, section="Feedback", own_line=False,
        desc="x" * 40):
    return {"kind": "file", "section": section, "stem": stem, "name": stem,
            "desc": desc, "pinned": pinned, "tier": "hot", "rs": rs,
            "channel": "endorsed" if pinned else "user", "stale": False,
            "stale_reason": None, "is_project": False, "is_live_state": False,
            "status": None, "subdir": None, "born": born,
            "is_cluster": stem.startswith("feedback_cluster_"),
            "own_line": own_line, "pending": None}


class TestSectionBudget(unittest.TestCase):
    def test_sheds_pinned_newest_first_and_never_clusters(self):
        hot = [_mk("feedback_cluster_a", "2026-01-01T00:00:00", desc="c" * 200),
               _mk("feedback_oldest", "2026-02-01T00:00:00"),
               _mk("feedback_middle", "2026-03-01T00:00:00"),
               _mk("feedback_newest", "2026-04-01T00:00:00"),
               _mk("user_other", "2026-05-01T00:00:00", section="About the User")]
        full = digest._section_bytes("Feedback", hot[:4])
        # cap so that exactly two plain entries must go
        one_line = full - digest._section_bytes("Feedback", hot[:3])
        cap = full - one_line - 1
        excluded = []
        dropped = digest.shed_section_budgets(hot, {"Feedback": cap}, excluded)
        self.assertEqual([e["stem"] for e in dropped], ["feedback_newest", "feedback_middle"])
        self.assertEqual({e["reason"] for e in excluded}, {"section_budget"})
        self.assertIn("feedback_cluster_a", [e["stem"] for e in hot])
        self.assertIn("user_other", [e["stem"] for e in hot])      # other sections untouched
        self.assertLessEqual(digest._section_bytes(
            "Feedback", [e for e in hot if e["section"] == "Feedback"]), cap)

    def test_cluster_alone_over_budget_is_never_shed(self):
        hot = [_mk("feedback_cluster_a", "2026-01-01T00:00:00", desc="c" * 500)]
        dropped = digest.shed_section_budgets(hot, {"Feedback": 256}, [])
        self.assertEqual(dropped, [])
        self.assertEqual(len(hot), 1)

    def test_own_line_entries_shed_after_plain_ones(self):
        hot = [_mk("feedback_justified", "2026-09-01T00:00:00", own_line=True),
               _mk("feedback_plain_old", "2026-01-01T00:00:00")]
        full = digest._section_bytes("Feedback", hot)
        dropped = digest.shed_section_budgets(hot, {"Feedback": full - 1}, [])
        self.assertEqual([e["stem"] for e in dropped], ["feedback_plain_old"])

    def test_ties_on_born_break_by_rs_asc(self):
        hot = [_mk("feedback_a", "2026-01-01T00:00:00", rs=0.9),
               _mk("feedback_b", "2026-01-01T00:00:00", rs=0.2)]
        full = digest._section_bytes("Feedback", hot)
        dropped = digest.shed_section_budgets(hot, {"Feedback": full - 1}, [])
        self.assertEqual(dropped[0]["stem"], "feedback_b")

    def test_param_validation(self):
        self.assertTrue(forget._param_ok("section_budgets", {"Feedback": 4096}))
        self.assertFalse(forget._param_ok("section_budgets", {"Feedback": 10}))
        self.assertFalse(forget._param_ok("section_budgets", {"Feedback": "big"}))
        self.assertFalse(forget._param_ok("section_budgets", ["Feedback"]))
        self.assertFalse(forget._param_ok("section_budgets", {"": 4096}))
        self.assertTrue(forget._param_ok("feedback_born_unindexed", False))
        self.assertFalse(forget._param_ok("feedback_born_unindexed", 1))
        self.assertEqual(forget.PANEL_PARAM_DEFAULTS["section_budgets"], {"Feedback": 7168})

    def test_load_params_merges_section_budgets(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "canonical.json"
            p.write_text(json.dumps({"params": {"section_budgets": {"Feedback": 2048,
                                                                    "Work": 1024}}}),
                         encoding="utf-8")
            params, warns = forget.load_params(p)
            self.assertEqual(params["section_budgets"], {"Feedback": 2048, "Work": 1024})
            self.assertEqual(warns, [])
            p.write_text(json.dumps({"params": {"section_budgets": {"Feedback": 1}}}),
                         encoding="utf-8")
            params, warns = forget.load_params(p)
            self.assertEqual(params["section_budgets"], {"Feedback": 7168})
            self.assertEqual(len(warns), 1)


class TestSectionBudgetEndToEnd(Base):
    def test_render_sheds_pinned_feedback_and_records_reason(self):
        for i in range(6):
            self.write(f"feedback_rule_{i}.md",
                       f"---\nname: Rule {i}\ndescription: {'d' * 60}\ntype: feedback\n"
                       f"why_own_line: test\nsection: Feedback\n---\nr\n")
        weights = self.mem / ".weights"
        weights.mkdir()
        (weights / "canonical.json").write_text(json.dumps(
            {"version": 1, "memories": {}, "params": {"section_budgets": {"Feedback": 400}}}),
            encoding="utf-8")
        out = br.bridge_render(self.conn, self.mem, sync_claude=False)
        self.assertTrue(out["ok"], out)
        text = (self.mem / "MEMORY.md").read_text(encoding="utf-8")
        self.assertIn("feedback_cluster_verify.md", text)          # never shed
        self.assertIn("user_editor.md", text)
        shed = json.loads((weights / "shed.json").read_text(encoding="utf-8"))
        self.assertGreater(shed["by_reason"].get("section_budget", 0), 0)
        self.assertLessEqual(shed["sections"]["Feedback"]["bytes"], 400)
        self.assertEqual(shed["sections"]["Feedback"]["budget"], 400)
        # the audit treats section_budget as an explained absence, not an orphan
        findings, _f, _ = memsom_audit.run_audit(self.mem)
        self.assertFalse([f for f in findings if f["name"] == "orphan-file"], findings)


# --- 3. consolidate-feedback / consolidate-projects ----------------------------

class TestConsolidateFeedback(Base):
    def _import(self):
        bi.import_all(self.conn, self.mem, dry_run=False)

    def test_dry_run_proposes_nearest_cluster_and_writes_json(self):
        self._import()
        props = consolidate.propose_feedback(self.conn, self.mem, min_age_days=0)
        self.assertEqual([p["stem"] for p in props], ["feedback_old_check"])
        self.assertEqual(props[0]["cluster"], "feedback_cluster_verify")
        self.assertGreater(props[0]["score"], 0)
        path = consolidate._write_proposals(self.mem, "feedback", props)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["feedback"]["proposals"][0]["cluster"], "feedback_cluster_verify")
        # dry-run touched neither file
        self.assertEqual((self.mem / "feedback_old_check.md").read_text(encoding="utf-8"), OLD)
        self.assertEqual((self.mem / "feedback_cluster_verify.md").read_text(encoding="utf-8"),
                         CLUSTER)

    def test_min_age_filters_young_files(self):
        self._import()
        self.assertEqual(consolidate.propose_feedback(self.conn, self.mem, min_age_days=14), [])

    def test_why_own_line_and_unindexed_are_skipped(self):
        self.write("feedback_old_check.md",
                   OLD.replace("section: Feedback\n", "section: Feedback\nwhy_own_line: keep\n"))
        self.write("feedback_gone.md", OLD.replace("section: Feedback", "section: none"))
        self._import()
        self.assertEqual(consolidate.propose_feedback(self.conn, self.mem, min_age_days=0), [])

    def test_apply_absorbs_into_cluster_and_withdraws_file(self):
        self._import()
        props = consolidate.propose_feedback(self.conn, self.mem, min_age_days=0)
        res = consolidate.apply_feedback(self.mem, props)
        self.assertEqual(res["absorbed"], 1)
        ctext = (self.mem / "feedback_cluster_verify.md").read_text(encoding="utf-8")
        # the cluster already linked this stem -> no duplicate bullet, no heading
        self.assertEqual(ctext.count("[[feedback_old_check]]"), 1)
        self.assertNotIn("## Absorbed", ctext)
        ftext = (self.mem / "feedback_old_check.md").read_text(encoding="utf-8")
        self.assertEqual(_fm(ftext).get("section"), "none")
        self.assertTrue((self.mem / "feedback_old_check.md").exists())   # never deleted
        # after re-render the file is out of MEMORY.md, cluster still in
        out = br.bridge_render(self.conn, self.mem, sync_claude=False)
        self.assertTrue(out["ok"], out)
        text = (self.mem / "MEMORY.md").read_text(encoding="utf-8")
        self.assertNotIn("feedback_old_check.md", text)
        self.assertIn("feedback_cluster_verify.md", text)

    def test_apply_appends_new_bullet_when_not_linked(self):
        self.write("feedback_cluster_verify.md",
                   CLUSTER.replace("\n- [[feedback_old_check]] — check output before claiming\n", "\n"))
        self._import()
        props = consolidate.propose_feedback(self.conn, self.mem, min_age_days=0)
        consolidate.apply_feedback(self.mem, props)
        ctext = (self.mem / "feedback_cluster_verify.md").read_text(encoding="utf-8")
        self.assertIn("- [[feedback_old_check]] — check output before claiming", ctext)
        self.assertLess(ctext.index("## Absorbed"), ctext.index("- [[feedback_old_check]]"))

    def test_no_cluster_means_no_match_not_crash(self):
        (self.mem / "feedback_cluster_verify.md").unlink()
        self._import()
        props = consolidate.propose_feedback(self.conn, self.mem, min_age_days=0)
        self.assertEqual(props[0]["cluster"], None)
        res = consolidate.apply_feedback(self.mem, props)
        self.assertEqual(res, {"absorbed": 0, "skipped": 1, "applied": []})

    def test_cli_dry_run_report(self):
        self._import()
        from argparse import Namespace
        buf = io.StringIO()
        with redirect_stdout(buf):
            consolidate._cmd_feedback(Namespace(memory_dir=str(self.mem), apply=False,
                                                min_age_days=0))
        out = buf.getvalue()
        self.assertIn("| feedback_old_check | feedback_cluster_verify |", out)
        self.assertIn("DRY-RUN", out)
        self.assertTrue((self.mem / ".weights" / "consolidate_proposals.json").exists())


class TestConsolidateProjects(Base):
    PARENT = ("---\nname: Widget\ndescription: the widget project\ntype: project\n---\n\n"
              "## Threads\n\n| thread | state | notes |\n|---|---|---|\n"
              "| [[project_widget_api]] | active | rest layer |\n")
    SUB = ("---\nname: Widget UI\ndescription: the old UI thread\ntype: project\n"
           "status: closed\n---\n\ndone\n")

    def _layout(self, parent=None):
        d = self.mem / "projects" / "widget"
        d.mkdir(parents=True)
        (d / "project_widget.md").write_text(parent or self.PARENT, encoding="utf-8")
        (d / "project_widget_api.md").write_text(
            self.SUB.replace("status: closed\n", "").replace("Widget UI", "Widget API"),
            encoding="utf-8")
        (d / "project_widget_ui.md").write_text(self.SUB, encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        return d

    def test_proposes_closed_subproject_only(self):
        self._layout()
        props = consolidate.propose_projects(self.conn, self.mem, min_age_days=0)
        self.assertEqual([(p["stem"], p["parent"], p["why"]) for p in props],
                         [("project_widget_ui", "project_widget", "closed")])
        self.assertEqual(consolidate.propose_projects(self.conn, self.mem, min_age_days=14), [])

    def test_apply_appends_table_row_and_withdraws(self):
        d = self._layout()
        props = consolidate.propose_projects(self.conn, self.mem, min_age_days=0)
        res = consolidate.apply_projects(self.mem, props)
        self.assertEqual(res["withdrawn"], 1)
        ptext = (d / "project_widget.md").read_text(encoding="utf-8")
        rows = [l for l in ptext.splitlines() if l.startswith("| [[project_widget_ui]]")]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].count("|"), 4)                    # 3 columns, like the header
        self.assertIn("closed", rows[0])
        stext = (d / "project_widget_ui.md").read_text(encoding="utf-8")
        self.assertEqual(_fm(stext).get("index"), "false")
        self.assertTrue((d / "project_widget_ui.md").exists())
        # second proposal pass: withdrawn file is no longer a candidate
        bi.import_all(self.conn, self.mem, dry_run=False)
        self.assertEqual(consolidate.propose_projects(self.conn, self.mem, min_age_days=0), [])
        out = br.bridge_render(self.conn, self.mem, sync_claude=False)
        self.assertTrue(out["ok"], out)
        idx = (self.mem / "projects" / "INDEX.md").read_text(encoding="utf-8")
        self.assertNotIn("project_widget_ui.md", idx)
        self.assertIn("project_widget_api.md", idx)

    def test_apply_without_table_uses_closed_threads_list(self):
        d = self._layout(parent="---\nname: Widget\ndescription: d\ntype: project\n---\n\nprose\n")
        props = consolidate.propose_projects(self.conn, self.mem, min_age_days=0)
        consolidate.apply_projects(self.mem, props)
        ptext = (d / "project_widget.md").read_text(encoding="utf-8")
        self.assertIn("## Closed threads\n- [[project_widget_ui]] — closed ", ptext)


# --- 4. visible counts --------------------------------------------------------

class TestIndexStats(Base):
    def test_counts_match_rendered_sections_and_budgets(self):
        out = br.bridge_render(self.conn, self.mem, sync_claude=False)
        self.assertTrue(out["ok"], out)
        st = index_stats.index_stats(self.mem)
        text = (self.mem / "MEMORY.md").read_text(encoding="utf-8")
        self.assertEqual(st["bytes"], len(text.encode("utf-8")))
        fb = st["sections"]["Feedback"]
        self.assertEqual(fb["budget"], 7168)
        self.assertEqual(fb["lines"], 3)          # header + 2 entries (last section)
        block = text[text.index("## Feedback"):]
        self.assertEqual(fb["bytes"], len(block.encode("utf-8")))
        self.assertIsNone(st["sections"]["About the User"]["budget"])
        rendered = index_stats.format_stats(st)
        self.assertIn("Feedback: 3 lines /", rendered)
        self.assertIn("(budget 7168)", rendered)

    def test_bridge_render_prints_feedback_counts(self):
        from argparse import Namespace
        buf = io.StringIO()
        with redirect_stdout(buf):
            br._cmd_bridge_render(Namespace(memory_dir=str(self.mem)))
        self.assertRegex(buf.getvalue(), r"\[bridge\] feedback: \d+ lines / \d+ bytes \(budget 7168\)")

    def test_audit_json_and_report_carry_sections(self):
        br.bridge_render(self.conn, self.mem, sync_claude=False)
        from argparse import Namespace
        buf = io.StringIO()
        with redirect_stdout(buf):
            memsom_audit._cmd_audit(Namespace(memory_dir=str(self.mem), json=True))
        data = json.loads(buf.getvalue())
        self.assertEqual(data["sections"]["Feedback"]["budget"], 7168)
        buf = io.StringIO()
        with redirect_stdout(buf):
            memsom_audit._cmd_audit(Namespace(memory_dir=str(self.mem), json=False))
        self.assertIn("Feedback: 3 lines /", buf.getvalue())

    def test_cli_registered(self):
        from memsom.interface import cli
        import argparse
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd")
        index_stats.register(sub)
        consolidate.register(sub)
        for name in ("index-stats", "consolidate-feedback", "consolidate-projects"):
            self.assertIn(name, sub.choices)
        self.assertTrue(all(hasattr(cli, m) for m in ("memsom_consolidate", "memsom_index_stats")))


# --- 6. wire-claude absolute executable path ----------------------------------

class TestWireAbsolutePath(unittest.TestCase):
    def test_resolve_exe_is_absolute_or_python_m_fallback(self):
        exe = wc.resolve_exe()
        if exe.startswith('"'):
            self.assertIn(" -m memsom.interface.cli", exe)
        else:
            self.assertTrue(Path(exe).is_absolute(), exe)
            self.assertTrue(Path(exe).exists(), exe)
        self.assertFalse(wc.is_bare_command(wc._cmd(exe, "hook-prompt")))

    def test_is_bare_command(self):
        self.assertTrue(wc.is_bare_command('"memsom" hook-prompt'))
        self.assertTrue(wc.is_bare_command("memsom bridge-render"))
        self.assertFalse(wc.is_bare_command('"/opt/venv/bin/memsom" hook-prompt'))
        self.assertFalse(wc.is_bare_command('"C:\\py\\Scripts\\memsom.exe" hook-prompt'))
        self.assertFalse(wc.is_bare_command(None))

    def test_bare_entries_are_upgraded_in_place(self):
        data = {"hooks": {
            "Stop": [{"hooks": [{"type": "command", "command": '"memsom" bridge-render'}]}],
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "memsom hook-prompt",
                                             "timeout": 5}]}]}}
        abs_exe = "/opt/venv/bin/memsom"
        changed = wc.merge_hooks(data, abs_exe)
        self.assertEqual(sorted(changed), ["Stop", "UserPromptSubmit"])
        self.assertEqual(data["hooks"]["Stop"][0]["hooks"][0]["command"],
                         '"/opt/venv/bin/memsom" bridge-render')
        self.assertEqual(data["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"],
                         '"/opt/venv/bin/memsom" hook-prompt')
        self.assertEqual(len(data["hooks"]["Stop"]), 1)                 # no duplicate group
        self.assertEqual(wc.merge_hooks(data, abs_exe), [])             # idempotent

    def test_custom_absolute_stop_entry_is_left_alone(self):
        data = {"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": '"/my/own/memsom" bridge-render --flag'}]}]}}
        self.assertEqual(wc.merge_hooks(data, "/opt/venv/bin/memsom", with_prompt_hook=False), [])
        self.assertEqual(data["hooks"]["Stop"][0]["hooks"][0]["command"],
                         '"/my/own/memsom" bridge-render --flag')

    def test_bare_resolution_never_upgrades_to_bare(self):
        data = {"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": '"memsom" bridge-render'}]}]}}
        self.assertEqual(wc.merge_hooks(data, "memsom", with_prompt_hook=False), [])

    def test_python_m_fallback_is_not_double_quoted(self):
        exe = '"/usr/bin/python3" -m memsom.interface.cli'
        self.assertEqual(wc._cmd(exe, "hook-prompt"),
                         '"/usr/bin/python3" -m memsom.interface.cli hook-prompt')

    def test_wire_settings_upgrades_existing_bare_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            p.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
                {"type": "command", "command": '"memsom" bridge-render'}]}]}}), encoding="utf-8")
            res = wc.wire_settings(p, "/opt/venv/bin/memsom")
            self.assertEqual(res["action"], "merged")
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["hooks"]["Stop"][0]["hooks"][0]["command"],
                             '"/opt/venv/bin/memsom" bridge-render')
            self.assertTrue(p.with_name("settings.json.bak").exists())


if __name__ == "__main__":
    unittest.main()
