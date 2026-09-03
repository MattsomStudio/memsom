"""Tests for memsom.bridge.project — structured project memory (node + sub-notes).

Pure file I/O (no DB): a project node and its sub-notes are files; they land in
the store on the next bridge-import, exactly like fact-set. So these tests need
only a tmp memory dir.

Run:  python -m unittest discover -s . -p test_memsom_project.py
"""

import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

from memsom.bridge import project as P
from memsom.kernel.frontmatter import split_frontmatter
from memsom.distill import digest


def _sec(node_path, h2):
    body = split_frontmatter(node_path.read_text(encoding="utf-8"))[1]
    return P._sections(body).get(h2, [])


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = Path(self.tmp.name) / "memory"
        self.mem.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def node(self, slug="demo"):
        return P._node_path(self.mem, slug)


class TestScaffold(Base):
    def test_init_creates_seven_files_idempotent(self):
        out = P.init_project(self.mem, "demo")
        self.assertEqual(len(out), 7)
        self.assertTrue(all(v == "created" for v in out.values()))
        d = P._proj_dir(self.mem, "demo")
        self.assertEqual(sorted(p.name for p in d.glob("*.md")), sorted([
            "project_demo.md", "project_demo_spec.md", "project_demo_gotchas.md",
            "project_demo_decisions.md", "project_demo_interface_io.md",
            "project_demo_architecture.md", "project_demo_tests.md"]))
        # idempotent: second run overwrites nothing
        out2 = P.init_project(self.mem, "demo")
        self.assertTrue(all(v == "present" for v in out2.values()))

    def test_fresh_scaffold_checks_clean(self):
        P.init_project(self.mem, "demo")
        self.assertEqual(P.check(self.mem, "demo"), [])


class TestFeatures(Base):
    def setUp(self):
        super().setUp()
        P.init_project(self.mem, "demo")

    def test_feature_writes_node_index_and_note_together(self):
        P.set_feature(self.mem, "demo", "f-one", name="Feature One", status="planned")
        # node line
        feats = P._parse_features(_sec(self.node(), "Features"))
        self.assertEqual([f["id"] for f in feats], ["f-one"])
        # spec index line
        idx = P._note_path(self.mem, "demo", "spec")
        self.assertIn("f-one", "\n".join(_sec(idx, "Features")))
        # feature note exists with the six sections
        fp = P._feature_path(self.mem, "demo", "f-one")
        self.assertTrue(fp.exists())
        secs = P._sections(split_frontmatter(fp.read_text(encoding="utf-8"))[1])
        for s in P.FEATURE_SECTIONS:
            self.assertIn(s, secs)

    def test_implemented_refuses_without_evidence(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        with self.assertRaises(P.ProjectError):
            P.set_feature(self.mem, "demo", "f-one", status="implemented")

    def test_status_change_writes_changes_line(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        P.set_feature(self.mem, "demo", "f-one", status="implemented",
                      evidence="(MEASURED) works 2026-09-02")
        fp = P._feature_path(self.mem, "demo", "f-one")
        changes = P._sections(split_frontmatter(fp.read_text(encoding="utf-8"))[1])["Changes"]
        self.assertTrue(any("implemented" in ln for ln in changes))

    def test_spec_set_edits_one_section_and_logs_why(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        P.set_spec(self.mem, "demo", "f-one", section="behaviour",
                   value="takes X, returns Y", why="clarified the contract")
        fp = P._feature_path(self.mem, "demo", "f-one")
        secs = P._sections(split_frontmatter(fp.read_text(encoding="utf-8"))[1])
        self.assertIn("takes X, returns Y", "\n".join(secs["Behaviour"]))
        self.assertTrue(any("clarified the contract" in ln for ln in secs["Changes"]))


class TestChecks(Base):
    def setUp(self):
        super().setUp()
        P.init_project(self.mem, "demo")

    def _names(self, slug="demo"):
        return [f["name"] for f in P.check(self.mem, slug)]

    def test_feature_without_spec_note_is_red(self):
        # add a node feature line but delete the spec note
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        P._feature_path(self.mem, "demo", "f-one").unlink()
        self.assertIn("project-schema", self._names())

    def test_spec_note_without_node_line_is_red(self):
        (P._proj_dir(self.mem, "demo") / "project_demo_spec_ghost.md").write_text(
            "---\nname: project_demo_spec_ghost\nkind: project-ref\n---\n"
            "## Purpose\n## Behaviour\n## Interfaces\n## Acceptance\n## Status\nplanned\n## Changes\n",
            encoding="utf-8")
        self.assertIn("project-schema", self._names())

    def test_status_mismatch_across_three_is_red(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        # tamper the feature note's Status to disagree with the node line
        fp = P._feature_path(self.mem, "demo", "f-one")
        fp.write_text(fp.read_text(encoding="utf-8").replace(
            "## Status\nplanned", "## Status\narchived"), encoding="utf-8")
        self.assertIn("project-schema", self._names())

    def test_left_naming_implemented_feature_is_red(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="implemented",
                      evidence="(MEASURED) done 2026-09-02")
        P.set_status(self.mem, "demo", left="finish f-one")
        self.assertIn("project-schema", self._names())

    def test_features_section_is_exempt_from_the_cap(self):
        # 70 feature lines blow the old 80-line / 6000-byte whole-body cap, but
        # ## Features is exempt (it is the source of truth for what's left and is
        # never injected in full) — so no cap finding fires.
        for i in range(70):
            P.set_feature(self.mem, "demo", f"f-{i:02d}", name=f"Feature {i}",
                          status="planned")
        msgs = [f["msg"] for f in P.check(self.mem, "demo")]
        self.assertFalse(any("cap" in m for m in msgs),
                         f"## Features must be exempt from the cap: {msgs}")

    def test_bloated_nonfeature_prose_trips_the_cap(self):
        # the tight 60-line / 4000-byte budget still applies to everything that is
        # NOT the Features list — bloat ## What and the cap fires.
        np = self.node()
        text = np.read_text(encoding="utf-8")
        text = text.replace("## What\n", "## What\n" + ("x " * 2500) + "\n", 1)
        np.write_text(text, encoding="utf-8")
        msgs = [f["msg"] for f in P.check(self.mem, "demo")]
        self.assertTrue(any("excl. Features" in m for m in msgs),
                        f"bloated non-feature prose must trip the cap: {msgs}")

    def test_active_decision_not_in_needs_matt_is_red(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="active-decision",
                      decision="D-20260902-01")
        # active-decision must be echoed under Needs Matt
        self.assertIn("project-schema", self._names())
        # once listed, the finding clears
        P.set_status(self.mem, "demo", ask="decide f-one")
        self.assertNotIn("project-schema", self._names())

    def test_stale_spec_is_red(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="implemented",
                      evidence="(MEASURED) 2026-09-01")
        # freeze the feature note's Changes to an old date
        fp = P._feature_path(self.mem, "demo", "f-one")
        txt = fp.read_text(encoding="utf-8")
        import re as _re
        txt = _re.sub(r"\d{4}-\d{2}-\d{2}", "2026-08-01", txt)
        fp.write_text(txt, encoding="utf-8")
        # a later decisions entry names the feature id
        P.log_entry(self.mem, "demo", "decisions", "**rework f-one contract**",
                    why="new requirement")
        self.assertIn("project-schema", self._names())

    def test_feature_note_missing_section_is_red(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        fp = P._feature_path(self.mem, "demo", "f-one")
        fp.write_text(fp.read_text(encoding="utf-8").replace("## Acceptance\n", ""),
                      encoding="utf-8")
        self.assertIn("project-schema", self._names())

    def test_feature_note_over_fence_is_red(self):
        P.set_feature(self.mem, "demo", "f-one", name="F", status="planned")
        fp = P._feature_path(self.mem, "demo", "f-one")
        fp.write_text(fp.read_text(encoding="utf-8") + "\n" + "\n".join(
            f"line {i}" for i in range(P.FEATURE_FENCE + 5)), encoding="utf-8")
        self.assertIn("project-schema", self._names())

    def test_missing_h2_section_is_red(self):
        # the gate inversion: drop ## Rules & gates
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "## Rules & gates\n\n", ""), encoding="utf-8")
        self.assertIn("project-schema", self._names())

    def test_creds_value_is_red(self):
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "## Creds\n", "## Creds\n- password: hunter2isquitelong\n"), encoding="utf-8")
        self.assertIn("project-creds-value", self._names())

    def test_alias_clash_is_red(self):
        P.init_project(self.mem, "beta")
        for slug in ("demo", "beta"):
            n = P._node_path(self.mem, slug)
            n.write_text(n.read_text(encoding="utf-8").replace(
                "status: active", "status: active\naliases: shared, %s-only" % slug),
                encoding="utf-8")
        names = [f["name"] for f in P.check(self.mem)]
        self.assertIn("project-alias-clash", names)

    def test_loose_file_is_warn(self):
        (P._proj_dir(self.mem, "demo") / "project_demo_scratch.md").write_text(
            "---\nname: project_demo_scratch\n---\nloose\n", encoding="utf-8")
        got = [(f["name"], f["sev"]) for f in P.check(self.mem, "demo")]
        self.assertIn(("project-loose-file", "WARN"), got)

    def test_nested_frontmatter_is_warn(self):
        P.init_project(self.mem, "nested")
        n = P._node_path(self.mem, "nested")
        n.write_text("---\nmetadata:\n  name: project_nested\n  kind: project-node\n---\n"
                     "## What\n", encoding="utf-8")
        names = [f["name"] for f in P.check(self.mem, "nested")]
        self.assertIn("project-nested-frontmatter", names)


class TestLog(Base):
    def setUp(self):
        super().setUp()
        P.init_project(self.mem, "demo")

    def test_log_appends_deterministic_dated_id(self):
        r = P.log_entry(self.mem, "demo", "gotchas", "**boom**", cause="x", fix="y",
                        source="session:2026-09-02")
        self.assertEqual(r["status"], "added")
        self.assertRegex(r["id"], r"^G-\d{8}-01$")
        # a second, different gotcha same day increments the sequence
        r2 = P.log_entry(self.mem, "demo", "gotchas", "**other**", cause="z", fix="w")
        self.assertTrue(r2["id"].endswith("-02"))

    def test_repeat_decision_reaffirms(self):
        P.log_entry(self.mem, "demo", "decisions", "**go with X**", why="simpler")
        r = P.log_entry(self.mem, "demo", "decisions", "**go with X**", why="again")
        self.assertEqual(r["status"], "reaffirmed")
        note = P._note_path(self.mem, "demo", "decisions").read_text(encoding="utf-8")
        self.assertIn("reaffirmed:", note)

    def test_cap_refuses_with_code_two(self):
        orig = dict(P.LOG_CAPS)
        P.LOG_CAPS["gotchas"] = 3
        try:
            with self.assertRaises(P.ProjectError) as cm:
                for i in range(6):
                    P.log_entry(self.mem, "demo", "gotchas", f"**g{i}**", cause="c", fix="f")
            self.assertEqual(cm.exception.code, 2)
        finally:
            P.LOG_CAPS.clear()
            P.LOG_CAPS.update(orig)

    def test_status_targets_the_right_h3(self):
        P.set_status(self.mem, "demo", done="shipped the thing")
        done = P._h3(_sec(self.node(), "Status")).get("Done", [])
        self.assertTrue(any("shipped the thing" in ln for ln in done))


class TestDigestNotesLine(Base):
    """The digest renders a project-node group as ONE notes: line and hides the
    fixed sub-notes; a legacy group (no node) is unchanged."""

    def _entries(self, kinds):
        # minimal entry dicts shaped like digest._entry output
        out = []
        for stem, node_kind in kinds:
            out.append({"kind": "file", "stem": stem, "name": stem, "desc": "",
                        "rs": 1.0, "stale": False, "status": "active",
                        "tier": "hot", "node_kind": node_kind,
                        "subdir": "projects/demo"})
        return out

    def test_structured_group_renders_one_notes_line(self):
        ents = self._entries([
            ("project_demo", "project-node"),
            ("project_demo_spec", None), ("project_demo_gotchas", None),
            ("project_demo_decisions", None), ("project_demo_interface_io", None),
            ("project_demo_architecture", None), ("project_demo_tests", None),
            ("project_demo_spec_f-one", None)])
        txt = digest.render_projects_index(ents, title="# Projects")
        self.assertEqual(txt.count("\n  notes: "), 1)
        # the six sub-notes and the per-feature spec note are NOT subproject lines
        self.assertNotIn("  - [project_demo_gotchas]", txt)
        self.assertNotIn("project_demo_spec_f-one", txt)

    def test_legacy_group_unchanged(self):
        ents = self._entries([("project_demo", None),
                              ("project_demo_sub", None)])
        txt = digest.render_projects_index(ents, title="# Projects")
        self.assertNotIn("notes:", txt)
        self.assertIn("project_demo_sub", txt)


class TestReorg(Base):
    """`project reorg` — the deterministic /reorgmem half: check() findings tagged
    content, the mechanical fixes (sync-conflict merge/delete + count refresh),
    and the headless --sweep pending file."""

    def setUp(self):
        super().setUp()
        P.init_project(self.mem, "demo")
        # a real index_hook so the placeholder INFO doesn't cloud these tests
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "index_hook: (project node — set the Status headline)",
            "index_hook: real hook"), encoding="utf-8")

    def _conflict(self, suffix, text):
        d = P._proj_dir(self.mem, "demo")
        p = d / f"project_demo_{suffix}.sync-conflict-20260101-120000-ABCDEF.md"
        p.write_text(text, encoding="utf-8")
        return p

    def test_check_findings_are_tagged_content(self):
        # a broken node link is a schema-independent content finding
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "## Pointers", "## Pointers\n- [[no_such_memory]]"), encoding="utf-8")
        r = P.reorg(self.mem, slug="demo")
        self.assertTrue(all(f.get("fix") == "content" for f in r["content"]))
        self.assertEqual({f["name"] for f in r["content"]}, {"reorg-link-broken"})

    def test_apply_deletes_identical_conflict_copy(self):
        g = P._note_path(self.mem, "demo", "gotchas")
        c = self._conflict("gotchas", g.read_text(encoding="utf-8"))
        r = P.reorg(self.mem, slug="demo", apply=True)
        self.assertFalse(c.exists())
        self.assertTrue(any(a["fix"] == "sync-conflict" and a["result"] == "deleted"
                            for a in r["applied"]))

    def test_apply_union_merges_divergent_log_conflict(self):
        P.log_entry(self.mem, "demo", "gotchas", "**boom**", cause="c", fix="f")
        g = P._note_path(self.mem, "demo", "gotchas")
        # a conflict copy carrying a DIFFERENT dated entry id
        extra = g.read_text(encoding="utf-8").replace(
            "## Entries\n", "## Entries\n- G-20200101-01 (2020-01-01) **older thing**\n")
        c = self._conflict("gotchas", extra)
        r = P.reorg(self.mem, slug="demo", apply=True)
        merged = g.read_text(encoding="utf-8")
        self.assertIn("G-20200101-01", merged)        # from the conflict copy
        self.assertIn("**boom**", merged)             # from canonical
        self.assertFalse(c.exists())
        self.assertTrue(any(a.get("result") == "merged" for a in r["applied"]))

    def test_apply_keeps_divergent_nonlog_conflict(self):
        # a conflict copy of the NODE that differs cannot be auto-merged
        n = self.node()
        c = self._conflict("", n.read_text(encoding="utf-8") + "\n- extra divergence\n")
        # name it as a node conflict, not a subnote
        node_conflict = c.parent / "project_demo.sync-conflict-20260101-120000-ABCDEF.md"
        c.rename(node_conflict)
        P.reorg(self.mem, slug="demo", apply=True)
        self.assertTrue(node_conflict.exists())       # kept for a human

    def test_apply_refreshes_subnote_counts(self):
        P.log_entry(self.mem, "demo", "gotchas", "**one**", cause="c", fix="f")
        P.log_entry(self.mem, "demo", "gotchas", "**two**", cause="c", fix="f")
        P.reorg(self.mem, slug="demo", apply=True)
        body = split_frontmatter(self.node().read_text(encoding="utf-8"))[1]
        line = next(l for l in P._sections(body)["Sub-notes"]
                    if "project_demo_gotchas" in l)
        self.assertTrue(line.rstrip().endswith("— 2"), line)

    def test_sweep_writes_pending_and_log(self):
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "## Pointers", "## Pointers\n- [[no_such_memory]]"), encoding="utf-8")
        r = P.reorg(self.mem, slug="demo", sweep=True)
        import json
        pending = json.loads(Path(r["pending_path"]).read_text(encoding="utf-8"))
        self.assertEqual(pending["version"], 1)
        self.assertIn("demo", pending["projects"])
        self.assertEqual({f["name"] for f in pending["projects"]["demo"]["findings"]},
                         {"reorg-link-broken"})
        log = (self.mem / ".weights" / "reorgmem_log.jsonl").read_text(encoding="utf-8")
        self.assertIn('"mode": "sweep"', log)

    def test_sweep_touches_no_content_file(self):
        # counts already fresh (apply once) → a second sweep with only a content
        # finding must not rewrite any project file (RULES: mechanical-only headless)
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "## Pointers", "## Pointers\n- [[no_such_memory]]"), encoding="utf-8")
        P.reorg(self.mem, slug="demo", apply=True)     # settle counts
        import hashlib
        d = P._proj_dir(self.mem, "demo")
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in d.glob("*.md")}
        P.reorg(self.mem, slug="demo", sweep=True)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in d.glob("*.md")}
        self.assertEqual(before, after)


class TestCache(Base):
    """P2 auto-load: the project_aliases.json the prompt hook reads."""

    def test_write_cache_shape_and_block(self):
        P.init_project(self.mem, "demo", aliases="alpha, beta gamma")
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8")
                     .replace("### Done", "### Done\n- shipped it")
                     .replace("## Creds\n", "## Creds\n- pw in [[reference_x]]\n")
                     .replace("## Rules & gates\n", "## Rules & gates\n- r1\n- r2\n- r3\n"),
                     encoding="utf-8")
        P.set_feature(self.mem, "demo", "f-one", name="One", status="planned")
        out = P.write_cache(self.mem, project_bytes=1024, max_n=2)
        self.assertEqual(out["projects"], 1)
        cache = P.load_cache(self.mem)
        e = cache["projects"]["demo"]
        self.assertEqual(e["aliases"], ["alpha", "beta gamma"])
        self.assertIn("## Status", e["block"])
        self.assertIn("## Creds", e["block"])
        # only the first two Rules lines are carried
        self.assertIn("- r1", e["block"])
        self.assertIn("- r2", e["block"])
        self.assertNotIn("- r3", e["block"])
        self.assertEqual(e["features"], "1 planned")

    def test_block_truncated_on_line_boundary_to_byte_cap(self):
        P.init_project(self.mem, "demo")
        n = self.node()
        big = "\n".join(f"- done item {i} " + "x" * 40 for i in range(60))
        n.write_text(n.read_text(encoding="utf-8").replace("### Done", "### Done\n" + big),
                     encoding="utf-8")
        P.write_cache(self.mem, project_bytes=300, max_n=2)
        block = P.load_cache(self.mem)["projects"]["demo"]["block"]
        self.assertLessEqual(len(block.encode("utf-8")), 300)
        for ln in block.split("\n"):        # whole lines only, never a half
            self.assertFalse(ln.endswith("x" * 39 + "x") and len(ln) > 300)

    def test_fact_refs_resolved_when_resolver_given(self):
        P.init_project(self.mem, "demo")
        n = self.node()
        n.write_text(n.read_text(encoding="utf-8").replace(
            "### Done", "### Done\n- runs at [[fact_speed]]"), encoding="utf-8")
        body = split_frontmatter(n.read_text(encoding="utf-8"))[1]
        block = P.build_inject_block(body, resolve=lambda t: t.replace("[[fact_speed]]", "9ms"))
        self.assertIn("9ms", block)
        self.assertNotIn("[[fact_speed]]", block)

    def test_clashing_alias_dropped_from_both(self):
        P.init_project(self.mem, "demo", aliases="shared, demo-only")
        P.init_project(self.mem, "beta", aliases="shared, beta-only")
        P.write_cache(self.mem, project_bytes=1024, max_n=2)
        cache = P.load_cache(self.mem)
        self.assertNotIn("shared", cache["projects"]["demo"]["aliases"])
        self.assertNotIn("shared", cache["projects"]["beta"]["aliases"])
        self.assertIn("demo-only", cache["projects"]["demo"]["aliases"])

    def test_load_cache_failopen_on_corrupt(self):
        (self.mem / ".weights").mkdir(parents=True)
        (self.mem / ".weights" / "project_aliases.json").write_text("{ not json",
                                                                    encoding="utf-8")
        self.assertIsNone(P.load_cache(self.mem))


if __name__ == "__main__":
    unittest.main()
