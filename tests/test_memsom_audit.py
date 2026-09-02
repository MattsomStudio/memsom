"""Tests for memsom_audit — structural integrity audit of the flat memory store.

Run:  python -m unittest discover -s . -p test_memsom_audit.py
"""

import os
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.bridge import bridge_import as bi
from memsom.lifecycle import forget as forget
from memsom.interface import audit as audit


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
        self._embed_prev = os.environ.get("MEMDAG_EMBED_BACKEND")
        os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"   # no network in tests
        self.mem = self.root / "memory"
        self.mem.mkdir()
        for n, t in FILES.items():
            (self.mem / n).write_text(t, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")
        # import + run the forgetting pass so the store has live nodes with tiers
        self.conn = memsom.get_connection()
        bi.migrate(self.conn)
        forget.migrate(self.conn)
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)
        self.conn.close()

    def tearDown(self):
        os.environ.pop("MEMDAG_DB", None)
        if self._embed_prev is None:
            os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        else:
            os.environ["MEMDAG_EMBED_BACKEND"] = self._embed_prev
        self.tmp.cleanup()

    def names(self, findings):
        return {f["name"] for f in findings}

    def audit(self):
        findings, _files, _ = audit.run_audit(self.mem)
        return findings


class TestClean(Base):
    def test_clean_store_has_no_errors(self):
        findings = self.audit()
        errs = [f for f in findings if f["sev"] == "ERROR"]
        self.assertEqual(errs, [], f"expected clean, got {errs}")


class TestProjectFindingsMerged(Base):
    def test_project_check_findings_appear_in_audit(self):
        from memsom.bridge import project as P
        P.init_project(self.mem, "demo")
        n = P._node_path(self.mem, "demo")
        n.write_text(n.read_text(encoding="utf-8").replace(
            "## Creds\n", "## Creds\n- token=abcd1234efgh5678zz (inline)\n"),
            encoding="utf-8")
        names = self.names(self.audit())
        self.assertIn("project-creds-value", names)


class TestChecks(Base):
    def test_dead_index_link(self):
        (self.mem / "user_editor.md").unlink()             # linked but now missing
        names = self.names(self.audit())
        self.assertIn("dead-index-link", names)

    def test_orphan_live_but_unindexed(self):
        # a live, imported file dropped from the index (and not cold) = real orphan
        idx = INDEX.replace("- [Widget](project_widget.md) — status\n", "")
        (self.mem / "MEMORY.md").write_text(idx, encoding="utf-8")
        findings = self.audit()
        orphans = [f for f in findings if f["name"] == "orphan-file"]
        self.assertTrue(any(f["target"] == "project_widget.md" for f in orphans))
        self.assertTrue(all(f["sev"] == "ERROR" for f in orphans))

    def test_withdrawn_file_is_info_not_orphan(self):
        # `section: none` in the file = withdrawn by design; the render records it
        # as unsectioned in shed.json; neither must surface as an orphan ERROR
        import json
        (self.mem / "feedback_tests.md").write_text(
            "---\nname: Run tests\ndescription: always run tests\ntype: feedback\n"
            "section: none\n---\nr\n", encoding="utf-8")
        idx = INDEX.replace("- [Run tests](feedback_tests.md) — always run tests\n", "")
        (self.mem / "MEMORY.md").write_text(idx, encoding="utf-8")
        conn = memsom.get_connection()
        bi.import_all(conn, self.mem, dry_run=False)
        conn.close()
        findings = self.audit()
        self.assertFalse([f for f in findings if f["name"] == "orphan-file"
                          and f["target"] == "feedback_tests.md"])
        w = [f for f in findings if f["name"] == "withdrawn"]
        self.assertEqual([(f["sev"], f["target"]) for f in w], [("INFO", "feedback_tests.md")])
        # the shed.json receipt alone (no frontmatter marker) is enough too
        (self.mem / "feedback_tests.md").write_text(FILES["feedback_tests.md"], encoding="utf-8")
        conn = memsom.get_connection()
        bi.import_all(conn, self.mem, dry_run=False)
        conn.close()
        (self.mem / ".weights").mkdir(exist_ok=True)
        (self.mem / ".weights" / "shed.json").write_text(json.dumps(
            {"excluded": [{"stem": "feedback_tests", "reason": "unsectioned", "rs": 1.0}]}),
            encoding="utf-8")
        findings = self.audit()
        self.assertEqual({f["target"] for f in findings if f["name"] == "withdrawn"},
                         {"feedback_tests.md"})
        self.assertFalse([f for f in findings if f["name"] == "orphan-file"])

    def test_pending_import_is_info_not_error(self):
        # a brand-new file not yet imported: INFO, never an error
        (self.mem / "reference_new.md").write_text(
            "---\nname: New\ndescription: d\ntype: reference\n---\nx\n", encoding="utf-8")
        findings = self.audit()
        pend = [f for f in findings if f["name"] == "pending-import"]
        self.assertTrue(any(f["target"] == "reference_new.md" for f in pend))
        self.assertEqual([f for f in findings if f["sev"] == "ERROR"], [])

    def test_frontmatter_missing(self):
        (self.mem / "user_broken.md").write_text("---\nname: x\n---\nno type/desc\n",
                                                  encoding="utf-8")
        names = self.names(self.audit())
        self.assertIn("frontmatter-missing", names)

    def test_bad_type(self):
        (self.mem / "project_widget.md").write_text(
            "---\nname: W\ndescription: d\ntype: bogus\n---\ns\n", encoding="utf-8")
        names = self.names(self.audit())
        self.assertIn("bad-type", names)

    def test_broken_wikilink_is_info(self):
        (self.mem / "user_editor.md").write_text(
            "---\nname: Editor\ndescription: d\ntype: user\n---\nsee [[nonexistent_thing]]\n",
            encoding="utf-8")
        findings = self.audit()
        bw = [f for f in findings if f["name"] == "broken-wikilink"]
        self.assertTrue(bw)
        self.assertTrue(all(f["sev"] == "INFO" for f in bw))

    def test_budget_breach(self):
        big = "# Memory\n\n## About the User\n" + ("- [x](user_editor.md) — " + "z" * 200 + "\n") * 120
        (self.mem / "MEMORY.md").write_text(big, encoding="utf-8")
        names = self.names(self.audit())
        self.assertIn("budget-breach", names)


    def test_uppercase_broken_wikilink_flagged(self):
        (self.mem / "user_editor.md").write_text(
            "---\nname: Editor\ndescription: d\ntype: user\n---\nsee [[Nonexistent]]\n",
            encoding="utf-8")
        self.assertIn("broken-wikilink", self.names(self.audit()))

    def test_fact_type_not_flagged_bad_type(self):
        (self.mem / "fact_tps.md").write_text(
            "---\nname: tps\ndescription: d\ntype: fact\nvalue: 42\n---\nx\n",
            encoding="utf-8")
        bad = [f for f in self.audit() if f["name"] == "bad-type"]
        self.assertEqual(bad, [])

    def test_personal_type_not_flagged_bad_type(self):
        (self.mem / "personal_note.md").write_text(
            "---\nname: Note\ndescription: d\ntype: personal\n---\nx\n", encoding="utf-8")
        bad = [f for f in self.audit() if f["name"] == "bad-type"]
        self.assertEqual(bad, [])


class TestDBUnavailableDegrades(Base):
    def test_orphan_downgrades_to_warn_without_store(self):
        idx = INDEX.replace("- [Widget](project_widget.md) — status\n", "")
        (self.mem / "MEMORY.md").write_text(idx, encoding="utf-8")
        # force the store unreachable -> _store_tiers returns db_ok=False
        orig = memsom.get_connection
        memsom.get_connection = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no db"))
        try:
            findings, _f, (_t, db_ok) = audit.run_audit(self.mem)
        finally:
            memsom.get_connection = orig
        self.assertFalse(db_ok)
        orphans = [f for f in findings if f["name"] == "orphan-file"]
        self.assertTrue(orphans and all(f["sev"] == "WARN" for f in orphans))


class TestCLI(Base):
    def test_json_exit_code_clean(self):
        rc = audit._cmd_audit(Namespace(memory_dir=str(self.mem), json=True))
        self.assertEqual(rc, 0)

    def test_exit_code_nonzero_on_error(self):
        (self.mem / "user_editor.md").unlink()             # dead-index-link = ERROR
        rc = audit._cmd_audit(Namespace(memory_dir=str(self.mem), json=True))
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
