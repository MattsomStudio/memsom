"""Tests for memsom_tombstone — the sanctioned flat-memory delete path.

Run:  python -m unittest discover -s . -p test_memsom_tombstone.py
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
from memsom.integrity import tombstone as tomb


FILES = {
    "user_editor.md": "---\nname: Editor\ndescription: prefers tabs\ntype: user\n---\nbody\n",
    "feedback_tests.md": "---\nname: Run tests\ndescription: always run\ntype: feedback\n---\nr\n",
    "project_widget.md": "---\nname: Widget\ndescription: status\ntype: project\n---\ns\n",
    "reference_doc.md": "---\nname: Doc\ndescription: where\ntype: reference\n---\np\n",
}
INDEX = """# Memory

## About the User
- [Editor](user_editor.md) — prefers tabs

## Personal projects
- [Widget](project_widget.md) — status

## References
- [Doc](reference_doc.md) — where

## Feedback
- [Run tests](feedback_tests.md) — always run
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
        self.conn = memsom.get_connection()
        bi.migrate(self.conn)
        forget.migrate(self.conn)
        bi.import_all(self.conn, self.mem, dry_run=False)
        forget.recompute_forget(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def node_state(self, stem):
        return self.conn.execute(
            "SELECT tombstoned FROM nodes WHERE source_ref = ? ORDER BY id DESC LIMIT 1",
            (f"memory:{stem}",)).fetchone()


class TestTombstone(Base):
    def test_revokes_node_and_deletes_file(self):
        res = tomb.tombstone_memory(self.conn, self.mem, "project_widget", reason="obsolete")
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["revoked"], 1)
        self.assertTrue(res["file_deleted"])
        self.assertFalse((self.mem / "project_widget.md").exists())
        self.assertEqual(self.node_state("project_widget")[0], 1)   # tombstoned

    def test_accepts_filename_form(self):
        res = tomb.tombstone_memory(self.conn, self.mem, "reference_doc.md")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["stem"], "reference_doc")

    def test_pinned_refused_without_force(self):
        res = tomb.tombstone_memory(self.conn, self.mem, "user_editor")
        self.assertEqual(res["status"], "refused-pinned")
        self.assertFalse(res["file_deleted"])
        self.assertTrue((self.mem / "user_editor.md").exists())     # untouched
        self.assertEqual(self.node_state("user_editor")[0], 0)      # still live

    def test_pinned_force_overrides(self):
        res = tomb.tombstone_memory(self.conn, self.mem, "feedback_tests", force=True)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["file_deleted"])
        self.assertEqual(self.node_state("feedback_tests")[0], 1)

    def test_list_includes_tombstoned(self):
        tomb.tombstone_memory(self.conn, self.mem, "project_widget", reason="obsolete")
        stems = {s for s, _at, _r in tomb.list_tombstoned(self.conn)}
        self.assertIn("project_widget", stems)

    def test_file_absent_still_revokes_node(self):
        (self.mem / "project_widget.md").unlink()                   # gone from disk
        res = tomb.tombstone_memory(self.conn, self.mem, "project_widget")
        self.assertEqual(res["status"], "ok")
        self.assertFalse(res["file_deleted"])                       # nothing to delete
        self.assertEqual(self.node_state("project_widget")[0], 1)   # node still revoked


    def test_path_traversal_refused(self):
        victim = self.root / "victim.md"          # one level OUTSIDE the memory dir
        victim.write_text("important\n", encoding="utf-8")
        res = tomb.tombstone_memory(self.conn, self.mem, "../victim")
        self.assertEqual(res["status"], "refused-traversal")
        self.assertFalse(res["file_deleted"])
        self.assertTrue(victim.exists())          # not deleted

    def test_traversal_into_dotclaude_refused(self):
        res = tomb.tombstone_memory(self.conn, self.mem, "../../.claude/CLAUDE")
        self.assertEqual(res["status"], "refused-traversal")


class TestHardTombstone(Base):
    """--hard: the sanctioned delete path also ERASES the payload (DB + disk +
    blame), closing the leak where a soft delete/edit leaves old plaintext
    readable through blame. Plain tombstone must stay non-destructive to payload."""

    def _nid(self, stem):
        return self.conn.execute(
            "SELECT id FROM nodes WHERE source_ref=? ORDER BY id DESC LIMIT 1",
            (f"memory:{stem}",)).fetchone()[0]

    def test_hard_erases_payload_db_disk_blame(self):
        from memsom.interface import blame as memsom_blame
        nid = self._nid("project_widget")
        self.assertNotEqual(
            self.conn.execute("SELECT content FROM nodes WHERE id=?", (nid,)).fetchone()[0], "")
        res = tomb.tombstone_memory(self.conn, self.mem, "project_widget",
                                    reason="secret", hard=True)
        self.assertEqual(res["status"], "ok")
        self.assertGreaterEqual(res["erased"], 1)
        self.assertTrue(res["file_deleted"])
        # DB: payload destroyed, node redacted AND revoked
        self.assertEqual(
            self.conn.execute("SELECT content, redacted, tombstoned FROM nodes WHERE id=?",
                              (nid,)).fetchone(), ("", 1, 1))
        # disk: gone
        self.assertFalse((self.mem / "project_widget.md").exists())
        # blame: shows [REDACTED], never the old plaintext
        entries = memsom_blame.blame(self.conn, nid)
        self.assertTrue(any(e["line"] == "[REDACTED]" for e in entries))

    def test_soft_delete_keeps_payload(self):
        # the DEFAULT path revokes only — payload survives, and it never touches the
        # redact machinery (the redacted column isn't even migrated in).
        nid = self._nid("project_widget")
        res = tomb.tombstone_memory(self.conn, self.mem, "project_widget", reason="obsolete")
        self.assertEqual(res.get("erased", 0), 0)
        content = self.conn.execute(
            "SELECT content FROM nodes WHERE id=?", (nid,)).fetchone()[0]
        self.assertNotEqual(content, "")            # revoke, not erase: payload survives

    def test_hard_pinned_refused_without_force(self):
        res = tomb.tombstone_memory(self.conn, self.mem, "user_editor", hard=True)
        self.assertEqual(res["status"], "refused-pinned")
        nid = self._nid("user_editor")
        self.assertNotEqual(                        # pinned payload untouched
            self.conn.execute("SELECT content FROM nodes WHERE id=?", (nid,)).fetchone()[0], "")


class TestCLI(Base):
    def _ns(self, **kw):
        base = dict(stem=None, reason="", force=False, hard=False,
                    memory_dir=str(self.mem))
        base.update(kw)
        return Namespace(**base)

    def test_cli_ok_exit_zero(self):
        # close the shared conn so the CLI opens its own against the same DB file
        self.conn.close()
        rc = tomb._cmd_tombstone(self._ns(stem="project_widget", reason="x"))
        self.assertEqual(rc, 0)
        self.conn = memsom.get_connection()                        # reopen for tearDown

    def test_cli_pinned_exit_two(self):
        self.conn.close()
        rc = tomb._cmd_tombstone(self._ns(stem="user_editor"))
        self.assertEqual(rc, 2)
        self.conn = memsom.get_connection()


if __name__ == "__main__":
    unittest.main()
