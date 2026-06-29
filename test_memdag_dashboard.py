"""Tests for memdag_dashboard — the memory telemetry dashboard.

Run:  python -m unittest discover -s . -p test_memdag_dashboard.py
"""

import os
import sys
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import scrub_gate                       # noqa: E402
import memdag                           # noqa: E402
import memdag_bridge_import as bi       # noqa: E402
import memdag_forget as forget          # noqa: E402
import memdag_dashboard as dash         # noqa: E402


FILES = {
    "user_editor.md": "---\nname: Editor\ndescription: prefers tabs\ntype: user\n---\nbody\n",
    "feedback_tests.md": "---\nname: Run tests\ndescription: always run\ntype: feedback\n---\nr\n",
    "project_widget.md": "---\nname: Widget\ndescription: status\ntype: project\n---\nsee [[user_editor]]\n",
}
INDEX = """# Memory

## About the User
- [Editor](user_editor.md) — prefers tabs

## Personal projects
- [Widget](project_widget.md) — status

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
        os.environ["MEMDAG_BRIDGE_MEMORY_DIR"] = str(self.mem)
        for n, t in FILES.items():
            (self.mem / n).write_text(t, encoding="utf-8")
        (self.mem / "MEMORY.md").write_text(INDEX, encoding="utf-8")
        conn = memdag.get_connection()
        bi.migrate(conn)
        forget.migrate(conn)
        bi.import_all(conn, self.mem, dry_run=False)
        forget.recompute_forget(conn)
        conn.close()
        # the optional episodic sessions DB exists on a dev machine — stub it to a
        # nonexistent path so the session card is deterministically absent.
        self._orig_sessions = dash._sessions_db
        dash._sessions_db = lambda: self.root / "no_sessions.db"

    def tearDown(self):
        dash._sessions_db = self._orig_sessions
        for k in ("MEMDAG_DB", "MEMDAG_BRIDGE_MEMORY_DIR"):
            os.environ.pop(k, None)
        self.tmp.cleanup()


class TestTelemetry(Base):
    def test_shape(self):
        t = dash.build_telemetry()
        self.assertEqual(t["totals"]["total"], 3)
        self.assertEqual(t["totals"]["hot"] + t["totals"]["cold"], 3)
        self.assertIn("user", t["types"])
        self.assertEqual(len(t["scatter"]), 3)
        self.assertIsNotNone(t["budget"])              # MEMORY.md exists
        self.assertTrue(t["graph"]["nodes"])           # sections + memories
        self.assertIsNone(t["sessions"])               # stubbed absent

    def test_wikilink_becomes_graph_link(self):
        t = dash.build_telemetry()
        link_kinds = {l["kind"] for l in t["graph"]["links"]}
        self.assertIn("link", link_kinds)              # [[user_editor]] cross-link


class TestRender(Base):
    def test_html_self_contained_and_identity_free(self):
        t = dash.build_telemetry()
        out = self.root / "dash.html"
        dash.render(t, out)
        html = out.read_text(encoding="utf-8")
        self.assertIn("<title>Memory Telemetry</title>", html)
        self.assertIn("user_editor", html)             # data embedded
        self.assertEqual(scrub_gate.scan_text(html), [], "dashboard HTML leaks identity")


class TestSecurity(Base):
    def test_malicious_section_cannot_break_out_of_script(self):
        t = dash.build_telemetry()
        payload = "</script><img src=x onerror=alert(1)>"
        t["graph"]["sections"].append(payload)
        out = self.root / "x.html"
        dash.render(t, out)
        html = out.read_text(encoding="utf-8")
        self.assertNotIn("</script><img", html)        # raw breakout absent
        self.assertIn("<" + chr(92) + "/script>", html)  # escaped form present

    def test_body_pin_line_does_not_mark_pinned(self):
        # only frontmatter pin counts; a body line "pin: yes" must not (bug_012)
        (self.mem / "project_pinbody.md").write_text(
            "---\nname: P\ndescription: d\ntype: project\n---\nnotes:\npin: yes please\n",
            encoding="utf-8")
        import memdag, memdag_bridge_import as bi2, memdag_forget as f2
        conn = memdag.get_connection()
        bi2.import_all(conn, self.mem, dry_run=False)
        f2.recompute_forget(conn)
        conn.close()
        rows = {r["stem"]: r for r in dash.load_weights()}
        self.assertEqual(rows["project_pinbody"]["pinned"], 0)


class TestCLI(Base):
    def test_no_open_writes_file(self):
        out = self.root / "out.html"
        rc = dash._cmd_dashboard(Namespace(out=str(out), no_open=True))
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())


class TestMissingDB(unittest.TestCase):
    def test_load_weights_raises_when_db_absent(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["MEMDAG_DB"] = str(Path(d) / "nope.db")
            try:
                with self.assertRaises(SystemExit):
                    dash.load_weights()
            finally:
                os.environ.pop("MEMDAG_DB", None)


if __name__ == "__main__":
    unittest.main()
