"""Tests for memsom_verify_stale — verification-age staleness.

Run:  python -m unittest discover -s . -p test_memsom_verify_stale.py
"""

import os
import tempfile
import unittest
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_stale
import memsom_bridge_import as bi
import memsom_verify_stale as vs


NOW = datetime(2026, 6, 25, tzinfo=timezone.utc)


def _ns(dt):
    """obsidian_mtime string for a given datetime."""
    return f"{int(dt.timestamp() * 1e9)}:100"


# --- pure assess() logic (no DB) ---------------------------------------------

class TestAssess(unittest.TestCase):
    def test_state_bearing_old_marks(self):
        old = _ns(NOW - timedelta(days=60))
        stale, reason = vs.assess("status: NOT deployed yet", old, NOW, 21)
        self.assertTrue(stale)
        self.assertTrue(reason.startswith("unverified since"))

    def test_state_bearing_recent_fresh(self):
        recent = _ns(NOW - timedelta(days=3))
        stale, _ = vs.assess("feature NOT deployed", recent, NOW, 21)
        self.assertFalse(stale)        # state-bearing but recent -> fresh

    def test_not_state_bearing_never_stale(self):
        old = _ns(NOW - timedelta(days=2000))
        stale, _ = vs.assess("the user prefers tabs over spaces", old, NOW, 21)
        self.assertFalse(stale)        # age alone never flags a stable fact

    def test_due_past_immediate(self):
        recent = _ns(NOW - timedelta(days=1))   # mtime recent on purpose
        stale, reason = vs.assess("progress check DUE 2026-01-01", recent, NOW, 21)
        self.assertTrue(stale)
        self.assertIn("overdue", reason)
        self.assertIn("2026-01-01", reason)

    def test_due_future_not_stale(self):
        recent = _ns(NOW - timedelta(days=1))
        stale, _ = vs.assess("ship DUE 2099-01-01", recent, NOW, 21)
        self.assertFalse(stale)

    def test_last_verified_overrides_mtime(self):
        old = _ns(NOW - timedelta(days=900))
        content = "---\nlast-verified: 2026-06-20\n---\nthing NOT deployed"
        stale, _ = vs.assess(content, old, NOW, 21)
        self.assertFalse(stale)        # last-verified (5d ago) wins over old mtime

    def test_garbage_mtime_not_age_stale(self):
        stale, _ = vs.assess("NOT applied", "garbage", NOW, 21)
        self.assertFalse(stale)        # can't prove age -> don't flag
        # but a past DUE still flags even with garbage mtime
        stale2, _ = vs.assess("NOT applied DUE 2020-01-01", "garbage", NOW, 21)
        self.assertTrue(stale2)

    def test_threshold_zero_disables(self):
        old = _ns(NOW - timedelta(days=900))
        stale, _ = vs.assess("NOT deployed", old, NOW, 0)
        self.assertFalse(stale)


# --- reconciler against a DB --------------------------------------------------

FILES = {
    # state-bearing + (mtime forced old in setUp) -> should mark
    "project_site.md": "---\nname: Site\ndescription: site\ntype: project\n---\nmattsomstudio.ca NOT deployed\n",
    # stable fact -> never marks
    "user_adhd.md": "---\nname: ADHD\ndescription: adhd\ntype: user\n---\nhas ADHD\n",
    # past DUE -> marks regardless of mtime
    "project_due.md": "---\nname: Due\ndescription: due\ntype: project\n---\ncheck DUE 2026-01-01 passed\n",
}
INDEX = """# Memory - Test

## Current Setup & Learning
- [Site](project_site.md) — site
- [Due](project_due.md) — due

## About the User
- [ADHD](user_adhd.md) — adhd
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
        bi.import_all(self.conn, self.mem, dry_run=False)
        # force project_site's mtime old so age-staleness fires deterministically
        self._set_mtime("project_site", NOW - timedelta(days=60))
        self._set_mtime("user_adhd", NOW - timedelta(days=60))

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def _set_mtime(self, stem, dt):
        self.conn.execute("UPDATE nodes SET obsidian_mtime = ? WHERE source_ref = ?",
                          (_ns(dt), f"memory:{stem}"))
        self.conn.commit()

    def _stale_stems(self):
        rows = self.conn.execute(
            "SELECT source_ref FROM nodes WHERE stale = 1 AND tombstoned = 0").fetchall()
        return {r[0].split(":", 1)[1] for r in rows}


class TestReconcile(Base):
    def test_marks_state_bearing_and_due(self):
        res = vs.recompute_verify_stale(self.conn, now=NOW)
        self.assertEqual(self._stale_stems(), {"project_site", "project_due"})
        self.assertNotIn("user_adhd", self._stale_stems())   # stable fact untouched

    def test_stale_stems_are_bare(self):
        res = vs.recompute_verify_stale(self.conn, now=NOW)
        self.assertIn("project_site", res["stale_stems"])
        self.assertNotIn("memory:project_site", res["stale_stems"])

    def test_idempotent(self):
        vs.recompute_verify_stale(self.conn, now=NOW)
        res2 = vs.recompute_verify_stale(self.conn, now=NOW)
        self.assertEqual(res2["marked"], [])                 # second run marks nothing

    def test_kill_switch_clears_owned(self):
        vs.recompute_verify_stale(self.conn, now=NOW)
        self.assertTrue(self._stale_stems())
        res = vs.recompute_verify_stale(self.conn, now=NOW, threshold_days=0)
        self.assertEqual(self._stale_stems(), set())         # disabled -> cleared

    def test_does_not_clear_supersession_stale(self):
        # a NON-owned stale flag (simulated supersession cascade) on a node that
        # verify would otherwise leave fresh (user_adhd is not state-bearing)
        nid = self.conn.execute(
            "SELECT id FROM nodes WHERE source_ref = 'memory:user_adhd'").fetchone()[0]
        memsom_stale.mark_stale_cascade(self.conn, nid, "source superseded by node 999")
        vs.recompute_verify_stale(self.conn, now=NOW)
        row = self.conn.execute(
            "SELECT stale, stale_reason FROM nodes WHERE id = ?", (nid,)).fetchone()
        self.assertEqual(row[0], 1)                           # still stale
        self.assertIn("superseded", row[1])                   # reason untouched

    def test_reversible_on_edit(self):
        vs.recompute_verify_stale(self.conn, now=NOW)
        self.assertIn("project_site", self._stale_stems())
        # edit the file to drop the state-bearing phrase, re-import (tombstone+new)
        (self.mem / "project_site.md").write_text(
            "---\nname: Site\ndescription: site\ntype: project\n---\nmattsomstudio.ca is LIVE\n",
            encoding="utf-8")
        bi.import_all(self.conn, self.mem, dry_run=False)
        vs.recompute_verify_stale(self.conn, now=NOW)
        self.assertNotIn("project_site", self._stale_stems())  # fresh node, no flag


if __name__ == "__main__":
    unittest.main()
