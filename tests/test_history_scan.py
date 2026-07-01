#!/usr/bin/env python3
"""Tests for history_scan — the pre-push gate that scans the diffs of commits being
pushed, not just the final tree.

Run:  python -m unittest -v test_history_scan
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
import history_scan  # noqa: E402

# Assembled at runtime so this test file itself stays clean of the literal token
# (otherwise the tree scrub_gate would flag its own test). Same trick as test_scrub.
USERNAME = "Fur" + "io"


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


def _rev(cwd, ref="HEAD"):
    return subprocess.run(["git", "rev-parse", ref], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


class _TempRepo(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _git("init", "-q", "-b", "main", cwd=self.repo)
        _git("config", "user.email", "t@t.test", cwd=self.repo)
        _git("config", "user.name", "Tester", cwd=self.repo)
        _git("config", "commit.gpgsign", "false", cwd=self.repo)
        # history_scan relies on cwd (the pre-push hook runs at repo root).
        os.chdir(self.repo)

    def tearDown(self):
        os.chdir(self._cwd)
        self.tmp.cleanup()

    def _commit(self, name, body):
        (self.repo / name).write_text(body, encoding="utf-8")
        _git("add", name, cwd=self.repo)
        _git("commit", "-q", "-m", f"add {name}", cwd=self.repo)
        return _rev(self.repo)


class TestHistoryScan(_TempRepo):
    def test_clean_range_passes(self):
        base = self._commit("a.md", "nothing private here\n")
        tip = self._commit("b.md", "still clean\n")
        self.assertEqual(history_scan.scan_push([(tip, base)]), [])

    def test_planted_leak_is_caught(self):
        base = self._commit("a.md", "clean\n")
        tip = self._commit("leak.md", f"path is C:/Users/{USERNAME}/memsom\n")
        findings = history_scan.scan_push([(tip, base)])
        self.assertTrue(findings, "leak added in a pushed commit was not caught")
        self.assertTrue(any(tok == USERNAME.lower() for _, tok, _, _ in findings))

    def test_leak_added_then_removed_still_caught(self):
        # THE gap: leak lives in commit1, gone by the tip tree. Tree gate would pass;
        # history gate must still bite because the commit ships in the push.
        base = self._commit("a.md", "clean\n")
        self._commit("leak.md", f"user {USERNAME} lives here\n")
        _git("rm", "-q", "leak.md", cwd=self.repo)
        _git("commit", "-q", "-m", "remove leak", cwd=self.repo)
        tip = _rev(self.repo)
        # Tree at tip is clean...
        self.assertFalse((self.repo / "leak.md").exists())
        # ...but the push (base..tip) still carries the leak in history.
        findings = history_scan.scan_push([(tip, base)])
        self.assertTrue(findings, "add-then-remove leak escaped the history scan")
        self.assertTrue(any(tok == USERNAME.lower() for _, tok, _, _ in findings))

    def test_new_branch_scans_reachable(self):
        # remote oid all-zero => new branch; everything reachable should be scanned.
        self._commit("a.md", "clean\n")
        tip = self._commit("leak.md", f"{USERNAME}\n")
        zero = "0" * 40
        findings = history_scan.scan_push([(tip, zero)])
        self.assertTrue(any(tok == USERNAME.lower() for _, tok, _, _ in findings))

    def test_scan_all_catches_leak_anywhere_in_history(self):
        # CI's mode: scan every commit on every ref, no range needed. A leak buried
        # mid-history (tree since cleaned) must still surface.
        self._commit("a.md", "clean\n")
        self._commit("leak.md", f"{USERNAME} was here\n")
        _git("rm", "-q", "leak.md", cwd=self.repo)
        _git("commit", "-q", "-m", "remove leak", cwd=self.repo)
        findings = history_scan.scan_all()
        self.assertTrue(any(tok == USERNAME.lower() for _, tok, _, _ in findings))

    def test_scan_all_clean_history_passes(self):
        self._commit("a.md", "clean\n")
        self._commit("b.md", "also clean\n")
        self.assertEqual(history_scan.scan_all(), [])

    def test_deletion_is_noop(self):
        self._commit("a.md", "clean\n")
        zero = "0" * 40
        # local oid all-zero => branch deletion, nothing published to scan.
        self.assertEqual(history_scan.scan_push([(zero, _rev(self.repo))]), [])


if __name__ == "__main__":
    unittest.main()
