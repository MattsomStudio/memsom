"""Tests for memsom_wire_claude — install the Claude Code memory loop safely.

Run:  python -m unittest discover -s . -p test_wire_claude.py
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

from memsom.bridge import wire_claude as wc

EXE = "/abs/path/memsom"


def _make_skills_src(root):
    src = Path(root) / "claude" / "skills"
    for name in ("saveall", "audit"):
        d = src / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return src


class TestHookMerge(unittest.TestCase):
    def test_merge_adds_stop_hook(self):
        data = {}
        changed = wc.merge_hooks(data, EXE)
        self.assertEqual(changed, ["Stop"])
        self.assertTrue(wc._has_command(data["hooks"]["Stop"], "bridge-render"))

    def test_merge_is_idempotent(self):
        data = {}
        wc.merge_hooks(data, EXE)
        self.assertEqual(wc.merge_hooks(data, EXE), [])      # second run: no change

    def test_merge_preserves_existing_hooks(self):
        data = {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other thing"}]}]}}
        wc.merge_hooks(data, EXE)
        cmds = [h["command"] for g in data["hooks"]["Stop"] for h in g["hooks"]]
        self.assertIn("other thing", cmds)                   # user's hook kept
        self.assertTrue(any("bridge-render" in c for c in cmds))

    def test_gate_opt_in_adds_pre_post(self):
        data = {}
        changed = wc.merge_hooks(data, EXE, with_gate=True)
        self.assertIn("PreToolUse", changed)
        self.assertIn("PostToolUse", changed)
        self.assertTrue(wc._has_command(data["hooks"]["PreToolUse"], "hook-pre"))

    def test_malformed_hooks_raises(self):
        with self.assertRaises(ValueError):
            wc.merge_hooks({"hooks": "not-an-object"}, EXE)


class TestSettingsIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "settings.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_created_when_absent(self):
        res = wc.wire_settings(self.path, EXE)
        self.assertEqual(res["action"], "created")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(wc._has_command(data["hooks"]["Stop"], "bridge-render"))

    def test_merged_and_backed_up(self):
        self.path.write_text(json.dumps(
            {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "x"}]}]}}),
            encoding="utf-8")
        res = wc.wire_settings(self.path, EXE)
        self.assertEqual(res["action"], "merged")
        self.assertTrue(self.path.with_name("settings.json.bak").exists())
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("SessionStart", data["hooks"])         # preserved
        self.assertIn("Stop", data["hooks"])

    def test_unchanged_on_rerun(self):
        wc.wire_settings(self.path, EXE)
        res = wc.wire_settings(self.path, EXE)
        self.assertEqual(res["action"], "unchanged")

    def test_malformed_refuses_to_write(self):
        self.path.write_text("{not json", encoding="utf-8")
        res = wc.wire_settings(self.path, EXE)
        self.assertEqual(res["action"], "malformed")
        self.assertIn("snippet", res)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{not json")  # untouched

    def test_print_only_writes_nothing(self):
        res = wc.wire_settings(self.path, EXE, print_only=True)
        self.assertEqual(res["action"], "print")
        self.assertFalse(self.path.exists())

    def test_non_list_stop_is_malformed_not_silent(self):
        # a non-list Stop must surface as malformed, not silently report "unchanged"
        self.path.write_text(json.dumps({"hooks": {"Stop": "oops"}}), encoding="utf-8")
        res = wc.wire_settings(self.path, EXE)
        self.assertEqual(res["action"], "malformed")

    def test_string_hook_entry_does_not_crash(self):
        # a stray non-dict entry must not crash; our hook still gets installed
        self.path.write_text(json.dumps({"hooks": {"Stop": ["a string"]}}), encoding="utf-8")
        res = wc.wire_settings(self.path, EXE)
        self.assertIn(res["action"], ("merged", "created"))
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(wc._has_command(data["hooks"]["Stop"], "bridge-render"))


class TestSkillsCopy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src = _make_skills_src(self.root)
        self.dst = self.root / "home" / ".claude" / "skills"

    def tearDown(self):
        self.tmp.cleanup()

    def test_installs_new_skills(self):
        res = dict(wc.wire_skills(self.src, self.dst))
        self.assertEqual(res, {"saveall": "installed", "audit": "installed"})
        self.assertTrue((self.dst / "saveall" / "SKILL.md").exists())

    def test_existing_skill_is_protected_without_force(self):
        (self.dst / "saveall").mkdir(parents=True)
        (self.dst / "saveall" / "SKILL.md").write_text("MY OWN VERSION\n", encoding="utf-8")
        res = dict(wc.wire_skills(self.src, self.dst))
        self.assertEqual(res["saveall"], "exists-skipped")
        # the user's content is untouched
        self.assertEqual((self.dst / "saveall" / "SKILL.md").read_text(encoding="utf-8"),
                         "MY OWN VERSION\n")

    def test_force_overwrites_with_backup(self):
        (self.dst / "saveall").mkdir(parents=True)
        (self.dst / "saveall" / "SKILL.md").write_text("MY OWN VERSION\n", encoding="utf-8")
        res = dict(wc.wire_skills(self.src, self.dst, force=True))
        self.assertEqual(res["saveall"], "updated")
        self.assertIn("saveall", (self.dst / "saveall" / "SKILL.md").read_text(encoding="utf-8"))
        bak = self.dst / "saveall.bak" / "SKILL.md"
        self.assertEqual(bak.read_text(encoding="utf-8"), "MY OWN VERSION\n")

    def test_print_only(self):
        res = dict(wc.wire_skills(self.src, self.dst, print_only=True))
        self.assertEqual(set(res.values()), {"print"})
        self.assertFalse(self.dst.exists())

    def test_force_removes_stale_files(self):
        (self.dst / "saveall").mkdir(parents=True)
        (self.dst / "saveall" / "SKILL.md").write_text("OLD\n", encoding="utf-8")
        (self.dst / "saveall" / "stale_extra.md").write_text("leftover\n", encoding="utf-8")
        wc.wire_skills(self.src, self.dst, force=True)
        self.assertFalse((self.dst / "saveall" / "stale_extra.md").exists())  # replaced, not merged
        self.assertTrue((self.dst / "saveall" / "SKILL.md").exists())
        self.assertTrue((self.dst / "saveall.bak" / "stale_extra.md").exists())  # preserved in .bak


class TestOrchestration(unittest.TestCase):
    def test_wire_claude_full_run_isolated_home(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = _make_skills_src(root)
            home = root / "home"
            out = wc.wire_claude(home=home, abs_exe=EXE, skills_src=src)
            # skills installed under the temp home
            self.assertTrue((home / ".claude" / "skills" / "saveall" / "SKILL.md").exists())
            # Stop hook created
            self.assertEqual(out["settings"]["action"], "created")
            # CLAUDE.md seeded under the SAME home (never the real one)
            cm = home / ".claude" / "CLAUDE.md"
            self.assertTrue(cm.exists())
            self.assertIn("memsom:managed:start", cm.read_text(encoding="utf-8"))

    def test_claude_md_error_carries_detail(self):
        # a claude-sync failure must capture the exception detail (so the CLI can
        # surface it instead of printing a blank "[claude.md] error -> ").
        from memsom.bridge import claude as memsom_claude
        orig = memsom_claude.sync
        memsom_claude.sync = lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied"))
        try:
            with tempfile.TemporaryDirectory() as d:
                src = _make_skills_src(Path(d))
                out = wc.wire_claude(home=Path(d) / "home", abs_exe=EXE, skills_src=src)
        finally:
            memsom_claude.sync = orig
        self.assertEqual(out["claude_md"]["action"], "error")
        self.assertIn("PermissionError", out["claude_md"]["detail"])


if __name__ == "__main__":
    unittest.main()
