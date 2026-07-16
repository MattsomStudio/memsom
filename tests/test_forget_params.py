"""Tests for the runtime-tunables loader (forget.load_params) and the live
MEMORY.md budget resolution (digest.resolve_budget).

These are the read side of the panel's knob layer: canonical.json `params` must
be honored when sane, ignored (with a warning) when degenerate, and absent-file
must behave exactly like the pre-panel hardcoded defaults.
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

from memsom.distill import digest
from memsom.lifecycle import forget


def _write_canonical(dirpath, params):
    weights = Path(dirpath) / ".weights"
    weights.mkdir(parents=True, exist_ok=True)
    path = weights / "canonical.json"
    path.write_text(json.dumps({"version": 1, "params": params, "memories": {}}),
                    encoding="utf-8")
    return path


class LoadParams(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _defaults(self):
        return {**forget.DEFAULTS, **forget.PANEL_PARAM_DEFAULTS}

    def test_absent_file_is_pure_defaults(self):
        params, warns = forget.load_params(self.dir / "nope" / "canonical.json")
        self.assertEqual(params, self._defaults())
        self.assertEqual(warns, [])

    def test_corrupt_json_warns_and_defaults(self):
        p = self.dir / "canonical.json"
        p.write_text("{not json", encoding="utf-8")
        params, warns = forget.load_params(p)
        self.assertEqual(params, self._defaults())
        self.assertEqual(len(warns), 1)
        self.assertIn("unreadable", warns[0])

    def test_params_not_object_warns_and_defaults(self):
        p = self.dir / "canonical.json"
        p.write_text(json.dumps({"params": [1, 2, 3]}), encoding="utf-8")
        params, warns = forget.load_params(p)
        self.assertEqual(params, self._defaults())
        self.assertEqual(len(warns), 1)

    def test_valid_overrides_apply(self):
        p = _write_canonical(self.dir, {"rs_gain": 0.3, "memory_budget": 8192,
                                        "grace_days": 10})
        params, warns = forget.load_params(p)
        self.assertEqual(warns, [])
        self.assertEqual(params["rs_gain"], 0.3)
        self.assertEqual(params["memory_budget"], 8192)
        self.assertEqual(params["grace_days"], 10)
        self.assertEqual(params["decay_base"], forget.DEFAULTS["decay_base"])

    def test_legacy_and_unknown_keys_ignored(self):
        # the live canonical.json still carries pre-rename cruft: cap/decay/gain/seed
        p = _write_canonical(self.dir, {"cap": 9.0, "decay": 9.0, "gain": 9.0,
                                        "seed": 9.0, "banana": True, "rs_gain": 0.2})
        params, warns = forget.load_params(p)
        self.assertEqual(warns, [])
        self.assertEqual(params["rs_gain"], 0.2)
        self.assertNotIn("cap", params)
        self.assertNotIn("banana", params)

    def test_degenerate_values_rejected_with_warning(self):
        p = _write_canonical(self.dir, {
            "decay_base": 1.5,        # >1: RS would grow with disuse
            "rs_cap": 0,              # zero cap
            "memory_budget": 100,     # below floor
            "grace_days": -3,         # negative
        })
        params, warns = forget.load_params(p)
        self.assertEqual(params["decay_base"], forget.DEFAULTS["decay_base"])
        self.assertEqual(params["rs_cap"], forget.DEFAULTS["rs_cap"])
        self.assertEqual(params["memory_budget"],
                         forget.PANEL_PARAM_DEFAULTS["memory_budget"])
        self.assertEqual(params["grace_days"], forget.DEFAULTS["grace_days"])
        self.assertEqual(len(warns), 4)

    def test_bool_is_not_a_number(self):
        p = _write_canonical(self.dir, {"rs_gain": True})
        params, warns = forget.load_params(p)
        self.assertEqual(params["rs_gain"], forget.DEFAULTS["rs_gain"])
        self.assertEqual(len(warns), 1)

    def test_ss_floor_above_cap_cross_check(self):
        p = _write_canonical(self.dir, {"ss_floor": 2.0, "ss_cap": 1.5})
        params, warns = forget.load_params(p)
        self.assertEqual(params["ss_floor"], forget.DEFAULTS["ss_floor"])
        self.assertEqual(params["ss_cap"], 1.5)  # cap itself was sane
        self.assertTrue(any("ss_floor" in w for w in warns))


class ResolveBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_absent_store_falls_back_to_constant(self):
        self.assertEqual(digest.resolve_budget(self.dir), digest.BUDGET)

    def test_live_override_wins(self):
        _write_canonical(self.dir, {"memory_budget": 4096})
        self.assertEqual(digest.resolve_budget(self.dir), 4096)

    def test_degenerate_override_falls_back(self):
        _write_canonical(self.dir, {"memory_budget": 12})
        self.assertEqual(digest.resolve_budget(self.dir), digest.BUDGET)

    def test_write_shadow_renders_with_live_budget(self):
        # The shadow render must use the SAME live cap as the real render —
        # this was the mismatch where _cmd_shadow printed the live budget while
        # write_shadow rendered against the hardcoded default. Seam test: spy
        # on render_digest and assert the resolved budget reaches it.
        _write_canonical(self.dir, {"memory_budget": 2048})
        captured = {}
        orig = digest.render_digest

        def spy(conn, *, title=None, budget=None):
            captured["budget"] = budget
            return "# Memory\n"

        digest.render_digest = spy
        try:
            digest.write_shadow(None, self.dir)
        finally:
            digest.render_digest = orig
        self.assertEqual(captured["budget"], 2048)


if __name__ == "__main__":
    unittest.main()
