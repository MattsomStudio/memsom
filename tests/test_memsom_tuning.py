"""Tests for memsom.tuning -- ARCH-09: every knob has a type/bounds, and
resolve() never raises on a malformed or out-of-bounds env value.

Run:  python -m unittest tests.test_memsom_tuning
"""

import logging
import os
import unittest

from memsom import tuning


_VALID_TYPES = {int, float, bool, str, "path", "enum"}


class RegistryEnv(unittest.TestCase):
    """Base: snapshot + restore any env var a test touches, and reset the
    tuning module's per-process warn-dedup + override state so tests don't
    leak into each other."""

    def setUp(self):
        self._saved_env = dict(os.environ)
        tuning._clear_warned()
        tuning._overrides.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        tuning._clear_warned()
        tuning._overrides.clear()


class TestEveryKnobHasAType(unittest.TestCase):
    def test_every_registered_knob_has_a_type(self):
        self.assertGreater(len(tuning.REGISTRY), 0)
        for key, knob in tuning.REGISTRY.items():
            self.assertIn(knob.type, _VALID_TYPES, f"{key} has no valid type: {knob.type!r}")

    def test_numeric_knobs_have_bounds_containing_default(self):
        for key, knob in tuning.REGISTRY.items():
            if knob.type in (int, float) and knob.bounds is not None:
                lo, hi = knob.bounds
                self.assertLessEqual(lo, knob.default, f"{key}: default below its own bounds")
                self.assertLessEqual(knob.default, hi, f"{key}: default above its own bounds")

    def test_enum_knobs_have_choices_containing_default(self):
        for key, knob in tuning.REGISTRY.items():
            if knob.type == "enum":
                self.assertTrue(knob.choices, f"{key}: enum knob with no choices")
                if knob.default:
                    self.assertIn(knob.default, knob.choices,
                                  f"{key}: default {knob.default!r} not in {knob.choices}")


class TestIntBoundsFailOpen(RegistryEnv):
    KEY = "lifecycle.verify_stale_days"
    ENV = "MEMDAG_VERIFY_STALE_DAYS"

    def test_in_bounds_value_passes_through_with_the_right_type(self):
        os.environ[self.ENV] = "45"
        raw = tuning.resolve(self.KEY)
        self.assertEqual(raw, "45")            # raw string passes through unmodified
        self.assertEqual(int(raw), 45)         # and it IS the knob's declared type (int)

    def test_out_of_bounds_falls_back_to_default_without_raising(self):
        os.environ[self.ENV] = "999999"        # way outside (0, 36500)
        try:
            value = tuning.resolve(self.KEY)
        except Exception as exc:   # pragma: no cover -- this is exactly what must NOT happen
            self.fail(f"resolve() raised on an out-of-bounds value: {exc!r}")
        self.assertEqual(value, tuning.REGISTRY[self.KEY].default)

    def test_out_of_bounds_emits_exactly_one_warning(self):
        os.environ[self.ENV] = "999999"
        with self.assertLogs("memsom.tuning", level="WARNING") as cm:
            tuning.resolve(self.KEY)
            tuning.resolve(self.KEY)   # second call, same key: must NOT warn again
        self.assertEqual(len(cm.records), 1)

    def test_non_coercible_value_falls_back_without_raising(self):
        os.environ[self.ENV] = "abc"
        try:
            value = tuning.resolve(self.KEY)
        except Exception as exc:   # pragma: no cover
            self.fail(f"resolve() raised on a non-coercible value: {exc!r}")
        self.assertEqual(value, tuning.REGISTRY[self.KEY].default)
        self.assertIsInstance(value, int)


class TestFloatBoundsFailOpen(RegistryEnv):
    KEY = "contradict.nli_threshold"
    ENV = "MEMDAG_CONTRADICT_NLI_THRESHOLD"

    def test_in_bounds_value_passes_through_with_the_right_type(self):
        os.environ[self.ENV] = "0.6"
        raw = tuning.resolve(self.KEY)
        self.assertEqual(raw, "0.6")
        self.assertEqual(float(raw), 0.6)

    def test_out_of_bounds_falls_back_to_default(self):
        os.environ[self.ENV] = "5.0"           # outside (0.0, 1.0)
        value = tuning.resolve(self.KEY)
        self.assertEqual(value, tuning.REGISTRY[self.KEY].default)
        self.assertIsInstance(value, float)

    def test_non_coercible_falls_back_without_raising(self):
        os.environ[self.ENV] = "not-a-float"
        value = tuning.resolve(self.KEY)
        self.assertEqual(value, tuning.REGISTRY[self.KEY].default)


class TestBoolCoercion(RegistryEnv):
    KEY = "contradict.nli_enabled"
    ENV = "MEMDAG_CONTRADICT_NLI"

    def test_accepts_1_0_true_false_yes_no_case_insensitively(self):
        for spelling in ("1", "0", "true", "FALSE", "Yes", "nO", "True", "TRUE"):
            os.environ[self.ENV] = spelling
            raw = tuning.resolve(self.KEY)
            self.assertEqual(raw, spelling, f"valid bool spelling {spelling!r} got replaced")

    def test_invalid_bool_falls_back_without_raising(self):
        os.environ[self.ENV] = "maybe"
        value = tuning.resolve(self.KEY)
        self.assertEqual(value, tuning.REGISTRY[self.KEY].default)
        self.assertIsInstance(value, bool)

    def test_unset_returns_typed_default(self):
        os.environ.pop(self.ENV, None)
        value = tuning.resolve(self.KEY)
        self.assertIs(value, False)


class TestEnumCoercion(RegistryEnv):
    def test_valid_choice_passes_through(self):
        os.environ["MEMDAG_HOOK_MODE"] = "Enforcing"
        self.assertEqual(tuning.resolve("bridge.hook_mode"), "Enforcing")

    def test_invalid_choice_falls_back_without_raising(self):
        os.environ["MEMDAG_EMBED_BACKEND"] = "nonsense"
        value = tuning.resolve("embed.backend")
        self.assertEqual(value, tuning.REGISTRY["embed.backend"].default)


class TestRoundtripScriptStillHappy(unittest.TestCase):
    """Cheap in-process equivalent of the gate line
    `memsom tuning list --json | python scripts/check_panel_roundtrip.py`."""

    def test_every_knob_roundtrips(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import check_panel_roundtrip

        knobs = list(tuning.as_json().values())
        failures = [k["name"] for k in knobs if not check_panel_roundtrip._roundtrips(k)]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
