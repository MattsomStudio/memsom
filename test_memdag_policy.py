#!/usr/bin/env python3
"""Tests for memdag_policy — declarative capability policy (Gate #3).

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memdag_policy.py -t <repo> -v
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag_policy as P


def policy(d):
    """Normalize a raw dict through the public loader (via a temp file)."""
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "policy.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        return P.load_policy(p)


# ---------------------------------------------------------------------------
# matching + lookup
# ---------------------------------------------------------------------------

class TestLookup(unittest.TestCase):
    def setUp(self):
        self.pol = policy({
            "default": "deny",
            "rules": [
                {"tool": "fetch.fetch",      "required": "external", "taints": "external"},
                {"tool": "github.delete_*",  "required": "endorsed"},
                {"tool": "gmail.send_*",     "required": "user"},
                {"tool": "*.read_*",         "required": "external"},
            ],
        })

    def test_first_match_wins(self):
        # gmail.send_message matches send_* (user), not the later *.read_*
        self.assertEqual(P.required_floor(self.pol, "gmail.send_message"), 2)

    def test_read_tool_is_external_and_not_consequential(self):
        self.assertEqual(P.required_floor(self.pol, "fetch.fetch"), 0)
        self.assertFalse(P.is_consequential(self.pol, "fetch.fetch"))

    def test_consequential_tools(self):
        self.assertTrue(P.is_consequential(self.pol, "github.delete_repo"))
        self.assertTrue(P.is_consequential(self.pol, "gmail.send_message"))

    def test_taints_only_on_flagged_rule(self):
        self.assertEqual(P.taints(self.pol, "fetch.fetch"), 0)        # external
        self.assertIsNone(P.taints(self.pol, "gmail.send_message"))   # no taints key

    def test_unmatched_uses_default_deny(self):
        # default 'deny' -> sentinel 4 -> consequential and never satisfiable
        self.assertEqual(P.required_floor(self.pol, "weird.unknown_tool"), P.DENY)
        self.assertTrue(P.is_consequential(self.pol, "weird.unknown_tool"))

    def test_glob_wildcards(self):
        self.assertEqual(P.required_floor(self.pol, "drive.read_file"), 0)


# ---------------------------------------------------------------------------
# default allow
# ---------------------------------------------------------------------------

class TestDefaultAllow(unittest.TestCase):
    def test_default_allow_permits_unmatched(self):
        pol = policy({"default": "allow", "rules": []})
        self.assertEqual(P.required_floor(pol, "anything.at_all"), P.ALLOW)
        self.assertFalse(P.is_consequential(pol, "anything.at_all"))

    def test_default_floor_by_name(self):
        pol = policy({"default": "user", "rules": []})
        self.assertEqual(P.required_floor(pol, "x.y"), 2)

    def test_absent_default_is_deny(self):
        pol = policy({"rules": []})
        self.assertEqual(P.required_floor(pol, "x.y"), P.DENY)


# ---------------------------------------------------------------------------
# malformed -> raise (never silently allow)
# ---------------------------------------------------------------------------

class TestMalformed(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            P.load_policy("/no/such/policy/file.json")

    def test_bad_json_raises(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "p.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                P.load_policy(p)

    def test_non_object_raises(self):
        with self.assertRaises(ValueError):
            policy([1, 2, 3])

    def test_rule_without_tool_raises(self):
        with self.assertRaises(ValueError):
            policy({"rules": [{"required": "user"}]})

    def test_rule_without_required_raises(self):
        with self.assertRaises(ValueError):
            policy({"rules": [{"tool": "x.y"}]})

    def test_bad_required_value_raises(self):
        with self.assertRaises(ValueError):
            policy({"rules": [{"tool": "x.y", "required": "banana"}]})

    def test_bad_taints_value_raises(self):
        with self.assertRaises(ValueError):
            policy({"rules": [{"tool": "x.y", "required": "user", "taints": 9}]})

    def test_rules_not_a_list_raises(self):
        with self.assertRaises(ValueError):
            policy({"rules": {"tool": "x"}})


if __name__ == "__main__":
    unittest.main()
