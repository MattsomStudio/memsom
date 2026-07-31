#!/usr/bin/env python3
r"""No compiled fence in this package may anchor with `$` — F-16.

Seam S3 of the memsom-panel white-box engagement.

In Python `$` is a LINE anchor. It matches at the end of the string OR
immediately before a single trailing newline, and `re.match` does not imply
`fullmatch`. So `re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$").match("nodes\n")` is a
match, and a fence whose comment says "must be a plain SQL identifier" accepts
one that is not.

The finding was filed against five filename fences in the panel package, which
is where it is reachable: an id with a trailing newline becomes a filename, and
on POSIX that creates a file nobody can name again. This package carries the
seventh instance of the same mistake, in `storage/schema.py`, and it is NOT
reachable — the callers of `_check_ident` pass module constants and column names
read back from `PRAGMA table_info`, never a request field. It is fixed anyway,
because the defect is a habit rather than a bug: seven fences written at
different times all stated an exact-match invariant and all expressed it with a
line anchor. Fixing six and leaving one is how the seventh gets copied.

WHY THIS PARSES INSTEAD OF GREPPING. A text gate for a banned pattern fires on
the prose explaining the banned pattern — this docstring contains the shape
twice and would fail itself. The scan walks the AST, so it only ever sees
`ast.Call` nodes; a comment, a docstring, or a string that merely quotes the
shape is not one. `test_the_gate_matches_the_ast_not_the_characters` proves that
rather than asserting it.
"""

import ast
import pathlib
import re
import textwrap
import unittest

import memsom
from memsom.storage import schema

PKG = pathlib.Path(memsom.__file__).resolve().parent

#: (module path relative to the package parent, variable name) pairs whose `$`
#: is a LINE anchor applied to an already-split line — markdown and frontmatter
#: parsers, not end-of-string fences. MEASURED: each is used with `.match()` on
#: one line at a time, where matching before a trailing newline is the point.
#:
#: An entry here is a standing exception, so `test_no_dead_allowances` deletes
#: it when the site goes. An allowlist that outlives its call site silently
#: pre-authorises the next thing that reuses the name.
_LINE_PARSERS = {
    ("memsom/bridge/bridge_import.py", "_PRIMARY_RE"),
    ("memsom/bridge/obsidian.py", "_FM_LIST_ITEM"),
    ("memsom/bridge/obsidian.py", "_FM_KV"),
    ("memsom/integrity/contradict.py", "_META_TOK"),
}


def _dollar_anchored_fences(path, label=None):
    r"""Module-level ``NAME = re.compile("…$")`` assignments in one file.

    AST, not text: the node must be an ``ast.Assign`` at module scope whose
    value is a ``Call`` to ``<something>.compile`` whose first argument is a
    string LITERAL ending in ``$``. A pattern ending in ``\$`` is an escaped
    literal dollar rather than an anchor and is excluded.
    """
    rel = label if label is not None else path.as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "compile"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)):
            continue
        pattern = call.args[0].value
        if not pattern.endswith("$") or pattern.endswith("\\$"):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                yield rel, tgt.id, node.lineno


def _package_fences():
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        yield from _dollar_anchored_fences(p, p.relative_to(PKG.parent).as_posix())


class IdentifierFenceAnchorTest(unittest.TestCase):

    def test_ident_fence_rejects_a_trailing_newline(self):
        """The site this seam actually changed."""
        self.assertTrue(schema._IDENT.match("nodes"),
                        "the fence must still accept a real identifier")
        self.assertFalse(schema._IDENT.match("nodes\n"),
                         "_IDENT still uses `$` instead of `\\Z`")
        self.assertFalse(schema._IDENT.match("nodes\r\n"))
        self.assertFalse(schema._IDENT.match("\nnodes"))

    def test_check_ident_rejects_a_trailing_newline(self):
        """The wrapper is what callers use; a fixed pattern behind a wrapper
        that re-strips would be a green regex and an open gate."""
        schema._check_ident("nodes")                     # must not raise
        with self.assertRaises(ValueError):
            schema._check_ident("nodes\n")

    def test_no_fence_anchors_with_dollar(self):
        offenders = [f"{rel}:{line} {name}"
                     for rel, name, line in _package_fences()
                     if (rel, name) not in _LINE_PARSERS]
        self.assertEqual(offenders, [], (
            "these patterns end with `$`, which matches before a trailing "
            "newline. Use `\\Z`. If the pattern really is a line parser "
            "applied to an already-split line, add the (module, name) pair to "
            "_LINE_PARSERS:\n  " + "\n  ".join(offenders)))

    def test_no_dead_allowances(self):
        """An allowlist entry that outlives its call site is an amnesty."""
        live = {(rel, name) for rel, name, _ in _package_fences()}
        dead = sorted(_LINE_PARSERS - live)
        self.assertEqual(dead, [], (
            "these _LINE_PARSERS entries no longer match anything in the "
            f"package; delete them rather than leaving a standing exception: {dead}"))

    def test_the_allowlisted_parsers_really_are_line_parsers(self):
        """Not a formality. The exception is granted because these patterns
        contain parser machinery — groups, whitespace classes, wildcards — and
        are applied one line at a time. An entry that is a bare character-class
        fence is a fence somebody excused, and this catches that.
        """
        by_name = {(rel, name): None for rel, name, _ in _package_fences()}
        self.assertTrue(set(_LINE_PARSERS) <= set(by_name),
                        "an allowlisted name is not in the scan at all")
        for p in sorted(PKG.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(PKG.parent).as_posix()
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                call = node.value
                if not (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "compile"
                        and call.args
                        and isinstance(call.args[0], ast.Constant)
                        and isinstance(call.args[0].value, str)):
                    continue
                for tgt in node.targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    if (rel, tgt.id) not in _LINE_PARSERS:
                        continue
                    pat = call.args[0].value
                    self.assertTrue(
                        any(tok in pat for tok in ("(", "\\s", ".*", ".+")),
                        f"{rel} {tgt.id} is allowlisted as a line parser but "
                        f"looks like a plain fence: {pat!r}")


class GateBehaviourTest(unittest.TestCase):
    """The gate's own control tests. Without these the inventory gate above is
    permanently green and stops demonstrating that it can ever fail."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_gate_matches_the_ast_not_the_characters(self):
        decoy = self.dir / "decoy.py"
        decoy.write_text(textwrap.dedent(r'''
            """Quotes the defect: _X = re.compile(r"^[a-z]{1,8}$")"""
            import re

            # And again in a comment: _Y = re.compile(r"^[a-z]{1,8}$")
            _NOT_A_FENCE = "_Z = re.compile(r'^[a-z]{1,8}$')"
            _FIXED = re.compile(r"^[a-z]{1,8}\Z")
            _ESCAPED_DOLLAR = re.compile(r"^costs \$")
            _BUILT = re.compile("^" + "[a-z]+" + "$")
        ''').lstrip(), encoding="utf-8")
        self.assertEqual(
            list(_dollar_anchored_fences(decoy)), [],
            "the gate fired on a comment, a docstring, a string literal, an "
            "escaped dollar or a concatenation — it matches characters, not AST")

    def test_the_gate_still_sees_a_real_offender(self):
        """'Found nothing' is worthless unless the same probe finds the thing
        when it is really there."""
        real = self.dir / "real.py"
        real.write_text('import re\n_REAL = re.compile(r"^[a-z]{1,8}$")\n',
                        encoding="utf-8")
        found = [(n, ln) for _, n, ln in _dollar_anchored_fences(real)]
        self.assertEqual(found, [("_REAL", 2)],
                         f"the gate is dead, not clean: {found}")

    def test_the_pre_fix_ident_pattern_really_admitted_a_newline(self):
        """The premise, frozen. If this ever fails, F-16 was wrong about this
        site and the fix above is cargo."""
        self.assertTrue(re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$').match("nodes\n"))
        self.assertFalse(re.compile(r'^[A-Za-z_][A-Za-z0-9_]*\Z').match("nodes\n"))


if __name__ == "__main__":
    unittest.main()
