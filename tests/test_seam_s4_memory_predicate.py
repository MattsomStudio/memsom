"""The `memory:` namespace predicate is hand-rolled in every module that needs it.

S4 / F-33's last unowned proof gate, and the only one of the thirteen that lands
RED. Written by refactor Phase 7 (the white-box re-run), which is the phase that
opens these files -- amendment A-8 refused to assign it to a phase that never
would, on the grounds that "a checkbox nobody can honestly tick" is worse than a
stated gap.

WHY THIS IS A SECURITY PROPERTY AND NOT A TIDINESS ONE
------------------------------------------------------
`source_ref LIKE 'memory:%'` is the fence between two trust classes. Nodes in
that namespace are BRIDGE-OWNED: they are minted by `bridge_import` from files
the memory directory actually holds, they carry `bridge_path`, and the
reconciler is allowed to delete them because it can see the file that justifies
them. Everything else is caller-minted.

The original engagement's finding #1 is what happens when the two classes blur:
one MCP `ingest_text` call declaring `source_ref="memory:user_adhd"` implanted
attacker text into the always-loaded `MEMORY.md` on both machines, and the
reconcile sweep could not remove it, because a fileless node is invisible to a
sweep keyed on `bridge_path`. Three separate legs closed that -- the channel
ceiling, `enforce_source_ref_namespace`, and `digest._rows`' render predicate.

**The third leg is the one this gate is about, and it exists in exactly one of
the fourteen places the predicate is written.** `digest.py:148` is the only site
that carries the `bridge_path IS NOT NULL` half. Every other site asks "is this
in the memory namespace" and gets a different answer to "is this a node the
bridge actually owns".

WHAT IS MEASURED, AND WHY IT IS AN AST WALK AND NOT A GREP
----------------------------------------------------------
A text search over this repository counts the comment explaining the predicate.
`forget.py:21` and `forget.py:261` are a module docstring and a function
docstring that quote the SQL verbatim in prose; `bridge_import.py:282` explains
the literal-hash scheme. Three false positives out of seventeen textual hits --
and this project has paid at least four times for a gate that fired on the
comment describing the thing it bans.

So this walks the AST and reads only string constants that are NOT a docstring:
the module docstring, and the first statement of every function and class body,
are dropped by construction before any matching happens.

THE CONTROL, AND WHY IT SITS AT THE EDGE (A-14 rule 2)
-------------------------------------------------------
The obvious control is "does the detector see a predicate I plant in a module".
That is the centre, and it proves nothing about the failure this gate exists to
catch, which is a detector that has quietly stopped reading source at all.

So the control has two arms and they pull in opposite directions:

  * `test_the_detector_does_not_count_a_predicate_in_a_docstring` plants the
    exact banned string inside a docstring and asserts the count does NOT move.
    That is the edge on the false-positive side, and it is where the four prior
    failures in this project landed.

  * `test_the_detector_counts_a_predicate_in_executable_code` plants the same
    string as a real expression and asserts the count DOES move. Without this
    arm, a detector that returns an empty list for every input satisfies the
    first arm perfectly.

If both controls pass, what is still invisible: a module that composes the
predicate at runtime out of fragments (`"memory:" + "%"`), or one that reads it
from a constant defined elsewhere. This gate reads string literals in place. A
future site that hides the predicate behind an f-string or a module constant
would be a *third* dialect and this gate would not see it -- which is an
argument for the shared primitive, not against the gate.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import pytest

#: The namespace fence. Anything asking a question about this prefix in SQL is
#: re-deriving a rule that `digest.py` gets right and the other sites do not.
_NEEDLE = "memory:"

#: Files that legitimately own the predicate once a primitive exists. Kept as a
#: named set rather than a magic number so the exemption is a decision somebody
#: made out loud, in the shape A-1's ownership table requires.
_OWNERS: frozenset[str] = frozenset()

_ROOT = pathlib.Path(__file__).resolve().parent.parent / "memsom"


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """id() of every Constant that is a docstring.

    `ast.get_docstring` only covers Module/FunctionDef/AsyncFunctionDef/ClassDef,
    which is exactly the set Python treats as docstrings -- so this is the
    definition, not an approximation of it.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.add(id(first.value))
    return out


def sites_in_source(src: str) -> list[int]:
    """Line numbers of executable string constants naming the namespace.

    Reads `lineno`, not walk order: `ast.walk` is breadth-first, so a gate that
    reported "the first hit" from walk order would name a line that is not the
    first line. This one sorts.
    """
    tree = ast.parse(src)
    skip = _docstring_nodes(tree)
    hits: list[int] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in skip
                and _NEEDLE in node.value
                and "LIKE" in node.value.upper()):
            hits.append(node.lineno)
    return sorted(hits)


def survey() -> dict[str, list[int]]:
    """module path (repo-relative, posix) -> executable predicate line numbers."""
    found: dict[str, list[int]] = {}
    for path in sorted(_ROOT.rglob("*.py")):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):  # pragma: no cover - not reachable in-tree
            continue
        lines = sites_in_source(src)
        if lines:
            rel = path.relative_to(_ROOT.parent).as_posix()
            if rel not in _OWNERS:
                found[rel] = lines
    return found


class TheMemoryNamespacePredicateHasOneOwner(unittest.TestCase):

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "MEASURED 2026-08-03 by refactor Phase 7: 14 executable occurrences "
            "across 7 modules, in 3 mutually inconsistent shapes -- bare "
            "`memory:%`, `memory:%` AND NOT `memory:literal:%`, and "
            "`memory:literal:%`. Only distill/digest.py:148 carries the "
            "`bridge_path IS NOT NULL` half that makes finding #1's render fence "
            "structural. There is no shared primitive for the namespace half, "
            "although `memsom.storage.schema.taint_filter_clauses` is exactly the "
            "shape one should take for the adjacent taint question. strict=True: "
            "when a primitive lands and this passes, the xfail becomes a FAILURE "
            "demanding the marker come off -- the gate must not quietly stay "
            "yellow after it is fixed."),
    )
    def test_no_module_hand_rolls_a_memory_predicate(self):
        found = survey()
        detail = "\n".join(
            f"  {mod}:{','.join(str(n) for n in lines)}"
            for mod, lines in sorted(found.items()))
        self.assertEqual(
            found, {},
            "the `memory:` namespace predicate is written by hand in "
            f"{len(found)} module(s), {sum(len(v) for v in found.values())} "
            f"site(s):\n{detail}\n"
            "One of them (distill/digest.py) carries the `bridge_path IS NOT "
            "NULL` half and the rest do not, so 'is this a memory node' and 'is "
            "this a node the bridge owns' answer differently depending on which "
            "module asks. Route them through one primitive, in the shape of "
            "memsom.storage.schema.taint_filter_clauses, and add it to _OWNERS.")

    def test_the_shape_of_the_gap_is_recorded_not_just_its_size(self):
        """A count is not a finding; the DIALECTS are.

        This passes today. It exists so that the RED gate above cannot be
        'fixed' by deleting sites until the number reaches zero while the
        inconsistency survives -- the thing that makes this a security property
        is that the sites disagree, not that there are many of them.
        """
        found = survey()
        self.assertGreater(len(found), 1,
                           "if this drops to one module the gate above should "
                           "be passing; re-check before relaxing anything")
        digest = [m for m in found if m.endswith("distill/digest.py")]
        self.assertEqual(len(digest), 1,
                         "distill/digest.py is the site that carries the "
                         "bridge_path half; if it stops matching, the render "
                         "fence for finding #1 has moved and this file's "
                         "argument needs re-deriving")


class TheDetectorItself(unittest.TestCase):
    """Both directions, at the edge rather than the centre. See the module docstring."""

    def test_the_detector_does_not_count_a_predicate_in_a_docstring(self):
        planted = (
            '"""Module doc: reads WHERE source_ref LIKE \'memory:%\' from the store."""\n'
            "\n"
            "def f():\n"
            "    \"\"\"Function doc: also WHERE source_ref LIKE 'memory:%'.\"\"\"\n"
            "    return 1\n"
            "\n"
            "class C:\n"
            "    \"\"\"Class doc: WHERE source_ref LIKE 'memory:literal:%'.\"\"\"\n"
            "    x = 1\n"
        )
        self.assertEqual(
            sites_in_source(planted), [],
            "the detector counted a docstring. This is the failure this "
            "project has paid for four times: a gate for a banned pattern "
            "firing on the comment that explains the banned pattern.")

    def test_the_detector_counts_a_predicate_in_executable_code(self):
        planted = (
            '"""Doc only."""\n'
            "\n"
            "def f(conn):\n"
            "    return conn.execute(\n"
            "        \"SELECT id FROM nodes WHERE source_ref LIKE 'memory:%'\")\n"
        )
        self.assertEqual(
            sites_in_source(planted), [5],
            "the detector missed a real predicate -- without this arm, a "
            "detector that returns [] for every input passes the docstring "
            "control perfectly.")

    def test_the_detector_reads_lineno_and_not_walk_order(self):
        """`ast.walk` is breadth-first, so 'the first hit' is not the first line."""
        planted = (
            "def outer(conn):\n"
            "    def inner():\n"
            "        return \"WHERE source_ref LIKE 'memory:literal:%'\"\n"
            "    return conn.execute(\"WHERE source_ref LIKE 'memory:%'\")\n"
        )
        self.assertEqual(sites_in_source(planted), [3, 4])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
