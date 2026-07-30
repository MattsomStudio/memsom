"""Inventory lock on the path primitives.

Written 2026-07-30 for seam S3 of the memsom-panel white-box engagement.

`test_paths.py` proves `safe_join` behaves. This file proves the codebase
actually *uses* it — which is the failure that keeps happening. The audit filed
two resolve-before-fence sites; an AST sweep of the live tree found seven,
including one inside the scope checker itself and three in code a review had
written up as "well-defended." Nobody was careless. The sites are individually
reasonable and collectively invisible, because `Path.resolve()` reads like a
predicate and grep cannot tell you which call is a fence and which is a lookup.

So: freeze the set of functions allowed to call `.resolve()` on a path. A new
one fails this test, and the author has to either route through
`memsom.paths.safe_join` or add an entry here with a reason. That turns the
eighth instance into a code-review conversation instead of a silent regression.

This is a source-shape assertion, not a behavioural one. It is deliberately
annoying to add to.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "memsom"

# ---------------------------------------------------------------------------
# Every function permitted to call `.resolve()`, and why.
#
# Key:   "<path/relative/to/memsom>::<enclosing function>"
# Value: why this one is not a fence that needs safe_join.
#
# Three legitimate categories, and nothing else should appear here:
#   SELF     - Path(__file__).resolve(), locating the package's own files.
#   ROOT     - resolving a TRUSTED containing directory (config, a CLI arg).
#              The root is the thing safe_join fences *against*; resolving it
#              is required, not a smell.
#   PRIMITIVE- inside memsom/paths.py, which is the implementation.
# ---------------------------------------------------------------------------
ALLOWED: dict[str, str] = {
    "__init__.py::<module>":
        "SELF - HOME = Path(__file__).resolve().parent, the packaged-resource dir.",
    "bridge/wire_claude.py::default_skills_src":
        "SELF - Path(__file__).resolve().parents[2], locates the bundled claude/ dir.",
    "bridge/obsidian.py::_walk_markdown":
        "ROOT + walked path. `vroot` is the trusted vault root. `ap.resolve()` is a "
        "path built by os.walk of that same tree, re-checked with _within to drop "
        "symlinks/junctions escaping a Syncthing-shared vault. The input is not "
        "model-authored, so there is nothing for safe_join to fence.",
    "bridge/obsidian.py::export_note":
        "ROOT - `vault = Path(vault).resolve()`. The model-authored `folder` and "
        "`title` now go through safe_join immediately below it.",
    "integrity/tombstone.py::tombstone_memory":
        "ROOT - `mem_root = Path(mem_dir).resolve()`, and it runs AFTER safe_join "
        "has already decided the model-supplied `stem` is contained.",
    "lifecycle/compact.py::_llm_summarize":
        "NOT A PATH - `memsom_llm.resolve(model, base_url)` picks a model endpoint. "
        "Same attribute name, unrelated call. Listed so the sweep's false positive "
        "is on the record rather than being rediscovered every time.",
    "paths.py::safe_join":
        "PRIMITIVE - the root resolve and the post-containment symlink re-check. "
        "This is the implementation the rest of the codebase defers to.",
}


def _resolve_sites() -> dict[str, list[int]]:
    """Every `.resolve()` call in the package, keyed by file::function."""
    found: dict[str, list[int]] = {}
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue

        scopes = [
            (n.lineno, getattr(n, "end_lineno", n.lineno), n.name)
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        def owner(line: int) -> str:
            best = None
            for start, end, name in scopes:
                if start <= line <= end and (best is None or start > best[0]):
                    best = (start, name)
            return best[1] if best else "<module>"

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve"
            ):
                found.setdefault(f"{rel}::{owner(node.lineno)}", []).append(node.lineno)
    return found


def test_no_unreviewed_resolve_site():
    """A `.resolve()` in a function nobody signed off on is a new fence.

    If this fails on code you just wrote, the question to answer is: does this
    call touch a string that came from a model, a request body, a store column,
    or a synced file? If yes, it belongs behind `memsom.paths.safe_join`. If no,
    add it to ALLOWED with which of SELF / ROOT / PRIMITIVE it is.
    """
    found = _resolve_sites()
    unreviewed = sorted(set(found) - set(ALLOWED))
    assert not unreviewed, (
        "unreviewed .resolve() call site(s) — route untrusted input through "
        "memsom.paths.safe_join, or add an annotated entry to ALLOWED in "
        + __file__
        + ":\n"
        + "\n".join(f"  memsom/{k}  (line{'s' if len(found[k]) > 1 else ''} "
                    f"{', '.join(str(n) for n in found[k])})" for k in unreviewed)
    )


def test_allowlist_has_no_dead_entries():
    """A stale entry silently widens the gate.

    If a function is deleted or renamed and its allowlist entry stays, the next
    function to take that name inherits permission it never earned.
    """
    found = _resolve_sites()
    dead = sorted(set(ALLOWED) - set(found))
    assert not dead, (
        "ALLOWED names function(s) that no longer call .resolve(); remove them:\n"
        + "\n".join(f"  {k}" for k in dead)
    )


@pytest.mark.parametrize("key,reason", sorted(ALLOWED.items()))
def test_every_allowlist_entry_states_its_category(key, reason):
    """An entry without a category is an unexplained suppression."""
    assert any(reason.startswith(c) for c in ("SELF", "ROOT", "PRIMITIVE", "NOT A PATH")), (
        f"{key} has no category — say which of SELF / ROOT / PRIMITIVE / NOT A PATH it is"
    )


def test_the_sweep_actually_finds_things():
    """Premise check: if the AST pass silently stopped matching, both gates above
    would pass vacuously and this file would be decoration."""
    found = _resolve_sites()
    assert len(found) >= 5, f"AST sweep found only {len(found)} sites — it is probably broken"
    assert "paths.py::safe_join" in found, "the sweep cannot even see the primitive itself"


def _join_then_resolve_sites() -> list[str]:
    """`(a / b).resolve()` — resolve applied straight to a joined path.

    This is the resolve-before-fence shape itself, matched structurally rather
    than by text. A textual check here is worse than useless: the first version
    of this test failed on the *comment* in obsidian.py that explains what the
    old form was, which is the exact way a source-shape gate becomes something
    people delete instead of fix.
    """
    out: list[str] = []
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "resolve"):
                continue
            recv = node.func.value
            if isinstance(recv, ast.BinOp) and isinstance(recv.op, ast.Div):
                out.append(f"memsom/{rel}:{node.lineno}")
    return out


def _resolve_then_contain_sites() -> list[str]:
    """`x.resolve().is_relative_to(...)` — the fence one step too late.

    The predicate is correct; the ordering is not. By the time `is_relative_to`
    runs, `resolve()` has already made the syscall — and for a UNC path that
    syscall is a DNS lookup, a TCP/445 connection and an NTLM exchange offering
    this process's credentials to a host the attacker named.
    """
    out: list[str] = []
    for f in sorted(PKG.rglob("*.py")):
        rel = f.relative_to(PKG).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("is_relative_to", "startswith", "commonpath")):
                continue
            recv = node.func.value
            if (isinstance(recv, ast.Call)
                    and isinstance(recv.func, ast.Attribute)
                    and recv.func.attr == "resolve"):
                out.append(f"memsom/{rel}:{node.lineno}")
    return out


def test_no_join_then_resolve():
    """`(root / untrusted).resolve()` must not exist anywhere in the package.

    Both halves of it are wrong. `/` DISCARDS `root` as soon as `untrusted`
    carries a drive or a UNC prefix, so the join is not containment; and the
    resolve then touches the network before anything has decided it wanted to.
    """
    sites = _join_then_resolve_sites()
    assert not sites, (
        "resolve-before-fence: `(a / b).resolve()` at\n"
        + "\n".join(f"  {s}" for s in sites)
        + "\nUse memsom.paths.safe_join(a, b) — it fences on the string first."
    )


def test_no_resolve_then_containment_check():
    sites = _resolve_then_contain_sites()
    assert not sites, (
        "containment checked AFTER resolve() — the syscall already happened:\n"
        + "\n".join(f"  {s}" for s in sites)
        + "\nUse memsom.paths.safe_join, which decides before touching the disk."
    )


def test_the_shape_detectors_actually_detect():
    """Premise check for the two gates above.

    Both assert an empty list, so a detector that silently stopped matching
    would leave them passing forever. Feed each one the shape it hunts and
    require a hit.
    """
    joined = ast.parse("(root / user_input).resolve()")
    node = joined.body[0].value
    assert (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resolve"
            and isinstance(node.func.value, ast.BinOp)
            and isinstance(node.func.value.op, ast.Div)), "join-then-resolve matcher is broken"

    late = ast.parse("p.resolve().is_relative_to(root)")
    node = late.body[0].value
    recv = node.func.value
    assert (isinstance(node.func, ast.Attribute)
            and node.func.attr == "is_relative_to"
            and isinstance(recv, ast.Call)
            and recv.func.attr == "resolve"), "resolve-then-contain matcher is broken"


def test_the_fixed_sites_no_longer_resolve_untrusted_input():
    """The three sites this seam repaired must not have regressed."""
    found = _resolve_sites()
    assert "integrity/redact.py::_unlink_within" not in found, (
        "_unlink_within resolves again — it must fence with safe_join first"
    )


def test_fixed_sites_call_the_shared_primitive():
    """Positive half of the check above: they route through safe_join."""
    for src in ("integrity/redact.py", "integrity/tombstone.py", "bridge/obsidian.py"):
        text = (PKG / src).read_text(encoding="utf-8")
        assert "safe_join(" in text, f"{src} does not use the shared primitive"
        assert "from memsom.paths import" in text, f"{src} does not import it"
