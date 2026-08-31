"""Regression tests for memsom.paths — the shared containment primitive.

Written 2026-07-30 as the proof gate for seam S3 of the memsom-panel white-box
engagement. Two halves:

* the behaviour table the fix spec published, run row by row, so a future
  "simplification" of safe_join has to break a named case rather than a vibe;
* an adversarial half covering what the spec's table did NOT list — sibling
  prefix confusion, absolute-not-first, and the one syscall the function makes.

The sibling-prefix case is the one to keep if you ever trim this file. A
containment check written as ``str(candidate).startswith(str(root))`` — without
the separator — accepts ``.../memsom-agentic-os/backend`` for a root of
``.../memsom``. That is the single most common way this class of fence is
written wrong, it looks correct in review, and only a test that names a real
sibling directory catches it.
"""

from __future__ import annotations

import os
import sys

try:
    import pytest
except ImportError:  # stdlib-only unittest discover (CI runs these under pytest)
    import unittest
    raise unittest.SkipTest("pytest-style module; run under the CI pytest step")

from memsom.paths import UnsafePath, is_unc_or_device, safe_join

WINDOWS = os.name == "nt"
win_only = pytest.mark.skipif(not WINDOWS, reason="Win32 path semantics")


@pytest.fixture()
def root(tmp_path):
    """A real directory to fence against. Resolved, as safe_join resolves it."""
    r = tmp_path / "root"
    r.mkdir()
    return r


def rel(root, got) -> str:
    """The accepted path expressed relative to the root, for comparison."""
    return os.path.relpath(str(got), str(root.resolve()))


# --------------------------------------------------------------------------
# Accepted: the shapes real callers emit.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("README.md",), "README.md"),
        (("sub/deep/ok.md",), os.path.join("sub", "deep", "ok.md")),
        (("sub", "deep", "ok.md"), os.path.join("sub", "deep", "ok.md")),
        (("./README.md",), "README.md"),
        ((".",), "."),
        (("sub//deep//x.md",), os.path.join("sub", "deep", "x.md")),
        (("sub/",), "sub"),
        ((".hidden",), ".hidden"),
        (("a.b.c",), "a.b.c"),
    ],
)
def test_accepts_ordinary_relative_paths(root, parts, expected):
    got = safe_join(root, *parts, resolve_symlinks=False)
    assert rel(root, got) == expected


@win_only
@pytest.mark.parametrize("name", ["CONSOLE.txt", "NULL.md", "COM0", "LPT10", "console"])
def test_reserved_name_check_does_not_overreach(root, name):
    """Only COM1-9 / LPT1-9 and the four bare names are reserved.

    An over-broad check here silently breaks legitimate filenames, and it would
    do so rarely enough that nobody would connect it to this function.
    """
    got = safe_join(root, name, resolve_symlinks=False)
    assert rel(root, got) == name


# --------------------------------------------------------------------------
# Refused: the string is rejected before any syscall.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("part", "because"),
    [
        ("../evil", "'..'"),
        ("sub/../README.md", "'..'"),
        ("", "empty"),
        ("a\x00b", "NUL"),
        ("README.md\n", "whitespace"),
        (" README.md", "whitespace"),
    ],
)
def test_refuses_everywhere(root, part, because):
    with pytest.raises(UnsafePath) as exc:
        safe_join(root, part, resolve_symlinks=False)
    assert because in str(exc.value)


@win_only
@pytest.mark.parametrize(
    ("part", "because"),
    [
        ("..\\..\\evil", "'..'"),
        ("//evil.example/share/a", "UNC or device"),
        ("\\\\evil.example\\share\\a", "UNC or device"),
        ("//?/C:/x", "UNC or device"),
        ("//./PhysicalDrive0", "UNC or device"),
        ("CON", "reserved device"),
        ("nul.txt", "reserved device"),
        ("x.", "trailing dot"),
    ],
)
def test_refuses_win32_traps(root, part, because):
    with pytest.raises(UnsafePath) as exc:
        safe_join(root, part, resolve_symlinks=False)
    assert because in str(exc.value)


@win_only
def test_unc_is_refused_even_with_allow_absolute(root):
    """There is no root a UNC path could legitimately be inside.

    This is the case that costs a network round trip if it gets through:
    resolving \\\\host\\share goes out over the MUP and offers an NTLM
    challenge-response for whatever account the process runs as.
    """
    with pytest.raises(UnsafePath, match="UNC or device"):
        safe_join(root, r"\\evil.example\share\a",
                  allow_absolute=True, resolve_symlinks=False)


@pytest.mark.parametrize("bad", [b"README.md", 42, None, 3.5, ["README.md"]])
def test_refuses_non_str_components(root, bad):
    with pytest.raises(UnsafePath, match="must be str"):
        safe_join(root, bad, resolve_symlinks=False)


def test_refuses_call_with_no_components(root):
    with pytest.raises(UnsafePath, match="at least one"):
        safe_join(root)


# --------------------------------------------------------------------------
# allow_absolute: accepted only when LEXICALLY inside the root.
# --------------------------------------------------------------------------

def test_absolute_refused_by_default(root):
    inside = str(root / "README.md")
    with pytest.raises(UnsafePath, match="absolute"):
        safe_join(root, inside, resolve_symlinks=False)


def test_absolute_inside_root_accepted_when_allowed(root):
    inside = str(root / "README.md")
    got = safe_join(root, inside, allow_absolute=True, resolve_symlinks=False)
    assert rel(root, got) == "README.md"


def test_absolute_outside_root_refused_even_when_allowed(root, tmp_path):
    outside = str(tmp_path / "elsewhere" / "x.md")
    with pytest.raises(UnsafePath, match="outside the root"):
        safe_join(root, outside, allow_absolute=True, resolve_symlinks=False)


@win_only
def test_absolute_match_is_case_insensitive(root):
    """Windows paths are case-insensitive; a raw startswith is a fence that a
    different capitalisation walks straight through."""
    shouty = str(root / "README.md").upper()
    got = safe_join(root, shouty, allow_absolute=True, resolve_symlinks=False)
    assert rel(root, got).lower() == "readme.md"


def test_sibling_directory_prefix_is_refused(tmp_path):
    """THE regression that matters.

    ``.../rootEVIL`` and ``.../root-agentic-os`` both start with ``.../root``
    as a plain string. Containment must be checked with the separator, or a
    sibling directory reads as contained.
    """
    root = tmp_path / "root"
    root.mkdir()
    for sibling in ("rootEVIL", "root-agentic-os", "root2"):
        (tmp_path / sibling).mkdir()
        target = str(tmp_path / sibling / "loot.md")
        with pytest.raises(UnsafePath, match="outside the root"):
            safe_join(root, target, allow_absolute=True, resolve_symlinks=False)


def test_absolute_component_may_only_appear_first(root):
    """A later absolute part would silently discard everything before it —
    that is exactly the os.path.join behaviour this module exists to stop."""
    inside = str(root / "README.md")
    with pytest.raises(UnsafePath, match="may only appear first"):
        safe_join(root, "sub", inside, allow_absolute=True, resolve_symlinks=False)


def test_root_itself_is_contained(root):
    got = safe_join(root, str(root), allow_absolute=True, resolve_symlinks=False)
    assert os.path.normcase(str(got)) == os.path.normcase(str(root.resolve()))


@win_only
@pytest.mark.parametrize("part", ["/etc/passwd", "C:README.md"])
def test_rooted_and_drive_relative_forms_are_outside(root, part):
    with pytest.raises(UnsafePath, match="outside the root"):
        safe_join(root, part, allow_absolute=True, resolve_symlinks=False)


# --------------------------------------------------------------------------
# The one syscall: a symlink planted inside the root, pointing out.
# --------------------------------------------------------------------------

def _can_symlink(tmp_path) -> bool:
    probe = tmp_path / "_probe_link"
    try:
        probe.symlink_to(tmp_path)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


def test_symlink_escape_is_caught_by_default(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation not permitted in this environment")
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("stolen", encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    # Lexically contained. Only the resolve step can catch it.
    with pytest.raises(UnsafePath, match="via a link"):
        safe_join(root, "escape/secret.txt")


def test_resolve_symlinks_false_is_documented_as_weaker(tmp_path):
    """Pinned deliberately: with the syscall off, the lexical fence alone does
    NOT stop a symlink out of the root. Callers opting out must know that."""
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation not permitted in this environment")
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("stolen", encoding="utf-8")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    got = safe_join(root, "escape/secret.txt", resolve_symlinks=False)
    real = os.path.realpath(str(got))
    assert not os.path.normcase(real).startswith(
        os.path.normcase(str(root.resolve())) + os.sep
    )


# --------------------------------------------------------------------------
# is_unc_or_device — the pre-check for sites that cannot use safe_join.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("//evil/share/a", True),
        (r"\\evil\share\a", True),
        ("//?/C:/x", True),
        ("//./PhysicalDrive0", True),
        ("C:/Windows", False),
        ("README.md", False),
        ("/etc/passwd", False),
        ("", False),
        (None, False),
        (42, False),
    ],
)
def test_is_unc_or_device(raw, expected):
    assert is_unc_or_device(raw) is expected


# --------------------------------------------------------------------------
# The premise the whole module rests on. If this ever stops being true the
# module is solving a problem that no longer exists — and this test says so
# out loud rather than leaving safe_join looking like superstition.
# --------------------------------------------------------------------------

@win_only
def test_premise_join_discards_the_left_operand():
    from pathlib import PureWindowsPath

    assert str(PureWindowsPath(r"C:\a") / "//h/s/x") == r"\\h\s\x"
    assert os.path.join(r"C:\a", r"\\h\s\x") == r"\\h\s\x"
    assert os.path.join(r"C:\a", r"C:\Windows") == r"C:\Windows"


def test_premise_join_discards_the_left_operand_posix():
    assert os.path.join("/a/b", "/etc/passwd") == "/etc/passwd"


# --------------------------------------------------------------------------
# No syscall before the fence. The performance claim is a security claim: a
# rejected UNC path must never have caused a network round trip.
# --------------------------------------------------------------------------

def test_rejection_happens_before_any_resolve(root, monkeypatch):
    """Nothing may touch the filesystem on the rejection path except the one
    root resolve that happens up front."""
    calls = []
    import pathlib

    real_resolve = pathlib.Path.resolve

    def counting_resolve(self, *a, **kw):
        calls.append(str(self))
        return real_resolve(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "resolve", counting_resolve)

    bad = r"\\evil.example\share\a" if WINDOWS else "../../etc/passwd"
    with pytest.raises(UnsafePath):
        safe_join(root, bad)

    # Exactly one resolve, and it is the root's — never the attacker's string.
    assert len(calls) == 1, calls
    assert os.path.normcase(calls[0]) == os.path.normcase(str(root))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
