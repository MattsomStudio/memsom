"""Regression tests for memsom.childenv — the child-process credential fence.

Written 2026-07-30 as the proof gate for F-19 + F-21 of the memsom-panel
white-box engagement. The finding: `subprocess` inherits the parent environment
by default and not one spawn site opted out, so every child — including the
DETACHED, long-lived ones that outlive their parent — carried the Anthropic and
OpenAI keys.

Rated hygiene, honestly: both keys are in HKCU\\Environment in plaintext, so a
same-user attacker reads them with one `reg query` and inheritance adds nothing
to their reach. What these tests protect is the *other* direction — a child that
never receives a credential cannot write one into a log, a crash dump or a
debug dump, however its stdout is redirected later.

The last test is the one to keep if this file is ever trimmed: a denylist whose
membership is asserted nowhere silently becomes an empty tuple after a bad
merge, and every test above it would still pass.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from memsom.childenv import (CREDENTIAL_ENV_NAMES, child_env, minimal_env)


@pytest.fixture()
def planted(monkeypatch):
    """A parent environment carrying every credential plus one innocent."""
    for name in CREDENTIAL_ENV_NAMES:
        monkeypatch.setenv(name, f"REAL-SECRET-{name}")
    monkeypatch.setenv("KEEP_ME", "still-here")
    monkeypatch.setenv("SystemRoot", os.environ.get("SystemRoot", r"C:\Windows"))
    return None


# --------------------------------------------------------------------------
# child_env: removes credentials, keeps everything else.
# --------------------------------------------------------------------------

def test_every_credential_name_is_removed(planted):
    env = child_env()
    for name in CREDENTIAL_ENV_NAMES:
        assert name not in env, f"{name} survived into the child environment"


def test_removed_not_blanked(planted):
    """`NAME in os.environ` must be False in the child, not True-with-''.

    A blanked variable still tells the child the name exists and is worth
    reading, and some libraries treat an empty credential as "configured but
    broken" rather than "absent".
    """
    env = child_env()
    assert "MEMSOM_ANTHROPIC_KEY" not in env
    assert env.get("MEMSOM_ANTHROPIC_KEY", None) is None


def test_ordinary_variables_survive(planted):
    """The control. Without this, a child_env that returned {} would pass
    every assertion above and break every model server on the box."""
    env = child_env()
    assert env["KEEP_ME"] == "still-here"
    # Case-insensitive: CPython UPPERCASES every os.environ key on Windows.
    assert any(k.upper() == "SYSTEMROOT" for k in env)
    assert len(env) > 5, "a near-empty environment is an allowlist, not this"


def test_keep_readmits_a_named_credential(planted):
    # saveall.py uses this: the child IS an Anthropic client, so it may
    # legitimately need to authenticate with them.
    env = child_env(keep=("ANTHROPIC_API_KEY",))
    assert env["ANTHROPIC_API_KEY"] == "REAL-SECRET-ANTHROPIC_API_KEY"
    assert "OPENAI_API_KEY" not in env


def test_drop_adds_a_call_site_specific_name(planted):
    monkeypatch_free = child_env(drop=("KEEP_ME",))
    assert "KEEP_ME" not in monkeypatch_free


def test_absent_names_are_not_an_error(monkeypatch):
    for name in CREDENTIAL_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    assert isinstance(child_env(), dict)


def test_the_parent_environment_is_not_mutated(planted):
    child_env()
    assert os.environ["MEMSOM_ANTHROPIC_KEY"].startswith("REAL-SECRET")


# --------------------------------------------------------------------------
# minimal_env: a real allowlist, for children whose needs are known and tiny.
# --------------------------------------------------------------------------

def test_minimal_env_admits_only_known_names(planted):
    env = minimal_env()
    assert "KEEP_ME" not in env
    for name in CREDENTIAL_ENV_NAMES:
        assert name not in env


def test_minimal_env_carries_enough_to_start_a_console_child(planted):
    env = minimal_env()
    if sys.platform == "win32":
        # THE regression. _MINIMAL_ENV_NAMES is written in conventional casing
        # ("SystemRoot", "windir") but os.environ's keys are uppercased, so an
        # exact-match allowlist admits NEITHER and the child cannot start.
        assert any(k.upper() == "SYSTEMROOT" for k in env), \
            "a Windows child cannot start without SystemRoot"
        assert any(k.upper() == "WINDIR" for k in env)
        assert len(env) >= 8, sorted(env)


def test_minimal_env_extra_admits_a_named_variable(planted):
    assert minimal_env(extra=("KEEP_ME",))["KEEP_ME"] == "still-here"


@pytest.mark.skipif(os.name != "nt", reason="Win32 env-name casing")
def test_name_matching_is_case_insensitive_on_windows(planted):
    """Both helpers must fold case, or every name written in its conventional
    casing silently matches nothing."""
    assert "MEMSOM_ANTHROPIC_KEY" not in child_env(drop=("memsom_anthropic_key",))
    assert child_env(keep=("memsom_anthropic_key",))["MEMSOM_ANTHROPIC_KEY"]
    assert any(k.upper() == "KEEP_ME" for k in minimal_env(extra=("keep_me",)))


# --------------------------------------------------------------------------
# End to end: a real child, a real environment.
# --------------------------------------------------------------------------

def test_a_real_child_cannot_see_the_credential(planted):
    """The property under test, through an actual process boundary rather than
    a dict comparison."""
    probe = ("import os; "
             "print(os.environ.get('MEMSOM_ANTHROPIC_KEY', '<absent>')); "
             "print(os.environ.get('KEEP_ME', '<absent>'))")

    scoped = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, timeout=60, env=child_env())
    assert scoped.stdout.splitlines() == ["<absent>", "still-here"], scoped.stdout

    # The control: without the fence the child DOES see it. Without this, the
    # assertion above would pass just as happily for a variable that was never
    # in the parent environment either.
    inherited = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                               text=True, timeout=60)
    assert "REAL-SECRET" in inherited.stdout, inherited.stdout


# --------------------------------------------------------------------------
# The list itself. A denylist nobody asserts on quietly becomes ().
# --------------------------------------------------------------------------

def test_the_denylist_still_names_the_keys_this_box_actually_has():
    # MEASURED 2026-07-30: MEMSOM_ANTHROPIC_KEY (len 108) and OPENAI_API_KEY
    # (len 164) are the two set in HKCU\Environment. If a refactor empties or
    # shortens this tuple, every other test in this file still passes.
    for required in ("MEMSOM_ANTHROPIC_KEY", "OPENAI_API_KEY",
                     "ANTHROPIC_API_KEY", "HF_TOKEN", "GITHUB_TOKEN",
                     "AWS_SECRET_ACCESS_KEY", "MEMSOM_PANEL_TOKEN_DIR"):
        assert required in CREDENTIAL_ENV_NAMES
    assert len(CREDENTIAL_ENV_NAMES) >= 11
    assert len(set(CREDENTIAL_ENV_NAMES)) == len(CREDENTIAL_ENV_NAMES)


def test_childenv_stays_stdlib_only():
    """This module lives in the public repo, which advertises zero runtime
    dependencies. It is imported by the panel's spawn choke point, so a stray
    import here would propagate into every child spawn path."""
    import ast
    from pathlib import Path
    import memsom.childenv as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported <= {"os", "__future__"}, imported


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
