"""GATE for MS-18 -- a failed hook-post taint write must not mint a clean
session on the next hook-pre.

`cmd_hook_post` wraps its whole taint write in `except Exception: ... return`,
and `ensure_session` never lowers an existing session (first sight only).
Before the fix, a transient failure in the `lower_floor` UPDATE (the finding's
own trigger: a routine write held past 5s elsewhere in the store) silently
discarded the taint -- the session row is untouched, so the NEXT hook-pre call
sees a clean floor and lets a consequential tool through as if the untrusted
fetch never happened.

CONTROL-TESTED: this test fails on the pre-fix code (monkeypatch forces
lower_floor to raise once; the session's floor after `ensure_session` comes
back `user` -- clean -- instead of `external`).
"""

import argparse
import io
import json
import sqlite3

import pytest

import memsom
from memsom.bridge import hook as memsom_hook
from memsom.kernel import policy as memsom_policy
from memsom.storage import session as memsom_session


def _default_policy():
    return memsom_policy._normalize(memsom_hook.DEFAULT_HOOK_POLICY)


def test_failed_taint_write_is_reconciled_not_lost(conn, monkeypatch):
    sid = "ms18-session"
    policy = _default_policy()

    real_lower = memsom_session.lower_floor
    calls = {"n": 0}

    def flaky_lower_floor(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_lower(*a, **kw)

    monkeypatch.setattr(memsom_session, "lower_floor", flaky_lower_floor)

    with pytest.raises(sqlite3.OperationalError):
        memsom_hook.apply_post(conn, policy, sid, "WebFetch")

    floor_now = memsom_session.current_floor(conn, sid)
    assert floor_now == memsom.RANK["user"], (
        "precondition: the write really failed and the floor is still clean")

    # The next hook-pre: ensure_session must reconcile the pending taint
    # instead of returning a clean floor.
    floor_after = memsom_session.ensure_session(conn, sid, "user")
    assert floor_after == memsom.RANK["external"], (
        f"a failed hook-post taint write was lost: the next hook-pre saw a "
        f"clean session at floor={floor_after} instead of the external taint "
        f"that WebFetch should have applied")


def test_failed_taint_write_is_logged_to_stderr(conn, monkeypatch, capsys):
    sid = "ms18-session-log"

    def always_fails(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(memsom_hook, "apply_post", always_fails)
    monkeypatch.setattr(
        memsom_hook.sys, "stdin",
        io.StringIO(json.dumps({"session_id": sid, "tool_name": "WebFetch"})))
    monkeypatch.setattr(memsom, "get_connection", lambda: conn)

    memsom_hook.cmd_hook_post(argparse.Namespace())

    captured = capsys.readouterr()
    assert "post error" in captured.err, (
        f"a failed hook-post write produced no stderr signal: {captured.err!r}")


def test_control_a_clean_write_still_taints_normally(conn):
    """GREEN control: proves the reconciliation path did not break the
    ordinary, no-failure case."""
    sid = "ms18-control"
    policy = _default_policy()
    memsom_hook.apply_post(conn, policy, sid, "WebFetch")
    assert memsom_session.current_floor(conn, sid) == memsom.RANK["external"]
