"""GATE for MS-03 -- derive_node's liveness check, and the redact cascade's atomicity.

CONTROL-TESTED against memsom @ 9d165b1. Reproduce the raw failures with:

    pytest gates/test_gate_redact_derive_taint.py --runxfail -q

MS-06 FIXED, Phase 1: see the atomicity test's own docstring below for why its
harness had to change shape along with the fix (it does not just flip a marker).

WHAT A FIX LOOKS LIKE
---------------------
  * `derive_node` (memsom/__init__.py:113) widens its liveness predicate from
    `tombstoned` to the full taint set -- ideally by calling
    `schema.taint_filter_clauses` rather than re-listing columns, so the next
    taint dimension is inherited automatically.
  * `redact.redact_node` (memsom/integrity/redact.py:181) takes `BEGIN
    IMMEDIATE` BEFORE `cascade_set`, so the target set and the writes are one
    atomic unit -- the same protection `derive_node` already has against the
    revoke cascade (memsom/__init__.py:117-118).
"""

import sys
import threading
from pathlib import Path

import pytest

import memsom
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from _meta.tools import interleave as memsom_interleave

MARKER = "PASSPHRASE-9f3a2b"
SECRET = f"The lighthouse CA passphrase is {MARKER}, topsecret compartment only."


def test_derive_node_refuses_a_redacted_parent(conn):
    vid = memsom.insert_node(conn, SECRET, "user")
    conn.commit()
    memsom_redact.redact_node(conn, vid, "gate", cascade=True)
    with pytest.raises(ValueError):
        memsom.derive_node(conn, "summary of the redacted node", [vid])


def test_derive_node_refuses_a_quarantined_parent(conn):
    ext = memsom.insert_node(conn, "untrusted forum text about lighthouses", "external")
    d, _ = memsom.derive_node(conn, "derived from the untrusted forum text", [ext])
    conn.commit()
    memsom_quarantine.consolidate(conn)
    assert conn.execute("SELECT status FROM nodes WHERE id=?",
                        (d,)).fetchone()[0] == "quarantined", "precondition"
    with pytest.raises(ValueError):
        memsom.derive_node(conn, "second hop off the quarantined node", [d])


def test_derive_node_refuses_an_archived_parent(conn):
    a = memsom.insert_node(conn, "an episode that gets compacted away", "user")
    conn.commit()
    conn.execute("UPDATE nodes SET archived=1, archived_at=? WHERE id=?",
                 (memsom.now_iso(), a))
    conn.commit()
    with pytest.raises(ValueError):
        memsom.derive_node(conn, "summary chaining off the archived episode", [a])


def test_redact_cascade_is_atomic_against_a_concurrent_derive(conn):
    """MS-06 FIXED: redact_node now takes BEGIN IMMEDIATE BEFORE cascade_set, so
    the target set and the writes are one atomic unit -- there is no longer a
    gap between reading the cascade and writing it for a monkeypatched pause to
    sit in (that would just deadlock the racer against SQLite's own write lock,
    since cascade_set now runs WHILE redact already holds it).

    So this harness switched shape along with the fix: naturally-scheduled
    concurrent trials via the shared interleave harness (the same method that
    MEASURED the pre-fix escape rate at 29/40), asserting zero hits across N
    trials post-fix. Two outcomes are both correct now:
      (a) the racer wins the lock first and mints its child BEFORE redact's
          (now lock-protected) cascade_set runs -- cascade_set sees that fresh
          child in the live DAG and redacts it too, or
      (b) redact wins the lock first -- the racer blocks on SQLite's own lock,
          and once redact commits, MS-05's widened liveness check refuses the
          now-redacted victim as a parent, so no child is ever minted.
    Either way the secret cannot survive live and un-redacted.
    """
    def setup_fn():
        # sqlite3 connections are thread-affine (check_same_thread default) --
        # each function below opens its OWN connection inside its OWN thread,
        # never sharing one across threads.
        c = memsom.get_connection()
        victim = memsom.insert_node(c, SECRET, "user")
        c.commit()
        c.close()
        return {"victim": victim, "child": None}

    def victim_fn(ctx, barrier):
        barrier.wait(5)
        c = memsom.get_connection()
        try:
            memsom_redact.redact_node(c, ctx["victim"], "gate", cascade=True)
        finally:
            c.close()

    def racer_fn(ctx, barrier):
        barrier.wait(5)
        c2 = memsom.get_connection()
        try:
            nid, _ = memsom.derive_node(c2, f"- {SECRET}", [ctx["victim"]])
            ctx["child"] = nid
        except ValueError:
            pass  # refused as a redacted parent -- the correct closed outcome
        finally:
            c2.close()

    def detect_fn(ctx):
        child = ctx.get("child")
        if child is None:
            return False
        c = memsom.get_connection()
        try:
            row = c.execute(
                "SELECT redacted, content FROM nodes WHERE id=?", (child,)).fetchone()
        finally:
            c.close()
        if row is None:
            return False
        red, content = row
        return red != 1 and MARKER in (content or "")

    result = memsom_interleave.natural_rate(20, setup_fn, victim_fn, racer_fn, detect_fn)
    assert result.hits == 0, (
        f"secret escaped the redact cascade in {result.hits}/{result.trials} "
        f"trials -- a node derived concurrently with redact survives live and "
        f"un-redacted: {result.errors}")


def test_control_derive_node_does_refuse_a_tombstoned_parent(conn):
    """GREEN and must stay green. The liveness check is not absent -- it is
    one-dimensional. This proves the xfails above measure a MISSING dimension,
    not a missing check."""
    vid = memsom.insert_node(conn, SECRET, "user")
    conn.commit()
    memsom.revoke_cascade(conn, vid, "control")
    conn.commit()
    with pytest.raises(ValueError, match="tombstoned parent"):
        memsom.derive_node(conn, "summary of the tombstoned node", [vid])
