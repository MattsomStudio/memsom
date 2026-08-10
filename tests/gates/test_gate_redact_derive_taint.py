"""GATE for MS-03 -- derive_node's liveness check, and the redact cascade's atomicity.

CONTROL-TESTED against memsom @ 9d165b1. Reproduce the raw failures with:

    pytest gates/test_gate_redact_derive_taint.py --runxfail -q

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

import threading

import pytest

import memsom
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact

MARKER = "PASSPHRASE-9f3a2b"
SECRET = f"The lighthouse CA passphrase is {MARKER}, topsecret compartment only."


@pytest.mark.xfail(strict=True, reason="MS-03: derive_node checks tombstoned only")
def test_derive_node_refuses_a_redacted_parent(conn):
    vid = memsom.insert_node(conn, SECRET, "user")
    conn.commit()
    memsom_redact.redact_node(conn, vid, "gate", cascade=True)
    with pytest.raises(ValueError):
        memsom.derive_node(conn, "summary of the redacted node", [vid])


@pytest.mark.xfail(strict=True, reason="MS-03: derive_node checks tombstoned only")
def test_derive_node_refuses_a_quarantined_parent(conn):
    ext = memsom.insert_node(conn, "untrusted forum text about lighthouses", "external")
    d, _ = memsom.derive_node(conn, "derived from the untrusted forum text", [ext])
    conn.commit()
    memsom_quarantine.consolidate(conn)
    assert conn.execute("SELECT status FROM nodes WHERE id=?",
                        (d,)).fetchone()[0] == "quarantined", "precondition"
    with pytest.raises(ValueError):
        memsom.derive_node(conn, "second hop off the quarantined node", [d])


@pytest.mark.xfail(strict=True, reason="MS-03: derive_node checks tombstoned only")
def test_derive_node_refuses_an_archived_parent(conn):
    a = memsom.insert_node(conn, "an episode that gets compacted away", "user")
    conn.commit()
    conn.execute("UPDATE nodes SET archived=1, archived_at=? WHERE id=?",
                 (memsom.now_iso(), a))
    conn.commit()
    with pytest.raises(ValueError):
        memsom.derive_node(conn, "summary chaining off the archived episode", [a])


@pytest.mark.xfail(strict=True, reason="MS-04: redact_node reads cascade_set "
                                       "outside any transaction (redact.py:181)")
def test_redact_cascade_is_atomic_against_a_concurrent_derive(conn):
    """Forced interleave. MEASURED natural (uninstrumented) hit rate at
    9d165b1: 29/40 -- this is not a narrow window."""
    victim = memsom.insert_node(conn, SECRET, "user")
    conn.commit()

    in_window = threading.Event()
    derive_done = threading.Event()
    state = {}
    real = memsom.cascade_set

    def instrumented(c, seed):
        rows = real(c, seed)
        in_window.set()          # redact.py:181 has returned; redact.py:187 has
        derive_done.wait(10)     # not yet opened its deferred transaction
        return rows

    def racer():
        in_window.wait(10)
        c2 = memsom.get_connection()
        try:
            state["child"] = memsom.derive_node(c2, f"- {SECRET}", [victim])[0]
        except Exception as exc:            # noqa: BLE001
            state["error"] = repr(exc)
        finally:
            c2.close()
            derive_done.set()

    t = threading.Thread(target=racer, daemon=True)
    t.start()
    memsom.cascade_set = instrumented
    try:
        memsom_redact.redact_node(conn, victim, "gate", cascade=True)
    finally:
        memsom.cascade_set = real
    t.join(15)

    child = state.get("child")
    assert child is not None, f"racer failed to derive: {state.get('error')}"
    red, content = conn.execute(
        "SELECT redacted, content FROM nodes WHERE id=?", (child,)).fetchone()
    assert red == 1 or MARKER not in (content or ""), (
        f"node {child} was derived inside the redact window and escaped the "
        f"cascade: the destroyed payload survives verbatim in a live, "
        f"un-redacted descendant")


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
