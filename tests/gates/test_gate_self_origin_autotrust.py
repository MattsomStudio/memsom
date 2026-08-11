"""GATE for MS-14 -- self-origin auto-trust must be refuseable.

WRITTEN in Phase 1 (SECURITY-REMEDIATION.md §3.2): `findings.json` names a gate
file (`test_gate_cross_lens.py`) for this finding that carries no matching
assertion -- a dangling pointer, not a mislabel. This is the gate.

CONTROL-TESTED reasoning against memsom @ 9d165b1 (pre-fix): `migrate()`
unconditionally `INSERT OR IGNORE`d `default_origin()` into `trusted_origins`
on EVERY call -- including the one `import_changeset()` makes at its own top
-- so an operator's explicit `DELETE FROM trusted_origins WHERE
origin=default_origin()` was undone by the very next import. There was no
configuration in which federation was default-deny for a changeset claiming
to BE this machine. `test_self_origin_changeset_takes_no_trusted_branch`'s
own precondition assertion (`assert not is_trusted(...)` right after the
first `migrate()` call) reproduces this directly: it FAILS on the pre-fix
tree (migrate re-adds the trust before the assertion runs) and PASSES here.

WHAT A FIX LOOKS LIKE
---------------------
  * `migrate()` no longer auto-inserts `default_origin()` into
    `trusted_origins`.
  * `import_changeset` refuses (treats as untrusted) any changeset whose
    header origin equals `default_origin()`, regardless of the allowlist.
"""

import memsom
from memsom.federation import federation as memsom_fed

MARKER = "PASSPHRASE-9f3a2b"


def test_self_origin_changeset_takes_no_trusted_branch(conn, monkeypatch):
    """With trusted_origins emptied, a changeset whose header origin ==
    default_origin() must take NO trusted branch: no endorsed injection, no
    conf-floor skip, no destroy of an existing local node."""
    monkeypatch.setenv("MEMDAG_ORIGIN", "THIS-MACHINE")
    memsom_fed.migrate(conn)
    # Precondition -- this is the bug itself: migrate() must not have
    # auto-trusted this machine's own origin.
    assert not memsom_fed.is_trusted(conn, "THIS-MACHINE"), (
        "migrate() auto-registered this machine's own origin as trusted -- "
        "self-trust is unremovable if migrate() re-adds it on every call")

    local_nid = memsom.insert_node(conn, "a local memory", "endorsed")
    conn.commit()
    memsom_fed.backfill_uuids(conn, "THIS-MACHINE")
    local_uuid = conn.execute(
        "SELECT uuid FROM nodes WHERE id=?", (local_nid,)).fetchone()[0]

    # A changeset claiming to BE this machine: injects a new endorsed node at
    # conf_label=PUBLIC, and attempts to destroy the local node. The
    # same-origin "owned" check would otherwise pass trivially, since the
    # header origin equals the node's own stored origin.
    changeset = {
        "format": "memsom-changeset-v1", "origin": "THIS-MACHINE",
        "nodes": [
            {"uuid": "THIS-MACHINE:injected", "content": f"{MARKER} injected as self",
             "channel": "endorsed", "label": 3, "conf_label": 0, "status": "live",
             "tombstoned": 0, "created_at": memsom.now_iso(), "origin": "THIS-MACHINE"},
            {"uuid": local_uuid, "content": "", "channel": "endorsed", "label": 3,
             "conf_label": 0, "status": "quarantined", "tombstoned": 1,
             "tombstoned_at": memsom.now_iso(), "revoke_reason": "self-forged",
             "redacted": 1, "redacted_at": memsom.now_iso(),
             "redact_reason": "self-forged", "created_at": memsom.now_iso(),
             "origin": "THIS-MACHINE"},
        ],
        "edges": [],
    }
    memsom_fed.import_changeset(conn, changeset)

    # No endorsed injection at full trust / no conf-floor skip: the injected
    # node must be clamped exactly like any other untrusted origin's node.
    row = conn.execute(
        "SELECT channel, label, conf_label FROM nodes WHERE uuid=?",
        ("THIS-MACHINE:injected",)).fetchone()
    assert row is not None, "the injected node should still be inserted (clamped, not dropped)"
    channel, label, conf_label = row
    assert channel == "external" and label == 0, (
        f"a self-origin changeset injected an ENDORSED node without ever "
        f"crossing the trust boundary: {row}")
    assert conf_label >= memsom_fed.CONF_FLOOR_UNTRUSTED, (
        f"a self-origin changeset skipped the untrusted confidentiality floor: {row}")

    # No destroy: the local node must survive untouched.
    local_row = conn.execute(
        "SELECT tombstoned, redacted, status FROM nodes WHERE id=?",
        (local_nid,)).fetchone()
    assert local_row == (0, 0, "live"), (
        f"a self-origin changeset tombstoned/redacted/quarantined a local node: {local_row}")


def test_migrate_does_not_reinsert_a_deleted_self_trust(conn, monkeypatch):
    """Even if something else once trusted this machine's own name, migrate()
    must never re-add it after an explicit revocation -- the mechanism that
    made the old bug unremovable (migrate() runs at the top of every
    federation call, including import_changeset)."""
    monkeypatch.setenv("MEMDAG_ORIGIN", "RE-TRUST-ME")
    memsom_fed.migrate(conn)
    memsom_fed.register_origin(conn, "RE-TRUST-ME", by="test")
    assert memsom_fed.is_trusted(conn, "RE-TRUST-ME")

    conn.execute("DELETE FROM trusted_origins WHERE origin='RE-TRUST-ME'")
    conn.commit()
    assert not memsom_fed.is_trusted(conn, "RE-TRUST-ME")

    memsom_fed.migrate(conn)
    assert not memsom_fed.is_trusted(conn, "RE-TRUST-ME"), (
        "migrate() re-added self-trust after an explicit revocation")


def test_control_a_different_registered_origin_is_unaffected(conn, monkeypatch):
    """GREEN and must stay green: refusing SELF is specific to self, not a
    blanket refusal of the whole trusted_origins mechanism."""
    monkeypatch.setenv("MEMDAG_ORIGIN", "THIS-MACHINE")
    memsom_fed.migrate(conn)
    memsom_fed.register_origin(conn, "trusted-peer", by="test")

    changeset = {
        "format": "memsom-changeset-v1", "origin": "trusted-peer",
        "nodes": [{"uuid": "trusted-peer:1", "content": "peer content",
                   "channel": "endorsed", "label": 3, "conf_label": 0,
                   "status": "live", "tombstoned": 0,
                   "created_at": memsom.now_iso(), "origin": "trusted-peer"}],
        "edges": [],
    }
    memsom_fed.import_changeset(conn, changeset)
    row = conn.execute(
        "SELECT channel, label FROM nodes WHERE uuid='trusted-peer:1'").fetchone()
    assert row == ("endorsed", 3), (
        f"a genuinely trusted, non-self peer's changeset should NOT be clamped: {row}")
