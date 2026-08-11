"""GATES for the highest-severity findings from the federation, redaction and
effects lenses.

CONTROL-TESTED against memsom @ 9d165b1. Every xfail here was run WITHOUT its
marker and FAILED. Reproduce with:

    pytest gates/test_gate_cross_lens.py --runxfail -q

Each `xfail(strict=True)` -> the day the fix lands, pytest reports XPASS ->
FAILED, which is the signal to delete the marker.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import memsom
from memsom import childenv
from memsom.federation import federation as memsom_fed
from memsom.integrity import confid as memsom_confid
from memsom.integrity import corroborate as memsom_corr
from memsom.integrity import redact as memsom_redact
from memsom.interface import mcp as memsom_mcp
from memsom.lifecycle import compact as memsom_compact

MARKER = "PASSPHRASE-9f3a2b"


# ---------------------------------------------------------------------------
# FEDERATION
# ---------------------------------------------------------------------------

def test_delisted_origin_cannot_destroy_local_nodes(conn, monkeypatch):
    """Removing an origin from trusted_origins must be a REVOCATION. Today the
    de-listed origin keeps tombstone/redact/quarantine power over every node
    whose stored origin matches its name."""
    monkeypatch.setenv("MEMDAG_ORIGIN", "STOLEN-LAPTOP")
    memsom_fed.migrate(conn)
    nid = memsom.insert_node(conn, "a memory the stolen laptop owns", "user")
    conn.commit()
    # migrate() does NOT backfill uuid/origin on existing rows -- only
    # backfill_uuids() does (export_changeset calls it internally; a
    # hand-built changeset like this one below has to call it explicitly).
    memsom_fed.backfill_uuids(conn, "STOLEN-LAPTOP")
    conn.commit()
    uuid = conn.execute("SELECT uuid FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    assert uuid, "precondition: node carries a uuid"

    conn.execute("DELETE FROM trusted_origins WHERE origin='STOLEN-LAPTOP'")
    conn.commit()

    changeset = {
        "format": "memsom-changeset-v1", "origin": "STOLEN-LAPTOP",
        "nodes": [{"uuid": uuid, "content": "", "channel": "user", "label": 2,
                   "conf_label": 0, "status": "quarantined",
                   "tombstoned": 1, "tombstoned_at": memsom.now_iso(),
                   "revoke_reason": "pwned", "redacted": 1,
                   "redacted_at": memsom.now_iso(), "redact_reason": "pwned",
                   "created_at": memsom.now_iso(), "origin": "STOLEN-LAPTOP"}],
        "edges": [],
    }
    memsom_fed.import_changeset(conn, changeset)
    row = conn.execute("SELECT tombstoned, redacted, status FROM nodes WHERE id=?",
                       (nid,)).fetchone()
    assert row == (0, 0, "live"), (
        f"a de-listed origin tombstoned/redacted/quarantined a local node: {row}")


@pytest.mark.xfail(strict=True, reason="MS-15 (FED-7): the existing-node merge "
                                       "branch (federation.py:574-655) never "
                                       "merges conf_label")
def test_conf_label_raise_propagates(conn, monkeypatch):
    """Bell-LaPadula max-ceiling must be monotonic across an instance boundary.
    Today a later `classify <id> secret` never reaches the peer."""
    monkeypatch.setenv("MEMDAG_ORIGIN", "PEER-A")
    memsom_fed.migrate(conn)
    nid = memsom.insert_node(conn, f"{MARKER} lighthouse CA backup", "user")
    conn.commit()
    memsom_fed.migrate(conn)
    conn.commit()
    uuid = conn.execute("SELECT uuid FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    memsom_fed.add_trusted_origin(conn, "PEER-B", by="gate") \
        if hasattr(memsom_fed, "add_trusted_origin") else \
        conn.execute("INSERT OR IGNORE INTO trusted_origins(origin) VALUES ('PEER-B')")
    conn.commit()
    cs = {"format": "memsom-changeset-v1", "origin": "PEER-B",
          "nodes": [{"uuid": uuid, "content": f"{MARKER} lighthouse CA backup",
                     "channel": "user", "label": 2, "conf_label": 3,
                     "status": "live", "tombstoned": 0,
                     "created_at": memsom.now_iso(), "origin": "PEER-B"}],
          "edges": []}
    memsom_fed.import_changeset(conn, cs)
    conf = conn.execute("SELECT conf_label FROM nodes WHERE id=?", (nid,)).fetchone()[0]
    assert conf == 3, (
        f"an incoming conf_label RAISE (0 -> 3) was discarded; the peer keeps "
        f"serving the secret at conf_label={conf} to a PUBLIC reader")


@pytest.mark.xfail(strict=True, reason="MS-16 (FED-1): `archived` is in neither "
                                       "_SELECT_COLS nor the import INSERT")
def test_archived_survives_a_federation_roundtrip(conn, monkeypatch):
    monkeypatch.setenv("MEMDAG_ORIGIN", "PEER-A")
    memsom_fed.migrate(conn)
    a = memsom.insert_node(conn, "an episode that gets compacted away", "user")
    conn.commit()
    conn.execute("UPDATE nodes SET archived=1, archived_at=? WHERE id=?",
                 (memsom.now_iso(), a))
    conn.commit()
    memsom_fed.migrate(conn)
    conn.commit()
    cs = memsom_fed.export_changeset(conn)
    node = next(n for n in cs["nodes"]
                if "compacted away" in (n.get("content") or ""))
    assert "archived" in node, (
        "the changeset format carries tombstoned/redacted/status/conf_label but "
        "not archived, so a compacted-away episode lands LIVE and content-intact "
        "on the peer, undoing the consolidation")


# ---------------------------------------------------------------------------
# REDACTION COMPLETENESS
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="MS-17 (RED-01): redact never touches "
                                       "claims/claim_assertions")
def test_redact_reaps_extracted_claims(conn):
    """`extract_claim` deliberately lifts the HIGHEST-value substrings out of a
    document (hashes, IPs, host:port, semver, key=value) into `claims.value`.
    Redaction never reaps them and `claims-list` has no taint filter."""
    memsom_corr.migrate(conn)
    digest = "deadbeef" * 7 + "cafebabe"
    nid = memsom.insert_node(conn, f"The release sha256 is {digest} for build 9.",
                             "user")
    conn.commit()
    triple = memsom_corr.extract_claim(f"sha256 is {digest}")
    assert triple, "precondition: a claim was extractable"
    memsom_corr.register_root(conn, "vendor-a", by="gate")
    memsom_corr.assert_claim(conn, nid, triple, "vendor-a")
    conn.commit()
    memsom_redact.redact_node(conn, nid, "gate", cascade=True)
    conn.commit()
    surviving = json.dumps(memsom_corr.list_claims(conn))
    assert digest not in surviving, (
        "the redacted document's highest-value substring survives verbatim in "
        "claims.value and is printed by `memsom claims-list`")


def test_connection_enables_secure_delete(conn):
    """Without `PRAGMA secure_delete`, `UPDATE nodes SET content=''` leaves the
    original overflow pages on the freelist with their bytes intact. The store
    is Syncthing-replicated, so a redaction is pushed to the peer with those
    pages still readable."""
    val = conn.execute("PRAGMA secure_delete").fetchone()[0]
    assert val, (
        "secure_delete is OFF; redacted plaintext is byte-recoverable from the "
        ".db file until ordinary writes recycle the freed pages")


# ---------------------------------------------------------------------------
# EFFECTS
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="MS-19 (EFF-01): mcp.py:553 appends `node` "
                                       "as a bare positional with str(), no int()")
def test_mcp_obsidian_export_node_arg_cannot_inject_an_option():
    """`node` is declared `nargs="?"` in the CLI, so a value of `--vault=<dir>`
    is parsed as the --vault OPTION and never meets `_checked_vault`. The
    containment root becomes caller-chosen, which paths.py explicitly names as
    the thing it cannot defend."""
    payload = {"node": r"--vault=C:\Users\test\.claude", "query": "x",
               "folder": ".", "title": "poc"}
    try:
        argv = memsom_mcp._tool_argv("obsidian_export", payload)
    except (ValueError, TypeError):
        return   # rejected -> fixed
    assert not any(a.startswith("--vault") for a in argv[:2]), (
        f"the node argument became an option: {argv}")


@pytest.mark.xfail(strict=True, reason="MS-20 (EFF-02): broker.py:176 rebuilds "
                                       "the full os.environ")
def test_broker_upstream_env_excludes_credentials(monkeypatch):
    """The broker's upstream MCP servers are the least-trusted, longest-lived
    children in the tree -- exactly what childenv was written for -- and are the
    one class that explicitly reconstructs os.environ."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI-gate")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_gate")
    from memsom.federation import broker as memsom_broker
    up = memsom_broker.Upstream("x", {"command": sys.executable,
                                      "args": ["-c", "pass"]})
    leaked = [n for n in childenv.CREDENTIAL_ENV_NAMES if n in up.env]
    assert not leaked, f"upstream child inherits denylisted credentials: {leaked}"


# ---------------------------------------------------------------------------
# CONTROLS -- GREEN today, must stay green
# ---------------------------------------------------------------------------

def test_control_childenv_does_block_the_denylist(monkeypatch):
    """Proves MS-20 is a caller gap, not a broken primitive."""
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI-gate")
    env = childenv.child_env()
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_control_untrusted_origin_is_clamped_on_the_new_node_path(conn, monkeypatch):
    """Proves MS-14 is specific to the `owned` branch: the SAME untrusted
    changeset is correctly clamped on the new-node path."""
    monkeypatch.setenv("MEMDAG_ORIGIN", "LOCAL")
    memsom_fed.migrate(conn)
    cs = {"format": "memsom-changeset-v1", "origin": "STRANGER",
          "nodes": [{"uuid": "STRANGER:1", "content": "injected", "channel": "endorsed",
                     "label": 3, "conf_label": 0, "status": "live", "tombstoned": 0,
                     "created_at": memsom.now_iso(), "origin": "STRANGER"}],
          "edges": []}
    memsom_fed.import_changeset(conn, cs)
    row = conn.execute(
        "SELECT channel, label FROM nodes WHERE uuid='STRANGER:1'").fetchone()
    assert row == ("external", 0), f"untrusted clamp did not fire: {row}"


def test_control_check_action_has_no_exception_handler():
    """Proves the enforcement point itself is fail-CLOSED -- the strongest
    property available. MS-13 is about the STALE INPUT it trusts, not about a
    fail-open inside it."""
    import ast
    import inspect
    from memsom.integrity import gate as memsom_gate
    src = inspect.getsource(memsom_gate.check_action)
    tree = ast.parse(src.lstrip())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Try)], (
        "check_action grew an exception handler -- the only enforcement point "
        "must never be able to degrade to allow")
