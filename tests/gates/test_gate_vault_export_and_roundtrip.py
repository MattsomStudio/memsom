"""GATE for MS-05 -- vault export by node id, and the DAG-severing round-trip.

CONTROL-TESTED against memsom @ 9d165b1. Reproduce the raw failures with:

    pytest gates/test_gate_vault_export_and_roundtrip.py --runxfail -q

WHAT A FIX LOOKS LIKE
---------------------
  * `obsidian.export_note`'s node_id branch (obsidian.py:641-647) applies
    `schema.taint_filter_clauses(conn, clearance=parse_conf(clearance))` instead
    of checking `tombstoned` alone, and the MCP `obsidian_export` tool schema
    (mcp.py:246) grows a `clearance` property so the parameter is reachable.
  * A memsom-authored note carries its source node's `conf_label` and quarantine
    status in frontmatter, and `sync_vault` restores them -- or re-links the
    re-ingested note to the original node id rather than minting an orphan.
  * `heal.check` gains an invariant for `channel='agent-derived' AND no parents`
    -- a state `derive_node` itself forbids (memsom/__init__.py:115).
"""

import shutil
import tempfile
from pathlib import Path

import pytest

import memsom
from memsom.bridge import obsidian as memsom_obsidian
from memsom.integrity import confid as memsom_confid
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact
from memsom.lifecycle import heal as memsom_heal

MARKER = "PASSPHRASE-9f3a2b"


@pytest.fixture()
def vault(request):
    p = Path(tempfile.gettempdir()) / "memsom_pentest_gates" / f"v_{request.node.name[:40]}"
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    yield p
    shutil.rmtree(p, ignore_errors=True)


# MS-05 FIXED (Phase 4): the node_id branch now routes through taint_filter_clauses.
def test_export_note_refuses_a_quarantined_node(conn, vault):
    ext = memsom.insert_node(conn, "untrusted forum text about NAT traversal", "external")
    d, _ = memsom.derive_node(conn, "derived from the untrusted forum text", [ext])
    conn.commit()
    memsom_quarantine.consolidate(conn)
    assert conn.execute("SELECT status FROM nodes WHERE id=?",
                        (d,)).fetchone()[0] == "quarantined", "precondition"
    with pytest.raises(ValueError):
        memsom_obsidian.export_note(conn, vault, node_id=d, title="q")


# MS-05 FIXED (Phase 4): clearance is now parsed into the taint_filter_clauses call.
def test_export_note_honours_clearance_on_the_node_id_branch(conn, vault):
    s = memsom.insert_node(conn, f"passphrase {MARKER} topsecret", "user")
    memsom_confid.classify(conn, s, "topsecret")
    conn.commit()
    try:
        p = memsom_obsidian.export_note(conn, vault, node_id=s,
                                        clearance="public", title="ts")
    except ValueError:
        return  # refused -> fixed
    assert MARKER not in p.read_text(encoding="utf-8"), (
        "clearance='public' wrote a topsecret payload to disk")


# MS-06 FIXED (Phase 4): export_note stamps memsom-conf; sync_vault restores it
# on re-ingest for a note that passes the memsom-authored check.
def test_vault_roundtrip_does_not_declassify(conn, vault):
    s = memsom.insert_node(conn, f"passphrase {MARKER} topsecret compartment", "user")
    memsom_confid.classify(conn, s, "topsecret")
    conn.commit()
    memsom_obsidian.export_note(conn, vault, node_id=s, title="ts")
    before = {r[0] for r in conn.execute("SELECT id FROM nodes")}
    memsom_obsidian.sync_vault(conn, vault, default_channel="user")
    conn.commit()
    new = [r for r in conn.execute(
        "SELECT id, conf_label, content FROM nodes") if r[0] not in before]
    carriers = [r for r in new if MARKER in (r[2] or "")]
    assert carriers, "precondition: the round-trip did re-ingest the payload"
    for nid, conf, _c in carriers:
        assert conf == 3, (
            f"round-trip node {nid} carries the topsecret payload at "
            f"conf_label={conf} (0 = PUBLIC): declassification")


# MS-34 FIXED (Phase 4): heal.check() gains _check_parentless_agent_derived.
def test_heal_flags_a_parentless_agent_derived_node(conn):
    conn.execute(
        "INSERT INTO nodes(content, channel, label, created_at) VALUES (?,?,?,?)",
        ("an orphan claiming to be derived", "agent-derived", 1, memsom.now_iso()))
    conn.commit()
    findings = memsom_heal.check(conn)
    assert findings, (
        "heal.check() reports zero findings on a store containing an "
        "agent-derived node with no parents -- a state derive_node itself "
        "forbids (memsom/__init__.py:115), and the exact signature the vault "
        "round-trip produces")


# FIXED (Phase 6): export_note stamps obsidian_path on a fresh single-node
# export, so _purge_backing_files has something to unlink.
def test_redact_purges_a_node_id_export(conn, vault):
    s = memsom.insert_node(conn, f"passphrase {MARKER} export target", "user")
    conn.commit()
    out = memsom_obsidian.export_note(conn, vault, node_id=s, title="export-target")
    conn.commit()
    assert out.exists(), "precondition: the export actually wrote a file"

    stats = {}
    memsom_redact.redact_node(conn, s, "gate", cascade=False, vault=vault,
                              purge_stats=stats)
    assert stats != {"purged": 0, "failed": 0}, (
        "redact reported a clean purge with the exported plaintext still on "
        f"disk: {out}, exists={out.exists()}")
    assert not out.exists(), f"exported plaintext survived redaction: {out}"


def test_control_effective_channel_still_holds(conn):
    """GREEN and must stay green. The frontmatter channel clamp is NOT the leak
    -- min(default, declared) is correct in both directions. This proves the
    round-trip xfails measure the DAG severing, not a broken channel clamp."""
    assert memsom_obsidian.effective_channel("user", "endorsed") == "user"
    assert memsom_obsidian.effective_channel("user", "external") == "external"
    assert memsom_obsidian.effective_channel("user", "nonsense") == "user"
    assert memsom_obsidian.effective_channel("external", "endorsed") == "external"
