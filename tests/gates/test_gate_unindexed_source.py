"""GATE for MS-31 -- an ingested node invisible to every retrieve().

CONTROL-TESTED reasoning against memsom @ 9d165b1: `interface/ingest.py`'s
`_try_index` swallowed everything `index_node` raised (`try: ... except
Exception: pass`), and `index_node` writes postings in its own transaction
AFTER `ingest_text`'s own commit. A node whose indexing failed was fully
live, answerable via the pool-based `ask`, yet `retrieve()`/`ask --retrieve`
returned nothing for a question the store CAN answer, and nothing in the
codebase ever reported the gap.

WRITTEN in Phase 4 (SECURITY-REMEDIATION.md Sec3.2): no gate existed for this
finding before Phase 4.

WHAT A FIX LOOKS LIKE (PLAN.md Sec2.3 -- "never a third option")
--------------------------------------------------------------------
The write path emits "node_ingested" (kernel.events, which never swallows)
instead of a try-wrapped upward import. `lifecycle.heal.check()` gains an
`unindexed-source` invariant that is independent of whether that event ever
had a subscriber, so a node whose indexing failed is either INDEXED (the
event's subscriber succeeds) or REPORTED (heal flags it) -- never silently
neither. `rebuild_derived` repairs it.
"""

import memsom
from memsom.integrity import ingest as memsom_ingest
from memsom.lifecycle import heal as memsom_heal
from memsom.retrieval import retrieve as memsom_retrieve


def test_a_node_whose_indexing_failed_is_reported_by_heal(conn, monkeypatch):
    """Force index_node itself to fail (the real subscriber's own call) and
    assert heal.check() flags the resulting gap."""
    monkeypatch.setattr(memsom_retrieve, "index_node",
                        lambda conn_, nid: (_ for _ in ()).throw(RuntimeError("index down")))

    ids = memsom_ingest.ingest_text(conn, "a fact retrieve must be able to find", "user")
    conn.commit()
    assert len(ids) == 1, "precondition: the node was still minted despite the failure"
    nid = ids[0]

    findings = [v for v in memsom_heal.check(conn) if v.get("node") == nid]
    assert findings and findings[0]["kind"] == "unindexed-source", (
        f"heal.check() did not flag node {nid} as unindexed after index_node "
        f"raised -- MS-31 is exactly this silence")


def test_rebuild_derived_repairs_an_unindexed_source(conn, monkeypatch):
    """The 'indexed' half of "either indexed or reported": rebuild_derived
    must close the gap heal.check() flags."""
    monkeypatch.setattr(memsom_retrieve, "index_node",
                        lambda conn_, nid: (_ for _ in ()).throw(RuntimeError("index down")))
    ids = memsom_ingest.ingest_text(conn, "another fact retrieve must find", "user")
    conn.commit()
    nid = ids[0]
    assert conn.execute(
        "SELECT COUNT(*) FROM docstats WHERE node_id=?", (nid,)
    ).fetchone()[0] == 0, "precondition: never indexed"
    monkeypatch.undo()

    summary = memsom_heal.rebuild_derived(conn)
    assert summary["reindexed"] >= 1

    assert conn.execute(
        "SELECT COUNT(*) FROM docstats WHERE node_id=?", (nid,)
    ).fetchone()[0] == 1, "rebuild_derived must leave the node indexed"

    residual = [v for v in memsom_heal.check(conn) if v.get("node") == nid]
    assert not residual, "heal.check() must be clean after the repair"

    results = memsom_retrieve.retrieve(conn, "another fact retrieve must find")
    assert any(r[0] == nid for r in results), (
        "the repaired node must actually be retrievable, not just indexed in name")


def test_control_a_cleanly_indexed_node_is_not_flagged(conn):
    """GREEN control: normal ingest (no forced failure) never trips this check."""
    ids = memsom_ingest.ingest_text(conn, "a normally-indexed fact", "user")
    conn.commit()
    nid = ids[0]
    assert conn.execute(
        "SELECT COUNT(*) FROM docstats WHERE node_id=?", (nid,)
    ).fetchone()[0] == 1, "precondition: the normal event subscriber indexed it"
    findings = [v for v in memsom_heal.check(conn) if v.get("node") == nid]
    assert not findings
