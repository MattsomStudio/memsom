"""GATES for MS-20 and MS-21 -- the bridge importer's own write path.

WRITTEN in Phase 4 (SECURITY-REMEDIATION.md Sec3.2): neither had a gate
before Phase 4.

MS-20 -- frontmatter `type:` dictates the trust channel
---------------------------------------------------------
CONTROL-TESTED reasoning against memsom @ 9d165b1: `import_memory_dir` reads
`channel = CHANNEL_BY_TYPE.get(memory_type(stem, fm), DEFAULT_CHANNEL)` from
the file's OWN frontmatter, then called `memsom.insert_node` directly -- the
one write path's F-13 channel-ceiling guard was never in the call graph. A
`type: feedback` (or `personal`/`user`) file landed `endorsed`, which is
PINNED in the digest and permanently loaded into every session, regardless of
`MEMDAG_CHANNEL_CEILING`.

MS-21 -- `depends_on:` mints raw edges without Biba re-derivation
-------------------------------------------------------------------
CONTROL-TESTED reasoning: `relate_fact_deps` issued a bare
`INSERT OR IGNORE INTO edges(child, parent)` into the SAME table
CASCADE_CTE walks -- the table derive_node's own `label = min(parents)` clamp
also writes through -- without ever recomputing the child's label.

GUESS FLAGGED IN THE PLAN, RESOLVED HERE: PLAN.md's own MS-21 row warns that
routing the edge through a full Biba floor recompute could "quietly demote or
unpin" a fact-protocol memory, because `bridge_import.CHANNEL_BY_TYPE` stamps
every file-imported node (every `fact_*.md` included) at a FIXED,
channel-authoritative label -- never `agent-derived`. `recompute_node`
(memsom.integrity.recompute) already encodes exactly the right boundary: it
is a no-op fixed point for any non-`agent-derived` node by design, and only
re-floors a node whose label is MEANT to be computed. Routing through it is
therefore Biba-aware where that is meaningful and correctly inert on a
fact-protocol child -- the second test below is the control proving the
protocol is not the collision the plan worried about.
"""

import pytest

import memsom
from memsom.bridge import bridge_import as bi
from memsom.integrity import ingest as memsom_ingest


def test_channel_ceiling_bounds_a_memory_files_own_type(conn, tmp_path, monkeypatch):
    bi.migrate(conn)
    mem = tmp_path / "memdir"
    mem.mkdir()
    (mem / "feedback_secret.md").write_text(
        "---\nname: s\ndescription: d\n---\n\nAn attacker-controlled memory file.\n",
        encoding="utf-8")
    (mem / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")

    monkeypatch.setenv(memsom_ingest.CHANNEL_CEILING_ENV, "user")
    try:
        with pytest.raises(ValueError):
            bi.import_memory_dir(conn, mem, dry_run=False)
    finally:
        monkeypatch.delenv(memsom_ingest.CHANNEL_CEILING_ENV, raising=False)

    # Nothing landed above the ceiling: the file's own claimed channel
    # (feedback -> endorsed) must never reach the DB when refused.
    row = conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE source_ref = 'memory:feedback_secret'"
    ).fetchone()
    assert row[0] == 0, "a refused channel must not partially land"


def test_channel_ceiling_permissive_by_default(conn, tmp_path):
    """GREEN control: unset ceiling still allows endorsed, proving the refusal
    above is the ceiling firing, not a broken import."""
    bi.migrate(conn)
    mem = tmp_path / "memdir2"
    mem.mkdir()
    (mem / "feedback_secret.md").write_text(
        "---\nname: s\ndescription: d\n---\n\nAn attacker-controlled memory file.\n",
        encoding="utf-8")
    (mem / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")

    bi.import_memory_dir(conn, mem, dry_run=False)
    row = conn.execute(
        "SELECT channel FROM nodes WHERE source_ref = 'memory:feedback_secret'"
    ).fetchone()
    assert row == ("endorsed",)


def _write_fact_pair(mem):
    (mem / "fact_secret.md").write_text(
        "---\nname: s\ndescription: d\ntype: fact\n---\n\n"
        "An externally-sourced measurement.\n", encoding="utf-8")
    (mem / "fact_dependent.md").write_text(
        "---\nname: d\ndescription: d\ntype: fact\ndepends_on: fact_secret\n---\n\n"
        "A derived measurement.\n", encoding="utf-8")
    (mem / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")


def test_depends_on_edge_routes_through_recompute(conn, tmp_path, monkeypatch):
    """The raw edge insert is no longer un-recomputed: recompute_node fires
    for the child on every materialised depends_on edge."""
    bi.migrate(conn)
    mem = tmp_path / "memdir3"
    mem.mkdir()
    _write_fact_pair(mem)
    bi.import_memory_dir(conn, mem, dry_run=False)

    from memsom.integrity import recompute as memsom_recompute
    calls = []
    orig = memsom_recompute.recompute_node

    def _spy(conn_, nid):
        calls.append(nid)
        return orig(conn_, nid)

    monkeypatch.setattr(memsom_recompute, "recompute_node", _spy)
    bi.relate_fact_deps(conn, mem, dry_run=False)

    dep = conn.execute(
        "SELECT id FROM nodes WHERE source_ref = 'memory:fact_dependent'").fetchone()[0]
    assert dep in calls, (
        "materialising a depends_on edge must call recompute_node(child) -- "
        "MS-21: the raw INSERT was never followed by any Biba re-derivation")


def test_depends_on_does_not_demote_the_fact_protocol(conn, tmp_path):
    """CONTROL: the exact collision PLAN.md flags as a risk must NOT happen --
    a fact-protocol (channel=user) child keeps its channel-authoritative label
    even after its depends_on parent is forced to a lower label."""
    bi.migrate(conn)
    mem = tmp_path / "memdir4"
    mem.mkdir()
    _write_fact_pair(mem)
    bi.import_all(conn, mem, dry_run=False)

    dep = conn.execute(
        "SELECT id FROM nodes WHERE source_ref = 'memory:fact_dependent'").fetchone()[0]
    secret = conn.execute(
        "SELECT id FROM nodes WHERE source_ref = 'memory:fact_secret'").fetchone()[0]
    dep_before = conn.execute("SELECT label FROM nodes WHERE id=?", (dep,)).fetchone()[0]
    assert dep_before == memsom.RANK["user"], "precondition: fact lands at channel=user"

    conn.execute("UPDATE nodes SET label = 0 WHERE id = ?", (secret,))
    conn.commit()
    bi.relate_fact_deps(conn, mem, dry_run=False)

    dep_after = conn.execute("SELECT label FROM nodes WHERE id=?", (dep,)).fetchone()[0]
    assert dep_after == dep_before, (
        f"a fact-protocol child's channel-authoritative label changed "
        f"({dep_before} -> {dep_after}) -- this is the demotion PLAN.md's "
        f"MS-21 row explicitly warned a naive Biba clamp would cause")
