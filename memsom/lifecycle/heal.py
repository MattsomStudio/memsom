"""memsom_heal — self-healing: detect-and-report invariant violations + deterministic rebuild.

Source nodes are the source-of-truth and are NEVER modified.  Derived state is
recomputed from them.  No LLM remediation, ever.

Public API
----------
check(conn)          -> list[dict]  (each violation has 'kind', 'detail', plus 'node' or 'edge')
rebuild_derived(conn)-> dict        summary {'integrity_fixed', 'conf_fixed',
                                             'cascades_repaired', 'content_wiped',
                                             'dangling_edges_reported'}

CLI
---
check                        prints violations, exits 0 (OK) or 1 (violations found)
rebuild-derived [--yes]      dry-run default; --yes applies fixes

register(subparsers) mounts this into a unified CLI.
main(argv=None) standalone entry point.
"""

import sys
import argparse

import memsom
from memsom.storage import schema as memsom_schema
from memsom.integrity import recompute as memsom_recompute

# Defensive optional imports — heal still runs if either module is absent
try:
    from memsom.integrity import confid as memsom_confid
except ImportError:
    memsom_confid = None

try:
    from memsom.integrity import redact as memsom_redact
except ImportError:
    memsom_redact = None

try:
    from memsom.retrieval import retrieve as memsom_retrieve
except ImportError:
    memsom_retrieve = None


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Run migrations for optional modules that may own columns we read.

    We don't own any new columns ourselves — forward to optional modules so their
    columns exist before check() probes them.
    """
    if memsom_redact is not None:
        memsom_redact.migrate(conn)
    if memsom_confid is not None:
        memsom_confid.migrate(conn)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_dangling_edges(conn):
    """(a) Edges whose child or parent has no nodes row."""
    rows = conn.execute(
        "SELECT e.child, e.parent FROM edges e"
        " LEFT JOIN nodes c ON c.id = e.child"
        " LEFT JOIN nodes p ON p.id = e.parent"
        " WHERE c.id IS NULL OR p.id IS NULL"
        " ORDER BY e.child, e.parent"
    ).fetchall()
    violations = []
    for child, parent in rows:
        violations.append({
            "kind": "dangling-edge",
            "detail": f"edge child={child} parent={parent} has no nodes row",
            "edge": (child, parent),
        })
    return violations


def _check_integrity_mismatch(conn):
    """(b) Live agent-derived node where stored label != effective label.

    Uses memsom_recompute.effective_labels — ONE shared-memo bulk pass
    (O(V+E)) instead of a fresh full DFS per node, and the exact same
    computation recompute_all() uses, so this check flags a row iff
    rebuild_derived()'s recompute step would change it.  Elevation fixed
    points are skipped by effective_labels (they can never mismatch).
    """
    violations = []
    for nid, stored, expected in memsom_recompute.effective_labels(conn):
        if expected != stored:
            violations.append({
                "kind": "integrity-mismatch",
                "detail": f"node {nid}: stored label={stored}, expected={expected}",
                "node": nid,
                "expected": expected,
                "actual": stored,
            })
    return violations


def _check_conf_mismatch(conn):
    """(c) Live derived node where stored conf != EFFECTIVE conf — only if memsom_confid present.

    HEAL-1: the old check compared against the immediate parents' STORED conf
    (single level), so a multi-hop laundered chain a->d1->d2 (both d1 and d2
    raw-downgraded) was missed — d2's parent d1 now stores 0, so single-level
    expected==stored==0 and check() reported clean while d2 was served below
    clearance. memsom_confid.effective_confs computes the transitive high-water
    max (the same value recompute_conf_all would write), mirroring the integrity
    check's use of effective_labels, so check() now flags every row rebuild fixes.
    """
    if memsom_confid is None:
        return []
    if not memsom_schema.column_exists(conn, "nodes", "conf_label"):
        return []
    eff = memsom_confid.effective_confs(conn)
    # tombstoned=0 dimension sourced from the ONE taint-filter primitive
    # (storage.schema.taint_filter_clauses) rather than a hand-rolled literal.
    tomb_clause = memsom_schema.taint_filter_clauses(conn)[0][0]
    rows = conn.execute(
        f"SELECT id, conf_label FROM nodes WHERE {tomb_clause} AND channel='agent-derived'"
        " ORDER BY id"
    ).fetchall()
    violations = []
    for nid, stored_conf in rows:
        expected = eff.get(nid, stored_conf)
        if expected != stored_conf:
            violations.append({
                "kind": "conf-mismatch",
                "detail": f"node {nid}: stored conf={stored_conf}, expected={expected}",
                "node": nid,
                "expected": expected,
                "actual": stored_conf,
            })
    return violations


def _check_live_child_of_tombstoned(conn):
    """(d) Live node having any tombstoned IMMEDIATE parent (cascade should have caught it)."""
    rows = conn.execute(
        "SELECT DISTINCT e.child FROM edges e"
        " JOIN nodes p ON p.id = e.parent"
        " JOIN nodes c ON c.id = e.child"
        " WHERE p.tombstoned = 1 AND c.tombstoned = 0"
        " ORDER BY e.child"
    ).fetchall()
    violations = []
    for (nid,) in rows:
        violations.append({
            "kind": "live-child-of-tombstoned",
            "detail": f"node {nid} is live but has a tombstoned immediate parent",
            "node": nid,
        })
    return violations


def _check_redacted_with_content(conn):
    """(e) redacted=1 AND content != '' — only if redacted column exists."""
    if not memsom_schema.column_exists(conn, "nodes", "redacted"):
        return []
    rows = conn.execute(
        "SELECT id FROM nodes WHERE redacted = 1 AND content != ''"
        " ORDER BY id"
    ).fetchall()
    violations = []
    for (nid,) in rows:
        violations.append({
            "kind": "redacted-with-content",
            "detail": f"node {nid} is redacted but still has content",
            "node": nid,
        })
    return violations


def _check_parentless_agent_derived(conn):
    """(f) A live channel='agent-derived' node with ZERO parent edges.

    derive_node itself forbids this (raises ValueError on an empty parent_ids
    list) -- it can only be reached by a path that bypasses derive_node
    entirely, which is exactly the signature MS-08's vault round-trip
    produces: a re-ingested memsom-authored note lands as a plain SOURCE row
    (via ingest_text) but keeps the exported channel='agent-derived' stamp.
    """
    rows = conn.execute(
        "SELECT n.id FROM nodes n"
        " LEFT JOIN edges e ON e.child = n.id"
        " WHERE n.tombstoned = 0 AND n.channel = 'agent-derived' AND e.child IS NULL"
        " ORDER BY n.id"
    ).fetchall()
    return [{
        "kind": "parentless-agent-derived",
        "detail": f"node {nid}: channel=agent-derived with zero parents -- a "
                  f"state derive_node itself forbids",
        "node": nid,
    } for (nid,) in rows]


def _check_unindexed_sources(conn):
    """(g) A live, non-derived, non-redacted, non-archived source with NO
    postings row -- MS-31: index_node's failure used to be a try-wrapped
    upward import caught by a bare `except`, so a node could be fully live
    and answerable via the enhanced pool / compose() while retrieve()
    silently never surfaced it -- "no results" for a question the store CAN
    answer. Flagged here independent of whether the indexing call ever ran,
    so a failure is visible even when nothing happened to be watching.
    """
    if memsom_retrieve is None or not memsom_schema.table_exists(conn, "docstats"):
        return []
    clauses = ["tombstoned = 0", "channel != 'agent-derived'"]
    for col in ("redacted", "archived"):
        if memsom_schema.column_exists(conn, "nodes", col):
            clauses.append(f"COALESCE({col}, 0) = 0")
    rows = conn.execute(
        "SELECT id FROM nodes WHERE " + " AND ".join(clauses) +
        " AND id NOT IN (SELECT node_id FROM docstats) ORDER BY id"
    ).fetchall()
    return [{
        "kind": "unindexed-source",
        "detail": f"node {nid}: live source has no postings row -- invisible to retrieve()",
        "node": nid,
    } for (nid,) in rows]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check(conn):
    """Return a list of violation dicts, deterministic order: kinds a-e, then node/edge id.

    Each dict has at minimum {'kind', 'detail'} plus either 'node' (int) or 'edge' (tuple).
    """
    violations = []
    violations.extend(_check_dangling_edges(conn))          # (a)
    violations.extend(_check_integrity_mismatch(conn))      # (b)
    violations.extend(_check_conf_mismatch(conn))           # (c)
    violations.extend(_check_live_child_of_tombstoned(conn))# (d)
    violations.extend(_check_redacted_with_content(conn))   # (e)
    violations.extend(_check_parentless_agent_derived(conn))# (f)
    violations.extend(_check_unindexed_sources(conn))       # (g)
    return violations


def rebuild_derived(conn):
    """Deterministic rebuild of derived state.  Source nodes are NEVER modified.

    Returns a summary dict:
      integrity_fixed      int  — count of nodes whose label was corrected  (b)
      conf_fixed           int  — count of nodes whose conf was corrected    (c)
      cascades_repaired    int  — count of live children of tombstoned nodes now tombstoned (d)
      content_wiped        int  — count of redacted nodes whose content was cleared  (e)
      dangling_edges_reported int — count of dangling edges (reported only, not deleted)  (a)
    """
    summary = {
        "integrity_fixed": 0,
        "conf_fixed": 0,
        "cascades_repaired": 0,
        "content_wiped": 0,
        "dangling_edges_reported": 0,
        "reindexed": 0,
    }

    # (a) dangling edges — report only, NEVER delete (rows and edges always survive)
    summary["dangling_edges_reported"] = len(_check_dangling_edges(conn))

    # (b) integrity labels — recompute_all handles the full graph deterministically
    changes = memsom_recompute.recompute_all(conn)
    summary["integrity_fixed"] = len(changes)

    # (c) conf labels — delegate to memsom_confid if present and column exists
    if memsom_confid is not None and memsom_schema.column_exists(conn, "nodes", "conf_label"):
        conf_changes = memsom_confid.recompute_conf_all(conn)
        summary["conf_fixed"] = len(conf_changes) if conf_changes is not None else 0

    # (d) live children of tombstoned parents — re-run cascade for each tombstoned seed
    #     that still has live descendants (first-death-wins: already-dead rows keep their record)
    live_child_violations = _check_live_child_of_tombstoned(conn)
    if live_child_violations:
        # Find the tombstoned parents of these live children (one level up only for the seed)
        # We need to find tombstoned seeds: any tombstoned node that has live descendants
        # Strategy: for each live child with a tombstoned immediate parent, find the tombstoned
        # parents, then re-run revoke_cascade on each tombstoned parent that is itself not a
        # descendant of another tombstoned node driving this (to avoid double-counting).
        # Simpler and correct: collect ALL tombstoned nodes that have live descendants;
        # run revoke_cascade on each (first-death-wins means already-dead rows untouched).
        tombstoned_seeds = set()
        for v in live_child_violations:
            child_id = v["node"]
            parent_rows = conn.execute(
                "SELECT p.id FROM edges e JOIN nodes p ON p.id = e.parent"
                " WHERE e.child = ? AND p.tombstoned = 1",
                (child_id,)
            ).fetchall()
            for (pid,) in parent_rows:
                tombstoned_seeds.add(pid)

        repaired = 0
        for seed_id in sorted(tombstoned_seeds):
            # Re-run revoke_cascade: first-death-wins preserves existing tombstone records,
            # only wrongly-live descendants get tombstoned
            n = memsom.revoke_cascade(conn, seed_id, f"cascade from node {seed_id}")
            # revoke_cascade counts newly tombstoned nodes; seed itself is already dead so
            # only the live descendants count
            repaired += n
        summary["cascades_repaired"] = repaired

    # (e) redacted nodes with content — wipe content
    if memsom_schema.column_exists(conn, "nodes", "redacted"):
        with conn:
            cur = conn.execute("UPDATE nodes SET content='' WHERE redacted=1 AND content != ''")
        summary["content_wiped"] = conn.execute("SELECT changes()").fetchone()[0]

    # (g) MS-31: a live source with no postings row -- either indexed or
    # reported; this is the "indexed" half. index_node itself decides skip
    # vs index (redacted/archived/tombstoned all already excluded above).
    for v in _check_unindexed_sources(conn):
        if memsom_retrieve.index_node(conn, v["node"]):
            summary["reindexed"] += 1

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_check(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        violations = check(conn)
        for v in violations:
            print(v["detail"])
        if not violations:
            print("OK - no violations")
        else:
            print(f"{len(violations)} violation(s)")
            sys.exit(1)
    finally:
        conn.close()


def cmd_rebuild_derived(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        violations = check(conn)
        if not violations:
            print("OK - no violations found; nothing to rebuild.")
        else:
            for v in violations:
                print(v["detail"])
            print(f"{len(violations)} violation(s) found.")

        if not args.yes:
            print("dry run - re-run with --yes to apply.")
            return

        summary = rebuild_derived(conn)
        print(summary)

        # Re-run check and print residuals
        residuals = check(conn)
        if not residuals:
            print("OK - no violations remain")
        else:
            print(f"{len(residuals)} residual violation(s) (dangling edges remain by design):")
            for v in residuals:
                print(v["detail"])
    finally:
        conn.close()


def register(subparsers):
    p_check = subparsers.add_parser("check",
                                     help="check invariant violations")
    p_check.set_defaults(func=cmd_check)

    p_rebuild = subparsers.add_parser("rebuild-derived",
                                       help="deterministic rebuild of derived state")
    p_rebuild.add_argument("--yes", action="store_true",
                           help="apply fixes (default is dry run)")
    p_rebuild.set_defaults(func=cmd_rebuild_derived)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_heal")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
