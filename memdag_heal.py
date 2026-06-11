"""memdag_heal — self-healing: detect-and-report invariant violations + deterministic rebuild.

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

import memdag
import memdag_schema
import memdag_recompute

# Defensive optional imports — heal still runs if either module is absent
try:
    import memdag_confid as memdag_confid
except ImportError:
    memdag_confid = None

try:
    import memdag_redact as memdag_redact
except ImportError:
    memdag_redact = None


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Run migrations for optional modules that may own columns we read.

    We don't own any new columns ourselves — forward to optional modules so their
    columns exist before check() probes them.
    """
    if memdag_redact is not None:
        memdag_redact.migrate(conn)
    if memdag_confid is not None:
        memdag_confid.migrate(conn)


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
    """(b) Live agent-derived node where stored label != recompute_label."""
    rows = conn.execute(
        "SELECT id, label FROM nodes WHERE tombstoned=0 AND channel='agent-derived'"
        " ORDER BY id"
    ).fetchall()
    violations = []
    for nid, stored in rows:
        try:
            expected = memdag_recompute.recompute_label(conn, nid)
        except ValueError:
            continue
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
    """(c) Live derived node where stored conf != max(parent conf) — only if memdag_confid present."""
    if memdag_confid is None:
        return []
    if not memdag_schema.column_exists(conn, "nodes", "conf_label"):
        return []
    # Read-only expectation: max of live immediate parents' conf; keep stored when zero live parents.
    rows = conn.execute(
        "SELECT id, conf_label FROM nodes WHERE tombstoned=0 AND channel='agent-derived'"
        " ORDER BY id"
    ).fetchall()
    violations = []
    for nid, stored_conf in rows:
        # compute expected conf: max of live immediate parents' conf_label
        parent_rows = conn.execute(
            "SELECT n.conf_label FROM edges e JOIN nodes n ON n.id = e.parent"
            " WHERE e.child = ? AND n.tombstoned = 0",
            (nid,)
        ).fetchall()
        if not parent_rows:
            # zero live parents: keep stored — not a violation
            continue
        expected = max(r[0] for r in parent_rows)
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
    if not memdag_schema.column_exists(conn, "nodes", "redacted"):
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
    }

    # (a) dangling edges — report only, NEVER delete (rows and edges always survive)
    summary["dangling_edges_reported"] = len(_check_dangling_edges(conn))

    # (b) integrity labels — recompute_all handles the full graph deterministically
    changes = memdag_recompute.recompute_all(conn)
    summary["integrity_fixed"] = len(changes)

    # (c) conf labels — delegate to memdag_confid if present and column exists
    if memdag_confid is not None and memdag_schema.column_exists(conn, "nodes", "conf_label"):
        conf_changes = memdag_confid.recompute_conf_all(conn)
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
            # Get the reason stored on the seed node so the cascade reason is consistent
            seed_node = memdag.get_node(conn, seed_id)
            # Re-run revoke_cascade: first-death-wins preserves existing tombstone records,
            # only wrongly-live descendants get tombstoned
            n = memdag.revoke_cascade(conn, seed_id, f"cascade from node {seed_id}")
            # revoke_cascade counts newly tombstoned nodes; seed itself is already dead so
            # only the live descendants count
            repaired += n
        summary["cascades_repaired"] = repaired

    # (e) redacted nodes with content — wipe content
    if memdag_schema.column_exists(conn, "nodes", "redacted"):
        with conn:
            cur = conn.execute("UPDATE nodes SET content='' WHERE redacted=1 AND content != ''")
        summary["content_wiped"] = conn.execute("SELECT changes()").fetchone()[0]

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_check(args):
    conn = memdag.get_connection()
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
    conn = memdag.get_connection()
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
    p = argparse.ArgumentParser(prog="memdag_heal")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
