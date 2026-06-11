"""memdag_blame — "git blame" for the derivation DAG.

Traces any claim or answer back to its ROOT sources: the non-derived nodes it
ultimately came from, each with channel, integrity, ref, and state.

History is immutable; tombstoned or redacted ancestors are INCLUDED with their
state — blame is a history tool.

Public API
----------
migrate(conn)                         — ensure redact + quarantine columns exist
blame(conn, nid) -> list[dict]        — trace nid to root sources; ValueError unknown
format_blame(conn, nid) -> list[str]  — human-readable lines (no print/exit)

CLI
---
blame <id>        print format_blame output; unknown id -> stderr + exit 1
register(subparsers)
main(argv=None)
"""

import sys

import memdag
import memdag_schema
import memdag_redact
import memdag_quarantine


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Ensure redact and quarantine state columns exist. Idempotent."""
    memdag_redact.migrate(conn)
    memdag_quarantine.migrate(conn)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _state_str(tombstoned, redacted, status):
    """Compose a state string from the three independent flags/columns.

    Possible values:
      'live', 'tombstoned', 'redacted', 'tombstoned+redacted', 'quarantined'
    tombstoned/redacted take priority; quarantined is reported only when
    neither of the other flags is set (a quarantined node is typically live).
    """
    parts = []
    if tombstoned:
        parts.append("tombstoned")
    if redacted:
        parts.append("redacted")
    if parts:
        return "+".join(parts)
    if status == "quarantined":
        return "quarantined"
    return "live"


def _fetch_full(conn, nid):
    """Return a dict of blame-relevant columns for node *nid*, or None if not found.

    Gracefully handles databases where 'redacted' and/or 'status' columns have
    not been added yet (migrate() not called): defaults to 0 / 'live'.
    """
    has_redacted = memdag_schema.column_exists(conn, "nodes", "redacted")
    has_status = memdag_schema.column_exists(conn, "nodes", "status")

    extra = ""
    if has_redacted:
        extra += ", redacted"
    if has_status:
        extra += ", status"

    row = conn.execute(
        f"SELECT id, content, channel, label, source_ref, tombstoned{extra}"
        " FROM nodes WHERE id=?",
        (nid,)
    ).fetchone()

    if row is None:
        return None

    d = {
        "id": row[0],
        "content": row[1],
        "channel": row[2],
        "label": row[3],
        "source_ref": row[4],
        "tombstoned": row[5],
    }
    idx = 6
    d["redacted"] = row[idx] if has_redacted else 0
    if has_redacted:
        idx += 1
    d["status"] = row[idx] if has_status else "live"
    return d


def _build_entry(node):
    """Turn a _fetch_full dict into a blame result dict."""
    tombstoned = bool(node["tombstoned"])
    redacted = bool(node["redacted"])
    status = node["status"] or "live"
    state = _state_str(tombstoned, redacted, status)
    line = "[REDACTED]" if redacted else memdag.snippet(node["content"])
    return {
        "id": node["id"],
        "channel": node["channel"],
        "label": node["label"],
        "label_name": memdag.NAME[node["label"]],
        "source_ref": node["source_ref"],
        "state": state,
        "line": line,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blame(conn, nid):
    """Trace node *nid* back to its non-derived root ancestors.

    Returns a list of dicts, one per unique root source node, ordered by
    label DESC, id ASC (same trust order as live_sources).

    If *nid* itself is non-derived (channel != 'agent-derived'), returns a
    single-entry list for itself — a source is its own root.

    Tombstoned and redacted ancestors are INCLUDED with their state; they are
    never silently skipped (blame is a history tool).

    Raises ValueError for an unknown *nid*.
    """
    node = _fetch_full(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id: {nid}")

    # Source nodes are their own roots
    if node["channel"] != "agent-derived":
        return [_build_entry(node)]

    # Recursive CTE: walk ALL ancestors, dedup via UNION
    # Keep only non-derived (channel != 'agent-derived') nodes as roots.
    # ORDER BY label DESC, id ASC mirrors live_sources trust order.
    has_redacted = memdag_schema.column_exists(conn, "nodes", "redacted")
    has_status = memdag_schema.column_exists(conn, "nodes", "status")

    extra_cols = ""
    if has_redacted:
        extra_cols += ", n.redacted"
    if has_status:
        extra_cols += ", n.status"

    sql = (
        "WITH RECURSIVE anc(id) AS ("
        "  SELECT parent FROM edges WHERE child=?"
        "  UNION"
        "  SELECT e.parent FROM edges e JOIN anc a ON e.child = a.id"
        ")"
        " SELECT n.id, n.content, n.channel, n.label, n.source_ref, n.tombstoned"
        + extra_cols +
        " FROM nodes n"
        " WHERE n.id IN (SELECT id FROM anc)"
        "   AND n.channel != 'agent-derived'"
        " ORDER BY n.label DESC, n.id ASC"
    )

    rows = conn.execute(sql, (nid,)).fetchall()

    results = []
    for row in rows:
        d = {
            "id": row[0],
            "content": row[1],
            "channel": row[2],
            "label": row[3],
            "source_ref": row[4],
            "tombstoned": row[5],
        }
        idx = 6
        d["redacted"] = row[idx] if has_redacted else 0
        if has_redacted:
            idx += 1
        d["status"] = row[idx] if has_status else "live"
        results.append(_build_entry(d))

    return results


def format_blame(conn, nid):
    """Return a list of human-readable strings describing the blame result for *nid*.

    Raises ValueError for an unknown *nid*.
    Separated from printing so the MCP server can reuse it.

    Output shape:
      node [<nid>] came from:
        [<id>] <channel>  integrity=<NAME>  <state>  <ref-or-(stated directly)>
              "<line>..."
    """
    entries = blame(conn, nid)
    lines = [f"node [{nid}] came from:"]
    for e in entries:
        ref = e["source_ref"] or "(stated directly)"
        lines.append(
            f"  [{e['id']}] {e['channel']}"
            f"  integrity={e['label_name']}"
            f"  {e['state']}"
            f"  {ref}"
        )
        lines.append(f'      "{e["line"]}..."')
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_blame(args):
    conn = memdag.get_connection()
    migrate(conn)
    try:
        try:
            lines = format_blame(conn, args.id)
        except ValueError as exc:
            print(f"[memdag] {exc}", file=sys.stderr)
            sys.exit(1)
        for line in lines:
            print(line)
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser("blame",
                               help="trace a node back to its root source(s)")
    p.add_argument("id", type=int, help="node id to blame")
    p.set_defaults(func=cmd_blame)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memdag_blame")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
