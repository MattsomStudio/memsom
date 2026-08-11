"""memsom_blame — "git blame" for the derivation DAG.

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

import memsom
from memsom.storage import schema as memsom_schema
from memsom.integrity import redact as memsom_redact
from memsom.integrity import quarantine as memsom_quarantine


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Ensure redact and quarantine state columns exist. Idempotent."""
    memsom_redact.migrate(conn)
    memsom_quarantine.migrate(conn)


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
    has_redacted = memsom_schema.column_exists(conn, "nodes", "redacted")
    has_status = memsom_schema.column_exists(conn, "nodes", "status")
    has_conf = memsom_schema.column_exists(conn, "nodes", "conf_label")

    extra = ""
    if has_redacted:
        extra += ", redacted"
    if has_status:
        extra += ", status"
    if has_conf:
        extra += ", conf_label"

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
    if has_status:
        idx += 1
    d["conf_label"] = row[idx] if has_conf else 0
    return d


def _build_entry(node, clearance=None):
    """Turn a _fetch_full dict into a blame result dict.

    BLAME-CONF-1: blame is a read path and must honour the confidentiality axis
    like ask/retrieve. When *clearance* is given (a parsed int 0-3) and the node's
    conf_label exceeds it, the content snippet is suppressed to '[ABOVE CLEARANCE]'
    (BLP no-read-up) — metadata (id/channel/label/state/ref) stays visible so the
    provenance shape is still auditable. clearance=None preserves full admin/history
    use (the default).
    """
    tombstoned = bool(node["tombstoned"])
    redacted = bool(node["redacted"])
    status = node["status"] or "live"
    state = _state_str(tombstoned, redacted, status)
    if redacted:
        line = "[REDACTED]"
    elif clearance is not None and node.get("conf_label", 0) > clearance:
        line = "[ABOVE CLEARANCE]"
    else:
        line = memsom.snippet(node["content"])
    return {
        "id": node["id"],
        "channel": node["channel"],
        "label": node["label"],
        "label_name": memsom.NAME[node["label"]],
        "source_ref": node["source_ref"],
        "state": state,
        "line": line,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blame(conn, nid, clearance=None):
    """Trace node *nid* back to its non-derived root ancestors.

    Returns a list of dicts, one per unique root source node, ordered by
    label DESC, id ASC (same trust order as live_sources).

    If *nid* itself is non-derived (channel != 'agent-derived'), returns a
    single-entry list for itself — a source is its own root.

    Tombstoned and redacted ancestors are INCLUDED with their state; they are
    never silently skipped (blame is a history tool).

    BLAME-CONF-1: *clearance* (parsed int 0-3, or None for no filter) suppresses
    the content snippet of any root above clearance — blame must not become a
    provenance oracle that leaks high-confidentiality source content.

    Raises ValueError for an unknown *nid*.
    """
    node = _fetch_full(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id: {nid}")

    # Source nodes are their own roots
    if node["channel"] != "agent-derived":
        return [_build_entry(node, clearance)]

    # Recursive CTE: walk ALL ancestors, dedup via UNION
    # Keep only non-derived (channel != 'agent-derived') nodes as roots.
    # ORDER BY label DESC, id ASC mirrors live_sources trust order.
    has_redacted = memsom_schema.column_exists(conn, "nodes", "redacted")
    has_status = memsom_schema.column_exists(conn, "nodes", "status")
    has_conf = memsom_schema.column_exists(conn, "nodes", "conf_label")

    extra_cols = ""
    if has_redacted:
        extra_cols += ", n.redacted"
    if has_status:
        extra_cols += ", n.status"
    if has_conf:
        extra_cols += ", n.conf_label"

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
        if has_status:
            idx += 1
        d["conf_label"] = row[idx] if has_conf else 0
        results.append(_build_entry(d, clearance))

    return results


def format_blame(conn, nid, clearance=None):
    """Return a list of human-readable strings describing the blame result for *nid*.

    Raises ValueError for an unknown *nid*.
    Separated from printing so the MCP server can reuse it.
    *clearance* (parsed int 0-3 or None) is threaded to blame() for BLP gating.

    Output shape:
      node [<nid>] came from:
        [<id>] <channel>  integrity=<NAME>  <state>  <ref-or-(stated directly)>
              "<line>..."
    """
    entries = blame(conn, nid, clearance)
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
    conn = memsom.get_connection()
    try:
        # BLAME-1: migrate() inside the try so a failure here still closes conn.
        migrate(conn)
        clearance = None
        if getattr(args, "clearance", None) is not None:
            try:
                from memsom.integrity import confid as memsom_confid
                clearance = memsom_confid.parse_conf(args.clearance)
            except ValueError as exc:
                print(f"[memsom] invalid --clearance: {exc}", file=sys.stderr)
                sys.exit(1)
        try:
            lines = format_blame(conn, args.id, clearance)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        for line in lines:
            print(line)
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser("blame",
                               help="trace a node back to its root source(s)")
    p.add_argument("id", type=int, help="node id to blame")
    p.add_argument("--clearance", default=None,
                   help="confidentiality ceiling (public|internal|secret|topsecret or 0-3); "
                        "suppresses content of roots above it (default: no filter)")
    p.set_defaults(func=cmd_blame)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_blame")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
