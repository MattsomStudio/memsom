"""memsom_profile — leaf-origin PROFILE for the derivation DAG.

The PROFILE is DISPLAY-ONLY. It must NEVER feed a gate or any action decision.
The FLOOR (min of parents, already on the label column) is the ONLY thing that
gates — and it is enforced elsewhere (memsom_gate), never here.

Public API
----------
migrate(conn)                        — no-op migration stub (CLI uniformity)
profile(conn, nid) -> dict           — leaf-origin histogram; ValueError for unknown nid
format_profile(p) -> str             — one-line summary string (pure, no I/O)

CLI
---
profile <id>    print floor + histogram + external leaf ids
register(subparsers)
main(argv=None)
"""

import sys

import memsom
import memsom_schema


# ---------------------------------------------------------------------------
# Recursive CTE: walk ALL ancestors, dedup via UNION (same discipline as
# memsom.CASCADE_CTE — terminates on cycles, visits diamonds once).
# ---------------------------------------------------------------------------

ANC_CTE = (
    "WITH RECURSIVE ancestors(id) AS ("
    "SELECT parent FROM edges WHERE child = ? "
    "UNION "
    "SELECT e.parent FROM edges e JOIN ancestors a ON e.child = a.id"
    ")"
)


# ---------------------------------------------------------------------------
# Migration stub — no schema changes needed for profile
# ---------------------------------------------------------------------------

def migrate(conn):
    """No-op migration; provided for CLI uniformity."""
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile(conn, nid):
    """Return a provenance profile dict for node *nid*.

    Dict keys:
      node             — the queried node id
      floor            — stored label (READ from nodes.label; never recomputed here)
      floor_name       — human-readable floor label (e.g. "EXTERNAL")
      hist             — dict[int, int]: label -> count of leaves with that label (>0 only)
      leaf_total       — total live, non-redacted non-derived leaf count
      external_leaf_ids— sorted list of leaf ids whose label == 0
      summary          — format_profile(d) one-liner

    Raises ValueError for an unknown nid.
    Does NOT gate, write, or compare against any threshold.
    """
    node = memsom.get_node(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id: {nid}")

    floor = node["label"]

    # --- determine live-leaf rows ---
    if node["channel"] != "agent-derived":
        # Bare source node: it is its own single leaf, IF it is live.
        has_red = memsom_schema.column_exists(conn, "nodes", "redacted")
        is_tombstoned = bool(node["tombstoned"])
        is_redacted = False
        if has_red:
            row = conn.execute(
                "SELECT redacted FROM nodes WHERE id=?", (nid,)
            ).fetchone()
            is_redacted = bool(row[0]) if row else False

        if is_tombstoned or is_redacted:
            rows = []
        else:
            rows = [(nid, floor)]
    else:
        # Derived node: CTE walk to find all non-derived ancestors that are live.
        has_red = memsom_schema.column_exists(conn, "nodes", "redacted")
        red_clause = " AND n.redacted = 0" if has_red else ""
        sql = (
            ANC_CTE
            + " SELECT n.id, n.label FROM nodes n"
            " WHERE n.id IN (SELECT id FROM ancestors)"
            " AND n.channel != 'agent-derived'"
            " AND n.tombstoned = 0"
            + red_clause
            + " ORDER BY n.id"
        )
        rows = conn.execute(sql, (nid,)).fetchall()

    # --- build histogram ---
    hist = {}
    for _lid, llabel in rows:
        hist[llabel] = hist.get(llabel, 0) + 1

    leaf_total = len(rows)
    external_leaf_ids = sorted([lid for (lid, llabel) in rows if llabel == 0])

    d = {
        "node": nid,
        "floor": floor,
        "floor_name": memsom.NAME[floor],
        "hist": hist,
        "leaf_total": leaf_total,
        "external_leaf_ids": external_leaf_ids,
    }
    d["summary"] = format_profile(d)
    return d


def format_profile(p):
    """Return a one-line provenance summary string (pure — no I/O, no conn).

    Example (with external leaves):
      "floor: EXTERNAL (gates) | provenance: 2 of 3 leaves endorsed/user, 1 external [mem:3] - inspect"

    Example (no external leaves):
      "floor: ENDORSED (gates) | provenance: 4 of 4 leaves endorsed/user"

    ASCII only (hyphen, not em-dash) — output safe under cp1252 in subprocess tests.
    """
    trusted = p["leaf_total"] - len(p["external_leaf_ids"])
    base = (
        f"floor: {p['floor_name']} (gates)"
        f" | provenance: {trusted} of {p['leaf_total']} leaves endorsed/user"
    )
    if p["external_leaf_ids"]:
        ids_str = ",".join(str(i) for i in p["external_leaf_ids"])
        base += f", {len(p['external_leaf_ids'])} external [mem:{ids_str}] - inspect"
    return base


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_profile(args):
    conn = memsom.get_connection()
    try:
        try:
            p = profile(conn, args.id)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        print(format_profile(p))
        for label in sorted(p["hist"], reverse=True):
            print(f"  {memsom.NAME[label]:<13} {p['hist'][label]}")
        if p["external_leaf_ids"]:
            print(f"  external leaves: {', '.join('[' + str(i) + ']' for i in p['external_leaf_ids'])}")
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser(
        "profile",
        help="leaf-origin provenance histogram (display-only; floor gates, profile never does)",
    )
    p.add_argument("id", type=int)
    p.set_defaults(func=cmd_profile)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:  # io.StringIO under tests has no reconfigure
            pass
    p = argparse.ArgumentParser(prog="memsom_profile")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
