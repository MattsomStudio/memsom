"""memsom_trust — integrity lattice (meet/join) + audited manual elevation.

The ONLY legal way a node's label rises is through elevate().  Every elevation
is recorded in the `elevations` audit table so that memsom_recompute treats the
node as a fixed point and will not claw the label back down.

Invariants upheld here:
  - Labels come from channels; elevation is manual only, with a paper trail.
  - to_label must be strictly greater than from_label (raise only; lowering is
    what recompute/revoke are for).
  - Jumping from external(0) directly to endorsed(3) is blocked without force=True.
  - After elevation, memsom_recompute.recompute_all() re-floors all descendants
    against the new (higher) label.

Public API
----------
meet(a, b)  -> int                   lattice meet (min), pure, validates 0..3
join(a, b)  -> int                   lattice join (max), pure, validates 0..3
elevate(conn, nid, to_label, reason, by, force=False) -> dict
elevations_for(conn, nid)           -> list[dict]

CLI
---
elevate <id> --to <name> --reason <r> --by <who> [--force]
meet <a> <b>
join <a> <b>
elevations <id>

register(subparsers) mounts into a unified CLI.
main(argv=None) is provided as a standalone entry point.
"""

import argparse
import sys

import memsom
import memsom_schema
import memsom_recompute

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Create the elevations audit table if it does not exist."""
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS elevations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  node INTEGER NOT NULL REFERENCES nodes(id),
  from_label INTEGER NOT NULL,
  to_label INTEGER NOT NULL,
  reason TEXT NOT NULL,
  elevated_by TEXT NOT NULL,
  forced INTEGER NOT NULL DEFAULT 0,
  ts TEXT NOT NULL
);""")


# ---------------------------------------------------------------------------
# Lattice operations (pure functions, no I/O)
# ---------------------------------------------------------------------------

def _validate_label_int(v):
    """Raise ValueError if *v* is not an integer in 0..3."""
    if not isinstance(v, int):
        raise ValueError(f"label must be an int 0..3, got {v!r}")
    if v < 0 or v > 3:
        raise ValueError(f"label out of range 0..3: {v}")


def meet(a, b):
    """Lattice meet: min of two integrity labels.

    Both values must be integers in 0..3, otherwise ValueError is raised.
    """
    _validate_label_int(a)
    _validate_label_int(b)
    return min(a, b)


def join(a, b):
    """Lattice join: max of two integrity labels.

    Both values must be integers in 0..3, otherwise ValueError is raised.
    """
    _validate_label_int(a)
    _validate_label_int(b)
    return max(a, b)


# ---------------------------------------------------------------------------
# Label resolution helper
# ---------------------------------------------------------------------------

def _resolve_label(raw):
    """Accept an int 0..3, a string digit '0'..'3', or a case-insensitive RANK name.

    Raises ValueError on anything else.
    """
    if isinstance(raw, int):
        _validate_label_int(raw)
        return raw
    if isinstance(raw, str):
        # Try numeric string first
        if raw.lstrip('-').isdigit():
            v = int(raw)
            _validate_label_int(v)
            return v
        key = raw.lower()
        if key in memsom.RANK:
            return memsom.RANK[key]
        # Try case-insensitive scan (handles 'ENDORSED', 'User', etc.)
        for k, kv in memsom.RANK.items():
            if k.lower() == key:
                return kv
        raise ValueError(f"unknown label name: {raw!r}")
    raise ValueError(f"label must be int or str, got {type(raw).__name__!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def elevate(conn, nid, to_label, reason, by, force=False):
    """Manually raise the integrity label of node *nid* to *to_label*.

    Parameters
    ----------
    conn       : open sqlite3.Connection (schema must include the elevations table)
    nid        : int — node id
    to_label   : int 0..3 OR case-insensitive RANK name (e.g. 'user', 'ENDORSED')
    reason     : str — why this elevation is happening (audit trail)
    by         : str — who is performing the elevation (audit trail)
    force      : bool — if True, allow the external(0) -> endorsed(3) jump

    Returns
    -------
    dict with keys:
        node                int
        from                int   (label before elevation)
        to                  int   (label after elevation)
        forced              bool
        changed_descendants list[(id, old_label, new_label)]

    Raises
    ------
    ValueError  — node unknown, tombstoned, same label, downward move, blocked jump
    """
    # Resolve label arg
    to_int = _resolve_label(to_label)

    # Ensure the elevations table exists
    migrate(conn)

    # TRUST-2: fetch + tombstone/liveness validation and the mutation must run
    # UNDER one write lock. The old code validated tombstoned outside any
    # transaction, so a revoke landing in the window let elevate write a higher
    # label onto a now-dead node (resurrection at a raised label). Mirror
    # derive_node: BEGIN IMMEDIATE before the SELECT.
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        # Fetch node (under the lock)
        node = memsom.get_node(conn, nid)
        if node is None:
            raise ValueError(f"unknown node id: {nid}")
        if node["tombstoned"]:
            raise ValueError(f"node {nid} is tombstoned; cannot elevate a revoked node")

        from_int = node["label"]

        if to_int == from_int:
            raise ValueError("already at that label — elevation must change something")
        if to_int < from_int:
            raise ValueError("elevate only raises; lowering is what recompute/revoke are for")

        # Policy: a node whose PROVENANCE FLOOR is external(0) reaching ENDORSED(3)
        # requires --force.  Gate on the floor, not on a proxy:
        #   - Keying on from_int (mutable label) let a stepwise 0->1->2->3 walk slip
        #     past entirely (F-08): once the node moved to 1 the single-hop check
        #     never fired again.
        #   - Keying on chan_rank (immutable channel) fixed F-08 for SOURCE nodes but
        #     left TRUST-1: a DERIVED node is always channel 'agent-derived'
        #     (chan_rank 1), so a node derived from pure-external parents (true floor
        #     0) laundered straight to ENDORSED with no force and a falsified forced=0
        #     audit row.
        # The floor that actually matters is min over the node's LIVE parents'
        # effective labels (a source node has none -> use its channel rank).  This is
        # immune to BOTH proxies: it sees through the agent-derived channel string AND
        # it does not move when the node itself is elevated, so stepwise can't escape.
        chan_rank = memsom.RANK.get(node["channel"], from_int)
        live_parent_ids = [r[0] for r in memsom.parents_of(conn, nid) if r[6] == 0]
        if live_parent_ids:
            prov_floor = min(
                memsom_recompute.recompute_label(conn, pid) for pid in live_parent_ids
            )
        else:
            prov_floor = chan_rank
        needs_force = (prov_floor == 0 and to_int == 3)
        if needs_force and not force:
            raise ValueError(
                "external -> endorsed blocked: this node's provenance floor is "
                "external(0); reaching ENDORSED requires --force (logged as forced=1)"
            )

        forced_int = 1 if needs_force else 0
        ts = memsom.now_iso()

        # UPDATE nodes + INSERT elevation audit row — same write-locked unit.
        conn.execute("UPDATE nodes SET label=? WHERE id=?", (to_int, nid))
        conn.execute(
            "INSERT INTO elevations(node, from_label, to_label, reason, elevated_by, forced, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (nid, from_int, to_int, reason, by, forced_int, ts)
        )

    # After commit: re-floor descendants against the new (higher) label.
    # Because the elevations row now exists, recompute treats this node as a
    # fixed point and will not claw the label back down.
    changed = memsom_recompute.recompute_all(conn)

    return {
        "node": nid,
        "from": from_int,
        "to": to_int,
        "forced": bool(forced_int),
        "changed_descendants": changed,
    }


def elevations_for(conn, nid):
    """Return all audit rows for *nid*, ordered by ts ascending.

    Each row is a dict with keys:
        id, node, from_label, to_label, reason, by (mapped from elevated_by),
        forced, ts
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id, node, from_label, to_label, reason, elevated_by, forced, ts"
        " FROM elevations WHERE node=? ORDER BY ts ASC, id ASC",
        (nid,)
    ).fetchall()
    result = []
    for row in rows:
        eid, node, from_label, to_label, reason, elevated_by, forced, ts = row
        result.append({
            "id": eid,
            "node": node,
            "from_label": from_label,
            "to_label": to_label,
            "reason": reason,
            "by": elevated_by,      # mapped: column is elevated_by, exposed as 'by'
            "forced": forced,
            "ts": ts,
        })
    return result


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _label_arg(raw):
    """argparse type: accept int or name."""
    try:
        return _resolve_label(int(raw))
    except (ValueError, TypeError):
        pass
    try:
        return _resolve_label(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"unknown label: {raw!r}")


def cmd_elevate(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            result = elevate(conn, args.id, args.to, args.reason, args.by,
                             force=args.force)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        from_name = memsom.NAME[result["from"]]
        to_name = memsom.NAME[result["to"]]
        flag = " [FORCED]" if result["forced"] else ""
        print(f"[{result['node']}] {from_name} -> {to_name} (by {args.by}): {args.reason}{flag}")
        for nid, old, new in result["changed_descendants"]:
            print(f"  re-floored [{nid}] {memsom.NAME[old]} -> {memsom.NAME[new]}")
    finally:
        conn.close()


def cmd_meet(args):
    try:
        a = _resolve_label(args.a)
        b = _resolve_label(args.b)
    except ValueError as exc:
        print(f"[memsom] {exc}", file=sys.stderr)
        sys.exit(1)
    print(memsom.NAME[meet(a, b)])


def cmd_join(args):
    try:
        a = _resolve_label(args.a)
        b = _resolve_label(args.b)
    except ValueError as exc:
        print(f"[memsom] {exc}", file=sys.stderr)
        sys.exit(1)
    print(memsom.NAME[join(a, b)])


def cmd_elevations(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        rows = elevations_for(conn, args.id)
        if not rows:
            print("no elevations")
        else:
            for r in rows:
                print(f"[{r['id']}] node={r['node']}  "
                      f"{memsom.NAME[r['from_label']]} -> {memsom.NAME[r['to_label']]}  "
                      f"by={r['by']}  forced={r['forced']}  ts={r['ts']}  reason={r['reason']}")
    finally:
        conn.close()


def register(subparsers):
    """Mount trust subcommands onto *subparsers*."""
    # elevate
    p_elev = subparsers.add_parser("elevate",
                                    help="manually raise a node's integrity label (audited)")
    p_elev.add_argument("id", type=int)
    p_elev.add_argument("--to", required=True,
                         help="target label name or int (e.g. 'user', 'endorsed', 2)")
    p_elev.add_argument("--reason", required=True, help="why this elevation is happening")
    p_elev.add_argument("--by", required=True, help="who is performing the elevation")
    p_elev.add_argument("--force", action="store_true",
                         help="allow external->endorsed jump (logged as forced=1)")
    p_elev.set_defaults(func=cmd_elevate)

    # meet
    p_meet = subparsers.add_parser("meet", help="lattice meet (min) of two labels")
    p_meet.add_argument("a", help="label name or int")
    p_meet.add_argument("b", help="label name or int")
    p_meet.set_defaults(func=cmd_meet)

    # join
    p_join = subparsers.add_parser("join", help="lattice join (max) of two labels")
    p_join.add_argument("a", help="label name or int")
    p_join.add_argument("b", help="label name or int")
    p_join.set_defaults(func=cmd_join)

    # elevations
    p_hist = subparsers.add_parser("elevations", help="show elevation audit history for a node")
    p_hist.add_argument("id", type=int)
    p_hist.set_defaults(func=cmd_elevations)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_trust")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
