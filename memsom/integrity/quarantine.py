"""memsom_quarantine — consolidation gates + quarantine.

External taint can NEVER silently promote into trusted knowledge.
Every node derived (even partially) from an external-channel source is
quarantined by consolidate(); a human reviewer must call promote() with
a fully-endorsed ancestor chain before it re-enters the live pool.

Public API
----------
migrate(conn)                        -> None        idempotent schema migration
consolidate(conn)                    -> list[(id, cause_str)]
quarantine_node(conn, nid, reason)   -> bool        True=flipped, False=no-op
promote(conn, nid, by)               -> None        raises ValueError on gate fail
list_quarantined(conn)               -> list[dict]
live_source_ids(conn)                -> list[int]
live_unquarantined_sources(conn)     -> list[tuple]

CLI sub-commands (via register(subparsers))
-------------------------------------------
consolidate
quarantine <id> --reason <r>
promote    <id> --by <who>
quarantine-list
"""

import argparse
import sqlite3
import sys

import memsom
from memsom.storage import schema as memsom_schema

# ---------------------------------------------------------------------------
# Ancestry CTE — UNION (not UNION ALL) deduplicates, terminates on cycles
# ---------------------------------------------------------------------------
ANC_CTE = (
    "WITH RECURSIVE anc(id) AS ("
    "SELECT parent FROM edges WHERE child=? "
    "UNION "
    "SELECT e.parent FROM edges e JOIN anc a ON e.child=a.id"
    ")"
)


# ---------------------------------------------------------------------------
# Schema migration (additive only)
# ---------------------------------------------------------------------------

def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: add quarantine columns to nodes if they are not yet present."""
    memsom_schema.add_column(conn, "nodes", "status",
                             "TEXT NOT NULL DEFAULT 'live'")
    memsom_schema.add_column(conn, "nodes", "quarantine_reason", "TEXT")
    memsom_schema.add_column(conn, "nodes", "quarantined_at", "TEXT")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_live_ancestor_channel(conn: sqlite3.Connection, nid: int, channel: str) -> bool:
    """Return True if *nid* has at least one live (non-tombstoned) ancestor
    whose channel == *channel*."""
    row = conn.execute(
        ANC_CTE
        + " SELECT 1 FROM nodes"
        "  WHERE id IN (SELECT id FROM anc)"
        "    AND channel = ?"
        "    AND tombstoned = 0"
        "  LIMIT 1",
        (nid, channel),
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def consolidate(conn: sqlite3.Connection) -> list:
    """Scan live agent-derived nodes; quarantine any whose integrity label is
    EXTERNAL (0) OR that has any live external-channel ancestor.

    Returns list of (id, cause_str) for every node that was quarantined.
    Second call returns [] (idempotent).
    """
    migrate(conn)
    candidates = conn.execute(
        "SELECT id, label FROM nodes"
        " WHERE tombstoned = 0 AND channel = 'agent-derived' AND status = 'live'"
        " ORDER BY id"
    ).fetchall()

    quarantined = []
    now = memsom.now_iso()

    with conn:
        for nid, label in candidates:
            cause = None
            if label == 0:
                cause = "integrity=EXTERNAL"
            elif _has_live_ancestor_channel(conn, nid, "external"):
                cause = "live external ancestor"

            if cause is None:
                continue

            conn.execute(
                "UPDATE nodes SET status='quarantined',"
                " quarantined_at=?, quarantine_reason=?"
                " WHERE id=?",
                (now, "consolidation gate: " + cause, nid),
            )
            quarantined.append((nid, "consolidation gate: " + cause))

    return quarantined


def quarantine_node(conn: sqlite3.Connection, nid: int, reason: str) -> bool:
    """Manually quarantine a node.

    Returns True if the node was quarantined, False if already quarantined.
    Raises ValueError for unknown or tombstoned ids.
    """
    migrate(conn)
    row = conn.execute(
        "SELECT tombstoned, status FROM nodes WHERE id = ?", (nid,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown node id: {nid}")
    tombstoned, status = row
    if tombstoned:
        raise ValueError(f"node {nid} is tombstoned; cannot quarantine a dead node")
    if status == "quarantined":
        return False

    with conn:
        conn.execute(
            "UPDATE nodes SET status='quarantined', quarantine_reason=?, quarantined_at=?"
            " WHERE id=?",
            (reason, memsom.now_iso(), nid),
        )
    return True


def promote(conn: sqlite3.Connection, nid: int, by: str) -> None:
    """Promote a quarantined node back to live.

    Gate conditions (both must hold):
      1. At least one live endorsed-channel ancestor.
      2. Zero live external-channel ancestors.

    Raises ValueError on unknown / not-quarantined node or gate failure.
    Leaves quarantined_at intact as history; overwrites quarantine_reason with
    an audit breadcrumb.
    """
    migrate(conn)
    # QUAR-PROMOTE-TOCTOU: the gate reads (status + ancestor channels) and the
    # status flip must be ONE write-locked unit. Otherwise a concurrent
    # revoke/derive between the gate check and the UPDATE could change the ancestor
    # set, letting a node that should fail the gate be promoted back to live.
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            "SELECT tombstoned, status FROM nodes WHERE id = ?", (nid,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown node id: {nid}")
        tombstoned, status = row
        if tombstoned or status != "quarantined":
            raise ValueError(f"node {nid} is not quarantined (status={status!r})")

        has_endorsed = _has_live_ancestor_channel(conn, nid, "endorsed")
        has_external = _has_live_ancestor_channel(conn, nid, "external")

        if not has_endorsed and has_external:
            raise ValueError(
                "manual endorsement required: no endorsed ancestor AND live external ancestor present"
            )
        if not has_endorsed:
            raise ValueError(
                "manual endorsement required: no endorsed ancestor"
            )
        if has_external:
            raise ValueError(
                "manual endorsement required: live external ancestor present"
            )

        now = memsom.now_iso()
        conn.execute(
            "UPDATE nodes SET status='live', quarantine_reason=?"
            " WHERE id=?",
            (f"promoted by {by} at {now}", nid),
        )


def list_quarantined(conn: sqlite3.Connection) -> list:
    """Return list of dicts for all live-quarantined nodes, ordered by id."""
    migrate(conn)
    rows = conn.execute(
        "SELECT id, channel, label, quarantine_reason, quarantined_at"
        " FROM nodes WHERE status='quarantined' AND tombstoned=0 ORDER BY id"
    ).fetchall()
    keys = ("id", "channel", "label", "quarantine_reason", "quarantined_at")
    return [dict(zip(keys, r)) for r in rows]


def live_source_ids(conn: sqlite3.Connection) -> list:
    """Return ids of live_sources rows whose status == 'live'.

    memsom.live_sources is the source of truth for which rows are sources;
    this function just filters out quarantined ones.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id FROM nodes"
        " WHERE tombstoned = 0 AND channel != 'agent-derived' AND status = 'live'"
        " ORDER BY label DESC, id ASC"
    ).fetchall()
    return [r[0] for r in rows]


def live_unquarantined_sources(conn: sqlite3.Connection) -> list:
    """Same row shape as memsom.live_sources (id, content, channel, label, source_ref),
    filtered to exclude quarantined rows."""
    migrate(conn)
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE tombstoned = 0 AND channel != 'agent-derived' AND status != 'quarantined'"
        " ORDER BY label DESC, id ASC"
    ).fetchall()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_consolidate(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        results = consolidate(conn)
        if not results:
            print("nothing to quarantine")
        else:
            for nid, cause in results:
                print(f"[{nid}] quarantined - {cause}")
    finally:
        conn.close()


def _cmd_quarantine(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            flipped = quarantine_node(conn, args.id, args.reason)
        except ValueError as exc:
            print(f"[memsom-quarantine] {exc}", file=sys.stderr)
            sys.exit(1)
        if flipped:
            print(f"[{args.id}] quarantined")
        else:
            print(f"[{args.id}] already quarantined - no-op")
    finally:
        conn.close()


def _cmd_promote(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            promote(conn, args.id, args.by)
        except ValueError as exc:
            print(f"[memsom-quarantine] {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"[{args.id}] promoted to live by {args.by}")
    finally:
        conn.close()


def _cmd_quarantine_list(args) -> None:
    conn = memsom.get_connection()
    try:
        migrate(conn)
        rows = list_quarantined(conn)
        if not rows:
            print("no quarantined nodes")
        else:
            for r in rows:
                print(
                    f"[{r['id']}] {r['channel']:<13}"
                    f" integrity={memsom_schema.NAME[r['label']]:<13}"
                    f" reason={r['quarantine_reason']!r}"
                    f" at={r['quarantined_at']}"
                )
    finally:
        conn.close()


def register(subparsers) -> None:
    """Mount quarantine sub-commands onto an existing argparse subparsers object."""
    sp_con = subparsers.add_parser("consolidate", help="run consolidation gate")
    sp_con.set_defaults(func=_cmd_consolidate)

    sp_q = subparsers.add_parser("quarantine", help="manually quarantine a node")
    sp_q.add_argument("id", type=int)
    sp_q.add_argument("--reason", required=True)
    sp_q.set_defaults(func=_cmd_quarantine)

    sp_p = subparsers.add_parser("promote", help="promote a quarantined node back to live")
    sp_p.add_argument("id", type=int)
    sp_p.add_argument("--by", required=True)
    sp_p.set_defaults(func=_cmd_promote)

    sp_ql = subparsers.add_parser("quarantine-list", help="list quarantined nodes")
    sp_ql.set_defaults(func=_cmd_quarantine_list)


def main(argv=None) -> None:
    """Thin CLI wrapper — delegates to register()."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-quarantine")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
