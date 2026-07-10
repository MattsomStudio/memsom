"""memsom_gate — action-time integrity floor enforcement (Move 4).

check_action() is the ONLY place the floor is enforced. READ paths
(ask/explain/profile/blame/compose) never gate — they always return
their answer regardless of integrity level. The display profile never
feeds this function — it takes only (conn, node_id, required_floor).

The signature itself enforces the invariant: there is no parameter
through which a profile/histogram could influence the decision.

NAMING NOTE: The original design brief named the CLI subcommand
`check`, but that name is already registered by memsom_heal.register()
(memsom_heal.py line 317). Using it would cause argparse to raise
on duplicate add_parser. The subcommand is therefore named
`check-action` here. The MCP/programmatic function name stays
check_action (no hyphen).

Schema: gate_log(id, node, required, floor, decision, culprit, ts)
        A row is inserted on every check_action() call.

Module functions:
  migrate(conn)                              — idempotent DDL
  check_action(conn, node_id, required_floor) -> dict
  recent_gate_log(conn, limit=20)            -> list[dict]

CLI:
  check-action <id> --require <name|int>   (exits 0=allow, 2=deny)
  gate-log [--limit N]
"""

import argparse
import sys

import memsom
from memsom.interface import blame as memsom_blame
from memsom.storage import schema as memsom_schema

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_GATE_LOG_SQL = """CREATE TABLE IF NOT EXISTS gate_log (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  node     INTEGER NOT NULL REFERENCES nodes(id),
  required TEXT    NOT NULL,
  floor    INTEGER NOT NULL,
  decision TEXT    NOT NULL CHECK (decision IN ('allow','deny')),
  culprit  INTEGER,
  ts       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gate_log_node ON gate_log(node);"""


def migrate(conn):
    """Ensure gate_log table and index exist. Idempotent."""
    memsom_schema.ensure_table(conn, _GATE_LOG_SQL)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _parse_required(value) -> int:
    """Parse *value* into an integer integrity rank 0..3.

    Accepts:
      - int 0..3
      - numeric string '0'..'3'
      - case-insensitive RANK name: 'external', 'agent-derived', 'user', 'endorsed'

    Raises ValueError for anything else.
    """
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        raise ValueError(f"unknown required floor: {value!r}")

    s = str(value).strip()
    # Try numeric string first
    if s.isdigit():
        n = int(s)
        if 0 <= n <= 3:
            return n
        raise ValueError(f"unknown required floor: {value!r}")

    # Case-insensitive name lookup
    lower = s.lower()
    if lower in memsom.RANK:
        return memsom.RANK[lower]

    raise ValueError(f"unknown required floor: {value!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_action(conn, node_id, required_floor) -> dict:
    """Check whether *node_id* meets *required_floor* and log the decision.

    This is THE ONLY place the floor is enforced.  READ paths must never
    call this function.

    Parameters
    ----------
    conn          : sqlite3.Connection  (already open; caller owns lifecycle)
    node_id       : int
    required_floor: int 0..3, str '0'..'3', or integrity name ('user', etc.)

    Returns
    -------
    dict with keys:
      decision      'allow' | 'deny'
      floor         int — stored label of node_id (already min(parents))
      floor_name    str — human-readable label name
      required      int — parsed required_floor
      required_name str — human-readable required name
      culprit       int | None — id of weakest live leaf ancestor (deny only)
      reason        str — human-readable explanation

    Raises
    ------
    ValueError — unknown node_id or bad required_floor value.
    No gate_log row is written for unknown node ids.
    """
    # 1. Parse required floor — may raise ValueError for bad names/values
    req = _parse_required(required_floor)

    # 2. Fetch node — unknown id is a hard error; no log row written
    node = memsom.get_node(conn, node_id)
    if node is None:
        raise ValueError(f"unknown node id: {node_id}")

    # 3. Floor = stored label (already = min(parents) by frozen invariant).
    #    NEVER recompute, NEVER mutate.
    floor = node["label"]

    # 4/5. Decision
    if floor >= req:
        decision = "allow"
        culprit = None
        reason = (
            f"floor {memsom.NAME[floor]} >= required {memsom.NAME[req]}"
        )
    else:
        decision = "deny"
        # Walk leaf ancestors via blame (handles missing redacted/status columns)
        entries = memsom_blame.blame(conn, node_id)
        # Live = anything that is not tombstoned.
        # Redacted/quarantined nodes still count — redaction does not change labels,
        # so a redacted leaf can still be the floor-setter.
        live = [e for e in entries if "tombstoned" not in e["state"]]
        if live:
            culprit = min(live, key=lambda e: (e["label"], e["id"]))["id"]
            reason = (
                f"floor {memsom.NAME[floor]} < required {memsom.NAME[req]};"
                f" weakest live leaf = [{culprit}]"
            )
        else:
            culprit = None
            reason = (
                f"floor {memsom.NAME[floor]} < required {memsom.NAME[req]};"
                " no live leaf ancestors"
            )

    # 6. Log every evaluated call (migrate is idempotent — cheap)
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT INTO gate_log(node, required, floor, decision, culprit, ts)"
            " VALUES (?,?,?,?,?,?)",
            (node_id, memsom.NAME[req], floor, decision, culprit, memsom.now_iso()),
        )

    # 7. Return result dict
    return {
        "decision": decision,
        "floor": floor,
        "floor_name": memsom.NAME[floor],
        "required": req,
        "required_name": memsom.NAME[req],
        "culprit": culprit,
        "reason": reason,
    }


def recent_gate_log(conn, limit=20) -> list:
    """Return the most recent *limit* gate_log rows as a list of dicts.

    Columns: id, node, required, floor, decision, culprit, ts.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id, node, required, floor, decision, culprit, ts"
        " FROM gate_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    keys = ("id", "node", "required", "floor", "decision", "culprit", "ts")
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def cmd_check_action(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        try:
            r = check_action(conn, args.id, args.require)
        except ValueError as exc:
            print(f"[memsom] {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"{r['decision'].upper()}  node [{args.id}]"
            f"  floor={r['floor_name']}  required={r['required_name']}"
        )
        if r["decision"] == "deny":
            if r["culprit"] is not None:
                print(f"  culprit: [{r['culprit']}]")
            else:
                print("  culprit: none (no live leaf ancestors)")
            print(f"  {r['reason']}")
        if r["decision"] == "deny":
            sys.exit(2)
    finally:
        conn.close()


def cmd_gate_log(args):
    conn = memsom.get_connection()
    try:
        rows = recent_gate_log(conn, args.limit)
        if not rows:
            print("gate log empty")
        else:
            for r in rows:
                line = (
                    f"[{r['id']}] {r['ts']}"
                    f"  node={r['node']}"
                    f"  required={r['required']}"
                    f"  floor={memsom.NAME[r['floor']]}"
                    f"  {r['decision'].upper()}"
                )
                if r["culprit"] is not None:
                    line += f"  culprit=[{r['culprit']}]"
                print(line)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# register — mount subcommands into a shared subparsers object
# ---------------------------------------------------------------------------

def register(subparsers):
    p = subparsers.add_parser(
        "check-action",
        help="action-time integrity gate: allow/deny by floor (the ONLY gate)",
    )
    p.add_argument("id", type=int)
    p.add_argument(
        "--require",
        required=True,
        help="minimum integrity floor name or int (e.g. user, endorsed, 2)",
    )
    p.set_defaults(func=cmd_check_action)

    q = subparsers.add_parser(
        "gate-log",
        help="recent gate decisions (audit)",
    )
    q.add_argument("--limit", type=int, default=20)
    q.set_defaults(func=cmd_gate_log)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_gate")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
