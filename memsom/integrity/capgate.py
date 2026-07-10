"""memsom_capgate — capability gate: the Gate #3 decision + audit (action gate).

The runtime sibling of memsom_gate.check_action (memsom_gate.py:98).  Where
check_action compares a NODE's integrity label against a required floor,
check_capability compares the SESSION's integrity floor (memsom_session) against
a TOOL's required floor (memsom_policy).  Same RANK vocabulary, same
allow/deny + mandatory audit discipline.

This is the ONLY place a capability decision is made + logged.  The broker calls
it BEFORE forwarding any upstream tool call; deny means the call never reaches
the upstream server.

A separate `capability_log` table (not gate_log) keeps the node-scoped gate and
its frozen test suite untouched.

Schema: capability_log(id, session_id, tool, required, floor, decision, reason, ts)
        One row per check_capability() call.

Required floors are ints 0..4: 0..3 are the RANK lattice, 4 is the policy DENY
sentinel (never satisfiable).  The comparison is uniform: allow iff
session_floor >= required.

Public API
----------
  migrate(conn)                                                   -> None
  check_capability(conn, session_id, session_floor, tool, required) -> dict
  recent_capability_log(conn, limit=20)                           -> list[dict]
"""

import argparse
import sys

import memsom
from memsom.storage import schema as memsom_schema

_CAP_LOG_SQL = """CREATE TABLE IF NOT EXISTS capability_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT    NOT NULL,
  tool       TEXT    NOT NULL,
  required   TEXT    NOT NULL,
  floor      INTEGER NOT NULL,
  decision   TEXT    NOT NULL CHECK (decision IN ('allow','deny')),
  reason     TEXT,
  ts         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_capability_log_session
  ON capability_log(session_id);"""


def migrate(conn):
    """Ensure capability_log table + index exist. Idempotent."""
    memsom_schema.ensure_table(conn, _CAP_LOG_SQL)


def _required_name(required: int) -> str:
    """Human-readable name for a required floor 0..4 (4 = policy DENY sentinel)."""
    return memsom.NAME.get(required, "DENY")


def check_capability(conn, session_id, session_floor, tool, required) -> dict:
    """Decide whether a session at *session_floor* may invoke *tool* (which needs
    *required*), log the decision, and return it.

    Parameters
    ----------
    conn          : sqlite3.Connection (caller owns lifecycle)
    session_id    : str  (for the audit row; not dereferenced here)
    session_floor : int 0..3  — the current session taint floor
    tool          : str  — the namespaced tool name being gated
    required      : int 0..4 — minimum floor to allow (4 = always deny)

    Returns
    -------
    dict: decision ('allow'|'deny'), floor, floor_name, required, required_name,
          tool, reason.

    Raises ValueError for an out-of-range session_floor or required.
    """
    if not isinstance(session_floor, int) or isinstance(session_floor, bool) \
            or not (0 <= session_floor <= 3):
        raise ValueError(f"bad session_floor: {session_floor!r}")
    if not isinstance(required, int) or isinstance(required, bool) \
            or not (0 <= required <= 4):
        raise ValueError(f"bad required: {required!r}")

    req_name = _required_name(required)
    if session_floor >= required:
        decision = "allow"
        reason = f"session floor {memsom.NAME[session_floor]} >= required {req_name}"
    else:
        decision = "deny"
        reason = f"session floor {memsom.NAME[session_floor]} < required {req_name}"

    migrate(conn)
    with conn:
        conn.execute(
            "INSERT INTO capability_log(session_id, tool, required, floor, decision, reason, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (session_id, tool, req_name, session_floor, decision, reason, memsom.now_iso()),
        )

    return {
        "decision": decision,
        "floor": session_floor,
        "floor_name": memsom.NAME[session_floor],
        "required": required,
        "required_name": req_name,
        "tool": tool,
        "reason": reason,
    }


def recent_capability_log(conn, limit=20) -> list:
    """Return the most recent *limit* capability_log rows (most recent first)."""
    migrate(conn)
    rows = conn.execute(
        "SELECT id, session_id, tool, required, floor, decision, reason, ts"
        " FROM capability_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    keys = ("id", "session_id", "tool", "required", "floor", "decision", "reason", "ts")
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_capability_log(args):
    conn = memsom.get_connection()
    try:
        rows = recent_capability_log(conn, args.limit)
        if not rows:
            print("capability log empty")
        else:
            for r in rows:
                print(
                    f"[{r['id']}] {r['ts']}  session={r['session_id'][:8]}"
                    f"  tool={r['tool']}"
                    f"  floor={memsom.NAME[r['floor']]}"
                    f"  required={r['required']}"
                    f"  {r['decision'].upper()}"
                )
    finally:
        conn.close()


def register(subparsers):
    q = subparsers.add_parser(
        "capability-log",
        help="recent capability-gate decisions (Gate #3 audit)",
    )
    q.add_argument("--limit", type=int, default=20)
    q.set_defaults(func=cmd_capability_log)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_capgate")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
