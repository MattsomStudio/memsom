"""memsom_session — per-session integrity taint floor (Gate #3 substrate).

A "session" is one run of the memsom broker (memsom_broker.py): the process
serves a single MCP client for its lifetime, so process-lifetime == session.

The session carries ONE integer integrity floor on the existing RANK lattice
(memsom.py:33  endorsed3 > user2 > agent-derived1 > external0).  It starts HIGH
(default `user`) and only ever DROPS — the runtime dual of derive_node's
`min(parent labels)` (memsom.py:122).  When untrusted content flows through the
broker (a web fetch, or any tool the policy flags as tainting), the floor is
lowered via `min(old, incoming)`.  It is never raised inside a session; a new
process is the only way back to a clean floor (pessimistic, lowest-water-mark,
deliberately consistent with the Biba model — no per-value declassification).

`lower_floor` is the ONLY mutator and the ONE place the watermark invariant
(floor never rises) is enforced, by construction (`min`).

Schema:
  session_taint(session_id, started_at, start_floor, floor, updated_at)
  session_taint_event(id, session_id, tool, from_floor, to_floor, reason, ts)

Library discipline: no prints, no sys.exit; callers own the connection.

Public API
----------
  migrate(conn)                                          -> None  (idempotent DDL)
  begin_session(conn, start_floor=RANK['user'])          -> str   (session_id)
  current_floor(conn, session_id)                        -> int
  lower_floor(conn, session_id, incoming_label, tool, reason) -> int  (new floor)
  session_log(conn, session_id, limit=20)                -> list[dict]
"""

import argparse
import sys
import uuid

import memsom
from memsom.storage import schema as memsom_schema

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """CREATE TABLE IF NOT EXISTS session_taint (
  session_id  TEXT    PRIMARY KEY,
  started_at  TEXT    NOT NULL,
  start_floor INTEGER NOT NULL CHECK (start_floor BETWEEN 0 AND 3),
  floor       INTEGER NOT NULL CHECK (floor BETWEEN 0 AND 3),
  updated_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS session_taint_event (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT    NOT NULL REFERENCES session_taint(session_id),
  tool       TEXT    NOT NULL,
  from_floor INTEGER NOT NULL,
  to_floor   INTEGER NOT NULL,
  reason     TEXT    NOT NULL,
  ts         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_taint_event_sid
  ON session_taint_event(session_id);"""


def migrate(conn):
    """Ensure session tables + index exist. Idempotent."""
    memsom_schema.ensure_table(conn, _SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _parse_floor(value) -> int:
    """Parse *value* into an integer integrity rank 0..3.

    Accepts int 0..3, numeric string '0'..'3', or a case-insensitive RANK name
    ('external', 'agent-derived', 'user', 'endorsed').  Raises ValueError
    otherwise.  Mirrors memsom_gate._parse_required but kept local so this
    module does not depend on the gate.
    """
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise ValueError(f"bad floor: {value!r}")
    if isinstance(value, int):
        if 0 <= value <= 3:
            return value
        raise ValueError(f"bad floor: {value!r}")
    s = str(value).strip()
    if s.isdigit():
        n = int(s)
        if 0 <= n <= 3:
            return n
        raise ValueError(f"bad floor: {value!r}")
    if s.lower() in memsom.RANK:
        return memsom.RANK[s.lower()]
    raise ValueError(f"bad floor: {value!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def begin_session(conn, start_floor=None) -> str:
    """Create a new session row and return its generated session_id.

    *start_floor* defaults to RANK['user'] (2): a session begins with
    user-directed intent.  Accepts an int or a RANK name.
    """
    floor = _parse_floor(start_floor) if start_floor is not None else memsom.RANK["user"]
    sid = uuid.uuid4().hex
    now = memsom.now_iso()
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT INTO session_taint(session_id, started_at, start_floor, floor, updated_at)"
            " VALUES (?,?,?,?,?)",
            (sid, now, floor, floor, now),
        )
    return sid


def ensure_session(conn, session_id, start_floor=None) -> int:
    """Idempotently ensure a session row exists for an EXTERNALLY-supplied id
    (e.g. a Claude Code session_id passed in by a hook) and return its current
    floor.  Never lowers or resets an existing session — first sight creates the
    row at *start_floor* (default `user`); later calls are no-ops.  This is the
    bridge that lets the hook layer key taint on the live Claude session.
    """
    floor = _parse_floor(start_floor) if start_floor is not None else memsom.RANK["user"]
    now = memsom.now_iso()
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO session_taint"
            "(session_id, started_at, start_floor, floor, updated_at)"
            " VALUES (?,?,?,?,?)",
            (session_id, now, floor, floor, now),
        )
    return current_floor(conn, session_id)


def current_floor(conn, session_id) -> int:
    """Return the current integrity floor for *session_id*.

    Raises ValueError for an unknown session id.
    """
    migrate(conn)
    row = conn.execute(
        "SELECT floor FROM session_taint WHERE session_id=?", (session_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown session id: {session_id!r}")
    return row[0]


def lower_floor(conn, session_id, incoming_label, tool, reason) -> int:
    """Lower the session floor to min(current, incoming_label) and return it.

    This is the ONLY mutator of the session floor.  The watermark invariant —
    the floor can never RISE within a session — holds by construction (`min`):
    passing an *incoming_label* at or above the current floor is a safe no-op.

    A session_taint_event row is written only on an actual drop (a no-op records
    nothing, keeping the event table a true transition log).  Raises ValueError
    for an unknown session id or a bad incoming_label.
    """
    incoming = _parse_floor(incoming_label)
    migrate(conn)
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT floor FROM session_taint WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown session id: {session_id!r}")
        old = row[0]
        new = min(old, incoming)  # watermark: new <= old, always
        if new != old:
            now = memsom.now_iso()
            conn.execute(
                "UPDATE session_taint SET floor=?, updated_at=? WHERE session_id=?",
                (new, now, session_id),
            )
            conn.execute(
                "INSERT INTO session_taint_event"
                "(session_id, tool, from_floor, to_floor, reason, ts)"
                " VALUES (?,?,?,?,?,?)",
                (session_id, tool, old, new, reason, now),
            )
    return new


def session_log(conn, session_id, limit=20) -> list:
    """Return the most recent *limit* taint-transition events for *session_id*.

    Columns: id, session_id, tool, from_floor, to_floor, reason, ts.  Most
    recent first.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id, session_id, tool, from_floor, to_floor, reason, ts"
        " FROM session_taint_event WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    keys = ("id", "session_id", "tool", "from_floor", "to_floor", "reason", "ts")
    return [dict(zip(keys, r)) for r in rows]


def recent_events(conn, limit=20) -> list:
    """Recent taint-transition events across ALL sessions (audit view)."""
    migrate(conn)
    rows = conn.execute(
        "SELECT id, session_id, tool, from_floor, to_floor, reason, ts"
        " FROM session_taint_event ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    keys = ("id", "session_id", "tool", "from_floor", "to_floor", "reason", "ts")
    return [dict(zip(keys, r)) for r in rows]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_session_log(args):
    conn = memsom.get_connection()
    try:
        rows = session_log(conn, args.session, args.limit) if args.session \
            else recent_events(conn, args.limit)
        if not rows:
            print("session taint log empty")
        else:
            for r in rows:
                print(
                    f"[{r['id']}] {r['ts']}  session={r['session_id'][:8]}"
                    f"  {memsom.NAME[r['from_floor']]} -> {memsom.NAME[r['to_floor']]}"
                    f"  via {r['tool']} ({r['reason']})"
                )
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser(
        "session-log",
        help="recent session taint-floor transitions (Gate #3 audit)",
    )
    p.add_argument("--session", help="filter to one session id")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_session_log)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_session")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
