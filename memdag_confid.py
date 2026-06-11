"""memdag_confid — Bell-LaPadula CONFIDENTIALITY axis, orthogonal to Biba integrity.

Integrity uses MIN (low-water-mark; trust floors with the weakest parent).
Confidentiality uses MAX (high-water-mark; reading one secret source makes the
derived answer secret — no read-up).

Schema migration: adds conf_label INTEGER NOT NULL DEFAULT 0 to nodes.
Default 0 (public) keeps demo #1 byte-identical.

CONF_RANK / CONF_NAME constants; parse_conf() for CLI + sibling modules.

Public API:
    migrate(conn)
    classify(conn, nid, level)
    recompute_conf(conn, nid) -> (old, new)
    recompute_conf_all(conn) -> list[(id, old, new)]
    sources_for_clearance(conn, clearance) -> list[int]

CLI sub-commands registered via register(subparsers).
"""

import argparse
import sqlite3

import memdag
import memdag_schema

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_RANK = {"public": 0, "internal": 1, "secret": 2, "topsecret": 3}
CONF_NAME = {0: "PUBLIC", 1: "INTERNAL", 2: "SECRET", 3: "TOPSECRET"}


def parse_conf(value) -> int:
    """Accept int 0-3 or string name (case-insensitive). Raise ValueError otherwise."""
    if isinstance(value, int):
        if value not in CONF_NAME:
            raise ValueError(f"conf level {value!r} out of range 0-3")
        return value
    if isinstance(value, str):
        key = value.lower()
        if key in CONF_RANK:
            return CONF_RANK[key]
        # try numeric string
        try:
            n = int(key)
        except ValueError:
            raise ValueError(f"unknown conf level {value!r}") from None
        if n not in CONF_NAME:
            raise ValueError(f"conf level {n!r} out of range 0-3")
        return n
    raise ValueError(f"unrecognised conf level type: {type(value)}")


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: add conf_label column to nodes if absent. DEFAULT 0 = public."""
    memdag_schema.add_column(
        conn, "nodes", "conf_label",
        "INTEGER NOT NULL DEFAULT 0"
    )


# ---------------------------------------------------------------------------
# Library API (no print, no sys.exit)
# ---------------------------------------------------------------------------

def classify(conn: sqlite3.Connection, nid: int, level) -> None:
    """Set conf_label on node *nid* to *level* (via parse_conf).

    Raises ValueError for unknown id or bad level.
    Works on any channel — sources are classification roots; derived nodes can
    also be manually classified (overridable later by recompute_conf).
    """
    level = parse_conf(level)
    node = memdag.get_node(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id {nid}")
    with conn:
        conn.execute("UPDATE nodes SET conf_label = ? WHERE id = ?", (level, nid))


def recompute_conf(conn: sqlite3.Connection, nid: int):
    """Recompute confidentiality for a single derived node.

    Returns (old, new) ints.
    - Non-derived (source) nodes are classification roots: returns (c, c) untouched.
    - Derived nodes: new = MAX(conf_label of live immediate parents).
      If there are zero live parents, keep the stored label (no crash).
    - Writes to DB only when old != new.
    """
    node = memdag.get_node(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id {nid}")

    # Fetch current conf_label
    row = conn.execute("SELECT conf_label FROM nodes WHERE id = ?", (nid,)).fetchone()
    old = row[0]

    if node["channel"] != "agent-derived":
        # Source nodes are classification roots — don't touch them.
        return (old, old)

    # Live immediate parents only (tombstoned == 0)
    parent_rows = conn.execute(
        "SELECT n.conf_label FROM edges e "
        "JOIN nodes n ON n.id = e.parent "
        "WHERE e.child = ? AND n.tombstoned = 0",
        (nid,)
    ).fetchall()

    if not parent_rows:
        # No live parents: keep stored label, no write.
        return (old, old)

    new = max(r[0] for r in parent_rows)

    if new != old:
        with conn:
            conn.execute("UPDATE nodes SET conf_label = ? WHERE id = ?", (new, nid))

    return (old, new)


def recompute_conf_all(conn: sqlite3.Connection):
    """Recompute conf_label for every live agent-derived node in id order.

    Returns list of (id, old, new) for nodes whose label changed.

    id order = monotone derivation order: a parent's row is always finalized
    before any child that derives from it is visited, so one ordered pass
    propagates the high-water mark correctly through multi-hop chains.

    Idempotent: a second call on an already-correct DB returns [].
    """
    rows = conn.execute(
        "SELECT id FROM nodes WHERE tombstoned = 0 AND channel = 'agent-derived' ORDER BY id"
    ).fetchall()

    changed = []
    with conn:
        for (nid,) in rows:
            row = conn.execute("SELECT conf_label FROM nodes WHERE id = ?", (nid,)).fetchone()
            old = row[0]

            parent_rows = conn.execute(
                "SELECT n.conf_label FROM edges e "
                "JOIN nodes n ON n.id = e.parent "
                "WHERE e.child = ? AND n.tombstoned = 0",
                (nid,)
            ).fetchall()

            if not parent_rows:
                continue

            new = max(r[0] for r in parent_rows)
            if new != old:
                conn.execute("UPDATE nodes SET conf_label = ? WHERE id = ?", (new, nid))
                changed.append((nid, old, new))

    return changed


def sources_for_clearance(conn: sqlite3.Connection, clearance) -> list:
    """Return ids of live source nodes whose conf_label <= clearance (no read-up).

    Clearance is parsed via parse_conf (int 0-3 or string name).
    """
    clearance = parse_conf(clearance)
    rows = conn.execute(
        "SELECT id FROM nodes "
        "WHERE tombstoned = 0 AND channel != 'agent-derived' AND conf_label <= ? "
        "ORDER BY id",
        (clearance,)
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# CLI layer
# ---------------------------------------------------------------------------

def _cmd_classify(args):
    conn = memdag.get_connection()
    try:
        migrate(conn)
        level = parse_conf(args.level)
        classify(conn, args.id, level)
        print(f"[{args.id}] conf={CONF_NAME[level]}")
    finally:
        conn.close()


def _cmd_conf_recompute(args):
    conn = memdag.get_connection()
    try:
        migrate(conn)
        if args.all:
            changed = recompute_conf_all(conn)
            if not changed:
                print("conf-recompute: nothing changed")
            else:
                for nid, old, new in changed:
                    print(f"[{nid}] conf {CONF_NAME[old]} -> {CONF_NAME[new]}")
        elif args.id is not None:
            old, new = recompute_conf(conn, args.id)
            if old == new:
                print(f"[{args.id}] conf {CONF_NAME[old]} (unchanged)")
            else:
                print(f"[{args.id}] conf {CONF_NAME[old]} -> {CONF_NAME[new]}")
        else:
            import sys
            print("conf-recompute: specify <id> or --all", file=sys.stderr)
            sys.exit(1)
    finally:
        conn.close()


def register(subparsers) -> None:
    """Mount classify and conf-recompute sub-commands onto *subparsers*."""
    # classify <id> --level <name>
    p_classify = subparsers.add_parser("classify", help="Set confidentiality label on a node")
    p_classify.add_argument("id", type=int)
    p_classify.add_argument("--level", required=True,
                            help="public | internal | secret | topsecret (or 0-3)")
    p_classify.set_defaults(func=_cmd_classify)

    # conf-recompute [id] [--all]
    p_recompute = subparsers.add_parser(
        "conf-recompute",
        help="Recompute confidentiality for one node or all derived nodes"
    )
    p_recompute.add_argument("id", type=int, nargs="?", default=None)
    p_recompute.add_argument("--all", action="store_true")
    p_recompute.set_defaults(func=_cmd_conf_recompute)


def main(argv=None):
    """Thin CLI wrapper — for direct invocation and test harness."""
    import sys
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memdag_confid")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
