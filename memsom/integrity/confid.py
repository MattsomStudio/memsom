"""memsom_confid — Bell-LaPadula CONFIDENTIALITY axis, orthogonal to Biba integrity.

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

import memsom
from memsom.storage import schema as memsom_schema

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
    memsom_schema.add_column(
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

    Bypass-2G guard (conf-laundering): an *archived* node is frozen — it has
    already been consolidated into one or more derived summary nodes whose
    confidentiality is the high-water mark of these very parents.  Lowering an
    archived source's conf would let the next recompute_conf drag the
    SECRET-derived summary down to PUBLIC.  Archived nodes therefore accept
    raises but REFUSE any downgrade.  (Live, non-archived sources remain freely
    reclassifiable by the operator — that is the legitimate classification root.)
    """
    level = parse_conf(level)
    node = memsom.get_node(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id {nid}")
    row = conn.execute("SELECT conf_label FROM nodes WHERE id = ?", (nid,)).fetchone()
    current = row[0]
    if level < current and memsom_schema.column_exists(conn, "nodes", "archived"):
        arow = conn.execute("SELECT archived FROM nodes WHERE id = ?", (nid,)).fetchone()
        if arow and arow[0]:
            raise ValueError(
                f"cannot lower confidentiality on archived node {nid} "
                f"({CONF_NAME[current]} -> {CONF_NAME[level]}): archived nodes are "
                f"frozen — downgrading them would launder a derived summary"
            )
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
    node = memsom.get_node(conn, nid)
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
    """Recompute conf_label (high-water MAX) for every live agent-derived node,
    ORDER-INDEPENDENTLY.

    Returns list of (id, old, new) for nodes whose label net-changed.

    CONFID-1: the old single `ORDER BY id` pass assumed id order == topological
    order (parent.id < child.id).  That holds for in-process derive_node but is
    FALSE after a federation import, which assigns local ids in changeset arrival
    order — a child can land with a lower id than its parent, get computed against
    a stale parent conf, and never be re-raised.  The integrity counterpart
    (memsom_recompute.effective_labels) was hardened to an order-independent walk;
    this is the confidentiality mirror.  We iterate the ordered pass to a fixpoint
    (Gauss-Seidel over the finite {0..3} lattice with source nodes as constants),
    so a child is re-visited after its parent settles regardless of id order.

    Idempotent (CONFID-2): on an already-correct DB the first pass changes
    nothing, the loop exits, and [] is returned.

    Residual: a provenance CYCLE among agent-derived nodes (only reachable via a
    federation import with no acyclicity check — see RECOMPUTE-1) can sustain a
    stable spurious high value; the pass count is bounded so this can never hang.
    """
    rows = conn.execute(
        "SELECT id FROM nodes WHERE tombstoned = 0 AND channel = 'agent-derived' ORDER BY id"
    ).fetchall()
    ids = [r[0] for r in rows]

    # Track each changed node's ORIGINAL conf and its FINAL conf so the returned
    # (id, old, new) reflects the net change across all passes.
    original = {}
    final = {}
    with conn:
        # A DAG converges in <= depth passes (<= len(ids)); +1 guards the exit.
        for _ in range(len(ids) + 1):
            progressed = False
            for nid in ids:
                old = conn.execute(
                    "SELECT conf_label FROM nodes WHERE id = ?", (nid,)
                ).fetchone()[0]
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
                    if nid not in original:
                        original[nid] = old
                    final[nid] = new
                    progressed = True
            if not progressed:
                break

    return [
        (nid, original[nid], final[nid])
        for nid in ids
        if nid in final and final[nid] != original[nid]
    ]


def effective_confs(conn: sqlite3.Connection) -> dict:
    """Read-only: {id: effective_conf} for every live node — the value
    recompute_conf_all WOULD write, computed without mutating.

    Source nodes are fixed points (stored conf); a derived node = max over its
    live parents' effective conf (high-water, transitive), order-independent via
    the same bounded fixpoint recompute_conf_all uses. This is the confidentiality
    mirror of memsom_recompute.effective_labels, so an auditor (memsom_heal) can
    flag exactly the rows recompute_conf_all would fix — including multi-hop
    laundered chains a single-level check misses (HEAL-1).
    """
    nodes = conn.execute(
        "SELECT id, channel, conf_label FROM nodes WHERE tombstoned = 0"
    ).fetchall()
    conf = {nid: c for nid, _ch, c in nodes}
    derived = [nid for nid, ch, _c in nodes if ch == "agent-derived"]
    parents = {}
    for child, parent in conn.execute(
        "SELECT e.child, e.parent FROM edges e"
        " JOIN nodes n ON n.id = e.parent WHERE n.tombstoned = 0"
    ):
        parents.setdefault(child, []).append(parent)
    # Bounded Gauss-Seidel to a fixpoint (sources constant); +1 guards a cycle.
    for _ in range(len(derived) + 1):
        progressed = False
        for nid in derived:
            ps = parents.get(nid)
            if not ps:
                continue  # zero live parents: keep stored (matches recompute_conf_all)
            new = max(conf[p] for p in ps)
            if new != conf[nid]:
                conf[nid] = new
                progressed = True
        if not progressed:
            break
    return conf


def sources_for_clearance(conn: sqlite3.Connection, clearance) -> list:
    """Return ids of live source nodes whose conf_label <= clearance (no read-up).

    Clearance is parsed via parse_conf (int 0-3 or string name).
    """
    clearance = parse_conf(clearance)
    # Defense-in-depth (Bypass-2G): archived nodes are consolidated-away and must
    # never re-enter a clearance read pool — exclude them when the column exists.
    archived_clause = ""
    if memsom_schema.column_exists(conn, "nodes", "archived"):
        archived_clause = " AND archived = 0"
    rows = conn.execute(
        "SELECT id FROM nodes "
        "WHERE tombstoned = 0 AND channel != 'agent-derived' AND conf_label <= ?"
        + archived_clause +
        " ORDER BY id",
        (clearance,)
    ).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# CLI layer
# ---------------------------------------------------------------------------

def _cmd_classify(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        level = parse_conf(args.level)
        classify(conn, args.id, level)
        print(f"[{args.id}] conf={CONF_NAME[level]}")
    finally:
        conn.close()


def _cmd_conf_recompute(args):
    conn = memsom.get_connection()
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
    p = argparse.ArgumentParser(prog="memsom_confid")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
