"""memdag_recompute — multi-hop integrity recompute (the cut 'min beyond one hop').

After a manual elevation or any label change on a source node, descendant
agent-derived nodes must re-floor to the true transitive minimum — but must
NOT clobber manually elevated nodes (those are fixed points).

Public API
----------
recompute_label(conn, nid)     -> int          (ValueError if unknown id)
recompute_node(conn, nid)      -> (old, new)   (writes only if changed and not fixed point)
recompute_all(conn)            -> list[(id, old, new)]   (changed rows only)

CLI
---
recompute [id] [--all]   (exactly one of the two required)

register(subparsers) mounts this into a unified CLI.
"""

import sys

import memdag
import memdag_schema


# ---------------------------------------------------------------------------
# Migration stub — no schema changes needed for recompute
# ---------------------------------------------------------------------------

def migrate(conn):
    """No-op migration; provided for CLI uniformity."""
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _elevated_ids(conn):
    """Return the set of node ids that have a row in the elevations table.

    Detects the table WITHOUT importing memdag_trust (no circular import).
    Returns an empty set if the table does not exist yet.
    """
    if memdag_schema.table_exists(conn, "elevations"):
        return {r[0] for r in conn.execute("SELECT DISTINCT node FROM elevations")}
    return set()


def _is_fixed_point(node, elevated):
    """A node is a fixed point when its stored label is authoritative and the
    walk must NOT look past it.

    Fixed-point conditions:
      (a) channel != 'agent-derived' — source nodes keep their channel label.
      (b) the node has a row in the elevations table — manual trust elevation.
    """
    return node["channel"] != "agent-derived" or node["id"] in elevated


def _effective(conn, start_id, memo, elevated):
    """Compute the effective integrity label for *start_id* using an iterative
    post-order DFS (explicit stack; a 1500-deep chain must not hit the recursion
    limit).

    Algorithm
    ---------
    - If *start_id* is already in memo -> return immediately.
    - If *start_id* is a fixed point -> its stored label is authoritative.
    - Otherwise, effective(n) = min(effective(p) for each LIVE immediate parent p).
    - Zero live parents (orphaned derived node) -> fall back to stored label.
    - Cycle guard: an in_progress set detects back-edges; when a back-edge hits
      a node currently being processed, use its stored label and continue.

    Stack: each entry is [nid, parent_iter_or_None, partial_min_or_None]
    - parent_iter_or_None = None  -> node not yet expanded (first visit)
    - parent_iter_or_None = iter  -> node expanded, iterating over parents
    partial_min accumulates min over processed parents.

    When a parent result comes back we pop the parent frame and the callee's
    frame is now top-of-stack — we incorporate the result into its partial_min.
    """
    if start_id in memo:
        return memo[start_id]

    in_progress = set()

    # Stack frame: [nid, parent_iter, partial_min, stored_label]
    # parent_iter=None => first visit not yet done
    stack = [[start_id, None, None, None]]

    while stack:
        nid, parent_iter, partial_min, stored = stack[-1]

        # Case A: already memoised (possible via diamond — another path got here first)
        if nid in memo:
            val = memo[nid]
            stack.pop()
            if stack:
                caller = stack[-1]
                # Incorporate this result into caller's partial_min
                caller[2] = val if caller[2] is None else min(caller[2], val)
            continue

        if parent_iter is None:
            # ---- First visit ----

            # Cycle detection
            if nid in in_progress:
                # Back-edge: use stored label to break the cycle
                node = memdag.get_node(conn, nid)
                val = node["label"] if node else 0
                stack.pop()
                if stack:
                    caller = stack[-1]
                    caller[2] = val if caller[2] is None else min(caller[2], val)
                continue

            node = memdag.get_node(conn, nid)
            if node is None:
                # Unknown id — be safe
                stack.pop()
                if stack:
                    caller = stack[-1]
                    caller[2] = 0 if caller[2] is None else min(caller[2], 0)
                continue

            node_stored = node["label"]

            if _is_fixed_point(node, elevated):
                memo[nid] = node_stored
                stack.pop()
                if stack:
                    caller = stack[-1]
                    caller[2] = node_stored if caller[2] is None else min(caller[2], node_stored)
                continue

            # Not a fixed point — expand live immediate parents
            in_progress.add(nid)
            parent_rows = memdag.parents_of(conn, nid)
            # parents_of tuple: (id, content, channel, label, source_ref, created_at,
            #                    tombstoned, tombstoned_at, revoke_reason)
            live_parent_ids = [r[0] for r in parent_rows if r[6] == 0]

            if not live_parent_ids:
                # Zero live parents: fall back to stored label
                memo[nid] = node_stored
                in_progress.discard(nid)
                stack.pop()
                if stack:
                    caller = stack[-1]
                    caller[2] = node_stored if caller[2] is None else min(caller[2], node_stored)
                continue

            # Update frame: mark expanded, store the parent iterator and stored label
            stack[-1][1] = iter(live_parent_ids)
            stack[-1][2] = None   # partial_min reset
            stack[-1][3] = node_stored

        else:
            # ---- Subsequent visits: try to push the next parent ----
            next_pid = next(parent_iter, None)
            if next_pid is not None:
                # Push the next parent for processing
                stack.append([next_pid, None, None, None])
            else:
                # All parents processed — compute result from partial_min
                computed = stack[-1][2]
                node_stored = stack[-1][3]
                memo[nid] = computed if computed is not None else node_stored
                in_progress.discard(nid)
                stack.pop()
                if stack:
                    caller = stack[-1]
                    val = memo[nid]
                    caller[2] = val if caller[2] is None else min(caller[2], val)

    return memo.get(start_id, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recompute_label(conn, nid):
    """Return the correct effective integrity label for node *nid*.

    Raises ValueError if the node does not exist.
    If *nid* is a fixed point (source channel or manually elevated), the stored
    label is returned immediately without any traversal.
    """
    node = memdag.get_node(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id: {nid}")

    elevated = _elevated_ids(conn)
    if _is_fixed_point(node, elevated):
        return node["label"]

    memo = {}
    return _effective(conn, nid, memo, elevated)


def recompute_node(conn, nid):
    """Recompute and (if changed) update the label for node *nid*.

    Returns (old_label, new_label).  If *nid* is a fixed point the stored label
    is returned as both old and new — no write is performed.

    Raises ValueError if *nid* does not exist.
    """
    node = memdag.get_node(conn, nid)
    if node is None:
        raise ValueError(f"unknown node id: {nid}")

    old = node["label"]
    elevated = _elevated_ids(conn)

    if _is_fixed_point(node, elevated):
        return (old, old)

    memo = {}
    new = _effective(conn, nid, memo, elevated)

    if new != old:
        with conn:
            conn.execute("UPDATE nodes SET label=? WHERE id=?", (new, nid))

    return (old, new)


def recompute_all(conn):
    """Recompute labels for every live agent-derived node that is not a fixed point.

    Returns a list of (id, old_label, new_label) for rows that changed.
    All updates are applied in a single transaction.

    Guarantees:
    - Order-independent: eff() is defined off fixed points, not off other nodes'
      stored labels, so the result is the same regardless of processing order.
    - A second call with no external label changes is a strict no-op (returns []).
    """
    elevated = _elevated_ids(conn)
    memo = {}
    changes = []

    rows = conn.execute(
        "SELECT id, label FROM nodes WHERE tombstoned=0 AND channel='agent-derived'"
        " ORDER BY id"
    ).fetchall()

    for nid, old in rows:
        if nid in elevated:
            continue
        new = _effective(conn, nid, memo, elevated)
        if new != old:
            changes.append((nid, old, new))

    if changes:
        with conn:
            for nid, _old, new in changes:
                conn.execute("UPDATE nodes SET label=? WHERE id=?", (new, nid))

    return changes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_recompute(args):
    has_id = args.id is not None
    has_all = args.all

    if has_id == has_all:  # both True or both False
        print("usage: recompute <id> | recompute --all", file=sys.stderr)
        sys.exit(2)

    conn = memdag.get_connection()
    try:
        if has_all:
            changes = recompute_all(conn)
            if not changes:
                print("no changes")
            else:
                for nid, old, new in changes:
                    print(f"[{nid}] AGENT-DERIVED  {memdag.NAME[old]} -> {memdag.NAME[new]}")
        else:
            nid = args.id
            try:
                old, new = recompute_node(conn, nid)
            except ValueError as exc:
                print(f"[memdag] {exc}", file=sys.stderr)
                sys.exit(1)
            if old == new:
                print(f"[{nid}] unchanged ({memdag.NAME[old]})")
            else:
                print(f"[{nid}] AGENT-DERIVED  {memdag.NAME[old]} -> {memdag.NAME[new]}")
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser("recompute",
                               help="recompute multi-hop integrity labels for derived nodes")
    p.add_argument("id", type=int, nargs="?", default=None,
                   help="node id to recompute (mutually exclusive with --all)")
    p.add_argument("--all", action="store_true",
                   help="recompute all live agent-derived nodes")
    p.set_defaults(func=cmd_recompute)
