"""memsom_stale — STALENESS CASCADE (verifiable correctness-over-time).

The fourth flag on a node, orthogonal to the three security flags:

  - tombstoned = revoked (liveness dies)            [memsom.py]
  - redacted   = payload destroyed                  [memsom_redact.py]
  - quarantined/archived = consolidation states     [memsom_quarantine/compact]
  - stale      = a SOURCE this node derives from has since CHANGED

Staleness is DELIBERATELY NOT a security dimension. It never gates reads on its
own: a stale node still answers, the answer is just FLAGGED, and exclusion is an
explicit operator opt-in (`ask --fresh-only`). So `stale` is NOT folded into
memsom_schema.taint_filter_clauses (the always-on dead filter); it lives here as
a separate, opt-in WHERE fragment. This keeps "outdated" from silently becoming
"dead" (availability) while still letting the high-stakes caller demand fresh.

Mechanism (reuses the revocation-cascade engine, aimed at a new trigger):
  1. ingest detects a SOURCE change (same source_ref, new content_hash, live old
     version) -> records old->new in source_supersedes + fires the cascade.
  2. mark_stale_cascade marks the old version + every transitive descendant stale
     (the memsom.revoke_cascade twin: same CASCADE_CTE walk, sets stale=1 not
     tombstoned=1, first-staleness-wins).
  3. ask/retrieve DISCLOSE staleness by default; `--fresh-only` excludes.
  4. freshen(node) — the ONLY thing that rewrites provenance — rewires the stale
     node's edge old->fresh source, then regenerates (memsom_rederive.regenerate),
     archiving the old and re-flooring the label. Explicit + audited; never automatic.

Migration adds three columns to nodes (stale triple), a source_supersedes link
table, and a stale_log audit table.

Public API
----------
migrate(conn)                                                idempotent
mark_stale_cascade(conn, seed, reason)       -> int          newly-stale count
stale_set(conn, seed)                        -> list[(id, channel, stale)]
unstale(conn, node_id)                       -> int          cleared count (audited)
record_source_supersession(conn, old_id, new_id, source_ref, detected_at=None)
superseding_version(conn, old_id)            -> int | None   direct successor
fresh_version_for(conn, node_id)             -> int | None   live head of the chain
stale_exclude_clauses(conn)                  -> (clauses, params)   OPT-IN only
stale_annotations(conn, node_ids)            -> dict[int, dict]
stale_status(conn, node_id)                  -> dict
on_reingest_supersede(conn, old_id, new_id, source_ref) -> int
freshen_preview(conn, node_id)               -> list[(old_parent, fresh_parent)]
freshen(conn, node_id)                       -> dict          (OUTSIDE a txn)

CLI
---
stale-cascade <id> [--reason r] [--yes]    (dry-run by default)
freshen <id> [--yes]                       (dry-run by default)
stale-status <id>
unstale <id> [--yes]
register(subparsers)   — standard mount point
main(argv=None)        — thin wrapper for tests
"""

import argparse
import sys

import memsom
import memsom_schema
import memsom_rederive


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Add the stale triple + supersession + audit tables idempotently."""
    memsom_schema.add_column(conn, "nodes", "stale",        "INTEGER NOT NULL DEFAULT 0")
    memsom_schema.add_column(conn, "nodes", "stale_at",     "TEXT")
    memsom_schema.add_column(conn, "nodes", "stale_reason", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_stale ON nodes(stale)")
    # source_supersedes: old->new for SOURCE nodes. derivation_recipe.supersedes
    # only covers DERIVED nodes; a re-ingested source has no recipe, so it needs
    # its own home. old_id PK = each old version superseded at most once FORWARD,
    # so re-ingest builds a chain 7->8->9 and fresh_version_for walks to the head.
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS source_supersedes (
    old_id      INTEGER PRIMARY KEY REFERENCES nodes(id),
    new_id      INTEGER NOT NULL    REFERENCES nodes(id),
    source_ref  TEXT,
    detected_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_source_supersedes_new ON source_supersedes(new_id);
  CREATE INDEX IF NOT EXISTS idx_source_supersedes_ref ON source_supersedes(source_ref);""")
    # stale_log: first-class audit, mirrors redaction_log — staleness is the moat,
    # every mark / supersede / rewire / regen / unstale is recorded.
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS stale_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event      TEXT NOT NULL,
    node_id    INTEGER,
    related_id INTEGER,
    source_ref TEXT,
    reason     TEXT,
    at         TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_stale_log_node ON stale_log(node_id);""")


def _now():
    return memsom.now_iso()


def _log(conn, event, node_id=None, related_id=None, source_ref=None, reason=None, at=None):
    """Append one stale_log row. Caller owns the transaction boundary."""
    conn.execute(
        "INSERT INTO stale_log(event, node_id, related_id, source_ref, reason, at)"
        " VALUES (?,?,?,?,?,?)",
        (event, node_id, related_id, source_ref, reason, at or _now()))


# ---------------------------------------------------------------------------
# Cascade engine (the memsom.revoke_cascade twin: marks stale, never kills)
# ---------------------------------------------------------------------------

def mark_stale_cascade(conn, seed, reason):
    """Mark *seed* and every transitive descendant STALE (not tombstoned).

    Structurally identical to memsom.revoke_cascade: one atomic UPDATE over the
    CASCADE_CTE (UNION-deduped, so cycles terminate and diamonds are visited
    once), `WHERE stale = 0` so the FIRST staleness wins (re-marking preserves a
    node's original stale_at/reason). Liveness and edges are untouched. Returns
    the count of newly-stale nodes (SELECT changes(), since WITH-prefixed DML
    misreports cursor.rowcount).
    """
    migrate(conn)
    ts = _now()
    with conn:
        conn.execute(
            memsom.CASCADE_CTE + """
            UPDATE nodes SET stale = 1, stale_at = ?,
                   stale_reason = CASE WHEN id = ? THEN ?
                                       ELSE 'stale cascade from node ' || ? END
             WHERE id IN (SELECT id FROM descendants) AND stale = 0""",
            (seed, ts, seed, reason, seed))
        n = conn.execute("SELECT changes()").fetchone()[0]
        _log(conn, "cascade", node_id=seed, reason=reason, at=ts)
    return n


def stale_set(conn, seed):
    """Preview the cascade set: (id, channel, stale) for seed + all descendants."""
    migrate(conn)
    return conn.execute(
        memsom.CASCADE_CTE + " SELECT n.id, n.channel, COALESCE(n.stale, 0) FROM nodes n"
        " WHERE n.id IN (SELECT id FROM descendants) ORDER BY n.id", (seed,)).fetchall()


def unstale(conn, node_id):
    """Clear the stale flag on ONE node (audited). Returns 1 if cleared, else 0.

    Used after a freshen rewire whose regenerate was a byte-identical no-op (the
    node now points at the fresh source and its content is current), and exposed
    as an operator escape hatch.
    """
    migrate(conn)
    with conn:
        conn.execute(
            "UPDATE nodes SET stale = 0, stale_at = NULL, stale_reason = NULL"
            " WHERE id = ? AND stale = 1", (node_id,))
        n = conn.execute("SELECT changes()").fetchone()[0]
        if n:
            _log(conn, "unstale", node_id=node_id)
    return n


# ---------------------------------------------------------------------------
# Source supersession (the new old->new link; derivation_recipe is DERIVED-only)
# ---------------------------------------------------------------------------

def record_source_supersession(conn, old_id, new_id, source_ref, detected_at=None):
    """Record that *new_id* supersedes *old_id* for *source_ref*. Idempotent.

    INSERT OR IGNORE on the old_id PK -> once-forward; a second re-ingest records
    new->newer separately, building a chain the resolvers walk to the live head.
    """
    migrate(conn)
    ts = detected_at or _now()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO source_supersedes(old_id, new_id, source_ref, detected_at)"
            " VALUES (?,?,?,?)", (old_id, new_id, source_ref, ts))
        _log(conn, "supersede", node_id=old_id, related_id=new_id,
             source_ref=source_ref, at=ts)


def superseding_version(conn, old_id):
    """The node that directly superseded *old_id*, or None."""
    if not memsom_schema.column_exists(conn, "nodes", "stale"):
        return None  # module never ran -> source_supersedes absent
    row = conn.execute(
        "SELECT new_id FROM source_supersedes WHERE old_id = ?", (old_id,)).fetchone()
    return row[0] if row else None


def fresh_version_for(conn, node_id):
    """Walk the supersession chain from *node_id* to its live head.

    Returns the newest version id, or None if *node_id* was never superseded.
    Cycle-guarded (a malformed chain cannot loop).
    """
    seen = {node_id}
    cur = node_id
    while True:
        nxt = superseding_version(conn, cur)
        if nxt is None or nxt in seen:
            break
        seen.add(nxt)
        cur = nxt
    return cur if cur != node_id else None


# ---------------------------------------------------------------------------
# Read-time disclosure + opt-in exclusion
# ---------------------------------------------------------------------------

def stale_exclude_clauses(conn):
    """OPT-IN WHERE fragment to drop stale nodes from a pool (`--fresh-only`).

    Returns (["stale = 0"], []) once the module has run, else ([], []). This is
    deliberately SEPARATE from memsom_schema.taint_filter_clauses: staleness is
    not a security dimension, so it must never exclude by default. Carries no
    bound parameter, so appending it never disturbs a positional `conf_label <= ?`.
    """
    if memsom_schema.column_exists(conn, "nodes", "stale"):
        return (["stale = 0"], [])
    return ([], [])


def stale_annotations(conn, node_ids):
    """{id: {stale_at, reason, fresh_id}} for whichever of *node_ids* are stale.

    fresh_id = the live head of the supersession chain (what to repoint at), or
    None. Used by `ask` to flag exactly the cited sources that have gone stale.
    """
    if not node_ids or not memsom_schema.column_exists(conn, "nodes", "stale"):
        return {}
    ids = list(dict.fromkeys(node_ids))  # de-dup, preserve order
    qmarks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, stale_at, stale_reason FROM nodes"
        f" WHERE id IN ({qmarks}) AND stale = 1", tuple(ids)).fetchall()
    return {
        nid: {"stale_at": at, "reason": reason, "fresh_id": fresh_version_for(conn, nid)}
        for nid, at, reason in rows
    }


def _passes_pool_gate(conn, nid, clearance_int):
    """True if *nid* may enter an answer pool — the EXACT filter
    _build_retrieve_pool applies, for one id. A fresh node injected by
    substitute_fresh bypassed retrieval, so it must re-clear this gate: a
    supersedes hop can never become a backdoor that pulls a tombstoned / redacted
    / archived / quarantined / above-clearance node into the answer.
    """
    clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=clearance_int)
    clauses.append("channel != 'agent-derived'")
    return conn.execute(
        "SELECT 1 FROM nodes WHERE id = ? AND " + " AND ".join(clauses),
        [nid] + params).fetchone() is not None


def substitute_fresh(conn, pool, clearance_int):
    """Staleness-aware hybrid serving (read-time, NON-mutating).

    For each STALE source in *pool* that has a live fresh head clearing the pool
    gate, REPLACE the stale row with the fresh row (the deterministic supersedes
    edge rescuing what retrieval surfaced) — so the composed answer resolves to
    the CURRENT value instead of serving the stale one. Substitution, not
    augmentation: keeping both would still serve stale.

    *pool* is a list of (id, content, channel, label, source_ref) rows. Returns
    (new_pool, substitutions) where substitutions = [(old_id, fresh_id), ...].
    The store is untouched — this is a per-answer pool swap, distinct from the
    `freshen` VERB which rewrites provenance. A stale node with no fresh head, or
    whose fresh head fails the gate, is KEPT (falls back to default disclosure).
    """
    if not pool:
        return pool, []
    ann = stale_annotations(conn, [r[0] for r in pool])  # one call over all ids
    if not ann:
        return pool, []

    present = {r[0] for r in pool}
    new_pool = []
    substitutions = []
    for row in pool:
        sid = row[0]
        info = ann.get(sid)
        fresh_id = info.get("fresh_id") if info else None
        if fresh_id is not None and _passes_pool_gate(conn, fresh_id, clearance_int):
            substitutions.append((sid, fresh_id))
            if fresh_id in present:
                continue  # drop stale; fresh already in the pool (dedup)
            fresh_row = conn.execute(
                "SELECT id, content, channel, label, source_ref FROM nodes WHERE id = ?",
                (fresh_id,)).fetchone()
            if fresh_row is not None:
                new_pool.append(fresh_row)
                present.add(fresh_id)
            else:                     # fresh head vanished between calls -> keep stale
                new_pool.append(row)
        else:
            new_pool.append(row)      # no fresh / gate-fail -> keep + let [STALE] disclose
    return new_pool, substitutions


def stale_status(conn, node_id):
    """Display reader for `stale-status <id>`. ValueError on unknown id."""
    migrate(conn)
    row = conn.execute(
        "SELECT id, stale, stale_at, stale_reason FROM nodes WHERE id = ?",
        (node_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown node id: {node_id}")
    return {
        "id": row[0], "stale": bool(row[1]),
        "stale_at": row[2], "stale_reason": row[3],
        "superseded_by": superseding_version(conn, node_id),
        "fresh_head": fresh_version_for(conn, node_id),
        "stale_descendants": [r[0] for r in stale_set(conn, node_id) if r[2]],
    }


# ---------------------------------------------------------------------------
# Ingest auto-detect trigger
# ---------------------------------------------------------------------------

def on_reingest_supersede(conn, old_id, new_id, source_ref):
    """Record old->new and fire the staleness cascade from old_id.

    Called best-effort from ingest after a changed-content re-ingest mints a new
    node. Returns the count of newly-stale nodes. Runs its own transactions; the
    new node is already committed by ingest before this is called.
    """
    migrate(conn)
    record_source_supersession(conn, old_id, new_id, source_ref)
    return mark_stale_cascade(
        conn, old_id,
        f"source superseded by node {new_id} (re-ingest of {source_ref})")


# ---------------------------------------------------------------------------
# Resolution: freshen (the ONLY provenance rewrite; explicit + audited)
# ---------------------------------------------------------------------------

def _is_live_source(conn, nid):
    """True if *nid* is a fully-untainted node (safe to repoint onto)."""
    clauses = ["tombstoned = 0"]
    for col in ("redacted", "archived"):
        if memsom_schema.column_exists(conn, "nodes", col):
            clauses.append(f"COALESCE({col}, 0) = 0")
    if memsom_schema.column_exists(conn, "nodes", "status"):
        clauses.append("(status IS NULL OR status != 'quarantined')")
    return conn.execute(
        "SELECT 1 FROM nodes WHERE id = ? AND " + " AND ".join(clauses),
        (nid,)).fetchone() is not None


def freshen_preview(conn, node_id):
    """The rewires `freshen` WOULD apply: list of (old_parent, fresh_parent).

    A direct parent qualifies when it has a superseding LIVE version (T3: never
    repoint onto a tombstoned/redacted/archived/quarantined node).
    """
    migrate(conn)
    if memsom.get_node(conn, node_id) is None:
        raise ValueError(f"unknown node id: {node_id}")
    parent_ids = [r[0] for r in conn.execute(
        "SELECT parent FROM edges WHERE child = ?", (node_id,)).fetchall()]
    rewires = []
    for p in parent_ids:
        head = fresh_version_for(conn, p)
        if head is not None and head != p and _is_live_source(conn, head):
            rewires.append((p, head))
    return rewires


def freshen(conn, node_id):
    """Repoint *node_id* at the fresh version of each stale source parent, then
    regenerate. The ONE provenance-rewriting operation — explicit, atomic, audited.

    Steps: (1) drop edge node->old_parent, add edge node->fresh_parent (one txn +
    audit row per rewire); (2) memsom_rederive.regenerate(node) replays the recipe
    over the now-fresh live parents, mints a new node, archives the old, chains
    supersedes, and re-floors integrity (min) + confidentiality (max). If
    regenerate is a byte-identical no-op (the change didn't move the answer), the
    node is simply un-staled in place.

    MUST be called OUTSIDE an open transaction (regenerate manages its own).
    Returns {node_id, rewired, regenerated}.
    """
    rewires = freshen_preview(conn, node_id)
    if not rewires:
        return {"node_id": node_id, "rewired": [], "regenerated": None,
                "note": "no stale parent with a fresh version"}

    ts = _now()
    with conn:
        for old_p, new_p in rewires:
            conn.execute("DELETE FROM edges WHERE child = ? AND parent = ?",
                         (node_id, old_p))
            conn.execute("INSERT OR IGNORE INTO edges(child, parent) VALUES (?,?)",
                         (node_id, new_p))
            _log(conn, "freshen-rewire", node_id=node_id, related_id=new_p,
                 reason=f"rewired parent {old_p}->{new_p}", at=ts)

    # regenerate reads node_id's now-fresh live parents, mints + archives + chains.
    new_id = memsom_rederive.regenerate(conn, node_id)
    if new_id is None:
        # Rewired to fresh parents but content is byte-identical -> the node is
        # current now; clear its stale flag in place.
        unstale(conn, node_id)
    with conn:
        _log(conn, "freshen-regen", node_id=node_id, related_id=new_id,
             reason="regenerated" if new_id else "regenerate noop (unstaled in place)")
    return {"node_id": node_id, "rewired": rewires, "regenerated": new_id}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_stale_cascade(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        if memsom.get_node(conn, args.id) is None:
            print(f"[memsom-stale] no node [{args.id}]", file=sys.stderr)
            sys.exit(1)
        preview = stale_set(conn, args.id)
        pending = [r for r in preview if not r[2]]
        print(f"will mark {len(pending)} node(s) stale:")
        for nid, ch, st in preview:
            role = "seed" if nid == args.id else "descendant"
            note = "  - already stale, skipped" if st else ""
            print(f"  [{nid}] {ch} ({role}){note}")
        if not args.yes:
            print("nodes stay LIVE and readable; answers that cite them are flagged.")
            print("dry run - re-run with --yes to apply.")
            return
        n = mark_stale_cascade(conn, args.id, args.reason)
        print(f"done - {n} node(s) marked stale, 0 tombstoned, all edges intact.")
    finally:
        conn.close()


def cmd_freshen(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        if memsom.get_node(conn, args.id) is None:
            print(f"[memsom-stale] no node [{args.id}]", file=sys.stderr)
            sys.exit(1)
        rewires = freshen_preview(conn, args.id)
        if not rewires:
            print(f"[{args.id}] has no stale parent with a fresh version - nothing to freshen.")
            return
        print(f"will repoint node [{args.id}] onto {len(rewires)} fresh source(s):")
        for old_p, new_p in rewires:
            print(f"  parent [{old_p}] -> [{new_p}]")
        if not args.yes:
            print("then regenerate from the fresh parents (archives the old, re-floors label).")
            print("dry run - re-run with --yes to apply.")
            return
        result = freshen(conn, args.id)
        new_id = result["regenerated"]
        if new_id:
            print(f"done - regenerated as node [{new_id}]; node [{args.id}] archived.")
        else:
            print(f"done - rewired; answer unchanged, node [{args.id}] un-staled in place.")
    finally:
        conn.close()


def cmd_stale_status(args):
    conn = memsom.get_connection()
    try:
        st = stale_status(conn, args.id)
        flag = "STALE" if st["stale"] else "fresh"
        print(f"node [{st['id']}]: {flag}")
        if st["stale"]:
            print(f"  since:  {st['stale_at']}")
            print(f"  reason: {st['stale_reason']}")
        if st["superseded_by"]:
            print(f"  superseded by: [{st['superseded_by']}] (live head [{st['fresh_head']}])")
        if st["stale_descendants"]:
            ds = ", ".join(f"[{d}]" for d in st["stale_descendants"] if d != st["id"])
            if ds:
                print(f"  stale descendants: {ds}")
    except ValueError as exc:
        print(f"[memsom-stale] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


def cmd_unstale(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        if memsom.get_node(conn, args.id) is None:
            print(f"[memsom-stale] no node [{args.id}]", file=sys.stderr)
            sys.exit(1)
        if not args.yes:
            print(f"will clear the stale flag on node [{args.id}] (single node, no cascade).")
            print("dry run - re-run with --yes to apply.")
            return
        n = unstale(conn, args.id)
        print(f"done - {n} node un-staled." if n else "node was not stale; nothing to do.")
    finally:
        conn.close()


def register(subparsers):
    p_sc = subparsers.add_parser(
        "stale-cascade", help="mark a node + descendants stale (source changed)")
    p_sc.add_argument("id", type=int)
    p_sc.add_argument("--reason", default="source changed")
    p_sc.add_argument("--yes", action="store_true",
                      help="apply; without this flag the command is a dry run")
    p_sc.set_defaults(func=cmd_stale_cascade)

    p_f = subparsers.add_parser(
        "freshen", help="repoint a stale node at the fresh source + regenerate")
    p_f.add_argument("id", type=int)
    p_f.add_argument("--yes", action="store_true",
                     help="apply; without this flag the command is a dry run")
    p_f.set_defaults(func=cmd_freshen)

    p_ss = subparsers.add_parser("stale-status", help="show staleness of a node")
    p_ss.add_argument("id", type=int)
    p_ss.set_defaults(func=cmd_stale_status)

    p_us = subparsers.add_parser("unstale", help="clear the stale flag on one node")
    p_us.add_argument("id", type=int)
    p_us.add_argument("--yes", action="store_true",
                      help="apply; without this flag the command is a dry run")
    p_us.set_defaults(func=cmd_unstale)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-stale")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
