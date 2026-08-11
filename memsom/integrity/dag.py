"""memsom.integrity.dag -- the derivation-DAG store primitives.

insert_node / derive_node / get_node / live_sources / parents_of /
cascade_set / revoke_cascade / CASCADE_CTE moved out of memsom/__init__.py
(Phase 2, the core split).

Two security fixes land as part of this move (SECURITY-REMEDIATION.md Sec3.3):

  MS-10 -- derive_node never wrote conf_label. ARCHITECTURE.md:103 attributes
  "conf = max(used)" to it; only the enhanced CLI remembered to call
  recompute_conf right after derive. Set it here, inside derive_node itself,
  so no caller can forget it again.

  MS-13 -- live_sources filtered tombstoned only. Routed through the ONE
  taint-pool primitive (storage.schema.taint_filter_clauses) instead, the
  same shape interface/cli.py._build_pool already uses. No *clearance*
  argument here on purpose: live_sources feeds the frozen-core `ask`, which
  has no clearance concept at all -- so the safe default is PUBLIC (0), not
  "no filter".
"""

import memsom

# Timestamps go through memsom.now_iso(), not a direct kernel.text import:
# many tests (and the bridge/federation/lifecycle callers) do
# patch.object(memsom, "now_iso", ...) to control clocks, and a direct
# import binds a local reference that patch can never reach.

CASCADE_CTE = """WITH RECURSIVE descendants(id) AS (
  SELECT ? UNION SELECT e.child FROM edges e JOIN descendants d ON e.parent = d.id
)"""  # UNION (not UNION ALL): dedupes -> terminates on cycles, visits diamonds once


def insert_node(conn, content, channel, label=None, source_ref=None):
    if label is None:
        label = memsom.RANK[channel]  # labels come from the channel; explicit label is the
    cur = conn.execute(                # derive/manual-elevation path only
        "INSERT INTO nodes(content, channel, label, source_ref, created_at) VALUES (?,?,?,?,?)",
        (content, channel, label, source_ref, memsom.now_iso()))
    return cur.lastrowid


def derive_node(conn, content, parent_ids):
    if not parent_ids:
        raise ValueError("derived node needs at least one parent")
    # MS-05: widen the liveness check from tombstoned-only to the full taint
    # set (redacted / quarantined / archived) -- a redacted or quarantined
    # parent could otherwise still be derived from (MS-06's delivery
    # mechanism). Deferred import: the frozen core must not gain a load-time
    # dependency on storage.schema; column_exists is a plain read, harmless
    # before BEGIN IMMEDIATE below. Columns are optional -- a bare frozen-core
    # DB with none of the extension modules migrated has none of them, and
    # there is nothing to exclude.
    from memsom.storage import schema as memsom_schema
    redacted_col = "redacted" if memsom_schema.column_exists(conn, "nodes", "redacted") else "0"
    status_col = "status" if memsom_schema.column_exists(conn, "nodes", "status") else "'live'"
    archived_col = "archived" if memsom_schema.column_exists(conn, "nodes", "archived") else "0"
    has_conf = memsom_schema.column_exists(conn, "nodes", "conf_label")
    conf_col = "conf_label" if has_conf else "0"
    qmarks = ",".join("?" * len(parent_ids))
    with conn:  # liveness check + node + edges are ONE unit, write-locked up front:
        if not conn.in_transaction:  # a revoke can't land between check and insert (TOCTOU)
            conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            f"SELECT id, label, tombstoned, {redacted_col}, {status_col}, {archived_col},"
            f" {conf_col}"
            f" FROM nodes WHERE id IN ({qmarks})",
            tuple(parent_ids)).fetchall()
        if len(rows) != len(set(parent_ids)):
            raise ValueError("unknown parent id")
        for _id, _label, tombstoned, redacted, status, archived, _conf in rows:
            if tombstoned:
                raise ValueError("tombstoned parent")
            if redacted:
                raise ValueError("redacted parent")
            if status == "quarantined":
                raise ValueError("quarantined parent")
            if archived:
                raise ValueError("archived parent")
        label = min(r[1] for r in rows)
        nid = insert_node(conn, content, "agent-derived", label)
        conn.executemany("INSERT INTO edges(child, parent) VALUES (?,?)",
                         [(nid, p) for p in set(parent_ids)])
        # MS-10: confidentiality high-water mark, set inside the same
        # transaction as the insert -- mirrors integrity's min-floor above,
        # so a caller that forgets recompute_conf cannot declassify by omission.
        if has_conf:
            conf = max(r[6] for r in rows)
            conn.execute("UPDATE nodes SET conf_label = ? WHERE id = ?", (conf, nid))
    return nid, label


def get_node(conn, nid):
    row = conn.execute(
        "SELECT id, content, channel, label, source_ref, created_at,"
        " tombstoned, tombstoned_at, revoke_reason FROM nodes WHERE id = ?", (nid,)).fetchone()
    if not row:
        return None
    keys = ("id", "content", "channel", "label", "source_ref", "created_at",
            "tombstoned", "tombstoned_at", "revoke_reason")
    return dict(zip(keys, row))


def live_sources(conn):
    from memsom.storage import schema as memsom_schema
    clauses, params = memsom_schema.taint_filter_clauses(conn)
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE channel != 'agent-derived' AND " + " AND ".join(clauses) +
        " ORDER BY label DESC, id ASC", params).fetchall()


def parents_of(conn, nid):
    return conn.execute(
        "SELECT n.id, n.content, n.channel, n.label, n.source_ref, n.created_at,"
        " n.tombstoned, n.tombstoned_at, n.revoke_reason"
        " FROM edges e JOIN nodes n ON n.id = e.parent"
        " WHERE e.child = ? ORDER BY n.label DESC, n.id ASC", (nid,)).fetchall()


def cascade_set(conn, seed):
    return conn.execute(
        CASCADE_CTE + " SELECT n.id, n.channel, n.tombstoned FROM nodes n"
        " WHERE n.id IN (SELECT id FROM descendants) ORDER BY n.id", (seed,)).fetchall()


def revoke_cascade(conn, seed, reason):
    with conn:
        conn.execute(
            CASCADE_CTE + """
            UPDATE nodes SET tombstoned = 1, tombstoned_at = ?,
                   revoke_reason = CASE WHEN id = ? THEN ? ELSE 'cascade from node ' || ? END
             WHERE id IN (SELECT id FROM descendants) AND tombstoned = 0""",
            (seed, memsom.now_iso(), seed, reason, seed))
        # cursor.rowcount is -1 for WITH-prefixed DML (sqlite3 misdetects it as a query)
        n = conn.execute("SELECT changes()").fetchone()[0]
    return n  # first death wins: already-dead rows keep their record
