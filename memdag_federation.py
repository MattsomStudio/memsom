"""memdag_federation — multi-machine sync that FIXES the Syncthing additive-deletion bug.

The core invariant: death (tombstone) and redaction are MONOTONIC and PROPAGATE.
Once a node is dead/redacted on ANY machine it stays dead everywhere. Importing
an older still-live copy of a locally-tombstoned/redacted node must NOT resurrect it
(first-death-wins across machines).

Public API
----------
default_origin() -> str
backfill_uuids(conn, origin) -> int
export_changeset(conn, since=None, origin=None) -> dict
write_jsonl(path, changeset)
read_jsonl(path) -> dict
import_changeset(conn, changeset) -> dict  # stats

CLI
---
export <file> [--since <iso>] [--origin <name>]
import <file>
register(subparsers)
main(argv=None)
"""

import argparse
import json
import os
import platform
import sys

import memdag
import memdag_schema
import memdag_redact
import memdag_quarantine
import memdag_confid


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Idempotent federation migration.

    Runs dependency migrations first, then adds federation-specific columns.
    Also ensures secondary columns needed for full changeset fidelity are present.
    """
    # Pull in dependency schema
    memdag_redact.migrate(conn)
    memdag_quarantine.migrate(conn)
    memdag_confid.migrate(conn)

    # Federation identity columns
    memdag_schema.add_column(conn, "nodes", "uuid", "TEXT")
    memdag_schema.add_column(conn, "nodes", "origin", "TEXT")

    # Ensure redact/quarantine timestamp columns exist (dependency modules add them,
    # but guard here too for forward-compat / standalone testing)
    memdag_schema.add_column(conn, "nodes", "redacted_at", "TEXT")
    memdag_schema.add_column(conn, "nodes", "redact_reason", "TEXT")
    memdag_schema.add_column(conn, "nodes", "quarantined_at", "TEXT")
    memdag_schema.add_column(conn, "nodes", "quarantine_reason", "TEXT")

    # Unique index: multiple NULLs are fine under SQLite UNIQUE (pre-federation rows)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_uuid ON nodes(uuid)"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_origin():
    """Return the default origin for this machine."""
    return os.environ.get("MEMDAG_ORIGIN") or platform.node() or "unknown"


def backfill_uuids(conn, origin):
    """Assign uuid=f'{origin}:{id}' and origin=origin to all rows where uuid IS NULL.

    Idempotent: rows that already have a uuid are untouched.
    Returns the count of rows updated.
    """
    migrate(conn)
    rows = conn.execute(
        "SELECT id FROM nodes WHERE uuid IS NULL ORDER BY id"
    ).fetchall()
    if not rows:
        return 0
    with conn:
        for (nid,) in rows:
            conn.execute(
                "UPDATE nodes SET uuid=?, origin=? WHERE id=? AND uuid IS NULL",
                (f"{origin}:{nid}", origin, nid)
            )
    return len(rows)


def _row_to_node_dict(row):
    """Convert a DB row (fetched with all federation columns) to an export dict."""
    (nid, content, channel, label, conf_label, status, source_ref, created_at,
     tombstoned, tombstoned_at, revoke_reason,
     redacted, redacted_at, redact_reason,
     quarantined_at, quarantine_reason,
     uuid, origin) = row

    return {
        "uuid": uuid,
        "origin": origin,
        "content": content,
        "channel": channel,
        "label": label,
        "conf_label": conf_label if conf_label is not None else 0,
        "status": status if status is not None else "live",
        "tombstoned": tombstoned,
        "tombstoned_at": tombstoned_at,
        "revoke_reason": revoke_reason,
        "redacted": redacted if redacted is not None else 0,
        "redacted_at": redacted_at,
        "redact_reason": redact_reason,
        "quarantine_reason": quarantine_reason,
        "quarantined_at": quarantined_at,
        "source_ref": source_ref,
        "created_at": created_at,
    }


_SELECT_COLS = (
    "id, content, channel, label, "
    "COALESCE(conf_label, 0) as conf_label, "
    "COALESCE(status, 'live') as status, "
    "source_ref, created_at, "
    "tombstoned, tombstoned_at, revoke_reason, "
    "COALESCE(redacted, 0) as redacted, redacted_at, redact_reason, "
    "quarantined_at, quarantine_reason, "
    "uuid, origin"
)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_changeset(conn, since=None, origin=None):
    """Build a changeset dict for all (or recently changed) nodes.

    since:  ISO-8601 TEXT cutoff. When given, include only nodes where any of
            created_at / tombstoned_at / redacted_at / quarantined_at >= since.
            When None, export all nodes.
    origin: override the default origin name (default_origin() if not given).

    Returns a dict with keys: format, origin, exported_at, nodes, edges.
    """
    migrate(conn)
    origin = origin or default_origin()
    backfill_uuids(conn, origin)
    now = memdag.now_iso()

    if since is None:
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM nodes ORDER BY id"
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_SELECT_COLS} FROM nodes WHERE "
            "created_at >= ? OR tombstoned_at >= ? OR redacted_at >= ? "
            "OR quarantined_at >= ? ORDER BY id",
            (since, since, since, since)
        ).fetchall()

    node_dicts = [_row_to_node_dict(r) for r in rows]

    # Build uuid -> local id map for edge resolution
    selected_ids = set()
    for r in rows:
        selected_ids.add(r[0])  # r[0] = id

    # Collect edges where child is in selected set
    # Also resolve parent uuid (parent node always exists locally — FK constraint)
    uuid_by_id = {}
    for r in rows:
        nid = r[0]
        uuid = r[16]  # uuid column index in the SELECT
        if uuid is not None:
            uuid_by_id[nid] = uuid

    # For since-filtered exports, we also need parent uuids even if parent not exported
    # Fetch uuids for all local nodes to resolve parent references
    all_uuids = conn.execute("SELECT id, uuid FROM nodes WHERE uuid IS NOT NULL").fetchall()
    for nid, uuid in all_uuids:
        if nid not in uuid_by_id:
            uuid_by_id[nid] = uuid

    edge_rows = []
    if selected_ids:
        qmarks = ",".join("?" * len(selected_ids))
        edge_db_rows = conn.execute(
            f"SELECT child, parent FROM edges WHERE child IN ({qmarks}) ORDER BY child, parent",
            tuple(sorted(selected_ids))
        ).fetchall()
        for child_id, parent_id in edge_db_rows:
            child_uuid = uuid_by_id.get(child_id)
            parent_uuid = uuid_by_id.get(parent_id)
            if child_uuid and parent_uuid:
                edge_rows.append([child_uuid, parent_uuid])

    return {
        "format": "memdag-changeset-v1",
        "origin": origin,
        "exported_at": now,
        "nodes": node_dicts,
        "edges": edge_rows,
    }


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------

def write_jsonl(path, changeset):
    """Write changeset to a JSONL file.

    Line 1: header with format/origin/exported_at.
    Then one line per node, one line per edge.
    UTF-8, ensure_ascii=False.
    """
    with open(path, "w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "format": changeset["format"],
            "origin": changeset["origin"],
            "exported_at": changeset["exported_at"],
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for node in changeset.get("nodes", []):
            record = {"type": "node"}
            record.update(node)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for child_uuid, parent_uuid in changeset.get("edges", []):
            f.write(json.dumps(
                {"type": "edge", "child": child_uuid, "parent": parent_uuid},
                ensure_ascii=False
            ) + "\n")


def read_jsonl(path):
    """Reassemble a changeset dict from a JSONL file.

    Raises ValueError on missing or malformed header.
    Returns dict matching export_changeset format.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        raise ValueError("empty JSONL file")

    header = json.loads(lines[0])
    if header.get("type") != "header" or header.get("format") != "memdag-changeset-v1":
        raise ValueError(f"bad or missing header in JSONL: {lines[0]!r}")

    nodes = []
    edges = []
    for line in lines[1:]:
        rec = json.loads(line)
        rec_type = rec.get("type")
        if rec_type == "node":
            d = {k: v for k, v in rec.items() if k != "type"}
            nodes.append(d)
        elif rec_type == "edge":
            edges.append([rec["child"], rec["parent"]])

    return {
        "format": header["format"],
        "origin": header["origin"],
        "exported_at": header["exported_at"],
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_changeset(conn, changeset):
    """Import a changeset into this database.

    Returns stats dict: nodes_new, nodes_updated, edges_new, edges_skipped,
    resurrections_blocked.

    Monotonic rules (all applied in a single transaction):
      Tombstone: incoming dead + local live -> mark dead (copy fields).
                 incoming live + local dead -> BLOCKED (resurrection prevented).
      Redact:    incoming redacted + local unredacted -> redact locally.
                 incoming unredacted + local redacted -> BLOCKED.
      Quarantine: incoming quarantined + local live -> quarantine locally.
                  incoming live + local quarantined -> BLOCKED (promote is a local act).
      Content/label/channel/created_at: NEVER overwritten on existing rows.
    """
    if changeset.get("format") != "memdag-changeset-v1":
        raise ValueError(
            f"unsupported changeset format: {changeset.get('format')!r}"
        )

    migrate(conn)

    stats = {
        "nodes_new": 0,
        "nodes_updated": 0,
        "edges_new": 0,
        "edges_skipped": 0,
        "resurrections_blocked": 0,
    }

    with conn:
        # ----- Pass 1: nodes -----
        for node in changeset.get("nodes", []):
            uuid = node.get("uuid")
            if not uuid:
                continue

            local = conn.execute(
                "SELECT id, tombstoned, tombstoned_at, revoke_reason, "
                "redacted, redacted_at, redact_reason, "
                "COALESCE(status, 'live') as status, quarantined_at, quarantine_reason "
                "FROM nodes WHERE uuid=?",
                (uuid,)
            ).fetchone()

            if local is None:
                # --- New node: INSERT with all fields ---
                conn.execute(
                    "INSERT INTO nodes("
                    "  content, channel, label, conf_label, status, source_ref, created_at,"
                    "  tombstoned, tombstoned_at, revoke_reason,"
                    "  redacted, redacted_at, redact_reason,"
                    "  quarantined_at, quarantine_reason,"
                    "  uuid, origin"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        node.get("content", ""),
                        node.get("channel", "external"),
                        node.get("label", 0),
                        node.get("conf_label", 0),
                        node.get("status", "live"),
                        node.get("source_ref"),
                        node.get("created_at", memdag.now_iso()),
                        node.get("tombstoned", 0),
                        node.get("tombstoned_at"),
                        node.get("revoke_reason"),
                        node.get("redacted", 0),
                        node.get("redacted_at"),
                        node.get("redact_reason"),
                        node.get("quarantined_at"),
                        node.get("quarantine_reason"),
                        uuid,
                        node.get("origin"),
                    )
                )
                stats["nodes_new"] += 1

            else:
                # --- Existing node: monotonic merge ---
                (local_id, local_tomb, local_tomb_at, local_tomb_reason,
                 local_redacted, local_redacted_at, local_redact_reason,
                 local_status, local_q_at, local_q_reason) = local

                updated = False

                # (a) Tombstone monotonic rule
                inc_tomb = node.get("tombstoned", 0)
                if inc_tomb and not local_tomb:
                    # incoming dead, local live -> propagate death
                    conn.execute(
                        "UPDATE nodes SET tombstoned=1, tombstoned_at=?, revoke_reason=?"
                        " WHERE id=?",
                        (node.get("tombstoned_at"), node.get("revoke_reason"), local_id)
                    )
                    updated = True
                elif not inc_tomb and local_tomb:
                    # incoming live, local dead -> BLOCK resurrection
                    stats["resurrections_blocked"] += 1

                # (b) Redact monotonic rule
                inc_redacted = node.get("redacted", 0)
                if inc_redacted and not local_redacted:
                    # incoming redacted, local unredacted -> redact locally
                    conn.execute(
                        "UPDATE nodes SET content='', redacted=1, redacted_at=?, redact_reason=?"
                        " WHERE id=?",
                        (node.get("redacted_at"), node.get("redact_reason"), local_id)
                    )
                    updated = True
                elif not inc_redacted and local_redacted:
                    # incoming unredacted over local redacted -> BLOCK
                    stats["resurrections_blocked"] += 1

                # (c) Quarantine monotonic rule (status field)
                inc_status = node.get("status", "live")
                if inc_status == "quarantined" and local_status == "live":
                    conn.execute(
                        "UPDATE nodes SET status='quarantined', quarantine_reason=?,"
                        " quarantined_at=? WHERE id=?",
                        (node.get("quarantine_reason"), node.get("quarantined_at"), local_id)
                    )
                    updated = True
                elif inc_status == "live" and local_status == "quarantined":
                    # incoming live over local quarantined -> BLOCK
                    # (promote is a deliberate local act, never an import side effect)
                    stats["resurrections_blocked"] += 1

                if updated:
                    stats["nodes_updated"] += 1

        # ----- Pass 2: edges -----
        # Build uuid -> local id map (all local nodes, including just-inserted)
        uuid_to_id = {
            r[0]: r[1]
            for r in conn.execute("SELECT uuid, id FROM nodes WHERE uuid IS NOT NULL")
        }

        for child_uuid, parent_uuid in changeset.get("edges", []):
            child_id = uuid_to_id.get(child_uuid)
            parent_id = uuid_to_id.get(parent_uuid)
            if child_id is None or parent_id is None:
                stats["edges_skipped"] += 1
                continue
            # INSERT OR IGNORE: idempotent
            conn.execute(
                "INSERT OR IGNORE INTO edges(child, parent) VALUES (?,?)",
                (child_id, parent_id)
            )
            # Check if it was actually inserted
            # (sqlite3 doesn't easily tell us, so we track via changes())
            n = conn.execute("SELECT changes()").fetchone()[0]
            if n:
                stats["edges_new"] += 1
            else:
                stats["edges_skipped"] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_export(args):
    conn = memdag.get_connection()
    try:
        changeset = export_changeset(
            conn,
            since=getattr(args, "since", None),
            origin=getattr(args, "origin", None),
        )
        write_jsonl(args.file, changeset)
        print(f"exported {len(changeset['nodes'])} node(s),"
              f" {len(changeset['edges'])} edge(s) -> {args.file}")
    finally:
        conn.close()


def cmd_import(args):
    changeset = read_jsonl(args.file)
    conn = memdag.get_connection()
    try:
        stats = import_changeset(conn, changeset)
        for k, v in stats.items():
            print(f"{k}: {v}")
    finally:
        conn.close()


def register(subparsers):
    p_exp = subparsers.add_parser("export", help="export a changeset to JSONL")
    p_exp.add_argument("file", help="output .jsonl path")
    p_exp.add_argument("--since", default=None,
                       help="ISO-8601 cutoff; only nodes changed since this time")
    p_exp.add_argument("--origin", default=None,
                       help="override the machine origin name")
    p_exp.set_defaults(func=cmd_export)

    p_imp = subparsers.add_parser("import", help="import a changeset from JSONL")
    p_imp.add_argument("file", help="input .jsonl path")
    p_imp.set_defaults(func=cmd_import)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memdag_federation")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
