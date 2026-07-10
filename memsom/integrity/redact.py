"""memsom_redact — shape-preserving REDACTION (Guarantee #6 second mode).

Distinct from revoke:
  - revoke  = tombstone (liveness dies, payload survives, DAG shape survives)
  - redact  = payload destroyed (content=''), row / label / dates / ALL edges
              survive so blame / cascade still works; liveness is unaffected
              (tombstoned stays 0).

NOTE: nodes.content is NOT NULL in the frozen schema and cannot be relaxed
additively.  The redaction marker is content='' PLUS redacted=1.  Demo #1
untouched: the three new columns all default so existing code never reads them.

Migration adds three columns to nodes.

Public API
----------
migrate(conn)                                          idempotent
redact_node(conn, nid, reason, cascade=False) -> list[int]
is_redacted(conn, nid)                        -> bool
live_unredacted_sources(conn)                 -> list[tuple]  (same shape as live_sources)
describe(conn, nid)                           -> list[str]    (fmt_node-style)

CLI
---
redact <id> --reason <r> [--cascade] [--yes]   (dry-run by default)
register(subparsers)   — standard mount point
main(argv=None)        — thin wrapper for tests
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import memsom
from memsom.storage import schema as memsom_schema


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Add redaction columns idempotently.  Safe to call multiple times."""
    memsom_schema.add_column(conn, "nodes", "redacted",      "INTEGER NOT NULL DEFAULT 0")
    memsom_schema.add_column(conn, "nodes", "redacted_at",   "TEXT")
    memsom_schema.add_column(conn, "nodes", "redact_reason", "TEXT")
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS redaction_log (
    uuid        TEXT PRIMARY KEY,
    redacted_at TEXT
  );""")


# ---------------------------------------------------------------------------
# Library functions (no prints, no sys.exit)
# ---------------------------------------------------------------------------

def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _unlink_within(root, relpath):
    """Unlink ``root/relpath`` iff it resolves INSIDE *root*. Returns one of
    'purged' | 'noop' | 'failed'. A path that escapes the root (crafted '../'
    bridge_path/obsidian_path — the store treats ingested content as untrusted)
    is 'failed', never followed. Best-effort: OS errors are 'failed', not raised."""
    try:
        root = Path(root).resolve()
        target = (root / relpath).resolve()
        if not target.is_relative_to(root):
            return "failed"                    # traversal attempt — surfaced, not followed
        if target.exists():
            target.unlink()
            return "purged"
        return "noop"
    except OSError:
        return "failed"


def _resolve_memory_dir(memory_dir):
    if memory_dir is not None:
        return memory_dir
    try:
        from memsom.bridge.bridge_import import default_memory_dir
        return default_memory_dir()
    except Exception:  # noqa: BLE001
        return None


def _resolve_vault(vault):
    if vault is not None:
        return vault
    return os.environ.get("MEMDAG_OBSIDIAN_VAULT")


def _purge_backing_files(conn, ids, memory_dir=None, vault=None):
    """Unlink the on-disk source files backing *ids* — flat memory ``.md`` and any
    Obsidian vault note — so redaction reaches disk, not just the DB.

    Roots are resolved LAZILY and only when some redacted node actually carries a
    backing-file path: a plain node (insert_node never sets bridge_path/
    obsidian_path) resolves no root, so a bare redact_node() call — every test and
    POC — can never touch a real file. Containment-guarded and best-effort: a
    failed unlink is counted and reported, never fatal to the DB redaction.

    Returns {'purged': int, 'failed': int}.
    """
    out = {"purged": 0, "failed": 0}
    if not ids:
        return out
    has_bp = memsom_schema.column_exists(conn, "nodes", "bridge_path")
    has_op = memsom_schema.column_exists(conn, "nodes", "obsidian_path")
    if not (has_bp or has_op):
        return out
    bp_col = "bridge_path" if has_bp else "NULL"
    op_col = "obsidian_path" if has_op else "NULL"
    rows = []
    for tid in ids:
        r = conn.execute(
            f"SELECT {bp_col}, {op_col} FROM nodes WHERE id=?", (tid,)
        ).fetchone()
        if r and (r[0] or r[1]):
            rows.append(r)
    if not rows:                               # nothing file-backed: resolve no root
        return out

    mem_root = None                            # resolved on first flat-file hit
    vault_root = None                          # resolved on first vault-note hit
    for bp, op in rows:
        if bp:
            if mem_root is None:
                mem_root = _resolve_memory_dir(memory_dir) or False
            if mem_root:
                res = _unlink_within(mem_root, bp)
                if res == "purged":
                    out["purged"] += 1
                elif res == "failed":
                    out["failed"] += 1
        if op:
            if vault_root is None:
                vault_root = _resolve_vault(vault) or False
            if vault_root:
                res = _unlink_within(vault_root, op)
                if res == "purged":
                    out["purged"] += 1
                elif res == "failed":
                    out["failed"] += 1
    return out


def redact_node(conn, nid, reason, cascade=True, *, memory_dir=None, vault=None,
                purge_stats=None):
    """Destroy payload of *nid* (and optionally all descendants).

    Also reaches DISK: unlinks the flat memory ``.md`` and any Obsidian vault note
    backing a redacted node, so the plaintext can't survive on disk or resurrect on
    the next bridge-render. Roots are resolved lazily (from *memory_dir* / *vault*
    or the live locations) and ONLY when a redacted node actually carries a
    backing-file path — a plain node has none, so a bare call never touches disk.
    If *purge_stats* (a dict) is given it is filled with {'purged','failed'} counts.

    Returns a sorted list of node ids that were newly redacted in this call.
    Already-redacted targets are silently skipped (first redaction wins).
    Raises ValueError for unknown ids.
    """
    migrate(conn)
    row = conn.execute("SELECT id FROM nodes WHERE id = ?", (nid,)).fetchone()
    if row is None:
        raise ValueError(f"unknown node id: {nid}")

    if cascade:
        targets_raw = memsom.cascade_set(conn, nid)
        target_ids = [r[0] for r in targets_raw]
    else:
        target_ids = [nid]

    redacted_ids = []
    ts = _now_iso()
    with conn:
        for tid in target_ids:
            r_reason = reason if tid == nid else f"cascade from node {nid}"
            conn.execute(
                "UPDATE nodes SET content='', redacted=1, redacted_at=?, redact_reason=?"
                " WHERE id=? AND redacted=0",
                (ts, r_reason, tid))
            changed = conn.execute("SELECT changes()").fetchone()[0]
            if changed:
                redacted_ids.append(tid)

    # F-15: purge the retrieval index (postings/docstats/embeddings) for every
    # node whose payload was just destroyed, so a stale BM25/vector posting can't
    # surface the node id after redaction. Best-effort: retrieval is optional.
    if redacted_ids:
        try:
            from memsom.retrieval import retrieve as memsom_retrieve
            for tid in redacted_ids:
                memsom_retrieve.deindex_node(conn, tid)
        except Exception:  # noqa: BLE001
            pass

    # Reach disk: unlink the flat memory file + any vault note backing a redacted
    # node, so the plaintext is gone from disk and can't resurrect on the next
    # bridge-render. Best-effort, containment-guarded (see _purge_backing_files).
    if redacted_ids:
        stats = _purge_backing_files(conn, redacted_ids, memory_dir, vault)
        if purge_stats is not None:
            purge_stats.update(stats)

    # Record redaction EVENTS for any redacted node carrying a federation uuid,
    # so the redaction propagates as a first-class record (F-07 closeable part).
    if redacted_ids and memsom_schema.column_exists(conn, 'nodes', 'uuid'):
        with conn:
            for tid in redacted_ids:
                u = conn.execute('SELECT uuid FROM nodes WHERE id=?', (tid,)).fetchone()
                if u and u[0]:
                    conn.execute(
                        'INSERT OR IGNORE INTO redaction_log(uuid, redacted_at) VALUES (?,?)',
                        (u[0], ts)
                    )

    return sorted(redacted_ids)


def is_redacted(conn, nid):
    """Return True if node *nid* has been redacted.  ValueError on unknown id."""
    migrate(conn)
    row = conn.execute("SELECT redacted FROM nodes WHERE id = ?", (nid,)).fetchone()
    if row is None:
        raise ValueError(f"unknown node id: {nid}")
    return bool(row[0])


def live_unredacted_sources(conn):
    """Live, non-redacted sources — same row shape as memsom.live_sources.

    This is the safe source-pool helper for compose/ask.  Even if a redacted row
    somehow leaks past this filter its content is '' so nothing sensitive can be
    quoted.
    """
    migrate(conn)
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE tombstoned = 0 AND redacted = 0 AND channel != 'agent-derived'"
        " ORDER BY label DESC, id ASC"
    ).fetchall()


def describe(conn, nid):
    """Render a node like memsom.fmt_node, replacing the snippet with a redaction
    marker when the node has been redacted.

    Returns a list of strings (no trailing newline).
    Raises ValueError for unknown id.
    """
    migrate(conn)
    row = conn.execute(
        "SELECT id, content, channel, label, source_ref, created_at,"
        " tombstoned, tombstoned_at, revoke_reason,"
        " redacted, redacted_at, redact_reason"
        " FROM nodes WHERE id = ?", (nid,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown node id: {nid}")

    (nid_, content, channel, label, source_ref, created_at,
     tombstoned, tombstoned_at, revoke_reason,
     redacted, redacted_at, redact_reason) = row

    # Build a node dict compatible with memsom.fmt_node
    node = {
        "id": nid_, "content": content, "channel": channel, "label": label,
        "source_ref": source_ref, "created_at": created_at,
        "tombstoned": tombstoned, "tombstoned_at": tombstoned_at,
        "revoke_reason": revoke_reason,
    }

    # fmt_node returns a list; the last line is the snippet line
    lines = memsom.fmt_node(node)

    if redacted:
        local = memsom.local_date(redacted_at)
        lines[-1] = f'      "[REDACTED {local}: {redact_reason}]"'

    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_redact(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        node = memsom.get_node(conn, args.id)
        if node is None:
            print(f"[memsom-redact] no node [{args.id}]", file=sys.stderr)
            sys.exit(1)

        # cascade=True by default; --single opts out
        cascade = not getattr(args, "single", False)

        # Build target set for dry-run display
        if cascade:
            targets_raw = memsom.cascade_set(conn, args.id)
            target_ids = [r[0] for r in targets_raw]
            target_channels = {r[0]: r[1] for r in targets_raw}
        else:
            target_ids = [args.id]
            target_channels = {args.id: node["channel"]}

        # Determine already-redacted members
        already = set()
        for tid in target_ids:
            r = conn.execute("SELECT redacted FROM nodes WHERE id=?", (tid,)).fetchone()
            if r and r[0]:
                already.add(tid)

        pending_count = len([t for t in target_ids if t not in already])

        print(f"will redact {pending_count} node(s):")
        for tid in target_ids:
            role = "seed" if tid == args.id else "descendant"
            ch = target_channels.get(tid, "?")
            note = "  - already redacted, skipped" if tid in already else ""
            print(f"  [{tid}] {ch} ({role}){note}")

        if not args.yes:
            print("payloads destroyed; rows, edges, labels, dates survive.")
            print("dry run - re-run with --yes to apply.")
            return

        ps = {}
        mem_dir = getattr(args, "memory_dir", None)
        newly = redact_node(conn, args.id, args.reason, cascade=cascade,
                            memory_dir=mem_dir, purge_stats=ps)
        purged, failed = ps.get("purged", 0), ps.get("failed", 0)
        fail_note = f", {failed} unlink FAILED (see above)" if failed else ""
        print(f"done - {len(newly)} redacted, {purged} file(s) purged from disk, "
              f"0 rows deleted, all edges intact{fail_note}.")
    finally:
        conn.close()


def register(subparsers):
    p = subparsers.add_parser("redact", help="destroy payload while preserving shape")
    p.add_argument("id", type=int)
    p.add_argument("--reason", required=True, help="why the payload is being destroyed")
    p.add_argument("--cascade", action="store_true",
                   help="(default) redact all transitive descendants")
    p.add_argument("--single", action="store_true",
                   help="redact ONLY this node (opt out of cascade)")
    p.add_argument("--yes", action="store_true",
                   help="apply; without this flag the command is a dry run")
    p.add_argument("--memory-dir", default=None,
                   help="override the memory dir whose flat file is unlinked on redact")
    p.set_defaults(func=cmd_redact)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-redact")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
