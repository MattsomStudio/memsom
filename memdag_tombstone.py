"""memdag_tombstone — the sanctioned delete path for a flat memory file.

Deleting a memory file by hand leaves its node live in the store, so the GENERATED
MEMORY.md keeps rendering it. This makes deletion first-class and auditable: it
REVOKES the memory's node (memdag's native tombstone + cascade — history stays
explainable, nothing is hard-deleted from the DAG) and removes the flat ``.md`` so
the bridge importer won't recreate it. MEMORY.md drops it on the next
``bridge-render``.

Refuses to tombstone a pinned (endorsed: ``user_`` / ``feedback_`` / ``personal_``)
memory unless ``--force`` — those are identity / operating-rule memories.

  memdag tombstone <stem> --reason "..."   # revoke the node + delete the file
  memdag tombstone-list                     # show tombstoned memory nodes

Read the store via $MEMDAG_DB (defaults to ~/.memdag/memdag.db).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import memdag
from memdag_bridge_import import (split_frontmatter, fm_top_level,
                                  default_memory_dir)

ENDORSED_PREFIXES = ("user_", "feedback_", "personal_")


def _is_pinned(stem, fm, channel=None):
    """A memory is pinned if its node is endorsed, its stem carries an endorsed
    prefix, or its frontmatter sets pin truthy."""
    if channel == "endorsed":
        return True
    if stem.startswith(ENDORSED_PREFIXES):
        return True
    return str(fm.get("pin", "")).strip().lower() in ("1", "true", "yes")


def _live_node(conn, stem):
    """(id, channel) of the live node for memory:<stem>, or (None, None)."""
    row = conn.execute(
        "SELECT id, channel FROM nodes "
        "WHERE source_ref = ? AND tombstoned = 0 ORDER BY id DESC LIMIT 1",
        (f"memory:{stem}",)).fetchone()
    return (row[0], row[1]) if row else (None, None)


def tombstone_memory(conn, mem_dir, stem, *, reason="", force=False, delete_file=True):
    """Revoke the node for *stem* and remove its flat file. Returns a result dict.

    status: 'ok' | 'refused-pinned'. Never hard-deletes the DAG row — revoke
    tombstones it (cascading to descendants) so blame/explain still walk it."""
    stem = stem[:-3] if stem.endswith(".md") else stem
    filename = f"{stem}.md"
    local = Path(mem_dir) / filename
    fm = {}
    if local.exists():
        fm_lines, _b, _ = split_frontmatter(local.read_text(encoding="utf-8", errors="replace"))
        fm = fm_top_level(fm_lines)

    node_id, channel = _live_node(conn, stem)

    if _is_pinned(stem, fm, channel) and not force:
        return {"status": "refused-pinned", "stem": stem,
                "node_id": node_id, "revoked": 0, "file_deleted": False}

    revoked = 0
    if node_id is not None:
        revoked = memdag.revoke_cascade(conn, node_id, reason or "tombstoned via memdag tombstone")
        conn.commit()

    file_deleted = False
    if delete_file and local.exists():
        local.unlink()
        file_deleted = True

    return {"status": "ok", "stem": stem, "node_id": node_id,
            "revoked": revoked, "file_deleted": file_deleted}


def list_tombstoned(conn):
    """[(stem, tombstoned_at, reason)] for tombstoned memory FILE nodes."""
    rows = conn.execute(
        "SELECT source_ref, tombstoned_at, revoke_reason FROM nodes "
        "WHERE tombstoned = 1 AND source_ref LIKE 'memory:%' "
        "AND source_ref NOT LIKE 'memory:literal:%' ORDER BY tombstoned_at DESC"
    ).fetchall()
    out = []
    for sref, at, reason in rows:
        out.append((sref.split(":", 1)[1], at, reason or ""))
    return out


# --- CLI ---------------------------------------------------------------------

def _default_db():
    return str(Path.home() / ".memdag" / "memdag.db")


def _cmd_tombstone(args):
    os.environ.setdefault("MEMDAG_DB", _default_db())
    mem = Path(args.memory_dir) if args.memory_dir else default_memory_dir()
    conn = memdag.get_connection()
    try:
        res = tombstone_memory(conn, mem, args.stem, reason=args.reason, force=args.force)
    finally:
        conn.close()
    if res["status"] == "refused-pinned":
        print(f"[tombstone] REFUSED: {res['stem']} is pinned (identity/feedback/personal). "
              f"Re-run with --force only if you truly mean to forget it.", file=sys.stderr)
        return 2
    print(f"[tombstone] {res['stem']}: "
          f"node {'revoked (+'+str(res['revoked'])+' cascaded)' if res['node_id'] else 'not in store'} | "
          f"file {'deleted' if res['file_deleted'] else 'absent'}")
    print("  MEMORY.md drops it on the next `memdag bridge-render`.")
    return 0


def _cmd_tombstone_list(args):
    os.environ.setdefault("MEMDAG_DB", _default_db())
    conn = memdag.get_connection()
    try:
        rows = list_tombstoned(conn)
    finally:
        conn.close()
    if not rows:
        print("no tombstoned memories.")
        return 0
    print(f"{len(rows)} tombstoned memor{'y' if len(rows) == 1 else 'ies'}:")
    for stem, at, reason in rows:
        print(f"  {at or '?':<26} {stem:<44} {reason}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("tombstone",
                       help="the sanctioned delete path: revoke a memory's node + remove its file")
    p.add_argument("stem", help="memory stem or filename, e.g. project_old_thing")
    p.add_argument("--reason", default="", help="why it's being deleted (audited on the node)")
    p.add_argument("--force", action="store_true",
                   help="allow tombstoning a pinned user_/feedback_/personal_ memory")
    p.add_argument("--memory-dir", default=None, help="override the memory dir")
    p.set_defaults(func=_cmd_tombstone)

    q = sub.add_parser("tombstone-list", help="list tombstoned memory nodes")
    q.set_defaults(func=_cmd_tombstone_list)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="memdag_tombstone", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = ap.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
