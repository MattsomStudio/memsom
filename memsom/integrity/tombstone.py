"""memsom_tombstone — the sanctioned delete path for a flat memory file.

Deleting a memory file by hand leaves its node live in the store, so the GENERATED
MEMORY.md keeps rendering it. This makes deletion first-class and auditable: it
REVOKES the memory's node (memsom's native tombstone + cascade — history stays
explainable, nothing is hard-deleted from the DAG) and removes the flat ``.md`` so
the bridge importer won't recreate it. MEMORY.md drops it on the next
``bridge-render``.

Refuses to tombstone a pinned (endorsed: ``user_`` / ``feedback_`` / ``personal_``)
memory unless ``--force`` — those are identity / operating-rule memories.

  memsom tombstone <stem> --reason "..."   # revoke the node + delete the file
  memsom tombstone-list                     # show tombstoned memory nodes

Store path resolved by memsom.db_path() (MEMDAG_DB > MEMDAG_HOME > ~/.memdag).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import memsom
from memsom.bridge.bridge_import import split_frontmatter, fm_top_level, default_memory_dir

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


def tombstone_memory(conn, mem_dir, stem, *, reason="", force=False, delete_file=True,
                     hard=False):
    """Revoke the node for *stem* and remove its flat file. Returns a result dict.

    status: 'ok' | 'refused-pinned' | 'refused-traversal'. Never hard-deletes the
    DAG row — revoke tombstones it (cascading to descendants) so blame/explain
    still walk it.

    hard=True additionally ERASES the payload: after revoking, it runs
    memsom_rederive.erase over the node's whole supersedes lineage, so the
    content is destroyed in the DB, de-indexed, purged from disk, and shown as
    [REDACTED] in blame — closing the leak where a soft delete or edit leaves the
    old plaintext readable through blame. The result dict then carries 'erased'
    (count of nodes whose payload was destroyed)."""
    stem = stem[:-3] if stem.endswith(".md") else stem
    filename = f"{stem}.md"
    mem_root = Path(mem_dir).resolve()
    local = mem_root / filename
    # Containment: a crafted stem ("../../.claude/CLAUDE.md") must never resolve
    # outside the memory dir. This is the documented agent surface and the store
    # treats ingested content as untrusted, so a prompt-injected stem must not be
    # able to escape and delete arbitrary files (the pin guard is also bypassable
    # by a "../" prefix, so this check has to come first).
    if not local.resolve().is_relative_to(mem_root):
        return {"status": "refused-traversal", "stem": stem,
                "node_id": None, "revoked": 0, "file_deleted": False}
    fm = {}
    existed_before = local.exists()      # for accurate file_deleted when hard-erase unlinks first
    if existed_before:
        fm_lines, _b, _ = split_frontmatter(local.read_text(encoding="utf-8", errors="replace"))
        fm = fm_top_level(fm_lines)

    node_id, channel = _live_node(conn, stem)

    if _is_pinned(stem, fm, channel) and not force:
        return {"status": "refused-pinned", "stem": stem,
                "node_id": node_id, "revoked": 0, "file_deleted": False}

    revoked = 0
    if node_id is not None:
        revoked = memsom.revoke_cascade(conn, node_id, reason or "tombstoned via memsom tombstone")
        conn.commit()

    # hard delete: erase the payload across the whole lineage so nothing survives
    # in the DB / on disk / in blame. Pass mem_root so the on-disk purge targets
    # this call's memory dir (honours --memory-dir), not just the live location.
    erased = 0
    if hard and node_id is not None:
        from memsom.retrieval import rederive as memsom_rederive
        erased = len(memsom_rederive.erase(
            conn, node_id, reason or "hard tombstone", memory_dir=mem_root))
        conn.commit()

    # file_deleted means THIS call removed a file. In hard mode redact_node's purge
    # may have unlinked it first, so credit that too — but a file that was already
    # absent before the call was not deleted by us.
    file_deleted = False
    if delete_file:
        if local.exists():
            local.unlink()
            file_deleted = True
        elif hard and existed_before:
            file_deleted = True          # erase's purge removed it

    return {"status": "ok", "stem": stem, "node_id": node_id,
            "revoked": revoked, "erased": erased, "file_deleted": file_deleted}


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

def _cmd_tombstone(args):
    # DB path resolution belongs to memsom.db_path() (MEMDAG_DB > MEMDAG_HOME >
    # ~/.memdag), which get_connection() already uses. The old setdefault here
    # pinned MEMDAG_DB=~/.memdag/memdag.db whenever it was unset, silently
    # overriding a MEMDAG_HOME-relocated store.
    mem = Path(args.memory_dir) if args.memory_dir else default_memory_dir()
    conn = memsom.get_connection()
    try:
        res = tombstone_memory(conn, mem, args.stem, reason=args.reason,
                               force=args.force, hard=args.hard)
    finally:
        conn.close()
    if res["status"] == "refused-traversal":
        print(f"[tombstone] REFUSED: {args.stem!r} escapes the memory dir "
              f"(path traversal blocked).", file=sys.stderr)
        return 2
    if res["status"] == "refused-pinned":
        print(f"[tombstone] REFUSED: {res['stem']} is pinned (identity/feedback/personal). "
              f"Re-run with --force only if you truly mean to forget it.", file=sys.stderr)
        return 2
    erased_note = f" | payload ERASED ({res['erased']} node(s))" if res.get("erased") else ""
    print(f"[tombstone] {res['stem']}: "
          f"node {'revoked (+'+str(res['revoked'])+' cascaded)' if res['node_id'] else 'not in store'} | "
          f"file {'deleted' if res['file_deleted'] else 'absent'}{erased_note}")
    if args.hard:
        print("  hard delete: content destroyed in DB + disk + blame ([REDACTED]).")
    print("  MEMORY.md drops it on the next `memsom bridge-render`.")
    return 0


def _cmd_tombstone_list(args):
    conn = memsom.get_connection()
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
    p.add_argument("--hard", action="store_true",
                   help="also ERASE the payload across the lineage (DB + disk + blame), "
                        "not just revoke it")
    p.add_argument("--memory-dir", default=None, help="override the memory dir")
    p.set_defaults(func=_cmd_tombstone)

    q = sub.add_parser("tombstone-list", help="list tombstoned memory nodes")
    q.set_defaults(func=_cmd_tombstone_list)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass  # reconfigure unsupported on this stream — keep its default encoding
    ap = argparse.ArgumentParser(prog="memsom_tombstone", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = ap.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
