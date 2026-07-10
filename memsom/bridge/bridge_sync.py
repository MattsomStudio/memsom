"""memsom_bridge_sync — Mac<->PC memory sync over federation changesets (Phase 4).

The memory DATA must reach both machines, but the SQLite DB is gitignored and
binary (no line-diff, merge hell) — so it is NOT git-synced and NOT whole-file
Syncthing-synced.  Instead each machine exports its DB to a TEXT changeset JSONL
that rides the existing Syncthing-over-Nebula `claude-sync/` tree, and imports
the other machine's changeset into its own LOCAL DB.  memsom_federation already
provides the safe primitives (monotonic first-death-wins merge, default-deny
trust boundary, resurrection-block, post-import re-floor) — this module is the
thin per-machine orchestration over them.

Operational note (set at cutover, not here): to avoid the SAME memory existing
under two origins (pc:5 AND mac:7), ONE machine is the importer-of-record for the
flat files (PC, mirroring mem_reconcile's single-writer discipline); the other
receives those nodes purely via the changeset.  This module's round-trip is
origin-keyed and idempotent, so re-importing a changeset never duplicates.

Frozen core untouched.  Library functions never print; only _cmd_* do I/O.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import memsom
from memsom.federation import federation as fed

_SUFFIX = ".changeset.jsonl"


def migrate(conn) -> None:
    """Federation owns the schema (uuid/origin/trusted_origins); just chain it."""
    fed.migrate(conn)


def export_for_sync(conn, out_dir, *, origin=None):
    """Write a FULL changeset for this machine to out_dir/<origin>.changeset.jsonl.

    Full (not incremental) export: the federation import is monotonic + idempotent,
    so re-shipping the whole DB each time is lossless and survives a missed sync
    cycle.  For a ~100-200 node memory store the file is tiny.
    Returns (path, node_count).
    """
    origin = origin or fed.default_origin()
    changeset = fed.export_changeset(conn, origin=origin)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{origin}{_SUFFIX}"
    fed.write_jsonl(path, changeset)
    return path, len(changeset.get("nodes", []))


def import_from_sync(conn, in_dir, *, self_origin=None, trust=True):
    """Import every peer changeset in in_dir (skipping this machine's own file).

    With trust=True (default for the single-operator Mac<->PC mesh, where only the
    owner's machines write into the Nebula-synced tree) each peer origin is
    registered trusted before import, so its nodes keep their real channel instead
    of being clamped to external by the default-deny boundary.  Returns
    {origin: import_stats}.
    """
    self_origin = self_origin or fed.default_origin()
    in_dir = Path(in_dir)
    results = {}
    if not in_dir.is_dir():
        return results
    for path in sorted(in_dir.glob(f"*{_SUFFIX}")):
        origin = path.name[:-len(_SUFFIX)]
        if origin == self_origin:
            continue  # never import our own export
        changeset = fed.read_jsonl(path)
        if trust:
            fed.register_origin(conn, changeset.get("origin") or origin,
                                descr="bridge peer")
        results[origin] = fed.import_changeset(conn, changeset)
    return results


# --- CLI ----------------------------------------------------------------------

def _cmd_export(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        path, n = export_for_sync(conn, args.dir, origin=args.origin)
        print(f"[bridge-sync] exported {n} nodes -> {path}")
    finally:
        conn.close()


def _cmd_import(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        res = import_from_sync(conn, args.dir, self_origin=args.origin,
                               trust=not args.no_trust)
        if not res:
            print("[bridge-sync] no peer changesets found")
        for origin, st in res.items():
            print(f"[bridge-sync] imported {origin}: "
                  f"+{st.get('nodes_new', 0)} new, {st.get('nodes_updated', 0)} updated, "
                  f"{st.get('resurrections_blocked', 0)} resurrections blocked")
    finally:
        conn.close()


def register(sub) -> None:
    pe = sub.add_parser("bridge-sync-export",
                        help="export this machine's memory changeset for sync (Phase 4)")
    pe.add_argument("dir", help="output dir (e.g. the synced claude-sync/memsom/)")
    pe.add_argument("--origin", default=None, help="origin name (default: $MEMDAG_ORIGIN)")
    pe.set_defaults(func=_cmd_export)

    pi = sub.add_parser("bridge-sync-import",
                        help="import peer memory changesets from a sync dir (Phase 4)")
    pi.add_argument("dir", help="input dir holding peer <origin>.changeset.jsonl files")
    pi.add_argument("--origin", default=None, help="this machine's origin (skip own file)")
    pi.add_argument("--no-trust", action="store_true",
                    help="do NOT auto-register peers (imports clamp to external)")
    pi.set_defaults(func=_cmd_import)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass  # reconfigure unsupported on this stream — keep its default encoding
    ap = argparse.ArgumentParser(prog="memsom_bridge_sync", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
