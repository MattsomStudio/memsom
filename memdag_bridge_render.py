"""memdag_bridge_render — regenerate MEMORY.md from the memdag store.

The shippable Stop-hook command behind the bridge.  The flat ``memory/*.md`` files
are the live INPUT (a memory-write skill such as ``/saveall`` edits them); this
re-imports them into memdag, runs the forgetting pass, flags verification-age
staleness, and rewrites ``MEMORY.md`` from the store via the FAIL-SAFE digest
writer.  A bad or oversized render leaves the existing ``MEMORY.md`` untouched — it
never blanks the always-on brain.

Wired as a Claude Code Stop hook by ``memdag wire-claude``::

    "Stop": [{"hooks": [{"type": "command", "command": "<memdag> bridge-render"}]}]

NEVER raises into the hook chain: any error is caught, logged to stdout as
``[bridge] ...``, and the process exits 0 with ``MEMORY.md`` unchanged.

Single-writer note: on a multi-machine setup, render on exactly one machine and let
the others receive the rendered ``MEMORY.md`` via your sync layer — set
``MEMDAG_BRIDGE_AUTHOR=0`` on the non-author machines to skip the render (they still
import + run the forgetting pass to keep their mirror warm).

Env knobs:
  MEMDAG_DB                 store path (defaults to ~/.memdag/memdag.db — the
                            bootstrap data dir — so a Stop hook needs no env wiring).
  MEMDAG_DIGEST_TITLE       H1 of the rendered MEMORY.md (digest default: "# Memory").
  MEMDAG_VERIFY_STALE_DAYS  verification-age threshold; <= 0 disables the pass.
  MEMDAG_BRIDGE_AUTHOR      "0" => mirror-only (no render) for non-author machines.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import memdag
import memdag_bridge_import as bi
import memdag_digest as digest
import memdag_forget as forget


def _is_author() -> bool:
    return os.environ.get("MEMDAG_BRIDGE_AUTHOR", "1") != "0"


def _verify_stale_days() -> int:
    try:
        return int(os.environ.get("MEMDAG_VERIFY_STALE_DAYS", "14"))
    except ValueError:
        return 14


def bridge_render(conn, memory_dir, *, render=True, sync_claude=True):
    """Run import -> forget -> (verify-stale) -> write_live over *memory_dir*.

    Returns a result dict.  This is the pure orchestration core; the CLI wrapper
    (`_cmd_bridge_render_safe`) is the fail-safe boundary that swallows errors so
    they never break the Stop-hook chain.
    """
    bi.migrate(conn)
    forget.migrate(conn)
    bi.import_all(conn, memory_dir, dry_run=False)
    usage = Path(memory_dir) / ".weights" / "usage"
    forget.recompute_forget(conn, usage_dir=str(usage))

    if not render:
        return {"rendered": False, "reason": "non-author (MEMDAG_BRIDGE_AUTHOR=0)"}

    stale_marked = 0
    days = _verify_stale_days()
    if days > 0:
        # imported lazily: non-author machines never reach here, so a not-yet-synced
        # module can't break a mirror-only run.
        try:
            import memdag_verify_stale as verify
            vstats = verify.recompute_verify_stale(conn, threshold_days=days)
            stale_marked = len(vstats.get("marked", []))
        except Exception as exc:  # staleness is advisory — never block the render
            print(f"[bridge] verify-stale skipped: {exc!r}")

    ok, info = digest.write_live(conn, str(memory_dir))

    # Keep the CLAUDE.md managed block current in the same pass (idempotent: a
    # second run is a no-op). Best-effort — a CLAUDE.md problem must never stop the
    # MEMORY.md render from being reported. Honors $CLAUDE_MD_PATH.
    claude = None
    if sync_claude:
        try:
            import memdag_claude
            claude = memdag_claude.sync()
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] claude-sync skipped: {exc!r}")

    return {"rendered": True, "ok": ok, "info": info,
            "stale_marked": stale_marked, "claude": claude}


def _default_db() -> str:
    # Friend-beta convention: bootstrap creates the DB in ~/.memdag. Default to it so
    # a bare `memdag bridge-render` Stop hook resolves the right store with no env
    # wiring; an explicit MEMDAG_DB always wins (set via setdefault below).
    return str(Path.home() / ".memdag" / "memdag.db")


def _cmd_bridge_render(args):
    os.environ.setdefault("MEMDAG_DB", _default_db())
    mem = Path(args.memory_dir) if args.memory_dir else bi.default_memory_dir()
    conn = memdag.get_connection()
    try:
        result = bridge_render(conn, mem, render=_is_author())
    finally:
        conn.close()

    if not result.get("rendered"):
        print(f"[bridge] mirror updated; render skipped ({result.get('reason')})")
    elif result.get("ok"):
        print(f"[bridge] MEMORY.md regenerated {result['info']} "
              f"stale_marked={result['stale_marked']}")
    else:
        print(f"[bridge] MEMORY.md unchanged (render rejected): {result['info']}")


def _cmd_bridge_render_safe(args):
    """Fail-safe CLI boundary: catch everything so a Stop hook always exits clean."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        _cmd_bridge_render(args)
    except Exception as exc:  # noqa: BLE001 — the hook must never crash the session
        print(f"[bridge] render skipped (MEMORY.md unchanged): {exc!r}")


def register(sub) -> None:
    p = sub.add_parser(
        "bridge-render",
        help="regenerate MEMORY.md from the memdag store (the Stop-hook command)")
    p.add_argument("memory_dir", nargs="?", default=None,
                   help="memory dir (default: auto-detected ~/.claude/projects/*/memory)")
    p.set_defaults(func=_cmd_bridge_render_safe)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="memdag_bridge_render", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    _cmd_bridge_render_safe(ap.parse_args(argv))


if __name__ == "__main__":
    main()
