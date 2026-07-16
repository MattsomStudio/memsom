"""memsom_bridge_render — regenerate MEMORY.md from the memsom store.

The shippable Stop-hook command behind the bridge.  The flat ``memory/*.md`` files
are the live INPUT (a memory-write skill such as ``/saveall`` edits them); this
re-imports them into memsom, runs the forgetting pass, flags verification-age
staleness, and rewrites ``MEMORY.md`` from the store via the FAIL-SAFE digest
writer.  A bad or oversized render leaves the existing ``MEMORY.md`` untouched — it
never blanks the always-on brain.

Wired as a Claude Code Stop hook by ``memsom wire-claude``::

    "Stop": [{"hooks": [{"type": "command", "command": "<memsom> bridge-render"}]}]

NEVER raises into the hook chain: any error is caught, logged to stdout as
``[bridge] ...``, and the process exits 0 with ``MEMORY.md`` unchanged.

Single-writer note: on a multi-machine setup, render on exactly one machine and let
the others receive the rendered ``MEMORY.md`` via your sync layer — set
``MEMDAG_BRIDGE_AUTHOR=0`` on the non-author machines to skip the render (they still
import + run the forgetting pass to keep their mirror warm).

Env knobs:
  MEMDAG_DB / MEMDAG_HOME   store path, resolved by memsom.db_path()
                            (MEMDAG_DB > MEMDAG_HOME/memdag.db > ~/.memdag/memdag.db
                            — so a Stop hook needs no env wiring).
  MEMDAG_DIGEST_TITLE       H1 of the rendered MEMORY.md (digest default: "# Memory").
  MEMDAG_VERIFY_STALE_DAYS  verification-age threshold; <= 0 disables the pass.
  MEMDAG_BRIDGE_AUTHOR      "0" => mirror-only (no render) for non-author machines.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import memsom
from memsom.bridge import bridge_import as bi
from memsom.distill import digest as digest
from memsom.lifecycle import forget as forget


def _is_author() -> bool:
    return os.environ.get("MEMDAG_BRIDGE_AUTHOR", "1") != "0"


def bridge_render(conn, memory_dir, *, render=True, sync_claude=True):
    """Run import -> forget -> (verify-stale) -> write_live over *memory_dir*.

    Returns a result dict.  This is the pure orchestration core; the CLI wrapper
    (`_cmd_bridge_render_safe`) is the fail-safe boundary that swallows errors so
    they never break the Stop-hook chain.
    """
    bi.migrate(conn)
    forget.migrate(conn)
    bi.import_all(conn, memory_dir, dry_run=False)
    weights = Path(memory_dir) / ".weights"
    # Runtime tunables: the same canonical.json the original mem_weights.py
    # maintains.  This is what makes panel-written params live — the render pass
    # actually computes with them.  Warnings are logged HERE (forget never prints).
    params, param_warnings = forget.load_params(weights / "canonical.json")
    for w in param_warnings:
        print(f"[bridge] tunables: {w}")
    forget.recompute_forget(conn, usage_dir=str(weights / "usage"), params=params)

    if not render:
        return {"rendered": False, "reason": "non-author (MEMDAG_BRIDGE_AUTHOR=0)"}

    stale_marked = 0
    # imported lazily: non-author machines never reach here, so a not-yet-synced
    # module can't break a mirror-only run. Threshold comes from the verify_stale
    # module's own resolver (single source of truth — no divergent default here);
    # <= 0 disables the pass.
    try:
        from memsom.integrity import verify_stale as verify
        if verify._threshold_days() > 0:
            vstats = verify.recompute_verify_stale(conn)
            stale_marked = len(vstats.get("marked", []))
    except Exception as exc:  # staleness is advisory — never block the render
        print(f"[bridge] verify-stale skipped: {exc!r}")

    # Budget comes from the same loaded params (memory_budget) so one knob write
    # moves the render threshold and the digest cap together, atomically.
    ok, info = digest.write_live(conn, str(memory_dir),
                                 budget=params["memory_budget"])

    # Keep the CLAUDE.md managed block current in the same pass (idempotent: a
    # second run is a no-op). Best-effort — a CLAUDE.md problem must never stop the
    # MEMORY.md render from being reported. Honors $CLAUDE_MD_PATH.
    claude = None
    if sync_claude:
        try:
            from memsom.bridge import claude as memsom_claude
            claude = memsom_claude.sync()
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] claude-sync skipped: {exc!r}")

    return {"rendered": True, "ok": ok, "info": info,
            "stale_marked": stale_marked, "claude": claude}


def _cmd_bridge_render(args):
    # DB path resolution belongs to memsom.db_path() (MEMDAG_DB > MEMDAG_HOME >
    # ~/.memdag), which get_connection() already uses — so a bare Stop hook still
    # resolves ~/.memdag/memdag.db with no env wiring. The old setdefault here
    # pinned MEMDAG_DB=~/.memdag/memdag.db whenever it was unset, silently
    # overriding a MEMDAG_HOME-relocated store.
    mem = Path(args.memory_dir) if args.memory_dir else bi.default_memory_dir()
    conn = memsom.get_connection()
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
            pass  # reconfigure unsupported on this stream — keep its default encoding
    try:
        _cmd_bridge_render(args)
    except Exception as exc:  # noqa: BLE001 — the hook must never crash the session
        print(f"[bridge] render skipped (MEMORY.md unchanged): {exc!r}")


def register(sub) -> None:
    p = sub.add_parser(
        "bridge-render",
        help="regenerate MEMORY.md from the memsom store (the Stop-hook command)")
    p.add_argument("memory_dir", nargs="?", default=None,
                   help="memory dir (default: auto-detected ~/.claude/projects/*/memory)")
    p.set_defaults(func=_cmd_bridge_render_safe)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="memsom_bridge_render", description=__doc__)
    ap.add_argument("memory_dir", nargs="?", default=None)
    _cmd_bridge_render_safe(ap.parse_args(argv))


if __name__ == "__main__":
    main()
