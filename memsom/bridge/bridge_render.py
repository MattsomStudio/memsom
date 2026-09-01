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
import sys
from pathlib import Path

import memsom
from memsom.bridge import bridge_import as bi
from memsom.distill import digest as digest
from memsom.lifecycle import forget as forget
from memsom import tuning as memsom_tuning


def _is_author() -> bool:
    return memsom_tuning.resolve("bridge.author") != "0"


def _write_shed_manifest(memory_dir, excluded, budget, rendered_bytes,
                         max_lines=None, rendered_lines=None, sections=None,
                         section_budgets=None) -> None:
    """Record which memories this render left OUT of MEMORY.md, and why.

    `.weights/shed.json` is the receipt for every index exclusion memsom decides
    (projects / cold / unsectioned / budget / lines).  Consumers (mem_audit) treat a listed stem as
    a tracked, explained absence rather than an orphan.  Always rewritten in
    full — it describes the LAST render only, so a memory that comes back into
    the index drops out of it on its own with no reconciliation step.

    Best-effort by construction: the manifest is diagnostic, and a render that
    already wrote MEMORY.md successfully must never be reported as failed
    because a side-file could not be written.
    """
    import json
    try:
        weights = Path(memory_dir) / ".weights"
        weights.mkdir(parents=True, exist_ok=True)
        by_reason = {}
        for e in excluded:
            by_reason[e.get("reason", "?")] = by_reason.get(e.get("reason", "?"), 0) + 1
        payload = {
            "version": 2,
            "rendered_at": forget.now_iso(),
            "budget": budget,
            "rendered_bytes": rendered_bytes,
            "max_lines": max_lines,
            "lines": rendered_lines,
            "count": len(excluded),
            "by_reason": by_reason,
            # per-section line/byte counts next to their budgets (anti-creep
            # mechanism 4: drift is visible every session end)
            "sections": section_table(sections, section_budgets),
            "excluded": excluded,
        }
        tmp = weights / "shed.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(weights / "shed.json")
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] shed manifest skipped: {exc!r}")


def section_table(sections, section_budgets) -> dict:
    """{section: {"lines", "bytes", "budget" (or None)}} — the rendered counts
    (digest.section_stats) joined with the `section_budgets` param."""
    out = {}
    for sec, st in (sections or {}).items():
        out[sec] = {"lines": st.get("lines", 0), "bytes": st.get("bytes", 0),
                    "budget": (section_budgets or {}).get(sec)}
    for sec, cap in (section_budgets or {}).items():
        out.setdefault(sec, {"lines": 0, "bytes": 0, "budget": cap})
    return out


def format_section_lines(table) -> list:
    """One `name: N lines / B bytes (budget X)` string per budgeted section."""
    out = []
    for sec, st in (table or {}).items():
        if st.get("budget") is None:
            continue
        over = " OVER" if st["bytes"] > st["budget"] else ""
        out.append(f"{sec.lower()}: {st['lines']} lines / {st['bytes']} bytes "
                   f"(budget {st['budget']}){over}")
    return out


def _write_projects_index(conn, memory_dir):
    """Render projects/INDEX.md atomically next to MEMORY.md (creates projects/).

    No byte cap (read on demand, never always-loaded). Returns the path.
    """
    proj_dir = Path(memory_dir) / bi.PROJECTS_SUBDIR
    proj_dir.mkdir(parents=True, exist_ok=True)
    text = digest.render_projects_index(digest.project_entries(conn), conn=conn)
    path = proj_dir / digest.PROJECTS_INDEX_NAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))  # LF on Windows too, like MEMORY.md
    tmp.replace(path)
    return path


def bridge_render(conn, memory_dir, *, render=True, sync_claude=True):
    """Run import -> forget -> (verify-stale) -> write_live over *memory_dir*.

    Also writes projects/INDEX.md (the project_ sub-index) after a successful
    MEMORY.md write.  Returns a result dict.  This is the pure orchestration core; the CLI wrapper
    (`_cmd_bridge_render_safe`) is the fail-safe boundary that swallows errors so
    they never break the Stop-hook chain.
    """
    bi.migrate(conn)
    forget.migrate(conn)
    # First-run scaffold (create-if-absent only): projects/, projects/INDEX.md,
    # .weights/canonical.json with the panel defaults — so a fresh install is
    # line-aware and the pointer line has a target before anything is saved.
    try:
        bi.scaffold_memory_dir(memory_dir)
    # FAILOPEN: allowed, the scaffold is advisory -- a failure here must never block the render.
    except Exception as exc:
        print(f"[bridge] scaffold skipped: {exc!r}")
    weights = Path(memory_dir) / ".weights"
    # Runtime tunables: the same canonical.json the original mem_weights.py
    # maintains.  This is what makes panel-written params live — the render pass
    # actually computes with them.  Warnings are logged HERE (forget never prints).
    params, param_warnings = forget.load_params(weights / "canonical.json")
    for w in param_warnings:
        print(f"[bridge] tunables: {w}")
    imp = bi.import_all(conn, memory_dir, dry_run=False, params=params)
    f = imp.get("files") or {}
    if f.get("dedup") or f.get("quarantined"):
        print(f"[bridge] duplicate stems healed: {f['dedup']} identical deleted, "
              f"{f['quarantined']} differing -> .weights/dup_quarantine/")
    if f.get("born_unindexed"):
        print(f"[bridge] {f['born_unindexed']} new feedback file(s) born unindexed "
              f"(no why_own_line:) — merge into a feedback_cluster_*")
    forget.recompute_forget(conn, usage_dir=str(weights / "usage"), params=params)

    if not render:
        return {"rendered": False, "reason": "non-author (MEMDAG_BRIDGE_AUTHOR=0)"}

    stale_marked = 0
    # imported lazily: non-author machines never reach here, so a not-yet-synced
    # module can't break a mirror-only run. Threshold comes from the verify_stale
    # module's own resolver (single source of truth — no divergent default here);
    # <= 0 disables the pass.
    try:
        from memsom.lifecycle import verify_stale as verify
        if verify._threshold_days() > 0:
            vstats = verify.recompute_verify_stale(conn)
            stale_marked = len(vstats.get("marked", []))
    except Exception as exc:  # staleness is advisory — never block the render
        print(f"[bridge] verify-stale skipped: {exc!r}")

    # Budget + line cap come from the same loaded params (memory_budget /
    # memory_max_lines) so one knob write moves the render threshold and the
    # digest caps together, atomically.
    ok, info = digest.write_live(conn, str(memory_dir),
                                 budget=params["memory_budget"],
                                 max_lines=params["memory_max_lines"],
                                 section_budgets=params["section_budgets"])

    # Budget eviction is a real removal from the always-loaded index, but it is
    # NOT the forgetting layer's hot->cold demote and it never reaches
    # canonical.json (memsom only READS that file; mem_weights.py owns it). Left
    # unrecorded it produced 86 memories that were on disk, absent from
    # MEMORY.md, and unexplainable — the audit could not tell a budget drop from
    # a corrupted index. So the render publishes its own manifest: single-author
    # (this code path only), so no writer ever contends with the weights layer.
    if ok:
        _write_shed_manifest(memory_dir, info.get("excluded") or [],
                             params["memory_budget"], info.get("bytes"),
                             params["memory_max_lines"], info.get("lines"),
                             info.get("sections"), params["section_budgets"])
        # projects/INDEX.md: the project_ memories MEMORY.md no longer carries
        # (digest module docstring, "Projects split"). Written only after the
        # main index succeeded, so the pointer line and its target move together.
        try:
            info["projects_index"] = str(_write_projects_index(conn, memory_dir))
        # FAILOPEN: allowed, the projects sub-index is advisory and must never block the main render.
        except Exception as exc:
            print(f"[bridge] projects index skipped: {exc!r}")

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
        info = dict(result["info"])
        sections = info.pop("sections", None)
        budgets = info.pop("section_budgets", None)
        print(f"[bridge] MEMORY.md regenerated {info} "
              f"stale_marked={result['stale_marked']}")
        for line in format_section_lines(section_table(sections, budgets)):
            print(f"[bridge] {line}")
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
