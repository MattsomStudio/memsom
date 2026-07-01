#!/usr/bin/env python3
"""memsom_wire_claude — install the Claude Code memory loop for a tester.

Sibling of memsom_config (the MCP wiring); same safety contract: BACK UP first,
MERGE never overwrite, IDEMPOTENT, malformed config -> refuse + print. Three pieces:

  1. SKILLS   — copy the bundled skills (claude/skills/*) into ~/.claude/skills/.
                NEVER overwrites an existing skill dir without --force (this is what
                protects a user's own same-named skills); --force backs up to *.bak.
  2. STOP HOOK— merge a Stop hook that runs `<memsom> bridge-render` into
                ~/.claude/settings.json (regenerates MEMORY.md from the store on
                session end). Deduped on re-run; everything else preserved.
  3. CLAUDE.md— seed/refresh the memsom-managed memory block (via memsom_claude).

Gate #3 (PreToolUse/PostToolUse taint hooks) is OPT-IN behind --with-gate — it can
deny tools, so a tester never gets a blocking gate unasked.

settings.json structure (Claude Code): {"hooks": {"Stop": [ {"hooks": [ {"type":
"command", "command": ...} ]} ], "PreToolUse": [ {"matcher": ..., "hooks": [...]} ]}}.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def settings_path(home=None):
    return Path(home or Path.home()) / ".claude" / "settings.json"


def skills_dst_dir(home=None):
    return Path(home or Path.home()) / ".claude" / "skills"


def default_skills_src():
    # Works when running from the repo (python memsom_cli.py wire-claude); bootstrap
    # passes an explicit --skills-src for the installed case.
    return Path(__file__).resolve().parent / "claude" / "skills"


def _default_exe():
    return shutil.which("memsom") or "memsom"


# ---------------------------------------------------------------------------
# Hook construction (pure)
# ---------------------------------------------------------------------------

def _cmd(abs_exe, sub):
    # Quote the exe so a path with spaces survives the shell; bare names quote fine.
    return f'"{abs_exe}" {sub}'


def stop_group(abs_exe):
    return {"hooks": [{"type": "command", "command": _cmd(abs_exe, "bridge-render")}]}


def gate_event_groups(abs_exe):
    """The opt-in Gate #3 taint hooks (mirrors memsom_hook._CONFIG_SNIPPET)."""
    return {
        "PostToolUse": [{"matcher": "WebFetch|WebSearch",
                         "hooks": [{"type": "command", "command": _cmd(abs_exe, "hook-post")}]}],
        "PreToolUse": [{"matcher": "Bash|Edit|Write|MultiEdit|NotebookEdit",
                        "hooks": [{"type": "command", "command": _cmd(abs_exe, "hook-pre")}]}],
    }


def _has_command(groups, substr):
    """True if any hook command under *groups* contains *substr* (dedupe probe).
    Tolerates malformed (non-dict) entries without crashing."""
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        for h in (g.get("hooks") or []):
            if isinstance(h, dict) and substr in (h.get("command") or ""):
                return True
    return False


def merge_hooks(data, abs_exe, *, with_gate=False):
    """Mutate *data* (a settings dict) to add our hooks. Returns the list of events
    actually changed (empty => already current). Raises ValueError if the existing
    'hooks' shape is not a dict (caller treats as malformed)."""
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("settings 'hooks' is not an object")
    changed = []

    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        raise ValueError("settings 'hooks.Stop' is not a list")
    if not _has_command(stop, "bridge-render"):
        stop.append(stop_group(abs_exe))
        changed.append("Stop")

    if with_gate:
        for event, groups in gate_event_groups(abs_exe).items():
            probe = "hook-post" if event == "PostToolUse" else "hook-pre"
            arr = hooks.setdefault(event, [])
            if not isinstance(arr, list):
                raise ValueError(f"settings 'hooks.{event}' is not a list")
            if not _has_command(arr, probe):
                arr.extend(groups)
                changed.append(event)
    return changed


# ---------------------------------------------------------------------------
# settings.json IO (mirrors memsom_config.wire_json contract)
# ---------------------------------------------------------------------------

def _backup(path):
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    return bak


def wire_settings(path, abs_exe, *, with_gate=False, print_only=False):
    path = Path(path)
    fresh = {"hooks": {"Stop": [stop_group(abs_exe)]}}
    if with_gate:
        fresh["hooks"].update(gate_event_groups(abs_exe))
    snippet = json.dumps(fresh, indent=2)

    if print_only:
        return {"action": "print", "path": str(path), "snippet": snippet}

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snippet + "\n", encoding="utf-8")
        return {"action": "created", "path": str(path)}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {"action": "malformed", "path": str(path), "snippet": snippet}
    if not isinstance(data, dict):
        return {"action": "malformed", "path": str(path), "snippet": snippet}

    try:
        changed = merge_hooks(data, abs_exe, with_gate=with_gate)
    except ValueError:
        return {"action": "malformed", "path": str(path), "snippet": snippet}

    if not changed:
        return {"action": "unchanged", "path": str(path)}
    _backup(path)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {"action": "merged", "path": str(path), "events": changed}


# ---------------------------------------------------------------------------
# Skills copy (no-overwrite guard is the whole point)
# ---------------------------------------------------------------------------

def wire_skills(src_dir, dst_dir, *, force=False, print_only=False):
    """Copy each skill subdir from *src_dir* into *dst_dir*. Returns a list of
    (name, action) where action is installed | updated | exists-skipped | print."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    results = []
    if not src_dir.is_dir():
        return results
    for skill in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        dst = dst_dir / skill.name
        if print_only:
            results.append((skill.name, "print"))
            continue
        if dst.exists():
            if not force:
                results.append((skill.name, "exists-skipped"))   # protect user's own
                continue
            bak = dst.with_name(dst.name + ".bak")
            if bak.exists():
                shutil.rmtree(bak)
            shutil.copytree(dst, bak)                              # back up before clobber
            shutil.rmtree(dst)                                     # replace, not merge
            shutil.copytree(skill, dst)                            # (stale files don't survive)
            results.append((skill.name, "updated"))
        else:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(skill, dst)
            results.append((skill.name, "installed"))
    return results


# ---------------------------------------------------------------------------
# Orchestration + CLI
# ---------------------------------------------------------------------------

def wire_claude(*, home=None, abs_exe=None, skills_src=None, with_gate=False,
                force=False, print_only=False):
    abs_exe = abs_exe or _default_exe()
    skills_src = Path(skills_src) if skills_src else default_skills_src()
    out = {}
    out["skills"] = wire_skills(skills_src, skills_dst_dir(home),
                                force=force, print_only=print_only)
    out["settings"] = wire_settings(settings_path(home), abs_exe,
                                     with_gate=with_gate, print_only=print_only)
    # CLAUDE.md (managed block) — imported lazily so this module is independent.
    # When a home is given, target THAT home's CLAUDE.md (so the loop stays self-
    # consistent and a test/scratch home never touches the real file); otherwise let
    # memsom_claude resolve its default ($CLAUDE_MD_PATH or ~/.claude/CLAUDE.md).
    import memsom_claude
    claude_path = (Path(home) / ".claude" / "CLAUDE.md") if home else None
    if print_only:
        out["claude_md"] = {"action": "print", "snippet": memsom_claude.render_block()}
    else:
        try:
            out["claude_md"] = memsom_claude.sync(path=claude_path)
        except Exception as exc:  # noqa: BLE001
            out["claude_md"] = {"action": "error", "detail": repr(exc)}
    return out


def cmd_wire_claude(args):
    res = wire_claude(abs_exe=args.exe, skills_src=args.skills_src,
                      with_gate=args.with_gate, force=args.force,
                      print_only=args.print_only)
    failed = False

    for name, action in res["skills"]:
        print(f"[skill:{name}] {action}")
    s = res["settings"]
    print(f"[settings] {s['action']} -> {s.get('path', '')}")
    if s.get("snippet") and s["action"] in ("print", "malformed"):
        print(s["snippet"])
    cm = res["claude_md"]
    print(f"[claude.md] {cm['action']}: {cm.get('detail') or cm.get('path', '')}")
    if cm.get("snippet") and cm["action"] == "print":
        print(cm["snippet"])

    if not args.print_only:
        # success whitelist (mirrors wire-config): anything else is a soft failure.
        if s["action"] not in ("created", "merged", "unchanged"):
            failed = True
        if cm["action"] == "error":
            failed = True
    return 1 if failed else 0


def register(subparsers):
    p = subparsers.add_parser(
        "wire-claude",
        help="install the Claude Code memory loop (skills + Stop hook + CLAUDE.md)")
    p.add_argument("--exe", default=None,
                   help="absolute path to the memsom executable (default: resolve on PATH)")
    p.add_argument("--skills-src", default=None,
                   help="dir of bundled skills (default: <repo>/claude/skills)")
    p.add_argument("--with-gate", action="store_true",
                   help="also wire the opt-in Gate #3 taint hooks (can deny tools)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing same-named skill (backs it up to *.bak)")
    p.add_argument("--print-only", action="store_true",
                   help="print what would be wired; touch nothing")
    p.set_defaults(func=cmd_wire_claude)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="memsom-wire-claude")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--skills-src", default=None)
    ap.add_argument("--with-gate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    sys.exit(cmd_wire_claude(ap.parse_args()))
