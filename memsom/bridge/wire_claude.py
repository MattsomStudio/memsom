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
     + PROMPT HOOK — a UserPromptSubmit hook running `<memsom> hook-prompt`
                (top memories as added context; see interface/prompt_hook.py).
                Text-compare upgraded in place on re-run; --no-prompt-hook skips.
  3. CLAUDE.md— seed/refresh the memsom-managed memory block (via memsom_claude).

Gate #3 (PreToolUse/PostToolUse taint hooks) is OPT-IN behind --with-gate — it can
deny tools, so a tester never gets a blocking gate unasked.

settings.json structure (Claude Code): {"hooks": {"Stop": [ {"hooks": [ {"type":
"command", "command": ...} ]} ], "PreToolUse": [ {"matcher": ..., "hooks": [...]} ]}}.
"""
from __future__ import annotations

import argparse
import json
import os
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
    # The bundled claude/ dir sits NEXT TO the memsom package — repo root for a
    # checkout, site-packages for a wheel (pyproject force-include installs it as
    # site-packages/claude). This file is memsom/bridge/wire_claude.py, so that
    # shared parent is parents[2].
    return Path(__file__).resolve().parents[2] / "claude" / "skills"


def resolve_exe():
    """An ABSOLUTE path to the memsom console script, or a `python -m` fallback.

    A hook entry of bare ``"memsom"`` only works when the directory holding the
    console script is on the PATH Claude Code itself was launched with.  With a
    venv install that is rarely true (the venv's ``bin``/``Scripts`` is only on
    PATH inside an activated shell; a GUI-launched Claude never sees it), so the
    Stop/prompt hooks failed silently.  Resolution order:

      1. the console script that is RUNNING us (``sys.argv[0]``, when it is an
         absolute existing file named memsom) — the one path guaranteed to work;
      2. the sibling of the running interpreter (``<venv>/bin/memsom`` or
         ``<venv>\\Scripts\\memsom.exe``);
      3. ``shutil.which("memsom")``;
      4. ``"<sys.executable>" -m memsom.interface.cli`` — the interpreter we run
         under can always import the package it was installed into.
    """
    argv0 = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if argv0 and argv0.is_absolute() and argv0.is_file() \
            and argv0.stem.lower() == "memsom":
        return str(argv0)
    exe_dir = Path(sys.executable).resolve().parent
    for name in ("memsom", "memsom.exe"):
        cand = exe_dir / name
        if cand.is_file():
            return str(cand)
    found = shutil.which("memsom")
    if found:
        return str(Path(found).resolve())
    return f'"{sys.executable}" -m memsom.interface.cli'


def _default_exe():
    return resolve_exe()


def is_bare_command(cmd) -> bool:
    """True for a hook command whose executable token is an unqualified
    ``memsom`` (``memsom ...`` or ``"memsom" ...``) — i.e. PATH-dependent."""
    if not isinstance(cmd, str):
        return False
    head = cmd.strip().split(" ", 1)[0].strip('"').strip("'")
    return head.lower() in ("memsom", "memsom.exe")


# ---------------------------------------------------------------------------
# Hook construction (pure)
# ---------------------------------------------------------------------------

def _cmd(abs_exe, sub):
    # Quote the exe so a path with spaces survives the shell; bare names quote
    # fine.  A pre-composed `"<python>" -m memsom.interface.cli` fallback
    # (resolve_exe step 4) is already quoted and must not be wrapped again.
    if abs_exe.startswith('"') and " -m " in abs_exe:
        return f"{abs_exe} {sub}"
    return f'"{abs_exe}" {sub}'


def stop_group(abs_exe):
    return {"hooks": [{"type": "command", "command": _cmd(abs_exe, "bridge-render")}]}


PROMPT_HOOK_TIMEOUT_S = 5   # Claude Code cancels past this; the hook's own
                            # deadline (prompt_hook_deadline_ms) is far shorter.


def prompt_hook_entry(abs_exe):
    """The canonical UserPromptSubmit hook handler (retrieval -> added context)."""
    return {"type": "command", "command": _cmd(abs_exe, "hook-prompt"),
            "timeout": PROMPT_HOOK_TIMEOUT_S}


def prompt_group(abs_exe):
    return {"hooks": [prompt_hook_entry(abs_exe)]}


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


def _upsert_command(groups, substr, entry):
    """Text-compare upgrade (same discipline as the managed CLAUDE.md block):
    find every handler whose command contains *substr*; if the first one is
    byte-identical to *entry* nothing changes, otherwise it is replaced in
    place and any later duplicates are dropped. No match -> append a new group.
    Returns True when *groups* was modified."""
    found = []
    for gi, g in enumerate(groups):
        if not isinstance(g, dict):
            continue
        for hi, h in enumerate(g.get("hooks") or []):
            if isinstance(h, dict) and substr in (h.get("command") or ""):
                found.append((gi, hi))
    if not found:
        groups.append({"hooks": [dict(entry)]})
        return True
    gi, hi = found[0]
    current = groups[gi]["hooks"][hi]
    changed = current != entry
    if changed:
        groups[gi]["hooks"][hi] = dict(entry)
    for gj, hj in reversed(found[1:]):
        del groups[gj]["hooks"][hj]
        if not groups[gj]["hooks"]:
            del groups[gj]
        changed = True
    return changed


def _upgrade_bare(groups, substr, entry):
    """Replace an existing PATH-dependent (bare ``memsom``) handler for *substr*
    with *entry* in place, iff *entry* itself is not bare.  Returns True when
    something changed.  A user's own absolute/custom command is left alone."""
    if is_bare_command(entry.get("command")):
        return False
    changed = False
    for g in groups:
        if not isinstance(g, dict):
            continue
        for hi, h in enumerate(g.get("hooks") or []):
            if (isinstance(h, dict) and substr in (h.get("command") or "")
                    and is_bare_command(h.get("command"))):
                g["hooks"][hi] = dict(entry)
                changed = True
    return changed


def merge_hooks(data, abs_exe, *, with_gate=False, with_prompt_hook=True):
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
    elif _upgrade_bare(stop, "bridge-render", stop_group(abs_exe)["hooks"][0]):
        changed.append("Stop")

    if with_prompt_hook:
        ups = hooks.setdefault("UserPromptSubmit", [])
        if not isinstance(ups, list):
            raise ValueError("settings 'hooks.UserPromptSubmit' is not a list")
        if _upsert_command(ups, "hook-prompt", prompt_hook_entry(abs_exe)):
            changed.append("UserPromptSubmit")

    if with_gate:
        for event, groups in gate_event_groups(abs_exe).items():
            probe = "hook-post" if event == "PostToolUse" else "hook-pre"
            arr = hooks.setdefault(event, [])
            if not isinstance(arr, list):
                raise ValueError(f"settings 'hooks.{event}' is not a list")
            if not _has_command(arr, probe):
                arr.extend(groups)
                changed.append(event)
            elif _upgrade_bare(arr, probe, groups[0]["hooks"][0]):
                changed.append(event)
    return changed


# ---------------------------------------------------------------------------
# settings.json IO (mirrors memsom_config.wire_json contract)
# ---------------------------------------------------------------------------

def _backup(path):
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    return bak


def wire_settings(path, abs_exe, *, with_gate=False, print_only=False,
                  with_prompt_hook=True):
    path = Path(path)
    fresh = {"hooks": {"Stop": [stop_group(abs_exe)]}}
    if with_prompt_hook:
        fresh["hooks"]["UserPromptSubmit"] = [prompt_group(abs_exe)]
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
        changed = merge_hooks(data, abs_exe, with_gate=with_gate,
                              with_prompt_hook=with_prompt_hook)
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

def memory_dir_candidates(home=None):
    """Existing ``<home>/.claude/projects/*/memory`` dirs (Claude Code creates
    one per working directory).  $MEMDAG_BRIDGE_MEMORY_DIR wins when no
    explicit home is given."""
    if home is None and os.environ.get("MEMDAG_BRIDGE_MEMORY_DIR"):
        p = Path(os.environ["MEMDAG_BRIDGE_MEMORY_DIR"])
        return [p] if p.is_dir() else []
    base = Path(home) if home else Path.home()
    return sorted(d for d in (base / ".claude" / "projects").glob("*/memory")
                  if d.is_dir())


def scaffold_memory(home=None, *, print_only=False):
    """Scaffold the memory layout (projects/, projects/INDEX.md, canonical.json)
    in the largest existing memory dir.  A fresh machine may have none yet —
    then this is skipped and ``bridge-render`` scaffolds on its first run."""
    from memsom.bridge import bridge_import as bi
    cands = memory_dir_candidates(home)
    if not cands:
        return {"action": "skipped",
                "detail": "no memory dir yet (bridge-render scaffolds it on first run)"}
    target = max(cands, key=lambda d: len(list(d.glob("*.md"))))
    if print_only:
        return {"action": "print", "path": str(target)}
    try:
        return {"action": "scaffolded", "path": str(target),
                **bi.scaffold_memory_dir(target)}
    except Exception as exc:  # noqa: BLE001
        return {"action": "error", "path": str(target), "detail": repr(exc)}


def wire_claude(*, home=None, abs_exe=None, skills_src=None, with_gate=False,
                force=False, print_only=False, with_prompt_hook=True):
    abs_exe = abs_exe or _default_exe()
    skills_src = Path(skills_src) if skills_src else default_skills_src()
    out = {}
    out["skills"] = wire_skills(skills_src, skills_dst_dir(home),
                                force=force, print_only=print_only)
    out["settings"] = wire_settings(settings_path(home), abs_exe,
                                     with_gate=with_gate, print_only=print_only,
                                     with_prompt_hook=with_prompt_hook)
    # CLAUDE.md (managed block) — imported lazily so this module is independent.
    # When a home is given, target THAT home's CLAUDE.md (so the loop stays self-
    # consistent and a test/scratch home never touches the real file); otherwise let
    # memsom_claude resolve its default ($CLAUDE_MD_PATH or ~/.claude/CLAUDE.md).
    from memsom.bridge import claude as memsom_claude
    claude_path = (Path(home) / ".claude" / "CLAUDE.md") if home else None
    if print_only:
        out["claude_md"] = {"action": "print", "snippet": memsom_claude.render_block()}
    else:
        try:
            out["claude_md"] = memsom_claude.sync(path=claude_path)
        except Exception as exc:  # noqa: BLE001
            out["claude_md"] = {"action": "error", "detail": repr(exc)}
    out["memory_dir"] = scaffold_memory(home, print_only=print_only)
    out["exe"] = abs_exe
    return out


def cmd_wire_claude(args):
    res = wire_claude(abs_exe=args.exe, skills_src=args.skills_src,
                      with_gate=args.with_gate, force=args.force,
                      print_only=args.print_only,
                      with_prompt_hook=not args.no_prompt_hook)
    failed = False

    for name, action in res["skills"]:
        print(f"[skill:{name}] {action}")
    s = res["settings"]
    print(f"[settings] {s['action']} -> {s.get('path', '')}")
    if is_bare_command(res.get("exe", "")):
        print("[settings] WARNING: memsom could not be resolved to an absolute path; "
              "hooks will only fire if `memsom` is on the PATH Claude Code was "
              "launched with. Re-run with --exe <abs path>.")
    if s.get("snippet") and s["action"] in ("print", "malformed"):
        print(s["snippet"])
    cm = res["claude_md"]
    print(f"[claude.md] {cm['action']}: {cm.get('detail') or cm.get('path', '')}")
    md = res.get("memory_dir") or {}
    print(f"[memory-dir] {md.get('action')}: {md.get('detail') or md.get('path', '')}")
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
    p.add_argument("--no-prompt-hook", action="store_true",
                   help="skip the UserPromptSubmit retrieval hook (memsom hook-prompt)")
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
    ap.add_argument("--no-prompt-hook", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-only", action="store_true")
    sys.exit(cmd_wire_claude(ap.parse_args()))
