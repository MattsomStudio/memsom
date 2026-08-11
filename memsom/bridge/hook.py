"""memsom_hook — Gate #3 native-tool arm via Claude Code hooks.

The broker (memsom_broker) guards MCP doors; this guards the doors Anthropic
owns.  Claude Code's native tools (WebFetch/WebSearch/Bash/Edit/...) are not MCP
and bypass the broker entirely.  Hooks are the supported interception point:

  PostToolUse  on WebFetch/WebSearch  -> hook-post: untrusted content entered,
                                         drop the session's taint floor.
  PreToolUse   on Bash/Edit/Write/... -> hook-pre:  if the session is tainted
                                         below the tool's required floor, emit a
                                         `permissionDecision: deny` and Claude
                                         Code blocks the tool.

SHADOW FIRST (PLAN.md Phase 9, Matt's Q2).  `bridge.hook_mode` defaults to
"shadow": every PreToolUse decision is computed and logged to
`~/.claude/gate3_shadow.jsonl` (or $MEMDAG_HOOK_SHADOW_LOG) exactly as it would
be, but a would-have-denied verdict never reaches stdout -- nothing is ever
blocked.  Flipping a rule to "enforcing" (per action) is a separate change.

Both key on the Claude `session_id` delivered in the hook payload (bridged into
memsom via memsom_session.ensure_session), so taint follows the live
conversation.  The shared brain is memsom_session + memsom_policy +
memsom_capgate — identical to the broker, different hands.

The hook is wired in settings.json (run `memsom hook-print-config`):
  "PostToolUse": [{"matcher": "WebFetch|WebSearch",
                   "hooks": [{"type": "command", "command": "memsom hook-post"}]}],
  "PreToolUse":  [{"matcher": "Bash|Edit|Write|MultiEdit",
                   "hooks": [{"type": "command", "command": "memsom hook-pre"}]}]

POLICY DEFAULT DIFFERS FROM THE BROKER.  Native tools are an OPEN set, so a
default-deny would brick the agent the first time it ran an unlisted tool.  The
hook layer therefore defaults to ALLOW with an explicit consequential list
(below).  Override with a file at $MEMDAG_HOOK_POLICY or ~/.memdag/hook_policy.json.

FAIL-OPEN ON INTERNAL ERROR (deliberate).  A hook that crashed closed would
block every Bash/Edit and make Claude Code unusable.  This is a single-user dev
tool and a defense-in-depth layer (the broker is the stricter one), so on an
unexpected exception the hook logs to stderr and ALLOWS.  The gate still fires
whenever memsom can read the session floor; only true internal faults fail open.

Hook contract verified against code.claude.com/docs/en/hooks.md (2026-06-23):
PreToolUse deny via {"hookSpecificOutput":{"hookEventName":"PreToolUse",
"permissionDecision":"deny","permissionDecisionReason":...}}; PostToolUse stdin
carries session_id, tool_name, tool_output.
"""

import argparse
import json
import sys
from pathlib import Path

import memsom
from memsom.integrity import capgate as memsom_capgate
from memsom.kernel import policy as memsom_policy
from memsom.storage import session as memsom_session
from memsom import tuning as memsom_tuning

# Built-in default: ALLOW unlisted tools; gate the known consequential ones;
# taint on the untrusted-ingress tools.  Used when no override file is present.
DEFAULT_HOOK_POLICY = {
    "default": "allow",
    "rules": [
        {"tool": "WebFetch", "required": "external", "taints": "external"},
        {"tool": "WebSearch", "required": "external", "taints": "external"},
        {"tool": "Bash", "required": "user"},
        {"tool": "Edit", "required": "user"},
        {"tool": "Write", "required": "user"},
        {"tool": "MultiEdit", "required": "user"},
        {"tool": "NotebookEdit", "required": "user"},
    ],
}


def hook_policy_path():
    env = memsom_tuning.resolve("bridge.hook_policy_path")
    if env:
        return Path(env).expanduser()
    p = Path.home() / ".memdag" / "hook_policy.json"
    return p if p.exists() else None


def load_hook_policy():
    """Load the override policy if present; else the built-in default.  A
    malformed override falls back to the built-in (still gates consequential
    tools) rather than disabling the gate."""
    path = hook_policy_path()
    if path is not None:
        try:
            return memsom_policy.load_policy(path)
        except Exception as exc:  # noqa: BLE001 — fall back, never disable the gate
            print(f"[memsom-hook] policy {path} unusable, using built-in default: {exc}",
                  file=sys.stderr)
    return memsom_policy._normalize(DEFAULT_HOOK_POLICY)


def hook_mode() -> str:
    """'shadow' (default, PLAN.md Phase 9) or 'enforcing'."""
    return str(memsom_tuning.resolve("bridge.hook_mode") or "shadow").strip().lower()


def shadow_log_path() -> Path:
    override = memsom_tuning.resolve("bridge.hook_shadow_log")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude" / "gate3_shadow.jsonl"


def log_shadow_decision(verdict: dict, tool: str) -> None:
    """Append one would-be PreToolUse decision to the shadow log (scripts/
    shadow_summary.py + shadow_falsepos.py's input). Never raises -- a shadow
    log write failure must not turn into a hook failure."""
    row = {"ts": memsom.now_iso(), "event": "pre", "action": tool, "rule": tool,
           "decision": verdict["decision"], "required": verdict["required_name"],
           "floor": verdict["floor_name"]}
    path = shadow_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Core (testable without stdin/CLI)
# ---------------------------------------------------------------------------

def decide_pre(conn, policy, session_id, tool) -> dict:
    """Ensure the session, then return the capability verdict for *tool*."""
    memsom_session.ensure_session(conn, session_id, "user")
    floor = memsom_session.current_floor(conn, session_id)
    required = memsom_policy.required_floor(policy, tool)
    return memsom_capgate.check_capability(conn, session_id, floor, tool, required)


def apply_post(conn, policy, session_id, tool):
    """If *tool* taints (per policy), ensure the session and lower its floor.
    Returns the new floor on a taint, or None if the tool does not taint.

    MS-18: the lower_floor write is the one that can hit a transient lock (the
    finding names a routine write elsewhere in the store held past 5s as the
    realistic trigger). If it fails, record_pending_taint makes one more
    attempt at a SEPARATE, less-contended row before re-raising to the caller,
    which still logs to stderr -- so the taint is either applied now, recorded
    for the next ensure_session to reconcile, or (only if BOTH writes fail)
    lost with a loud message rather than a silent one.
    """
    taint = memsom_policy.taints(policy, tool)
    if taint is None:
        return None
    memsom_session.ensure_session(conn, session_id, "user")
    reason = f"hook:{tool}"
    try:
        return memsom_session.lower_floor(conn, session_id, taint, tool, reason=reason)
    # FAILOPEN: not actually open -- caught only to record a pending taint
    # before propagating; the bare `raise` below re-throws unchanged so
    # cmd_hook_post's own handler still logs it to stderr.
    except Exception:
        memsom_session.record_pending_taint(conn, session_id, taint, tool, reason)
        raise


def pre_output(verdict, tool):
    """Build the PreToolUse stdout JSON for a verdict, or None to allow silently."""
    if verdict["decision"] != "deny":
        return None
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"memsom Gate #3: this session was tainted to {verdict['floor_name']} "
            f"because an untrusted source (e.g. a fetched web page) entered the "
            f"conversation, and {tool} requires {verdict['required_name']}. "
            f"The action is blocked to prevent injected instructions from taking "
            f"effect. Start a fresh session to perform it."),
    }}


# ---------------------------------------------------------------------------
# CLI handlers — read the hook JSON on stdin, fail OPEN on internal error
# ---------------------------------------------------------------------------

def _read_stdin_json():
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:  # noqa: BLE001
        return {}


def cmd_hook_pre(args):
    data = _read_stdin_json()
    sid = data.get("session_id")
    tool = data.get("tool_name", "")
    if not sid or not tool:
        return  # nothing to gate -> allow
    try:
        policy = load_hook_policy()
        conn = memsom.get_connection()
        try:
            verdict = decide_pre(conn, policy, sid, tool)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — fail OPEN (availability), logged
        print(f"[memsom-hook] pre error (failing open): {exc}", file=sys.stderr)
        return
    try:
        log_shadow_decision(verdict, tool)
    except Exception as exc:  # noqa: BLE001 -- logging must never break the gate
        print(f"[memsom-hook] shadow log write failed: {exc}", file=sys.stderr)
    if hook_mode() != "enforcing":
        return  # shadow mode: decision logged, nothing ever blocked
    out = pre_output(verdict, tool)
    if out is not None:
        print(json.dumps(out))


def cmd_hook_post(args):
    data = _read_stdin_json()
    sid = data.get("session_id")
    tool = data.get("tool_name", "")
    if not sid or not tool:
        return
    try:
        policy = load_hook_policy()
        conn = memsom.get_connection()
        try:
            apply_post(conn, policy, sid, tool)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[memsom-hook] post error: {exc}", file=sys.stderr)
        return


_CONFIG_SNIPPET = {
    "hooks": {
        "PostToolUse": [
            {"matcher": "WebFetch|WebSearch",
             "hooks": [{"type": "command", "command": "memsom hook-post"}]}
        ],
        "PreToolUse": [
            {"matcher": "Bash|Edit|Write|MultiEdit|NotebookEdit",
             "hooks": [{"type": "command", "command": "memsom hook-pre"}]}
        ],
    }
}


def cmd_hook_print_config(args):
    print("# Add to your Claude Code settings.json (~/.claude/settings.json):")
    print(json.dumps(_CONFIG_SNIPPET, indent=2))


# ---------------------------------------------------------------------------
# register / main
# ---------------------------------------------------------------------------

def register(subparsers):
    a = subparsers.add_parser("hook-pre",
                              help="PreToolUse hook: deny consequential tools on a tainted session")
    a.set_defaults(func=cmd_hook_pre)

    b = subparsers.add_parser("hook-post",
                              help="PostToolUse hook: taint the session after untrusted ingress")
    b.set_defaults(func=cmd_hook_post)

    c = subparsers.add_parser("hook-print-config",
                              help="print the settings.json hooks block for Gate #3")
    c.set_defaults(func=cmd_hook_print_config)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_hook")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
