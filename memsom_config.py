#!/usr/bin/env python3
"""memsom_config — safely wire `memsom-mcp` into MCP client config files.

The dangerous part of the whole project: these config files belong to the user
and may already hold other MCP servers. Every write here is guarded:
  - BACK UP the file first (<name>.bak).
  - MERGE, never overwrite — preserve all other servers/settings.
  - IDEMPOTENT — re-running yields a byte-identical file.
  - MALFORMED existing config -> do NOT write; return the snippet to paste.
  - ALWAYS write an ABSOLUTE path to the executable (GUI clients like Claude
    Desktop do not inherit the shell PATH — a bare name fails with spawn ENOENT).

Clients:
  - claude-code    : prefer `claude mcp add --scope user`; fallback ~/.claude.json (JSON)
  - claude-desktop : claude_desktop_config.json (JSON; mac/win/linux)
  - codex          : ~/.codex/config.toml (TOML; stdlib has no writer -> hand-append)
"""

import json
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def client_config_path(client, os_name=None, home=None):
    home = Path(home or Path.home())
    osn = (os_name or platform.system()).lower()
    if client == "codex":
        return home / ".codex" / "config.toml"
    if client == "claude-code":
        return home / ".claude.json"
    if client == "claude-desktop":
        if osn == "darwin":
            return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if osn == "windows":
            return home / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
        return home / ".config" / "Claude" / "claude_desktop_config.json"  # linux + other
    raise ValueError(f"unknown client {client!r}")


def _backup(path):
    bak = path.with_name(path.name + ".bak")
    shutil.copy2(path, bak)
    return bak


# ---------------------------------------------------------------------------
# JSON clients (Claude Desktop; Claude Code fallback)
# ---------------------------------------------------------------------------

def _json_entry(abs_exe, db_path):
    return {"command": abs_exe, "args": [], "env": {"MEMDAG_DB": db_path}}


def wire_json(path, abs_exe, db_path, print_only=False):
    path = Path(path)
    entry = _json_entry(abs_exe, db_path)
    snippet = json.dumps({"mcpServers": {"memsom": entry}}, indent=2)

    if print_only:
        return {"action": "print", "path": str(path), "snippet": snippet}

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mcpServers": {"memsom": entry}}, indent=2) + "\n",
                        encoding="utf-8")
        return {"action": "created", "path": str(path)}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return {"action": "malformed", "path": str(path), "snippet": snippet}
    if not isinstance(data, dict):
        return {"action": "malformed", "path": str(path), "snippet": snippet}

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers

    if servers.get("memsom") == entry:
        return {"action": "unchanged", "path": str(path)}

    _backup(path)
    servers["memsom"] = entry  # merge: only our key changes, others untouched
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {"action": "merged", "path": str(path)}


# ---------------------------------------------------------------------------
# Codex (TOML)
# ---------------------------------------------------------------------------

def _toml_block(abs_exe, db_path):
    # Single-quoted TOML literal strings: no escape processing, so Windows
    # backslash paths (C:\Users\...) pass through verbatim. tomllib reads them back.
    return ("\n[mcp_servers.memsom]\n"
            f"command = '{abs_exe}'\n"
            "args = []\n\n"
            "[mcp_servers.memsom.env]\n"
            f"MEMDAG_DB = '{db_path}'\n")


def wire_toml(path, abs_exe, db_path, print_only=False):
    path = Path(path)
    block = _toml_block(abs_exe, db_path)
    want = {"command": abs_exe, "args": [], "env": {"MEMDAG_DB": db_path}}

    # CONFIG-1: validate the generated block ROUND-TRIPS to exactly the intended
    # table before writing. The old guard only rejected a single quote; a newline
    # (or any other literal-string-breaking char) in a path slipped through and
    # produced an unparseable config.toml. tomllib here is read-only (stdlib).
    try:
        roundtrip = tomllib.loads(block).get("mcp_servers", {}).get("memsom")
    except tomllib.TOMLDecodeError:
        roundtrip = None
    if roundtrip != want:
        return {"action": "print", "path": str(path), "snippet": block}
    if print_only:
        return {"action": "print", "path": str(path), "snippet": block}

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block.lstrip("\n"), encoding="utf-8")
        return {"action": "created", "path": str(path)}

    text = path.read_text(encoding="utf-8")
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {"action": "malformed", "path": str(path), "snippet": block}

    existing = parsed.get("mcp_servers", {}).get("memsom")
    if existing == want:
        return {"action": "unchanged", "path": str(path)}
    if existing is not None:
        # Present but different (e.g. a stale path). We cannot edit TOML in place
        # without a writer and must not append a duplicate table key. Leave it and
        # ask the user to update the block by hand.
        return {"action": "exists_differs", "path": str(path), "snippet": block}

    new = text if text.endswith("\n") else text + "\n"
    merged = new + block
    # CONFIG-MERGE-INLINE-1: if the file already defines mcp_servers as an INLINE
    # table, parsed[...].get('memsom') is None (so existing is None above), but
    # appending a [mcp_servers.memsom] header produces invalid TOML. Validate the
    # merged result re-parses to our table BEFORE writing; otherwise refuse and ask
    # the user to merge by hand rather than corrupt their config.
    try:
        check = tomllib.loads(merged).get("mcp_servers", {}).get("memsom")
    except tomllib.TOMLDecodeError:
        check = None
    if check != want:
        return {"action": "exists_differs", "path": str(path), "snippet": block}

    _backup(path)
    path.write_text(merged, encoding="utf-8")
    return {"action": "merged", "path": str(path)}


# ---------------------------------------------------------------------------
# Claude Code: prefer the CLI, fall back to JSON
# ---------------------------------------------------------------------------

def wire_claude_code(abs_exe, db_path, print_only=False, home=None):
    cli = shutil.which("claude")
    cmd = ["claude", "mcp", "add", "memsom", "--scope", "user",
           "--env", f"MEMDAG_DB={db_path}", "--", abs_exe]
    if print_only:
        return {"action": "print", "client": "claude-code",
                "snippet": " ".join(cmd) if cli else json.dumps(
                    {"mcpServers": {"memsom": _json_entry(abs_exe, db_path)}}, indent=2)}
    if cli:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return {"action": "claude-cli", "path": "(claude mcp add --scope user)"}
            # CLI failed -> fall back to hand-editing the JSON config
        except Exception:
            pass  # subprocess itself failed (missing CLI, timeout) — same fallback
    return wire_json(client_config_path("claude-code", home=home), abs_exe, db_path)


# ---------------------------------------------------------------------------
# Dispatch + CLI
# ---------------------------------------------------------------------------

def wire(client, abs_exe, db_path, print_only=False, home=None, os_name=None):
    if client == "claude-code":
        return wire_claude_code(abs_exe, db_path, print_only=print_only, home=home)
    path = client_config_path(client, os_name=os_name, home=home)
    if client == "codex":
        return wire_toml(path, abs_exe, db_path, print_only=print_only)
    return wire_json(path, abs_exe, db_path, print_only=print_only)


def _default_exe():
    return shutil.which("memsom-mcp") or "memsom-mcp"


def register(subparsers):
    p = subparsers.add_parser("wire-config",
                              help="merge memsom into an MCP client's config (backup + idempotent)")
    p.add_argument("--client", choices=["claude-code", "claude-desktop", "codex", "all"],
                   default="all")
    p.add_argument("--exe", default=None, help="absolute path to memsom-mcp (default: resolve on PATH)")
    p.add_argument("--db", default=None, help="MEMDAG_DB path to wire (default: memsom's default)")
    p.add_argument("--print-only", action="store_true", help="print snippets; touch nothing")
    p.set_defaults(func=cmd_wire_config)


def cmd_wire_config(args):
    import memsom
    abs_exe = args.exe or _default_exe()
    db_path = args.db or str(memsom.db_path())
    clients = (["claude-code", "claude-desktop", "codex"]
               if args.client == "all" else [args.client])
    failed = False
    for c in clients:
        res = wire(c, abs_exe, db_path, print_only=args.print_only)
        action = res.get("action")
        if action in ("print", "malformed", "exists_differs"):
            print(f"[{c}] {action} -> {res.get('path', '')}")
            if res.get("snippet"):
                print(res["snippet"])
        else:
            print(f"[{c}] {action} -> {res.get('path', '')}")
        # BOOTSTRAP-1 / CFG-PRINT-SOFTFAIL-1: on a real (non print-only) run, treat
        # anything that is NOT a confirmed wire as a soft failure — a SUCCESS
        # whitelist, not a failure denylist. The denylist missed 'print' (wire_toml/
        # wire_json refusing to write, e.g. an apostrophe in the path), which left
        # the client unconfigured while wire-config still exited 0.
        if not args.print_only and action not in ("created", "merged", "unchanged", "claude-cli"):
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="memsom-wire-config")
    ap.add_argument("--client", choices=["claude-code", "claude-desktop", "codex", "all"],
                    default="all")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--print-only", action="store_true")
    sys.exit(cmd_wire_config(ap.parse_args()))
