#!/usr/bin/env python3
"""bootstrap.py — one-command setup for a memdag friend-beta tester.

Pure stdlib, cross-platform. From a cloned repo:

    python3 bootstrap.py            # macOS / Linux
    py bootstrap.py                 # Windows

What it does, in order:
  1. Check Python >= 3.12 (the only hard stop).
  2. Install memdag into an isolated env (pipx if present, else a venv in ~/.memdag).
  3. Install Ollama + pull the embedding model (graceful-fail: prints manual steps
     and CONTINUES — memdag still works on BM25 until Ollama is present).
  4. `memdag init` -> create + migrate the DB, capture its path.
  5. Opt-in: offer to seed from your OWN local chat history.
  6. Wire your MCP client config(s) to launch memdag (absolute exe path; backup+merge).
  7. Print a final report.

The orchestration lives in main(); the decision logic (OS, install plan, exe path,
opt-in) is factored into pure functions so it can be unit-tested without running
any installer.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PY = (3, 12)
DATA_DIR_DEFAULT = "~/.memdag"
EMBED_MODEL = "nomic-embed-text"
CLIENTS = ["claude-code", "claude-desktop", "codex"]


# ---------------------------------------------------------------------------
# Pure, testable helpers
# ---------------------------------------------------------------------------

def python_ok(info=None):
    info = info or sys.version_info
    return (info[0], info[1]) >= MIN_PY


def os_key(os_name=None):
    n = (os_name or platform.system()).lower()
    if n == "darwin":
        return "macos"
    if n == "windows":
        return "windows"
    return "linux"


def ollama_install_plan(os_name=None, machine=None):
    k = os_key(os_name)
    if k == "macos":
        return {"os": "macos", "primary": ["brew", "install", "ollama"], "needs": "brew",
                "manual": "Install Ollama from https://ollama.com/download (the .dmg has no "
                          "silent flag), then run: ollama pull " + EMBED_MODEL}
    if k == "windows":
        return {"os": "windows", "primary": ["winget", "install", "--silent", "Ollama.Ollama"],
                "needs": None,
                "manual": "Install OllamaSetup.exe from https://ollama.com/download, then run: "
                          "ollama pull " + EMBED_MODEL}
    return {"os": "linux",
            "primary": ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            "needs": None,
            "manual": "Run: curl -fsSL https://ollama.com/install.sh | sh  (needs sudo + "
                      "systemd), then: ollama pull " + EMBED_MODEL}


def _bin_dir_name(os_name=None):
    return "Scripts" if os_key(os_name) == "windows" else "bin"


def _exe_suffix(os_name=None):
    return ".exe" if os_key(os_name) == "windows" else ""


def venv_exe_path(data_dir, name, os_name=None):
    base = Path(data_dir).expanduser() / "venv" / _bin_dir_name(os_name)
    return base / f"{name}{_exe_suffix(os_name)}"


def pipx_exe_path(name, os_name=None, home=None):
    home = Path(home or Path.home())
    return (home / ".local" / "pipx" / "venvs" / "memdag" / _bin_dir_name(os_name)
            / f"{name}{_exe_suffix(os_name)}")


def resolve_exe_path(method, data_dir, name="memdag-mcp", os_name=None, home=None):
    """Absolute path to an installed console script. Always absolute — GUI MCP
    clients do not inherit the shell PATH, so a bare name fails with spawn ENOENT."""
    if method == "venv":
        return venv_exe_path(data_dir, name, os_name=os_name)
    return pipx_exe_path(name, os_name=os_name, home=home)


def should_ingest(answer):
    return answer.strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# IO steps (each prints; injectable runner/which for testing the Ollama seam)
# ---------------------------------------------------------------------------

def _pull_model(runner):
    try:
        r = runner(["ollama", "pull", EMBED_MODEL], capture_output=True, text=True, timeout=1800)
        return {"ok": getattr(r, "returncode", 1) == 0, "model": EMBED_MODEL}
    except Exception as exc:
        return {"ok": False, "model": EMBED_MODEL, "reason": str(exc)}


def install_ollama(plan, runner=None, which=None):
    """Install Ollama + pull the embed model. NEVER raises — returns a status dict.
    On any failure the caller prints plan['manual'] and continues."""
    which = which or shutil.which
    runner = runner or subprocess.run
    if which("ollama"):
        return _pull_model(runner)
    needs = plan.get("needs")
    if needs and not which(needs):
        return {"ok": False, "reason": f"{needs} not found", "manual": plan["manual"]}
    try:
        r = runner(plan["primary"], timeout=900)
        if getattr(r, "returncode", 1) != 0:
            return {"ok": False, "reason": "installer returned non-zero", "manual": plan["manual"]}
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "manual": plan["manual"]}
    status = _pull_model(runner)
    if not status.get("ok"):
        status["manual"] = plan["manual"]
    return status


def _run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    # BOOTSTRAP-1: a failed install step must raise, not be silently ignored and
    # leave a half-configured install that still prints "=== done ===".
    kw.setdefault("check", True)
    return subprocess.run(cmd, **kw)


def install_memdag(repo_dir, data_dir, os_name=None):
    """Install memdag isolated; return (method, mcp_exe, cli_exe)."""
    if shutil.which("pipx"):
        _run(["pipx", "install", "--force", str(repo_dir)])
        method = "pipx"
    else:
        venv = Path(data_dir).expanduser() / "venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        pip = venv_exe_path(data_dir, "pip", os_name=os_name)
        _run([str(pip), "install", str(repo_dir)])
        method = "venv"
    mcp = resolve_exe_path(method, data_dir, "memdag-mcp", os_name=os_name)
    cli = resolve_exe_path(method, data_dir, "memdag", os_name=os_name)
    return method, mcp, cli


def run_init(cli_exe, data_dir):
    r = subprocess.run([str(cli_exe), "init", "--data-dir", data_dir],
                       capture_output=True, text=True)
    # BOOTSTRAP-1: on failure, abort with the captured stderr instead of guessing
    # a DB path and proceeding as if setup succeeded.
    if r.returncode != 0:
        raise RuntimeError(
            f"`memdag init` failed (exit {r.returncode}): {(r.stderr or '').strip()}"
        )
    db = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else \
        str(Path(data_dir).expanduser() / "memdag.db")
    return db


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="bootstrap.py", description="Set up memdag for testing.")
    ap.add_argument("--data-dir", default=DATA_DIR_DEFAULT)
    ap.add_argument("--print-only", action="store_true",
                    help="print client-config snippets instead of writing them")
    ap.add_argument("--no-ingest", action="store_true", help="skip the chat-history seeding offer")
    ap.add_argument("--clients", default=None,
                    help="comma-separated subset of: " + ",".join(CLIENTS))
    args = ap.parse_args(argv)

    print("=== memdag bootstrap ===")

    # 1. Python gate (the one hard stop)
    if not python_ok():
        print(f"ERROR: Python {MIN_PY[0]}.{MIN_PY[1]}+ required; you have "
              f"{sys.version_info[0]}.{sys.version_info[1]}. Install a newer Python and re-run.",
              file=sys.stderr)
        return 1

    repo_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir
    os_name = platform.system()

    # 2. Install memdag
    print("\n[1/6] Installing memdag (isolated)...")
    try:
        method, mcp_exe, cli_exe = install_memdag(repo_dir, data_dir, os_name=os_name)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: memdag install failed (exit {exc.returncode}). "
              f"Aborting — nothing was wired.", file=sys.stderr)
        return 1
    print(f"  installed via {method}; server exe -> {mcp_exe}")

    # 3. Ollama (graceful-fail, continue)
    print("\n[2/6] Installing Ollama + embedding model (required for semantic search)...")
    plan = ollama_install_plan(os_name)
    ostatus = install_ollama(plan)
    if ostatus.get("ok"):
        print(f"  ollama ready; {EMBED_MODEL} pulled.")
    else:
        print(f"  WARNING: Ollama not ready ({ostatus.get('reason', '?')}). memdag will run on "
              f"BM25-only until you fix this:\n    {ostatus.get('manual', '')}")

    # 4. init
    print("\n[3/6] Creating the database...")
    try:
        db_path = run_init(cli_exe, data_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  DB at {db_path}")

    # 5. Opt-in chat ingest
    if not args.no_ingest:
        print("\n[4/6] Seed from your OWN chat history? This reads local Claude Code / Codex "
              "transcripts and ingests them stamped 'user'. Nothing leaves your machine.")
        if should_ingest(input("  Ingest your chat history now? [y/N] ")):
            subprocess.run([str(cli_exe), "ingest-chats", "--yes"])
        else:
            print("  skipped (you can run `memdag ingest-chats` later).")
    else:
        print("\n[4/6] Chat ingest skipped (--no-ingest).")

    # 6. Wire client configs
    print("\n[5/6] Wiring MCP client config(s)...")
    wire = [str(cli_exe), "wire-config", "--exe", str(mcp_exe), "--db", db_path]
    if args.print_only:
        wire.append("--print-only")
    # BOOTSTRAP-1: a real (non print-only) wiring run must abort on failure like
    # install/init now do — otherwise a failed wire still printed "=== done ===".
    rcs = []
    if args.clients:
        for c in args.clients.split(","):
            rcs.append(subprocess.run(wire + ["--client", c.strip()]).returncode)
    else:
        rcs.append(subprocess.run(wire).returncode)
    if not args.print_only and any(rc != 0 for rc in rcs):
        print("ERROR: MCP client wiring failed; setup is incomplete. "
              "Run `memdag doctor` and re-run bootstrap.", file=sys.stderr)
        return 1

    # 7. Wire the Claude Code memory loop (skills + Stop hook + CLAUDE.md block).
    # Soft step: a failure here leaves the MCP server working, so warn but don't abort.
    print("\n[6/6] Wiring the Claude Code memory loop (skills + Stop hook + CLAUDE.md)...")
    wc = [str(cli_exe), "wire-claude",
          "--exe", str(cli_exe),
          "--skills-src", str(repo_dir / "claude" / "skills")]
    if args.print_only:
        wc.append("--print-only")
    if subprocess.run(wc).returncode != 0 and not args.print_only:
        print("  WARNING: memory-loop wiring incomplete (skills/hook/CLAUDE.md). The "
              "MCP server still works; re-run `memdag wire-claude` to finish.")

    print("\n=== done ===")
    print(f"DB: {db_path}")
    print("Restart your MCP client, then ask it to use the memdag tools.")
    print("For a bug report, run:  memdag doctor   (paste the output into a GitHub issue)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
