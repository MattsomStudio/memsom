"""Detached headless /saveall runner for the DECK button.

Normally /saveall runs inside a live Claude session. This runs it OUT OF BAND:
it finds the most-recently-active session transcript, then spawns

    claude --resume <sid> --model claude-sonnet-5 --effort high -p /saveall

fully DETACHED, logging to a file. Because the transcript is captured at spawn
and the process is detached from the panel + the source chat, the save survives
the app closing AND a /clear of the chat you were working in. The DECK polls
status()/log for a live monitor.

The panel server (which spawns it) is never killed by the desktop app, and the
child is detached anyway — so "close the app / clear the chat, it still saves"
holds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from memsom.lifecycle import forget

# Detached + no console window: survives the parent, never flashes a terminal.
_DETACHED = 0
if sys.platform == "win32":  # pragma: no branch - platform constant
    _DETACHED = (getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _runs_dir(claude_dir) -> Path:
    return Path(claude_dir) / "episodic" / "saveall"


def _state_path(claude_dir) -> Path:
    return _runs_dir(claude_dir) / "latest.json"


def find_latest_session(claude_dir) -> "dict | None":
    """Newest ``*.jsonl`` across all project folders under
    ``<claude_dir>/projects`` — the chat you were most recently in."""
    proj = Path(claude_dir) / "projects"
    newest = None
    for p in proj.glob("*/*.jsonl"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if newest is None or m > newest[1]:
            newest = (p, m)
    if newest is None:
        return None
    return {"session_id": newest[0].stem, "path": str(newest[0]), "mtime": newest[1]}


def start(claude_dir, *, cli_path: str = "claude", model: str = "claude-sonnet-5",
          effort: str = "high", session_id: "str | None" = None,
          resume_cwd: "str | None" = None) -> dict:
    # Explicit session_id (hook-driven auto-save of a cloned/original transcript)
    # takes precedence; otherwise fall back to newest session (DECK button).
    if session_id:
        sid = session_id
    else:
        latest = find_latest_session(claude_dir)
        if not latest:
            raise RuntimeError("no session transcript found to save")
        sid = latest["session_id"]
    runs = _runs_dir(claude_dir)
    runs.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    log = runs / f"{run_id}.log"
    argv = [cli_path, "--resume", sid, "--model", model, "--effort", effort,
            "-p", "/saveall"]
    logfh = open(log, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            argv, stdout=logfh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=_DETACHED,
            cwd=resume_cwd or os.environ.get("USERPROFILE") or None,
            close_fds=True,
            # POSIX: new session (setsid) so the save survives a SessionEnd
            # /clear teardown that would otherwise kill the hook's children.
            start_new_session=(sys.platform != "win32"))
    except (FileNotFoundError, OSError) as exc:
        logfh.close()
        raise RuntimeError(f"failed to launch claude: {exc}") from exc
    state = {"run_id": run_id, "pid": proc.pid, "session_id": sid,
             "log": str(log), "argv": argv, "started": forget.now_iso()}
    _state_path(claude_dir).write_text(json.dumps(state, indent=1), encoding="utf-8")
    return {"ok": True, "run_id": run_id, "session_id": sid, "pid": proc.pid}


def status(claude_dir, *, tail_bytes: int = 8000) -> dict:
    sp = _state_path(claude_dir)
    if not sp.is_file():
        return {"exists": False}
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"exists": False}
    running = _pid_alive(st.get("pid"))
    log_text = ""
    try:
        p = Path(st.get("log", ""))
        if p.is_file():
            log_text = p.read_text(encoding="utf-8", errors="replace")[-tail_bytes:]
    except OSError:
        pass
    return {"exists": True, "running": running, "session_id": st.get("session_id"),
            "started": st.get("started"), "run_id": st.get("run_id"),
            "log": log_text}


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=5,
                                 creationflags=_NO_WINDOW)
            return str(pid) in out.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False
