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

import csv
import io
import itertools
import json
import os
import sys
import threading
import uuid
from pathlib import Path

from memsom.effects import proc as memsom_proc
from memsom.lifecycle import forget

# Detached + no console window: survives the parent, never flashes a terminal.
_DETACHED = 0
if sys.platform == "win32":  # pragma: no branch - platform constant
    _DETACHED = (memsom_proc.DETACHED_PROCESS
                 | memsom_proc.CREATE_NEW_PROCESS_GROUP
                 | memsom_proc.CREATE_NO_WINDOW)
_NO_WINDOW = memsom_proc.CREATE_NO_WINDOW if sys.platform == "win32" else 0

#: One /saveall for one transcript at a time, and a ceiling on the total.
#:
#: NOT a singleton, deliberately. `saveall_hook.py` fires on PreCompact and on
#: SessionEnd(clear) with an explicit session_id and swallows every error to a
#: log, so a blanket "one at a time" refusal would SILENTLY LOSE a legitimate
#: second save whenever two sessions end near each other. Dedupe is by session
#: id — which is exactly the double-fired-DECK-button case — and the cap is what
#: stops an unbounded spawn.
_MAX_CONCURRENT_SAVEALLS = 3

#: In-process only, and that is the honest limit of this guard: two panel
#: processes would each hold their own lock and both spawn. There is one panel.
_START_LOCK = threading.Lock()
_TMP_COUNTER = itertools.count()


def _runs_dir(claude_dir) -> Path:
    return Path(claude_dir) / "episodic" / "saveall"


def _state_path(claude_dir) -> Path:
    """The NEWEST run. Kept for GET /api/saveall/status and the DECK monitor,
    whose SaveallStatus shape is a published contract with the frontend."""
    return _runs_dir(claude_dir) / "latest.json"


def _registry_path(claude_dir) -> Path:
    """EVERY live run. `latest.json` only ever held the newest, which is why a
    second spawn made the first untracked and therefore unreapable.

    If this file ever wedges the dedupe guard — a recycled pid that is also a
    live `claude.exe` — deleting it clears the state completely. Nothing else
    reads it.
    """
    return _runs_dir(claude_dir) / "runs.json"


class AlreadyRunning(RuntimeError):
    """Refusal, not failure — the route turns this into a 409, not a 500."""


def _atomic_write_json(target: Path, obj) -> None:
    """Unique tmp name per call — a fixed one collides with a concurrent writer
    mid-replace, and a torn `latest.json` makes `status()` report
    ``{'exists': false}`` for a run that is very much alive."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(
        f"{target.name}.saveall-{os.getpid()}-{next(_TMP_COUNTER)}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, target)


def _expected_image(argv) -> "str | None":
    """The Windows image name this pid should carry. Windows recycles pids fast
    and a recycled pid read as 'still running' would wedge the dedupe guard
    permanently — the same reason `procman._pid_alive` takes an image.

    MEASURED 2026-07-30: on this box ``shutil.which("claude")`` resolves to a
    real ``claude.EXE``, so the derived name is the image tasklist reports. If
    a profile ever sets ``cli_path`` to a ``.cmd``/``.ps1`` shim the real image
    becomes cmd/node,
    the check stops matching, and the dedupe degrades to "never fires" — i.e.
    back to today's behaviour. That is the safe direction for a robustness
    fix: it can double-spawn, it can never silently refuse a real save.
    """
    if not argv:
        return None
    exe = os.path.basename(str(argv[0])).lower()
    return exe if exe.endswith(".exe") else exe + ".exe"


def _load_registry(claude_dir) -> list:
    p = _registry_path(claude_dir)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def live_runs(claude_dir) -> list:
    """Registry entries whose process is still alive, registry pruned to match."""
    loaded = _load_registry(claude_dir)
    alive = [r for r in loaded
             if _pid_alive(r.get("pid"), _expected_image(r.get("argv")))]
    if len(alive) != len(loaded):
        _atomic_write_json(_registry_path(claude_dir), alive)
    return alive


def kill(claude_dir, run_id: str) -> dict:
    """Reap a tracked run. Detached + a new process group means nothing else
    can: closing the app and restarting the panel both leave it running."""
    for r in _load_registry(claude_dir):
        if r.get("run_id") != run_id:
            continue
        pid = r.get("pid")
        killed = False
        if _pid_alive(pid, _expected_image(r.get("argv"))):
            if sys.platform == "win32":
                try:
                    memsom_proc.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                    capture_output=True, text=True, timeout=10,
                                    creationflags=_NO_WINDOW)
                    killed = True
                except (FileNotFoundError, memsom_proc.TimeoutExpired):
                    pass
            else:
                try:
                    os.kill(int(pid), 15)
                    killed = True
                except OSError:
                    pass
        live_runs(claude_dir)  # prune
        return {"ok": True, "run_id": run_id, "pid": pid, "killed": killed}
    return {"ok": False, "error": f"no tracked saveall run {run_id!r}"}


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

    # Dedupe by SESSION ID under one lock, not by "is anything running".
    # `start()` used to have no identity for a run beyond "the newest one", so
    # it could not tell a duplicate from a legitimate second save and therefore
    # could not refuse the first without refusing the second.
    with _START_LOCK:
        alive = live_runs(claude_dir)
        for r in alive:
            if r.get("session_id") == sid:
                raise AlreadyRunning(
                    f"a /saveall for session {sid} is already running "
                    f"(pid {r.get('pid')}, run {r.get('run_id')})")
        if len(alive) >= _MAX_CONCURRENT_SAVEALLS:
            raise AlreadyRunning(
                f"{len(alive)} /saveall runs already in flight "
                f"(cap {_MAX_CONCURRENT_SAVEALLS}); kill one or wait")

        runs = _runs_dir(claude_dir)
        runs.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        log = runs / f"{run_id}.log"
        argv = [cli_path, "--resume", sid, "--model", model, "--effort", effort,
                "-p", "/saveall"]
        logfh = open(log, "w", encoding="utf-8")
        try:
            # env: the credential denylist (S7 / F-19). This is the site that
            # most needs it — the child is DETACHED, outlives the session, and
            # its stdout already goes to a LOG FILE, so anything it holds is one
            # careless `claude --debug` away from being written down.
            #
            # ANTHROPIC_* is kept because this child IS an Anthropic client: it
            # may legitimately authenticate with them, and dropping them would
            # break /saveall at 2am on a machine that authenticates by env var
            # rather than by the OAuth credentials in ~/.claude. That exception
            # belongs here, at the call site, rather than as a hole in the
            # shared list. MEASURED on this box: both are unset, so today this
            # changes nothing either way.
            #
            # KEEP THIS LINE. The S2/F-43 fix spec respelled this whole call
            # WITHOUT an env override, which would have silently reverted S7/F-19
            # on the one child in this system that outlives everything that could
            # notice. MEASURED 2026-07-30 against the spec text. (Phase 5: `env=
            # child_env(keep=...)` became `keep=...` -- proc.popen() builds the
            # same child_env() call itself; the ANTHROPIC_* exception still lives
            # at this call site, not as a hole in the shared denylist.)
            proc = memsom_proc.popen(
                argv, stdout=logfh, stderr=memsom_proc.STDOUT,
                stdin=memsom_proc.DEVNULL, creationflags=_DETACHED,
                cwd=resume_cwd or os.environ.get("USERPROFILE") or None,
                close_fds=True,
                keep=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                # POSIX: new session (setsid) so the save survives a SessionEnd
                # /clear teardown that would otherwise kill the hook's children.
                start_new_session=(sys.platform != "win32"))
        except (FileNotFoundError, OSError) as exc:
            logfh.close()
            raise RuntimeError(f"failed to launch claude: {exc}") from exc
        # Close OUR copy. Popen has already duplicated the handle into the
        # child, so the detached process keeps writing the log after this —
        # MEASURED 2026-07-30 with a detached child under the same
        # creationflags. Without it the panel, which is long-lived and now
        # spawns up to _MAX_CONCURRENT_SAVEALLS at a time, leaks one open file
        # handle per run forever and keeps the log locked against every reader
        # on Windows.
        logfh.close()
        state = {"run_id": run_id, "pid": proc.pid, "session_id": sid,
                 "log": str(log), "argv": argv, "started": forget.now_iso()}
        # Both files atomically: `latest.json` was written with a plain
        # write_text, so a concurrent status() could read a truncated file and
        # report {"exists": false} for a live run.
        _atomic_write_json(_registry_path(claude_dir), alive + [state])
        _atomic_write_json(_state_path(claude_dir), state)
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


def _pid_alive(pid, image: "str | None" = None) -> bool:
    """True if *pid* is running — and, when *image* is given, running the image
    we expect.

    The image argument is only supplied by the dedupe path (`live_runs`).
    `status()` deliberately does NOT pass it: a wrong `cli_path` making the
    image never match would flip the DECK monitor to "not running" for a live
    save, which is a worse lie than the recycled-pid one it fixes. On the
    dedupe path the same mismatch just means "no duplicate found", which is
    exactly today's behaviour.

    `/FO CSV` rather than the default table, because the image check needs a
    DELIMITED field: the table format pads the image name with spaces, so no
    exact comparison is possible against it. Given a parsed row, checking the
    pid field costs nothing extra and is exact.

    To be precise about what was NOT wrong: the previous `str(pid) in
    out.stdout` was not exploitable. MEASURED 2026-07-30 — `/FI "PID eq N"`
    returns either the one matching row or `INFO: No tasks are running...`,
    never a foreign row, so the memory column could not supply a false match.
    This is a tightening the image check required, not a bug being fixed.
    """
    if not pid:
        return False
    if sys.platform == "win32":
        try:
            out = memsom_proc.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=_NO_WINDOW)
        except (FileNotFoundError, memsom_proc.TimeoutExpired):
            return False
        for row in csv.reader(io.StringIO(out.stdout or "")):
            # ["Image Name", "PID", "Session Name", "Session#", "Mem Usage"].
            # A no-match run prints an INFO line, which parses to one field.
            if len(row) < 2 or row[1].strip() != str(pid):
                continue
            return image is None or row[0].strip().lower() == image
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False
