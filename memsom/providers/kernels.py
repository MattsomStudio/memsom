"""Persistent CLI kernels — long-lived Claude/Codex agents as panel objects.

A kernel is NOT a process. It is a named, durable pointer to a Claude Code
(or Codex) session: the CLI's own transcript carries the context, so the
kernel survives the app closing, the panel server restarting, even a reboot.
Each prompt is one fresh headless CLI call that RESUMES the session and
streams its output into the same durable JSONL buffers inference uses
(providers/session.py) — the UI polls /api/inference with the run's id.

The rolling pointer (verified against claude-code issues #10806/#12235):
``claude --resume <sid> -p`` does NOT append under <sid> — every turn mints a
NEW session id, returned in the stream-json ``result`` event. So
``session_ptr`` is a rolling pointer, re-persisted after every prompt; the
stable identity is the kernel_id. Codex differs: ``codex exec resume``
appends to the same thread, so its pointer is stable once captured.

Concurrency (claude-code has NO session lock — concurrent resumes fork
silently and race ~/.claude.json): a per-kernel lock serializes headless
prompts (busy -> 409 at the handler). The interactive-terminal fence lives in
the app UI (sessiond state is invisible to this server by design); the
``refresh_ptr`` scan at the top of every prompt picks up whatever an
interactive ``claude --resume`` terminal did in the meantime.

Prompt bodies ride stdin (never argv — process listers see argv) and are
never audited; audit lines carry action metadata only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

from memsom.providers.base import ProviderError, now
from memsom.providers.session import FileSink, valid_session_id

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

_KERNEL_ID_RE = re.compile(r"^[a-f0-9]{12}$")
# Claude session ids are UUIDs; codex thread ids are UUIDs too. Fence anything
# we splice into argv or a glob.
_PTR_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")

_DEFAULT_TIMEOUT_S = 1800
_PROMPT_CAP_BYTES = 8 * 1024


def _now_ts() -> float:
    return now()


class KernelStore:
    """One atomic JSON file per kernel under <state_dir>/ (peer of the
    inference/ and agents/ dirs — derived from the audit log's parent)."""

    def __init__(self, state_dir) -> None:
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kernel_id: str) -> Path:
        if not _KERNEL_ID_RE.match(kernel_id or ""):
            raise ProviderError("invalid kernel_id")
        return self.dir / f"{kernel_id}.json"

    def create(self, name: str, engine: str, model: Optional[str],
               effort: Optional[str], cwd: str, cli_path: str) -> dict:
        kernel = {
            "kernel_id": uuid.uuid4().hex[:12],
            "name": (name or "kernel")[:64],
            "engine": engine,
            "model": model,
            "effort": effort,
            "cwd": cwd,
            "cli_path": cli_path,
            "session_ptr": None,
            "created": _now_ts(),
            "last_prompt_ts": None,
            "last_run_id": None,
            "status": "idle",
            "prompt_count": 0,
        }
        self._write(kernel)
        return kernel

    def get(self, kernel_id: str) -> dict:
        path = self._path(kernel_id)
        if not path.is_file():
            raise ProviderError(f"no kernel {kernel_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, include_archived: bool = False) -> list:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            if not _KERNEL_ID_RE.match(p.stem):
                continue
            try:
                k = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if k.get("status") == "archived" and not include_archived:
                continue
            out.append(k)
        out.sort(key=lambda k: k.get("created") or 0)
        return out

    def update(self, kernel_id: str, **fields) -> dict:
        """Read-modify-write under the store lock semantics of one writer per
        kernel (the runner's per-kernel lock); atomic tmp+replace on disk."""
        kernel = self.get(kernel_id)
        kernel.update(fields)
        self._write(kernel)
        return kernel

    def _write(self, kernel: dict) -> None:
        path = self._path(kernel["kernel_id"])
        tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        tmp.write_text(json.dumps(kernel, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, path)


class KernelRunner:
    """Spawns per-prompt CLI processes and streams their output into durable
    session files (same dir + format as inference — the read path is free)."""

    def __init__(self, store: KernelStore, sessions_dir, claude_dir) -> None:
        self.store = store
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.claude_dir = Path(claude_dir)
        self._locks: dict = {}
        self._locks_guard = threading.Lock()
        self._procs: dict = {}

    def _lock_for(self, kernel_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(kernel_id, threading.Lock())

    # -- pointer maintenance -------------------------------------------------

    def _project_dir(self, cwd: str) -> Path:
        # Claude Code encodes a project cwd by replacing every non-alphanumeric
        # character with '-' (C:\Users\X -> C--Users-X).
        encoded = re.sub(r"[^A-Za-z0-9-]", "-", cwd or "")
        return self.claude_dir / "projects" / encoded

    def refresh_ptr(self, kernel: dict) -> dict:
        """Roll the pointer to the newest transcript in the kernel's project
        dir when something (an interactive terminal) advanced it. NO-OP when
        the pointer is null — a kernel must never adopt a transcript it didn't
        create (the project dir may hold unrelated manual sessions)."""
        ptr = kernel.get("session_ptr")
        if not ptr or kernel.get("engine") != "claude":
            return kernel
        proj = self._project_dir(kernel.get("cwd") or "")
        current = proj / f"{ptr}.jsonl"
        if not current.is_file():
            return kernel
        try:
            cur_mtime = current.stat().st_mtime
            newest = max(proj.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        except (OSError, ValueError):
            return kernel
        if newest.stem != ptr and newest.stat().st_mtime > cur_mtime:
            kernel = self.store.update(kernel["kernel_id"], session_ptr=newest.stem)
        return kernel

    # -- prompting -----------------------------------------------------------

    def prompt(self, kernel_id: str, prompt_text: str) -> str:
        """Fire one prompt. Returns the run session id immediately (poll it
        via the inference read path). Raises ProviderError on busy/invalid."""
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ProviderError("prompt must be a non-empty string")
        if len(prompt_text.encode("utf-8")) > _PROMPT_CAP_BYTES:
            raise ProviderError("prompt exceeds 8KB cap")
        kernel = self.store.get(kernel_id)
        if kernel.get("status") == "archived":
            raise ProviderError("kernel is archived")

        lock = self._lock_for(kernel_id)
        if not lock.acquire(blocking=False):
            raise ProviderError("kernel busy: a prompt is already running")
        try:
            kernel = self.refresh_ptr(kernel)
            run_id = f"krn-{kernel_id}-{uuid.uuid4().hex[:8]}"
            assert valid_session_id(run_id)
            path = self.sessions_dir / f"{run_id}.jsonl"

            # start line, fsync'd, before the thread (session.py's ordering) —
            # a poll that races the thread still finds a well-formed file.
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "t": "start", "provider": f"kernel-{kernel['engine']}",
                    "model": kernel.get("model"), "turns": 1,
                    "params": {"kernel_id": kernel_id}, "ts": _now_ts(),
                }, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            self.store.update(kernel_id, status="busy", last_run_id=run_id,
                              last_prompt_ts=_now_ts())
            thread = threading.Thread(
                target=self._run, args=(kernel, prompt_text, path, lock),
                name=f"kernel-{kernel_id}", daemon=True)
            thread.start()
            return run_id
        except Exception:
            lock.release()
            raise

    def _argv(self, kernel: dict) -> list:
        cli = kernel.get("cli_path") or kernel["engine"]
        ptr = kernel.get("session_ptr")
        if ptr and not _PTR_RE.match(ptr):
            raise ProviderError("corrupt session pointer")
        if kernel["engine"] == "claude":
            argv = [cli, "-p", "--output-format", "stream-json", "--verbose"]
            if kernel.get("model"):
                argv += ["--model", kernel["model"]]
            if kernel.get("effort"):
                argv += ["--effort", kernel["effort"]]
            if ptr:
                argv += ["--resume", ptr]
            return argv
        if kernel["engine"] == "codex":
            argv = [cli, "exec"]
            if ptr:
                argv = [cli, "exec", "resume", ptr]
            if kernel.get("model"):
                argv += ["-m", kernel["model"]]
            return argv
        raise ProviderError(f"unknown engine {kernel['engine']}")

    def _run(self, kernel: dict, prompt_text: str, path: Path,
             lock: threading.Lock) -> None:
        kernel_id = kernel["kernel_id"]
        sink = FileSink(path)
        killed = threading.Event()
        proc = None
        try:
            argv = self._argv(kernel)
            creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=kernel.get("cwd") or None,
                creationflags=creationflags, text=True, encoding="utf-8",
                errors="replace")
            self._procs[kernel_id] = (proc, killed)

            def _reaper() -> None:
                killed.set()
                _kill_proc_tree(proc)

            timer = threading.Timer(_DEFAULT_TIMEOUT_S, _reaper)
            timer.daemon = True
            timer.start()
            try:
                assert proc.stdin is not None
                proc.stdin.write(prompt_text)
                proc.stdin.close()
                if kernel["engine"] == "claude":
                    self._stream_claude(kernel_id, proc, sink)
                else:
                    self._stream_codex(kernel_id, proc, sink)
            finally:
                timer.cancel()

            rc = proc.wait()
            if not sink._fh.closed:  # no result event reached the sink
                if killed.is_set():
                    sink.error("killed (timeout or user)")
                elif rc != 0:
                    err = (proc.stderr.read() or "").strip() if proc.stderr else ""
                    sink.error(f"cli exited {rc}: {err[:500]}")
                else:
                    sink.done({"tokens": sink.count})
        except ProviderError as exc:
            if not sink._fh.closed:
                sink.error(str(exc))
        except Exception as exc:  # never let the thread die silently
            if not sink._fh.closed:
                sink.error(f"internal error: {exc}")
        finally:
            self._procs.pop(kernel_id, None)
            try:
                self.store.update(kernel_id, status="idle",
                                  prompt_count=self.store.get(kernel_id).get(
                                      "prompt_count", 0) + 1)
            except ProviderError:
                pass
            lock.release()

    def _stream_claude(self, kernel_id: str, proc, sink: FileSink) -> None:
        """Map stream-json lines to sink events. Every event carries
        session_id (hooks fire before init on hook-heavy setups, so we take
        the FIRST id seen, whatever the event type); the result event's id is
        the ROLLED pointer — persist it the moment it arrives."""
        seen_init = False
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            sid = event.get("session_id")
            if sid and not seen_init and _PTR_RE.match(str(sid)):
                seen_init = True
                # Persist immediately: a crash mid-run must not strand the
                # minted session.
                self.store.update(kernel_id, session_ptr=str(sid))
            etype = event.get("type")
            if etype == "assistant":
                for item in (event.get("message") or {}).get("content") or []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and item.get("text"):
                        sink.token(item["text"])
                    elif item.get("type") == "tool_use":
                        sink.token(f"\n[tool: {item.get('name', '?')}]\n")
            elif etype == "result":
                rsid = event.get("session_id")
                if rsid and _PTR_RE.match(str(rsid)):
                    self.store.update(kernel_id, session_ptr=str(rsid))
                if event.get("is_error"):
                    sink.error(str(event.get("result") or "kernel run failed"))
                else:
                    sink.done({
                        "cost_usd": event.get("total_cost_usd"),
                        "num_turns": event.get("num_turns"),
                        "tokens": sink.count,
                        "usage": {k: v for k, v in (event.get("usage") or {}).items()
                                  if isinstance(v, (int, float))},
                    })
            # system / rate_limit_event / user(tool results) etc. — skipped.

    def _stream_codex(self, kernel_id: str, proc, sink: FileSink) -> None:
        """codex exec streams plain text; capture a UUID-ish session/thread id
        if one appears in a header line, else rely on `exec resume` with the
        captured pointer from the first run that revealed one."""
        assert proc.stdout is not None
        uuid_re = re.compile(
            r"session[_ id:]*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            re.IGNORECASE)
        for line in proc.stdout:
            m = uuid_re.search(line)
            if m:
                self.store.update(kernel_id, session_ptr=m.group(1))
            sink.token(line)

    # -- kill / reconcile ----------------------------------------------------

    def kill(self, kernel_id: str) -> bool:
        entry = self._procs.get(kernel_id)
        if not entry:
            return False
        proc, killed = entry
        killed.set()
        _kill_proc_tree(proc)
        return True

    def reconcile_on_boot(self) -> int:
        """Stamp kernels left 'busy' by a server restart (agents.py pattern):
        append an error line to their last run feed, reset to idle."""
        fixed = 0
        for kernel in self.store.list(include_archived=True):
            if kernel.get("status") != "busy":
                continue
            run_id = kernel.get("last_run_id")
            if run_id and valid_session_id(run_id):
                path = self.sessions_dir / f"{run_id}.jsonl"
                try:
                    if path.is_file():
                        status_line = json.dumps({
                            "t": "error",
                            "error": "interrupted: panel server restarted"})
                        text = path.read_text(encoding="utf-8",
                                              errors="replace")
                        if '"t": "done"' not in text and '"t":"done"' not in text \
                                and '"t": "error"' not in text and '"t":"error"' not in text:
                            with open(path, "a", encoding="utf-8") as fh:
                                fh.write(status_line + "\n")
                except OSError:
                    pass
            self.store.update(kernel["kernel_id"], status="idle")
            fixed += 1
        return fixed


def _kill_proc_tree(proc) -> None:
    """Kill the CLI and its subprocess tree (claude spawns children)."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                creationflags=_CREATE_NO_WINDOW)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
