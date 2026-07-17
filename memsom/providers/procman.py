"""Detached process manager for local model servers (llama.cpp, vLLM).

Starting a model server from the panel has one hard requirement: it must
**outlive the panel**. If llama-server were a child of the panel process it'd
die on a panel restart. So we spawn it detached and record its PID in a small
JSON registry on disk; ``stop`` reads the PID back and kills the tree. This is
the same "survives the app + the panel" discipline the plan calls for, done with
OS-level detach rather than a Job Object (which does the opposite — kill on
close).

Windows: ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``. WSL2-hosted servers
are just a command prefixed with ``wsl.exe`` — the PID we track is the wsl.exe
launcher on the Windows side; stop kills it and WSL reaps the child.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from memsom.providers.base import ProviderError, now, run_no_window

_DETACHED = 0
if sys.platform == "win32":  # pragma: no branch - platform constant
    _DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | \
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


class ProcessManager:
    def __init__(self, registry_path) -> None:
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def record(self, key: str) -> Optional[dict]:
        return self._load().get(key)

    def is_running(self, key: str) -> bool:
        rec = self.record(key)
        if not rec or not rec.get("pid"):
            return False
        return _pid_alive(rec["pid"])

    def start(self, key: str, argv: list, *, port: Optional[int] = None,
              model: Optional[str] = None, cwd: Optional[str] = None) -> dict:
        if self.is_running(key):
            raise ProviderError(f"{key} is already running")
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL, creationflags=_DETACHED, cwd=cwd,
                close_fds=True)
        except (FileNotFoundError, OSError) as exc:
            raise ProviderError(f"failed to spawn {argv[0]!r}: {exc}") from exc
        data = self._load()
        data[key] = {"pid": proc.pid, "port": port, "model": model,
                     "argv": argv, "started_ts": now()}
        self._save(data)
        return {"ok": True, "pid": proc.pid, "model": model, "port": port}

    def stop(self, key: str) -> dict:
        rec = self.record(key)
        if not rec or not rec.get("pid"):
            raise ProviderError(f"{key} is not running (no tracked pid)")
        pid = rec["pid"]
        _kill_tree(pid)
        data = self._load()
        data.pop(key, None)
        self._save(data)
        return {"ok": True, "stopped_pid": pid}


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            out = run_no_window(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5)
            return str(pid) in out.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        try:
            run_no_window(["taskkill", "/PID", str(pid), "/T", "/F"],
                          capture_output=True, text=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass
