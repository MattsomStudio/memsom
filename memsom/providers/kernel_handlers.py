"""HTTP-facing handlers for kernels — same texture as agent_handlers.py:
pure ``(deps, payload) -> (status, body)`` functions, thin panel routes,
two-phase audit (gated intent line) on every mutation. Prompt bodies are
NEVER audited — kernel/engine/run metadata only.
"""

from __future__ import annotations

import os
import shutil

from memsom.providers.base import ProviderError
from memsom.providers.handlers import _audit
from memsom.providers.kernels import KernelRunner, KernelStore

_ENGINES = ("claude", "codex")


def _busy(runner: KernelRunner, kernel: dict) -> bool:
    lock = runner._lock_for(kernel["kernel_id"])
    if lock.acquire(blocking=False):
        lock.release()
        return False
    return True


def handle_kernels_list(store: KernelStore, runner: KernelRunner,
                        include_archived: bool = False) -> tuple:
    kernels = store.list(include_archived=include_archived)
    for k in kernels:
        k["busy"] = _busy(runner, k)
    return 200, {"ok": True, "kernels": kernels}


def handle_kernel_create(store: KernelStore, profile: dict, audit_path,
                         payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    name = (payload.get("name") or "").strip()
    engine = payload.get("engine") or "claude"
    if engine not in _ENGINES:
        return 400, {"ok": False, "error": f"engine must be one of {_ENGINES}"}
    cwd = payload.get("cwd") or os.environ.get("USERPROFILE") \
        or os.environ.get("HOME") or "."
    if not os.path.isdir(cwd):
        return 400, {"ok": False, "error": f"cwd is not a directory: {cwd}"}

    # cli path from the provider spec (profile) — the terminal bridge needs
    # it. The live profile's `providers` block is a LIST of spec dicts keyed
    # by "id" (build_registry's input shape); tests/other profiles may use a
    # dict — accept both.
    raw = profile.get("providers") or {}
    if isinstance(raw, list):
        spec = next((s for s in raw
                     if isinstance(s, dict) and s.get("id") == engine), {})
    else:
        spec = raw.get(engine) or {}
    cli_path = spec.get("cli_path") or engine
    if shutil.which(cli_path) is None and not os.path.isfile(cli_path):
        return 501, {"ok": False,
                     "error": f"{engine} CLI not found ({cli_path})"}

    intent = {"action": "kernel-create", "engine": engine, "name": name}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    kernel = store.create(name or f"{engine}-kernel", engine,
                          payload.get("model"), payload.get("effort"),
                          cwd, cli_path)
    _audit(audit_path, {**intent, "result": "ok",
                        "kernel_id": kernel["kernel_id"]})
    return 201, {"ok": True, "kernel": kernel}


def handle_kernel_prompt(store: KernelStore, runner: KernelRunner, audit_path,
                         kernel_id: str, payload: dict) -> tuple:
    prompt = (payload or {}).get("prompt")
    intent = {"action": "kernel-prompt", "kernel_id": kernel_id}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        run_id = runner.prompt(kernel_id or "", prompt)
    except ProviderError as exc:
        msg = str(exc)
        _audit(audit_path, {**intent, "result": f"refused: {msg}"})
        if "busy" in msg:
            return 409, {"ok": False, "error": msg}
        if "no kernel" in msg:
            return 404, {"ok": False, "error": msg}
        return 400, {"ok": False, "error": msg}
    _audit(audit_path, {**intent, "result": "started", "run_id": run_id})
    return 202, {"ok": True, "kernel_id": kernel_id, "run_session_id": run_id}


def handle_kernel_kill(store: KernelStore, runner: KernelRunner, audit_path,
                       kernel_id: str) -> tuple:
    intent = {"action": "kernel-kill", "kernel_id": kernel_id}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        store.get(kernel_id or "")
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"refused: {exc}"})
        return 404, {"ok": False, "error": str(exc)}
    killed = runner.kill(kernel_id)
    _audit(audit_path, {**intent,
                        "result": "ok" if killed else "ok-nothing-running"})
    return 200, {"ok": True, "killed": killed}


def handle_kernel_archive(store: KernelStore, audit_path,
                          kernel_id: str) -> tuple:
    intent = {"action": "kernel-archive", "kernel_id": kernel_id}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        store.update(kernel_id or "", status="archived")
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"refused: {exc}"})
        return 404, {"ok": False, "error": str(exc)}
    _audit(audit_path, {**intent, "result": "ok"})
    return 200, {"ok": True}
