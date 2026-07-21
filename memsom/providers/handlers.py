"""HTTP-facing handlers for the provider control plane.

Pure functions: ``(registry / session_runner, payload) -> (http_status, body)``,
exactly like ``panel.handle_knob_write`` — so the panel routes stay thin JSON
wrappers and this is directly unit-testable. Every mutating action is wrapped in
the same two-phase audit the knob path uses (intent line gates the action, then
a result line), against the panel's audit log. Prompt bodies are never
audited — only the action metadata.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memsom.interface import telemetry
from memsom.providers.base import ProviderError, run_no_window

# actions and what they require
_MODEL_ACTIONS = {"load", "unload"}
_SERVICE_ACTIONS = {"start", "stop"}


# ---------------------------------------------------------------------------
# GET /api/providers
# ---------------------------------------------------------------------------


def build_providers_payload(registry: dict, *, run=None) -> dict:
    """Fan every probe out concurrently. Serially this cost the SUM of every
    adapter's timeouts — a single down provider on a filtered port burns its
    full HTTP timeout twice (status, then list_models re-probing), and the
    SERVICES/INFERENCE tabs waited on all of it. Concurrent, the payload costs
    the SLOWEST probe instead. Every call is a read-only status/list/metrics
    probe, so there is nothing to serialize for correctness.
    """
    run = run or run_no_window  # no flashing console on the 4s GPU poll
    jobs = {"gpu": lambda: telemetry._read_gpu(run)}
    for pid, adapter in registry.items():
        jobs[(pid, "status")] = lambda a=adapter: a.status().as_dict()
        jobs[(pid, "models")] = lambda a=adapter: [m.as_dict() for m in a.list_models()]
        jobs[(pid, "metrics")] = lambda a=adapter: a.metrics()

    results = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=min(32, len(jobs))) as pool:
            futures = {pool.submit(fn): key for key, fn in jobs.items()}
            for fut, key in futures.items():
                try:
                    results[key] = fut.result()
                except Exception:
                    results[key] = None

    providers = []
    for pid, adapter in registry.items():
        status = results.get((pid, "status"))
        providers.append({
            "id": pid,
            "kind": adapter.kind,
            "label": adapter.label,
            "transport": getattr(adapter, "transport", None),
            "capabilities": adapter.capabilities().as_dict(),
            "status": status if status is not None
                      else {"state": "down", "detail": "status failed"},
            "models": results.get((pid, "models")) or [],
            "metrics": results.get((pid, "metrics")) or {},
        })
    return {"providers": providers, "gpu": results.get("gpu")}


# ---------------------------------------------------------------------------
# POST /api/providers/... (load / unload / start / stop)
# ---------------------------------------------------------------------------


def handle_provider_action(registry: dict, audit_log_path, action: str,
                           payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    pid = payload.get("provider")
    adapter = registry.get(pid) if pid else None
    if adapter is None:
        return 404, {"ok": False, "error": f"unknown provider: {pid!r}"}

    model = payload.get("model")
    if action in _MODEL_ACTIONS and not model:
        return 400, {"ok": False, "error": f"{action} requires 'model'"}

    intent = {"provider": pid, "action": action}
    if model:
        intent["model"] = model
    try:
        _audit(audit_log_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}

    try:
        if action == "load":
            result = adapter.load(model)
        elif action == "unload":
            result = adapter.unload(model)
        elif action == "start":
            result = adapter.start(model) if model else adapter.start()
        elif action == "stop":
            result = adapter.stop()
        else:
            _audit(audit_log_path, {**intent, "result": "refused-unknown-action"})
            return 400, {"ok": False, "error": f"unknown action: {action!r}"}
    except ProviderError as exc:
        _audit(audit_log_path, {**intent, "result": f"failed: {exc}"})
        return 400, {"ok": False, "error": str(exc)}
    except Exception as exc:
        _audit(audit_log_path, {**intent, "result": f"error: {exc}"})
        return 500, {"ok": False, "error": str(exc)}

    _audit(audit_log_path, {**intent, "result": "ok"})
    return 200, {"ok": True, "result": result}


# ---------------------------------------------------------------------------
# GET /api/providers/{id}/vram-estimate
# ---------------------------------------------------------------------------


def handle_vram_estimate(registry: dict, provider_id: str, model: str,
                         ctx: int, kv_type: str = "fp16") -> tuple:
    adapter = registry.get(provider_id)
    if adapter is None:
        return 404, {"ok": False, "error": f"unknown provider: {provider_id!r}"}
    if not model:
        return 400, {"ok": False, "error": "missing 'model'"}
    try:
        est = adapter.estimate_vram(model, ctx, kv_type)
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}
    except Exception as exc:
        return 500, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, "estimate": est}


# ---------------------------------------------------------------------------
# Inference: POST start / GET poll / GET sessions
# ---------------------------------------------------------------------------


def handle_inference_start(registry: dict, session_runner, audit_log_path,
                           payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    pid = payload.get("provider")
    adapter = registry.get(pid) if pid else None
    if adapter is None:
        return 404, {"ok": False, "error": f"unknown provider: {pid!r}"}
    # accept a full conversation (messages) OR a single prompt (wrapped as one
    # user turn) for backward-compat.
    messages = payload.get("messages")
    if messages is None:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return 400, {"ok": False, "error": "missing 'messages' or 'prompt'"}
        messages = [{"role": "user", "content": prompt}]
    if (not isinstance(messages, list) or not messages
            or not all(isinstance(m, dict) and "content" in m for m in messages)):
        return 400, {"ok": False, "error": "'messages' must be a non-empty list of {role, content}"}
    model = payload.get("model") or ""
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        return 400, {"ok": False, "error": "'params' must be an object"}
    session_id = payload.get("session_id")

    # audit metadata only — never the prompt body
    intent = {"provider": pid, "action": "infer", "model": model,
              "params": {k: params.get(k) for k in
                         ("transport", "ctx", "temperature", "max_tokens")
                         if k in params}}
    try:
        _audit(audit_log_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}

    try:
        sid = session_runner.start(adapter, model, messages, params, session_id)
    except ProviderError as exc:
        _audit(audit_log_path, {**intent, "result": f"failed: {exc}"})
        return 400, {"ok": False, "error": str(exc)}
    _audit(audit_log_path, {**intent, "result": "started", "session_id": sid})
    return 200, {"ok": True, "session_id": sid, "cursor": 0}


def handle_inference_read(session_runner, session_id: str, cursor: int) -> tuple:
    try:
        data = session_runner.read_since(session_id, cursor)
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, **data}


def handle_inference_sessions(session_runner) -> tuple:
    # Kernel prompt runs share the sessions dir (krn- prefixed) so their read
    # path is free — but they are NOT chat history; keep them out of the
    # INFERENCE picker.
    sessions = [s for s in session_runner.list_sessions()
                if not str(s.get("session_id", "")).startswith("krn-")]
    return 200, {"ok": True, "sessions": sessions}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe(fn, fallback):
    try:
        return fn()
    except Exception:
        return fallback


def _audit(path, obj: dict, *, gate: bool = False) -> None:
    """Append one fsync'd JSONL line. When gate=True (the intent line) the
    OSError propagates so the caller can refuse the action; otherwise (result
    line) it's swallowed — the action already happened."""
    from memsom.lifecycle import forget  # local import: matches panel's now_iso use
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": forget.now_iso(), **obj}, ensure_ascii=False) + "\n"
        fh = open(p, "a", encoding="utf-8")
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fh.close()
    except OSError:
        if gate:
            raise
