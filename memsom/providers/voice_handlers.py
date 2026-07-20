"""HTTP-facing handlers for the voice tab (STT / chat / TTS).

Pure functions ``(registry / session_runner, payload) -> (http_status, body)``,
exactly like :mod:`memsom.providers.handlers` — so the panel routes stay thin
JSON wrappers and this is directly unit-testable.

The voice flow, end to end:
  1. mic -> POST /api/voice/stt        (Parakeet)  -> {text}
  2. text -> POST /api/voice/chat      (Claude, STREAMED) -> {session_id}
  3. GET  /api/voice/chat?session_id=&cursor=N  cursor-polls the transcript
  4. each finished sentence -> POST /api/voice/tts  (Kokoro) -> {audio_b64}

AUDIT DISCIPLINE (load-bearing): every mutation is two-phase audited (an intent
"pending" line gates the action, a result line follows) against the panel's
audit log — the SAME shape the knob path and provider actions use. Audited fields
are METADATA ONLY: ``{op, model, ctx_len, ms}``. The audio bytes and the text
(prompt or reply) NEVER touch the audit log — same rule as the inference start
handler, which persists param knobs but never the prompt body.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from memsom.providers import handlers as provider_handlers
from memsom.providers.base import ProviderError
from memsom.providers.parakeet import decode_audio_b64


# ---------------------------------------------------------------------------
# adapter lookup — voice adapters are found by KIND, not by a hardcoded id, so a
# profile is free to name them anything (or run two of a kind on diff ports).
# ---------------------------------------------------------------------------


def _find_by_kind(registry: dict, kind: str):
    for adapter in registry.values():
        if getattr(adapter, "kind", None) == kind:
            return adapter
    return None


# ---------------------------------------------------------------------------
# POST /api/voice/stt  — {audio_b64, format} -> {text}
# ---------------------------------------------------------------------------


def handle_voice_stt(registry: dict, audit_log_path, payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    adapter = _find_by_kind(registry, "parakeet")
    if adapter is None:
        return 404, {"ok": False, "error": "no parakeet (STT) provider configured"}

    fmt = payload.get("format") or "webm"
    try:
        audio_bytes = decode_audio_b64(payload.get("audio_b64"))
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}

    model = getattr(adapter, "default_model", "")
    intent = {"op": "stt", "model": model, "ctx_len": len(audio_bytes)}
    try:
        _audit(audit_log_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}

    t0 = _now()
    try:
        result = adapter.transcribe(audio_bytes, fmt)
    except ProviderError as exc:
        _audit(audit_log_path, {**intent, "result": f"failed: {exc}"})
        return 400, {"ok": False, "error": str(exc)}
    except Exception as exc:
        _audit(audit_log_path, {**intent, "result": f"error: {exc}"})
        return 500, {"ok": False, "error": str(exc)}

    ms = round((_now() - t0) * 1000, 1)
    _audit(audit_log_path, {**intent, "ms": ms, "result": "ok"})
    # Contract: res {text:string}. Extra diagnostic fields (model_installed,
    # detail, install) ride along so the UI can show the stub state cleanly.
    body = {"ok": True, "text": result.get("text", ""), "ms": ms}
    for k in ("model_installed", "detail", "install"):
        if k in result:
            body[k] = result[k]
    return 200, body


# ---------------------------------------------------------------------------
# POST /api/voice/chat  — {text, session_id?} -> {session_id}  (STREAMED)
# ---------------------------------------------------------------------------


def handle_voice_chat_start(registry: dict, voice_runner, audit_log_path,
                            payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"ok": False, "error": "missing 'text'"}
    adapter = _find_by_kind(registry, "claude")
    if adapter is None:
        return 404, {"ok": False, "error": "no claude provider configured"}

    session_id = payload.get("session_id")
    model = payload.get("model") or getattr(adapter, "api_model", "") or ""
    messages = [{"role": "user", "content": text}]
    # api transport + streaming: per-sentence TTS needs the token stream. No
    # thinking param — non-thinking is already the default (do not add one).
    params = {"transport": "api", "stream": True}
    if payload.get("temperature") is not None:
        params["temperature"] = payload["temperature"]

    intent = {"op": "chat", "model": model, "ctx_len": len(text)}
    try:
        _audit(audit_log_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}

    try:
        sid = voice_runner.start(adapter, model, messages, params, session_id)
    except ProviderError as exc:
        _audit(audit_log_path, {**intent, "result": f"failed: {exc}"})
        return 400, {"ok": False, "error": str(exc)}

    _audit(audit_log_path, {**intent, "result": "started", "session_id": sid})
    return 200, {"ok": True, "session_id": sid}


# ---------------------------------------------------------------------------
# GET /api/voice/chat?session_id=&cursor=N  -> {events, cursor, status, stats?}
# ---------------------------------------------------------------------------


def handle_voice_chat_read(voice_runner, session_id: str, cursor: int) -> tuple:
    # Reuse the inference read path verbatim — the session file format is
    # identical (start/tok/done/error jsonl), so the frontend cursor-poll gets
    # {events:[{t,text}], cursor, status, stats}. Read-only, no audit.
    return provider_handlers.handle_inference_read(voice_runner, session_id, cursor)


# ---------------------------------------------------------------------------
# POST /api/voice/tts  — {text} -> {audio_b64, format}
# ---------------------------------------------------------------------------


def handle_voice_tts(registry: dict, audit_log_path, payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return 400, {"ok": False, "error": "missing 'text'"}
    adapter = _find_by_kind(registry, "kokoro")
    if adapter is None:
        return 404, {"ok": False, "error": "no kokoro (TTS) provider configured"}

    model = getattr(adapter, "default_model", "")
    intent = {"op": "tts", "model": model, "ctx_len": len(text)}
    try:
        _audit(audit_log_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}

    t0 = _now()
    try:
        result = adapter.synthesize(text)
    except ProviderError as exc:
        _audit(audit_log_path, {**intent, "result": f"failed: {exc}"})
        return 400, {"ok": False, "error": str(exc)}
    except Exception as exc:
        _audit(audit_log_path, {**intent, "result": f"error: {exc}"})
        return 500, {"ok": False, "error": str(exc)}

    ms = round((_now() - t0) * 1000, 1)
    _audit(audit_log_path, {**intent, "ms": ms, "result": "ok"})
    body = {"ok": True, "audio_b64": result.get("audio_b64", ""),
            "format": result.get("format", "wav"), "ms": ms}
    for k in ("model_installed", "detail", "install"):
        if k in result:
            body[k] = result[k]
    return 200, body


# ---------------------------------------------------------------------------
# helpers — mirror handlers.py so audit shape/behavior stays identical
# ---------------------------------------------------------------------------


def _now() -> float:
    import time
    return time.perf_counter()


def _audit(path, obj: dict, *, gate: bool = False) -> None:
    """Append one fsync'd JSONL line. gate=True (intent) propagates OSError so
    the caller can refuse; result lines swallow it (the action already ran).
    Identical to handlers._audit — kept local so the two handler modules don't
    couple through a private import."""
    from memsom.lifecycle import forget
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
