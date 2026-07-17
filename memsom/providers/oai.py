"""OpenAI-compatible HTTP helpers, shared by the vLLM and llama.cpp adapters
(and available to the Codex API transport).

Both vLLM and llama.cpp's server speak the OpenAI ``/v1/chat/completions`` +
``/v1/models`` shape, so the streaming SSE parse and the model list live once
here rather than in each adapter. stdlib ``urllib`` only, matching the repo.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from memsom.providers.base import ProviderError, Sink


def list_models(base: str, timeout: float = 5, headers: dict | None = None) -> list:
    """GET /v1/models -> list of model id strings."""
    try:
        req = urllib.request.Request(base + "/v1/models", headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"/v1/models failed: {exc}") from exc
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


def chat_stream(base: str, model: str, messages: list, params: dict, sink: Sink,
                *, headers: dict | None = None) -> dict:
    """POST /v1/chat/completions with stream=true; push token deltas to *sink*.

    *messages* is the OpenAI-shaped conversation list ([{role, content}, ...]),
    so multi-turn chat carries prior turns. Returns a stats dict (usage counters
    when the server reports them). Parses the SSE ``data: {...}`` framing;
    ``data: [DONE]`` ends it."""
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    for k in ("temperature", "top_p", "max_tokens"):
        if params.get(k) is not None:
            body[k] = params[k]
    hdr = {"Content-Type": "application/json"}
    hdr.update(headers or {})

    stats: dict = {}
    try:
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"), headers=hdr)
        with urllib.request.urlopen(req, timeout=params.get("timeout", 600)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        sink.token(piece)
                if chunk.get("usage"):
                    u = chunk["usage"]
                    stats["prompt_tokens"] = u.get("prompt_tokens")
                    stats["eval_count"] = u.get("completion_tokens")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            TimeoutError) as exc:
        raise ProviderError(f"chat/completions failed: {exc}") from exc
    return stats
