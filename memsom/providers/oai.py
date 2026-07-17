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


def messages_to_openai(messages: list[dict]) -> list[dict]:
    """Translate the canonical internal message shape to the OpenAI wire shape.

    Canonical assistant tool_calls carry parsed-dict arguments; OpenAI wants
    them as a JSON string inside a ``{"type": "function", "function": ...}``
    envelope. Canonical ``role:"tool"`` messages carry a ``name`` field OpenAI
    doesn't take. Plain system/user/assistant text messages pass through."""
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [
                    {"id": tc.get("id", ""), "type": "function",
                     "function": {"name": tc.get("name", ""),
                                  "arguments": json.dumps(
                                      tc.get("arguments") or {})}}
                    for tc in m["tool_calls"]],
            })
        elif role == "tool":
            out.append({"role": "tool",
                        "tool_call_id": m.get("tool_call_id", ""),
                        "content": m.get("content", "")})
        else:
            out.append(m)
    return out


def _parse_arguments(raw) -> dict:
    """Function-call arguments as a dict, whatever the server sent. A malformed
    JSON string becomes ``{"_raw": <string>}`` rather than raising — the runner
    decides what to do with a garbled call, not the transport."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def chat_once(base: str, model: str, messages: list, params: dict, sink: Sink,
              *, headers: dict | None = None) -> dict:
    """POST /v1/chat/completions non-streaming — the tool-calling path.

    When ``params['tools']`` is present the request must resolve in one JSON
    response (a tool call has no meaningful token stream), so the final text —
    if any — goes to *sink* as ONE token, and tool calls come back canonically
    in ``stats["tool_calls"]`` as ``[{"id", "name", "arguments": dict}]``."""
    body = {
        "model": model,
        "messages": messages_to_openai(messages),
        "stream": False,
    }
    for k in ("temperature", "top_p", "max_tokens"):
        if params.get(k) is not None:
            body[k] = params[k]
    if params.get("tools"):
        body["tools"] = params["tools"]
        if params.get("tool_choice") is not None:
            body["tool_choice"] = params["tool_choice"]
    hdr = {"Content-Type": "application/json"}
    hdr.update(headers or {})
    try:
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"), headers=hdr)
        with urllib.request.urlopen(req, timeout=params.get("timeout", 600)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"chat/completions failed: {exc}") from exc

    message = (data.get("choices") or [{}])[0].get("message") or {}
    content = message.get("content")
    if content:
        sink.token(content)
    stats: dict = {}
    u = data.get("usage") or {}
    if u:
        stats["prompt_tokens"] = u.get("prompt_tokens")
        stats["eval_count"] = u.get("completion_tokens")
    tool_calls = []
    for i, tc in enumerate(message.get("tool_calls") or [], 1):
        fn = tc.get("function") or {}
        tool_calls.append({"id": tc.get("id") or f"tc_{i}",
                           "name": fn.get("name", ""),
                           "arguments": _parse_arguments(fn.get("arguments"))})
    if tool_calls:
        stats["tool_calls"] = tool_calls
    return stats


def chat_stream(base: str, model: str, messages: list, params: dict, sink: Sink,
                *, headers: dict | None = None) -> dict:
    """POST /v1/chat/completions with stream=true; push token deltas to *sink*.

    *messages* is the OpenAI-shaped conversation list ([{role, content}, ...]),
    so multi-turn chat carries prior turns. Returns a stats dict (usage counters
    when the server reports them). Parses the SSE ``data: {...}`` framing;
    ``data: [DONE]`` ends it."""
    body = {
        "model": model,
        "messages": messages_to_openai(messages),
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
