"""Claude adapter — dual transport (API or subscription CLI).

Two ways to reach Claude, chosen per call via ``params['transport']`` (default
from the profile):

* ``api`` — Anthropic Messages API over ``urllib``. Metered, pay-per-token. Key
  is read from the environment by NAME (``api_key_env``); it never appears in a
  request we log or a response we return.
* ``cli-subscription`` — drive the installed ``claude`` CLI headlessly
  (``claude -p --output-format json``, prompt on stdin). Runs on the flat-rate
  Claude Code subscription, no metered charge. Phase 1 is one fresh headless
  call per prompt (stateless); the persistent-kernel version is Phase 3.

No VRAM, no load/unload, no start/stop — it's a cloud/agent backend. The
per-prompt CLI subprocess is spawned by the generation thread, so it inherits
the same disk-buffer durability as every other adapter.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

from memsom.providers.base import (
    Capabilities,
    ModelInfo,
    Provider,
    ProviderError,
    ProviderStatus,
    Sink,
    run_no_window,
)

_API_BASE = "https://api.anthropic.com"
_API_VERSION = "2023-06-01"


class ClaudeAdapter(Provider):
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        self.transport = spec.get("transport", "cli-subscription")
        self.api_key_env = spec.get("api_key_env", "ANTHROPIC_API_KEY")
        self.api_base = spec.get("api_base", _API_BASE)
        self.cli_path = spec.get("cli_path", "claude")
        self.default_model = spec.get("model")
        self.api_model = spec.get("api_model", "claude-sonnet-4-5")
        self.max_tokens = spec.get("max_tokens", 4096)

    # ---- capability + status ----

    def capabilities(self) -> Capabilities:
        return Capabilities(can_start=False, can_load=False, has_vram=False,
                            can_estimate=False,
                            transports=("api", "cli-subscription"))

    def _has_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def status(self) -> ProviderStatus:
        if self.transport == "api":
            if not self._has_key():
                return ProviderStatus("unauthed",
                                      detail=f"${self.api_key_env} not set")
            try:
                self._api_get("/v1/models", timeout=4)
            except ProviderError as exc:
                if "401" in str(exc) or "403" in str(exc):
                    return ProviderStatus("unauthed", detail="key rejected")
                return ProviderStatus("down", detail=str(exc))
            return ProviderStatus("up", detail="api key ok")
        # cli-subscription
        if _which(self.cli_path):
            return ProviderStatus("up", detail=f"{self.cli_path} on PATH")
        return ProviderStatus("down", detail=f"{self.cli_path} not found")

    # ---- inventory ----

    def list_models(self) -> list[ModelInfo]:
        if self.transport == "api" and self._has_key():
            try:
                data = self._api_get("/v1/models", timeout=4)
                return [ModelInfo(name=m.get("id")) for m in data.get("data", [])
                        if m.get("id")]
            except ProviderError:
                pass
        name = self.default_model or self.api_model
        return [ModelInfo(name=name, meta={"note": "subscription CLI default"})]

    # ---- inference ----

    def infer(self, model: str, messages: list, params: dict, sink: Sink) -> dict:
        transport = params.get("transport") or self.transport
        if transport == "api":
            return self._infer_api(model, messages, params, sink)
        return self._infer_cli(model, messages, params, sink)

    def _infer_cli(self, model, messages, params, sink) -> dict:
        if params.get("tools"):
            raise ProviderError("custom tools not supported over cli transport")
        argv = [self.cli_path, "-p", "--output-format", "json"]
        if model:
            argv += ["--model", model]
        try:
            proc = run_no_window(
                argv, input=_flatten(messages), capture_output=True, text=True,
                timeout=params.get("timeout", 600))
        except FileNotFoundError as exc:
            raise ProviderError(f"claude CLI not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError("claude CLI timed out") from exc
        if proc.returncode != 0:
            raise ProviderError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        text, stats = _parse_cli_json(proc.stdout)
        sink.token(text)
        stats["transport"] = "cli-subscription"
        return stats

    def _infer_api(self, model, messages, params, sink) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"${self.api_key_env} not set")
        tools = params.get("tools")
        if tools:
            system, convo = _to_anthropic_tool_convo(messages)
        else:
            system, convo = _split_system(messages)
        body = {
            "model": model or self.api_model,
            "max_tokens": int(params.get("max_tokens", self.max_tokens)),
            "messages": convo,
        }
        if tools:
            body["tools"] = _tools_to_anthropic(tools)
        if system:
            body["system"] = system
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        try:
            req = urllib.request.Request(
                self.api_base + "/v1/messages",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "x-api-key": key, "anthropic-version": _API_VERSION})
            with urllib.request.urlopen(
                    req, timeout=params.get("timeout", 600)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"anthropic API {exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                json.JSONDecodeError) as exc:
            raise ProviderError(f"anthropic API failed: {exc}") from exc
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        if text or not tools:
            sink.token(text)
        usage = data.get("usage") or {}
        stats = {"transport": "api",
                 "prompt_tokens": usage.get("input_tokens"),
                 "eval_count": usage.get("output_tokens")}
        tool_calls = [{"id": b.get("id", ""), "name": b.get("name", ""),
                       "arguments": b.get("input") or {}}
                      for b in data.get("content", [])
                      if b.get("type") == "tool_use"]
        if tool_calls:
            stats["tool_calls"] = tool_calls
        return stats

    # ---- HTTP plumbing (api transport) ----

    def _api_get(self, path: str, timeout: float = 5) -> dict:
        key = os.environ.get(self.api_key_env)
        try:
            req = urllib.request.Request(
                self.api_base + path,
                headers={"x-api-key": key or "", "anthropic-version": _API_VERSION})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"{exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                json.JSONDecodeError) as exc:
            raise ProviderError(str(exc)) from exc


def _parse_cli_json(stdout: str) -> tuple:
    """Claude Code -p --output-format json emits a result object. Pull the text
    and any cost/usage; fall back to raw stdout if the shape drifts."""
    stdout = (stdout or "").strip()
    if not stdout:
        return "", {}
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, {}
    if isinstance(obj, dict):
        text = obj.get("result") or obj.get("text") or ""
        stats = {}
        if obj.get("total_cost_usd") is not None:
            stats["cost_usd"] = obj["total_cost_usd"]
        usage = obj.get("usage") or {}
        if usage.get("output_tokens"):
            stats["eval_count"] = usage["output_tokens"]
        if usage.get("input_tokens"):
            stats["prompt_tokens"] = usage["input_tokens"]
        return text, stats
    return stdout, {}


def _which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


def _split_system(messages: list) -> tuple:
    """Anthropic wants the system prompt as a top-level param, not a message.
    Returns (system_text, non_system_messages)."""
    system_parts, convo = [], []
    for m in messages or []:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            convo.append({"role": m.get("role", "user"),
                          "content": m.get("content", "")})
    if not convo:
        convo = [{"role": "user", "content": ""}]
    return ("\n\n".join(p for p in system_parts if p), convo)


def _tools_to_anthropic(tools: list) -> list:
    """Canonical OpenAI-wire tool definitions → Anthropic's tool shape
    (``parameters`` becomes ``input_schema``)."""
    out = []
    for t in tools or []:
        fn = t.get("function") or {}
        out.append({"name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}}})
    return out


def _to_anthropic_tool_convo(messages: list) -> tuple:
    """Canonical messages → (system_text, Anthropic content-block messages).

    Same system-vs-conversation split as :func:`_split_system`, but block-aware:
    assistant tool_calls become ``tool_use`` blocks (after a text block when the
    turn had text), and ``role:"tool"`` results become user-role
    ``tool_result`` blocks. Plain text messages keep string content."""
    system_parts, convo = [], []
    for m in messages or []:
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content", ""))
        elif role == "tool":
            convo.append({"role": "user", "content": [
                {"type": "tool_result",
                 "tool_use_id": m.get("tool_call_id", ""),
                 "content": m.get("content", "")}]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": tc.get("id", ""),
                               "name": tc.get("name", ""),
                               "input": tc.get("arguments") or {}})
            convo.append({"role": "assistant", "content": blocks})
        else:
            convo.append({"role": m.get("role", "user"),
                          "content": m.get("content", "")})
    if not convo:
        convo = [{"role": "user", "content": ""}]
    return ("\n\n".join(p for p in system_parts if p), convo)


def _flatten(messages: list) -> str:
    """Flatten a conversation into a single prompt for the headless CLI (which
    takes one prompt on stdin). Role-labels each turn so the model keeps context."""
    lines = []
    for m in messages or []:
        role = (m.get("role") or "user").capitalize()
        lines.append(f"{role}: {m.get('content', '')}")
    lines.append("Assistant:")
    return "\n\n".join(lines)
