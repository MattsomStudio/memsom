"""Ollama adapter — the reference backend.

Ollama is already installed and running on the PC, so it's the one that proves
the whole interface. Talks to it over stdlib ``urllib`` exactly like
``memsom.distill.llm`` does (127.0.0.1, not localhost, to dodge the Windows
dual-stack stall when it's down).

Endpoints used: ``/api/tags`` (inventory), ``/api/ps`` (what's resident in VRAM
+ its size), ``/api/show`` (architecture metadata for the estimator),
``/api/generate`` (load via empty prompt + keep_alive, unload via keep_alive:0,
and streaming inference).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from memsom.providers.base import (
    Capabilities,
    ModelInfo,
    Provider,
    ProviderError,
    ProviderStatus,
    Sink,
    now,
)
from memsom.providers import vram


class OllamaAdapter(Provider):
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        host = spec.get("host", "127.0.0.1")
        port = spec.get("port", 11434)
        self.base = spec.get("base_url") or f"http://{host}:{port}"
        # keep_alive string used by load(); -1 = keep forever, "0" = unload.
        self.keep_alive = spec.get("keep_alive", "60m")

    # ---- capability + status ----

    def capabilities(self) -> Capabilities:
        # can_start is False: the Ollama daemon is a system-level service; the
        # panel manages model residency (load/unload), not the daemon process.
        return Capabilities(can_start=False, can_load=True, has_vram=True,
                            can_estimate=True, transports=("native",))

    def status(self) -> ProviderStatus:
        t0 = now()
        try:
            self._get("/api/tags", timeout=self.spec.get("status_timeout_s", 2))
        except ProviderError as exc:
            return ProviderStatus("down", detail=str(exc))
        return ProviderStatus("up", ms=round((now() - t0) * 1000, 1))

    # ---- inventory ----

    def list_models(self) -> list[ModelInfo]:
        tags = self._get("/api/tags").get("models", [])
        resident = self._resident_names()
        out = []
        for m in tags:
            details = m.get("details") or {}
            out.append(ModelInfo(
                name=m.get("name") or m.get("model"),
                size_bytes=m.get("size"),
                quant=details.get("quantization_level"),
                ctx_max=details.get("context_length"),
                loaded=(m.get("name") in resident or m.get("model") in resident),
                meta={
                    "n_params": details.get("parameter_size"),
                    "quant": details.get("quantization_level"),
                    "embedding_length": details.get("embedding_length"),
                    "family": details.get("family"),
                },
            ))
        return out

    def _resident_names(self) -> set:
        try:
            ps = self._get("/api/ps").get("models", [])
        except ProviderError:
            return set()
        names = set()
        for m in ps:
            names.add(m.get("name"))
            names.add(m.get("model"))
        return {n for n in names if n}

    # ---- model residency ----

    def load(self, model: str) -> dict:
        # Empty prompt + keep_alive loads the model into VRAM and returns.
        self._post("/api/generate",
                   {"model": model, "prompt": "", "keep_alive": self.keep_alive},
                   timeout=self.spec.get("load_timeout_s", 120))
        return {"ok": True, "loaded": model}

    def unload(self, model: str) -> dict:
        self._post("/api/generate",
                   {"model": model, "prompt": "", "keep_alive": 0},
                   timeout=self.spec.get("load_timeout_s", 30))
        return {"ok": True, "unloaded": model}

    # ---- vram prediction ----

    def estimate_vram(self, model: str, ctx: int, kv_type: str = "fp16") -> dict:
        show = self._post("/api/show", {"model": model},
                          timeout=self.spec.get("show_timeout_s", 15))
        meta = _normalize_show(show)
        if not ctx:
            ctx = meta.get("ctx_max") or 8192
        est = vram.estimate_vram(meta, ctx, kv_type)
        est["model"] = model
        return est

    # ---- metrics ----

    def metrics(self) -> dict:
        try:
            ps = self._get("/api/ps").get("models", [])
        except ProviderError:
            ps = []
        loaded = [{"name": m.get("name"), "vram_mb": _mb(m.get("size_vram")),
                   "ctx": (m.get("details") or {}).get("context_length")}
                  for m in ps]
        return {"loaded": loaded,
                "vram_used_mb": round(sum(l["vram_mb"] or 0 for l in loaded), 1)}

    # ---- inference ----

    def infer(self, model: str, messages: list, params: dict, sink: Sink) -> dict:
        options = {}
        if params.get("ctx") or params.get("num_ctx"):
            options["num_ctx"] = int(params.get("ctx") or params.get("num_ctx"))
        if params.get("temperature") is not None:
            options["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            options["top_p"] = params["top_p"]
        if params.get("max_tokens"):
            options["num_predict"] = int(params["max_tokens"])
        if params.get("tools"):
            return self._infer_tools(model, messages, params, sink, options)
        # /api/chat carries the whole conversation (messages) so multi-turn works
        body = {"model": model, "messages": messages, "stream": True}
        if options:
            body["options"] = options
        ka = params.get("keep_alive")
        if ka is not None:
            body["keep_alive"] = ka

        stats: dict = {}
        try:
            req = urllib.request.Request(
                self.base + "/api/chat",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(
                    req, timeout=params.get("timeout", 600)) as resp:
                for raw in resp:
                    raw = raw.strip()
                    if not raw:
                        continue
                    chunk = json.loads(raw)
                    piece = (chunk.get("message") or {}).get("content")
                    if piece:
                        sink.token(piece)
                    if chunk.get("done"):
                        ec = chunk.get("eval_count")
                        ed = chunk.get("eval_duration")  # nanoseconds
                        if ec and ed:
                            stats["eval_count"] = ec
                            stats["eval_duration_s"] = ed / 1e9
                        if chunk.get("prompt_eval_count"):
                            stats["prompt_tokens"] = chunk["prompt_eval_count"]
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"ollama inference failed: {exc}") from exc
        return stats

    def _infer_tools(self, model: str, messages: list, params: dict, sink: Sink,
                     options: dict) -> dict:
        """Tool-calling turn — non-streaming /api/chat with ``tools`` in the
        body (a tool call has no meaningful token stream). The whole reply
        arrives as one JSON object: the final text (if any) goes to *sink* as
        ONE token, and tool calls come back canonically in
        ``stats["tool_calls"]`` with synthesized ids (Ollama has none)."""
        body = {"model": model, "messages": _messages_to_ollama(messages),
                "stream": False, "tools": params["tools"]}
        if options:
            body["options"] = options
        ka = params.get("keep_alive")
        if ka is not None:
            body["keep_alive"] = ka
        data = self._post("/api/chat", body,
                          timeout=params.get("timeout", 600))
        message = data.get("message") or {}
        if message.get("content"):
            sink.token(message["content"])
        stats: dict = {}
        ec = data.get("eval_count")
        ed = data.get("eval_duration")  # nanoseconds
        if ec and ed:
            stats["eval_count"] = ec
            stats["eval_duration_s"] = ed / 1e9
        if data.get("prompt_eval_count"):
            stats["prompt_tokens"] = data["prompt_eval_count"]
        tool_calls = []
        for i, tc in enumerate(message.get("tool_calls") or [], 1):
            fn = tc.get("function") or {}
            tool_calls.append({"id": f"tc_{i}", "name": fn.get("name", ""),
                               "arguments": _parse_arguments(fn.get("arguments"))})
        if tool_calls:
            stats["tool_calls"] = tool_calls
        return stats

    # ---- HTTP plumbing ----

    def _get(self, path: str, timeout: float = 5) -> dict:
        try:
            with urllib.request.urlopen(self.base + path, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"ollama {path} unreachable: {exc}") from exc

    def _post(self, path: str, body: dict, timeout: float = 30) -> dict:
        try:
            req = urllib.request.Request(
                self.base + path, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read().decode("utf-8")
            return json.loads(data) if data.strip() else {}
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"ollama {path} failed: {exc}") from exc


def _normalize_show(show: dict) -> dict:
    """Pull the architecture fields the estimator needs out of /api/show, which
    keys everything by an arch prefix (e.g. 'qwen2.block_count')."""
    mi = show.get("model_info") or {}
    details = show.get("details") or {}
    arch = mi.get("general.architecture")

    def by_suffix(suffix: str):
        # exact arch-prefixed key first, then any key ending in the suffix
        if arch and f"{arch}.{suffix}" in mi:
            return mi[f"{arch}.{suffix}"]
        for k, v in mi.items():
            if k.endswith("." + suffix) or k == suffix:
                return v
        return None

    return {
        "n_params": mi.get("general.parameter_count") or details.get("parameter_size"),
        "quant": details.get("quantization_level"),
        "n_layers": by_suffix("block_count"),
        "n_heads": by_suffix("attention.head_count"),
        "n_kv_heads": by_suffix("attention.head_count_kv"),
        "embedding_length": by_suffix("embedding_length"),
        "ctx_max": by_suffix("context_length") or details.get("context_length"),
    }


def _messages_to_ollama(messages: list) -> list:
    """Canonical internal messages → Ollama /api/chat shape. Ollama has no
    tool-call ids: assistant tool_calls keep ``{"function": {"name",
    "arguments"}}`` (arguments stays a dict), ``role:"tool"`` messages keep
    their content (name carried as ``tool_name``, which Ollama understands),
    and the canonical id/tool_call_id fields are dropped. Plain messages pass
    through untouched."""
    out = []
    for m in messages or []:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [
                    {"function": {"name": tc.get("name", ""),
                                  "arguments": tc.get("arguments") or {}}}
                    for tc in m["tool_calls"]],
            })
        elif role == "tool":
            msg = {"role": "tool", "content": m.get("content", "")}
            if m.get("name"):
                msg["tool_name"] = m["name"]
            out.append(msg)
        else:
            out.append(m)
    return out


def _parse_arguments(raw) -> dict:
    """Tool-call arguments as a dict. Ollama already sends a parsed object, but
    be defensive: a JSON string is parsed, malformed input becomes
    ``{"_raw": <value>}`` rather than raising."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"_raw": raw}


def _mb(nbytes) -> float:
    if not nbytes:
        return 0.0
    return round(nbytes / (1024 * 1024), 1)
