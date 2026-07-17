"""Codex adapter — dual transport (OpenAI API or subscription CLI).

Mirror of the Claude adapter against OpenAI:

* ``api`` — OpenAI Chat Completions over ``urllib``, key from ``api_key_env``.
* ``cli-subscription`` — drive the installed ``codex`` CLI headlessly
  (``codex exec``, prompt on stdin) on the flat-rate subscription.

Exact ``codex exec`` flags/output are verified at build time; the CLI path
tolerates plain-text stdout as well as a JSON envelope.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

from memsom.providers import oai
from memsom.providers.base import (
    Capabilities,
    ModelInfo,
    Provider,
    ProviderError,
    ProviderStatus,
    Sink,
    run_no_window,
)

_API_BASE = "https://api.openai.com"


class CodexAdapter(Provider):
    def __init__(self, spec: dict) -> None:
        super().__init__(spec)
        self.transport = spec.get("transport", "cli-subscription")
        self.api_key_env = spec.get("api_key_env", "OPENAI_API_KEY")
        self.api_base = spec.get("api_base", _API_BASE)
        self.cli_path = spec.get("cli_path", "codex")
        self.default_model = spec.get("model", "gpt-5-codex")
        self.api_model = spec.get("api_model", "gpt-5-codex")
        self.max_tokens = spec.get("max_tokens", 4096)

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
        if _which(self.cli_path):
            return ProviderStatus("up", detail=f"{self.cli_path} on PATH")
        return ProviderStatus("down", detail=f"{self.cli_path} not found")

    def list_models(self) -> list[ModelInfo]:
        if self.transport == "api" and self._has_key():
            try:
                data = self._api_get("/v1/models", timeout=4)
                return [ModelInfo(name=m.get("id")) for m in data.get("data", [])
                        if m.get("id")]
            except ProviderError:
                pass
        return [ModelInfo(name=self.default_model,
                          meta={"note": "subscription CLI default"})]

    def infer(self, model: str, messages: list, params: dict, sink: Sink) -> dict:
        transport = params.get("transport") or self.transport
        if transport == "api":
            return self._infer_api(model, messages, params, sink)
        return self._infer_cli(model, messages, params, sink)

    def _infer_cli(self, model, messages, params, sink) -> dict:
        if params.get("tools"):
            raise ProviderError("custom tools not supported over cli transport")
        argv = [self.cli_path, "exec"]
        if model:
            argv += ["-m", model]
        try:
            proc = run_no_window(
                argv, input=_flatten(messages), capture_output=True, text=True,
                timeout=params.get("timeout", 600))
        except FileNotFoundError as exc:
            raise ProviderError(f"codex CLI not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderError("codex CLI timed out") from exc
        if proc.returncode != 0:
            raise ProviderError(
                f"codex CLI exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        text = _parse_cli_out(proc.stdout)
        sink.token(text)
        return {"transport": "cli-subscription"}

    def _infer_api(self, model, messages, params, sink) -> dict:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"${self.api_key_env} not set")
        if params.get("tools"):
            stats = oai.chat_once(self.api_base, model or self.api_model,
                                  messages, params, sink,
                                  headers={"Authorization": f"Bearer {key}"})
            stats["transport"] = "api"
            return stats
        body = {
            "model": model or self.api_model,
            "messages": messages,
        }
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        if params.get("max_tokens"):
            body["max_tokens"] = int(params["max_tokens"])
        try:
            req = urllib.request.Request(
                self.api_base + "/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(
                    req, timeout=params.get("timeout", 600)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"openai API {exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                json.JSONDecodeError) as exc:
            raise ProviderError(f"openai API failed: {exc}") from exc
        choices = data.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content", "")
        sink.token(text)
        usage = data.get("usage") or {}
        return {"transport": "api",
                "prompt_tokens": usage.get("prompt_tokens"),
                "eval_count": usage.get("completion_tokens")}

    def _api_get(self, path: str, timeout: float = 5) -> dict:
        key = os.environ.get(self.api_key_env)
        try:
            req = urllib.request.Request(
                self.api_base + path,
                headers={"Authorization": f"Bearer {key or ''}"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"{exc.code}") from exc
        except (urllib.error.URLError, OSError, TimeoutError,
                json.JSONDecodeError) as exc:
            raise ProviderError(str(exc)) from exc


def _parse_cli_out(stdout: str) -> str:
    stdout = (stdout or "").strip()
    if not stdout:
        return ""
    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict):
            return obj.get("result") or obj.get("text") or stdout
    except json.JSONDecodeError:
        pass
    return stdout


def _which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


def _flatten(messages: list) -> str:
    """Flatten a conversation into one prompt for the headless CLI (stdin)."""
    lines = []
    for m in messages or []:
        role = (m.get("role") or "user").capitalize()
        lines.append(f"{role}: {m.get('content', '')}")
    lines.append("Assistant:")
    return "\n\n".join(lines)
