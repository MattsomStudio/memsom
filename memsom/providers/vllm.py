"""vLLM adapter — drives a ``vllm serve`` process (OpenAI-compatible).

Like llama.cpp, one served model per process, OpenAI API shape, usually launched
into WSL2 (``host_kind: wsl2``). Heavier GPU serving; same interface.
"""

from __future__ import annotations

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
    now,
)


class VllmAdapter(Provider):
    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec)
        self.host = spec.get("host", "127.0.0.1")
        self.port = spec.get("port", 8000)
        self.base = spec.get("base_url") or f"http://{self.host}:{self.port}"
        self.exec = spec.get("exec", "vllm")
        self.host_kind = spec.get("host_kind", "wsl2")
        self.extra_args = spec.get("args", [])
        self._procman = procman

    def capabilities(self) -> Capabilities:
        return Capabilities(can_start=True, can_load=False, has_vram=True,
                            can_estimate=False, transports=("native",))

    def status(self) -> ProviderStatus:
        t0 = now()
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2):
                pass
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError) as exc:
            return ProviderStatus("down", detail=f"{self.host}:{self.port} — {exc}")
        return ProviderStatus("up", ms=round((now() - t0) * 1000, 1))

    def list_models(self) -> list[ModelInfo]:
        if self.status().state != "up":
            model = self.spec.get("model")
            return [ModelInfo(name=model, loaded=False)] if model else []
        try:
            return [ModelInfo(name=n, loaded=True)
                    for n in oai.list_models(self.base, timeout=3)]
        except ProviderError:
            return []

    def start(self, model: str = None) -> dict:
        if self._procman is None:
            raise ProviderError("no process manager configured")
        model = model or self.spec.get("model")
        if not model:
            raise ProviderError("start requires a model name")
        argv = [self.exec, "serve", model, "--host", self.host,
                "--port", str(self.port), *self.extra_args]
        return self._procman.start(self.id, _wrap(argv, self.host_kind),
                                   port=self.port, model=model)

    def stop(self) -> dict:
        if self._procman is None:
            raise ProviderError("no process manager configured")
        return self._procman.stop(self.id)

    def infer(self, model: str, messages: list, params: dict, sink: Sink) -> dict:
        if params.get("tools"):
            return oai.chat_once(self.base, model, messages, params, sink)
        return oai.chat_stream(self.base, model, messages, params, sink)


def _wrap(argv: list, host_kind: str) -> list:
    if host_kind == "wsl2":
        return ["wsl.exe", "-e", *argv]
    return argv
