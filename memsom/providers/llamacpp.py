"""llama.cpp adapter — drives a ``llama-server`` process.

llama-server serves ONE model at a time and speaks the OpenAI-compatible API, so
"load a model" means "start the server on that GGUF" (can_load=False,
can_start=True). ``host_kind`` decides native Windows vs WSL2 — WSL is a *host*,
not a separate provider.

Model discovery scans one or more ``models_dirs`` RECURSIVELY (GGUFs live both
flat in a models folder AND nested in the HuggingFace cache under
``hub/models--*/snapshots/<hash>/*.gguf``) and skips the ``ggml-vocab-*``
tokenizer files, which are not runnable models. Each discovered basename maps to
its full path so ``start`` can launch it.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

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


class LlamaCppAdapter(Provider):
    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec)
        self.host = spec.get("host", "127.0.0.1")
        self.port = spec.get("port", 8080)
        self.base = spec.get("base_url") or f"http://{self.host}:{self.port}"
        self.exec = spec.get("exec", "llama-server")
        self.host_kind = spec.get("host_kind", "native")
        # accept a single models_dir OR a list of models_dirs
        dirs = spec.get("models_dirs")
        if not dirs and spec.get("models_dir"):
            dirs = [spec["models_dir"]]
        self.model_dirs = [Path(d) for d in (dirs or [])]
        self.extra_args = spec.get("args", [])
        self._procman = procman
        self._index: dict = {}  # basename -> full path (built by _discover)

    def capabilities(self) -> Capabilities:
        return Capabilities(can_start=True, can_load=False, has_vram=True,
                            can_estimate=False, transports=("native",))

    def status(self) -> ProviderStatus:
        t0 = now()
        try:
            with urllib.request.urlopen(
                    self.base + "/health",
                    timeout=self.spec.get("status_timeout_s", 0.75)):
                pass
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                TimeoutError) as exc:
            return ProviderStatus("down", detail=f"{self.host}:{self.port} — {exc}")
        return ProviderStatus("up", ms=round((now() - t0) * 1000, 1))

    # ---- model discovery ----

    def _discover(self) -> dict:
        """basename -> full path, recursively across all model dirs, minus the
        ggml-vocab-* tokenizer files. Cached on the adapter; rebuilt each list."""
        index: dict = {}
        for d in self.model_dirs:
            if not d.is_dir():
                continue
            for p in d.rglob("*.gguf"):
                if p.name.lower().startswith("ggml-vocab"):
                    continue
                # first winner keeps the name; HF snapshots have unique filenames
                index.setdefault(p.name, str(p))
        self._index = index
        return index

    def list_models(self) -> list[ModelInfo]:
        index = self._discover()
        served = set()
        if self.status().state == "up":
            try:
                served = set(oai.list_models(self.base, timeout=3))
            except ProviderError:
                served = set()
        out = []
        for name, path in sorted(index.items()):
            out.append(ModelInfo(name=name, size_bytes=_size(Path(path)),
                                 loaded=(name in served or path in served),
                                 meta={"path": path}))
        # a served model that isn't a discovered file (e.g. HF repo id) still shows
        for s in served:
            if s not in index and not any(s == m.name for m in out):
                out.append(ModelInfo(name=s, loaded=True))
        return out

    def start(self, model: str = None) -> dict:
        if self._procman is None:
            raise ProviderError("no process manager configured")
        model = model or self.spec.get("model")
        if not model:
            raise ProviderError("start requires a model (GGUF name or path)")
        path = self._resolve(model)
        argv = [self.exec, "-m", path, "--host", self.host,
                "--port", str(self.port), *self.extra_args]
        return self._procman.start(self.id, _wrap(argv, self.host_kind),
                                   port=self.port, model=model)

    def _resolve(self, model: str) -> str:
        """Map a model name to a launchable GGUF path."""
        if not self._index:
            self._discover()
        if model in self._index:
            return self._index[model]
        p = Path(model)
        if p.is_absolute() and p.is_file():
            return str(p)
        # bare name with extension we didn't index, or a relative path
        raise ProviderError(f"model not found: {model!r} "
                            f"(known: {', '.join(sorted(self._index)[:6]) or 'none'})")

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


def _size(p: Path):
    try:
        return p.stat().st_size
    except OSError:
        return None
