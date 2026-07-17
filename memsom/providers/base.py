"""The Provider contract — one interface, every backend.

A backend is anything that can serve tokens: a local model server (Ollama,
llama.cpp, vLLM) or a cloud/CLI agent (Claude, Codex). They are wildly
different underneath — different HTTP shapes, different auth, some have VRAM and
some don't, some you can start/stop and some you can't. The panel must not know
any of that, so every backend implements the SAME small surface and advertises
what it actually supports via :class:`Capabilities`. The UI greys out what a
backend can't do rather than pretending.

Design rules that keep this honest:

* **Capabilities are declared, not guessed.** A cloud adapter sets
  ``has_vram=False``; the panel then renders N/A for its VRAM gauge instead of
  showing a fake zero.
* **No secrets cross the wire.** API keys are read from the environment by NAME
  (the profile carries ``api_key_env``, never the key itself) and never appear
  in a response body or the audit log.
* **Streaming is push, durability is elsewhere.** ``infer`` pushes tokens into a
  :class:`Sink`; it does not know or care that the sink is a disk-backed file
  the panel polls. That decoupling is what lets a generation outlive the app
  that started it (see :mod:`memsom.providers.session`).
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# On Windows, a process spawned DETACHED (no console — which is how the panel
# server itself is launched) makes every console subprocess it runs allocate a
# NEW console window, which flashes on screen. CREATE_NO_WINDOW suppresses it.
# Use run_no_window for ANY subprocess on a polled/background path (nvidia-smi,
# tasklist, the CLI adapters) so the panel never flashes a console.
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) \
    if sys.platform == "win32" else 0


def run_no_window(*args, **kwargs):
    """subprocess.run with the no-console-window flag on Windows; identical
    signature so it drops in anywhere subprocess.run is expected."""
    if _CREATE_NO_WINDOW:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
    return subprocess.run(*args, **kwargs)


# ---------------------------------------------------------------------------
# Value objects (plain dataclasses — they serialize straight to the JSON the
# panel emits; `.as_dict()` where the field names need massaging).
# ---------------------------------------------------------------------------


@dataclass
class Capabilities:
    """What a backend can actually do. The UI keys every affordance off this."""

    can_start: bool = False   # can the panel spawn/kill the serving process?
    can_load: bool = False    # can models be loaded/unloaded into VRAM?
    has_vram: bool = False    # does it consume local VRAM (→ show gauges)?
    can_estimate: bool = False  # can pre-load VRAM be predicted from metadata?
    transports: tuple = ("native",)  # e.g. ("api", "cli-subscription") for cloud

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["transports"] = list(self.transports)
        return d


@dataclass
class ProviderStatus:
    """A backend's reachability at a moment in time.

    ``state`` is one of ``up`` (serving), ``down`` (unreachable / process not
    running), ``unauthed`` (reachable but no valid credential — cloud only).
    """

    state: str
    ms: Optional[float] = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {"state": self.state, "ms": self.ms, "detail": self.detail}


@dataclass
class ModelInfo:
    """One model a backend can serve.

    ``meta`` carries the architecture fields the VRAM estimator needs
    (n_params, n_layers, n_kv_heads, head_dim, quant, ...) when the backend can
    supply them; empty otherwise. ``loaded`` is best-effort (None = unknown).
    """

    name: str
    size_bytes: Optional[int] = None
    quant: Optional[str] = None
    ctx_max: Optional[int] = None
    loaded: Optional[bool] = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "size_bytes": self.size_bytes,
            "quant": self.quant,
            "ctx_max": self.ctx_max,
            "loaded": self.loaded,
            "meta": self.meta,
        }


class Sink:
    """Where ``infer`` pushes tokens as they arrive.

    Deliberately tiny: adapters call :meth:`token` per chunk and nothing else.
    The concrete sink used in production (:mod:`memsom.providers.session`)
    appends each token to an fsync'd file so a poller — or a reopened app — can
    replay it. A trivial in-memory sink is used in tests.
    """

    def token(self, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ListSink(Sink):
    """In-memory sink: collects tokens into ``.tokens``. For tests / probes."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def token(self, text: str) -> None:
        self.tokens.append(text)

    def text(self) -> str:
        return "".join(self.tokens)


class ProviderError(Exception):
    """Any backend failure surfaced to the panel. ``str(exc)`` is user-facing —
    keep it clean (no secrets, no stack noise)."""


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class Provider:
    """Base class every adapter subclasses.

    The defaults here are the "not supported" answers, so an adapter only
    overrides what it actually does. Anything a backend can't do raises
    :class:`ProviderError` (start/stop/load on a cloud adapter, etc.) — the
    panel should gate those on :meth:`capabilities` and never call them, but
    the raise is the backstop.
    """

    #: stable id from the profile (e.g. "ollama"); set by the registry.
    id: str = ""
    #: kind discriminator ("ollama" | "llamacpp" | "vllm" | "claude" | "codex").
    kind: str = ""
    #: human label for the UI.
    label: str = ""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.id = spec.get("id") or spec.get("kind", "")
        self.kind = spec.get("kind", "")
        self.label = spec.get("label") or self.id

    # ---- capability + status ----

    def capabilities(self) -> Capabilities:  # pragma: no cover - overridden
        return Capabilities()

    def status(self) -> ProviderStatus:  # pragma: no cover - overridden
        return ProviderStatus("down", detail="not implemented")

    # ---- inventory ----

    def list_models(self) -> list[ModelInfo]:
        return []

    # ---- lifecycle (local serving process) ----

    def start(self) -> dict:
        raise ProviderError(f"{self.label} cannot be started from the panel")

    def stop(self) -> dict:
        raise ProviderError(f"{self.label} cannot be stopped from the panel")

    # ---- model residency (VRAM) ----

    def load(self, model: str) -> dict:
        raise ProviderError(f"{self.label} does not support loading models")

    def unload(self, model: str) -> dict:
        raise ProviderError(f"{self.label} does not support unloading models")

    # ---- vram prediction ----

    def estimate_vram(self, model: str, ctx: int, kv_type: str = "fp16") -> dict:
        raise ProviderError(f"{self.label} cannot estimate VRAM")

    # ---- metrics (measured, live) ----

    def metrics(self) -> dict:
        """Live measured numbers for this backend. Default: nothing measurable
        (cloud). Local adapters fill vram_used_mb / vram_total_mb from
        nvidia-smi (shared, so usually filled by the handler, not here)."""
        return {}

    # ---- inference ----

    def infer(self, model: str, messages: list, params: dict, sink: Sink) -> dict:
        """Generate a reply for the conversation *messages* ([{role, content},
        ...]), pushing tokens to *sink* as they arrive. Carrying the full
        message list (not a single prompt) is what makes multi-turn chat work.
        Returns a stats dict that MAY include authoritative counters
        (``eval_count``, ``eval_duration_s``); the session runner fills in
        wall-clock TPS when the backend doesn't report its own. Raises
        :class:`ProviderError` on any failure."""
        raise ProviderError(f"{self.label} does not support inference")


# ---------------------------------------------------------------------------
# Small shared helpers used by more than one adapter.
# ---------------------------------------------------------------------------


def now() -> float:
    """Monotonic-ish wall clock for TPS timing. Kept in one place so tests can
    monkeypatch it; adapters never call time.time() directly."""
    return time.time()


def tcp_ms(probe: Callable[[str, int, float], Any], host: str, port: int,
           timeout: float) -> ProviderStatus:
    """Turn a TCP probe result into a ProviderStatus. *probe* is
    ``telemetry._probe_one`` (host, port, timeout) -> (ok, ms), injected so this
    stays testable and reuses the one socket-probe implementation in the repo."""
    ok, ms = probe(host, port, timeout)
    if ok:
        return ProviderStatus("up", ms=ms)
    return ProviderStatus("down", ms=ms, detail=f"no listener on {host}:{port}")
