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

import re
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
class LaunchOption:
    """One knob the panel may pass to :meth:`Provider.start`.

    Declared by the adapter, rendered by the UI. Same discipline as
    :class:`Capabilities`: the frontend hardcodes no backend's flags, it draws
    whatever the adapter says it accepts. ``key`` is what travels in the request
    body; the adapter maps it to a real command-line flag on its own side, so a
    flag name never crosses the wire and a request can never name one.
    """

    key: str
    label: str
    type: str = "int"           # "int" | "float" | "bool" | "text" | "select"
    default: Any = None         # None = "leave it to the backend's own default"
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    hint: str = ""
    #: for type="text": a full-match regex the value MUST satisfy. Declared here
    #: rather than in the adapter's code so the same rule serves as the server's
    #: gate and the UI's input hint — one source of truth.
    pattern: Optional[str] = None
    #: for type="select": the only accepted values. The UI renders a dropdown
    #: from exactly this list, so it cannot offer an option the server refuses.
    choices: tuple = ()
    #: model meta key holding this option's real ceiling (e.g. "n_layers"), so
    #: the UI can bound the field per selected model instead of guessing.
    max_from_meta: Optional[str] = None
    #: option only applies when this model meta key is truthy (e.g. "n_experts"
    #: for the MoE knobs) — the UI disables it otherwise.
    requires_meta: Optional[str] = None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["choices"] = list(self.choices)
        return d


@dataclass
class Capabilities:
    """What a backend can actually do. The UI keys every affordance off this."""

    can_start: bool = False   # can the panel spawn/kill the serving process?
    can_load: bool = False    # can models be loaded/unloaded into VRAM?
    has_vram: bool = False    # does it consume local VRAM (→ show gauges)?
    can_estimate: bool = False  # can pre-load VRAM be predicted from metadata?
    transports: tuple = ("native",)  # e.g. ("api", "cli-subscription") for cloud
    #: LaunchOption tuple — the knobs `start` accepts. Empty means START takes
    #: nothing but a model (or nothing at all).
    launch_options: tuple = ()

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["transports"] = list(self.transports)
        d["launch_options"] = [o.as_dict() for o in self.launch_options]
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

    def reasoning(self, text: str) -> None:
        """A chunk of the model's THINKING, not of its answer.

        Separate channel because they are separate things: a reasoning model
        streams its scratchpad into ``reasoning_content`` and its actual reply
        into ``content``, and merging them means the answer arrives buried in
        several hundred tokens of deliberation. Default is a no-op so every
        existing sink keeps working — a backend that has no notion of reasoning
        simply never calls it.
        """


class ListSink(Sink):
    """In-memory sink: collects tokens into ``.tokens``. For tests / probes."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.thoughts: list[str] = []

    def token(self, text: str) -> None:
        self.tokens.append(text)

    def reasoning(self, text: str) -> None:
        self.thoughts.append(text)

    def text(self) -> str:
        return "".join(self.tokens)

    def thinking(self) -> str:
        return "".join(self.thoughts)


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

    def start(self, model: str = None, options: dict = None) -> dict:
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


def coerce_launch_options(declared, options) -> dict:
    """Validate a REQUEST-SUPPLIED option dict against an adapter's declared
    :class:`LaunchOption` list, returning ``{key: coerced value}``.

    These values end up in an argv, so this is the gate. The rule the provider
    layer already follows (see :mod:`memsom.providers.registry`) is that nothing
    from a request body reaches a command line unvalidated — spawning is
    ``Popen(list)`` with no shell, so this is defence in depth rather than the
    only thing between a body and a process, but a whitelist is what makes that
    claim checkable.

    * an unknown key is an ERROR naming it, never a silent drop — a knob that
      does nothing is worse than one that refuses;
    * ints/floats are coerced and range-checked against the declaration;
    * ``bool`` accepts only real booleans;
    * ``text`` must full-match the declared ``pattern``.
    """
    if not options:
        return {}
    if not isinstance(options, dict):
        raise ProviderError("launch options must be a JSON object")
    by_key = {o.key: o for o in declared}
    unknown = [k for k in options if k not in by_key]
    if unknown:
        raise ProviderError(
            f"unknown launch option(s): {', '.join(sorted(unknown))} "
            f"(accepted: {', '.join(sorted(by_key)) or 'none'})")

    out: dict = {}
    for key, raw in options.items():
        if raw is None or raw == "":
            continue  # "leave it to the backend default"
        opt = by_key[key]
        if opt.type == "bool":
            if not isinstance(raw, bool):
                raise ProviderError(f"{key} must be true or false")
            if raw:
                out[key] = True
            continue
        if opt.type == "select":
            val = str(raw)
            if val not in opt.choices:
                raise ProviderError(
                    f"{key} must be one of: {', '.join(opt.choices)} (got {val!r})")
            out[key] = val
            continue
        if opt.type == "text":
            val = str(raw)
            if opt.pattern and not re.fullmatch(opt.pattern, val):
                raise ProviderError(f"{key} is not a valid value: {val!r}")
            out[key] = val
            continue
        # numeric. bool is an int subclass in Python — reject it explicitly so a
        # stray `true` can't silently become 1.
        if isinstance(raw, bool):
            raise ProviderError(f"{key} must be a number")
        try:
            val = int(raw) if opt.type == "int" else float(raw)
        except (TypeError, ValueError):
            raise ProviderError(f"{key} must be {'an integer' if opt.type == 'int' else 'a number'}") from None
        if opt.min is not None and val < opt.min:
            raise ProviderError(f"{key} must be >= {opt.min:g} (got {val:g})")
        if opt.max is not None and val > opt.max:
            raise ProviderError(f"{key} must be <= {opt.max:g} (got {val:g})")
        out[key] = val
    return out


def tcp_ms(probe: Callable[[str, int, float], Any], host: str, port: int,
           timeout: float) -> ProviderStatus:
    """Turn a TCP probe result into a ProviderStatus. *probe* is
    ``telemetry._probe_one`` (host, port, timeout) -> (ok, ms), injected so this
    stays testable and reuses the one socket-probe implementation in the repo."""
    ok, ms = probe(host, port, timeout)
    if ok:
        return ProviderStatus("up", ms=ms)
    return ProviderStatus("down", ms=ms, detail=f"no listener on {host}:{port}")
