"""Shared base for the load-on-demand voice adapters (STT + TTS).

Why this exists: Parakeet (speech->text) and Chatterbox (text->speech) are the
same *shape* of backend — a small local model you warm into VRAM on demand, hold
for a keep-alive window, then evict so the 30B chat model can have the card back.
That residency lifecycle is identical to Ollama's ``load``/``unload`` (see
``ollama.py``); only the actual inference call differs. So the lifecycle lives
here once and the two concrete adapters (``parakeet.py``, ``chatterbox.py``) each
add just their one inference method and their install metadata.

STUB BOUNDARY (deliberate, per the build guardrail): the *plumbing* is real —
capabilities, status, VRAM estimate, the two-phase-audited load/unload, and the
detached model-server launch via :class:`ProcessManager` are all wired. What is
stubbed is the inference itself: until the model runtime is actually installed on
the box, :meth:`installed` returns False and the inference methods return a clean
``model_installed: False`` JSON payload instead of pretending to transcribe or
speak. Installing the runtime (see each adapter's ``install_hint``) is the one
step that lights real inference up; there is a single, clearly marked seam in
each subclass where the real call goes.

VRAM residency: :meth:`load` runs an advisory pre-load estimate (reusing
``vram.estimate_vram`` + ``telemetry._read_gpu`` exactly as
``handlers.build_providers_payload`` does) so the UI can warn before committing a
load that won't fit next to whatever's already resident. A hard refuse-if-it-
won't-fit gate is opt-in per provider via ``hard_vram_gate`` in the profile.
"""

from __future__ import annotations

import importlib.util

from memsom.interface import telemetry
from memsom.providers import vram
from memsom.providers.base import (
    Capabilities,
    ModelInfo,
    Provider,
    ProviderError,
    ProviderStatus,
    now,
    run_no_window,
)


class VoiceAdapter(Provider):
    """Base for local, load-on-demand voice models. Subclasses set the class
    attributes below and implement exactly one inference method."""

    #: "parakeet" / "chatterbox" — also the registry kind. Set by subclass.
    kind = ""
    #: default model id served when a call omits ``model``.
    default_model = ""
    #: rough parameter count string ("0.6B") for the advisory VRAM estimate.
    param_count = None
    #: python import name that proves the runtime is installed (None = skip).
    runtime_module = None
    #: one-line, copy-pasteable "how to actually install this" for the stub.
    install_hint = ""

    def __init__(self, spec: dict, procman=None) -> None:
        super().__init__(spec)
        self.host = spec.get("host", "127.0.0.1")
        self.port = spec.get("port")
        self.exec = spec.get("exec")
        self.models_dir = spec.get("models_dir")
        # keep_alive holds the model resident after a call; "0" = evict now.
        # Mirrors ollama.py's keep_alive contract (string minutes or seconds).
        self.keep_alive = spec.get("keep_alive", "10m")
        self.default_model = spec.get("model") or self.default_model
        self._hard_vram_gate = bool(spec.get("hard_vram_gate", False))
        self._procman = procman
        # in-process residency marker. The panel is a single long-lived process
        # and adapters are registry singletons, so this is authoritative for the
        # "is it warm right now" question the UI asks.
        self._resident = None
        self._loaded_ts = None

    # ---- capability + status ----

    def capabilities(self) -> Capabilities:
        # Same stance as Ollama: the panel manages model RESIDENCY (load/unload),
        # not a separate daemon start/stop button — load() owns the server spawn.
        return Capabilities(can_start=False, can_load=True, has_vram=True,
                            can_estimate=True, transports=("native",))

    def installed(self) -> tuple:
        """(is_installed, detail). Today this gates the whole stub: False until
        the runtime module imports. Subclasses may override to also check for
        model files on disk."""
        if not self.runtime_module:
            return False, "no runtime configured"
        found = importlib.util.find_spec(self.runtime_module) is not None
        if found:
            return True, f"{self.runtime_module} present"
        return False, f"{self.runtime_module} not importable"

    def status(self) -> ProviderStatus:
        inst, why = self.installed()
        if self._resident:
            return ProviderStatus("up", detail=f"resident: {self._resident}")
        if not inst:
            return ProviderStatus("down", detail=f"stub - {why}")
        return ProviderStatus("down", detail="idle (not loaded)")

    # ---- inventory ----

    def list_models(self) -> list[ModelInfo]:
        name = self.default_model
        if not name:
            return []
        return [ModelInfo(name=name, loaded=(self._resident == name),
                          meta={"modality": self.kind,
                                "n_params": self.param_count,
                                "installed": self.installed()[0]})]

    # ---- vram prediction ----

    def estimate_vram(self, model: str, ctx: int, kv_type: str = "fp16") -> dict:
        """Advisory load-time estimate. Voice models are not KV-cache-dominated
        the way an LLM is, so this is essentially weights + overhead (the
        estimator returns ``partial: True``, which is honest here). Kept on the
        same code path as the LLM adapters so the UI reads one shape."""
        meta = {"n_params": self.param_count, "quant": self.spec.get("quant")}
        est = vram.estimate_vram(meta, ctx or 0, kv_type)
        est["model"] = model or self.default_model
        return est

    def _vram_advisory(self, model: str) -> dict:
        """estimate vs. currently-free VRAM. Best-effort — a missing GPU or a
        failed estimate degrades to ``{"checked": False}`` and never blocks a
        load (unless hard_vram_gate is on AND we actually have both numbers)."""
        try:
            est = self.estimate_vram(model, 0)
            estimate_mb = est.get("total_mb")
        except Exception:
            return {"checked": False}
        gpu = telemetry._read_gpu(run_no_window)
        if not gpu.get("available"):
            return {"checked": False, "estimate_mb": estimate_mb}
        total = gpu.get("vram_total_mb")
        used = gpu.get("vram_used_mb")
        free = (total - used) if (total is not None and used is not None) else None
        would_exceed = (free is not None and estimate_mb is not None
                        and estimate_mb > free)
        return {"checked": True, "estimate_mb": estimate_mb, "free_mb": free,
                "would_exceed": would_exceed, "partial": est.get("partial")}

    # ---- model residency ----

    def load(self, model: str = None) -> dict:
        model = model or self.default_model
        if not model:
            raise ProviderError(f"{self.label}: no model configured to load")
        advisory = self._vram_advisory(model)
        if self._hard_vram_gate and advisory.get("would_exceed"):
            raise ProviderError(
                f"refusing to load {model}: estimate "
                f"{advisory.get('estimate_mb')}MB > free {advisory.get('free_mb')}MB "
                f"(hard_vram_gate on)")

        inst, why = self.installed()
        if not inst:
            # Stub path: no runtime yet. Honest failure with the fix inline —
            # the action audit records this, and the UI shows the install hint.
            raise ProviderError(
                f"{self.label} runtime not installed ({why}). "
                f"Install: {self.install_hint}")

        # --- real path: lights up once the runtime is installed ---
        # Launch the model server DETACHED via procman so it survives a panel
        # restart (same discipline as llama.cpp/vLLM), then mark it resident.
        if self._procman is not None and self.exec:
            self._procman.start(self.id, self._server_argv(model),
                                port=self.port, model=model)
        self._resident = model
        self._loaded_ts = now()
        return {"ok": True, "loaded": model, "vram_estimate": advisory}

    def unload(self, model: str = None) -> dict:
        if self._procman is not None and self._procman.is_running(self.id):
            try:
                self._procman.stop(self.id)
            except ProviderError:
                pass  # already gone — evicting VRAM is the goal, not the pid
        freed = self._resident
        self._resident = None
        self._loaded_ts = None
        return {"ok": True, "unloaded": model or freed or self.default_model}

    def _server_argv(self, model: str) -> list:
        """argv for the detached model server. Subclasses that run a server
        override this; the default assumes ``exec model --host --port``."""
        return [self.exec, model, "--host", self.host, "--port", str(self.port)]

    # ---- metrics ----

    def metrics(self) -> dict:
        inst = self.installed()[0]
        return {"resident": self._resident, "keep_alive": self.keep_alive,
                "installed": inst,
                "server_running": bool(self._procman
                                       and self._procman.is_running(self.id))}

    # ---- inference (chat-style is not what voice models do) ----

    def infer(self, model, messages, params, sink) -> dict:
        raise ProviderError(
            f"{self.label} is a voice model — use the /api/voice endpoints, "
            f"not chat inference")

    # ---- stub helper for the concrete inference methods ----

    def _stub_or_raise(self):
        """Return an (installed, detail) tuple; concrete transcribe/synthesize
        call this first and short-circuit to a clean stub payload when the
        runtime isn't present."""
        return self.installed()
