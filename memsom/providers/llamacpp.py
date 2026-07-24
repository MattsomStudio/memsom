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

Launch knobs (``LAUNCH_OPTIONS`` below) are declared, not hardcoded in the UI:
the panel draws a control per declared option and posts them back by KEY. This
module owns the key -> flag mapping, so a request body never names a
command-line flag and cannot introduce one.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from memsom.providers import gguf, oai
from memsom.providers.base import (
    Capabilities,
    LaunchOption,
    ModelInfo,
    Provider,
    ProviderError,
    ProviderStatus,
    Sink,
    coerce_launch_options,
    now,
)

# --------------------------------------------------------------------------
# The launch surface: what the panel may set, and what each key becomes on the
# llama-server command line. Verified against llama-server --help (b9601).
#
# `ctx`/`n_predict`/`temp` are the server's DEFAULTS — a per-request temperature
# from the INFERENCE cockpit still overrides `--temp`, but `--ctx-size` is fixed
# at launch, which is exactly why it belongs here and not only there.
#
# The MoE knobs are the reason this form exists. On a 12 GB card a 30B-A3B won't
# fit with every expert resident, but its experts are sparse: `-ncmoe N` parks
# the MoE weights of the first N layers on the CPU and keeps attention on the
# GPU, which is the difference between "won't load" and "runs". `-ot` is the
# escape hatch for placing specific tensors by regex.
# --------------------------------------------------------------------------

#: key -> flag. A flag with `None` arity is a bare switch (bool option).
_LAUNCH_FLAGS = {
    "ctx": "--ctx-size",
    "n_predict": "--n-predict",
    "temp": "--temp",
    "n_gpu_layers": "--n-gpu-layers",
    "n_cpu_moe": "--n-cpu-moe",
    "cpu_moe": "--cpu-moe",
    "override_tensor": "--override-tensor",
    "reasoning": "--reasoning",
    "reasoning_budget": "--reasoning-budget",
}

#: emitted in this order so the argv reads the way the flags are documented
_LAUNCH_ORDER = ("ctx", "n_predict", "temp", "n_gpu_layers",
                 "cpu_moe", "n_cpu_moe", "override_tensor",
                 "reasoning", "reasoning_budget")

#: `-ot` takes `<tensor name pattern>=<buffer type>` (comma-separated pairs are
#: allowed), e.g. `blk\.(1[5-9]|2[0-9])\.ffn_.*_exps\.=CPU`. Regex
#: metacharacters have to be allowed for it to be useful at all, so the gate is
#: a strict character whitelist — no spaces, no quotes, no shell characters —
#: plus a leading lookahead requiring at least one `=`, since a value with no
#: buffer-type half is not an override, it's a typo.
_OT_PATTERN = r"(?=[^=]*=)[A-Za-z0-9_.,^$()|\[\]{}*+?\\=/-]{3,256}"

LAUNCH_OPTIONS = (
    LaunchOption(key="ctx", label="context", type="int", min=256, max=1048576,
                 step=1024, max_from_meta="ctx_max",
                 hint="--ctx-size; fixed for the life of the server"),
    LaunchOption(key="n_predict", label="max output", type="int",
                 min=-1, max=1048576, step=128,
                 hint="--n-predict; -1 = unbounded"),
    LaunchOption(key="temp", label="temperature", type="float",
                 min=0.0, max=2.0, step=0.05,
                 hint="--temp; the server default, overridden per request"),
    LaunchOption(key="n_gpu_layers", label="gpu layers", type="int",
                 min=0, max=1024, step=1, max_from_meta="n_layers",
                 hint="--n-gpu-layers; how many layers live in VRAM"),
    LaunchOption(key="cpu_moe", label="all experts on CPU", type="bool",
                 requires_meta="n_experts",
                 hint="--cpu-moe; every MoE weight stays in RAM"),
    LaunchOption(key="n_cpu_moe", label="MoE layers on CPU", type="int",
                 min=0, max=1024, step=1, max_from_meta="n_layers",
                 requires_meta="n_experts",
                 hint="--n-cpu-moe N; experts of the first N layers on CPU"),
    LaunchOption(key="override_tensor", label="tensor override", type="text",
                 pattern=_OT_PATTERN,
                 hint=r"--override-tensor, e.g. blk\.(1[5-9])\.ffn_.*_exps\.=CPU"),
    # A reasoning model streams its scratchpad into `reasoning_content` and its
    # reply into `content`. "off" makes it answer directly; "auto" (the
    # server's own default) decides from the chat template.
    LaunchOption(key="reasoning", label="thinking", type="select",
                 choices=("auto", "on", "off"), default="auto",
                 hint="--reasoning; off = answer directly, no scratchpad"),
    # Worth having next to the toggle: with thinking ON and a small max output,
    # deliberation can consume the ENTIRE budget and the reply comes back
    # empty. This caps the scratchpad instead of the answer.
    LaunchOption(key="reasoning_budget", label="thinking budget", type="int",
                 min=-1, max=1048576, step=256,
                 hint="--reasoning-budget; -1 unlimited, 0 none, N = token cap"),
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
                            can_estimate=False, transports=("native",),
                            launch_options=LAUNCH_OPTIONS)

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
            # architecture facts straight from the GGUF header (cached per
            # file), so the launch form can bound --n-cpu-moe by the real layer
            # count and hide the MoE knobs on a dense model instead of offering
            # a control that does nothing.
            arch = gguf.read_meta(path)
            out.append(ModelInfo(name=name, size_bytes=_size(Path(path)),
                                 quant=arch.get("quant"),
                                 ctx_max=arch.get("ctx_max"),
                                 loaded=(name in served or path in served),
                                 meta={"path": path,
                                       "n_layers": arch.get("n_layers"),
                                       "n_experts": arch.get("n_experts"),
                                       "n_experts_used":
                                           arch.get("n_experts_used")}))
        # a served model that isn't a discovered file (e.g. HF repo id) still shows
        for s in served:
            if s not in index and not any(s == m.name for m in out):
                out.append(ModelInfo(name=s, loaded=True))
        return out

    def start(self, model: str = None, options: dict = None) -> dict:
        if self._procman is None:
            raise ProviderError("no process manager configured")
        model = model or self.spec.get("model")
        if not model:
            raise ProviderError("start requires a model (GGUF name or path)")
        path = self._resolve(model)
        # validated FIRST — a bad knob must refuse the launch, not spawn a
        # server with the flag silently dropped.
        opts = coerce_launch_options(LAUNCH_OPTIONS, options)
        argv = [self.exec, "-m", path, "--host", self.host,
                "--port", str(self.port), *_flags(opts), *self.extra_args]
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


def _flags(opts: dict) -> list:
    """Coerced options -> argv fragment. Bools become bare switches; everything
    else becomes `--flag value` as two elements (never one interpolated string,
    which is what turns a value into a second argument)."""
    out: list = []
    for key in _LAUNCH_ORDER:
        if key not in opts:
            continue
        flag = _LAUNCH_FLAGS[key]
        val = opts[key]
        if val is True:
            out.append(flag)
        else:
            out += [flag, str(val)]
    return out


def _wrap(argv: list, host_kind: str) -> list:
    if host_kind == "wsl2":
        return ["wsl.exe", "-e", *argv]
    return argv


def _size(p: Path):
    try:
        return p.stat().st_size
    except OSError:
        return None
