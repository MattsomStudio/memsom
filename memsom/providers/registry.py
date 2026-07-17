"""Build the adapter registry from a profile's ``providers`` block.

Each entry is ``{"id", "kind", "label", ...kind-specific...}``. The kind selects
the adapter class. Local serving adapters (llama.cpp, vLLM) get the shared
:class:`ProcessManager` so their start/stop survives a panel restart. Unknown
kinds are skipped with a warning rather than crashing the panel — a typo in one
provider must not take down the tuning/telemetry the panel primarily exists for.

Everything host/port/path/key comes from the profile, resolved here, never from
a request body — the same anti-injection rule the knob path follows.
"""

from __future__ import annotations

import sys
from pathlib import Path

from memsom.providers.claude import ClaudeAdapter
from memsom.providers.codex import CodexAdapter
from memsom.providers.llamacpp import LlamaCppAdapter
from memsom.providers.ollama import OllamaAdapter
from memsom.providers.procman import ProcessManager
from memsom.providers.vllm import VllmAdapter

# kinds that manage a local serving process → need the ProcessManager
_PROC_KINDS = {"llamacpp", "vllm"}


def build_registry(profile: dict, *, servers_path=None) -> dict:
    """Return an ordered ``{provider_id: adapter}`` dict."""
    specs = profile.get("providers") or []
    procman = ProcessManager(servers_path) if servers_path else None
    registry: dict = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        kind = spec.get("kind")
        pid = spec.get("id") or kind
        if not pid or pid in registry:
            continue
        try:
            adapter = _construct(kind, spec, procman)
        except Exception as exc:  # never let one bad spec kill the panel
            print(f"[memsom-panel] skipping provider {pid!r} ({kind!r}): {exc}",
                  file=sys.stderr)
            continue
        if adapter is None:
            print(f"[memsom-panel] unknown provider kind {kind!r} for {pid!r}",
                  file=sys.stderr)
            continue
        registry[pid] = adapter
    return registry


def _construct(kind: str, spec: dict, procman):
    if kind == "ollama":
        return OllamaAdapter(spec)
    if kind == "llamacpp":
        return LlamaCppAdapter(spec, procman=procman)
    if kind == "vllm":
        return VllmAdapter(spec, procman=procman)
    if kind == "claude":
        return ClaudeAdapter(spec)
    if kind == "codex":
        return CodexAdapter(spec)
    return None
