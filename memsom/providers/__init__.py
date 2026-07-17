"""memsom.providers — the local-AI control plane.

One adapter interface (:mod:`memsom.providers.base`) that every model backend
implements — Ollama, llama.cpp, vLLM, and the cloud/CLI agents (Claude, Codex).
The panel server talks only to the interface; adding a backend is one new file.

Nothing here opens a DB or touches the memory store. Adapters talk to their
backends over stdlib ``urllib`` / ``subprocess`` only (matching the existing
``memsom.distill.llm`` idiom), and the panel wires them via
:func:`memsom.providers.registry.build_registry`.
"""

from memsom.providers.base import (
    Capabilities,
    ModelInfo,
    Provider,
    ProviderStatus,
    Sink,
)
from memsom.providers.registry import build_registry

__all__ = [
    "Capabilities",
    "ModelInfo",
    "Provider",
    "ProviderStatus",
    "Sink",
    "build_registry",
]
