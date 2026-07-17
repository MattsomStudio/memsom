"""Builtin tool catalog + agent-spec -> Tool instantiation.

An agent's config stores tools as small specs (``{"name", "type", "options"}``)
so the panel can round-trip them as JSON; this module is the one place those
specs become live :class:`Tool` objects, and the one place the UI asks "what
builtins exist and how are they configured" (``GET /api/agents/tools``).
Unknown types fail loudly at build time — a silently dropped tool is an agent
that mysteriously can't do what its config says it can.
"""

from __future__ import annotations

from memsom.providers.tools.base import Tool, ToolError
from memsom.providers.tools.builtins import (
    FileRead,
    HttpFetch,
    MemoryRecall,
    Shell,
    WebSearch,
)

BUILTIN_TOOLS: dict[str, type[Tool]] = {
    HttpFetch.type: HttpFetch,
    WebSearch.type: WebSearch,
    MemoryRecall.type: MemoryRecall,
    FileRead.type: FileRead,
    Shell.type: Shell,
}


def build_tools(specs: list[dict]) -> list[Tool]:
    """Instantiate the tools an agent spec names.

    Each spec is ``{"name": str, "type": str, "options": dict}``. The caller
    guarantees name uniqueness within one agent (suffixing duplicates); here we
    just stamp the name onto the instance. Unknown types raise
    :class:`ToolError` — never silently skip.
    """
    tools: list[Tool] = []
    for spec in specs:
        ttype = spec.get("type")
        cls = BUILTIN_TOOLS.get(ttype)
        if cls is None:
            raise ToolError(f"unknown tool type: {ttype!r}")
        tool = cls(spec.get("options") or {})
        tool.name = spec.get("name") or tool.type
        tools.append(tool)
    return tools


def tool_catalog() -> list[dict]:
    """What the panel's agent editor renders: one entry per builtin with a
    human label and the JSON schema of its CONFIG options (not the call
    arguments the model sees — those live on ``Tool.parameters``)."""
    return [
        {
            "type": ttype,
            "label": ttype.replace("_", " ").upper(),
            "description": cls.description,
            "options_schema": cls.options_schema,
        }
        for ttype, cls in BUILTIN_TOOLS.items()
    ]
