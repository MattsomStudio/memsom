"""The Tool contract — what an agent is allowed to do, one class per ability.

A chat model on its own can only emit text. Giving an agent *abilities* (fetch a
URL, search the web, recall memory, read a file, run a command) means giving the
runner something concrete to execute when the model asks — and that something
must be small, uniform, and auditable. So every ability implements the SAME tiny
surface: a JSON-schema description the model sees, and a ``run`` that takes the
model's arguments and returns a string. Nothing else.

Design rules that keep this honest:

* **Tools are pure executors.** They do not write audit lines — the runner
  writes the two-phase intent/result lines (same pattern as
  :func:`memsom.providers.handlers._audit`) around every call, so a tool cannot
  forget to be audited.
* **Failures are messages, not crashes.** A tool-level failure raises
  :class:`ToolError`; ``str(exc)`` is fed back to the model as the tool result,
  so keep it clean (no secrets, no stack noise) — exactly like
  :class:`memsom.providers.base.ProviderError`.
* **Output is bounded.** Model context is finite; every tool caps what it
  returns via :func:`truncate_output` / :class:`ToolContext` limits so one
  runaway ``cat`` cannot flood a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ToolError(Exception):
    """Any tool failure surfaced back to the model. ``str(exc)`` is the
    user/model-facing message — keep it clean (no secrets, no stack noise)."""


@dataclass
class ToolContext:
    """Per-run limits and plumbing handed to every tool call by the runner.

    ``audit_path`` is the agents' audit log (``agents/audit.jsonl``, two-phase
    intent/result lines) — carried here so the RUNNER can write around the
    call; tools themselves never touch it.
    """

    audit_path: Path
    timeout_s: int
    max_output_bytes: int


class Tool:
    """Base class every builtin tool subclasses.

    Class attributes describe the ability (``type``, ``description``,
    ``parameters`` — the JSON schema the model sees, OpenAI ``parameters``
    shape). ``name`` is the per-agent instance name: it defaults to ``type``
    but the registry overwrites it from the agent spec, so one agent can carry
    two differently-configured instances of the same builtin.
    """

    #: stable builtin type id (e.g. "http_fetch"); set on the subclass.
    type: str = ""
    #: instance name, unique per agent (may be suffixed); set by the registry.
    name: str = ""
    #: what the model reads to decide when to call this.
    description: str = ""
    #: JSON schema for the call arguments (OpenAI "parameters" shape).
    parameters: dict = {"type": "object", "properties": {}}
    #: JSON schema for the CONFIG options (what the panel edits, not the model).
    options_schema: dict = {"type": "object", "properties": {}}

    def __init__(self, options: dict) -> None:
        self.options = options or {}
        if not self.name:
            self.name = self.type

    def run(self, arguments: dict, ctx: ToolContext) -> str:
        """Execute with the model-supplied *arguments* under *ctx* limits.
        Returns the string fed back to the model. Raises :class:`ToolError`
        on any tool-level failure."""
        raise NotImplementedError


def to_openai_tools(tools: list[Tool]) -> list[dict]:
    """Render *tools* in the OpenAI ``tools`` wire shape every backend we talk
    to (llama.cpp server, vLLM, Ollama's OpenAI endpoint) understands."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def truncate_output(text: str, max_bytes: int) -> tuple[str, bool]:
    """Cap *text* at *max_bytes* of UTF-8 without splitting a multibyte
    character. Returns ``(text, truncated)`` — callers decide how to mark the
    cut (tools append a visible marker so the model knows it saw a prefix)."""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text, False
    # errors="ignore" drops the partial trailing character a byte-slice can
    # leave behind, keeping the result valid UTF-8.
    return raw[:max_bytes].decode("utf-8", errors="ignore"), True
