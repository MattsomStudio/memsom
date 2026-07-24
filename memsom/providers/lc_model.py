"""The LangChain seam — memsom's provider layer, callable from a graph.

LangGraph needs two things to run an agent: something that answers to
``BaseChatModel`` and something that answers to ``BaseTool``. The obvious move
is to reach for ``ChatOllama`` / ``ChatOpenAI`` and point them at the same
ports — and it is the wrong one. Those clients open their own HTTP connection
to the backend, which means they route *around* everything the provider layer
exists to do: no ``hard_vram_gate`` (so a second model can OOM the 12 GB card
mid-run), no ``procman`` start/stop (so a cold llama.cpp stays cold), no
telemetry, no audit. Worse, llama.cpp would become reachable two different ways
depending on which caller you went through, and the gate would only fire on one
of them.

So instead: ONE wrapper over :meth:`Provider.infer`, and all five adapters
(ollama, llamacpp, vllm, claude, codex) become LangGraph-callable for free. The
shapes already line up, which is why this file is thin:

* ``infer(model, messages, params, sink)`` already takes an OpenAI-style
  message list — the same list LangChain messages trivially flatten into;
* :func:`to_openai_tools` already emits exactly what ``bind_tools`` wants to
  put on the wire;
* canonical ``stats["tool_calls"]`` (``{id, name, arguments}``) maps 1:1 onto
  ``AIMessage.tool_calls`` (``{id, name, args}``) — one key rename.

The one genuine impedance mismatch is that ``infer`` returns STATS, not text:
the reply arrives asynchronously through the :class:`Sink`, because that is what
lets a generation outlive the app that started it. ``_generate`` has to return a
message, so it hands ``infer`` a *tee* sink that forwards every chunk to the
real :class:`AgentFileSink` (durability, the polling frontend) while keeping a
copy locally (the ``AIMessage`` content).

**The run log stays ours.** We do not build the JSONL from LangGraph's
``.stream()``; the ``turn`` / ``tool_call`` / ``tool_result`` events are emitted
from this file, in exactly the field set ``run_tool_loop`` emits today, because
RunMonitor parses them and the shape is a contract. The single change is
additive: ``turn`` gains a ``node`` field so a multi-agent graph can say which
agent is talking. Old readers ignore it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field as PydanticField

# _execute_tool and _LOOP_STRIKES are imported, never reimplemented: the
# two-phase audit and the loop threshold must be the SAME code the legacy
# run_tool_loop uses, or the two runtimes drift apart silently.
from memsom.providers.agents import _LOOP_STRIKES, _execute_tool
from memsom.providers.base import ProviderError, Sink, now
from memsom.providers.tools import Tool, ToolContext, to_openai_tools, truncate_output

__all__ = [
    "RunContext",
    "MemsomChatModel",
    "MemsomTool",
    "to_memsom_messages",
    "from_memsom_messages",
]

#: reserved kwarg name carrying the tool_call_id from BaseTool.run into
#: MemsomTool._run. Dunder-ish on purpose: it must not collide with a real
#: tool argument name coming out of a model.
_CALL_ID_KEY = "__memsom_tool_call_id__"


# ---------------------------------------------------------------------------
# Per-run state
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """Everything one run needs to carry through the graph, in one object.

    LangGraph's own state (``MessagesState``) is the *conversation*; it is
    checkpointable, serializable and shared between nodes. This is the other
    half — the run's plumbing, which is none of those things: an open file
    handle, a mutable counter, a strike tracker. Keeping it out of the graph
    state is deliberate, since a graph state that contains a file handle cannot
    be checkpointed and a strike counter that round-trips through serialization
    is a strike counter that resets.

    One instance per run, handed to every node's model and every tool wrapper,
    so a run's turns are numbered continuously across nodes and the final stats
    line describes the WHOLE graph rather than whichever node happened to speak
    last.
    """

    #: the run's AgentFileSink — tokens and structural events both land here.
    sink: Any
    #: agents/audit.jsonl; passed straight through to ``_execute_tool``.
    audit_path: Path
    #: the compiled limits dict (max_turns, tool_timeout_s, …).
    limits: dict
    #: turns taken so far, across every node. Incremented by _generate.
    turn: int = 0
    #: accumulated counters (prompt_tokens, eval_count, tool_calls) — what
    #: ``_final_stats`` folds into the terminal ``done`` line.
    stats: dict = field(default_factory=dict)
    #: id of the node currently executing; stamped onto the ``turn`` event.
    node_id: str = ""
    #: last executed (name, arguments) signature and how many times it has
    #: repeated back-to-back — the identical-call loop detector.
    last_sig: Optional[str] = None
    strikes: int = 0
    #: when the run began; the run-timeout gate compares against it.
    started: float = field(default_factory=now)
    #: the run's shared scratch dict — one object every agent's state tools
    #: read/write, so a value set by one agent is visible to the next.
    data: dict = field(default_factory=dict)

    def guard(self) -> None:
        """Enforce the run-wide turn ceiling and timeout, at turn entry.

        LangGraph's ``recursion_limit`` bounds NODE transitions, which is not
        the same question as "how many times may a model speak" — one agent
        node can take twenty turns inside its own ReAct subgraph without the
        parent graph advancing a single step. So the two limits the user
        actually configured are enforced here, at the one place every model
        call in the graph passes through, with the same messages and the same
        semantics ``run_tool_loop`` has: checked BEFORE the call, so the ceiling
        is a count of turns taken rather than turns finished.

        A ceiling shared across nodes is the deliberate reading of max_turns in
        a multi-agent graph: it is the run's budget, not each agent's.
        """
        limits = self.limits or {}
        timeout = limits.get("run_timeout_s")
        if timeout and now() - self.started > timeout:
            raise ProviderError(f"run timeout after {timeout}s")
        ceiling = limits.get("max_turns")
        if ceiling and self.turn >= ceiling:
            raise ProviderError(
                f"max turns reached ({ceiling}) without a final answer")

    def next_turn(self) -> int:
        self.turn += 1
        return self.turn

    def accumulate(self, stats: dict) -> None:
        """Fold one node's usage counters into the run-wide totals."""
        for key in ("prompt_tokens", "eval_count"):
            value = (stats or {}).get(key)
            if isinstance(value, (int, float)):
                self.stats[key] = self.stats.get(key, 0) + value

    def count_tool_call(self) -> int:
        self.stats["tool_calls"] = self.stats.get("tool_calls", 0) + 1
        return self.stats["tool_calls"]


class _TeeSink(Sink):
    """Forwards to the real sink, keeps a copy of the ANSWER text.

    ``infer`` returns stats, not text, so ``_generate`` would otherwise have
    nothing to put in the ``AIMessage``. Reasoning is forwarded but never
    accumulated: a reasoning model streams its scratchpad through
    ``reasoning()`` and its reply through ``token()``, and folding the
    scratchpad into ``content`` means the next node reads several hundred
    tokens of deliberation as if the model had said them out loud.
    """

    def __init__(self, inner: Sink) -> None:
        self._inner = inner
        self._buf: list[str] = []

    def token(self, text: str) -> None:
        if text:
            self._buf.append(text)
        self._inner.token(text)

    def reasoning(self, text: str) -> None:
        self._inner.reasoning(text)

    def text(self) -> str:
        return "".join(self._buf)


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------


def _text_of(message: BaseMessage) -> str:
    """Flatten a message's content to a plain string.

    LangChain content is either a string or a list of typed blocks; memsom's
    wire shape is a string. Non-text blocks (images) are dropped rather than
    stringified — no adapter accepts them, and a repr of a base64 blob in the
    prompt is worse than its absence.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def to_memsom_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    """LangChain messages → memsom's canonical OpenAI-style dicts.

    Canonical means ``arguments`` (a parsed dict), not ``args`` — the rename
    happens here and only here, because ``oai.messages_to_openai`` and every
    adapter downstream already agree on ``arguments``.
    """
    out: list[dict] = []
    for message in messages or []:
        text = _text_of(message)
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": text})
        elif isinstance(message, ToolMessage):
            out.append({"role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "name": message.name or "",
                        "content": text})
        elif isinstance(message, AIMessage):
            calls = [
                {"id": tc.get("id") or "",
                 "name": tc.get("name") or "",
                 "arguments": dict(tc.get("args") or {})}
                for tc in (message.tool_calls or [])
            ]
            record = {"role": "assistant", "content": text}
            if calls:
                record["tool_calls"] = calls
            out.append(record)
        elif isinstance(message, HumanMessage):
            out.append({"role": "user", "content": text})
        else:
            # An unknown subclass still has a type string; trust it rather than
            # silently dropping the turn, which would corrupt the transcript.
            out.append({"role": getattr(message, "type", "user") or "user",
                        "content": text})
    return out


def from_memsom_messages(messages: Sequence[dict]) -> list[BaseMessage]:
    """The inverse of :func:`to_memsom_messages`.

    Needed because the entry point into a graph is a memsom-shaped message list
    (the agent's system prompt plus the trigger input) while everything inside
    the graph is LangChain messages. Round-tripping is lossless for the fields
    memsom actually carries.
    """
    out: list[BaseMessage] = []
    for record in messages or []:
        role = (record or {}).get("role")
        content = record.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content,
                                   tool_call_id=record.get("tool_call_id") or "",
                                   name=record.get("name") or None))
        elif role == "assistant":
            calls = [
                {"id": tc.get("id") or "",
                 "name": tc.get("name") or "",
                 "args": dict(tc.get("arguments") or {}),
                 "type": "tool_call"}
                for tc in (record.get("tool_calls") or [])
            ]
            out.append(AIMessage(content=content, tool_calls=calls))
        else:
            out.append(HumanMessage(content=content))
    return out


def _as_openai_tool(spec: Any) -> dict:
    """Render one bound tool into the OpenAI wire shape adapters understand.

    Three inputs reach ``bind_tools`` in practice — a memsom :class:`Tool`, a
    LangChain :class:`BaseTool` (including our own :class:`MemsomTool`), or an
    already-rendered dict — and all three must come out the same, because the
    adapters only ever learned one shape.
    """
    if isinstance(spec, Tool):
        return to_openai_tools([spec])[0]
    if isinstance(spec, dict) and spec.get("type") == "function" \
            and isinstance(spec.get("function"), dict):
        return spec
    return convert_to_openai_tool(spec)


# ---------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------


class MemsomChatModel(BaseChatModel):
    """A ``BaseChatModel`` whose backend is a memsom :class:`Provider`.

    Verified against langchain-core 1.5.1: ``_generate`` and ``_llm_type`` are
    the only abstract members, so those are the only two we owe. ``_stream`` is
    deliberately NOT implemented — token streaming already happens, through the
    sink, into the file the frontend polls; implementing it a second time on the
    LangGraph side would put the same tokens on two paths with two different
    durability stories.
    """

    #: the memsom Provider adapter. Typed Any so pydantic doesn't try to
    #: validate a live adapter object (and so tests can pass a fake).
    adapter: Any
    #: model name as the adapter knows it.
    model: str
    #: adapter params (temperature, transport, ctx, …). ``tools`` is injected
    #: by bind_tools at call time rather than living here.
    params: dict = PydanticField(default_factory=dict)
    #: the shared per-run carrier.
    ctx: Any = None
    #: which canvas node this model belongs to; stamped onto the turn event.
    node_id: str = ""

    @property
    def _llm_type(self) -> str:
        return "memsom"

    def bind_tools(self, tools: Sequence[Any], *,
                   tool_choice: Optional[str] = None,
                   **kwargs: Any) -> Runnable:
        """Bind tools by rendering them into ``params['tools']``.

        ``BaseChatModel.bind_tools`` raises by default, which would make every
        agent node in the graph a text-only model. The override is a rename, not
        a mechanism: the OpenAI ``tools`` array that :func:`to_openai_tools`
        emits is already what every memsom adapter puts on the wire, so binding
        is just carrying that array to ``_generate`` — which ``.bind`` does by
        stuffing it into the call kwargs.
        """
        rendered = [_as_openai_tool(t) for t in (tools or [])]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=rendered, **kwargs)

    def _generate(self, messages: list[BaseMessage],
                  stop: Optional[list[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        ctx = self.ctx
        node = self.node_id or (getattr(ctx, "node_id", "") if ctx else "")
        params = dict(self.params or {})
        if kwargs.get("tools"):
            params["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice") is not None:
            params["tool_choice"] = kwargs["tool_choice"]
        if stop:
            params["stop"] = list(stop)

        if ctx is not None:
            ctx.node_id = node
            # Turn ceiling and run timeout are the model's gate, not the tool's:
            # a text-only agent that never calls a tool must still be bounded.
            ctx.guard()
            turn = ctx.next_turn()
            # Emitted BEFORE the call, not after: the frontend draws the turn
            # separator as soon as it appears, so a long generation shows up as
            # a turn in progress instead of nothing at all.
            ctx.sink.event({"t": "turn", "n": turn, "node": node, "ts": now()})
            tee = _TeeSink(ctx.sink)
        else:
            turn, tee = 0, _TeeSink(_NullSink())

        # A ProviderError propagates untouched — AgentRunner._run catches it and
        # writes the terminal {"t":"error"} line. Wrapping it here would only
        # bury a clean user-facing message inside a LangGraph traceback.
        stats = self.adapter.infer(self.model, to_memsom_messages(messages),
                                   params, tee) or {}

        tool_calls = [
            {"id": call.get("id") or f"tc_{i}",
             "name": call.get("name") or "",
             "args": call.get("arguments") if isinstance(
                 call.get("arguments"), dict) else {"_raw": str(call.get("arguments"))},
             "type": "tool_call"}
            for i, call in enumerate(stats.get("tool_calls") or [], 1)
        ]
        if ctx is not None:
            ctx.accumulate(stats)
            # Identical-call loop detection, per TURN-BATCH — the same shape the
            # legacy run_tool_loop used (a whole turn's calls hashed into one
            # signature), and deliberately HERE rather than in MemsomTool._run.
            # A turn's calls fan out to the tool node in parallel, so a per-call
            # check can neither see a two-call A,B,A,B loop (its signatures never
            # land back-to-back) nor tell a legitimate parallel fan-out of
            # identical calls from a loop (it false-fires on the first turn).
            # _generate is single-threaded and sees the batch, so it can. Only a
            # turn that asks for tools can loop; a plain-text turn is a final
            # answer and touches neither strikes nor last_sig.
            if tool_calls:
                sig = json.dumps([(c["name"], c["args"]) for c in tool_calls],
                                 sort_keys=True, default=str)
                ctx.strikes = ctx.strikes + 1 if sig == ctx.last_sig else 0
                ctx.last_sig = sig
                if ctx.strikes >= _LOOP_STRIKES - 1:
                    raise ProviderError(
                        f"tool loop detected: {_LOOP_STRIKES}x identical call(s)")

        message = AIMessage(content=tee.text(), tool_calls=tool_calls,
                            response_metadata={"node": node, "turn": turn})
        usage = _usage_metadata(stats)
        if usage:
            message.usage_metadata = usage
        return ChatResult(generations=[ChatGeneration(message=message)],
                          llm_output={"stats": stats})


class _NullSink(Sink):
    """Sink for a model driven without a RunContext (unit tests, probes)."""

    def token(self, text: str) -> None:
        return


def _usage_metadata(stats: dict) -> Optional[dict]:
    """Translate memsom counters into LangChain's usage shape, or None.

    Only when the backend actually reported both — a half-filled usage block
    reads as "0 output tokens", which is a lie that ends up in someone's cost
    dashboard."""
    prompt = (stats or {}).get("prompt_tokens")
    output = (stats or {}).get("eval_count")
    if not isinstance(prompt, int) or not isinstance(output, int):
        return None
    return {"input_tokens": prompt, "output_tokens": output,
            "total_tokens": prompt + output}


# ---------------------------------------------------------------------------
# The tool wrapper
# ---------------------------------------------------------------------------


class MemsomTool(BaseTool):
    """A ``BaseTool`` wrapping exactly one memsom :class:`Tool`.

    Verified against langchain-core 1.5.1: ``_run`` is the only abstract member.

    Everything that makes a tool call *safe* — the two-phase pending/result
    audit, unknown-tool and failure handling as a MESSAGE rather than a crash,
    the output cap — already lives in :func:`_execute_tool` and
    :func:`truncate_output`. This class adds nothing to that path; it delegates
    verbatim and then emits the two sink events. Reimplementing any of it would
    mean the legacy loop and the graph runtime could disagree about what got
    audited, which is the one thing an audit log may not do.
    """

    #: the wrapped memsom Tool.
    memsom_tool: Any = None
    #: the shared per-run carrier (turn number, limits, audit path, strikes).
    run_ctx: Any = None
    #: when true, the call PAUSES for a human APPROVE/DENY before it executes.
    require_approval: bool = False

    def __init__(self, tool: Tool, ctx: RunContext,
                 require_approval: bool = False, **kwargs: Any) -> None:
        # args_schema is the tool's raw JSON Schema dict, NOT a generated
        # pydantic model. langchain-core 1.5.1 accepts a dict here and, when it
        # sees one, passes the model's arguments through untouched — which is
        # what we want: the memsom tools validate their own arguments and
        # report a bad one back to the MODEL as a ToolError message. A pydantic
        # gate in front would raise instead, turning a recoverable model mistake
        # into a dead run.
        super().__init__(
            name=tool.name or tool.type,
            description=tool.description,
            args_schema=dict(tool.parameters or {"type": "object",
                                                 "properties": {}}),
            memsom_tool=tool,
            run_ctx=ctx,
            require_approval=require_approval,
            **kwargs,
        )

    def _to_args_and_kwargs(self, tool_input: Any,
                            tool_call_id: Optional[str]) -> tuple:
        """Smuggle the model's tool_call_id down into ``_run``.

        ``_run`` is handed the model's arguments and nothing else — langchain
        only injects the call id into a *pydantic* arg schema (via
        ``InjectedToolCallId``), and ours are deliberately raw JSON Schema
        dicts. Without the id the ``tool_call``/``tool_result`` events would
        carry a synthetic number that no reader could join back to the
        assistant message that asked for the call. It rides in the kwargs
        rather than on the RunContext because a graph may execute a turn's
        tool calls in parallel, and a shared "current call id" slot is a race.
        """
        args, kwargs = super()._to_args_and_kwargs(tool_input, tool_call_id)
        if tool_call_id:
            kwargs[_CALL_ID_KEY] = tool_call_id
        return args, kwargs

    def _run(self, *args: Any, **kwargs: Any) -> str:
        call_id = kwargs.pop(_CALL_ID_KEY, "")
        arguments = dict(kwargs)
        ctx: RunContext = self.run_ctx
        limits = ctx.limits

        # Human-in-the-loop gate. interrupt() is the FIRST thing here, before any
        # counter, event or audit, and that ordering is load-bearing: on the
        # first pass it raises GraphInterrupt (the run pauses, this function
        # unwinds having done nothing), and on resume LangGraph RE-RUNS the whole
        # tool node from the top — so anything with a side effect before
        # interrupt() would fire twice. With it first, the resume path runs the
        # count/events/execution below exactly once. The payload is what
        # run_graph turns into the awaiting_approval event the UI reads.
        if self.require_approval:
            from langgraph.types import interrupt
            decision = interrupt({"kind": "approval", "tool": self.name,
                                  "arguments": arguments,
                                  "id": call_id or None})
            if str(decision).lower() != "approve":
                # Denied: record it (the gate working IS the security event) and
                # hand the model a plain result it can react to and move on.
                nth = ctx.count_tool_call()
                cid = call_id or f"tc_{nth}"
                self._audit_denied(ctx, cid, arguments)
                ctx.sink.event({"t": "tool_call", "turn": ctx.turn, "id": cid,
                                "name": self.name, "arguments": arguments,
                                "ts": now()})
                denied = "DENIED by user: the tool was not executed."
                ctx.sink.event({"t": "tool_result", "turn": ctx.turn, "id": cid,
                                "name": self.name, "ok": False, "output": denied,
                                "bytes": len(denied.encode("utf-8")),
                                "truncated": False, "elapsed_s": 0.0})
                return denied
            # Approved: fall through and execute exactly like an ungated call.

        # Loop detection lives in MemsomChatModel._generate, over the whole
        # turn's batch of calls — not here. The tool node fans a turn's calls
        # out across a thread pool, so a per-call check is both racy and blind
        # to a two-call loop; the batch check upstream is single-threaded and
        # sees every call in the turn at once.
        nth = ctx.count_tool_call()
        call_id = call_id or f"tc_{nth}"
        ctx.sink.event({"t": "tool_call", "turn": ctx.turn, "id": call_id,
                        "name": self.name, "arguments": arguments, "ts": now()})

        tool_ctx = ToolContext(
            audit_path=ctx.audit_path,
            timeout_s=limits["tool_timeout_s"],
            max_output_bytes=limits["max_tool_output_bytes"],
            shared=ctx.data,
        )
        started = now()
        output, ok = _execute_tool(self.memsom_tool, self.name, arguments,
                                   tool_ctx, ctx.audit_path,
                                   available=[self.name])
        text, truncated = truncate_output(output, limits["max_tool_output_bytes"])
        ctx.sink.event({"t": "tool_result", "turn": ctx.turn, "id": call_id,
                        "name": self.name, "ok": ok, "output": text,
                        "bytes": len(output.encode("utf-8", "ignore")),
                        "truncated": truncated,
                        "elapsed_s": round(now() - started, 3)})
        return text

    def _audit_denied(self, ctx: RunContext, call_id: str,
                      arguments: dict) -> None:
        """Record a human-refused call in the same audit log a real call uses.

        A denied tool never reaches ``_execute_tool``, so it would otherwise
        leave no trace — but "the human stopped shell(rm -rf)" is exactly the
        event the audit exists to hold. Same redaction discipline as the two-
        phase path: tool name and stringified/clipped arguments only, never the
        prompt or model text."""
        from memsom.providers.handlers import _audit
        _audit(ctx.audit_path, {
            "action": "tool", "tool": self.name, "id": call_id,
            "arguments": {k: str(v)[:200] for k, v in arguments.items()},
            "result": "refused-by-user",
        })
