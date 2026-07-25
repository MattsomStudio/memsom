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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional, Sequence

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
from langchain_core.tools import BaseTool, InjectedToolCallId
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from pydantic import Field as PydanticField

# _execute_tool, _infer_with_deadline and _LOOP_STRIKES are imported, never
# reimplemented: the two-phase audit, the run-budget deadline and the loop
# threshold must be the SAME code the legacy run_tool_loop uses, or the two
# runtimes drift apart silently.
from memsom.providers.agents import (
    _LOOP_STRIKES, _execute_tool, _infer_with_deadline,
)
from memsom.providers.base import ProviderError, Sink, now
from memsom.providers.tools import Tool, ToolContext, to_openai_tools, truncate_output

__all__ = [
    "RunContext",
    "MemsomChatModel",
    "MemsomTool",
    "HandoffTool",
    "to_memsom_messages",
    "from_memsom_messages",
]

#: reserved kwarg name carrying the tool_call_id from BaseTool.run into
#: MemsomTool._run. Dunder-ish on purpose: it must not collide with a real
#: tool argument name coming out of a model.
_CALL_ID_KEY = "__memsom_tool_call_id__"

#: how hard :meth:`RunContext.sync_data` tries to land its atomic replace, and
#: how long it waits between tries. Small: the contention it rides out lasts as
#: long as somebody else's open file handle, and this sits on the tool-call path.
_REPLACE_ATTEMPTS = 5
_REPLACE_RETRY_S = 0.01


def _atomic_json_write(path: Path, payload: dict) -> None:
    """Write *payload* tmp-then-replace. Best-effort; OSError is swallowed.

    Extracted from :meth:`RunContext.sync_data` when the exactly-once record
    gained a second sidecar, because the retry loop below is not boilerplate —
    it is a measured Windows behaviour, and a second hand-rolled copy of it is a
    second place to get it subtly wrong.

    ``os.replace`` onto a path another handle currently has OPEN fails with
    ``PermissionError`` (WinError 5): in a probe with three reader threads, 2935
    of 3000 replaces failed. Swallowing that means the write is silently gone,
    and the readers are not hypothetical — a fan-out has several tool threads
    writing here, and a machine like this one also has Syncthing and a virus
    scanner touching a run directory. The failure is transient by construction
    (the other handle closes in microseconds), so a few short retries turn
    "sometimes loses the last write" into "effectively never", for a worst case
    of 40ms.

    ``default=str`` mirrors :meth:`StateGet.run`'s existing dumps tolerance: a
    value that is not JSON-serializable round-trips as its ``str()``. Deliberately
    not "fixed" into a pickle — these are JSON sidecars, and a pickle in a run
    directory is an execution primitive we do not want.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except OSError:
        return
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            tmp.replace(path)
            return
        except OSError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                return
            time.sleep(_REPLACE_RETRY_S)


def _args_sig(arguments: dict) -> str:
    """A stable signature for a tool call's arguments.

    Sorted keys so two dicts that differ only in insertion order compare equal —
    a model's arguments arrive through JSON parsing and their order is not
    meaningful, so treating it as meaningful would turn a genuine replay into a
    miss and re-execute the call this record exists to stop.
    """
    try:
        return json.dumps(arguments or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):        # pragma: no cover - default=str eats these
        return repr(sorted((arguments or {}).items()))


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

    **Shared across THREADS since fan-out.** Several agent nodes can now run at
    once, and they all hold this same object, so every counter here is a shared
    mutable. Two rules follow, and both are structural rather than defensive:

    * anything that increments a counter does it under ``_lock`` — a read--
      modify-write on ``turn`` from two threads loses a turn, and a lost turn is
      a lost ``max_turns`` enforcement, not just a wrong number;
    * anything PER-AGENT is keyed by node id (``node_turn``, ``node_loop``)
      rather than kept in one scalar. The loop detector is the sharp case: with
      one shared ``last_sig`` a sibling's interleaved call resets the strike
      counter of the node that was actually looping, and two siblings making the
      same call would trip each other. The state belongs to the node, so it is
      stored per node.
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
    #: id of the node LAST seen executing. Under fan-out several nodes write
    #: this and the value is whichever wrote last — a documented inert race,
    #: deliberately not "fixed": every MemsomChatModel and MemsomTool in a graph
    #: is constructed with its own ``node_id``, so the fallback read of this
    #: field is unreachable on the real path. Do not start relying on it.
    node_id: str = ""
    #: when the run began; the run-timeout gate compares against it.
    started: float = field(default_factory=now)
    #: the run's shared scratch dict — one object every agent's state tools
    #: read/write, so a value set by one agent is visible to the next.
    data: dict = field(default_factory=dict)
    #: what this RUN may touch (`memsom.providers.scope`). On the run rather than
    #: on a node because a scope that stopped at a node boundary would be undone
    #: by the first handoff or fan-out. Empty means unrestricted, which is every
    #: graph saved before scope existed.
    scope: dict = field(default_factory=dict)
    #: where :meth:`sync_data` persists ``data``, or None for a run that cannot
    #: pause (no checkpoint path, a throwaway, a unit test). See sync_data.
    data_path: Optional[Path] = None
    #: tool_call_id -> what that call returned, for calls that ALREADY RAN in
    #: this run. Resuming after an approval gate re-enters the node the gate
    #: paused inside, so every tool call batched into the SAME turn as the gated
    #: one is asked for a second time (MEASURED: 2 executions, same batch; 1 when
    #: the call was in an earlier turn). Without this the second execution is
    #: real — a scan, a POST, a payload firing again with nobody having approved
    #: the repeat.
    calls: dict = field(default_factory=dict)
    #: sibling of ``data_path``, keyed the same way. A SEPARATE file rather than
    #: a key inside the scratchpad because `load_data` does `data.update(stored)`
    #: — a reserved key would surface in the scratchpad `state_get` reads.
    calls_path: Optional[Path] = None
    #: True only when this run_graph call is resuming a paused run. The record is
    #: consulted ONLY then, which removes the whole false-positive class by
    #: construction: a run that never paused cannot be replaying anything.
    replaying: bool = False
    #: guards the sidecar file AND the decision to write it. An RLock rather
    #: than a Lock because :meth:`MemsomTool._run` holds it across the
    #: "did anything change?" comparison and the :meth:`sync_data` call that
    #: follows, and sync_data takes it again — with a plain Lock that is a
    #: deadlock, not a race.
    _data_lock: Any = field(default_factory=threading.RLock, repr=False,
                            compare=False)
    #: node id → the turn number that node is currently on. ``turn`` is the
    #: run-global latest, which mis-attributes a tool event the moment a sibling
    #: advances the counter while this node is mid-call; ``turn_of`` reads here.
    node_turn: dict = field(default_factory=dict)
    #: node id → (last tool-batch signature, consecutive strikes). Per node, so
    #: one agent's repetition cannot be reset — or falsely tripped — by a
    #: sibling running concurrently.
    node_loop: dict = field(default_factory=dict)
    #: provider id → a Semaphore(1) that must be held while that engine is
    #: generating. Populated by ``lc_runtime.run_graph`` for engines that hold
    #: local VRAM; empty for a graph whose engines are all remote. Two nodes on
    #: ONE 12 GB card are not parallelism, they are an OOM with extra steps.
    engine_locks: dict = field(default_factory=dict)
    #: guards every counter above. Not the sidecar — that has its own, because
    #: the sidecar lock is held across a file write and this one must never be.
    _lock: Any = field(default_factory=threading.Lock, repr=False,
                       compare=False)

    def load_data(self) -> None:
        """Rehydrate ``data`` from the sidecar written by a previous segment.

        Why a file and not a graph channel: a resume builds a FRESH RunContext,
        so anything an agent stored in ``data`` before an approval pause was
        simply gone — the v0.18.0 caveat. The two graph-native fixes were both
        measured dead. ``Command(graph=Command.PARENT)`` returned from a tool
        silently truncates the enclosing agent node's remaining turns in
        memsom's manually-invoked-subgraph topology, and ``InjectedState`` needs
        the pydantic arg schema :class:`MemsomTool` deliberately does not have
        (raw JSON Schema is what lets a bad argument come back to the model as a
        message instead of raising). So the scratchpad lives entirely OUTSIDE
        LangGraph's channel/checkpoint system: no state schema, no reducer, no
        checkpoint migration.

        A corrupt or unreadable sidecar is swallowed: the run continues with an
        empty scratchpad, which is exactly the behaviour it had before this
        existed. Losing shared state is a degradation; refusing to resume the
        run over it would be a regression.
        """
        if self.data_path is None:
            return
        with self._data_lock:
            try:
                raw = Path(self.data_path).read_text(encoding="utf-8")
            except OSError:
                return
            try:
                stored = json.loads(raw)
            except (ValueError, TypeError):
                return
            if isinstance(stored, dict):
                # update, not replace: `data` is the same object every
                # ToolContext.shared already points at.
                self.data.update(stored)

    def sync_data(self) -> None:
        """Persist ``data`` to the sidecar, atomically. Best-effort.

        Written tmp-then-``replace`` so a reader (or a crash) can never see a
        half-written file — the same discipline the checkpoint DB gets, for the
        same reason: this file is read back on resume and a torn one would
        resurrect the run with garbage.

        ``default=str`` mirrors :meth:`StateGet.run`'s existing dumps tolerance:
        a value the model stored that is not JSON-serializable round-trips a
        pause as its ``str()``. That is deliberately not "fixed" into a pickle —
        the scratchpad is a JSON scratchpad, and a pickle in a run directory is
        an execution primitive we do not want.

        OSError is swallowed: a failed persist costs cross-pause survival, not
        the run.

        The replace is RETRIED, and that is not belt-and-braces — it is a
        measured Windows behaviour with real consequences. ``os.replace`` onto a
        path another handle currently has OPEN fails with ``PermissionError``
        (WinError 5): in a probe with three reader threads, 2935 of 3000
        replaces failed. Swallowing that means the write is silently gone, and
        the readers in question are not hypothetical — a fan-out has several
        tool threads writing this file, and a machine like this one also has
        Syncthing and a virus scanner touching a run directory. The failure is
        transient by construction (the other handle closes in microseconds), so
        a few short retries turn "sometimes loses the last scratchpad write"
        into "effectively never", for a worst case of 40ms on a path that only
        runs when a tool actually touched the scratchpad.
        """
        if self.data_path is None:
            return
        with self._data_lock:
            # dict(self.data) first: the lock serializes the WRITERS of the
            # file, but nothing stops another tool thread inserting a key into
            # the live scratchpad while json walks it. The shallow copy is one
            # GIL-atomic C-level operation, so the dump never sees the dict
            # resize under it. (Nested values are still shared — the discipline
            # the state tools follow is one whole value per key, not in-place
            # mutation of a value another agent holds.)
            _atomic_json_write(Path(self.data_path), dict(self.data))

    def load_calls(self) -> None:
        """Rehydrate the exactly-once record. Same discipline as `load_data`.

        A corrupt or unreadable record is swallowed, and the consequence is worth
        stating: the run falls back to TODAY's behaviour, which is that a replayed
        call executes twice. A degradation, not a regression — and the alternative,
        refusing to resume over an unreadable sidecar, would strand a paused run
        over a file that only ever existed to make it safer.
        """
        if self.calls_path is None:
            return
        with self._data_lock:
            try:
                stored = json.loads(
                    Path(self.calls_path).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                return
            if isinstance(stored, dict):
                self.calls.update(stored)

    def record_call(self, call_id: str, name: str, arguments: dict,
                    output: str, ok: bool) -> None:
        """Remember that this tool call ran, and what it returned."""
        if not call_id:
            return
        with self._data_lock:
            self.calls[call_id] = {"name": name, "args": _args_sig(arguments),
                                   "output": output, "ok": ok}
            if self.calls_path is not None:
                _atomic_json_write(Path(self.calls_path), dict(self.calls))

    def recall_call(self, call_id: str, name: str, arguments: dict):
        """What this call returned last time, or None to execute it.

        Returns ``None`` on a fresh run no matter what the record says — see
        ``replaying``. That is the whole false-positive defence: a run that never
        resumed cannot possibly be replaying, so it never consults this.

        Matched on id AND name AND arguments rather than id alone. **The stated
        limit:** a model that reuses a tool_call_id inside one run for a call with
        the same name and the same arguments gets the first result back instead of
        a second execution. Real providers mint a unique id per call, so that is a
        model pathology rather than a case to design for — and given the choice,
        answering a duplicate id with the duplicate's own result is the more
        defensible half of the trade.
        """
        if not self.replaying or not call_id:
            return None
        with self._data_lock:
            hit = self.calls.get(call_id)
        if not hit:
            return None
        if hit.get("name") != name or hit.get("args") != _args_sig(arguments):
            return None
        return hit

    def begin_turn(self, node: str = "") -> int:
        """Open one model turn: check the budget, take a number, announce it.

        Enforces the run-wide turn ceiling and timeout at turn ENTRY. LangGraph's
        ``recursion_limit`` bounds NODE transitions, which is not the same
        question as "how many times may a model speak" — one agent node can take
        twenty turns inside its own ReAct subgraph without the parent graph
        advancing a single step. So the two limits the user actually configured
        are enforced here, at the one place every model call in the graph passes
        through, with the same messages and the same semantics
        ``run_tool_loop`` has: checked BEFORE the call, so the ceiling counts
        turns taken rather than turns finished. A ceiling shared across nodes is
        the deliberate reading of max_turns in a multi-agent graph: it is the
        run's budget, not each agent's.

        **The check, the increment and the ``turn`` line are ONE acquisition**,
        and that is the whole reason this is a single method instead of the
        ``guard()`` + ``next_turn()`` + ``sink.event()`` trio it replaces. Two
        threads allocating 5 and 6 and then appending in the other order put
        turn 6 above turn 5 in the file — and RunMonitor renders the file in
        order, so the transcript shows the run going backwards. Not theoretical:
        with the append moved outside the lock, five runs of six threads
        produced 33 out-of-order turn lines; with it inside, zero. The append
        does real I/O, which is a yield point, which is why this one is easy to
        hit rather than merely possible.

        The ceiling check belongs inside the same acquisition for the second
        reason: it is a read-modify-write on a budget, so two threads passing it
        together spend the last turn twice.
        """
        with self._lock:
            limits = self.limits or {}
            timeout = limits.get("run_timeout_s")
            if timeout and now() - self.started > timeout:
                raise ProviderError(f"run timeout after {timeout}s")
            ceiling = limits.get("max_turns")
            if ceiling and self.turn >= ceiling:
                raise ProviderError(
                    f"max turns reached ({ceiling}) without a final answer")
            self.turn += 1
            turn = self.turn
            if node:
                self.node_turn[node] = turn
            # Emitted BEFORE the call, not after: the frontend draws the turn
            # separator as soon as it appears, so a long generation shows up as
            # a turn in progress instead of nothing at all.
            self.sink.event({"t": "turn", "n": turn, "node": node, "ts": now()})
            return turn

    def turn_of(self, node: str = "") -> int:
        """The turn number *node* is currently on — what its events must carry.

        ``turn`` is the run-global latest, which was the same thing right up
        until two nodes could run at once: a sibling opening turn 7 while this
        node is still executing turn 6's tool call would stamp 7 onto that
        call's ``tool_result``, and the reader joining it back to its
        ``tool_call`` would land on the wrong agent's turn."""
        if not node:
            return self.turn
        return self.node_turn.get(node, self.turn)

    def check_loop(self, node: str, sig: str) -> None:
        """Per-node identical-batch loop detection. Raises when it trips.

        Two properties the shared scalar it replaces could not have. The
        signature is a whole TURN'S batch (a two-call A,B,A,B loop is invisible
        per call, and a legitimate parallel fan-out of identical calls false-
        fires per call) — that part is unchanged from v0.17.0. What is new is
        that the strike state is keyed by NODE: a sibling's interleaved batch
        would otherwise reset the counter of the node that is genuinely stuck,
        and two siblings legitimately making the same call would strike each
        other out.

        The raise happens outside the lock deliberately — nothing else needs to
        be atomic with it, and raising while holding a lock other threads are
        waiting on is a habit worth not forming.
        """
        with self._lock:
            last, strikes = self.node_loop.get(node, (None, 0))
            strikes = strikes + 1 if sig == last else 0
            self.node_loop[node] = (sig, strikes)
        if strikes >= _LOOP_STRIKES - 1:
            raise ProviderError(
                f"tool loop detected: {_LOOP_STRIKES}x identical call(s)")

    def accumulate(self, stats: dict) -> None:
        """Fold one node's usage counters into the run-wide totals."""
        with self._lock:
            for key in ("prompt_tokens", "eval_count"):
                value = (stats or {}).get(key)
                if isinstance(value, (int, float)):
                    self.stats[key] = self.stats.get(key, 0) + value

    def count_tool_call(self) -> int:
        with self._lock:
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

    *node* is stamped onto each ``tok`` line so an interleaved concurrent stream
    stays attributable — with two agents generating at once the run log is one
    interleaved sequence, and without the tag a reader cannot tell which agent
    said what. Only forwarded to a sink that declares ``accepts_node``; anything
    else (a ListSink, a test double, the null sink) gets the one-argument call
    it has always got.
    """

    def __init__(self, inner: Sink, node: str = "") -> None:
        self._inner = inner
        self._buf: list[str] = []
        self._node = node if getattr(inner, "accepts_node", False) else ""

    def token(self, text: str) -> None:
        if text:
            self._buf.append(text)
        if self._node:
            self._inner.token(text, self._node)
        else:
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

    A tool may also carry its OWN rendering in ``openai_schema`` and skip the
    generic path. :class:`HandoffTool` is the reason: its pydantic arg schema
    exists solely so LangGraph can inject the graph state, and
    ``convert_to_openai_tool`` drops ``json_schema_extra`` (measured), which is
    the only way pydantic could have expressed the branch ENUM. A hand-written
    wire shape gives the model the same hard enum the ``decide`` router's
    synthetic tool gets, without giving up state injection.
    """
    rendered = getattr(spec, "openai_schema", None)
    if isinstance(rendered, dict) and rendered:
        return rendered
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
            # One call, one lock: the budget check, the turn number and the
            # `turn` line are atomic together, so file order is turn order even
            # with a sibling node generating on another thread.
            turn = ctx.begin_turn(node)
            tee = _TeeSink(ctx.sink, node)
        else:
            turn, tee = 0, _TeeSink(_NullSink())

        # A ProviderError propagates untouched — AgentRunner._run catches it and
        # writes the terminal {"t":"error"} line. Wrapping it here would only
        # bury a clean user-facing message inside a LangGraph traceback.
        #
        # With a ctx the call goes through the shared deadline helper, so
        # ``run_timeout_s`` bounds the CALL and not just the gap before it (see
        # _infer_with_deadline) and an agent may opt into retries. Without one
        # there is no run to budget — the probe/unit-test path calls the adapter
        # exactly as it always did.
        if ctx is not None:
            stats = _infer_with_deadline(
                self.adapter, self.model, to_memsom_messages(messages), params,
                tee,
                run_timeout_s=(ctx.limits or {}).get("run_timeout_s"),
                started=ctx.started,
                max_attempts=(ctx.limits or {}).get("infer_retries", 1))
        else:
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
            # _generate sees a whole batch at once, so it can. Only a turn that
            # asks for tools can loop; a plain-text turn is a final answer and
            # touches no strike state. The strike state is PER NODE — a sibling
            # running concurrently must not be able to reset, or trip, this
            # node's counter (see RunContext.check_loop).
            if tool_calls:
                sig = json.dumps([(c["name"], c["args"]) for c in tool_calls],
                                 sort_keys=True, default=str)
                ctx.check_loop(node, sig)

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


def _normalize_decision(decision: Any) -> tuple:
    """One human answer to an approval gate → ``(verdict, edited_arguments)``.

    Two wire shapes, because approving is not the only useful answer. A bare
    string is the legacy vocabulary (``"approve"`` executes, anything else
    denies — unchanged, including the deliberate breadth of "anything else").
    A dict ``{"decision": "edit", "arguments": {…}}`` is the human saying "run
    it, but with THESE arguments instead" — the review case where the tool call
    was nearly right and denying it just makes the model guess again. A dict
    round-trips through ``Command(resume=…)`` into ``interrupt()`` untouched
    (verified against langgraph 1.2.9), so no encoding is needed.

    A malformed edit — the word without an ``arguments`` object — DENIES rather
    than falling back to the original call. The HTTP handler rejects that
    payload before it ever gets here, so reaching this branch means something
    upstream is confused, and a security gate that guesses when it is confused
    is not a gate. Fail closed.
    """
    if isinstance(decision, dict):
        verdict = str(decision.get("decision") or "").strip().lower()
        if verdict == "edit":
            arguments = decision.get("arguments")
            if isinstance(arguments, dict):
                return "edit", dict(arguments)
            return "deny", None
        return ("approve" if verdict == "approve" else "deny"), None
    return ("approve" if str(decision).lower() == "approve" else "deny"), None


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

    **Never return ``Command(graph=Command.PARENT)`` from here.** In memsom's
    topology the ReAct agent is a subgraph invoked BY HAND inside a parent node
    (``lc_runtime._agent_node``), and a parent-directed Command raised from a
    tool unwinds straight through that ``subgraph.invoke`` — silently discarding
    whatever turns the enclosing agent node had left to take. Measured, not
    theorised. It is therefore useless as a state write-through mechanism (which
    is why the shared scratchpad persists through :meth:`RunContext.sync_data`
    instead), and the one place it IS correct is a handoff tool, where ending
    the node's turn is the entire point.
    """

    #: the wrapped memsom Tool.
    memsom_tool: Any = None
    #: the shared per-run carrier (turn numbers, limits, audit path, strikes).
    run_ctx: Any = None
    #: when true, the call PAUSES for a human APPROVE/DENY before it executes.
    require_approval: bool = False
    #: which canvas agent node owns this tool instance. Every event this class
    #: emits is stamped with THAT node's turn rather than the run-global latest,
    #: because a sibling agent can advance the run counter while this call is
    #: still in flight. Empty for a tool driven outside a graph (a unit test),
    #: where ``turn_of("")`` falls back to the global counter — which is correct,
    #: because with no nodes there is nothing to mis-attribute to.
    node_id: str = ""

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
            verdict, edited = _normalize_decision(decision)
            if verdict == "edit":
                # Reassigned HERE, before the counter, the sink events and
                # _execute_tool — so the run log and the audit both record what
                # ACTUALLY ran rather than what was proposed. The proposal is
                # not lost: it is already on disk in the awaiting_approval
                # event, which is the line a reviewer compares against.
                arguments = edited
            elif verdict != "approve":
                # Denied: record it (the gate working IS the security event) and
                # hand the model a plain result it can react to and move on.
                nth = ctx.count_tool_call()
                cid = call_id or f"tc_{nth}"
                turn = ctx.turn_of(self.node_id)
                self._audit_denied(ctx, cid, arguments)
                ctx.sink.event({"t": "tool_call", "turn": turn, "id": cid,
                                "name": self.name, "arguments": arguments,
                                "ts": now()})
                denied = "DENIED by user: the tool was not executed."
                ctx.sink.event({"t": "tool_result", "turn": turn, "id": cid,
                                "name": self.name, "ok": False, "output": denied,
                                "bytes": len(denied.encode("utf-8")),
                                "truncated": False, "elapsed_s": 0.0})
                return denied
            # Approved: fall through and execute exactly like an ungated call.

        # EXACTLY ONCE. The comment above is true of THIS tool and says nothing
        # about its siblings: a gate protects the call it sits on, and the other
        # calls the model batched into the same turn have no interrupt() in front
        # of them at all. Resuming re-enters the node the gate paused inside, so
        # those siblings are asked a second time — MEASURED at 2 executions for a
        # same-batch call, against 1 when the call was in an earlier turn. Which
        # meant a scan, a POST or a payload could fire twice, the repeat approved
        # by nobody.
        #
        # Placed after the gate and before the counter on purpose: the gated tool
        # itself never ran (it interrupted first), so it must fall through and
        # execute normally, while a suppressed replay must not take a tool-call
        # number or emit a second tool_call/tool_result pair — that duplicate
        # pair is the visible half of the bug.
        seen = ctx.recall_call(call_id, self.name, arguments)
        if seen is not None:
            # Loud. Fixing the double execution and leaving no trace of the
            # replay would trade one invisible behaviour for another; a reader
            # has to be able to see BOTH that the call ran once and that a repeat
            # was refused.
            self._audit_replay(ctx, call_id, arguments)
            ctx.sink.event({"t": "replay", "id": call_id, "name": self.name,
                            "node": self.node_id,
                            "turn": ctx.turn_of(self.node_id), "ts": now()})
            return str(seen.get("output") or "")

        # Loop detection lives in MemsomChatModel._generate, over the whole
        # turn's batch of calls — not here. The tool node fans a turn's calls
        # out across a thread pool, so a per-call check is both racy and blind
        # to a two-call loop; the batch check upstream is single-threaded and
        # sees every call in the turn at once.
        nth = ctx.count_tool_call()
        call_id = call_id or f"tc_{nth}"
        # This node's turn, not the run's. See the node_id field: with two agent
        # nodes in flight the global counter belongs to whoever spoke last.
        turn = ctx.turn_of(self.node_id)
        ctx.sink.event({"t": "tool_call", "turn": turn, "id": call_id,
                        "name": self.name, "arguments": arguments, "ts": now()})

        tool_ctx = ToolContext(
            audit_path=ctx.audit_path,
            timeout_s=limits["tool_timeout_s"],
            max_output_bytes=limits["max_tool_output_bytes"],
            shared=ctx.data,
            scope=getattr(ctx, "scope", None),
        )
        started = now()
        # Snapshot around the call so the scratchpad is persisted only by the
        # tools that actually touch it: state_set writes a sidecar, http_fetch
        # never does. A file write per tool call would be pure overhead on the
        # overwhelmingly common case, and this is the only place that can tell
        # the difference without asking every Tool to declare itself.
        before = dict(ctx.data)
        output, ok = _execute_tool(self.memsom_tool, self.name, arguments,
                                   tool_ctx, ctx.audit_path,
                                   available=[self.name])
        # The comparison and the persist are ONE acquisition, so a sibling tool
        # thread cannot slip a write in between "nothing changed" and the
        # decision not to save. Writes to `data` ITSELF are unguarded and stay
        # that way: StateSet stores one whole value under one key, which is a
        # single GIL-atomic dict assignment — it cannot tear, and the worst a
        # concurrent one costs is a redundant sidecar write. The discipline that
        # makes that true is "one whole value per key", never in-place mutation
        # of a value another agent is holding. (RLock, so sync_data re-entering
        # is not a deadlock.)
        with ctx._data_lock:
            if ctx.data != before:
                ctx.sync_data()
        text, truncated = truncate_output(output, limits["max_tool_output_bytes"])
        ctx.sink.event({"t": "tool_result", "turn": turn, "id": call_id,
                        "name": self.name, "ok": ok, "output": text,
                        "bytes": len(output.encode("utf-8", "ignore")),
                        "truncated": truncated,
                        "elapsed_s": round(now() - started, 3)})
        # Recorded AFTER the call returned, so only a call that actually ran can
        # ever be suppressed later. A record written before execution would turn
        # a crash mid-call into a call that never happens and never retries.
        ctx.record_call(call_id, self.name, arguments, text, ok)
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

    def _audit_replay(self, ctx: RunContext, call_id: str,
                      arguments: dict) -> None:
        """Record a suppressed replay. The runner writes it, never the tool.

        Without this line the audit would show one ``pending``/``ok`` pair and
        nothing else — indistinguishable from a run that was never resumed at
        all. The interesting fact is not just that the call ran once; it is that
        the runtime was asked to run it AGAIN and declined, which is precisely
        what somebody reading an audit after an approval gate wants to know.
        """
        from memsom.providers.handlers import _audit
        _audit(ctx.audit_path, {
            "action": "tool", "tool": self.name, "id": call_id,
            "arguments": {k: str(v)[:200] for k, v in arguments.items()},
            "result": "suppressed-replay",
        })


# ---------------------------------------------------------------------------
# The handoff tool
# ---------------------------------------------------------------------------


def _handoff_args_schema(names: Sequence[str]) -> type:
    """The pydantic arg model a handoff tool needs, built per router.

    This is the ONE tool in memsom with a pydantic ``args_schema`` instead of a
    raw JSON Schema dict, and the exception is narrow enough to be worth stating:
    ``InjectedState`` is an ANNOTATION protocol. LangGraph's tool node reads the
    schema's annotations to decide what to inject, so a raw dict — which is what
    lets a user tool report a bad argument back to the model as a message rather
    than raising — cannot receive the graph state at all (measured). Without the
    state, a handoff cannot carry the agent's own transcript forward, and the
    next agent starts blind. See :meth:`HandoffTool._run`.

    The safety that motivated raw dicts elsewhere survives anyway: a model that
    sends the wrong shape gets LangGraph's validation error back AS A TOOL
    MESSAGE and takes another turn (measured against langgraph 1.2.9) — it does
    not raise out of the run. And ``branch`` is deliberately typed ``str`` rather
    than a ``Literal``, so a hallucinated branch name reaches :meth:`_run` and
    comes back as a sentence naming the real branches, instead of a pydantic
    traceback the model has to decode.
    """
    from langgraph.prebuilt import InjectedState

    class HandoffArgs(BaseModel):
        branch: str = PydanticField(
            description="Name of the branch to take. One of: "
                        + ", ".join(names))
        message: Optional[str] = PydanticField(
            default=None,
            description="Optional briefing handed to whoever runs next.")
        #: injected by LangGraph, never by the model — both are stripped from
        #: the schema the model is shown.
        state: Annotated[dict, InjectedState]
        tool_call_id: Annotated[str, InjectedToolCallId]

    return HandoffArgs


class HandoffTool(BaseTool):
    """The synthetic tool a ``handoff`` router binds into the feeding agent.

    Why it exists: ``decide`` mode asks a SECOND model call "which way now?"
    after the agent has already finished talking. A handoff asks the agent to
    say where it is going as part of the turn it was taking anyway — one
    inference instead of two, and the choice is made by the model that actually
    holds the context, not by a stateless referee reading the transcript.

    **This is the one place ``Command(graph=Command.PARENT)`` is correct**, and
    it is correct for exactly the reason :class:`MemsomTool` forbids it: raising
    a parent-directed Command unwinds straight out of the manual
    ``subgraph.invoke`` in ``lc_runtime._agent_node``, abandoning whatever turns
    the agent had left. For a state write-through that is data loss. For a
    handoff it is the entire point — the agent is done, by its own declaration.

    The unwinding has one consequence that had to be paid for rather than
    documented away: the node's ``run_node`` never returns, so nothing it
    produced would reach the parent's message thread. Measured — the next agent
    saw the trigger input and nothing else, silently breaking the "one shared
    conversation" invariant that ``decide`` and ``match`` both keep. So the
    Command carries the transcript itself, sliced out of the injected state at
    ``prior``: everything this node added, plus the tool's own result message,
    plus the optional briefing.
    """

    #: the router's branch dicts — used to describe the choice to the model.
    branches: list = []
    #: branch name → parent-graph node id (or END).
    target_map: dict = {}
    #: the router's canvas node id; stamped onto the ``route`` event so the
    #: monitor renders a handoff exactly like the other two modes.
    router_node_id: str = ""
    #: the shared per-run carrier.
    run_ctx: Any = None
    #: hand-rendered wire shape (see :func:`_as_openai_tool`).
    openai_schema: dict = {}
    #: how many messages were already in the thread when this node was entered.
    #: Set by ``run_node`` immediately before it invokes the subgraph, so the
    #: tool can carry forward what the NODE produced without re-sending what the
    #: parent already holds. Slicing rather than leaning on ``add_messages``
    #: id-dedup is the same call ``run_node`` itself makes for its normal return.
    prior: int = 0
    #: when true, the handoff PAUSES for a human APPROVE/DENY before it routes.
    #: This was the one tool call in memsom a human could not intercept, and it
    #: is the highest-leverage one to be able to: handing off is how an agent
    #: moves work into a DIFFERENT agent with DIFFERENT tools, so it is the
    #: single action that changes what everything downstream is allowed to do.
    require_approval: bool = False
    #: the AGENT node this tool belongs to, for turn attribution on its audit
    #: line — the same reason :class:`MemsomTool` carries one.
    node_id: str = ""

    def __init__(self, *, name: str, branches: list, target_map: dict,
                 router_node_id: str, ctx: RunContext, **kwargs: Any) -> None:
        names = [str(b.get("name") or "") for b in branches]
        super().__init__(
            name=name,
            description=_handoff_description(name, branches),
            args_schema=_handoff_args_schema(names),
            openai_schema=_handoff_openai_schema(
                name, _handoff_description(name, branches), names),
            branches=list(branches),
            target_map=dict(target_map),
            router_node_id=router_node_id,
            run_ctx=ctx,
            **kwargs,
        )

    def _run(self, branch: str = "", state: Any = None, tool_call_id: str = "",
             message: Optional[str] = None, **kwargs: Any) -> Any:
        ctx: RunContext = self.run_ctx
        if branch not in self.target_map:
            # A plain string, never an exception. An unknown branch is a model
            # mistake, and the cheapest correct response to a model mistake is
            # to tell it what the real options were and let it take another
            # turn — the same contract every memsom tool honours.
            return (f"unknown branch {branch!r}. Valid branches: "
                    + ", ".join(sorted(self.target_map)) + ".")

        # The gate, and it is FIRST for the same load-bearing reason MemsomTool's
        # is: on resume LangGraph re-runs the tool node from the top, so anything
        # before interrupt() happens twice. This tool has no re-entrancy guard at
        # all — it never touches RunContext.calls — so a gate placed after the
        # route event below would emit TWO route events for one fork, and a fork
        # that reads as having happened twice is worse than one nobody approved.
        if self.require_approval:
            from langgraph.types import interrupt
            decision = interrupt({"kind": "approval", "tool": self.name,
                                  "arguments": {"branch": branch,
                                                "message": message or ""},
                                  "id": tool_call_id or None})
            verdict, _edited = _normalize_decision(decision)
            if verdict != "approve":
                # A plain string, exactly like the unknown-branch path above, and
                # deliberately NOT a forced else-branch: "do not go there" is not
                # "go here instead", and picking a destination the human also did
                # not choose would be inventing a decision out of a refusal. The
                # agent keeps its turn; if it gives up, run_node's existing else
                # fallback already covers "never handed off".
                #
                # The honest consequence: a model that immediately re-calls
                # handoff asks the human again. Bounded by max_turns, and better
                # than silently overriding the agent's plan.
                self._audit(ctx, tool_call_id, branch, "refused-by-user")
                return (f"handoff to {branch!r} was REFUSED by the user. You "
                        "still hold this turn — do something else, or stop.")

        # The existing route event, verbatim — same keys, same order, just a
        # third value for `mode`. RunMonitor renders `mode` as free text and
        # `read_since` never learned the vocabulary, so a handoff needs no new
        # event type and no reader change.
        if ctx is not None:
            ctx.sink.event({"t": "route", "router": self.router_node_id,
                            "branch": branch, "mode": "handoff", "ts": now()})
        self._audit(ctx, tool_call_id, branch, f"handoff:{branch}")

        carried = list((state or {}).get("messages") or [])[self.prior:]
        carried.append(ToolMessage(content=f"handing off to {branch}",
                                   tool_call_id=tool_call_id or "handoff",
                                   name=self.name))
        if message:
            # A SystemMessage rather than more tool output: the briefing is
            # addressed to whoever runs NEXT, not to the agent that just wrote
            # it, and a ToolMessage reads as a reply to the caller.
            carried.append(SystemMessage(
                content=f"Handoff from the previous agent: {message}"))

        from langgraph.types import Command
        return Command(goto=self.target_map[branch],
                       update={"messages": carried}, graph=Command.PARENT)

    def _audit(self, ctx: RunContext, call_id: str, branch: str,
               result: str) -> None:
        """Record the handoff. The runner writes it; the tool never does.

        A handoff used to write NOTHING here — one ``route`` event and that was
        the whole record. Every other tool call in the runtime leaves a two-phase
        intent/result pair, so the one call that decides which agent runs next,
        with which tools, was the single one an audit could not see. Gating it
        without also recording it would have produced the odd shape where a
        REFUSED handoff is on disk and a taken one is not.

        Defensive because the audit path is: a run must not die because
        housekeeping could not write. `_execute_tool`'s intent line is allowed to
        kill a run — that one gates the action — but this is a record of
        something already decided.
        """
        if ctx is None:
            return
        from memsom.providers.handlers import _audit
        try:
            _audit(ctx.audit_path, {
                "action": "handoff", "tool": self.name,
                "id": call_id or "handoff",
                "arguments": {"branch": str(branch)[:200],
                              "router": self.router_node_id},
                "result": result,
            })
        except OSError:
            pass


def _handoff_description(name: str, branches: list) -> str:
    """What the model is told about the fork it is standing at."""
    catalogue = "\n".join(
        f"- {b.get('name')}: {b.get('when') or 'no description given'}"
        for b in branches)
    return (
        "Hand the conversation to whoever should continue it. Call this as "
        "your FINAL action, on its own — any other tool call you make in the "
        "same turn will not run, because the handoff ends your turn "
        "immediately.\nBranches:\n" + catalogue)


def _handoff_openai_schema(name: str, description: str,
                           names: Sequence[str]) -> dict:
    """The wire shape, hand-written so ``branch`` keeps a hard enum.

    Same structure as ``lc_runtime._route_tool`` — that similarity is the point.
    Whether the model is picking a branch for a ``decide`` router or for itself,
    it should be looking at the same kind of constrained choice."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "enum": list(names),
                               "description": "The branch to hand off to."},
                    "message": {"type": "string",
                                "description": "Optional briefing for whoever "
                                               "runs next."},
                },
                "required": ["branch"],
            },
        },
    }
