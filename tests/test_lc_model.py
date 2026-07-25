"""The LangChain seam, pinned: does a graph node behave like the loop does?

``lc_model.py`` is the only place where memsom's provider contract and
LangChain's chat/tool contracts touch, so it is the only place the two can
drift apart. Two things are worth pinning and nothing else is:

1. **Translation is lossless.** ``arguments`` ⇄ ``args``, tool turns, tool
   results. Get one of these wrong and the failure is a model that quietly
   stops seeing what it just did.
2. **The run log is byte-compatible with what RunMonitor already parses.** The
   ``turn`` / ``tool_call`` / ``tool_result`` field sets are a contract with a
   frontend we are not rebuilding, so they are asserted key-by-key rather than
   "contains roughly the right stuff".

Nothing here touches a network, a real adapter or a real tool: a FakeAdapter
returns scripted stats and an EchoTool returns a pure function of its
arguments, so every assertion is a fact about this file's code.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)

from memsom.providers.agents import _LOOP_STRIKES
from memsom.providers.base import ProviderError, now
from memsom.providers.lc_model import (
    MemsomChatModel,
    MemsomTool,
    RunContext,
    from_memsom_messages,
    to_memsom_messages,
)
from memsom.providers.session import AgentFileSink
from memsom.providers.tools.base import Tool, ToolError
from memsom.providers.tools.builtins import StateSet


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class FakeAdapter:
    """One scripted turn per entry: ``(text, thinking, stats)``.

    Text is pushed through ``sink.token`` and thinking through
    ``sink.reasoning`` exactly as a real adapter does, so the tee sink is
    exercised on both channels. Past the end of the script the last entry
    repeats — "the model keeps asking for the same tool" is a one-entry script.
    """

    def __init__(self, script: list) -> None:
        self.script = script
        self.calls: list = []          # (model, messages, params)

    def infer(self, model, messages, params, sink):
        self.calls.append((model, json.loads(json.dumps(messages, default=str)),
                           dict(params)))
        text, thinking, stats = self.script[min(len(self.calls) - 1,
                                                len(self.script) - 1)]
        if thinking:
            sink.reasoning(thinking)
        if text:
            sink.token(text)
        return dict(stats)


class BoomAdapter:
    """An engine that fails the way a real one does — ProviderError, not
    an internal exception."""

    def infer(self, model, messages, params, sink):
        raise ProviderError("llamacpp: connection refused on 127.0.0.1:8080")


class EchoTool(Tool):
    type = "echo"
    name = "echo"
    description = "echoes its arguments back"
    parameters = {"type": "object",
                  "properties": {"x": {"type": "string"}},
                  "required": ["x"]}

    def run(self, arguments: dict, ctx) -> str:
        return "ECHO:" + str(arguments.get("x", ""))


class BigTool(EchoTool):
    """Returns more than any sane limit — the truncation path."""

    def run(self, arguments: dict, ctx) -> str:
        return "z" * 5000


class AngryTool(EchoTool):
    def run(self, arguments: dict, ctx) -> str:
        raise ToolError("nope")


class FlakyThenOkAdapter:
    """Fails the first *n* calls, then answers. A cold engine, in miniature.

    Records every ``params`` it was handed so the injected deadline is
    observable — the whole point of the fix is a key the adapters already read.
    """

    def __init__(self, failures: int = 1, text: str = "recovered",
                 delay: float = 0.0) -> None:
        self.failures = failures
        self.text = text
        self.delay = delay             # a call that outlives its deadline
        self.calls: list = []          # params, one entry per attempt

    def infer(self, model, messages, params, sink):
        self.calls.append(dict(params))
        if self.delay:
            time.sleep(self.delay)
        if len(self.calls) <= self.failures:
            raise ProviderError("ollama: connection reset by peer")
        sink.token(self.text)
        return {}


class StreamThenBoomAdapter:
    """Puts a token on the wire and THEN dies — the one shape a retry must
    never touch, because the run log already has half an answer in it."""

    def __init__(self) -> None:
        self.calls: list = []

    def infer(self, model, messages, params, sink):
        self.calls.append(dict(params))
        sink.token("half an ans")
        raise ProviderError("llamacpp: stream closed mid-token")


LIMITS = {"max_turns": 8, "tool_timeout_s": 5,
          "max_tool_output_bytes": 64, "run_timeout_s": 60}


@pytest.fixture()
def ctx(tmp_path: Path) -> RunContext:
    sink = AgentFileSink(tmp_path / "run.jsonl")
    return RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                      limits=dict(LIMITS))


def events(ctx: RunContext) -> list[dict]:
    """Every JSONL line the run has written so far."""
    ctx.sink._fh.flush()
    text = Path(ctx.sink.path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def audit_lines(ctx: RunContext) -> list[dict]:
    if not Path(ctx.audit_path).is_file():
        return []
    text = Path(ctx.audit_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


# ---------------------------------------------------------------------------
# message translation
# ---------------------------------------------------------------------------


def test_to_memsom_messages_covers_every_role_including_a_tool_turn():
    lc = [
        SystemMessage(content="you are a thing"),
        HumanMessage(content="fetch it"),
        AIMessage(content="on it", tool_calls=[
            {"id": "tc_1", "name": "echo", "args": {"x": "hi"},
             "type": "tool_call"}]),
        ToolMessage(content="ECHO:hi", tool_call_id="tc_1", name="echo"),
        AIMessage(content="done"),
    ]
    assert to_memsom_messages(lc) == [
        {"role": "system", "content": "you are a thing"},
        {"role": "user", "content": "fetch it"},
        # canonical shape: "arguments" (a dict), NOT langchain's "args"
        {"role": "assistant", "content": "on it",
         "tool_calls": [{"id": "tc_1", "name": "echo",
                         "arguments": {"x": "hi"}}]},
        {"role": "tool", "tool_call_id": "tc_1", "name": "echo",
         "content": "ECHO:hi"},
        {"role": "assistant", "content": "done"},
    ]


def test_message_conversion_round_trips_through_both_directions():
    memsom = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "calling",
         "tool_calls": [{"id": "tc_1", "name": "echo",
                         "arguments": {"x": "hi"}}]},
        {"role": "tool", "tool_call_id": "tc_1", "name": "echo",
         "content": "ECHO:hi"},
    ]
    assert to_memsom_messages(from_memsom_messages(memsom)) == memsom


def test_multimodal_content_blocks_flatten_to_their_text():
    msg = HumanMessage(content=[{"type": "text", "text": "look: "},
                                {"type": "image_url", "image_url": {"url": "x"}},
                                {"type": "text", "text": "this"}])
    assert to_memsom_messages([msg]) == [{"role": "user",
                                          "content": "look: this"}]


# ---------------------------------------------------------------------------
# the chat model
# ---------------------------------------------------------------------------


def test_generate_returns_the_sink_text_as_content_and_maps_tool_calls(ctx):
    adapter = FakeAdapter([("thinking about it", "", {
        "prompt_tokens": 11, "eval_count": 3,
        "tool_calls": [{"id": "call_abc", "name": "echo",
                        "arguments": {"x": "hi"}}]})])
    model = MemsomChatModel(adapter=adapter, model="m", ctx=ctx,
                            node_id="agent:1")

    reply = model.invoke([HumanMessage(content="go")])

    # infer returns STATS ONLY; the text exists purely because the tee sink
    # caught it on its way to the file.
    assert reply.content == "thinking about it"
    assert reply.tool_calls == [
        {"name": "echo", "args": {"x": "hi"}, "id": "call_abc",
         "type": "tool_call"}]
    assert reply.usage_metadata == {"input_tokens": 11, "output_tokens": 3,
                                    "total_tokens": 14}
    # counters accumulate onto the RUN, not the node
    assert ctx.stats == {"prompt_tokens": 11, "eval_count": 3}
    assert ctx.turn == 1


def test_turn_event_carries_the_node_id_and_keeps_n_meaning_what_it_meant(ctx):
    adapter = FakeAdapter([("a", "", {}), ("b", "", {})])
    MemsomChatModel(adapter=adapter, model="m", ctx=ctx,
                    node_id="agent:one").invoke([HumanMessage(content="1")])
    MemsomChatModel(adapter=adapter, model="m", ctx=ctx,
                    node_id="agent:two").invoke([HumanMessage(content="2")])

    turns = [e for e in events(ctx) if e.get("t") == "turn"]
    assert [(t["n"], t["node"]) for t in turns] == [(1, "agent:one"),
                                                    (2, "agent:two")]
    assert all(set(t) == {"t", "n", "node", "ts"} for t in turns)


def test_reasoning_reaches_the_sink_but_never_the_message_content(ctx):
    adapter = FakeAdapter([("the answer", "deliberating at length", {})])
    reply = MemsomChatModel(adapter=adapter, model="m",
                            ctx=ctx).invoke([HumanMessage(content="go")])

    assert reply.content == "the answer"
    assert "deliberating" not in reply.content
    kinds = [(e["t"], e.get("text")) for e in events(ctx) if e["t"] in
             ("tok", "think")]
    assert kinds == [("think", "deliberating at length"), ("tok", "the answer")]


def test_bind_tools_lands_in_params_tools_in_the_openai_wire_shape(ctx):
    adapter = FakeAdapter([("ok", "", {})])
    model = MemsomChatModel(adapter=adapter, model="m", ctx=ctx)

    # memsom Tool in, and the same Tool wrapped as a MemsomTool: both must
    # render identically, because the adapters only learned one shape.
    for tool in (EchoTool({}), MemsomTool(EchoTool({}), ctx)):
        model.bind_tools([tool]).invoke([HumanMessage(content="go")])
        assert adapter.calls[-1][2]["tools"] == [{
            "type": "function",
            "function": {"name": "echo",
                         "description": "echoes its arguments back",
                         "parameters": EchoTool.parameters},
        }]


def test_provider_error_from_infer_propagates_unchanged(ctx):
    model = MemsomChatModel(adapter=BoomAdapter(), model="m", ctx=ctx)
    with pytest.raises(ProviderError) as exc:
        model.invoke([HumanMessage(content="go")])
    # AgentRunner writes str(exc) into the terminal {"t":"error"} line, so it
    # has to survive the trip through LangChain's callback machinery intact.
    assert str(exc.value) == "llamacpp: connection refused on 127.0.0.1:8080"


# ---------------------------------------------------------------------------
# the tool wrapper
# ---------------------------------------------------------------------------


def test_tool_writes_the_two_phase_audit_pair_and_the_two_sink_events(ctx):
    ctx.turn = 4
    tool = MemsomTool(EchoTool({}), ctx)

    out = tool.invoke({"type": "tool_call", "id": "call_abc", "name": "echo",
                       "args": {"x": "hi"}})

    assert out.content == "ECHO:hi"
    call, result = [e for e in events(ctx)
                    if e["t"] in ("tool_call", "tool_result")]
    assert call == {"t": "tool_call", "turn": 4, "id": "call_abc",
                    "name": "echo", "arguments": {"x": "hi"}, "ts": call["ts"]}
    assert set(result) == {"t", "turn", "id", "name", "ok", "output", "bytes",
                           "truncated", "elapsed_s"}
    assert (result["turn"], result["id"], result["name"], result["ok"],
            result["output"], result["bytes"], result["truncated"]) == \
        (4, "call_abc", "echo", True, "ECHO:hi", 7, False)

    # _execute_tool, verbatim: pending BEFORE the call, outcome after.
    assert [(a["action"], a["tool"], a["result"]) for a in audit_lines(ctx)] == [
        ("tool", "echo", "pending"), ("tool", "echo", "ok")]


def test_tool_failure_is_a_message_to_the_model_not_a_dead_run(ctx):
    out = MemsomTool(AngryTool({}), ctx).invoke({"x": "hi"})
    assert out == "tool error: nope"
    result = [e for e in events(ctx) if e["t"] == "tool_result"][0]
    assert result["ok"] is False
    assert audit_lines(ctx)[-1]["result"] == "failed: nope"


def test_truncated_output_is_capped_at_the_limit_and_flagged(ctx):
    MemsomTool(BigTool({}), ctx).invoke({"x": "hi"})

    result = [e for e in events(ctx) if e["t"] == "tool_result"][0]
    assert result["truncated"] is True
    assert len(result["output"]) == LIMITS["max_tool_output_bytes"]
    # `bytes` is the size BEFORE the cut — that is what tells a reader how much
    # of the tool's answer the model never saw.
    assert result["bytes"] == 5000


# Loop detection is per TURN-BATCH, in _generate — NOT per call in the tool
# wrapper. A turn's calls fan out to the tool node in parallel, so the whole
# batch is the unit that repeats; a batch signature is what run_tool_loop always
# used. These drive it through _generate (the seam that sees the batch), the way
# a real graph turn does.


def _batch(*names_and_args) -> dict:
    """A stats dict whose tool_calls are the given (name, args) pairs."""
    return {"tool_calls": [
        {"id": f"tc_{i}", "name": n, "arguments": a}
        for i, (n, a) in enumerate(names_and_args, 1)]}


def _turn(model, ctx) -> None:
    """One graph turn: bind nothing, just infer over a trivial history."""
    model._generate([HumanMessage(content="go")])


def test_a_repeated_tool_batch_trips_loop_detection(ctx):
    # Every turn asks for the same single call — the classic stuck loop.
    script = [("", None, _batch(("echo", {"x": "hi"})))]
    model = MemsomChatModel(adapter=FakeAdapter(script), model="m", ctx=ctx)
    _turn(model, ctx)                     # strike 0
    _turn(model, ctx)                     # strike 1
    with pytest.raises(ProviderError) as exc:
        _turn(model, ctx)                 # strike 2 → trips
    assert str(exc.value) == (
        f"tool loop detected: {_LOOP_STRIKES}x identical call(s)")


def test_a_repeated_PARALLEL_batch_trips_loop_detection(ctx):
    # The case a per-call check misses: a turn that emits TWO calls, repeated.
    # Per-call the signatures interleave A,B,A,B and never land back-to-back;
    # per-batch [A,B] == [A,B] is one signature that repeats and trips.
    script = [("", None, _batch(("echo", {"x": "a"}), ("echo", {"x": "b"})))]
    model = MemsomChatModel(adapter=FakeAdapter(script), model="m", ctx=ctx)
    _turn(model, ctx)
    _turn(model, ctx)
    with pytest.raises(ProviderError):
        _turn(model, ctx)


def test_one_turn_of_identical_parallel_calls_does_NOT_trip(ctx):
    # The false-positive a per-call check produced: three identical calls in a
    # SINGLE turn is legitimate fan-out, not a loop. One batch = one strike.
    script = [("", None, _batch(("echo", {"x": "hi"}),
                                ("echo", {"x": "hi"}),
                                ("echo", {"x": "hi"}))),
              ("done", None, {})]          # next turn is a plain answer
    model = MemsomChatModel(adapter=FakeAdapter(script), model="m", ctx=ctx)
    _turn(model, ctx)                      # must not raise
    _turn(model, ctx)                      # plain-text turn, resets nothing
    # The strike state moved from one scalar on the run to one entry per NODE,
    # so siblings running concurrently cannot reset or trip each other. This
    # model carries no node id, so its bucket is keyed "" — same assertion,
    # relocated: (last signature, consecutive strikes).
    assert ctx.node_loop[""][1] == 0


def test_a_differing_batch_resets_the_strike_counter(ctx):
    # a,a,b,a,a — the repeats are never three-in-a-row, so it never trips.
    script = [("", None, _batch(("echo", {"x": "a"}))),
              ("", None, _batch(("echo", {"x": "a"}))),
              ("", None, _batch(("echo", {"x": "b"}))),
              ("", None, _batch(("echo", {"x": "a"}))),
              ("", None, _batch(("echo", {"x": "a"})))]
    model = MemsomChatModel(adapter=FakeAdapter(script), model="m", ctx=ctx)
    for _ in script:
        _turn(model, ctx)                  # no raise across the whole sequence


def test_tool_call_ids_from_the_model_survive_onto_the_tool_events(ctx):
    """The id the MODEL chose has to reach the tool_result line, or the
    transcript can't join a result back to the assistant turn that asked.

    This is the whole reason ``_to_args_and_kwargs`` is overridden: the
    AIMessage carries the id, LangGraph's ToolNode passes it to ``run()``, and
    without the override it dies there instead of reaching ``_run``."""
    adapter = FakeAdapter([("", "", {"tool_calls": [
        {"id": "call_xyz", "name": "echo", "arguments": {"x": "hi"}}]})])
    reply = MemsomChatModel(adapter=adapter, model="m",
                            ctx=ctx).invoke([HumanMessage(content="go")])
    assert reply.tool_calls[0]["id"] == "call_xyz"

    MemsomTool(EchoTool({}), ctx).invoke(reply.tool_calls[0])
    assert [e["id"] for e in events(ctx)
            if e["t"] in ("tool_call", "tool_result")] == ["call_xyz",
                                                           "call_xyz"]


def test_a_call_with_no_id_still_numbers_itself_like_the_legacy_loop(ctx):
    """Invoked outside a model turn there is no id to inherit; the transcript
    must still be well-formed, and ``tc_<n>`` is what run_tool_loop emits."""
    tool = MemsomTool(EchoTool({}), ctx)
    tool.invoke({"x": "a"})
    tool.invoke({"x": "b"})
    assert [e["id"] for e in events(ctx) if e["t"] == "tool_call"] == ["tc_1",
                                                                      "tc_2"]


# ---------------------------------------------------------------------------
# the shared-scratchpad sidecar
# ---------------------------------------------------------------------------
#
# A resume builds a FRESH RunContext, so `data` used to come back empty on the
# far side of an approval pause. The sidecar is what carries it across; these
# pin the file mechanics, and test_provider_agents pins the pause it exists for.


def test_run_context_sync_and_load_data_roundtrip(tmp_path):
    sink = AgentFileSink(tmp_path / "run.jsonl")
    path = tmp_path / "shared" / "run_1.json"

    writer = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                        limits=dict(LIMITS), data_path=path)
    writer.data["finding"] = "42"
    writer.data["nested"] = {"a": [1, 2]}
    writer.sync_data()
    assert path.is_file()          # the dir is created on the way

    # a different object, the way resume gets one
    reader = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                        limits=dict(LIMITS), data_path=path)
    assert reader.data == {}
    reader.load_data()
    assert reader.data == {"finding": "42", "nested": {"a": [1, 2]}}


def test_load_data_is_a_no_op_without_a_path_or_a_file(tmp_path):
    sink = AgentFileSink(tmp_path / "run.jsonl")
    # no path at all (an uncheckpointed run): both calls must be harmless
    plain = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                       limits=dict(LIMITS))
    plain.data["x"] = 1
    plain.sync_data()
    plain.load_data()
    assert plain.data == {"x": 1}
    assert not (tmp_path / "shared").exists()

    # a path whose file was never written, and one holding garbage: the run
    # continues with an EMPTY scratchpad rather than refusing to resume.
    missing = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                         limits=dict(LIMITS),
                         data_path=tmp_path / "shared" / "nope.json")
    missing.load_data()
    assert missing.data == {}

    corrupt_path = tmp_path / "shared" / "corrupt.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{not json at all", encoding="utf-8")
    corrupt = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                         limits=dict(LIMITS), data_path=corrupt_path)
    corrupt.load_data()
    assert corrupt.data == {}


def test_memsom_tool_persists_shared_data_only_when_it_changes(tmp_path):
    """A tool that touches the scratchpad writes the sidecar; one that doesn't
    must not — a file write per tool call would be pure overhead on the
    overwhelmingly common case."""
    sink = AgentFileSink(tmp_path / "run.jsonl")
    path = tmp_path / "shared" / "run_1.json"
    ctx = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                     limits=dict(LIMITS), data_path=path)

    MemsomTool(EchoTool({}), ctx).invoke({"x": "hi"})
    assert not path.exists()

    MemsomTool(StateSet({}), ctx).invoke({"key": "finding", "value": "42"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"finding": "42"}


def test_sync_data_survives_concurrent_writers_without_corrupting_the_file(
        tmp_path):
    """Insurance for the parallel fan-out stage: several tool threads calling
    sync_data at once must never leave a half-written file on disk, because the
    next resume reads it back as the whole scratchpad."""
    sink = AgentFileSink(tmp_path / "run.jsonl")
    path = tmp_path / "shared" / "run_1.json"
    ctx = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                     limits=dict(LIMITS), data_path=path)

    # parties = the writers, the reader, AND this thread — everyone releases
    # together so the writes genuinely overlap.
    n, rounds = 8, 25
    start = threading.Barrier(n + 2, timeout=30)
    torn: list = []

    def writer(i: int) -> None:
        start.wait()
        for r in range(rounds):
            ctx.data[f"k{i}"] = r
            ctx.sync_data()

    def reader() -> None:
        start.wait()
        for _ in range(rounds * n):
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue        # mid-replace on Windows; not a tearing failure
            if not raw:
                continue
            try:
                json.loads(raw)
            except ValueError:
                torn.append(raw)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    start.wait()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    assert torn == []
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final == {f"k{i}": rounds - 1 for i in range(n)}


# ---------------------------------------------------------------------------
# the run-budget deadline + opt-in retry
# ---------------------------------------------------------------------------
#
# run_timeout_s used to be a turn-ENTRY gate only: checked before a turn and
# never again, so one infer that hung for an hour walked straight through a
# 120s budget. The fix injects what is LEFT of the budget as params['timeout'],
# which every adapter already honours. The retry rides along on the same helper
# and is off by default; its load-bearing rule is that nothing is ever retried
# after a token reached the run log.


@pytest.fixture()
def no_retry_delay(monkeypatch):
    """Collapse the inter-attempt pause. The delay is real (1.0s) and worth
    having in production; paying it once per retry test is not."""
    import memsom.providers.agents as agents_mod
    monkeypatch.setattr(agents_mod, "_INFER_RETRY_DELAY_S", 0.0)


def test_flaky_infer_retries_once_and_succeeds(ctx, no_retry_delay):
    ctx.limits["infer_retries"] = 2
    adapter = FlakyThenOkAdapter(failures=1)

    reply = MemsomChatModel(adapter=adapter, model="m", ctx=ctx,
                            node_id="agent:1").invoke([HumanMessage(content="go")])

    assert reply.content == "recovered"
    assert len(adapter.calls) == 2
    # One turn, one answer. The failed attempt emitted nothing, so the run log
    # a human reads shows a clean turn rather than a stutter.
    stream = [(e["t"], e.get("text")) for e in events(ctx)
              if e["t"] in ("turn", "tok")]
    assert stream == [("turn", None), ("tok", "recovered")]
    assert ctx.turn == 1


def test_no_retry_after_anything_streamed(ctx, no_retry_delay):
    """The safety rule the whole retry rests on: text on the wire means the
    attempt is final, no matter how many retries were asked for."""
    ctx.limits["infer_retries"] = 3
    adapter = StreamThenBoomAdapter()

    with pytest.raises(ProviderError) as exc:
        MemsomChatModel(adapter=adapter, model="m",
                        ctx=ctx).invoke([HumanMessage(content="go")])

    assert len(adapter.calls) == 1
    assert str(exc.value) == "llamacpp: stream closed mid-token"
    # exactly one partial answer in the log — not two glued together
    assert [e.get("text") for e in events(ctx) if e["t"] == "tok"] == \
        ["half an ans"]


def test_retries_are_bounded_by_the_configured_count(ctx, no_retry_delay):
    ctx.limits["infer_retries"] = 3
    adapter = FlakyThenOkAdapter(failures=99)

    with pytest.raises(ProviderError):
        MemsomChatModel(adapter=adapter, model="m",
                        ctx=ctx).invoke([HumanMessage(content="go")])
    assert len(adapter.calls) == 3


def test_injected_timeout_is_capped_to_remaining_run_budget(ctx):
    """The deadline can only ever TIGHTEN a call. A run with 1s of its budget
    left must not hand the adapter the ten-minute default."""
    ctx.limits["run_timeout_s"] = 5
    ctx.started = now() - 4          # 4s in, 1s left
    adapter = FlakyThenOkAdapter(failures=0)

    MemsomChatModel(adapter=adapter, model="m",
                    ctx=ctx).invoke([HumanMessage(content="go")])

    injected = adapter.calls[0]["timeout"]
    assert 0 < injected <= 1.0


def test_a_call_that_already_asked_for_less_keeps_its_own_timeout(ctx):
    adapter = FlakyThenOkAdapter(failures=0)
    MemsomChatModel(adapter=adapter, model="m", ctx=ctx,
                    params={"timeout": 3}).invoke([HumanMessage(content="go")])
    # budget says 60s remain; the call asked for 3 — the tighter one wins
    assert adapter.calls[0]["timeout"] == 3


def test_a_call_that_outlives_the_budget_is_reported_as_the_run_timeout(ctx):
    """THE bug, reproduced: the turn-entry gate passes (nothing has elapsed),
    then one call runs past the whole budget. Before this, that surfaced as a
    raw socket error from whichever adapter happened to be holding the call —
    true, but it reads as an engine fault when the real answer is "you gave
    this run 0.2 seconds". The entry gate's wording is reused verbatim so one
    condition has one message."""
    ctx.limits["run_timeout_s"] = 0.2
    ctx.started = now()              # entry gate passes: nothing elapsed yet
    adapter = FlakyThenOkAdapter(failures=99, delay=0.35)

    with pytest.raises(ProviderError) as exc:
        MemsomChatModel(adapter=adapter, model="m",
                        ctx=ctx).invoke([HumanMessage(content="go")])

    assert str(exc.value) == "run timeout after 0.2s"
    assert isinstance(exc.value.__cause__, ProviderError)
    assert len(adapter.calls) == 1   # a spent budget is never retried
    # and the deadline the adapter was HANDED was the remainder, not 600s —
    # a real adapter would have aborted itself at 0.2s.
    assert adapter.calls[0]["timeout"] <= 0.2


def test_the_turn_entry_gate_still_fires_before_the_call(ctx):
    """The deadline is additive: the cumulative many-fast-turns case is still
    caught at turn entry, without the adapter being touched at all."""
    ctx.limits["run_timeout_s"] = 5
    ctx.started = now() - 30         # budget long gone before this turn
    adapter = FlakyThenOkAdapter(failures=0)

    with pytest.raises(ProviderError) as exc:
        MemsomChatModel(adapter=adapter, model="m",
                        ctx=ctx).invoke([HumanMessage(content="go")])

    assert str(exc.value) == "run timeout after 5s"
    assert adapter.calls == []


def test_a_ctx_less_call_still_reaches_the_adapter_untouched():
    """The probe path (no RunContext) has no run to budget, so it must not
    grow a timeout key it never had."""
    adapter = FlakyThenOkAdapter(failures=0)
    MemsomChatModel(adapter=adapter, model="m").invoke(
        [HumanMessage(content="go")])
    assert "timeout" not in adapter.calls[0]


# ---------------------------------------------------------------------------
# 12. the approval vocabulary — approve / deny / edit
# ---------------------------------------------------------------------------
#
# The gate used to understand exactly one word. Widening it is a security
# surface, so the shape of every answer is pinned here rather than inferred
# from the one end-to-end test that happens to exercise it: what executes, what
# denies, and — the part that matters — which way an answer nobody planned for
# falls.


def test_a_bare_approve_is_the_only_string_that_executes():
    from memsom.providers.lc_model import _normalize_decision

    assert _normalize_decision("approve") == ("approve", None)
    assert _normalize_decision("APPROVE") == ("approve", None)
    # everything else denies, deliberately including the nonsense — a gate that
    # executes on anything it fails to parse is not a gate.
    for answer in ["deny", "DENY", "", "yes", "ok", None, 0, ["approve"]]:
        assert _normalize_decision(answer) == ("deny", None), answer


def test_an_edit_payload_returns_the_substituted_arguments():
    from memsom.providers.lc_model import _normalize_decision

    verdict, edited = _normalize_decision(
        {"decision": "edit", "arguments": {"q": "EDITED", "n": 2}})
    assert verdict == "edit"
    assert edited == {"q": "EDITED", "n": 2}
    # a COPY: the tool must not be able to mutate the caller's payload, and the
    # payload must not be able to change under the tool.
    payload = {"decision": "edit", "arguments": {"q": "x"}}
    _, edited = _normalize_decision(payload)
    edited["q"] = "mutated"
    assert payload["arguments"] == {"q": "x"}


def test_a_dict_decision_that_is_not_an_edit_reads_as_the_word_it_carries():
    from memsom.providers.lc_model import _normalize_decision

    assert _normalize_decision({"decision": "approve"}) == ("approve", None)
    assert _normalize_decision({"decision": "deny"}) == ("deny", None)


def test_a_malformed_edit_fails_closed():
    """No arguments object means the handler that should have rejected it
    didn't, so something upstream is confused. Deny, do not fall back to the
    original call: a security gate that guesses when it is confused is not a
    gate."""
    from memsom.providers.lc_model import _normalize_decision

    for broken in [{"decision": "edit"},
                   {"decision": "edit", "arguments": None},
                   {"decision": "edit", "arguments": "q=1"},
                   {"decision": "edit", "arguments": [("q", 1)]}]:
        assert _normalize_decision(broken) == ("deny", None), broken


# ---------------------------------------------------------------------------
# 13. the handoff tool
# ---------------------------------------------------------------------------
#
# The synthetic tool a `handoff` router binds into the FEEDING agent, so the
# branch choice happens inside the turn the agent was taking anyway. Two things
# make it unlike every other tool in this file and both are pinned here: it is
# the one place a parent-directed Command is correct, and it is the one tool
# with a pydantic arg schema — which exists solely so LangGraph will inject the
# graph state it has to carry forward.


BRANCHES = [{"name": "esc", "when": "something went wrong",
             "target_node": "B"},
            {"name": "ok", "when": "all clear", "target_node": "C"}]


def _handoff(ctx, **over):
    from memsom.providers.lc_model import HandoffTool

    kwargs = dict(name="handoff", branches=BRANCHES,
                  target_map={"esc": "B", "ok": "C"},
                  router_node_id="R", ctx=ctx)
    kwargs.update(over)
    return HandoffTool(**kwargs)


def test_a_known_branch_returns_a_parent_directed_command(ctx):
    from langgraph.types import Command

    tool = _handoff(ctx)
    out = tool._run(branch="esc", state={"messages": []}, tool_call_id="tc_9")
    assert isinstance(out, Command)
    assert out.goto == "B"
    # graph=PARENT is the whole mechanism: it unwinds run_node's manual
    # subgraph.invoke, which is exactly what "this agent is done" means.
    assert out.graph == Command.PARENT


def test_the_command_carries_the_nodes_own_messages_and_nothing_older(ctx):
    """The reason this tool needs the graph state at all.

    A parent-directed Command unwinds `run_node` before it can return what the
    node produced, so without this the next agent would start blind — measured,
    not theorised. `prior` is how the tool knows which messages the parent is
    already holding."""
    tool = _handoff(ctx)
    tool.prior = 2
    state = {"messages": [HumanMessage(content="OLD-1"),
                          AIMessage(content="OLD-2"),
                          AIMessage(content="MINE-1"),
                          AIMessage(content="MINE-2")]}
    out = tool._run(branch="ok", state=state, tool_call_id="tc_9",
                    message="BRIEFING")
    carried = out.update["messages"]
    texts = [m.content for m in carried]
    assert "OLD-1" not in texts and "OLD-2" not in texts
    assert texts[:2] == ["MINE-1", "MINE-2"]
    # then the tool's own result, whose id MUST echo the call or the model sees
    # an unanswered tool call, and finally the briefing — addressed to whoever
    # runs next, so a SystemMessage rather than more tool output.
    assert isinstance(carried[2], ToolMessage)
    assert carried[2].tool_call_id == "tc_9"
    assert isinstance(carried[3], SystemMessage)
    assert "BRIEFING" in carried[3].content


def test_no_briefing_means_no_extra_message(ctx):
    tool = _handoff(ctx)
    out = tool._run(branch="ok", state={"messages": []}, tool_call_id="tc_1")
    assert [type(m).__name__ for m in out.update["messages"]] == ["ToolMessage"]


def test_an_unknown_branch_is_a_plain_string_never_an_exception(ctx):
    """A hallucinated branch is a model mistake, and the cheapest correct answer
    to a model mistake is a sentence it can act on. Raising would kill a run
    over a typo."""
    tool = _handoff(ctx)
    out = tool._run(branch="ghost", state={"messages": []}, tool_call_id="x")
    assert isinstance(out, str)
    assert "ghost" in out and "esc" in out and "ok" in out
    # and nothing was routed: no route event, no Command
    assert [e for e in events(ctx) if e["t"] == "route"] == []


def test_a_successful_handoff_emits_the_existing_route_event(ctx):
    tool = _handoff(ctx)
    tool._run(branch="esc", state={"messages": []}, tool_call_id="x")
    route = [e for e in events(ctx) if e["t"] == "route"]
    assert len(route) == 1
    assert route[0]["router"] == "R"
    assert route[0]["branch"] == "esc"
    assert route[0]["mode"] == "handoff"
    # exactly the keys the two older router modes emit — a handoff needs no new
    # event type, so read_since and RunMonitor are untouched.
    assert set(route[0]) == {"t", "router", "branch", "mode", "ts"}


def test_the_wire_schema_constrains_branch_and_hides_the_injected_arguments(ctx):
    """What the MODEL is shown is hand-written, not derived from the pydantic
    schema, because convert_to_openai_tool drops json_schema_extra (measured) —
    and without it `branch` would be a free string where `decide` mode's
    equivalent gets a hard enum."""
    from memsom.providers.lc_model import _as_openai_tool

    rendered = _as_openai_tool(_handoff(ctx))
    params = rendered["function"]["parameters"]
    assert rendered["function"]["name"] == "handoff"
    assert params["properties"]["branch"]["enum"] == ["esc", "ok"]
    assert params["required"] == ["branch"]
    # state/tool_call_id are LangGraph's to fill in; a model that saw them could
    # forge them.
    assert set(params["properties"]) == {"branch", "message"}
    # the branch hints reach the model through the description
    assert "something went wrong" in rendered["function"]["description"]


def test_the_description_tells_the_model_a_handoff_ends_its_turn(ctx):
    """Documentation-only guidance, deliberately: a call queued behind the
    handoff in the same batch never runs (the Command unwinds the tool node
    first), and enforcing that in code costs real machinery for a case
    well-behaved models do not hit."""
    assert "FINAL action" in _handoff(ctx).description


# ---------------------------------------------------------------------------
# 14. the RunContext under concurrency — counters, per-node turns, per-node
#     loop state, and the sidecar's atomic replace
# ---------------------------------------------------------------------------
#
# Fan-out means several agent nodes hold ONE RunContext at once. These drive the
# carrier directly rather than through a graph: a race is a probability, and the
# only honest way to test one is to make it overwhelmingly likely and then
# assert an exact number.


def test_run_context_counters_dont_lose_updates_under_concurrent_nodes(ctx):
    """N threads hammering the counters; the totals must be EXACT.

    ``stats["tool_calls"] += 1`` is a read-modify-write, and two threads landing
    on it lose one. Not a rounding error — the counter is what the terminal
    ``done`` line reports, and the same class of unguarded increment on ``turn``
    would lose a max_turns enforcement."""
    threads, per_thread = 8, 200
    start = threading.Barrier(threads, timeout=30)

    def hammer() -> None:
        start.wait()
        for _ in range(per_thread):
            ctx.count_tool_call()
            ctx.accumulate({"prompt_tokens": 1, "eval_count": 2})

    workers = [threading.Thread(target=hammer) for _ in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=30)
        assert not t.is_alive()

    total = threads * per_thread
    assert ctx.stats["tool_calls"] == total
    assert ctx.stats["prompt_tokens"] == total
    assert ctx.stats["eval_count"] == total * 2


def test_begin_turn_numbers_every_turn_exactly_once_across_threads(ctx):
    """No duplicate and no skipped turn number, and the run log's line order
    matches the numbers — the increment and the append share one lock precisely
    so a reader rendering the file in order sees the turns in order."""
    ctx.limits["max_turns"] = 0          # no ceiling: this is about numbering
    threads, per_thread = 6, 40
    start = threading.Barrier(threads, timeout=30)
    handed: list = []
    hlock = threading.Lock()

    def hammer(i: int) -> None:
        start.wait()
        for _ in range(per_thread):
            n = ctx.begin_turn(f"node{i}")
            with hlock:
                handed.append(n)

    workers = [threading.Thread(target=hammer, args=(i,))
               for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=30)
        assert not t.is_alive()

    total = threads * per_thread
    assert sorted(handed) == list(range(1, total + 1))   # each exactly once
    logged = [e["n"] for e in events(ctx) if e.get("t") == "turn"]
    assert logged == sorted(logged)      # file order IS turn order
    assert len(logged) == total


def test_begin_turn_enforces_the_ceiling_exactly_once_under_contention(ctx):
    """The ceiling is a budget, and a lost increment spends it twice. With the
    check and the increment under one lock, exactly `max_turns` calls succeed
    however many threads race for them."""
    ctx.limits["max_turns"] = 25
    threads = 8
    start = threading.Barrier(threads, timeout=30)
    ok, refused = [], []
    rlock = threading.Lock()

    def hammer(i: int) -> None:
        start.wait()
        for _ in range(20):
            try:
                n = ctx.begin_turn(f"node{i}")
            except ProviderError:
                with rlock:
                    refused.append(1)
            else:
                with rlock:
                    ok.append(n)

    workers = [threading.Thread(target=hammer, args=(i,))
               for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=30)
        assert not t.is_alive()

    assert sorted(ok) == list(range(1, 26))
    assert len(refused) == threads * 20 - 25


def test_turn_of_reports_the_nodes_own_turn_not_the_runs_latest(ctx):
    b = ctx.begin_turn("B")
    c = ctx.begin_turn("C")
    assert (b, c) == (1, 2)
    assert ctx.turn == 2               # the run has moved on
    assert ctx.turn_of("B") == 1       # B has not
    assert ctx.turn_of("C") == 2
    # a caller with no node (a tool driven outside a graph) still gets the
    # global counter, which is correct: with no nodes there is nothing to
    # mis-attribute to.
    assert ctx.turn_of("") == 2
    assert ctx.turn_of("never-ran") == 2


def test_a_tools_events_carry_its_own_nodes_turn(ctx):
    """The failure this closes: a sibling advancing the run counter while this
    node's tool call is still in flight."""
    ctx.begin_turn("B")                # B is on turn 1
    ctx.begin_turn("C")                # C races ahead to turn 2
    tool = MemsomTool(EchoTool({}), ctx, node_id="B")

    tool.invoke({"type": "tool_call", "id": "call_b", "name": "echo",
                 "args": {"x": "hi"}})

    turns = [e["turn"] for e in events(ctx)
             if e["t"] in ("tool_call", "tool_result")]
    assert turns == [1, 1]             # B's turn, not the run's 2


def test_loop_strikes_are_kept_per_node(ctx):
    """A sibling's interleaved batch must neither reset nor trip this node.

    With one shared signature, C's identical call between B's two would make
    B's strikes read as a three-in-a-row (or reset them, depending which way the
    scalar fell). Per node, B's own repetition is the only thing that counts."""
    sig = json.dumps([("echo", {"x": "hi"})], sort_keys=True)
    other = json.dumps([("echo", {"x": "bye"})], sort_keys=True)

    ctx.check_loop("B", sig)           # B strike 0
    ctx.check_loop("C", sig)           # C's identical call: not B's business
    ctx.check_loop("C", other)         # and C keeps moving
    ctx.check_loop("B", sig)           # B strike 1 — still not a loop
    ctx.check_loop("C", other)         # C strike 1
    with pytest.raises(ProviderError) as exc:
        ctx.check_loop("B", sig)       # B strike 2 → trips, on its own record
    assert str(exc.value) == (
        f"tool loop detected: {_LOOP_STRIKES}x identical call(s)")
    # C, which interleaved the whole way through, is untouched
    assert ctx.node_loop["C"][1] == 1


def test_a_differing_batch_still_resets_that_nodes_strikes(ctx):
    sig = json.dumps([("echo", {"x": "hi"})], sort_keys=True)
    other = json.dumps([("echo", {"x": "bye"})], sort_keys=True)
    ctx.check_loop("B", sig)
    ctx.check_loop("B", sig)           # strike 1
    ctx.check_loop("B", other)         # different → back to 0
    ctx.check_loop("B", sig)
    ctx.check_loop("B", sig)           # strike 1 again, no raise
    assert ctx.node_loop["B"][1] == 1


def test_tok_lines_carry_the_node_that_produced_them(ctx):
    """What makes an interleaved concurrent stream readable. A single-agent
    caller passes no node and the line is byte for byte what it always was."""
    ctx.sink.token("plain")
    ctx.sink.token("tagged", "agent:7")
    toks = [e for e in events(ctx) if e["t"] == "tok"]
    assert toks[0] == {"t": "tok", "text": "plain"}
    assert toks[1] == {"t": "tok", "text": "tagged", "node": "agent:7"}


def test_the_file_sink_serializes_concurrent_writers(tmp_path):
    """The lock, measured. Many threads streaming small chunks at one sink:
    every line must parse, and the token count must equal what was written."""
    sink = AgentFileSink(tmp_path / "run.jsonl")
    threads, chunks = 8, 250
    start = threading.Barrier(threads, timeout=30)

    def writer(i: int) -> None:
        start.wait()
        for n in range(chunks):
            sink.token(f"w{i}#{n} ", f"node{i}")
            if n % 25 == 0:
                sink.event({"t": "turn", "n": n, "node": f"node{i}"})

    workers = [threading.Thread(target=writer, args=(i,))
               for i in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=30)
        assert not t.is_alive()
    sink._fh.flush()

    raw = [l for l in Path(sink.path).read_text(encoding="utf-8").split("\n")
           if l.strip()]
    lines = [json.loads(l) for l in raw]      # a torn line raises here
    toks = [l for l in lines if l["t"] == "tok"]
    assert len(toks) == threads * chunks
    # every chunk kept its own tag: a spliced line would carry one writer's
    # node over another's text.
    assert all(t["text"].startswith(t["node"].replace("node", "w") + "#")
               for t in toks)
    # and the counter survived too — it rides inside the same acquisition
    assert sink.count == threads * chunks


def test_sync_data_retries_a_replace_that_loses_a_race(tmp_path, monkeypatch):
    """On Windows ``os.replace`` onto a path another handle has OPEN fails with
    PermissionError — measured at 2935 failures in 3000 attempts under a
    reader. sync_data swallows OSError, so without a retry the write is simply
    gone, and the scratchpad silently does not survive the pause it exists for.
    The failure is transient by construction, so it is retried."""
    sink = AgentFileSink(tmp_path / "run.jsonl")
    path = tmp_path / "shared" / "run.json"
    ctx = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                     limits=dict(LIMITS), data_path=path)
    monkeypatch.setattr("memsom.providers.lc_model._REPLACE_RETRY_S", 0.0)

    real = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    ctx.data["k"] = "v"
    ctx.sync_data()

    assert calls["n"] == 3                       # two failures, then it landed
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": "v"}


def test_sync_data_gives_up_quietly_when_the_replace_never_lands(tmp_path,
                                                                 monkeypatch):
    """Still best-effort: a persistently unwritable sidecar costs cross-pause
    survival, never the run."""
    sink = AgentFileSink(tmp_path / "run.jsonl")
    path = tmp_path / "shared" / "run.json"
    ctx = RunContext(sink=sink, audit_path=tmp_path / "audit.jsonl",
                     limits=dict(LIMITS), data_path=path)
    monkeypatch.setattr("memsom.providers.lc_model._REPLACE_RETRY_S", 0.0)

    calls = {"n": 0}

    def always_denied(self, target):
        calls["n"] += 1
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", always_denied)
    ctx.data["k"] = "v"
    ctx.sync_data()                              # must not raise

    from memsom.providers.lc_model import _REPLACE_ATTEMPTS
    assert calls["n"] == _REPLACE_ATTEMPTS
    assert not path.exists()
