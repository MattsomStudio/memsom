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
from pathlib import Path

import pytest

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)

from memsom.providers.agents import _LOOP_STRIKES
from memsom.providers.base import ProviderError
from memsom.providers.lc_model import (
    MemsomChatModel,
    MemsomTool,
    RunContext,
    from_memsom_messages,
    to_memsom_messages,
)
from memsom.providers.session import AgentFileSink
from memsom.providers.tools.base import Tool, ToolError


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
    assert ctx.strikes == 0


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
