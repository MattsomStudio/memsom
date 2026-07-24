"""Characterization tests for the agent runtime as it behaves TODAY.

This path had zero coverage, which made the planned LangGraph swap
unfalsifiable: nothing on disk said what a run is supposed to look like, so
"it still works" would have been an opinion. These tests pin the OBSERVABLE
contract instead of the implementation — the AgentSpec that falls out of a
saved canvas, the verbatim compile errors the route turns into 400s, and above
all the exact ordered JSONL event stream the frontend cursor-polls. Swap the
loop for anything you like; if this file stays green, the app cannot tell.

Nothing here touches a network or a real provider: a FakeAdapter stands in for
the engine and a FakeTool is registered into BUILTIN_TOOLS for the duration of
a test, so the tool the model "calls" is a pure function of the test file.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import pytest

from memsom.providers.agents import (
    _DEFAULT_LIMITS,
    _DEFAULT_MAX_STEPS,
    _LOOP_STRIKES,
    _MAX_TURNS_CEILING,
    AgentRunner,
    AgentSpec,
    GraphSpec,
    RouterSpec,
    compile_graph,
)
from memsom.providers.base import Capabilities, ProviderError, ProviderStatus
from memsom.providers.session import AgentFileSink, new_session_id
from memsom.providers.tools import registry as tool_registry
from memsom.providers.tools.base import Tool, ToolError

# Canaries: strings that exist so a test can prove they DIDN'T leak somewhere.
SYSTEM_CANARY = "SYSTEM-PROMPT-CANARY-do-not-audit-me"
ANSWER_CANARY = "MODEL-TEXT-CANARY-do-not-audit-me"


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class FakeAdapter:
    """The smallest thing AgentRunner will drive: capabilities/status/infer.

    ``script`` is one entry per turn — ``(text, stats)``. The text is pushed to
    the sink exactly like a real adapter buffering a whole turn, and the stats
    dict is returned verbatim, which is how a turn asks for tools (canonical
    ``stats['tool_calls']``). Past the end of the script the last entry repeats,
    so a "keeps asking for the same tool forever" adapter is a one-entry script.
    """

    transport = None

    def __init__(self, script: list, *, state: str = "up",
                 gate: threading.Event = None) -> None:
        self.script = script
        self.state = state
        self.gate = gate
        self.calls: list = []          # (model, messages copy, params copy)

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.state)

    def infer(self, model: str, messages: list, params: dict, sink) -> dict:
        if self.gate is not None:
            # Deadline, not a block: a hung test should fail, not wedge pytest.
            self.gate.wait(timeout=30)
        self.calls.append((model, copy.deepcopy(messages), dict(params)))
        text, stats = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if text:
            sink.token(text)
        return copy.deepcopy(stats)


class FakeTool(Tool):
    """A builtin-shaped tool that only touches memory, never the network."""

    type = "fake_tool"
    description = "echoes its q argument"
    parameters = {"type": "object", "properties": {"q": {"type": "string"}},
                  "required": ["q"]}

    def run(self, arguments: dict, ctx) -> str:
        return f"FAKE-OUTPUT:{arguments.get('q')}"


class ExplodingTool(Tool):
    type = "boom_tool"
    description = "always fails"
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments: dict, ctx) -> str:
        raise ToolError("nope")


@pytest.fixture
def fake_tools(monkeypatch):
    """Register the doubles in the real registry for one test.

    ``build_tools`` resolves BUILTIN_TOOLS out of its own module globals, so the
    registry module is the thing to patch — patching the re-export in
    ``tools/__init__`` would not be seen.
    """
    monkeypatch.setitem(tool_registry.BUILTIN_TOOLS, FakeTool.type, FakeTool)
    monkeypatch.setitem(tool_registry.BUILTIN_TOOLS, ExplodingTool.type,
                        ExplodingTool)
    return tool_registry.BUILTIN_TOOLS


# ---------------------------------------------------------------------------
# the graph document
# ---------------------------------------------------------------------------


def _doc() -> dict:
    """A real saved canvas: trigger→agent, engine→agent, tool→agent, agent→output.

    Handles are load-bearing — ``compile_graph`` finds the engine and the tools
    by ``targetHandle``, not by node type alone, so a tool wired into the
    ``engine`` handle is invisible to it.
    """
    return {
        "id": "g_demo",
        "rev": 7,
        "nodes": [
            {"id": "n3", "type": "trigger",
             "config": {"mode": "manual", "input": "TRIGGER-INPUT"}},
            {"id": "n4", "type": "engine",
             "config": {"provider": "fake", "model": "qwen2.5:7b-instruct",
                        "transport": None}},
            {"id": "n5", "type": "tool",
             "config": {"tool": "http_fetch", "options": {"max_bytes": 100000}}},
            {"id": "n6", "type": "agent",
             "config": {"name": "RESEARCHER", "system": SYSTEM_CANARY,
                        "params": {"temperature": 0.2, "max_tokens": 512,
                                   "ctx": 8192},
                        "limits": {"max_turns": 4, "run_timeout_s": 120}}},
            {"id": "n7", "type": "output", "config": {}},
        ],
        "edges": [
            {"id": "e1", "source": "n3", "target": "n6",
             "sourceHandle": "run", "targetHandle": "trigger"},
            {"id": "e2", "source": "n4", "target": "n6",
             "sourceHandle": "engine", "targetHandle": "engine"},
            {"id": "e3", "source": "n5", "target": "n6",
             "sourceHandle": "tool", "targetHandle": "tools"},
            {"id": "e4", "source": "n6", "target": "n7",
             "sourceHandle": "out", "targetHandle": "in"},
        ],
    }


def _node(doc: dict, node_id: str) -> dict:
    return next(n for n in doc["nodes"] if n["id"] == node_id)


def _registry(adapter=None) -> dict:
    return {"fake": adapter if adapter is not None else FakeAdapter([("hi", {})])}


# ---------------------------------------------------------------------------
# 1. compile_graph — the happy path
# ---------------------------------------------------------------------------


def test_compile_graph_extracts_the_spec_from_the_canvas():
    spec = compile_graph(_doc(), _registry())
    # compile_graph now yields a GraphSpec (the whole graph), not a bare
    # AgentSpec — but a one-agent graph answers provider/model/system/tools/
    # limits through its entry agent exactly as the old AgentSpec did, which is
    # what list_runs and the audit line depend on.
    assert isinstance(spec, GraphSpec)
    assert spec.graph_id == "g_demo"
    assert spec.graph_rev == 7
    assert spec.agent_name == "RESEARCHER"
    assert spec.provider_id == "fake"
    assert spec.model == "qwen2.5:7b-instruct"
    assert spec.transport is None
    assert spec.system == SYSTEM_CANARY
    assert spec.params == {"temperature": 0.2, "max_tokens": 512, "ctx": 8192}
    assert [t["name"] for t in spec.tool_specs] == ["http_fetch"]
    assert spec.tool_specs[0] == {"name": "http_fetch", "type": "http_fetch",
                                  "options": {"max_bytes": 100000},
                                  "require_approval": False}


def test_compile_graph_merges_limits_over_the_defaults():
    spec = compile_graph(_doc(), _registry())
    # the two the node declares win; the two it doesn't keep the defaults
    assert spec.limits["max_turns"] == 4
    assert spec.limits["run_timeout_s"] == 120
    assert spec.limits["tool_timeout_s"] == _DEFAULT_LIMITS["tool_timeout_s"]
    assert spec.limits["max_tool_output_bytes"] == \
        _DEFAULT_LIMITS["max_tool_output_bytes"]
    assert spec.limits is not _DEFAULT_LIMITS  # never mutate the module default


def test_compile_graph_ignores_junk_limits_and_caps_max_turns():
    doc = _doc()
    _node(doc, "n6")["config"]["limits"] = {
        "max_turns": 999,          # over the hard ceiling
        "run_timeout_s": 0,        # non-positive → ignored
        "tool_timeout_s": "sixty",  # non-numeric → ignored
        "nonsense_knob": 5,        # unknown key → ignored
    }
    spec = compile_graph(doc, _registry())
    assert spec.limits["max_turns"] == _MAX_TURNS_CEILING
    assert spec.limits["run_timeout_s"] == _DEFAULT_LIMITS["run_timeout_s"]
    assert spec.limits["tool_timeout_s"] == _DEFAULT_LIMITS["tool_timeout_s"]
    assert "nonsense_knob" not in spec.limits


def test_compile_graph_takes_input_from_the_trigger():
    assert compile_graph(_doc(), _registry()).input == "TRIGGER-INPUT"


def test_compile_graph_input_override_wins_including_empty_string():
    spec = compile_graph(_doc(), _registry(), input_override="FROM-THE-API")
    assert spec.input == "FROM-THE-API"
    # "" is a real override, not "unset" — the sentinel is None, and a run
    # started with an explicitly blank input must not silently inherit the
    # canvas's saved prompt.
    assert compile_graph(_doc(), _registry(), input_override="").input == ""


def test_compile_graph_folds_engine_transport_into_the_infer_params():
    doc = _doc()
    _node(doc, "n4")["config"]["transport"] = "api"
    spec = compile_graph(doc, _registry())
    assert spec.transport == "api"
    assert spec.params["transport"] == "api"


def test_as_start_meta_is_the_run_files_head_line():
    spec = compile_graph(_doc(), _registry())
    # ADDITIVE change (per the plan): the entry-agent fields the head line has
    # always carried are unchanged, and an `agents` array describing every
    # agent node is appended. Old readers ignore the new key.
    assert spec.as_start_meta() == {
        "graph_id": "g_demo",
        "agent": "RESEARCHER",
        "provider": "fake",
        "model": "qwen2.5:7b-instruct",
        "tools": ["http_fetch"],
        "limits": spec.limits,
        "agents": [
            {"node_id": "n6", "name": "RESEARCHER", "provider": "fake",
             "model": "qwen2.5:7b-instruct", "tools": ["http_fetch"]},
        ],
    }
    # max_steps now rides on the graph-level limits, defaulted off the trigger.
    assert spec.limits["max_steps"] == _DEFAULT_MAX_STEPS


# ---------------------------------------------------------------------------
# 2. tool instance naming
# ---------------------------------------------------------------------------


def test_duplicate_tool_types_get_numeric_suffixes():
    doc = _doc()
    doc["nodes"].append({"id": "n8", "type": "tool",
                         "config": {"tool": "http_fetch", "options": {}}})
    doc["edges"].append({"id": "e5", "source": "n8", "target": "n6",
                         "sourceHandle": "tool", "targetHandle": "tools"})
    spec = compile_graph(doc, _registry())
    # first instance keeps the bare name so single-tool graphs read naturally;
    # the collision is what gets the suffix, and numbering starts at 2.
    assert [t["name"] for t in spec.tool_specs] == ["http_fetch", "http_fetch_2"]


def test_tool_labels_are_slugged_to_a_callable_name():
    doc = _doc()
    _node(doc, "n5")["config"]["label"] = "My Fetcher!"
    spec = compile_graph(doc, _registry())
    # lowercased, every non-alnum/underscore byte becomes "_" — the model has to
    # be able to emit this as a function name.
    assert spec.tool_specs[0]["name"] == "my_fetcher_"
    assert spec.tool_specs[0]["type"] == "http_fetch"


def test_slugged_labels_collide_and_suffix_on_the_slug_not_the_label():
    doc = _doc()
    _node(doc, "n5")["config"]["label"] = "Web One"
    doc["nodes"].append({"id": "n8", "type": "tool",
                         "config": {"tool": "http_fetch", "label": "web/one"}})
    doc["edges"].append({"id": "e5", "source": "n8", "target": "n6",
                         "sourceHandle": "tool", "targetHandle": "tools"})
    spec = compile_graph(doc, _registry())
    assert [t["name"] for t in spec.tool_specs] == ["web_one", "web_one_2"]


def test_tools_wired_to_the_wrong_handle_are_not_picked_up():
    doc = _doc()
    _node(doc, "n5")  # exists, but rewire it away from the tools handle
    doc["edges"][2]["targetHandle"] = "trigger"
    spec = compile_graph(doc, _registry())
    assert spec.tool_specs == []


# ---------------------------------------------------------------------------
# 3. compile_graph — every error path, verbatim
# ---------------------------------------------------------------------------


def test_error_zero_agent_nodes():
    doc = _doc()
    doc["nodes"] = [n for n in doc["nodes"] if n["type"] != "agent"]
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    # MODIFIED: the rule became "at least one agent" (multi-agent graphs are the
    # whole point of the swap), so the "exactly one … (found 0)" wording is gone.
    assert str(ei.value) == "graph must contain at least one agent node"


def test_two_agent_nodes_are_legal_and_each_is_validated():
    # MODIFIED: two agents used to be a compile error; now it's a valid graph.
    # A SECOND agent with no engine wired in is still rejected — per-agent, and
    # the message is qualified by name once there's more than one agent to
    # confuse.
    doc = _doc()
    doc["nodes"].append({"id": "n9", "type": "agent", "config": {"name": "B"}})
    doc["edges"].append({"id": "e5", "source": "n6", "target": "n9",
                         "sourceHandle": "next", "targetHandle": "in"})
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "agent 'B' needs exactly one engine wired in (found 0)"


def test_error_no_engine_wired_in():
    doc = _doc()
    doc["edges"] = [e for e in doc["edges"] if e["source"] != "n4"]
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "agent needs exactly one engine wired in (found 0)"


def test_error_two_engines_wired_in():
    doc = _doc()
    doc["nodes"].append({"id": "n9", "type": "engine",
                         "config": {"provider": "fake", "model": "m"}})
    doc["edges"].append({"id": "e5", "source": "n9", "target": "n6",
                         "sourceHandle": "engine", "targetHandle": "engine"})
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "agent needs exactly one engine wired in (found 2)"


def test_error_unknown_provider():
    doc = _doc()
    _node(doc, "n4")["config"]["provider"] = "nope"
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "unknown provider: 'nope'"


def test_error_missing_provider_reports_the_empty_string():
    doc = _doc()
    _node(doc, "n4")["config"].pop("provider")
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "unknown provider: ''"


def test_error_engine_without_a_model():
    doc = _doc()
    _node(doc, "n4")["config"]["model"] = ""
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "engine node has no model selected"


def test_error_unknown_tool_type_fails_at_compile_time_not_mid_run():
    doc = _doc()
    _node(doc, "n5")["config"]["tool"] = "teleport"
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "unknown tool type: 'teleport'"


def test_error_tools_on_a_cli_subscription_transport():
    doc = _doc()
    _node(doc, "n4")["config"]["transport"] = "cli-subscription"
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == (
        "custom tools not supported over cli transport — use the api "
        "transport or remove the tool nodes")


def test_cli_subscription_transport_is_fine_with_no_tools():
    doc = _doc()
    _node(doc, "n4")["config"]["transport"] = "cli-subscription"
    doc["edges"] = [e for e in doc["edges"] if e["source"] != "n5"]
    spec = compile_graph(doc, _registry())
    assert spec.transport == "cli-subscription"
    assert spec.tool_specs == []


def test_error_cli_transport_inherited_from_the_adapter():
    # The engine node may leave transport null and still land on a CLI-only
    # adapter — the gate has to read the RESOLVED transport, not the config.
    adapter = FakeAdapter([("hi", {})])
    adapter.transport = "cli-subscription"
    doc = _doc()
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry(adapter))
    assert str(ei.value).startswith("custom tools not supported over cli transport")


# ---------------------------------------------------------------------------
# 4. end to end through AgentRunner
# ---------------------------------------------------------------------------


def _e2e_doc() -> dict:
    doc = _doc()
    _node(doc, "n5")["config"] = {"tool": "fake_tool", "options": {}}
    return doc


def _runner(tmp_path: Path, adapter) -> AgentRunner:
    return AgentRunner(tmp_path / "runs", _registry(adapter),
                       tmp_path / "audit.jsonl")


def _drain(runner: AgentRunner, run_id: str, timeout: float = 10.0) -> tuple:
    """Cursor-poll to completion exactly like the frontend does.

    Returns ``(events, status, stats)``. A run that never terminates fails the
    test loudly at the deadline instead of hanging the suite — the whole point
    of the loop's own timeout/turn ceiling is that a hang is a bug.
    """
    deadline = time.monotonic() + timeout
    events, cursor, seen_status, stats = [], 0, "running", None
    while time.monotonic() < deadline:
        r = runner.read_since(run_id, cursor)
        events.extend(r["events"])
        cursor = r["cursor"]
        seen_status, stats = r["status"], r.get("stats")
        if seen_status in ("done", "error"):
            return events, seen_status, stats
        time.sleep(0.02)
    raise AssertionError(
        f"run {run_id} did not terminate within {timeout}s; "
        f"last status={seen_status} events={[e.get('t') for e in events]}")


def test_end_to_end_run_emits_the_exact_event_sequence(tmp_path, fake_tools):
    adapter = FakeAdapter([
        ("thinking about it", {"tool_calls": [
            {"id": "tc_1", "name": "fake_tool", "arguments": {"q": "hello"}}],
            "eval_count": 5, "prompt_tokens": 11}),
        (ANSWER_CANARY, {"eval_count": 7, "prompt_tokens": 13}),
    ])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_e2e_doc(), _registry(adapter))
    run_id = runner.start(spec, "manual")
    events, status, _ = _drain(runner, run_id)

    assert status == "done"
    # ADDITIVE change (per the plan): a `node` event marks entry into each
    # agent node. A one-agent graph emits exactly one, right after warmup;
    # everything after it is byte-for-byte the legacy stream.
    assert [e["t"] for e in events] == [
        "start", "warmup", "node", "turn", "tok",
        "tool_call", "tool_result",
        "turn", "tok", "done",
    ]


def test_end_to_end_payloads_are_what_the_frontend_reads(tmp_path, fake_tools):
    adapter = FakeAdapter([
        ("thinking about it", {"tool_calls": [
            {"id": "tc_1", "name": "fake_tool", "arguments": {"q": "hello"}}],
            "eval_count": 5}),
        (ANSWER_CANARY, {"eval_count": 7}),
    ])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_e2e_doc(), _registry(adapter))
    run_id = runner.start(spec, "manual")
    events, status, poll_stats = _drain(runner, run_id)
    assert status == "done"
    by_t = {}
    for ev in events:
        by_t.setdefault(ev["t"], []).append(ev)

    head = by_t["start"][0]
    assert head["run_id"] == run_id
    assert head["trigger"] == "manual"
    assert head["graph_id"] == "g_demo"
    assert head["agent"] == "RESEARCHER"
    assert head["provider"] == "fake"
    assert head["model"] == "qwen2.5:7b-instruct"
    assert head["tools"] == ["fake_tool"]

    # a warm engine is a no-op warmup, not a start
    assert by_t["warmup"][0]["action"] == "none"
    assert by_t["warmup"][0]["ok"] is True
    assert by_t["warmup"][0]["detail"] == "up"

    assert [e["n"] for e in by_t["turn"]] == [1, 2]

    call = by_t["tool_call"][0]
    assert call["turn"] == 1
    assert call["id"] == "tc_1"
    assert call["name"] == "fake_tool"
    assert call["arguments"] == {"q": "hello"}

    res = by_t["tool_result"][0]
    assert res["turn"] == 1
    assert res["id"] == "tc_1"
    assert res["name"] == "fake_tool"
    assert res["ok"] is True
    assert res["output"] == "FAKE-OUTPUT:hello"
    assert res["bytes"] == len(b"FAKE-OUTPUT:hello")
    assert res["truncated"] is False
    assert isinstance(res["elapsed_s"], float)

    stats = by_t["done"][0]["stats"]
    assert stats["turns"] == 2
    assert stats["tool_calls"] == 1
    assert stats["tokens"] == 2                 # one sink token per turn
    assert isinstance(stats["elapsed_s"], float)
    assert stats["eval_count"] == 12            # summed across turns
    assert poll_stats == stats                  # read_since surfaces the same dict


def test_end_to_end_feeds_the_conversation_back_to_the_adapter(tmp_path,
                                                               fake_tools):
    adapter = FakeAdapter([
        ("", {"tool_calls": [
            {"id": "tc_1", "name": "fake_tool", "arguments": {"q": "hello"}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_e2e_doc(), _registry(adapter))
    _drain(runner, runner.start(spec, "manual"))

    first_msgs = adapter.calls[0][1]
    assert [m["role"] for m in first_msgs] == ["system", "user"]
    assert first_msgs[0]["content"] == SYSTEM_CANARY
    assert first_msgs[1]["content"] == "TRIGGER-INPUT"
    # tools reach the adapter in the OpenAI wire shape, under params["tools"]
    assert adapter.calls[0][2]["tools"] == [{
        "type": "function",
        "function": {"name": "fake_tool", "description": FakeTool.description,
                     "parameters": FakeTool.parameters},
    }]
    # second turn carries the assistant tool-call turn plus the tool result
    second_msgs = adapter.calls[1][1]
    assert [m["role"] for m in second_msgs] == ["system", "user", "assistant",
                                                "tool"]
    assert second_msgs[2]["tool_calls"][0]["name"] == "fake_tool"
    assert second_msgs[3]["tool_call_id"] == "tc_1"
    assert second_msgs[3]["content"] == "FAKE-OUTPUT:hello"


def test_a_failing_tool_is_a_result_not_a_dead_run(tmp_path, fake_tools):
    # A tool blowing up is information for the model, not the end of the run —
    # that's the whole reason _execute_tool swallows ToolError.
    doc = _e2e_doc()
    _node(doc, "n5")["config"] = {"tool": "boom_tool", "options": {}}
    adapter = FakeAdapter([
        ("", {"tool_calls": [
            {"id": "tc_1", "name": "boom_tool", "arguments": {}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    events, status, _ = _drain(runner, runner.start(
        compile_graph(doc, _registry(adapter)), "manual"))
    assert status == "done"
    res = next(e for e in events if e["t"] == "tool_result")
    assert res["ok"] is False
    assert res["output"] == "tool error: nope"


def test_unknown_tool_name_is_a_recoverable_message_not_a_dead_run(tmp_path,
                                                                   fake_tools):
    # MODIFIED under the LangGraph swap: tool DISPATCH moved into langgraph's
    # ToolNode, so an unknown tool name is now handled there, not by our
    # _execute_tool(None, …). The surviving contract is the one that matters —
    # a bogus tool name does NOT kill the run; the model is handed an error
    # message naming the tools it actually has and gets to react (here it
    # answers on the next turn). What changed is an implementation detail the
    # legacy loop happened to expose: the exact "unknown tool 'X'; available: Y"
    # string and a synthetic tool_result event for a call our code never ran.
    adapter = FakeAdapter([
        ("", {"tool_calls": [
            {"id": "tc_1", "name": "teleport", "arguments": {}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    events, status, _ = _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    assert status == "done"
    # our MemsomTool never ran, so there is no tool_result event for it…
    assert [e for e in events if e["t"] == "tool_result"] == []
    # …but the model got the error, listing the real tool, on its second turn.
    tool_msg = next(m for m in adapter.calls[1][1] if m["role"] == "tool")
    assert tool_msg["name"] == "teleport"
    assert "fake_tool" in tool_msg["content"]


# ---------------------------------------------------------------------------
# 5. read_since cursor semantics
# ---------------------------------------------------------------------------


def test_read_since_never_repeats_and_never_skips(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    run_id = new_session_id()
    sink = AgentFileSink(runner._path(run_id))
    try:
        seen, cursor = [], 0
        writes = [
            lambda: sink.event({"t": "turn", "n": 1}),
            lambda: sink.token("A"),
            lambda: sink.token("B"),
            lambda: sink.event({"t": "tool_call", "id": "tc_1"}),
            lambda: sink.event({"t": "tool_result", "id": "tc_1"}),
        ]
        for write in writes:
            write()
            r = runner.read_since(run_id, cursor)
            seen.extend(r["events"])
            cursor = r["cursor"]
            # an immediate second poll at the same cursor must be empty — the
            # cursor is a line index, and overshooting it once buried every
            # later record forever (the split("\n") trailing-"" bug).
            assert runner.read_since(run_id, cursor)["events"] == []
        sink.done({"tokens": 2})
        r = runner.read_since(run_id, cursor)
        seen.extend(r["events"])
        assert [e["t"] for e in seen] == [
            "turn", "tok", "tok", "tool_call", "tool_result", "done"]
        assert r["status"] == "done"
        assert r["stats"] == {"tokens": 2}
        # and a fresh reader from 0 sees exactly the same stream
        assert runner.read_since(run_id, 0)["events"] == seen
    finally:
        try:
            sink._fh.close()
        except Exception:
            pass


def test_read_since_rejects_a_traversal_shaped_run_id(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    with pytest.raises(ProviderError) as ei:
        runner.read_since("../../etc/passwd")
    assert str(ei.value) == "invalid run_id"


def test_read_since_on_a_missing_run_is_not_an_error(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    assert runner.read_since(new_session_id(), 3) == {
        "events": [], "cursor": 3, "status": "unknown"}


# ---------------------------------------------------------------------------
# 6. list_runs
# ---------------------------------------------------------------------------


def _seed_run(runner: AgentRunner, run_id: str, head: dict,
              tail: dict = None) -> None:
    with open(runner._path(run_id), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(head) + "\n")
        if tail is not None:
            fh.write(json.dumps(tail) + "\n")


def test_list_runs_parses_the_head_line(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    rid = new_session_id()
    _seed_run(runner, rid, {
        "t": "start", "run_id": rid, "trigger": "schedule",
        "graph_id": "g_demo", "agent": "RESEARCHER", "provider": "ollama",
        "model": "qwen2.5:7b-instruct", "tools": ["http_fetch"],
        "limits": {}, "ts": 1234.5,
    }, {"t": "done", "stats": {"turns": 2}})
    rows = runner.list_runs()
    assert len(rows) == 1
    assert rows[0] == {
        "run_id": rid, "graph_id": "g_demo", "agent": "RESEARCHER",
        "provider": "ollama", "model": "qwen2.5:7b-instruct",
        "trigger": "schedule", "ts": 1234.5, "status": "done",
    }


def test_list_runs_statuses_and_limit(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    ids = {}
    for key, tail in (("done", {"t": "done", "stats": {}}),
                      ("error", {"t": "error", "error": "boom"}),
                      ("interrupted", None)):
        rid = new_session_id()
        ids[key] = rid
        _seed_run(runner, rid, {"t": "start", "run_id": rid}, tail)
        time.sleep(0.01)   # distinct mtimes so the newest-first order is stable
    by_id = {r["run_id"]: r for r in runner.list_runs()}
    assert by_id[ids["done"]]["status"] == "done"
    assert by_id[ids["error"]]["status"] == "error"
    # no terminal line and no live thread = the previous server died mid-run
    assert by_id[ids["interrupted"]]["status"] == "interrupted"
    assert len(runner.list_runs(limit=1)) == 1


# ---------------------------------------------------------------------------
# 7. the two-phase audit
# ---------------------------------------------------------------------------


def test_every_tool_call_writes_a_pending_then_a_result_audit_line(tmp_path,
                                                                   fake_tools):
    adapter = FakeAdapter([
        ("", {"tool_calls": [
            {"id": "tc_1", "name": "fake_tool", "arguments": {"q": "hello"}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    _, status, _ = _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    assert status == "done"

    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(l) for l in raw.split("\n") if l.strip()]
    assert len(lines) == 2
    pending, result = lines
    # phase one is written BEFORE the tool runs and gates the call: if the audit
    # log can't be written, the tool never executes.
    assert pending["action"] == "tool"
    assert pending["tool"] == "fake_tool"
    assert pending["arguments"] == {"q": "hello"}
    assert pending["result"] == "pending"
    assert result["tool"] == "fake_tool"
    assert result["result"] == "ok"
    assert all("ts" in rec for rec in lines)


def test_the_audit_log_never_records_the_prompt_or_the_models_text(tmp_path,
                                                                   fake_tools):
    adapter = FakeAdapter([
        ("intermediate reasoning", {"tool_calls": [
            {"id": "tc_1", "name": "fake_tool", "arguments": {"q": "hello"}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert SYSTEM_CANARY not in raw
    assert ANSWER_CANARY not in raw
    assert "intermediate reasoning" not in raw
    assert "TRIGGER-INPUT" not in raw
    assert "FAKE-OUTPUT" not in raw   # the tool's output stays out too


def test_audit_arguments_are_stringified_and_clipped(tmp_path, fake_tools):
    adapter = FakeAdapter([
        ("", {"tool_calls": [{"id": "tc_1", "name": "fake_tool",
                              "arguments": {"q": "z" * 500, "n": 7}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    pending = json.loads((tmp_path / "audit.jsonl")
                         .read_text(encoding="utf-8").split("\n")[0])
    assert pending["arguments"]["q"] == "z" * 200   # clipped at 200 chars
    assert pending["arguments"]["n"] == "7"         # every value becomes a str


def test_a_failing_tool_records_the_failure_in_phase_two(tmp_path, fake_tools):
    doc = _e2e_doc()
    _node(doc, "n5")["config"] = {"tool": "boom_tool", "options": {}}
    adapter = FakeAdapter([
        ("", {"tool_calls": [{"id": "tc_1", "name": "boom_tool",
                              "arguments": {}}]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    _drain(runner, runner.start(compile_graph(doc, _registry(adapter)),
                                "manual"))
    lines = [json.loads(l) for l in (tmp_path / "audit.jsonl")
             .read_text(encoding="utf-8").split("\n") if l.strip()]
    assert [rec["result"] for rec in lines] == ["pending", "failed: nope"]


def test_a_parallel_tool_batch_audits_every_call_with_no_torn_lines(tmp_path,
                                                                    fake_tools):
    # A turn that emits SEVERAL tool calls at once is ordinary parallel function
    # calling. LangGraph's ToolNode fans those across a thread pool; memsom runs
    # them sequentially (max_concurrency=1) so the fsync'd audit appends can't
    # interleave. Without that, pending lines for calls that DID run were lost
    # and the JSONL itself tore — the one safety property (an intent line before
    # every call) silently broken. Here: four calls in one turn → four intact
    # pending/ok pairs, four tool_call/tool_result events, nothing torn.
    adapter = FakeAdapter([
        ("", {"tool_calls": [
            {"id": f"tc_{i}", "name": "fake_tool", "arguments": {"q": c}}
            for i, c in enumerate("abcd", 1)]}),
        (ANSWER_CANARY, {}),
    ])
    runner = _runner(tmp_path, adapter)
    events, status, _ = _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    assert status == "done"

    raw = [l for l in (tmp_path / "audit.jsonl")
           .read_text(encoding="utf-8").split("\n") if l.strip()]
    lines = [json.loads(l) for l in raw]        # torn line → JSONDecodeError here
    assert len(lines) == len(raw)               # every line parsed: none torn
    results = [r["result"] for r in lines if r.get("action") == "tool"]
    assert results.count("pending") == 4
    assert results.count("ok") == 4
    assert len([e for e in events if e["t"] == "tool_call"]) == 4
    assert len([e for e in events if e["t"] == "tool_result"]) == 4


# ---------------------------------------------------------------------------
# 8. loop detection
# ---------------------------------------------------------------------------


def test_a_model_stuck_on_one_tool_call_terminates_with_an_error(tmp_path,
                                                                 fake_tools):
    # One-entry script = the adapter asks for the identical call forever. The
    # run must die on its own; a hang here would be the failure mode the strike
    # counter exists to prevent, and _drain's deadline catches it.
    adapter = FakeAdapter([
        ("", {"tool_calls": [{"id": "tc_1", "name": "fake_tool",
                              "arguments": {"q": "same"}}]}),
    ])
    runner = _runner(tmp_path, adapter)
    events, status, stats = _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    assert status == "error"
    assert events[-1]["t"] == "error"
    assert events[-1]["error"] == (
        f"tool loop detected: {_LOOP_STRIKES}x identical call(s)")
    assert stats == {"error": events[-1]["error"]}
    # it takes three identical turns to trip, and the third one never executes
    # the tool — the strike fires before the call goes out.
    assert [e["n"] for e in events if e["t"] == "turn"] == [1, 2, 3]
    assert len([e for e in events if e["t"] == "tool_call"]) == 2


def test_running_out_of_turns_is_a_terminal_error_not_a_silent_stop(tmp_path,
                                                                    fake_tools):
    # Distinct arguments each turn so the loop detector stays quiet and the
    # turn ceiling is what ends it.
    class Counting(FakeAdapter):
        def infer(self, model, messages, params, sink):
            self.calls.append((model, [], dict(params)))
            return {"tool_calls": [{"id": f"tc_{len(self.calls)}",
                                    "name": "fake_tool",
                                    "arguments": {"q": str(len(self.calls))}}]}

    adapter = Counting([])
    doc = _e2e_doc()
    _node(doc, "n6")["config"]["limits"] = {"max_turns": 3}
    runner = _runner(tmp_path, adapter)
    events, status, _ = _drain(runner, runner.start(
        compile_graph(doc, _registry(adapter)), "manual"))
    assert status == "error"
    assert events[-1]["error"] == "max turns reached (3) without a final answer"
    assert [e["n"] for e in events if e["t"] == "turn"] == [1, 2, 3]


def test_a_provider_failure_becomes_a_terminal_error_line(tmp_path, fake_tools):
    class Broken(FakeAdapter):
        def infer(self, model, messages, params, sink):
            raise ProviderError("engine went away")

    adapter = Broken([])
    runner = _runner(tmp_path, adapter)
    events, status, _ = _drain(runner, runner.start(
        compile_graph(_e2e_doc(), _registry(adapter)), "manual"))
    assert status == "error"
    assert events[-1] == {"t": "error", "error": "engine went away"}


# ---------------------------------------------------------------------------
# 9. reconcile_on_boot
# ---------------------------------------------------------------------------


def test_reconcile_on_boot_stamps_an_interrupted_run(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    rid = new_session_id()
    _seed_run(runner, rid, {"t": "start", "run_id": rid, "graph_id": "g_demo"})
    assert runner.read_since(rid)["status"] == "interrupted"

    runner.reconcile_on_boot()

    r = runner.read_since(rid)
    assert r["status"] == "error"
    assert r["events"][-1] == {
        "t": "error", "error": "interrupted: panel server restarted"}


def test_reconcile_on_boot_leaves_terminated_runs_alone(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    done_id, err_id = new_session_id(), new_session_id()
    _seed_run(runner, done_id, {"t": "start"}, {"t": "done", "stats": {"a": 1}})
    _seed_run(runner, err_id, {"t": "start"}, {"t": "error", "error": "boom"})

    runner.reconcile_on_boot()
    runner.reconcile_on_boot()   # idempotent

    assert len(runner.read_since(done_id)["events"]) == 2
    assert runner.read_since(err_id)["events"][-1]["error"] == "boom"


# ---------------------------------------------------------------------------
# 10. the global run slot
# ---------------------------------------------------------------------------


def test_a_second_run_while_one_is_active_is_refused(tmp_path, fake_tools):
    gate = threading.Event()
    adapter = FakeAdapter([(ANSWER_CANARY, {})], gate=gate)
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_e2e_doc(), _registry(adapter))
    first = runner.start(spec, "manual")
    try:
        # the slot is taken synchronously inside start(), before the thread
        # exists — so this is deterministic, no racing the worker.
        with pytest.raises(ProviderError) as ei:
            runner.start(spec, "manual")
        assert "already active" in str(ei.value)
        assert str(ei.value) == "an agent run is already active; try again later"
        # the refused start left no file behind
        assert len(list((tmp_path / "runs").glob("*.jsonl"))) == 1
    finally:
        gate.set()
    _, status, _ = _drain(runner, first)
    assert status == "done"
    # and the slot is released for the next run. The terminal `done` line is
    # written a hair BEFORE the worker thread reaches its finally-block release,
    # so a start() fired the instant _drain reports done can still lose the race
    # to the not-yet-run release — those two events were never atomic. Retry
    # briefly: this asserts the slot RECYCLES, not that release is synchronous
    # with the terminal line (which it isn't, and the "refused while active"
    # guarantee above doesn't depend on).
    deadline = time.monotonic() + 5.0
    while True:
        try:
            second = runner.start(spec, "manual")
            break
        except ProviderError as exc:
            assert "already active" in str(exc)
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)
    assert _drain(runner, second)[1] == "done"


def test_start_refuses_a_spec_whose_provider_left_the_registry(tmp_path):
    runner = AgentRunner(tmp_path / "runs", {}, tmp_path / "audit.jsonl")
    spec = AgentSpec(graph_id="g", graph_rev=1, agent_name="A",
                     provider_id="ghost", model="m", transport=None,
                     system="", params={}, tool_specs=[],
                     limits=dict(_DEFAULT_LIMITS), input="hi")
    with pytest.raises(ProviderError) as ei:
        runner.start(spec, "manual")
    assert str(ei.value) == "unknown provider: 'ghost'"
    assert list((tmp_path / "runs").glob("*.jsonl")) == []


# ---------------------------------------------------------------------------
# 11. multi-agent graphs — compile
# ---------------------------------------------------------------------------
#
# The docs below are the multi-node shape the swap exists to enable: two agent
# nodes wired next→in, a router with either mode, and a cycle. One shared engine
# node feeds both agents (that's how the canvas draws it — one engine, two
# edges), so a single FakeAdapter's sequential script scripts the whole run.

RESEARCH_CANARY = "RESEARCHER-SYSTEM-CANARY"
WRITER_CANARY = "WRITER-SYSTEM-CANARY"


def _chain_doc() -> dict:
    """trigger→A, A next→B, B→output; one engine feeding both agents."""
    return {
        "id": "g_chain",
        "rev": 1,
        "nodes": [
            {"id": "t", "type": "trigger",
             "config": {"mode": "manual", "input": "GO"}},
            {"id": "eng", "type": "engine",
             "config": {"provider": "fake", "model": "m"}},
            {"id": "A", "type": "agent",
             "config": {"name": "RESEARCHER", "system": RESEARCH_CANARY,
                        "limits": {"max_turns": 4}}},
            {"id": "B", "type": "agent",
             "config": {"name": "WRITER", "system": WRITER_CANARY,
                        "limits": {"max_turns": 4}}},
            {"id": "out", "type": "output", "config": {}},
        ],
        "edges": [
            {"source": "t", "target": "A",
             "sourceHandle": "run", "targetHandle": "trigger"},
            {"source": "eng", "target": "A",
             "sourceHandle": "engine", "targetHandle": "engine"},
            {"source": "eng", "target": "B",
             "sourceHandle": "engine", "targetHandle": "engine"},
            {"source": "A", "target": "B",
             "sourceHandle": "next", "targetHandle": "in"},
            {"source": "B", "target": "out",
             "sourceHandle": "out", "targetHandle": "in"},
        ],
    }


def _router_doc(mode: str, branches: list, else_branch: str,
                targets: dict) -> dict:
    """trigger→A, A next→router, router branches→{agents|output}.

    *targets* maps a branch name to a node id ("B", "C" or "out"). Agents B and
    C share the one engine with A.
    """
    edges = [
        {"source": "t", "target": "A",
         "sourceHandle": "run", "targetHandle": "trigger"},
        {"source": "eng", "target": "A",
         "sourceHandle": "engine", "targetHandle": "engine"},
        {"source": "eng", "target": "B",
         "sourceHandle": "engine", "targetHandle": "engine"},
        {"source": "eng", "target": "C",
         "sourceHandle": "engine", "targetHandle": "engine"},
        {"source": "A", "target": "R",
         "sourceHandle": "next", "targetHandle": "in"},
        {"source": "B", "target": "out",
         "sourceHandle": "out", "targetHandle": "in"},
        {"source": "C", "target": "out",
         "sourceHandle": "out", "targetHandle": "in"},
    ]
    for i, b in enumerate(branches):
        edges.append({"source": "R", "target": targets[b["name"]],
                      "sourceHandle": f"case_{i}", "targetHandle": "in"})
    return {
        "id": "g_router",
        "rev": 1,
        "nodes": [
            {"id": "t", "type": "trigger",
             "config": {"mode": "manual", "input": "GO"}},
            {"id": "eng", "type": "engine",
             "config": {"provider": "fake", "model": "m"}},
            {"id": "A", "type": "agent",
             "config": {"name": "A", "system": "", "limits": {"max_turns": 4}}},
            {"id": "B", "type": "agent",
             "config": {"name": "B", "system": "", "limits": {"max_turns": 4}}},
            {"id": "C", "type": "agent",
             "config": {"name": "C", "system": "", "limits": {"max_turns": 4}}},
            {"id": "R", "type": "router",
             "config": {"mode": mode, "branches": branches,
                        "else": else_branch}},
            {"id": "out", "type": "output", "config": {}},
        ],
        "edges": edges,
    }


def test_compile_two_agent_chain_wires_the_flow():
    spec = compile_graph(_chain_doc(), _registry())
    assert isinstance(spec, GraphSpec)
    assert spec.entry == "A"
    assert list(spec.agents) == ["A", "B"]           # entry first
    assert spec.agents["A"].system == RESEARCH_CANARY
    assert spec.agents["B"].system == WRITER_CANARY
    assert spec.flow_edges["A"] == ["B"]
    assert spec.flow_edges["B"] == ["out"]            # terminal → END
    # the head line still reports the ENTRY agent
    assert spec.provider_id == "fake"
    assert spec.agent_name == "RESEARCHER"
    assert [a["name"] for a in spec.as_start_meta()["agents"]] == \
        ["RESEARCHER", "WRITER"]


def test_compile_router_records_branches_else_and_feeder():
    branches = [{"name": "esc", "when": "ERROR"},
                {"name": "ok", "when": ".*"}]
    spec = compile_graph(
        _router_doc("decide", branches, "ok",
                    {"esc": "B", "ok": "C"}), _registry())
    assert isinstance(spec, GraphSpec)
    router = spec.routers["R"]
    assert isinstance(router, RouterSpec)
    assert router.mode == "decide"
    assert router.else_branch == "ok"
    assert router.source_agent == "A"
    assert {b["name"]: b["target_node"] for b in router.branches} == \
        {"esc": "B", "ok": "C"}


def test_compile_router_else_must_name_a_real_branch():
    branches = [{"name": "esc", "when": "ERROR"}]
    with pytest.raises(ProviderError) as ei:
        compile_graph(_router_doc("decide", branches, "nope",
                                  {"esc": "B"}), _registry())
    assert str(ei.value) == \
        "router 'R' needs an 'else' naming one of its branches (esc)"


def test_compile_router_branch_target_must_exist():
    branches = [{"name": "esc", "when": "ERROR"}]
    doc = _router_doc("decide", branches, "esc", {"esc": "B"})
    # point the branch edge at a node id that isn't on the canvas
    for e in doc["edges"]:
        if e["source"] == "R":
            e["target"] = "ghost"
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert "branch 'esc'" in str(ei.value)


def test_compile_rejects_an_unreachable_agent():
    # B has an engine and a name but nothing wires into it — a silent dead node.
    doc = _chain_doc()
    doc["edges"] = [e for e in doc["edges"]
                    if not (e["source"] == "A" and e["target"] == "B")]
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert str(ei.value) == "agent 'WRITER' is not reachable from the trigger"


def test_compile_rejects_agent_fanning_out_without_a_router():
    doc = _chain_doc()
    doc["nodes"].append({"id": "C", "type": "agent",
                         "config": {"name": "THIRD", "limits": {}}})
    doc["edges"].append({"source": "eng", "target": "C",
                         "sourceHandle": "engine", "targetHandle": "engine"})
    # A now hands off to BOTH B and C directly — ambiguous control flow.
    doc["edges"].append({"source": "A", "target": "C",
                         "sourceHandle": "next", "targetHandle": "in"})
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert "more than one outgoing control-flow edge" in str(ei.value)


def test_compile_reads_max_steps_off_the_trigger_and_clamps_it():
    from memsom.providers.agents import _MAX_STEPS_CEILING
    doc = _chain_doc()
    _node(doc, "t")["config"]["limits"] = {"max_steps": 999}
    assert compile_graph(doc, _registry()).limits["max_steps"] == \
        _MAX_STEPS_CEILING
    _node(doc, "t")["config"]["limits"] = {"max_steps": 6}
    assert compile_graph(doc, _registry()).limits["max_steps"] == 6


# ---------------------------------------------------------------------------
# 12. multi-agent graphs — end to end on the LangGraph runtime
# ---------------------------------------------------------------------------


def test_two_agent_chain_runs_both_nodes_on_a_shared_thread(tmp_path):
    # One shared adapter: call 0 is RESEARCHER, call 1 is WRITER.
    adapter = FakeAdapter([("A-SPOKE", {"eval_count": 3}),
                           ("B-SPOKE", {"eval_count": 4})])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_chain_doc(), _registry(adapter))
    events, status, stats = _drain(runner, runner.start(spec, "manual"))
    assert status == "done"

    # both nodes announced, in order, each attributed to its own agent
    nodes = [(e["id"], e["agent"]) for e in events if e["t"] == "node"]
    assert nodes == [("A", "RESEARCHER"), ("B", "WRITER")]
    assert [e.get("node") for e in events if e["t"] == "turn"] == ["A", "B"]
    assert stats["turns"] == 2
    assert stats["eval_count"] == 7                   # summed across both agents

    # shared thread, private prompt: WRITER's call carries its OWN system and
    # sees RESEARCHER's output, but never RESEARCHER's system prompt.
    writer_msgs = adapter.calls[1][1]
    assert writer_msgs[0]["role"] == "system"
    assert writer_msgs[0]["content"] == WRITER_CANARY
    assert RESEARCH_CANARY not in json.dumps(writer_msgs)
    assert any(m["role"] == "assistant" and m["content"] == "A-SPOKE"
               for m in writer_msgs)


class _RouteAdapter(FakeAdapter):
    """A shared adapter that speaks for both agents AND the decide router.

    The router's inference is the one turn that asks for the synthetic ``route``
    tool, so it's identified by the tool being on offer — not by counting calls,
    which a retry would throw off. ``branch`` is what it picks."""

    def __init__(self, branch: str) -> None:
        super().__init__([("agent spoke", {})])
        self.branch = branch

    def infer(self, model, messages, params, sink):
        self.calls.append((model, copy.deepcopy(messages), dict(params)))
        offered = {t.get("function", {}).get("name")
                   for t in (params.get("tools") or [])}
        if "route" in offered:
            return {"tool_calls": [{"id": "tc_r", "name": "route",
                                    "arguments": {"branch": self.branch}}]}
        sink.token("agent spoke")
        return {}


@pytest.mark.parametrize("branch,target", [("esc", "B"), ("ok", "C")])
def test_decide_router_takes_each_branch(tmp_path, branch, target):
    branches = [{"name": "esc", "when": "an error happened"},
                {"name": "ok", "when": "all clear"}]
    adapter = _RouteAdapter(branch)
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(
        _router_doc("decide", branches, "ok", {"esc": "B", "ok": "C"}),
        _registry(adapter))
    events, status, _ = _drain(runner, runner.start(spec, "manual"))
    assert status == "done"

    route = next(e for e in events if e["t"] == "route")
    assert route["router"] == "R"
    assert route["mode"] == "decide"
    assert route["branch"] == branch
    # the chosen agent ran; the other did not
    ran = {e["id"] for e in events if e["t"] == "node"}
    assert target in ran
    assert ({"B", "C"} - {target}).pop() not in ran


def test_match_router_routes_on_the_previous_agents_text(tmp_path):
    branches = [{"name": "esc", "when": "ERROR"},
                {"name": "ok", "when": "."}]
    # A's text contains ERROR → the first branch matches, no inference needed.
    adapter = FakeAdapter([("something ERROR happened", {}),
                           ("B handled it", {})])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(
        _router_doc("match", branches, "ok", {"esc": "B", "ok": "C"}),
        _registry(adapter))
    events, status, _ = _drain(runner, runner.start(spec, "manual"))
    assert status == "done"
    route = next(e for e in events if e["t"] == "route")
    assert route["mode"] == "match"
    assert route["branch"] == "esc"
    assert "B" in {e["id"] for e in events if e["t"] == "node"}


def test_match_router_falls_through_to_else_on_no_hit(tmp_path):
    branches = [{"name": "esc", "when": "ERROR"},
                {"name": "ok", "when": "WONT-MATCH-THIS"}]
    adapter = FakeAdapter([("nothing notable", {}), ("C handled it", {})])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(
        _router_doc("match", branches, "ok", {"esc": "B", "ok": "C"}),
        _registry(adapter))
    events, status, _ = _drain(runner, runner.start(spec, "manual"))
    assert status == "done"
    assert next(e for e in events if e["t"] == "route")["branch"] == "ok"


def _cycle_doc(max_steps: int) -> dict:
    """trigger→A, A next→router, router's only branch loops back to A.

    Deliberately minimal — one agent, one router, no orphan nodes — so the only
    thing that can end the run is the step budget, not a reachability error."""
    return {
        "id": "g_cycle",
        "rev": 1,
        "nodes": [
            {"id": "t", "type": "trigger",
             "config": {"mode": "manual", "input": "GO",
                        "limits": {"max_steps": max_steps}}},
            {"id": "eng", "type": "engine",
             "config": {"provider": "fake", "model": "m"}},
            {"id": "A", "type": "agent",
             "config": {"name": "A", "system": "", "limits": {"max_turns": 99}}},
            {"id": "R", "type": "router",
             "config": {"mode": "match",
                        "branches": [{"name": "again", "when": "."}],
                        "else": "again"}},
        ],
        "edges": [
            {"source": "t", "target": "A",
             "sourceHandle": "run", "targetHandle": "trigger"},
            {"source": "eng", "target": "A",
             "sourceHandle": "engine", "targetHandle": "engine"},
            {"source": "A", "target": "R",
             "sourceHandle": "next", "targetHandle": "in"},
            {"source": "R", "target": "A",
             "sourceHandle": "case_0", "targetHandle": "in"},
        ],
    }


def test_a_cycle_trips_max_steps_and_ends_in_a_clean_error(tmp_path):
    # A match router that always loops back to A. Every lap is a plain-text
    # turn (one entry, repeated), so nothing converges and the graph-level
    # step budget is the only thing that ends it.
    adapter = FakeAdapter([("still going", {})])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_cycle_doc(4), _registry(adapter))
    events, status, _ = _drain(runner, runner.start(spec, "manual"))
    assert status == "error"
    assert events[-1]["t"] == "error"
    assert events[-1]["error"] == \
        "step limit reached (4) without finishing the graph"


def test_a_missing_langgraph_extra_is_a_named_provider_error(tmp_path,
                                                             monkeypatch):
    # The runtime import is lazy so the core stays stdlib-only; when the extra
    # isn't installed, a run must fail with an actionable message naming it,
    # not an ImportError traceback at some random depth.
    import memsom.providers.lc_runtime as rt

    def _boom():
        raise ProviderError(
            "the agent graph runtime needs langgraph — install the optional "
            "extra: pip install 'memsom[agents]'")

    monkeypatch.setattr(rt, "_lc", _boom)
    adapter = FakeAdapter([("hi", {})])
    runner = _runner(tmp_path, adapter)
    spec = compile_graph(_chain_doc(), _registry(adapter))
    _, status, stats = _drain(runner, runner.start(spec, "manual"))
    assert status == "error"
    assert "pip install 'memsom[agents]'" in stats["error"]


# ---------------------------------------------------------------------------
# 9. human-in-the-loop approval gates + resume  (interrupt/checkpointer)
# ---------------------------------------------------------------------------


def _gated_doc(require: bool = True) -> dict:
    doc = _e2e_doc()
    _node(doc, "n5")["config"] = {"tool": "fake_tool", "options": {},
                                  "require_approval": require}
    return doc


def _approval_adapter() -> FakeAdapter:
    # turn 1 reaches for the gated tool; turn 2 (after it clears) answers.
    return FakeAdapter([
        ("", {"tool_calls": [{"id": "tc_1", "name": "fake_tool",
                              "arguments": {"q": "hi"}}]}),
        (ANSWER_CANARY, {}),
    ])


def _settle(runner: AgentRunner, rid: str, *stop: str,
            timeout: float = 10.0) -> tuple:
    """Poll until the run reaches one of *stop*. Unlike ``_drain`` this also
    stops at ``paused``/``resumable`` — an approval pause is not a terminal
    state, so ``_drain`` would (correctly) time out on it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = runner.read_since(rid, 0)
        if r["status"] in stop:
            return r["events"], r["status"], r.get("stats")
        time.sleep(0.02)
    raise AssertionError(
        f"run {rid} never reached {stop}; "
        f"last status={runner.read_since(rid, 0)['status']}")


def _tool_audit(tmp_path: Path) -> list:
    p = tmp_path / "audit.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("action") == "tool":
            out.append(rec.get("result"))
    return out


def test_a_gated_tool_pauses_the_run_and_does_not_execute(tmp_path, fake_tools):
    adapter = _approval_adapter()
    runner = _runner(tmp_path, adapter)
    rid = runner.start(compile_graph(_gated_doc(), _registry(adapter)), "manual")
    events, status, _ = _settle(runner, rid, "paused")
    assert status == "paused"
    aa = [e for e in events if e["t"] == "awaiting_approval"]
    assert len(aa) == 1
    assert aa[0]["tool"] == "fake_tool"
    assert aa[0]["arguments"] == {"q": "hi"}
    # nothing ran: no tool_result, and the audit log has no execution line yet
    assert not [e for e in events if e["t"] == "tool_result"]
    assert _tool_audit(tmp_path) == []


def test_deny_finishes_the_run_without_executing_the_tool(tmp_path, fake_tools):
    adapter = _approval_adapter()
    runner = _runner(tmp_path, adapter)
    rid = runner.start(compile_graph(_gated_doc(), _registry(adapter)), "manual")
    _settle(runner, rid, "paused")
    runner.resume(rid, "deny")
    events, status, _ = _settle(runner, rid, "done", "error")
    assert status == "done"
    tr = [e for e in events if e["t"] == "tool_result"]
    assert len(tr) == 1 and tr[0]["ok"] is False and "DENIED" in tr[0]["output"]
    # the refusal IS the security event, and it is audited
    assert _tool_audit(tmp_path) == ["refused-by-user"]
    assert [e["decision"] for e in events if e["t"] == "approval"] == ["deny"]


def test_approve_executes_the_tool_exactly_once(tmp_path, fake_tools):
    adapter = _approval_adapter()
    runner = _runner(tmp_path, adapter)
    rid = runner.start(compile_graph(_gated_doc(), _registry(adapter)), "manual")
    _settle(runner, rid, "paused")
    runner.resume(rid, "approve")
    events, status, _ = _settle(runner, rid, "done", "error")
    assert status == "done"
    trs = [e for e in events if e["t"] == "tool_result"]
    assert len(trs) == 1 and trs[0]["ok"] is True
    # executed once, with the ordinary two-phase audit — not the refusal line
    assert _tool_audit(tmp_path) == ["pending", "ok"]


def test_shell_tool_defaults_to_requiring_approval():
    doc = _e2e_doc()
    _node(doc, "n5")["config"] = {"tool": "shell", "options": {}}  # no flag set
    spec = compile_graph(doc, _registry())
    assert spec.tool_specs[0]["require_approval"] is True


def test_an_explicit_flag_overrides_the_shell_default():
    doc = _e2e_doc()
    _node(doc, "n5")["config"] = {"tool": "shell", "options": {},
                                  "require_approval": False}
    spec = compile_graph(doc, _registry())
    assert spec.tool_specs[0]["require_approval"] is False


def test_a_paused_run_that_lost_its_memory_is_resumable_and_resumes(
        tmp_path, fake_tools):
    # A server restart drops _paused but not the checkpoint. The run must then
    # read as `resumable`, and a resume with a recompiled spec (what the handler
    # rebuilds from the graph doc) must continue it off the checkpoint.
    adapter = _approval_adapter()
    reg = _registry(adapter)
    gdoc = _gated_doc()
    runner = _runner(tmp_path, adapter)
    rid = runner.start(compile_graph(gdoc, reg), "manual")
    _settle(runner, rid, "paused")

    runner._paused.clear()  # simulate the restart
    assert runner.read_since(rid, 0)["status"] == "resumable"

    runner.resume(rid, "approve", spec=compile_graph(gdoc, reg))
    events, status, _ = _settle(runner, rid, "done", "error")
    assert status == "done"
    assert [e for e in events if e["t"] == "tool_result"][0]["ok"] is True


def test_resume_refuses_when_no_checkpoint_survives(tmp_path, fake_tools):
    adapter = _approval_adapter()
    reg = _registry(adapter)
    runner = _runner(tmp_path, adapter)
    rid = runner.start(compile_graph(_gated_doc(), reg), "manual")
    _settle(runner, rid, "paused")
    # a resume against a run whose checkpoint is gone must refuse, not silently
    # start the graph over from the top
    runner._paused.clear()
    import sqlite3
    con = sqlite3.connect(str(runner.checkpoints))
    con.execute("DELETE FROM checkpoints WHERE thread_id=?", (rid,))
    con.commit()
    con.close()
    with pytest.raises(ProviderError) as ei:
        runner.resume(rid, "approve", spec=compile_graph(_gated_doc(), reg))
    assert "no checkpoint" in str(ei.value)


# ---------------------------------------------------------------------------
# 10. structured output, context hooks, shared state
# ---------------------------------------------------------------------------

_SCHEMA = {"title": "Verdict", "type": "object",
           "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
           "required": ["ok", "why"]}


def test_output_schema_must_be_an_object():
    doc = _doc()  # real http_fetch tool — reaches the schema check
    _node(doc, "n6")["config"]["output_schema"] = "not a dict"
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert "output_schema must be a JSON-schema object" in str(ei.value)


def test_context_mode_is_validated():
    doc = _doc()
    _node(doc, "n6")["config"]["context_mode"] = "hallucinate"
    with pytest.raises(ProviderError) as ei:
        compile_graph(doc, _registry())
    assert "context_mode must be" in str(ei.value)


def _no_tool_doc() -> dict:
    """The e2e chassis with the tool node removed — a plain text agent, so the
    ReAct loop ends on a plain answer and the structured step is the only call
    that binds a tool."""
    doc = _doc()
    doc["nodes"] = [n for n in doc["nodes"] if n["id"] != "n5"]
    doc["edges"] = [e for e in doc["edges"] if e["id"] != "e3"]
    return doc


def test_structured_output_lands_in_the_run(tmp_path):
    # Schema-aware adapter: with no agent tools the ReAct loop answers plainly,
    # then LangGraph's structured step binds the schema tool — THAT is the only
    # call with a tool bound, and the adapter answers it with valid data.
    class Schematic(FakeAdapter):
        def infer(self, model, messages, params, sink):
            self.calls.append((model, [], dict(params)))
            if params.get("tools"):
                name = params["tools"][0]["function"]["name"]
                return {"tool_calls": [{"id": "s1", "name": name,
                                        "arguments": {"ok": True, "why": "clean"}}]}
            sink.token("looks fine")
            return {}

    doc = _no_tool_doc()
    _node(doc, "n6")["config"]["output_schema"] = _SCHEMA
    adapter = Schematic([])
    runner = _runner(tmp_path, adapter)
    events, status, stats = _drain(runner, runner.start(
        compile_graph(doc, _registry(adapter)), "manual"))
    assert status == "done"
    structured = [e for e in events if e["t"] == "structured"]
    assert len(structured) == 1
    assert structured[0]["data"] == {"ok": True, "why": "clean"}
    assert stats["structured"] == {"ok": True, "why": "clean"}


def test_shared_state_passes_a_value_between_agents(tmp_path, fake_tools):
    # WRITER stores a value; READER (a later node) reads it back — proving the
    # scratchpad is one object across the whole run.
    class Scripted(FakeAdapter):
        def __init__(self):
            super().__init__([])
            self.script2 = [
                {"tool_calls": [{"id": "c1", "name": "state_set",
                                 "arguments": {"key": "finding", "value": "42"}}]},
                {},
                {"tool_calls": [{"id": "c2", "name": "state_get",
                                 "arguments": {"key": "finding"}}]},
                {},
            ]
            self.i = 0

        def infer(self, model, messages, params, sink):
            step = self.script2[min(self.i, len(self.script2) - 1)]
            self.i += 1
            if step.get("tool_calls"):
                return dict(step)
            sink.token("ok")
            return {}

    adapter = Scripted()
    runner = _runner(tmp_path, adapter)
    events, status, _ = _drain(runner, runner.start(
        compile_graph(_shared_state_doc(), _registry(adapter)), "manual"))
    assert status == "done"
    reads = [e for e in events if e["t"] == "tool_result" and e["name"] == "state_get"]
    assert reads and reads[0]["output"] == '"42"'


def _shared_state_doc() -> dict:
    return {"id": "g_ss", "rev": 1, "nodes": [
        {"id": "t", "type": "trigger", "config": {"mode": "manual", "input": "go"}},
        {"id": "e", "type": "engine",
         "config": {"provider": "fake", "model": "m"}},
        {"id": "ts", "type": "tool", "config": {"tool": "state_set", "options": {}}},
        {"id": "tg", "type": "tool", "config": {"tool": "state_get", "options": {}}},
        {"id": "a1", "type": "agent",
         "config": {"name": "WRITER", "system": "", "params": {},
                    "limits": {"max_turns": 4}}},
        {"id": "a2", "type": "agent",
         "config": {"name": "READER", "system": "", "params": {},
                    "limits": {"max_turns": 4}}},
        {"id": "o", "type": "output", "config": {}}],
        "edges": [
        {"id": "1", "source": "t", "target": "a1",
         "sourceHandle": "run", "targetHandle": "trigger"},
        {"id": "2", "source": "e", "target": "a1",
         "sourceHandle": "engine", "targetHandle": "engine"},
        {"id": "3", "source": "ts", "target": "a1",
         "sourceHandle": "tool", "targetHandle": "tools"},
        {"id": "4", "source": "e", "target": "a2",
         "sourceHandle": "engine", "targetHandle": "engine"},
        {"id": "5", "source": "tg", "target": "a2",
         "sourceHandle": "tool", "targetHandle": "tools"},
        {"id": "6", "source": "a1", "target": "a2",
         "sourceHandle": "next", "targetHandle": "in"},
        {"id": "7", "source": "a2", "target": "o",
         "sourceHandle": "out", "targetHandle": "in"}]}
