"""The graph runtime — a compiled canvas, executed on LangGraph.

``compile_graph`` produces a shape (:class:`GraphSpec`); this module turns that
shape into something that runs. One ``StateGraph`` node per agent node, each of
them a ``create_react_agent`` subgraph over a :class:`MemsomChatModel` bound to
that agent's engine and its own tools; routers become conditional edges; an
output node — or an agent with nothing downstream — becomes ``END``.

Why LangGraph owns the loop now: the infer→execute-tools→infer cycle, its
retry and limit semantics, and every future improvement to it stop being ours
to maintain. What we deliberately did NOT hand over is the run log. The JSONL
file is a contract (``AgentRunner.read_since`` cursor-walks it, RunMonitor
renders it), so the events are emitted from our own code — inside
:class:`MemsomChatModel` and :class:`MemsomTool` — rather than reconstructed
from ``.stream()``, whose vocabulary is LangGraph's to change.

Three design points worth stating out loud:

* **Shared thread, private prompt.** State is ``MessagesState``: one message
  list every agent reads. Each node prepends only ITS OWN system prompt for its
  own call (``create_react_agent(prompt=…)`` applies it at the model, not to
  the state) and appends only what it produced. That is what lets a RESEARCHER
  and a WRITER with opposite instructions share one conversation.
* **Synchronous.** ``.invoke``, never ``.ainvoke``. This runs on an
  ``AgentRunner`` thread inside the panel server, which already has a scheduler
  daemon thread; starting an event loop there to await a call that is
  blocking anyway would buy nothing and cost a whole class of shutdown bug.
* **No checkpointer.** The run log is the record of what happened. A
  ``SqliteSaver`` alongside it would be a second answer to "what did this run
  do", and two sources of truth is how they start disagreeing.

Everything langgraph/langchain is imported lazily inside these functions.
memsom's core is stdlib-only, so a machine that never runs an agent never pays
for the import, and one that tries without the extra installed gets a
:class:`ProviderError` naming what to install rather than a traceback.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memsom.providers.base import ProviderError, Sink, now
from memsom.providers.tools import build_tools

__all__ = ["build_state_graph", "run_graph"]

#: name of the synthetic tool a "decide" router hands the model. Not a real
#: tool: it is never registered, never executed, and exists only so the choice
#: comes back through the canonical stats["tool_calls"] path every adapter
#: already parses, instead of through prose we would have to guess at.
_ROUTE_TOOL = "route"

_MISSING_EXTRA = (
    "the agent graph runtime needs langgraph — install the optional extra: "
    "pip install 'memsom[agents]'"
)


def _lc() -> SimpleNamespace:
    """Resolve every langgraph/langchain symbol this module needs, or explain.

    One entry point for the whole lazy import so there is exactly one place
    that can raise the "install the extra" message, and exactly one import
    cost to pay on the first run of a server's life."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage
        from langgraph.errors import GraphRecursionError
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import create_react_agent

        from memsom.providers.lc_model import (
            MemsomChatModel, MemsomTool, RunContext, to_memsom_messages,
        )
    except ImportError as exc:
        raise ProviderError(_MISSING_EXTRA) from exc
    return SimpleNamespace(
        AIMessage=AIMessage, HumanMessage=HumanMessage,
        GraphRecursionError=GraphRecursionError, END=END, START=START,
        MessagesState=MessagesState, StateGraph=StateGraph,
        create_react_agent=create_react_agent,
        MemsomChatModel=MemsomChatModel, MemsomTool=MemsomTool,
        RunContext=RunContext, to_memsom_messages=to_memsom_messages,
    )


class _QuietSink(Sink):
    """Swallows tokens. Used for a router's own inference.

    A ``decide`` router runs a real inference, but its output is a routing
    decision, not part of the transcript — streaming its tokens into the run
    log would show the user a turn that no agent took, and inflate the token
    count the ``done`` line reports. The usage counters ARE folded into the
    run's stats, because those tokens were genuinely generated."""

    def token(self, text: str) -> None:
        return


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


def build_state_graph(spec, registry: dict, ctx) -> Any:
    """Compile *spec* into a runnable ``CompiledStateGraph``.

    *ctx* is the run's :class:`RunContext` — shared by every node's model and
    every tool wrapper, which is what makes turn numbers continuous across
    nodes and the terminal ``done`` line describe the whole graph rather than
    whichever agent happened to speak last.
    """
    lc = _lc()
    builder = lc.StateGraph(lc.MessagesState)
    for node_id, agent in spec.agents.items():
        builder.add_node(node_id, _agent_node(lc, spec, node_id, agent,
                                              registry, ctx))
    builder.add_edge(lc.START, spec.entry)

    for node_id in spec.agents:
        successors = spec.flow_edges.get(node_id) or []
        target = successors[0] if successors else None
        if target in spec.routers:
            router = spec.routers[target]
            # The path map is branch-name → node name, with any branch that
            # lands on an output node collapsing to END. Two branches may share
            # a target; LangGraph is fine with that and the canvas allows it.
            path_map = {
                branch["name"]: (branch["target_node"]
                                 if branch["target_node"] in spec.agents
                                 else lc.END)
                for branch in router.branches
            }
            builder.add_conditional_edges(
                node_id, _router_fn(lc, spec, router, registry, ctx), path_map)
        elif target in spec.agents:
            builder.add_edge(node_id, target)
        else:
            # No successor, or an output node: the run ends here.
            builder.add_edge(node_id, lc.END)
    return builder.compile()


def _agent_node(lc, spec, node_id: str, agent, registry: dict, ctx):
    """One canvas agent → one parent-graph node wrapping a ReAct subgraph.

    The subgraph is built ONCE, here, not per invocation: a cycle that returns
    to this agent would otherwise rebuild the model, re-render the tools and
    re-run ``bind_tools`` on every lap, which is pure overhead in the one case
    that already costs the most.
    """
    adapter = registry.get(agent.provider_id)
    if adapter is None:
        raise ProviderError(f"unknown provider: {agent.provider_id!r}")
    model = lc.MemsomChatModel(adapter=adapter, model=agent.model,
                               params=dict(agent.params), ctx=ctx,
                               node_id=node_id)
    tools = [lc.MemsomTool(tool, ctx) for tool in build_tools(agent.tool_specs)]
    subgraph = lc.create_react_agent(model, tools,
                                     prompt=agent.system or None)
    # The subgraph gets the same budget as the parent. Without it, an agent
    # whose model keeps reaching for tools would trip langgraph's default
    # recursion limit instead of the one the user set on the trigger.
    #
    # max_concurrency=1 forces langgraph's ToolNode to run a turn's tool calls
    # SEQUENTIALLY (its executor is get_executor_for_config, whose worker count
    # is exactly this value). memsom's audit log, the run's counters and the
    # loop detector were all built for the sequential model run_tool_loop uses;
    # letting the tool node fan calls across a thread pool corrupts the fsync'd
    # audit appends and races the counters. A single agent run holds one GPU
    # slot anyway, so parallel local tool calls buy little and cost correctness.
    config = {"recursion_limit": spec.limits["max_steps"], "max_concurrency": 1}
    agent_name = agent.agent_name

    def run_node(state: dict) -> dict:
        ctx.node_id = node_id
        ctx.sink.event({"t": "node", "id": node_id, "agent": agent_name,
                        "ts": now()})
        prior = len(state["messages"])
        result = subgraph.invoke({"messages": state["messages"]}, config)
        # Append only what this node produced. The inbound messages come back
        # out of the subgraph unchanged, and handing them to the parent's
        # add_messages reducer again would rely on id-dedup to stay correct.
        return {"messages": result["messages"][prior:]}

    return run_node


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def _router_fn(lc, spec, router, registry: dict, ctx):
    """The conditional-edge function for one router node.

    Contract: it ALWAYS returns a valid branch name. An undecidable result —
    no match, a model that answered with prose, an engine that fell over
    mid-decision — takes the else branch. A router that could raise would turn
    a fork in the road into a dead run, discarding whatever work the graph had
    already banked.
    """
    names = [branch["name"] for branch in router.branches]

    def route(state: dict) -> str:
        if router.mode == "match":
            choice = _match_branch(lc, router, state)
        else:
            choice = _decide_branch(lc, spec, router, registry, ctx, state)
        if choice not in names:
            choice = router.else_branch
        ctx.sink.event({"t": "route", "router": router.node_id,
                        "branch": choice, "mode": router.mode, "ts": now()})
        return choice

    return route


def _last_text(lc, state: dict) -> str:
    """The previous agent's final answer, as plain text."""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, lc.AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content)
    return ""


def _match_branch(lc, router, state: dict) -> str:
    """First branch whose ``when`` regex hits the previous agent's text.

    Free — no inference — and the weakest link in the whole design, because it
    is a regex against model prose. That is exactly why ``decide`` is the
    default mode: ``match`` is for when the upstream agent was instructed to
    end with a literal token, not for reading intent."""
    text = _last_text(lc, state)
    for branch in router.branches:
        pattern = branch.get("when") or ""
        if not pattern:
            continue
        try:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return branch["name"]
        except re.error:
            # A malformed pattern is a config mistake, not a run-ending one:
            # skip it and let the remaining branches (or the else) decide.
            continue
    return ""


def _decide_branch(lc, spec, router, registry: dict, ctx, state: dict) -> str:
    """Ask the feeding agent's engine to name a branch.

    One inference, one synthetic tool whose only argument is an enum of the
    branch names. Going through a tool call rather than asking for a bare word
    means the answer arrives in ``stats["tool_calls"]`` — the same canonical
    path every adapter already normalises — instead of in free text we would
    have to parse and could get wrong in five provider-specific ways.

    It borrows the FEEDING agent's engine rather than owning one because a
    router has no engine handle on the canvas, and the agent that just spoke is
    by definition loaded and warm.
    """
    agent = spec.agents.get(router.source_agent)
    adapter = registry.get(agent.provider_id) if agent else None
    if adapter is None:
        return ""
    names = [branch["name"] for branch in router.branches]
    catalogue = "\n".join(
        f"- {branch['name']}: {branch['when'] or 'no description given'}"
        for branch in router.branches)
    ask = (
        "Routing decision. Based on the conversation so far, choose exactly "
        f"one of these branches by calling the {_ROUTE_TOOL} tool:\n"
        f"{catalogue}\n"
        "Answer only with the tool call."
    )
    messages = lc.to_memsom_messages(state.get("messages") or [])
    messages.append({"role": "user", "content": ask})
    params = {**dict(agent.params), "tools": [_route_tool(names)]}
    # Streaming a routing decision would push tokens at a sink that throws them
    # away, and some adapters take a slower path to do it.
    params.pop("stream", None)

    try:
        stats = adapter.infer(agent.model, messages, params, _QuietSink()) or {}
    except Exception:
        # Deliberately broad, and deliberately silent: see _router_fn's
        # contract. If the engine is genuinely gone, the next agent node's own
        # inference raises with a real message a moment later.
        return ""
    ctx.accumulate(stats)
    for call in stats.get("tool_calls") or []:
        arguments = call.get("arguments")
        if isinstance(arguments, dict) and arguments.get("branch") in names:
            return str(arguments["branch"])
    return ""


def _route_tool(names: list) -> dict:
    """The synthetic ``route`` tool, in the OpenAI wire shape."""
    return {
        "type": "function",
        "function": {
            "name": _ROUTE_TOOL,
            "description": "Select the branch the conversation should take.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "enum": list(names),
                               "description": "The name of the chosen branch."},
                },
                "required": ["branch"],
            },
        },
    }


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def run_graph(spec, registry: dict, sink, audit_path) -> dict:
    """Build and run *spec* to completion; return the aggregated stats.

    The return shape is what ``_final_stats`` expects and what the terminal
    ``done`` line has always carried — ``turns``, ``tool_calls``, plus whatever
    usage counters the backends reported — so the frontend reads a graph run
    and a legacy single-agent run through the same code.
    """
    lc = _lc()
    ctx = lc.RunContext(sink=sink, audit_path=Path(audit_path),
                        limits=dict(spec.limits))
    graph = build_state_graph(spec, registry, ctx)
    max_steps = spec.limits["max_steps"]
    try:
        graph.invoke({"messages": [lc.HumanMessage(content=spec.input or "Begin.")]},
                     {"recursion_limit": max_steps, "max_concurrency": 1})
    except lc.GraphRecursionError as exc:
        # A cycle that never converges is the expected way to hit this, and the
        # user needs a terminal error line naming the knob they can turn — not
        # a run that sits at RUNNING forever.
        raise ProviderError(
            f"step limit reached ({max_steps}) without finishing the graph"
        ) from exc
    stats = dict(ctx.stats)
    stats["turns"] = ctx.turn
    stats.setdefault("tool_calls", 0)
    return stats
