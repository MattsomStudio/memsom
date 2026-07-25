"""Agent runs — a durable, multi-node graph of tool-using agents.

A *graph* is what the canvas draws: one or more agent nodes, each with its own
engine (provider + model), system prompt and tool instances, wired to each
other by control flow — a straight handoff, or a router that picks a branch,
or an edge back to an earlier agent to make a cycle. Running one is the panel
server's job, and every step lands in an append-only JSONL run file (the
session.py pattern with a wider event vocabulary), so the app can close
mid-run and re-poll the transcript later.

The graph *shape* is compiled here (:func:`compile_graph` → :class:`GraphSpec`);
the graph *execution* lives in ``lc_runtime.py`` on top of LangGraph, imported
lazily so memsom's core stays stdlib-only. What stays here is everything the
runtime is built out of — :func:`_execute_tool` (the two-phase audit) and
:func:`run_tool_loop`, the hand-rolled single-agent loop the voice path still
drives and which is therefore NOT dead code.

Run file — one JSON object per line at ``<runs_dir>/<run_id>.jsonl``:

    {"t":"start","run_id":..,"graph_id":..,"trigger":..,"provider":..,
     "model":..,"tools":[..],"limits":{..},"agents":[{..}],"ts":..}
    {"t":"warmup","action":"start"|"none","ok":true,"detail":"..","ts":..}
    {"t":"node","id":"n6","agent":"RESEARCHER","ts":..}   # graph runs only
    {"t":"turn","n":1,"node":"n6","ts":..}
    {"t":"tok","text":".."}                         # whole-turn text w/ tools
    {"t":"tool_call","turn":1,"id":"tc_1","name":..,"arguments":{..},"ts":..}
    {"t":"tool_result","turn":1,"id":"tc_1","name":..,"ok":true,
     "output":"..","bytes":123,"truncated":false,"elapsed_s":1.2}
    {"t":"route","router":"n8","branch":"escalate","mode":"decide","ts":..}
    {"t":"done","stats":{..}}                       # terminal, fsync'd, OR
    {"t":"error","error":"..","turn":2}

``node``/``route`` and the ``node`` field on ``turn`` are the multi-agent
additions, and they are strictly additive: a reader that switches on ``t``
ignores what it doesn't know, so a stale frontend degrades to the old view
rather than breaking.

Durability scope matches session.py: survives the app closing, not the panel
server restarting — ``reconcile_on_boot`` stamps orphaned run files with a
terminal error so the UI never shows an eternal RUNNING.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from memsom.providers.base import ProviderError, now
from memsom.providers.scope import check as scope_check, unscoped_tools
from memsom.providers.session import (
    AgentFileSink, new_session_id, valid_session_id, _final_stats, _first_json,
)
from memsom.providers.tools import (
    Tool, ToolContext, ToolError, build_tools, to_openai_tools,
)
from memsom.providers.tools.base import truncate_output

# hard ceilings — graph configs may tighten, never exceed
_MAX_TURNS_CEILING = 32
_DEFAULT_LIMITS = {
    "max_turns": 8,
    "tool_timeout_s": 60,
    "max_tool_output_bytes": 32768,
    "run_timeout_s": 900,
    # How many times ONE model call may be attempted. 1 = no retry, which is
    # the default on purpose: every adapter collapses a failure into a flat
    # ProviderError with no subtype, so nothing here can tell a transient
    # connection reset from a permanently wrong model name, and retrying the
    # latter just spends the run's budget slower. Opt in per agent when the
    # engine is known to be flaky (a cold llama.cpp, a rate-limited cloud API).
    "infer_retries": 1,
}
# Graph-level step budget: how many node transitions the whole graph may take
# before it is declared a runaway. Distinct from max_turns (how many times a
# MODEL may speak) because a cycle between two agents can burn steps without
# either agent exceeding its own turn ceiling — this is the one bound that
# makes a cycle safe to draw on the canvas.
_DEFAULT_MAX_STEPS = 24
_MAX_STEPS_CEILING = 64
# control-flow node types: what an agent or a router branch may hand off to.
_FLOW_TYPES = ("agent", "router", "output")
# identical consecutive tool calls before we call it a loop
_LOOP_STRIKES = 3
# hard ceiling on infer_retries, and the flat pause between attempts. Both are
# small on purpose: the retry exists to ride out a blip, not to sit on a dead
# engine, so the worst case a user can configure is ~4s of wasted wall clock.
_INFER_RETRIES_CEILING = 5
_INFER_RETRY_DELAY_S = 1.0
# what every adapter falls back to when params carries no timeout of its own;
# mirrored here so the deadline can never LENGTHEN a call, only shorten it.
_DEFAULT_INFER_TIMEOUT_S = 600
# bounded wait for a cold engine to come up
_WARMUP_TIMEOUT_S = 60.0
_WARMUP_POLL_S = 2.0
# How many TERMINAL runs keep their root checkpoints, so they stay forkable.
# A module constant and deliberately not a dashboard knob: it trades a few
# hundred KB against how far back "fork from here" reaches, and neither side of
# that is worth a control nobody will tune twice. Runs that are still live,
# paused or resumable are never counted and never evicted — retention is about
# forgetting history, never about reclaiming state something still needs.
_RETAIN_TERMINAL_RUNS = 20


@dataclass
class AgentSpec:
    """One validated agent NODE — engine, prompt, tools, limits.

    Unchanged field for field from when it *was* the whole compile output; what
    moved out from under it is "what does this GRAPH do", which is now
    :class:`GraphSpec`'s job. ``node_id`` is appended last and defaulted so the
    handful of callers that build one positionally by hand keep working.
    """

    graph_id: str
    graph_rev: int
    agent_name: str
    provider_id: str
    model: str
    transport: Optional[str]
    system: str
    params: dict
    tool_specs: list  # [{"name","type","options"}]
    limits: dict
    input: str
    node_id: str = ""
    #: JSON schema this agent's final answer must conform to, or None for free
    #: text (structured output — a separate structured LLM call after the loop).
    output_schema: Optional[dict] = None
    #: context management before each model call: "off" | "trim" | "summarize".
    context_mode: str = "off"
    #: message-count budget the context hook trims/summarizes down to (0 = the
    #: mode's built-in default).
    context_budget: int = 0
    #: guardrail applied to what this agent PRODUCES, after each model call:
    #: "off" | "scrub" (regex redaction, free) | "guard" (one extra inference
    #: that may block a tool call or withhold an answer).
    output_mode: str = "off"
    #: static pause points around this agent's node: "off" | "before" | "after"
    #: | "both". Distinct from a tool's require_approval, which is a DYNAMIC
    #: interrupt inside the node; these stop the graph at the node boundary.
    breakpoint_mode: str = "off"

    def as_start_meta(self) -> dict:
        # output_mode rides on the head line because the run log is the only
        # audit source and a guard that ALLOWS writes nothing. Without this,
        # "the guard judged every turn and approved" and "no guard was ever
        # configured" are the same bytes on disk, and the absence of a
        # `guardrail` event proves nothing. With it, absence has exactly one
        # meaning. Additive, so older readers ignore it.
        return {
            "graph_id": self.graph_id,
            "agent": self.agent_name,
            "provider": self.provider_id,
            "model": self.model,
            "tools": [t["name"] for t in self.tool_specs],
            "limits": self.limits,
            "output_mode": getattr(self, "output_mode", "off"),
        }


@dataclass
class RouterSpec:
    """A branch point — one compiled ``router`` node.

    A router is not an agent: it owns no engine and no prompt. Three modes, and
    what separates them is WHO decides and what it costs:

    * ``decide`` borrows the engine of the agent that fed it (``source_agent``)
      for one small extra inference whose only job is to name a branch;
    * ``match`` never infers at all and regexes that agent's final text;
    * ``handoff`` moves the decision INSIDE the feeding agent — a synthetic
      tool is bound into its own tool list, so the agent picks its successor as
      part of the turn it was taking anyway. No extra inference, at the price of
      the agent having to know it is standing at a fork.

    Every mode falls to ``else_branch`` when nothing decided, which is why
    compile time insists the else names one of this router's own branches — a
    router that can fail to route is a graph that can hang.
    """

    node_id: str
    mode: str                 # "decide" | "match" | "handoff"
    branches: list            # [{"name","when","target_node"}]
    else_branch: str          # always one of branches[*]["name"]
    source_agent: str         # node id of the agent whose output it reads


@dataclass
class GraphSpec:
    """A validated, runnable GRAPH — the output of ``compile_graph``.

    ``flow_edges`` is the control-flow adjacency and nothing else: for each
    agent and router node, the nodes it may hand off to. A router has one entry
    per branch; an agent has one — or, when it fans out, several sibling agents
    that run CONCURRENTLY. Resource wiring — engine→agent, tool→agent — is *not*
    in here; it was already folded into each :class:`AgentSpec`.

    ``joins`` names the barrier points that fan-out produces: a node every
    member of one fan-out set flows into, which must not run until all of them
    have. It is derived (not drawn) — the canvas has no join node, it just has
    edges — and it is kept separate from ``flow_edges`` because the two are
    wired with different LangGraph primitives and confusing them is the exact
    bug this stage exists to avoid.

    The entry-agent properties below are deliberate: ``list_runs`` and the
    RunMonitor history list read ``provider``/``model``/``tools`` off the run
    file's head line, ``handle_run_start`` audits them, and a one-agent graph
    should keep answering those questions exactly as it did before there were
    graphs at all.
    """

    graph_id: str
    graph_rev: int
    entry: str                # node id of the agent the trigger fires
    agents: dict              # node_id -> AgentSpec, entry first
    routers: dict             # node_id -> RouterSpec
    flow_edges: dict          # node_id -> [downstream node ids]
    limits: dict              # graph-level: entry agent's limits + max_steps
    input: str
    #: agent node ids the compiled graph must pause BEFORE / AFTER entering.
    #: Tuples rather than lists because a GraphSpec is handed to a worker thread
    #: and re-read on every resume; nothing should be able to edit the pause set
    #: out from under a run in flight. Empty by default, which is what every
    #: graph saved before breakpoints existed compiles to.
    breakpoints_before: tuple = ()
    breakpoints_after: tuple = ()
    #: join node id → the fan-out members that must ALL finish before it runs.
    #: Empty for every graph without a fan-out, which is every graph saved
    #: before this existed.
    joins: dict = field(default_factory=dict)
    #: what this run may touch — ``{"hosts": [...], "paths": [...]}``, read off
    #: the trigger. Empty means unrestricted, which is every graph saved before
    #: scope existed and is the deliberate default. See `memsom.providers.scope`.
    scope: dict = field(default_factory=dict)

    @property
    def entry_agent(self) -> AgentSpec:
        return self.agents[self.entry]

    @property
    def agent_name(self) -> str:
        return self.entry_agent.agent_name

    @property
    def provider_id(self) -> str:
        return self.entry_agent.provider_id

    @property
    def model(self) -> str:
        return self.entry_agent.model

    @property
    def transport(self) -> Optional[str]:
        return self.entry_agent.transport

    @property
    def system(self) -> str:
        return self.entry_agent.system

    @property
    def params(self) -> dict:
        return self.entry_agent.params

    @property
    def tool_specs(self) -> list:
        return self.entry_agent.tool_specs

    def engines(self) -> list:
        """Distinct provider ids across every agent node, entry first.

        Deduped because warming an engine twice is a wasted round trip, and
        ordered because the entry agent's engine is the one whose warmup the
        user is actually waiting on."""
        out = []
        for agent in self.agents.values():
            if agent.provider_id not in out:
                out.append(agent.provider_id)
        return out

    def as_start_meta(self) -> dict:
        entry = self.entry_agent
        return {
            "graph_id": self.graph_id,
            "agent": entry.agent_name,
            "provider": entry.provider_id,
            "model": entry.model,
            "tools": [t["name"] for t in entry.tool_specs],
            "limits": self.limits,
            # Per-agent output_mode, for the reason on AgentSpec.as_start_meta:
            # a guard that allows is silent, so "armed" has to be stated once
            # here or the absence of a guardrail event is unreadable. Per-agent
            # rather than graph-level because it IS per-agent.
            "output_mode": getattr(entry, "output_mode", "off"),
            # Same argument as output_mode, one layer out: a scope that is never
            # violated is SILENT, so "this run declared nothing" and "this run
            # declared a scope and stayed inside it" would otherwise be the same
            # bytes. `unscoped` names the tools no target rule can bound (shell)
            # so the hole is a fact on disk at run start rather than a discovery
            # made later by whoever reads the audit.
            "scope": dict(getattr(self, "scope", None) or {}),
            "unscoped": unscoped_tools(
                t["type"] for a in self.agents.values() for t in a.tool_specs),
            "agents": [
                {"node_id": a.node_id, "name": a.agent_name,
                 "provider": a.provider_id, "model": a.model,
                 "tools": [t["name"] for t in a.tool_specs],
                 "output_mode": getattr(a, "output_mode", "off")}
                for a in self.agents.values()
            ],
        }


def compile_graph(graph: dict, registry: dict, *,
                  input_override: Optional[str] = None) -> GraphSpec:
    """Validate a graph document into a GraphSpec. Every failure raises
    ProviderError with a verbatim, user-facing reason (the route maps these
    to 400s).

    The validations are ordered by what a user can act on: first the agents
    themselves (an agent with no engine is broken no matter how it's wired),
    then the entry point, then the routers, then reachability. A graph saved
    before routers existed has no ``next`` edges and no router nodes, so it
    falls straight through to a one-agent GraphSpec whose entry-agent
    properties answer exactly what the old AgentSpec answered.
    """
    nodes = {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict)}
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
    graph_id = str(graph.get("id") or "")
    graph_rev = int(graph.get("rev") or 0)

    agent_nodes = [n for n in nodes.values() if n.get("type") == "agent"]
    if not agent_nodes:
        raise ProviderError("graph must contain at least one agent node")

    # Naming an agent in an error message only helps when there is more than
    # one to confuse — and the unqualified sentence is the one the panel has
    # shown for every single-agent graph since this feature shipped.
    qualify = len(agent_nodes) > 1
    agents = {n["id"]: _compile_agent(n, nodes, edges, registry,
                                      graph_id=graph_id, graph_rev=graph_rev,
                                      qualify=qualify)
              for n in agent_nodes}

    triggers = [n for n in nodes.values() if n.get("type") == "trigger"]
    if len(triggers) != 1:
        raise ProviderError(
            f"graph must contain exactly one trigger node (found {len(triggers)})")
    trigger = triggers[0]
    fired = list(dict.fromkeys(
        e.get("target") for e in edges
        if e.get("source") == trigger["id"] and e.get("target") in agents))
    if len(fired) != 1:
        raise ProviderError(
            "the trigger must be wired into exactly one agent node "
            f"(found {len(fired)})")
    entry = fired[0]
    # entry first so as_start_meta()'s agents array leads with the agent whose
    # provider/model the head line already reports.
    agents = {entry: agents[entry],
              **{k: v for k, v in agents.items() if k != entry}}

    routers = {n["id"]: _compile_router(n, nodes, edges, agents)
               for n in nodes.values() if n.get("type") == "router"}

    flow_edges = _flow_edges(agents, routers, nodes, edges)
    _require_reachable(entry, agents, flow_edges)
    # Fan-out's two compile-time obligations, both after reachability for the
    # same reason breakpoints are: an unreachable agent is first and foremost an
    # unreachable agent, and that is the error the user can act on.
    joins = _fan_joins(agents, flow_edges)
    _require_no_gate_in_fan_out(agents, flow_edges)
    before, after = _validate_breakpoints(agents, routers, flow_edges)

    t_cfg = trigger.get("config") or {}
    limits = {**agents[entry].limits,
              "max_steps": _max_steps(t_cfg)}

    trigger_input = t_cfg.get("input") or ""
    return GraphSpec(
        graph_id=graph_id,
        graph_rev=graph_rev,
        entry=entry,
        agents=agents,
        routers=routers,
        flow_edges=flow_edges,
        limits=limits,
        input=input_override if input_override is not None else trigger_input,
        breakpoints_before=before,
        breakpoints_after=after,
        joins=joins,
        scope=_scope_of(t_cfg),
    )


def _scope_of(trigger_config: dict) -> dict:
    """Read the run's scope off the trigger node. ``{}`` means unrestricted.

    On the TRIGGER for the same reason ``max_steps`` is: it is a property of the
    run, not of any one agent. Per-agent scope would also be a hole rather than a
    feature — the first handoff or fan-out into a second agent would leave it
    behind, and a containment rule that stops at a node boundary contains
    nothing.

    Unknown keys are dropped rather than rejected. A scope is a safety
    declaration, and refusing to compile a graph because someone wrote
    ``"host"`` for ``"hosts"`` would turn a typo into a dead canvas — but
    silently honouring it would be worse, so only the two known dimensions are
    ever read, and what was read rides on the start meta where it can be seen.
    """
    raw = trigger_config.get("scope") or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key in ("hosts", "paths"):
        entries = raw.get(key)
        if isinstance(entries, (list, tuple)):
            cleaned = [str(e).strip() for e in entries if str(e).strip()]
            if cleaned:
                out[key] = cleaned
    return out


def _compile_agent(node: dict, nodes: dict, edges: list, registry: dict, *,
                   graph_id: str, graph_rev: int, qualify: bool) -> AgentSpec:
    """Resolve one agent node's engine, tools, params and limits."""
    a_cfg = node.get("config") or {}
    agent_name = a_cfg.get("name") or "AGENT"
    who = f"agent {agent_name!r} " if qualify else "agent "

    def _into(target_handle: str) -> list:
        return [nodes.get(e.get("source")) for e in edges
                if e.get("target") == node["id"]
                and e.get("targetHandle") == target_handle
                and nodes.get(e.get("source"))]

    engines = [n for n in _into("engine") if n.get("type") == "engine"]
    if len(engines) != 1:
        raise ProviderError(
            f"{who}needs exactly one engine wired in (found {len(engines)})")
    e_cfg = engines[0].get("config") or {}
    provider_id = e_cfg.get("provider") or ""
    adapter = registry.get(provider_id)
    if adapter is None:
        raise ProviderError(f"unknown provider: {provider_id!r}")
    model = e_cfg.get("model") or ""
    if not model:
        raise ProviderError("engine node has no model selected")

    transport = e_cfg.get("transport") or None
    tool_nodes = [n for n in _into("tools") if n.get("type") == "tool"]
    resolved_transport = transport or getattr(adapter, "transport", None)
    if tool_nodes and resolved_transport == "cli-subscription":
        raise ProviderError(
            "custom tools not supported over cli transport — use the api "
            "transport or remove the tool nodes")

    # unique tool instance names: auto-suffix duplicates (http_fetch_2, ...).
    # Uniqueness is PER AGENT, not per graph: each agent's model only ever sees
    # its own tool list, so two agents may both carry a plain "http_fetch".
    tool_specs, seen = [], {}
    for n in tool_nodes:
        t_cfg = n.get("config") or {}
        t_type = t_cfg.get("tool") or ""
        base_name = t_cfg.get("label") or t_type
        base_name = "".join(c if c.isalnum() or c == "_" else "_"
                            for c in base_name.lower()) or "tool"
        seen[base_name] = seen.get(base_name, 0) + 1
        name = base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]}"
        # A shell tool defaults to requiring approval; anything else defaults
        # off. Either way the tool node's own `require_approval` config wins.
        default_gate = t_type == "shell"
        require_approval = bool(t_cfg.get("require_approval", default_gate))
        tool_specs.append({"name": name, "type": t_type,
                           "options": t_cfg.get("options") or {},
                           "require_approval": require_approval})
    # surface unknown tool types now, not mid-run
    try:
        build_tools(tool_specs)
    except ToolError as exc:
        raise ProviderError(str(exc)) from exc

    limits = dict(_DEFAULT_LIMITS)
    for k, v in (a_cfg.get("limits") or {}).items():
        if k in limits and isinstance(v, (int, float)) and v > 0:
            limits[k] = int(v)
    limits["max_turns"] = min(limits["max_turns"], _MAX_TURNS_CEILING)
    limits["infer_retries"] = min(limits["infer_retries"], _INFER_RETRIES_CEILING)

    params = dict(a_cfg.get("params") or {})
    if transport:
        params["transport"] = transport

    output_schema = a_cfg.get("output_schema")
    if output_schema is not None and not isinstance(output_schema, dict):
        raise ProviderError(f"{who}output_schema must be a JSON-schema object")
    context_mode = a_cfg.get("context_mode") or "off"
    if context_mode not in ("off", "trim", "summarize"):
        raise ProviderError(
            f"{who}context_mode must be off, trim or summarize")
    context_budget = a_cfg.get("context_budget")
    context_budget = int(context_budget) if isinstance(context_budget, (int, float)) \
        and context_budget > 0 else 0
    output_mode = a_cfg.get("output_mode") or "off"
    if output_mode not in ("off", "scrub", "guard"):
        raise ProviderError(f"{who}output_mode must be off, scrub or guard")
    breakpoint_mode = a_cfg.get("breakpoint_mode") or "off"
    if breakpoint_mode not in ("off", "before", "after", "both"):
        raise ProviderError(
            f"{who}breakpoint_mode must be off, before, after or both")

    return AgentSpec(
        graph_id=graph_id,
        graph_rev=graph_rev,
        agent_name=agent_name,
        provider_id=provider_id,
        model=model,
        transport=transport,
        system=a_cfg.get("system") or "",
        params=params,
        tool_specs=tool_specs,
        limits=limits,
        input="",          # the graph owns the trigger input, not the node
        node_id=node["id"],
        output_schema=output_schema or None,
        context_mode=context_mode,
        context_budget=context_budget,
        output_mode=output_mode,
        breakpoint_mode=breakpoint_mode,
    )


def _compile_router(node: dict, nodes: dict, edges: list,
                    agents: dict) -> RouterSpec:
    """Resolve one router node's branches, else and feeding agent.

    A branch is matched to its outgoing edge by ``sourceHandle``: either the
    positional ``case_<i>`` the canvas emits, or the branch name itself, so
    renaming a branch on the canvas doesn't silently orphan its edge.
    """
    cfg = node.get("config") or {}
    label = cfg.get("label") or cfg.get("name") or node["id"]
    mode = str(cfg.get("mode") or "decide").lower()
    if mode not in ("decide", "match", "handoff"):
        raise ProviderError(
            f"router {label!r} has an unknown mode: {mode!r} "
            "(expected 'decide', 'match' or 'handoff')")

    raw = [b for b in (cfg.get("branches") or []) if isinstance(b, dict)]
    if not raw:
        raise ProviderError(f"router {label!r} needs at least one branch")

    out_edges = [e for e in edges if e.get("source") == node["id"]]
    branches = []
    for i, b in enumerate(raw):
        name = str(b.get("name") or f"case_{i}")
        handles = {f"case_{i}", name}
        targets = list(dict.fromkeys(
            e.get("target") for e in out_edges
            if str(e.get("sourceHandle") or "") in handles))
        if len(targets) != 1:
            raise ProviderError(
                f"router {label!r} branch {name!r} needs exactly one outgoing "
                f"edge (found {len(targets)})")
        target = nodes.get(targets[0])
        if target is None:
            raise ProviderError(
                f"router {label!r} branch {name!r} points at a missing node")
        if target.get("type") not in ("agent", "output"):
            raise ProviderError(
                f"router {label!r} branch {name!r} must target an agent or an "
                f"output node (targets a {target.get('type')!r} node)")
        branches.append({"name": name, "when": str(b.get("when") or ""),
                         "target_node": target["id"]})

    names = [b["name"] for b in branches]
    if len(set(names)) != len(names):
        raise ProviderError(f"router {label!r} has duplicate branch names")
    else_branch = str(cfg.get("else") or "")
    if else_branch not in names:
        raise ProviderError(
            f"router {label!r} needs an 'else' naming one of its branches "
            f"({', '.join(names)})")

    feeders = list(dict.fromkeys(
        e.get("source") for e in edges
        if e.get("target") == node["id"] and e.get("source") in agents))
    if len(feeders) != 1:
        raise ProviderError(
            f"router {label!r} needs exactly one agent wired into it "
            f"(found {len(feeders)})")

    return RouterSpec(node_id=node["id"], mode=mode, branches=branches,
                      else_branch=else_branch, source_agent=feeders[0])


def _flow_edges(agents: dict, routers: dict, nodes: dict, edges: list) -> dict:
    """Control-flow adjacency for every agent and router node.

    An agent hands off to one successor, or FANS OUT to several sibling agents
    that run concurrently. What it may not do is mix the two kinds: a router is
    a branch point — "go exactly one of these ways" — and a sibling is
    "go all of these ways at once", so a node wired to both is asking for two
    contradictory things and there is no reading of the canvas that is obviously
    right. Refused with the fix in the message rather than resolved by picking,
    which would make the drawing lie about what runs.

    (Fan-out used to be refused outright, on the argument that parallel agents
    sharing one message thread was a feature nobody had designed. It is designed
    now — see ``_fan_joins`` for the convergence half and ``lc_runtime`` for the
    concurrency safety it demands.)
    """
    flow: dict = {}
    for aid, agent in agents.items():
        outs = list(dict.fromkeys(
            e.get("target") for e in edges
            if e.get("source") == aid
            and (nodes.get(e.get("target")) or {}).get("type") in _FLOW_TYPES))
        live = [t for t in outs
                if nodes[t].get("type") in ("agent", "router")]
        if len(live) > 1 and any(nodes[t].get("type") != "agent" for t in live):
            raise ProviderError(
                f"agent {agent.agent_name!r} fans out to a mix of agents and a "
                "router — pick one: several agents to run them in parallel, or "
                "a single router to choose between them")
        # A live successor wins over a terminal output edge: an agent wired to
        # BOTH an output node and a next agent is mid-edit on the canvas, and
        # ending the run there would be the surprising reading.
        flow[aid] = live or outs
    for rid, router in routers.items():
        flow[rid] = list(dict.fromkeys(b["target_node"]
                                       for b in router.branches))
    return flow


def _require_reachable(entry: str, agents: dict, flow_edges: dict) -> None:
    """Every agent must be reachable from the trigger.

    An orphaned agent is always a wiring mistake, and it is the expensive kind:
    the canvas shows a configured agent that silently never runs, so the user
    debugs the prompt instead of the edge."""
    seen, stack = set(), [entry]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(flow_edges.get(node_id) or [])
    for node_id, agent in agents.items():
        if node_id not in seen:
            raise ProviderError(
                f"agent {agent.agent_name!r} is not reachable from the trigger")


def _fan_sets(agents: dict, flow_edges: dict) -> dict:
    """Every agent that fans out → the sibling agents it fans out TO.

    Shared by the join derivation, the approval refusal and (through
    ``GraphSpec``) the runtime's concurrency width, so all three agree about
    what "a fan-out" is. The ``len(live) == len(targets)`` test excludes the
    degenerate shape of an agent wired to two OUTPUT nodes: those aren't live
    successors, they both collapse to END, and calling that a fan-out would put
    END in a barrier."""
    fans: dict = {}
    for node_id, targets in (flow_edges or {}).items():
        if node_id not in agents:
            continue
        live = [t for t in targets if t in agents]
        if len(live) > 1 and len(live) == len(targets):
            fans[node_id] = live
    return fans


def _downstream(start: str, flow_edges: dict) -> set:
    """Every node reachable from *start* along the flow, *start* included.

    Deliberately includes non-agent targets (an output node, a router): a
    branch's shape is what decides whether two nodes can be in flight together,
    and that question does not care what type they are."""
    seen, stack = set(), [start]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(flow_edges.get(node_id) or [])
    return seen


def _fan_regions(agents: dict, flow_edges: dict) -> dict:
    """Every fan-out → ``{sibling: the nodes ONLY that branch can reach}``.

    The exclusive region is what makes "runs in parallel with" answerable one
    hop past the siblings themselves. A node reachable from two siblings is
    where the branches have already met — the join, or anything after it — and
    those run alone, so they are excluded from every region.

    Shared by the join derivation and the approval refusal so the two agree
    about where a parallel region ends; the one thing that made both of them
    wrong was measuring "parallel" as "is a DIRECT sibling"."""
    regions: dict = {}
    for source, siblings in _fan_sets(agents, flow_edges).items():
        reach = {s: _downstream(s, flow_edges) for s in siblings}
        regions[source] = {
            s: reach[s].difference(*(reach[o] for o in siblings if o != s))
            for s in siblings}
    return regions


def _fan_joins(agents: dict, flow_edges: dict) -> dict:
    """Derive the barrier joins a fan-out converges on. Refuses ambiguity.

    A join is a node several concurrent siblings flow into, and it must not run
    until they have ALL finished — otherwise it reads a half-written
    conversation, or runs once per arriving branch. LangGraph expresses that as
    a multi-start ``add_edge([starts], end)``; ``lc_runtime`` does the wiring,
    this decides where.

    The rule is attribution, not adjacency, and that distinction is the whole
    correctness of the feature. A predecessor of a candidate join belongs to a
    fan-out branch when exactly ONE sibling can reach it; the node is that
    fan-out's barrier when every one of its predecessors is attributable and
    every branch contributes exactly one. Branch DEPTH is irrelevant — the
    barrier over ``B->B2->B3`` and ``C->C2`` is ``[B3, C2]``, and deriving it
    only for the direct-sibling case (which is what "the inbound set IS the
    fan-out set" measured) left every deeper shape falling through to two plain
    edges: measured on a three-hop-vs-two-hop fan-out, the join ran TWICE, the
    first time on a conversation missing an entire branch, and the run still
    reported ``done``.

    Anything an attribution cannot resolve is refused by name rather than
    guessed at, because both plausible guesses are wrong in a way the user
    cannot see. Two predecessors from ONE branch (a router inside a branch, say)
    would make a barrier that waits for a start that may never run — measured,
    the barrier simply never fires and the graph ends without running the join.
    A predecessor from outside every branch is the mirror image.

    A node with several predecessors and NOTHING from a parallel branch among
    them is left alone. That shape predates this feature — two branches of a
    router converging, or a cycle's back-edge — and it is not a barrier: those
    predecessors are alternatives, only one of them ever runs.
    """
    fans = _fan_sets(agents, flow_edges)
    if not fans:
        return {}
    members = {m for siblings in fans.values() for m in siblings}
    regions = _fan_regions(agents, flow_edges)
    preds: dict = {}
    for node_id, targets in (flow_edges or {}).items():
        for target in targets:
            bucket = preds.setdefault(target, [])
            if node_id not in bucket:
                bucket.append(node_id)

    joins: dict = {}
    for node_id, sources in preds.items():
        if node_id not in agents or len(sources) < 2:
            continue
        match = _attribute_join(sources, fans, regions)
        if match is not None:
            # Ordered by the fan-out's own ordering, not by whatever order the
            # reverse walk happened to find them: the barrier's start list ends
            # up in a run's compiled graph, and a set would make it
            # non-deterministic.
            joins[node_id] = match
            continue
        # Refuse only what is genuinely CONCURRENT and unresolvable: a direct
        # sibling among the predecessors (the historical rule, kept verbatim),
        # or predecessors drawn from two different branches of one fan-out. Two
        # predecessors inside ONE branch are alternatives — a router forking and
        # rejoining within a branch — and that shape has always compiled to two
        # plain edges of which exactly one ever fires. Refusing it would be a
        # new false alarm dressed up as a safety net.
        if not (any(s in members for s in sources)
                or _straddles_branches(sources, fans, regions)):
            continue
        raise ProviderError(
            f"agent {agents[node_id].agent_name!r} is fed by "
            f"{len(sources)} agents that are not one parallel group, so "
            "there is no way to tell what it should wait for. Wire every "
            "member of the parallel group into it, or give the group its "
            "own dedicated join agent")
    return joins


def _straddles_branches(sources: list, fans: dict, regions: dict) -> bool:
    """Whether *sources* reach across two branches of the same fan-out.

    The test for "these predecessors can be in flight at once". Two of them from
    one branch cannot — that branch is sequential — so only a straddle is a
    barrier question, and only a straddle an attribution could not resolve is a
    refusal."""
    for source, siblings in fans.items():
        branches = regions.get(source) or {}
        owners = {s for s in siblings
                  if any(pred in branches.get(s, ()) for pred in sources)}
        if len(owners) >= 2:
            return True
    return False


def _attribute_join(sources: list, fans: dict, regions: dict):
    """*sources* as one fan-out's barrier starts, in sibling order, or None.

    Every predecessor must sit in exactly one sibling's exclusive region and no
    two may share a sibling — one start per branch is what a NamedBarrierValue
    can actually wait for."""
    for source, siblings in fans.items():
        branches = regions.get(source) or {}
        chosen: dict = {}
        for pred in sources:
            owner = next((s for s in siblings if pred in branches.get(s, ())),
                         None)
            if owner is None or owner in chosen:
                chosen = {}
                break
            chosen[owner] = pred
        if len(chosen) == len(siblings):
            return [chosen[s] for s in siblings]
    return None


def _require_no_gate_in_fan_out(agents: dict, flow_edges: dict) -> None:
    """Refuse an approval-gated tool on an agent that runs in parallel.

    Not a LangGraph limitation — it combines simultaneous interrupts from
    parallel branches into one ``GraphInterrupt`` perfectly well. It is a memsom
    one, and it is end to end: ``run_graph`` surfaces ``interrupts[0]`` and
    nothing else, ``AgentRunner.resume`` takes ONE flat decision, and
    ``handle_approve`` validates one verdict. Two gates opening at the same
    moment would therefore put a second pause on the graph that the UI never
    shows and no request can answer — a run stuck forever, with a transcript
    that looks like it is only waiting on one thing.

    Refused at compile time because that is the only place it can be said
    clearly. Per-``Interrupt.id`` resume is the real fix and a named follow-up;
    until then the honest answer is that this combination is not supported.

    The refusal covers the WHOLE parallel region, not the direct siblings.
    Checking only the siblings was measured to leave the failure fully
    reachable: ``A→{B,C}; B→B2; C→C2`` with gates on B2 and C2 compiled clean,
    both gates opened in one superstep, the transcript showed ONE
    ``awaiting_approval``, and the approve came back
    ``internal error: When there are multiple pending interrupts, you must
    specify the interrupt id when resuming`` — a terminal error with neither
    tool run and the human's approval recorded against a call that never
    happened. Worse than the hang the check was written to prevent.

    A node past the barrier is not in the region and keeps its gate: once the
    branches have joined, it runs alone again. The cost of the blanket rule is a
    single gate deep in one branch — safe in practice, since nothing else can
    interrupt with it — refused along with the unsafe ones. That is deliberate:
    "one gate anywhere in a parallel region" is a rule a user can hold, and it
    stops being a rule the moment editing an unrelated branch can invalidate it.
    """
    regions = _fan_regions(agents, flow_edges)
    for source, branches in regions.items():
        for owner, region in branches.items():
            others = {n for s, nodes in branches.items() if s != owner
                      for n in nodes if n in agents}
            # Walked in the graph's own agent order, not the region set's, so a
            # graph with two offending agents names the same one every time.
            for node_id, agent in agents.items():
                if node_id not in region:
                    continue
                if not any(spec.get("require_approval")
                           for spec in agent.tool_specs):
                    continue
                raise ProviderError(
                    f"agent {agent.agent_name!r} has a tool that requires "
                    f"approval and runs in parallel with "
                    f"{len(others)} other agent(s) after "
                    f"{agents[source].agent_name!r}. Two approval gates can open "
                    "at once and memsom can only surface one, so the run would "
                    "hang. Move the gated tool onto an agent that runs on its "
                    "own, or drop the approval requirement")


def _validate_breakpoints(agents: dict, routers: dict,
                          flow_edges: dict) -> tuple:
    """Split the agents' ``breakpoint_mode`` into the two compile() arguments.

    The one refusal here is not defensive polish, it is a measured trap:
    ``interrupt_after`` on a node whose only successor is END is a SILENT
    NO-OP. LangGraph pauses *before the next task*, and a node that ends the
    graph has no next task — so the run finishes, ``get_state`` reports nothing
    pending, and the user is left staring at a completed run wondering why
    their breakpoint never fired. Refusing at compile time turns a mystery into
    a sentence.

    A node feeding a ROUTER is deliberately allowed even though the pause lands
    before a branch nobody has picked yet: that is exactly when a human most
    wants to look, and the router's own inference has not run, so nothing is
    lost by stopping there.
    """
    before, after = [], []
    for node_id, agent in agents.items():
        mode = getattr(agent, "breakpoint_mode", "off")
        if mode in ("before", "both"):
            before.append(node_id)
        if mode in ("after", "both"):
            successors = flow_edges.get(node_id) or []
            # ANY live successor is enough — a fan-out source has several, and
            # the pause lands before the whole set runs. Identical to the old
            # ``successors[0]`` reading for every single-successor node.
            if not any(t in agents or t in routers for t in successors):
                raise ProviderError(
                    f"agent {agent.agent_name!r} has breakpoint_mode "
                    f"{mode!r} but no live successor — an 'after' breakpoint "
                    "on a node that ends the graph never pauses. Use 'before', "
                    "or wire the agent into another agent or a router.")
            after.append(node_id)
    return tuple(before), tuple(after)


def _max_steps(trigger_config: dict) -> int:
    """Read the graph's step budget off the trigger node, clamped.

    It lives on the trigger because that is the node that owns "how this graph
    runs" (the schedule is already there), not on any one agent — a cycle's
    budget belongs to nobody in particular. Both ``max_steps`` and
    ``limits.max_steps`` are accepted; the canvas writes one, hand-edited JSON
    tends to write the other."""
    raw = (trigger_config.get("limits") or {}).get(
        "max_steps", trigger_config.get("max_steps"))
    steps = _DEFAULT_MAX_STEPS
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        steps = int(raw)
    return min(steps, _MAX_STEPS_CEILING)


def run_tool_loop(adapter, model: str, messages: list, params: dict,
                  sink, *, tools: list, audit_path, limits: dict,
                  final_answer_on_exhaust: bool = False) -> dict:
    """The durable infer→execute-tools→infer loop, provider-agnostic.

    Extracted from :class:`AgentRunner` so both the agent layer AND the voice
    tool-loop drive ONE implementation (no reinvention). Semantics unchanged:
    render *tools* into ``params['tools']`` (OpenAI wire), then each turn call
    ``adapter.infer(model, messages, params, sink)``; a turn with no
    ``stats['tool_calls']`` is the final answer and returns; otherwise execute
    every call under the two-phase audit, feed results back, and loop. Emits
    ``turn``/``tool_call``/``tool_result`` events onto *sink* (an
    :class:`AgentFileSink`). *messages* is mutated in place — callers that
    reuse a base message list must pass a copy.

    Streaming: whether a turn streams tokens or returns them buffered is the
    ADAPTER's call (claude.py streams text deltas AND tool_use when
    ``params['stream']`` is set; without it, whole-turn text lands via
    ``sink.token`` at the end). Either way every token reaches *sink*.
    """
    by_name = {t.name: t for t in tools}
    ctx = ToolContext(
        audit_path=audit_path,
        timeout_s=limits["tool_timeout_s"],
        max_output_bytes=limits["max_tool_output_bytes"],
    )
    params = dict(params)
    if tools:
        params["tools"] = to_openai_tools(tools)

    started = now()
    last_sig, strikes = None, 0
    agg: dict = {}
    tool_call_count = 0

    for turn in range(1, limits["max_turns"] + 1):
        if now() - started > limits["run_timeout_s"]:
            raise ProviderError(f"run timeout after {limits['run_timeout_s']}s")
        sink.event({"t": "turn", "n": turn, "ts": now()})
        # The entry check above bounds "how many fast turns"; this bounds "how
        # long ONE of them may take". Both are needed — either alone lets a run
        # blow its budget, and this path (voice) has exactly the same bug the
        # graph path had.
        stats = _infer_with_deadline(
            adapter, model, messages, params, sink,
            run_timeout_s=limits["run_timeout_s"], started=started,
            max_attempts=limits.get("infer_retries", 1))
        for k in ("prompt_tokens", "eval_count"):
            if isinstance(stats.get(k), (int, float)):
                agg[k] = agg.get(k, 0) + stats[k]
        calls = stats.get("tool_calls") or []
        if not calls:
            agg["turns"] = turn
            agg["tool_calls"] = tool_call_count
            return agg

        # loop detection: the same call(s), repeatedly
        sig = json.dumps([(c.get("name"), c.get("arguments"))
                          for c in calls], sort_keys=True, default=str)
        strikes = strikes + 1 if sig == last_sig else 0
        last_sig = sig
        if strikes >= _LOOP_STRIKES - 1:
            raise ProviderError(
                f"tool loop detected: {_LOOP_STRIKES}x identical call(s)")

        assistant_text = ""  # text already went to the sink; echo minimal
        messages.append({"role": "assistant", "content": assistant_text,
                         "tool_calls": calls})
        for call in calls:
            tool_call_count += 1
            cid = call.get("id") or f"tc_{tool_call_count}"
            name = call.get("name") or ""
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {"_raw": str(arguments)}
            sink.event({"t": "tool_call", "turn": turn, "id": cid,
                        "name": name, "arguments": arguments, "ts": now()})
            t0 = now()
            output, ok = _execute_tool(by_name.get(name), name, arguments, ctx,
                                       audit_path, available=sorted(by_name))
            text, truncated = truncate_output(
                output, limits["max_tool_output_bytes"])
            sink.event({"t": "tool_result", "turn": turn, "id": cid,
                        "name": name, "ok": ok, "output": text,
                        "bytes": len(output.encode("utf-8", "ignore")),
                        "truncated": truncated,
                        "elapsed_s": round(now() - t0, 3)})
            messages.append({"role": "tool", "tool_call_id": cid,
                             "name": name, "content": text})
    if final_answer_on_exhaust:
        # Voice must always speak SOMETHING. Instead of erroring with no answer
        # when the model keeps reaching for tools, strip the tools and force one
        # final text turn — it summarizes what it has (or says it came up empty).
        sink.event({"t": "turn", "n": limits["max_turns"] + 1,
                    "final": True, "ts": now()})
        final_params = dict(params)
        final_params.pop("tools", None)
        # Deliberately NOT deadline-bounded. This turn only happens because the
        # model burned every turn reaching for tools, so the budget is usually
        # spent already — injecting the remainder would turn "voice always says
        # something" into "voice raises a timeout", which is the failure this
        # branch exists to prevent.
        stats = adapter.infer(model, messages, final_params, sink) or {}
        for k in ("prompt_tokens", "eval_count"):
            if isinstance(stats.get(k), (int, float)):
                agg[k] = agg.get(k, 0) + stats[k]
        agg["turns"] = limits["max_turns"] + 1
        agg["tool_calls"] = tool_call_count
        agg["exhausted"] = True
        return agg
    raise ProviderError(
        f"max turns reached ({limits['max_turns']}) without a final answer")


class _EmitGuardSink:
    """Wraps a sink and remembers whether anything reached the run log.

    This is the precondition the retry below rests on, not a convenience. The
    JSONL run file is the display AND audit source, and the events it carries
    are append-only: half a streamed sentence followed by a second attempt at
    the same turn would leave two half-answers glued together under one ``turn``
    line, with no way for a reader to tell which text the model actually stood
    behind. So one non-empty ``token``/``reasoning`` chunk makes an attempt
    final — a failure that already spoke is never retried.

    Everything else an adapter reaches for (``event``, a tee sink's ``text``)
    delegates straight through, so wrapping is invisible to both sides.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.emitted = False

    def token(self, text: str) -> None:
        if text:
            self.emitted = True
        self._inner.token(text)

    def reasoning(self, text: str) -> None:
        if text:
            self.emitted = True
        self._inner.reasoning(text)

    def __getattr__(self, name: str):
        try:
            inner = self.__dict__["_inner"]
        except KeyError:                    # pragma: no cover - defensive
            raise AttributeError(name) from None
        return getattr(inner, name)


def _infer_with_deadline(adapter, model: str, messages: list, params: dict,
                         sink, *, run_timeout_s, started: float,
                         max_attempts: int = 1) -> dict:
    """One model call, bounded by what is LEFT of the run's budget.

    ``run_timeout_s`` was only ever a turn-ENTRY gate: it was checked before
    each turn and never again, so a single ``infer`` that hung for an hour
    walked straight through a 120s budget and the run sat at RUNNING until
    somebody noticed. The fix costs nothing new on the wire, because every
    adapter already reads ``params['timeout']`` as its HTTP/subprocess timeout
    and merely defaults it to ten minutes — nothing had ever wired that knob to
    the run budget. So the deadline is injected here, per attempt, as
    ``min(whatever the call already asked for, what's left)``: the budget can
    only ever tighten a call, never lengthen one, and an agent that explicitly
    wants a short per-call timeout keeps it.

    A ProviderError raised once the budget is spent is re-labelled with the
    turn-entry gate's exact wording. The raw failure underneath is a socket
    timeout from whichever adapter happened to be in the call — true, but it
    reads as an engine fault when the real answer is "you gave this run 120
    seconds". The original is kept as ``__cause__``.

    Retries are opt-in (``max_attempts`` comes from ``limits['infer_retries']``,
    default 1 = today's behaviour byte for byte) and guarded by
    :class:`_EmitGuardSink`. They are audit-safe by construction: a turn's tool
    calls are only audited AFTER ``infer`` returns, so a failed attempt has
    written nothing to the audit log, and the emit guard means it has written
    nothing to the run log either.
    """
    attempts = max(1, int(max_attempts or 1))
    # what the caller already asked for wins if it is TIGHTER than the budget
    base = params.get("timeout")
    if isinstance(base, bool) or not isinstance(base, (int, float)) or base <= 0:
        base = _DEFAULT_INFER_TIMEOUT_S
    budget = float("inf")
    if isinstance(run_timeout_s, (int, float)) and not isinstance(
            run_timeout_s, bool) and run_timeout_s > 0:
        budget = float(run_timeout_s)

    for attempt in range(1, attempts + 1):
        call_params = dict(params)
        # recomputed per attempt: a retry spends real budget too, so attempt
        # two gets the deadline as it stands then, not as it stood at entry.
        remaining = budget - (now() - started)
        # never zero or negative — urllib reads that as "no timeout" on some
        # paths, which would be the exact bug this function exists to kill.
        call_params["timeout"] = min(base, max(remaining, 0.01))
        guarded = _EmitGuardSink(sink)
        try:
            return adapter.infer(model, messages, call_params, guarded) or {}
        except ProviderError as exc:
            if now() - started >= budget:
                raise ProviderError(
                    f"run timeout after {run_timeout_s}s") from exc
            if guarded.emitted or attempt >= attempts:
                raise
            time.sleep(_INFER_RETRY_DELAY_S)
    raise ProviderError("infer exhausted its attempts")  # pragma: no cover


def _execute_tool(tool: Optional[Tool], name: str, arguments: dict,
                  ctx: ToolContext, audit_path, *, available: list) -> tuple:
    """Run one tool call under the two-phase audit. Model-level mistakes
    (unknown tool, bad args, tool failure) come back as a failing result
    string — the model gets to react; only audit unavailability kills the
    run."""
    from memsom.providers.handlers import _audit
    intent = {"action": "tool", "tool": name,
              "arguments": {k: str(v)[:200] for k, v in arguments.items()}}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        raise ProviderError(f"audit unavailable; refused: {exc}") from exc
    if tool is None:
        _audit(audit_path, {**intent, "result": "refused-unknown-tool"})
        return (f"unknown tool {name!r}; available: "
                f"{', '.join(available) or 'none'}", False)
    # Scope is checked HERE, and the position is doing work. This function is the
    # only place `Tool.run` is called from in the repo, so there is one gate
    # rather than one per tool — and it sits AFTER the approval interrupt, so
    # arguments a human SUBSTITUTED at the gate (`lc_model`'s "edit" verdict) get
    # checked too. A check any earlier would have trusted the edit.
    #
    # Refused like an unknown tool rather than raised: an out-of-scope call is a
    # mistake the model can correct next turn, and killing the run over it would
    # throw away whatever the graph had already banked.
    refusal = scope_check(getattr(ctx, "scope", None), getattr(tool, "type", ""),
                          arguments)
    if refusal:
        _audit(audit_path, {**intent, "result": "refused-out-of-scope"})
        return f"REFUSED, out of scope: {refusal}", False
    try:
        out = tool.run(arguments, ctx)
    except ToolError as exc:
        _audit(audit_path, {**intent, "result": f"failed: {exc}"})
        return f"tool error: {exc}", False
    except Exception as exc:
        _audit(audit_path, {**intent, "result": f"error: {exc}"})
        return f"tool internal error: {exc}", False
    _audit(audit_path, {**intent, "result": "ok"})
    return out, True


class AgentRunner:
    """Owns the agent runs directory; one tool-loop thread per run.

    ``max_concurrent`` is a global slot (default 1): local engines share one
    GPU, and a second concurrent loop mostly means VRAM thrash. A start that
    finds the slot taken raises — the route maps it to 409, the scheduler
    records a skip."""

    def __init__(self, runs_dir, registry: dict, audit_path,
                 max_concurrent: int = 1) -> None:
        self.dir = Path(runs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.audit_path = Path(audit_path)
        # Execution-state store for pause/resume, a sibling of runs/. Holds only
        # in-flight runs (pruned on any terminal exit); never a display source.
        self.checkpoints = self.dir.parent / "checkpoints.db"
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._active: set[str] = set()
        # run_id -> spec for runs paused at a human-approval interrupt. Holds
        # what resume() needs to rebuild the graph; the pausable state itself
        # lives in the checkpoint DB, not here. Lost on restart — a paused run
        # is then recovered as "resumable" via its surviving checkpoint.
        self._paused: dict[str, "GraphSpec"] = {}
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.jsonl"

    def _lines(self, run_id: str) -> list:
        """A run file's raw lines, or []. Same newline-only split (and same
        reason) as ``read_since``; a missing or unreadable file reads as an
        empty run rather than raising into a status check."""
        try:
            return self._path(run_id).read_text(
                encoding="utf-8", errors="replace").split("\n")
        except OSError:
            return []

    # -- lifecycle ---------------------------------------------------------

    def start(self, spec: "GraphSpec", trigger: str) -> str:
        # The entry agent's provider is checked here, synchronously, so a
        # registry that lost an adapter between compile and run is a 400 the
        # caller sees rather than a run file that errors a millisecond later.
        adapter = self.registry.get(spec.provider_id)
        if adapter is None:
            raise ProviderError(f"unknown provider: {spec.provider_id!r}")
        if not self._slots.acquire(blocking=False):
            raise ProviderError("an agent run is already active; try again later")
        run_id = new_session_id()
        path = self._path(run_id)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "t": "start", "run_id": run_id, "trigger": trigger,
                    **spec.as_start_meta(), "ts": now(),
                }, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            self._slots.release()
            raise
        with self._lock:
            self._active.add(run_id)
        threading.Thread(target=self._run, args=(spec, run_id, path),
                         name=f"agent-{run_id}", daemon=True).start()
        return run_id

    def fork(self, spec: "GraphSpec", source_run_id: str, step: int,
             edit: str = None) -> str:
        """Start a NEW run continuing *source_run_id* from after its *step*.

        Deliberately a sibling of ``start`` rather than a flag on it: a fork has
        its own run id, its own file and its own slot, and everything downstream
        — history, polling, the audit — treats it as the ordinary new run it is.
        The only thing that marks it is an additive ``forked_from`` on the start
        line, which older frontends ignore and this one renders as a badge.

        ``trigger`` is "fork" for the same reason "manual" and "schedule" exist:
        the head line is where a reader finds out why a run happened at all.

        *spec* is compiled by the caller from the CURRENT graph doc — same
        precedent as a post-restart resume — so a fork taken after fixing a
        prompt runs the fix rather than the mistake it was forked to escape.
        """
        adapter = self.registry.get(spec.provider_id)
        if adapter is None:
            raise ProviderError(f"unknown provider: {spec.provider_id!r}")
        if not self._slots.acquire(blocking=False):
            raise ProviderError("an agent run is already active; try again later")
        run_id = new_session_id()
        path = self._path(run_id)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "t": "start", "run_id": run_id, "trigger": "fork",
                    **spec.as_start_meta(),
                    "forked_from": {"run_id": source_run_id, "step": step},
                    "ts": now(),
                }, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            self._slots.release()
            raise
        with self._lock:
            self._active.add(run_id)
        threading.Thread(
            target=self._run, args=(spec, run_id, path),
            kwargs={"fork_from": {"source_run_id": source_run_id,
                                  "step": step, "edit": edit}},
            name=f"agent-fork-{run_id}", daemon=True).start()
        return run_id

    def _run(self, spec: "GraphSpec", run_id: str, path: Path,
             resume_decision=None, fork_from=None) -> None:
        sink = AgentFileSink(path)
        # Retention runs HERE — at the head of a run, inside the slot it already
        # holds — and not in the finally, which is where it obviously belongs
        # and where it is wrong. The finally sits between the terminal `done`
        # line and the slot release, so anything expensive there is a window in
        # which a run reports finished while the next `start` still 409s
        # "already active". That window used to be microseconds; a sweep made it
        # milliseconds and started failing back-to-back runs for real (measured,
        # not theorised — three tests in this stage hit it). Sweeping on the way
        # IN costs a slot that is held for the whole run anyway, keeps the
        # strict "never overlaps another run's connection" guarantee, and only
        # means the cap is enforced lazily: a finished run's checkpoints are
        # evicted at the START of the next one, so the DB plateaus at
        # `_RETAIN_TERMINAL_RUNS + 1` rather than exactly at the constant.
        #
        # Skipped for a fork, which is about to READ a terminal run's
        # checkpoints: a sweep that happened to evict the source would turn a
        # fork the user just picked into "it aged out" a millisecond later.
        if fork_from is None:
            self._enforce_retention()
        try:
            if resume_decision is not None:
                # A resume appends to the same run file. Record the human's call
                # before continuing so the transcript shows what unblocked it.
                # Three shapes, three lines: stepping past a BREAKPOINT is not
                # an approve/deny and must not read as one in the audit trail
                # (nobody vouched for anything), and an EDIT is recorded with
                # the arguments the human substituted rather than as a raw dict
                # the frontend would have to guess at.
                if resume_decision == "continue":
                    sink.event({"t": "resume", "kind": "breakpoint",
                                "ts": now()})
                elif isinstance(resume_decision, dict):
                    sink.event({"t": "approval", "decision": "edit",
                                "arguments": resume_decision.get("arguments")
                                or {}, "ts": now()})
                else:
                    sink.event({"t": "approval", "decision": resume_decision,
                                "ts": now()})
            self._warmup(spec, sink)
            stats, paused = self._loop(spec, run_id, sink, resume_decision,
                                       fork_from=fork_from)
            if paused:
                # The run is waiting on a human. The awaiting_approval event is
                # already on disk; write no terminal line, keep the checkpoint,
                # and remember the spec so resume() can rebuild the graph.
                with self._lock:
                    self._paused[run_id] = spec
            else:
                sink.done(_final_stats(sink, stats))
        except ProviderError as exc:
            sink.error(str(exc))
        except Exception as exc:  # defensive: never die silently
            sink.error(f"internal error: {exc}")
        finally:
            # The slot is freed whether the run finished or is now waiting on a
            # human — a paused run consumes no compute.
            with self._lock:
                self._active.discard(run_id)
            self._slots.release()

    def resume(self, run_id: str, decision,
               spec: "GraphSpec" = None) -> str:
        """Continue a paused run — an approval gate OR a static breakpoint.

        ``decision`` is one of:

        * ``"approve"`` / ``"deny"`` — handed straight to the waiting
          ``interrupt()`` inside the tool;
        * ``{"decision": "edit", "arguments": {…}}`` — approve, but run the
          tool with the arguments the human substituted;
        * ``"continue"`` — step past a static breakpoint. There is no
          ``interrupt()`` waiting for a value in that case, so ``_loop``
          translates it into the bare-resume sentinel rather than a
          ``Command(resume=…)`` nothing would consume.

        The spec comes from memory for a live pause; for one that outlived a
        restart the caller recompiles it from the graph doc and passes it in.
        Either way the run's pausable state is in the checkpoint, so a missing
        checkpoint is a hard refusal, not a silent fresh start. Re-acquires the
        single run slot and continues on a fresh worker thread under the same
        run id.

        The STATUS gate is not belt-and-braces, it is the invariant that used to
        be enforced by accident. A finished run's checkpoints used to be deleted
        outright, so ``_has_checkpoint`` doubled as "is in flight" and a resume
        of a terminal run refused itself. Retention keeps that chain now, and
        with it "has a checkpoint" stopped meaning "is waiting for you":
        measured, an approve against an ordinary finished run returned 200, then
        appended an ``approval`` line and a SECOND zeroed ``done`` after the
        terminal one — into the file that is the only display and audit source —
        re-ran the last node's tools, and wrote a human approval into audit.jsonl
        for a gate nobody was ever asked about. A double-clicked APPROVE button
        on a run that errored a moment earlier is enough to reach it."""
        with self._lock:
            spec = spec or self._paused.get(run_id)
        if spec is None:
            raise ProviderError(f"run {run_id!r} cannot be resumed")
        if not self._has_checkpoint(run_id):
            raise ProviderError(
                f"run {run_id!r} has no checkpoint to resume from")
        # Second, and no longer implied by the first: presence used to mean
        # liveness, retention broke that, and this is where it is said again.
        lines = self._lines(run_id)
        status, _ = self._status_of(run_id, lines)
        if status not in ("paused", "resumable"):
            raise ProviderError(
                f"run {run_id!r} is {status}, not waiting for a decision")
        kind = self._pause_kind(lines)
        wants_value = decision != "continue"
        if kind == "breakpoint" and wants_value:
            raise ProviderError(
                f"run {run_id!r} is stopped at a breakpoint, not at an approval "
                "gate — send 'continue' to step past it")
        if kind == "approval" and not wants_value:
            raise ProviderError(
                f"run {run_id!r} is waiting on an approval gate — send "
                "'approve', 'deny' or an edit, not 'continue'")
        if not self._slots.acquire(blocking=False):
            raise ProviderError("an agent run is already active; try again later")
        with self._lock:
            self._paused.pop(run_id, None)
            self._active.add(run_id)
        path = self._path(run_id)
        threading.Thread(
            target=self._run, args=(spec, run_id, path),
            kwargs={"resume_decision": decision},
            name=f"agent-resume-{run_id}", daemon=True).start()
        return run_id

    def paused_spec(self, run_id: str) -> "Optional[GraphSpec]":
        """The in-memory spec of a live-paused run, or None (e.g. after a
        restart, when the caller must recompile from the graph doc)."""
        with self._lock:
            return self._paused.get(run_id)

    def head_graph_id(self, run_id: str) -> Optional[str]:
        """The graph id recorded on a run's start line — what a post-restart
        resume recompiles its spec from."""
        p = self._path(run_id)
        if not p.is_file():
            return None
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        except OSError:
            return None
        head = _first_json(lines) or {}
        return head.get("graph_id")

    # -- the loop ----------------------------------------------------------

    def _warmup(self, spec: "GraphSpec", sink: AgentFileSink) -> None:
        """Warm every DISTINCT engine the graph will speak to, entry first.

        A two-agent graph split across ollama and llama.cpp would otherwise
        stall halfway through on a cold second engine — after the first agent
        has already burned real tokens. Deduped by provider id, and still
        never unloads anyone else's model: VRAM admission control is
        deliberately not this layer's job."""
        for provider_id in spec.engines():
            adapter = self.registry.get(provider_id)
            if adapter is None:
                raise ProviderError(f"unknown provider: {provider_id!r}")
            self._warmup_one(adapter, provider_id, sink)

    def _warmup_one(self, adapter, provider_id: str,
                    sink: AgentFileSink) -> None:
        """Cold engine gets started; a warm one passes through."""
        try:
            state = adapter.status().state
        except Exception:
            state = "down"
        caps = adapter.capabilities()
        if state == "up" or not getattr(caps, "can_start", False):
            sink.event({"t": "warmup", "action": "none", "ok": True,
                        "detail": state, "provider": provider_id, "ts": now()})
            return
        sink.event({"t": "warmup", "action": "start", "ok": True,
                    "detail": "starting engine", "provider": provider_id,
                    "ts": now()})
        adapter.start()
        deadline = now() + _WARMUP_TIMEOUT_S
        while now() < deadline:
            try:
                if adapter.status().state == "up":
                    sink.event({"t": "warmup", "action": "start", "ok": True,
                                "detail": "engine up", "provider": provider_id,
                                "ts": now()})
                    return
            except Exception:
                pass
            time.sleep(_WARMUP_POLL_S)
        raise ProviderError("engine did not come up within warmup timeout")

    def _loop(self, spec: "GraphSpec", run_id: str, sink: AgentFileSink,
              resume_decision=None, fork_from=None) -> tuple:
        """Hand the compiled graph to the LangGraph runtime.

        Imported here and not at module scope on purpose: ``lc_runtime`` pulls
        in langgraph and langchain-core, and memsom's core is stdlib-only. A
        machine that never runs an agent never pays for the import, and one
        that tries gets a ProviderError naming the extra instead of a
        traceback at server boot.

        ``run_id`` doubles as the checkpoint thread id, so a run's pausable
        state is keyed to the same id its run-log file carries. Returns
        ``(stats, paused)``; ``resume_decision`` (None on a fresh run) is passed
        through so a resume continues the same checkpointed thread, and
        ``fork_from`` (None unless this run is a fork) names the run and step
        whose state seeds it."""
        from memsom.providers.lc_runtime import _CONTINUE, _UNSET, run_graph
        # Three inputs, three sentinels. "continue" (step past a breakpoint) is
        # NOT a value for a waiting interrupt() — there is none — so it maps to
        # the bare-resume sentinel; a Command(resume="continue") would sit
        # unconsumed and the graph would replay instead of advancing.
        if resume_decision is None:
            resume = _UNSET
        elif resume_decision == "continue":
            resume = _CONTINUE
        else:
            resume = resume_decision
        return run_graph(spec, self.registry, sink, self.audit_path,
                         run_id=run_id, checkpoint_path=self.checkpoints,
                         resume_decision=resume, fork_from=fork_from)

    # -- reads -------------------------------------------------------------

    def read_since(self, run_id: str, cursor: int = 0) -> dict:
        if not valid_session_id(run_id):
            raise ProviderError("invalid run_id")
        path = self._path(run_id)
        if not path.is_file():
            return {"events": [], "cursor": cursor, "status": "unknown"}
        # newline-only split; see SessionStore.read_since for why splitlines()
        # is wrong here (it breaks on U+2028/U+2029/U+0085 and drops records).
        # Drop the trailing "" from the final "\n" so cursor=len(lines) doesn't
        # overshoot and bury the next record below the cursor on later polls.
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        events = []
        for line in lines[cursor:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        status, stats = self._status_of(run_id, lines)
        return {"events": events, "cursor": len(lines), "status": status,
                "stats": stats}

    def list_runs(self, limit: int = 50) -> list:
        files = sorted(self.dir.glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        # One query for the whole page rather than `_has_checkpoint` per row.
        # This endpoint is POLLED, and 50 sqlite connections per poll to answer
        # a boolean the DB can answer once is the kind of thing that only shows
        # up as "the panel got sluggish" months later.
        live = self._checkpointed_threads()
        out = []
        for p in files:
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                continue
            head = _first_json(lines) or {}
            status, _ = self._status_of(p.stem, lines)
            out.append({
                "run_id": p.stem,
                "graph_id": head.get("graph_id"),
                "agent": head.get("agent"),
                "provider": head.get("provider"),
                "model": head.get("model"),
                "trigger": head.get("trigger"),
                "ts": head.get("ts"),
                "status": status,
                # Where this run came from, if anywhere — None for every run
                # that was not forked, which is every run written before this
                # existed. Read off the head line, so history is self-describing
                # without a second store to keep in step.
                "forked_from": head.get("forked_from"),
                # Whether "fork from here" can be offered at all. A live or
                # paused run has no settled steps to fork FROM, and a terminal
                # one whose checkpoints aged out has nothing left to seed with —
                # the same presence question resume asks, batched.
                "forkable": status in ("done", "error") and p.stem in live,
            })
        return out

    def _checkpointed_threads(self) -> set:
        """Every thread id with a surviving checkpoint. Same defensiveness as
        ``_has_checkpoint`` — a missing, locked or surprising DB reads as
        'nothing is checkpointed' rather than raising into a list read."""
        if not self.checkpoints.is_file():
            return set()
        try:
            con = sqlite3.connect(str(self.checkpoints))
            try:
                return {row[0] for row in con.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints").fetchall()}
            finally:
                con.close()
        except Exception:
            return set()

    def _pause_kind(self, lines) -> str:
        """Why this run stopped: ``approval``, ``breakpoint`` or ``""``.

        One backwards walk, and it stops at the first DECISION it meets — an
        ``approval`` or a ``resume`` line means everything before it has already
        been answered. Exactly the rule the monitor's ``latestPause`` uses to
        pick which card to show, which is the point: the server was accepting a
        decision the UI would never have offered, and writing it into the
        transcript as though a human had made it. A breakpoint has no
        ``interrupt()`` to consume a value, so an 'approve' against one simply
        stepped the graph and left ``{"t":"approval","decision":"approve"}`` in
        the audit trail for something nobody was asked to approve.

        Read from the run log rather than from ``_status_of``, deliberately:
        ``_status_of``'s ordering is load-bearing and a reason-branch inside it
        is what caused the v0.18.0 resume bug.
        """
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("t")
            if kind in ("approval", "resume"):
                return ""
            if kind == "awaiting_approval":
                return "approval"
            if kind == "paused_breakpoint":
                return "breakpoint"
        return ""

    def _sweep_sidecars(self) -> None:
        """Delete shared-scratchpad sidecars no run can still use.

        ``run_graph`` unlinks its own on a terminal exit, which covers every run
        that reaches one. What it cannot cover is a run whose PROCESS died:
        ``reconcile_on_boot`` stamps that run errored but prunes nothing, so its
        ``agents/shared/<run_id>.json`` outlives it with nobody left to read it.
        A file class that only ever grows is a leak however small the files are,
        and this is the sweep that was assigned to retention and not written.

        The keep rule is the sidecar's whole purpose: it exists to carry
        ``RunContext.data`` across a pause, so anything running, paused or
        resumable keeps it and only terminal or unrecoverable runs lose it.
        Reached from the head of a run holding the single slot, so nothing else
        is executing — and a resume is already in ``_active`` (status
        ``running``) by the time this walks the directory.

        Defensive like everything else on this path: housekeeping never raises
        into a run.
        """
        shared = self.checkpoints.parent / "shared"
        try:
            files = list(shared.glob("*.json")) if shared.is_dir() else []
        except OSError:
            return
        for path in files:
            try:
                status, _ = self._status_of(path.stem, self._lines(path.stem))
                if status in ("running", "paused", "resumable"):
                    continue
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def _enforce_retention(self) -> None:
        """Keep the newest ``_RETAIN_TERMINAL_RUNS`` terminal runs' checkpoints.

        A run's checkpoints used to vanish the instant it finished, so the DB
        only ever held work in flight. Forking needs finished runs to still be
        re-enterable, which turns that file into a persistent one — and a
        persistent file with no eviction is just a slow leak. This is the
        eviction.

        Only DONE and ERRORED runs are candidates. Anything running, paused or
        resumable is skipped outright rather than merely ranked low: the whole
        point of its checkpoint is that something still intends to use it, and a
        cap that could delete one would turn a busy afternoon into a lost run.
        Ordering is by the run's own start timestamp from the JSONL — the same
        head line ``list_runs`` orders history by — so what the user sees at the
        bottom of the list is what ages out first.

        Called at the head of a run (see ``_run`` for why not the tail), which
        means the cap is enforced one run late: the observable steady state is
        ``_RETAIN_TERMINAL_RUNS + 1`` threads, the extra one being the run that
        just finished and has not yet been swept past.

        Defensive end to end: raising here would kill a run before it started
        over a housekeeping failure. A sweep that fails costs disk, not work.
        """
        self._sweep_sidecars()
        if not self.checkpoints.is_file():
            return
        try:
            con = sqlite3.connect(str(self.checkpoints))
            try:
                threads = [row[0] for row in con.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints "
                    "WHERE checkpoint_ns=''").fetchall()]
            finally:
                con.close()
        except Exception:
            return

        terminal = []
        for thread_id in threads:
            path = self._path(thread_id)
            try:
                lines = path.read_text(encoding="utf-8",
                                       errors="replace").split("\n")
            except OSError:
                # Checkpoints with no run file at all: nothing can read them and
                # nothing can fork them, so they are the first thing to go. Sort
                # key 0 puts them behind every dated run.
                terminal.append((0.0, thread_id))
                continue
            status, _ = self._status_of(thread_id, lines)
            if status not in ("done", "error"):
                continue
            head = _first_json(lines) or {}
            ts = head.get("ts")
            terminal.append((ts if isinstance(ts, (int, float)) else 0.0,
                             thread_id))
        if len(terminal) <= _RETAIN_TERMINAL_RUNS:
            return
        terminal.sort(key=lambda row: row[0], reverse=True)
        # Two DELETEs rather than SqliteSaver.delete_thread, which is those two
        # DELETEs and nothing else (read, not assumed). Doing it in SQL keeps
        # retention working without importing langgraph — this method is reached
        # from a finally that must not be the first thing to raise "install the
        # extra" — and puts the schema coupling in the same shape `_has_checkpoint`
        # above already has, rather than half in the API and half out of it.
        try:
            con = sqlite3.connect(str(self.checkpoints))
            try:
                for _, thread_id in terminal[_RETAIN_TERMINAL_RUNS:]:
                    con.execute("DELETE FROM writes WHERE thread_id=?",
                                (thread_id,))
                    con.execute("DELETE FROM checkpoints WHERE thread_id=?",
                                (thread_id,))
                con.commit()
            finally:
                con.close()
        except Exception:
            return

    def _status_of(self, run_id: str, lines) -> tuple:
        last = None
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
            break
        # A terminal line on disk is authoritative — a done/errored run stays so.
        if last is not None:
            if last.get("t") == "done":
                return "done", last.get("stats")
            if last.get("t") == "error":
                return "error", {"error": last.get("error")}
        # Live membership beats a stale last line. During a resume the run is
        # back in _active while its last disk line is still awaiting_approval;
        # without this check that window reads as "resumable" and a poll can
        # abort the resume it just kicked off.
        with self._lock:
            if run_id in self._active:
                return "running", None
            if run_id in self._paused:
                return "paused", None
        # Not running here and not tracked in memory. An awaiting_approval tail
        # or any un-terminated run is recoverable iff its checkpoint survives;
        # otherwise it was lost to a crash.
        if self._has_checkpoint(run_id):
            return "resumable", None
        return "interrupted", None

    def _has_checkpoint(self, run_id: str) -> bool:
        """Whether a live checkpoint for *run_id* survives in the DB. Cheap and
        defensive — a missing DB, a locked one, or a schema surprise all read as
        'no checkpoint' rather than raising into a status read."""
        if not self.checkpoints.is_file():
            return False
        try:
            con = sqlite3.connect(str(self.checkpoints))
            try:
                row = con.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id=? LIMIT 1",
                    (run_id,)).fetchone()
            finally:
                con.close()
            return row is not None
        except Exception:
            return False

    def reconcile_on_boot(self) -> None:
        """Terminalise runs the previous server left dangling — but only the
        UNRECOVERABLE ones. A run whose checkpoint survives is marked neither
        done nor errored: ``_status_of`` reports it ``resumable`` and the user
        (or Phase-2 resume) continues it. Only a run with no checkpoint gets the
        error stamp, safe because that writer thread died with the process."""
        for p in self.dir.glob("*.jsonl"):
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                continue
            status, _ = self._status_of(p.stem, lines)
            if status == "interrupted":
                try:
                    with open(p, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "t": "error",
                            "error": "interrupted: panel server restarted",
                        }, ensure_ascii=False) + "\n")
                except OSError:
                    continue
