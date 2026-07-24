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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from memsom.providers.base import ProviderError, now
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
# bounded wait for a cold engine to come up
_WARMUP_TIMEOUT_S = 60.0
_WARMUP_POLL_S = 2.0


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

    def as_start_meta(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "agent": self.agent_name,
            "provider": self.provider_id,
            "model": self.model,
            "tools": [t["name"] for t in self.tool_specs],
            "limits": self.limits,
        }


@dataclass
class RouterSpec:
    """A branch point — one compiled ``router`` node.

    A router is not an agent: it owns no engine and no prompt. In ``decide``
    mode it borrows the engine of the agent that fed it (``source_agent``) for
    one small inference whose only job is to name a branch; in ``match`` mode
    it never infers at all and regexes that agent's final text. Either way an
    undecidable result falls to ``else_branch``, which is why compile time
    insists the else names one of this router's own branches — a router that
    can fail to route is a graph that can hang.
    """

    node_id: str
    mode: str                 # "decide" | "match"
    branches: list            # [{"name","when","target_node"}]
    else_branch: str          # always one of branches[*]["name"]
    source_agent: str         # node id of the agent whose output it reads


@dataclass
class GraphSpec:
    """A validated, runnable GRAPH — the output of ``compile_graph``.

    ``flow_edges`` is the control-flow adjacency and nothing else: for each
    agent and router node, the nodes it may hand off to. An agent has at most
    one entry (fan-out without a router would mean parallel agents sharing one
    message thread, which is a different feature); a router has one per branch.
    Resource wiring — engine→agent, tool→agent — is *not* in here; it was
    already folded into each :class:`AgentSpec`.

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
            "agents": [
                {"node_id": a.node_id, "name": a.agent_name,
                 "provider": a.provider_id, "model": a.model,
                 "tools": [t["name"] for t in a.tool_specs]}
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
    )


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
        tool_specs.append({"name": name, "type": t_type,
                           "options": t_cfg.get("options") or {}})
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

    params = dict(a_cfg.get("params") or {})
    if transport:
        params["transport"] = transport

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
    if mode not in ("decide", "match"):
        raise ProviderError(
            f"router {label!r} has an unknown mode: {mode!r} "
            "(expected 'decide' or 'match')")

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

    An agent may hand off to at most ONE agent or router. Fan-out is refused
    rather than silently truncated: two live successors sharing one message
    thread is a parallelism feature nobody designed, and picking one at random
    would make the canvas lie about what runs.
    """
    flow: dict = {}
    for aid, agent in agents.items():
        outs = list(dict.fromkeys(
            e.get("target") for e in edges
            if e.get("source") == aid
            and (nodes.get(e.get("target")) or {}).get("type") in _FLOW_TYPES))
        live = [t for t in outs
                if nodes[t].get("type") in ("agent", "router")]
        if len(live) > 1:
            raise ProviderError(
                f"agent {agent.agent_name!r} has more than one outgoing "
                "control-flow edge; wire it into a router to branch")
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
        stats = adapter.infer(model, messages, params, sink) or {}
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
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.jsonl"

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

    def _run(self, spec: "GraphSpec", run_id: str, path: Path) -> None:
        sink = AgentFileSink(path)
        try:
            self._warmup(spec, sink)
            stats = self._loop(spec, sink)
            sink.done(_final_stats(sink, stats))
        except ProviderError as exc:
            sink.error(str(exc))
        except Exception as exc:  # defensive: never die silently
            sink.error(f"internal error: {exc}")
        finally:
            with self._lock:
                self._active.discard(run_id)
            self._slots.release()

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

    def _loop(self, spec: "GraphSpec", sink: AgentFileSink) -> dict:
        """Hand the compiled graph to the LangGraph runtime.

        Imported here and not at module scope on purpose: ``lc_runtime`` pulls
        in langgraph and langchain-core, and memsom's core is stdlib-only. A
        machine that never runs an agent never pays for the import, and one
        that tries gets a ProviderError naming the extra instead of a
        traceback at server boot."""
        from memsom.providers.lc_runtime import run_graph
        return run_graph(spec, self.registry, sink, self.audit_path)

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
            })
        return out

    def _status_of(self, run_id: str, lines) -> tuple:
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("t") == "done":
                return "done", rec.get("stats")
            if rec.get("t") == "error":
                return "error", {"error": rec.get("error")}
            break
        with self._lock:
            if run_id in self._active:
                return "running", None
        return "interrupted", None

    def reconcile_on_boot(self) -> None:
        """Stamp a terminal line onto any run file the previous server left
        unterminated — safe because that writer thread died with the process."""
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
