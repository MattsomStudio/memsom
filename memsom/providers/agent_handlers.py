"""HTTP-facing handlers for the agent layer — same texture as handlers.py:
pure ``(deps, payload) -> (status, body)`` functions, thin panel routes,
two-phase audit on every mutation (graph saves, run starts; individual tool
executions are audited inside AgentRunner). Prompt/system bodies are never
audited — node counts and tool names only.
"""

from __future__ import annotations

from memsom.providers.agents import AgentRunner, _fan_sets, compile_graph
from memsom.providers.agent_store import GraphStore
from memsom.providers.base import ProviderError
from memsom.providers.handlers import _audit
from memsom.providers.tools import tool_catalog


def handle_graphs_list(store: GraphStore) -> tuple:
    return 200, {"ok": True, "graphs": store.list()}


def handle_graph_get(store: GraphStore, graph_id: str) -> tuple:
    try:
        return 200, {"ok": True, "graph": store.get(graph_id or "")}
    except ProviderError as exc:
        return 404, {"ok": False, "error": str(exc)}


def handle_graph_save(store: GraphStore, audit_path, payload: dict) -> tuple:
    if not isinstance(payload, dict) or not isinstance(payload.get("graph"), dict):
        return 400, {"ok": False, "error": "body must be {graph: {...}}"}
    graph = payload["graph"]
    intent = {"action": "graph-save", "graph_id": graph.get("id"),
              "nodes": len(graph.get("nodes") or []),
              "edges": len(graph.get("edges") or [])}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        gid, rev = store.save(graph)
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"failed: {exc}"})
        return 400, {"ok": False, "error": str(exc)}
    _audit(audit_path, {**intent, "result": "ok", "graph_id": gid, "rev": rev})
    return 200, {"ok": True, "id": gid, "rev": rev}


def handle_graph_delete(store: GraphStore, audit_path, payload: dict) -> tuple:
    gid = (payload or {}).get("id") or ""
    intent = {"action": "graph-delete", "graph_id": gid}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        store.delete(gid)
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"failed: {exc}"})
        return 404, {"ok": False, "error": str(exc)}
    _audit(audit_path, {**intent, "result": "ok"})
    return 200, {"ok": True}


def handle_tool_catalog() -> tuple:
    return 200, {"ok": True, "tools": tool_catalog()}


def handle_run_start(store: GraphStore, runner: AgentRunner, registry: dict,
                     audit_path, payload: dict) -> tuple:
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    gid = payload.get("graph_id") or ""
    input_override = payload.get("input")
    if input_override is not None and not isinstance(input_override, str):
        return 400, {"ok": False, "error": "'input' must be a string"}
    try:
        graph = store.get(gid)
        spec = compile_graph(graph, registry, input_override=input_override)
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}

    intent = {"action": "agent-run", "graph_id": gid,
              "provider": spec.provider_id, "model": spec.model,
              "tools": [t["name"] for t in spec.tool_specs]}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        run_id = runner.start(spec, trigger="manual")
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"failed: {exc}"})
        busy = "already active" in str(exc)
        return (409 if busy else 400), {"ok": False, "error": str(exc)}
    _audit(audit_path, {**intent, "result": "started", "run_id": run_id})
    return 200, {"ok": True, "run_id": run_id, "cursor": 0}


def handle_approve(store: GraphStore, runner: AgentRunner, registry: dict,
                   audit_path, payload: dict) -> tuple:
    """Resume a paused run — an approval gate OR a static breakpoint.

    ``{run_id, decision}`` where decision is ``approve``, ``deny``,
    ``continue`` (step past a breakpoint) or
    ``{"decision": "edit", "arguments": {…}}`` (approve, but with the arguments
    the human substituted). One endpoint for all four because they are one
    mechanism — ``AgentRunner.resume`` off the same checkpoint — and a second
    route would duplicate the spec-rebuild, the audit and the busy handling to
    say the same thing in a different URL.

    The spec comes from the runner's memory for a live pause; if the pause
    outlived a restart it is recompiled from the graph doc named on the run's
    start line — preferring the original in-memory spec so a mid-pause graph
    edit can't warp the resume. The decision is audited: approving a gated
    shell call is itself a security event, and so is rewriting its arguments.

    Whether the decision FITS the pause is ``AgentRunner.resume``'s call, not
    this function's — it reads the pause kind off the run log, which this layer
    is not the place to parse. Two things it refuses there and the 400s that
    surface here: a decision against a run that already finished (retention
    stopped "has a checkpoint" from meaning "is waiting for you"), and a verdict
    against a static breakpoint, which has no ``interrupt()`` to consume it and
    used to be recorded in the transcript as a human approval of nothing."""
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    run_id = payload.get("run_id") or ""
    raw = payload.get("decision")
    if isinstance(raw, dict):
        # Validated HERE rather than in the tool: a malformed edit that reaches
        # MemsomTool._run is refused (fail closed), which would burn the pause
        # on a typo. A 400 leaves the run paused and the human able to retry.
        if str(raw.get("decision") or "").lower() != "edit":
            return 400, {"ok": False,
                         "error": "the only object decision is "
                                  "{'decision':'edit','arguments':{…}}"}
        if not isinstance(raw.get("arguments"), dict):
            return 400, {"ok": False,
                         "error": "an 'edit' decision needs an 'arguments' object"}
        decision = {"decision": "edit", "arguments": dict(raw["arguments"])}
        audited = "edit"
    else:
        decision = str(raw or "").lower()
        if decision not in ("approve", "deny", "continue"):
            return 400, {"ok": False,
                         "error": "decision must be 'approve', 'deny', "
                                  "'continue' or "
                                  "{'decision':'edit','arguments':{…}}"}
        audited = decision

    spec = runner.paused_spec(run_id)
    if spec is None:
        # Pause survived a restart (or was never live here): rebuild from the doc.
        gid = runner.head_graph_id(run_id)
        if not gid:
            return 404, {"ok": False, "error": f"unknown run: {run_id!r}"}
        try:
            spec = compile_graph(store.get(gid), registry)
        except ProviderError as exc:
            return 400, {"ok": False, "error": str(exc)}

    # The audit records the KIND of decision, never the substituted arguments —
    # same redaction discipline as every other intent in this file (names and
    # counts, not payloads). The arguments that actually ran are audited by
    # _execute_tool a moment later, clipped, which is where they belong.
    intent = {"action": "agent-approve", "run_id": run_id, "decision": audited}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        runner.resume(run_id, decision, spec=spec)
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"failed: {exc}"})
        busy = "already active" in str(exc)
        return (409 if busy else 400), {"ok": False, "error": str(exc)}
    _audit(audit_path, {**intent, "result": "resumed"})
    return 200, {"ok": True, "run_id": run_id}


def fork_steps(events: list) -> list:
    """The superstep numbers a run can be forked from, ascending.

    Read off each ``node`` event's own ``step`` field, deduplicated — NOT
    counted. A resume REPLAYS the parent task it paused inside, so a run that
    stopped at an approval gate emits a second ``node`` line for a superstep
    that already ran (measured: a gated two-agent chain gives node events
    ``[A, A, B]`` across two supersteps). Counting them offered the user a step
    that had no checkpoint and shifted every later one onto the wrong state —
    picking "after A" silently seeded the state after B and produced a fork that
    ran nothing and reported ``done``. The replayed event carries the same
    number as the original, so deduplicating on it is exact.

    Runs written before the field existed fall back to the positional count,
    which is right for them: they are all runs that never paused, or the field
    would have been there.
    """
    nodes = [ev for ev in events if ev.get("t") == "node"]
    stamped = [ev["step"] for ev in nodes
               if isinstance(ev.get("step"), int)
               and not isinstance(ev.get("step"), bool)]
    if len(stamped) != len(nodes):
        return list(range(1, len(nodes) + 1))
    return sorted(set(stamped))


def handle_run_fork(store: GraphStore, runner: AgentRunner, registry: dict,
                    audit_path, payload: dict) -> tuple:
    """Re-run a finished run from after step N, optionally rewriting its answer.

    ``{run_id, step, edit?}``. Its own route rather than a flavour of
    ``/api/agents/approve`` because the lifecycles are different in kind: an
    approve continues the run you named, a fork SPAWNS one — new id, new file,
    new slot — and folding those into one endpoint would make the response mean
    two things.

    Everything this validates comes from the JSONL. The step list, the statuses
    and the prefill text a human forks with are all read from the run log; the
    checkpoint DB is not consulted here at all, and is touched exactly once
    downstream where its content becomes ``.invoke``'s input. That split is the
    whole discipline: two stores, one of which may answer "what happened".

    The spec is recompiled from the CURRENT graph doc — the same precedent a
    post-restart resume sets — because the usual reason to fork is that you just
    fixed the prompt that made the run go wrong.
    """
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "body must be a JSON object"}
    run_id = payload.get("run_id") or ""
    raw_step = payload.get("step")
    if not isinstance(raw_step, int) or isinstance(raw_step, bool):
        return 400, {"ok": False, "error": "'step' must be an integer"}
    edit = payload.get("edit")
    if edit is not None and not isinstance(edit, str):
        return 400, {"ok": False, "error": "'edit' must be a string"}

    try:
        data = runner.read_since(run_id, 0)
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}
    if data["status"] == "unknown":
        return 404, {"ok": False, "error": f"unknown run: {run_id!r}"}
    if data["status"] not in ("done", "error"):
        # A run still in flight has no settled steps: the state behind its last
        # node event is being written as we read it.
        return 400, {"ok": False,
                     "error": "only a finished run can be forked "
                              f"(this one is {data['status']})"}
    steps = fork_steps(data["events"])
    if raw_step not in steps:
        return 400, {"ok": False,
                     "error": f"step must be one of {steps} "
                              f"for run {run_id!r}"}

    gid = runner.head_graph_id(run_id)
    if not gid:
        return 404, {"ok": False, "error": f"unknown run: {run_id!r}"}
    try:
        spec = compile_graph(store.get(gid), registry)
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}

    # Fan-out graphs are refused, and the reason is the ordinal the picker is
    # built on. One parallel superstep runs N sibling nodes and writes ONE
    # checkpoint, so N node events in the JSONL collapse to a single step in the
    # DB and "fork from after B" has no state that means it. Checked with
    # `_fan_sets` rather than `spec.joins` because a fan-out that never
    # reconverges has no join to find but breaks the mapping just the same.
    if _fan_sets(spec.agents, spec.flow_edges):
        return 400, {"ok": False,
                     "error": "forking a fan-out graph is not supported yet"}

    intent = {"action": "agent-fork", "run_id": run_id, "step": raw_step,
              "edited": bool(edit)}
    try:
        _audit(audit_path, {**intent, "result": "pending"}, gate=True)
    except OSError as exc:
        return 503, {"ok": False, "error": f"audit unavailable; refused: {exc}"}
    try:
        new_id = runner.fork(spec, run_id, raw_step, edit=edit or None)
    except ProviderError as exc:
        _audit(audit_path, {**intent, "result": f"failed: {exc}"})
        busy = "already active" in str(exc)
        return (409 if busy else 400), {"ok": False, "error": str(exc)}
    _audit(audit_path, {**intent, "result": "started", "new_run_id": new_id})
    return 200, {"ok": True, "run_id": new_id, "cursor": 0}


def handle_run_read(runner: AgentRunner, run_id: str, cursor: int) -> tuple:
    try:
        data = runner.read_since(run_id or "", cursor)
    except ProviderError as exc:
        return 400, {"ok": False, "error": str(exc)}
    return 200, {"ok": True, **data}


def handle_runs_list(runner: AgentRunner) -> tuple:
    return 200, {"ok": True, "runs": runner.list_runs()}


def handle_scheduler_status(scheduler) -> tuple:
    """Scheduler liveness + per-graph schedule state for the AGENTS status chip.
    Defensive against a None scheduler (a hand-built PanelConfig in a test)."""
    if scheduler is None:
        return 200, {"ok": True, "running": False, "tick_s": None, "schedules": []}
    return 200, {"ok": True, **scheduler.status()}
