"""HTTP-facing handlers for the agent layer — same texture as handlers.py:
pure ``(deps, payload) -> (status, body)`` functions, thin panel routes,
two-phase audit on every mutation (graph saves, run starts; individual tool
executions are audited inside AgentRunner). Prompt/system bodies are never
audited — node counts and tool names only.
"""

from __future__ import annotations

from memsom.providers.agents import AgentRunner, compile_graph
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
