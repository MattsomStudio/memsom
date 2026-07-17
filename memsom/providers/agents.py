"""Agent runs — a durable, multi-turn tool-use loop over any provider.

An *agent* is a compiled graph: one engine (provider + model), a system
prompt, a set of tool instances, and hard limits. Running one is a loop the
panel server owns: infer → if the model asked for tools, execute them and
feed the results back → infer again → until a plain answer, a limit, or an
error. Every step lands in an append-only JSONL run file (the session.py
pattern with a wider event vocabulary), so the app can close mid-run and
re-poll the transcript later.

Run file — one JSON object per line at ``<runs_dir>/<run_id>.jsonl``:

    {"t":"start","run_id":..,"graph_id":..,"trigger":..,"provider":..,
     "model":..,"tools":[..],"limits":{..},"ts":..}
    {"t":"warmup","action":"start"|"none","ok":true,"detail":"..","ts":..}
    {"t":"turn","n":1,"ts":..}
    {"t":"tok","text":".."}                         # whole-turn text w/ tools
    {"t":"tool_call","turn":1,"id":"tc_1","name":..,"arguments":{..},"ts":..}
    {"t":"tool_result","turn":1,"id":"tc_1","name":..,"ok":true,
     "output":"..","bytes":123,"truncated":false,"elapsed_s":1.2}
    {"t":"done","stats":{..}}                       # terminal, fsync'd, OR
    {"t":"error","error":"..","turn":2}

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
    FileSink, new_session_id, valid_session_id, _final_stats, _first_json,
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
# identical consecutive tool calls before we call it a loop
_LOOP_STRIKES = 3
# bounded wait for a cold engine to come up
_WARMUP_TIMEOUT_S = 60.0
_WARMUP_POLL_S = 2.0


@dataclass
class AgentSpec:
    """A validated, runnable agent — the output of ``compile_graph``."""

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

    def as_start_meta(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "agent": self.agent_name,
            "provider": self.provider_id,
            "model": self.model,
            "tools": [t["name"] for t in self.tool_specs],
            "limits": self.limits,
        }


def compile_graph(graph: dict, registry: dict, *,
                  input_override: Optional[str] = None) -> AgentSpec:
    """Validate a graph document into an AgentSpec. Every failure raises
    ProviderError with a verbatim, user-facing reason (the route maps these
    to 400s)."""
    nodes = {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict)}
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]

    agents = [n for n in nodes.values() if n.get("type") == "agent"]
    if len(agents) != 1:
        raise ProviderError(
            f"graph must contain exactly one agent node (found {len(agents)})")
    agent = agents[0]
    a_cfg = agent.get("config") or {}

    def _into(target_handle: str) -> list:
        return [nodes.get(e.get("source")) for e in edges
                if e.get("target") == agent["id"]
                and e.get("targetHandle") == target_handle
                and nodes.get(e.get("source"))]

    engines = [n for n in _into("engine") if n.get("type") == "engine"]
    if len(engines) != 1:
        raise ProviderError(
            f"agent needs exactly one engine wired in (found {len(engines)})")
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

    # unique tool instance names: auto-suffix duplicates (http_fetch_2, ...)
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

    trigger_input = ""
    for n in nodes.values():
        if n.get("type") == "trigger":
            trigger_input = (n.get("config") or {}).get("input") or ""
            break

    params = dict(a_cfg.get("params") or {})
    if transport:
        params["transport"] = transport

    return AgentSpec(
        graph_id=str(graph.get("id") or ""),
        graph_rev=int(graph.get("rev") or 0),
        agent_name=a_cfg.get("name") or "AGENT",
        provider_id=provider_id,
        model=model,
        transport=transport,
        system=a_cfg.get("system") or "",
        params=params,
        tool_specs=tool_specs,
        limits=limits,
        input=input_override if input_override is not None else trigger_input,
    )


class AgentFileSink(FileSink):
    """FileSink plus free-form event lines (tool_call, turn, warmup...)."""

    def event(self, obj: dict, sync: bool = False) -> None:
        self._write(obj, sync=sync)


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

    def start(self, spec: AgentSpec, trigger: str) -> str:
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
        threading.Thread(target=self._run, args=(adapter, spec, run_id, path),
                         name=f"agent-{run_id}", daemon=True).start()
        return run_id

    def _run(self, adapter, spec: AgentSpec, run_id: str, path: Path) -> None:
        sink = AgentFileSink(path)
        try:
            self._warmup(adapter, sink)
            stats = self._loop(adapter, spec, run_id, sink)
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

    def _warmup(self, adapter, sink: AgentFileSink) -> None:
        """Cold engines get started; warm ones pass through. Never unloads
        anyone else's model — VRAM admission control is deliberately not
        this layer's job."""
        try:
            state = adapter.status().state
        except Exception:
            state = "down"
        caps = adapter.capabilities()
        if state == "up" or not getattr(caps, "can_start", False):
            sink.event({"t": "warmup", "action": "none", "ok": True,
                        "detail": state, "ts": now()})
            return
        sink.event({"t": "warmup", "action": "start", "ok": True,
                    "detail": "starting engine", "ts": now()})
        adapter.start()
        deadline = now() + _WARMUP_TIMEOUT_S
        while now() < deadline:
            try:
                if adapter.status().state == "up":
                    sink.event({"t": "warmup", "action": "start", "ok": True,
                                "detail": "engine up", "ts": now()})
                    return
            except Exception:
                pass
            time.sleep(_WARMUP_POLL_S)
        raise ProviderError("engine did not come up within warmup timeout")

    def _loop(self, adapter, spec: AgentSpec, run_id: str,
              sink: AgentFileSink) -> dict:
        tools = build_tools(spec.tool_specs)
        by_name = {t.name: t for t in tools}
        ctx = ToolContext(
            audit_path=self.audit_path,
            timeout_s=spec.limits["tool_timeout_s"],
            max_output_bytes=spec.limits["max_tool_output_bytes"],
        )
        params = dict(spec.params)
        if tools:
            params["tools"] = to_openai_tools(tools)

        messages: list = []
        if spec.system:
            messages.append({"role": "system", "content": spec.system})
        messages.append({"role": "user", "content": spec.input or "Begin."})

        started = now()
        last_sig, strikes = None, 0
        agg: dict = {}
        tool_call_count = 0

        for turn in range(1, spec.limits["max_turns"] + 1):
            if now() - started > spec.limits["run_timeout_s"]:
                raise ProviderError(
                    f"run timeout after {spec.limits['run_timeout_s']}s")
            sink.event({"t": "turn", "n": turn, "ts": now()})
            stats = adapter.infer(spec.model, messages, params, sink) or {}
            for k in ("prompt_tokens", "eval_count"):
                if isinstance(stats.get(k), (int, float)):
                    agg[k] = agg.get(k, 0) + stats[k]
            calls = stats.get("tool_calls") or []
            if not calls:
                agg["turns"] = turn
                agg["tool_calls"] = tool_call_count
                return agg

            # loop detection: the same single call, repeatedly
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
                output, ok = self._execute(by_name.get(name), name,
                                           arguments, ctx,
                                           available=sorted(by_name))
                text, truncated = truncate_output(
                    output, spec.limits["max_tool_output_bytes"])
                sink.event({"t": "tool_result", "turn": turn, "id": cid,
                            "name": name, "ok": ok, "output": text,
                            "bytes": len(output.encode("utf-8", "ignore")),
                            "truncated": truncated,
                            "elapsed_s": round(now() - t0, 3)})
                messages.append({"role": "tool", "tool_call_id": cid,
                                 "name": name, "content": text})
        raise ProviderError(
            f"max turns reached ({spec.limits['max_turns']}) without a final answer")

    def _execute(self, tool: Optional[Tool], name: str, arguments: dict,
                 ctx: ToolContext, *, available: list) -> tuple:
        """Run one tool call under the two-phase audit. Model-level mistakes
        (unknown tool, bad args, tool failure) come back as a failing result
        string — the model gets to react; only audit unavailability kills
        the run."""
        from memsom.providers.handlers import _audit
        intent = {"action": "tool", "tool": name,
                  "arguments": {k: str(v)[:200] for k, v in arguments.items()}}
        try:
            _audit(self.audit_path, {**intent, "result": "pending"}, gate=True)
        except OSError as exc:
            raise ProviderError(f"audit unavailable; refused: {exc}") from exc
        if tool is None:
            _audit(self.audit_path, {**intent, "result": "refused-unknown-tool"})
            return (f"unknown tool {name!r}; available: "
                    f"{', '.join(available) or 'none'}", False)
        try:
            out = tool.run(arguments, ctx)
        except ToolError as exc:
            _audit(self.audit_path, {**intent, "result": f"failed: {exc}"})
            return f"tool error: {exc}", False
        except Exception as exc:
            _audit(self.audit_path, {**intent, "result": f"error: {exc}"})
            return f"tool internal error: {exc}", False
        _audit(self.audit_path, {**intent, "result": "ok"})
        return out, True

    # -- reads -------------------------------------------------------------

    def read_since(self, run_id: str, cursor: int = 0) -> dict:
        if not valid_session_id(run_id):
            raise ProviderError("invalid run_id")
        path = self._path(run_id)
        if not path.is_file():
            return {"events": [], "cursor": cursor, "status": "unknown"}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
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
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
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
