"""Agent scheduler — the panel's first in-process background thread.

Everything before this was request/response; scheduled model runs were external
OS tasks (schtasks). This is a single daemon thread that wakes every ``tick_s``
seconds, scans the saved graph docs for enabled TRIGGER schedules, and fires the
ones that are due through the same ``AgentRunner`` a manual RUN uses.

Design contracts (from plan elegant-riding-treehouse):
  - **No schedules.json.** The schedule lives on the trigger node inside the
    graph doc (mode="schedule", enabled, schedule={kind,...}); the scheduler
    re-reads graphs each tick, so editing a schedule in the canvas is picked up
    with no separate registration step. The ONLY materialized state is
    ``scheduler_state.json`` — per-graph ``last_fired / last_run_id /
    last_status`` plus a ``daily_marker`` date, written atomically.
  - **One run slot.** AgentRunner is a global single slot (local engines share
    one GPU). A due graph that finds the slot busy is skipped and recorded — no
    queue, no double-run.
  - **No backfill.** A daily schedule missed while the panel was down (or busy
    past a grace window) is recorded as ``missed`` and NOT fired late. Interval
    schedules simply resume from now.
  - **Exception-fenced.** A bad tick never kills the thread — the loop is the
    autonomy layer, so it must outlive any single graph's failure.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import uuid
from pathlib import Path

from memsom.providers.agents import compile_graph
from memsom.providers.base import ProviderError, now

# A daily target still counts as "fire now" if we cross it within this window;
# past it (panel was down or busy through the whole window) it's a recorded
# miss, never a late fire. Comfortably covers a 30s tick's evaluation lag.
_DAILY_GRACE_S = 120.0
_MIN_INTERVAL_S = 60


class Scheduler:
    def __init__(self, store, runner, registry, audit_path, state_path,
                 *, tick_s: int = 30) -> None:
        self.store = store
        self.runner = runner
        self.registry = registry
        self.audit_path = Path(audit_path)
        self.state_path = Path(state_path)
        self.tick_s = tick_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict = self._load_state()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="agent-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t:
            t.join(timeout=2)
        self._thread = None

    def _loop(self) -> None:
        # wait-first: gives the server a beat to finish booting before the first
        # scan, and Event.wait returns True the instant stop() fires (clean exit
        # mid-interval instead of sleeping out the tick).
        while not self._stop.wait(self.tick_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — autonomy layer must not die
                pass

    # -- the tick ----------------------------------------------------------

    def tick(self, *, now_ts: float | None = None) -> None:
        """One scan. Public + now_ts-injectable so a test can drive it directly
        without the thread or real wall-clock."""
        now_ts = now() if now_ts is None else now_ts
        summaries = self.store.list()
        live_ids = set()
        dirty = False
        for s in summaries:
            sched = s.get("schedule")
            if not sched or not sched.get("enabled"):
                continue
            gid = s.get("id")
            if not gid:
                continue
            live_ids.add(gid)
            verdict, marker = self._due(gid, sched, now_ts)
            if verdict == "wait" or verdict == "none":
                continue
            dirty = True
            with self._lock:
                st = dict(self._state.get(gid) or {})
            if verdict == "missed":
                st.update(last_status="missed", daily_marker=marker)
            elif verdict == "invalid-time":
                st.update(last_status="invalid-time", daily_marker=marker)
            elif verdict == "fire":
                st = self._apply_fire(gid, st, now_ts, marker,
                                      daily=(marker is not None))
            with self._lock:
                self._state[gid] = st
        # drop state for graphs no longer scheduled, so it can't grow unbounded
        with self._lock:
            stale = [k for k in self._state if k not in live_ids]
            for k in stale:
                del self._state[k]
                dirty = True
        if dirty:
            self._save_state()

    def _due(self, gid: str, sched: dict, now_ts: float) -> tuple:
        """Return (verdict, daily_marker). verdict is one of wait/none/fire/
        missed/invalid-time. daily_marker is the resolved local date for daily
        (so the caller stamps it and won't re-resolve today), else None."""
        kind = sched.get("kind") or "interval"
        with self._lock:
            st = dict(self._state.get(gid) or {})
        if kind == "daily":
            return self._due_daily(st, sched.get("at") or "", now_ts)
        # interval (default)
        every = sched.get("every_s")
        try:
            every = max(_MIN_INTERVAL_S, int(every))
        except (TypeError, ValueError):
            every = _MIN_INTERVAL_S
        last = st.get("last_fired") or 0
        return ("fire", None) if now_ts - last >= every else ("wait", None)

    def _due_daily(self, st: dict, at_str: str, now_ts: float) -> tuple:
        lt = datetime.datetime.fromtimestamp(now_ts)  # local wall clock
        today = lt.strftime("%Y-%m-%d")
        if st.get("daily_marker") == today:
            return ("none", today)  # already resolved (fired/missed) today
        try:
            hh, mm = (int(x) for x in at_str.split(":", 1))
            target = lt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except (ValueError, TypeError):
            return ("invalid-time", today)
        if lt < target:
            return ("wait", today)
        if (lt - target).total_seconds() <= _DAILY_GRACE_S:
            return ("fire", today)
        return ("missed", today)

    def _apply_fire(self, gid: str, st: dict, now_ts: float,
                    marker, *, daily: bool) -> dict:
        result, run_id = self._fire(gid)
        if result == "started":
            st.update(last_fired=now_ts, last_run_id=run_id, last_status="fired")
            if daily:
                st["daily_marker"] = marker
        elif result == "skip-busy":
            # leave last_fired/daily_marker untouched so it retries next tick
            # (still within a daily grace window, or the next interval boundary).
            st["last_status"] = "skip-busy"
        else:  # error / refused-audit: resolve this cycle so it can't hammer
            st["last_status"] = result
            if daily:
                st["daily_marker"] = marker
            else:
                st["last_fired"] = now_ts
        return st

    def _fire(self, gid: str) -> tuple:
        """Compile + start one scheduled run under the two-phase audit. Returns
        (result, run_id) — result in started/skip-busy/refused-audit/error:*."""
        from memsom.providers.handlers import _audit
        intent = {"action": "agent-schedule-fire", "graph_id": gid}
        try:
            _audit(self.audit_path, {**intent, "result": "pending"}, gate=True)
        except OSError as exc:
            return (f"refused-audit: {exc}", None)
        try:
            graph = self.store.get(gid)
            spec = compile_graph(graph, self.registry)
        except ProviderError as exc:
            _audit(self.audit_path, {**intent, "result": f"failed: {exc}"})
            return (f"error: {exc}", None)
        try:
            run_id = self.runner.start(spec, trigger="schedule")
        except ProviderError as exc:
            busy = "already active" in str(exc)
            _audit(self.audit_path, {**intent,
                   "result": "skip-busy" if busy else f"failed: {exc}"})
            return ("skip-busy" if busy else f"error: {exc}", None)
        _audit(self.audit_path, {**intent, "result": "started", "run_id": run_id})
        return ("started", run_id)

    # -- status ------------------------------------------------------------

    def status(self) -> dict:
        now_ts = now()
        summaries = self.store.list()
        out = []
        with self._lock:
            state = {k: dict(v) for k, v in self._state.items()}
        for s in summaries:
            sched = s.get("schedule")
            if not sched or not sched.get("enabled"):
                continue
            gid = s.get("id")
            st = state.get(gid, {})
            out.append({
                "graph_id": gid,
                "name": s.get("name"),
                "kind": sched.get("kind") or "interval",
                "every_s": sched.get("every_s"),
                "at": sched.get("at"),
                "last_fired": st.get("last_fired"),
                "last_run_id": st.get("last_run_id"),
                "last_status": st.get("last_status"),
                "next_fire": self._next_fire(sched, st, now_ts),
            })
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "tick_s": self.tick_s,
            "schedules": out,
        }

    def _next_fire(self, sched: dict, st: dict, now_ts: float):
        """Best-effort epoch of the next fire, for the UI. None if unknowable."""
        kind = sched.get("kind") or "interval"
        if kind == "interval":
            try:
                every = max(_MIN_INTERVAL_S, int(sched.get("every_s")))
            except (TypeError, ValueError):
                return None
            last = st.get("last_fired") or 0
            return max(now_ts, last + every)
        try:
            hh, mm = (int(x) for x in (sched.get("at") or "").split(":", 1))
        except (ValueError, TypeError, AttributeError):
            return None
        lt = datetime.datetime.fromtimestamp(now_ts)
        target = lt.replace(hour=hh, minute=mm, second=0, microsecond=0)
        # if today's slot is already resolved or past, the next one is tomorrow
        if st.get("daily_marker") == lt.strftime("%Y-%m-%d") or lt >= target:
            target += datetime.timedelta(days=1)
        return target.timestamp()

    # -- persistence -------------------------------------------------------

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        with self._lock:
            snapshot = json.dumps(self._state, ensure_ascii=False, indent=2)
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
            tmp.write_text(snapshot, encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass  # a failed state write costs at worst a duplicate fire; never fatal
