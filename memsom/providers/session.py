"""Durable inference sessions — generation that outlives the app.

The problem: if the tokens stream back over the same HTTP connection the app
opened, closing the app kills the socket and kills the generation. The fix:
decouple them. A POST starts a generation on a background thread that writes
every token to an append-only file; the app just POLLS that file. Close the app
mid-stream and the thread keeps writing to disk (the panel server, which owns
the thread, is never killed by the app); reopen and re-poll from the last cursor
— the transcript is intact.

File format — one JSON object per line at
``<sessions_dir>/<session_id>.jsonl``:

    {"t":"start","provider":..,"model":..,"params":..,"ts":..}
    {"t":"tok","text":".."}          # many
    {"t":"done","stats":{..}}        # exactly one terminal line, OR
    {"t":"error","error":".."}

Cursor = line index. ``read_since(id, N)`` returns lines[N:] and the new cursor,
plus a status derived from whether a terminal line is present. Many readers, and
— since agent graphs learned to fan out — possibly several writers, so
:class:`FileSink` serializes every append on its own lock (see the class).

Durability scope (honest): this survives the *app* closing, because the panel
server keeps running. It does NOT survive the panel *server* itself restarting —
an in-flight thread dies with the process, leaving a file with no terminal line.
That's the Phase-3 concern (a detached session host); here a server restart is
rare and the partial transcript is still readable.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Optional

from memsom.providers.base import ProviderError, Sink, now

# session ids become filenames — fence hard against path traversal. Also the
# shape the app should generate.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def new_session_id() -> str:
    return uuid.uuid4().hex


def valid_session_id(session_id: str) -> bool:
    return bool(session_id) and bool(_SESSION_ID_RE.match(session_id))


class FileSink(Sink):
    """Append-only token sink over one session file.

    Flush-per-token (visible to other processes immediately) but fsync only on
    the terminal line — fsync-per-token would gate TPS on disk latency for no
    gain, since 'survives app close' only needs cross-process visibility on the
    same machine, which flush already gives.

    **Every append is serialized on ``_lock``, and the counters ride inside it.**
    This file used to have a single writer by construction (one generation
    thread), and the docstring said so. Fan-out made that false: two agent nodes
    running concurrently share one RunContext and therefore one sink.

    What the lock is and is not for, measured rather than assumed — because the
    obvious claim turned out to be wrong. A line does NOT tear without it:
    ``TextIOWrapper.write`` is internally locked in CPython, and 3200 concurrent
    writes at a 1µs switch interval produced zero unparseable lines. What IS
    unguarded is everything AROUND the write — ``count``, ``think_count``,
    ``t_first``, ``t_last`` are ordinary read-modify-writes, they feed
    ``_final_stats``, and a run whose token count disagrees with its own
    transcript is a wrong audit line. So the counters are held under the SAME
    acquisition as the append they describe, rather than a separate one.

    The lock also buys not depending on that CPython detail at all. Atomic
    ``write`` is an implementation property of the GIL-era io stack, not a
    guarantee of the io contract, and it is exactly the kind of thing a
    free-threaded build changes. A file this codebase treats as its audit source
    should not rest on it.

    Note what the lock does NOT fix, so nobody assumes it did: a ``token`` that
    arrives after ``done`` has closed the handle still raises, locked or not
    (measured both ways). That is a lifecycle question — the run is over — and
    it is unchanged from before fan-out existed.
    """

    #: this sink's ``token`` accepts an optional node id. Checked by callers
    #: (``lc_model._TeeSink``) rather than probed with a TypeError, because a
    #: TypeError raised from INSIDE an inner sink is indistinguishable from one
    #: raised by the call shape — and guessing wrong means a token written twice.
    accepts_node = True

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.count = 0
        self.think_count = 0
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def _write(self, obj: dict, sync: bool = False) -> None:
        with self._lock:
            self._write_locked(obj, sync)

    def _write_locked(self, obj: dict, sync: bool = False) -> None:
        """The append itself. Callers must already hold ``_lock`` — this exists
        so a method that also mutates counters can do both under ONE
        acquisition instead of releasing in between."""
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        if sync:
            os.fsync(self._fh.fileno())

    def token(self, text: str, node: str = "") -> None:
        """One answer chunk. *node* names the agent that produced it, and is
        omitted from the written object when empty — which is every caller
        outside a multi-agent graph, so a single-agent run's file is byte for
        byte what it always was."""
        if not text:
            return
        with self._lock:
            t = now()
            if self.t_first is None:
                self.t_first = t
            self.t_last = t
            self.count += 1
            obj = {"t": "tok", "text": text}
            if node:
                obj["node"] = node
            self._write_locked(obj)

    def reasoning(self, text: str) -> None:
        """A thinking chunk, written as its own event type so a reader can show
        it apart from the answer (or not at all). Counted separately: these ARE
        generated tokens and belong in the throughput timing, but calling them
        answer tokens would overstate how much reply you got."""
        if not text:
            return
        with self._lock:
            t = now()
            if self.t_first is None:
                self.t_first = t
            self.t_last = t
            self.think_count += 1
            self._write_locked({"t": "think", "text": text})

    def done(self, stats: dict) -> None:
        with self._lock:
            self._write_locked({"t": "done", "stats": stats}, sync=True)
            self._fh.close()

    def error(self, message: str) -> None:
        with self._lock:
            self._write_locked({"t": "error", "error": message}, sync=True)
            self._fh.close()

    def elapsed(self) -> float:
        if self.t_first is None or self.t_last is None:
            return 0.0
        return max(0.0, self.t_last - self.t_first)


class AgentFileSink(FileSink):
    """FileSink plus free-form event lines (turn, tool_call, tool_result…).

    A single-infer session only ever emits ``tok``/``done``/``error``; a
    tool-loop session ALSO needs to record the loop's structural events. They
    share one file format — ``read_since`` parses every line as an event, so a
    reader that only renders ``t=="tok"`` (the inference/voice frontend) simply
    ignores the extra lines. Home is here beside :class:`FileSink` so both the
    agent runner and the voice tool-loop can reuse it without importing across
    the agents↔session boundary."""

    def event(self, obj: dict, sync: bool = False) -> None:
        self._write(obj, sync=sync)


class SessionRunner:
    """Owns the inference sessions directory and spawns generation threads."""

    def __init__(self, sessions_dir) -> None:
        self.dir = Path(sessions_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.jsonl"

    def start(self, provider, model: str, messages: list, params: dict,
              session_id: Optional[str] = None) -> str:
        """Write the start line and launch the generation thread. *messages* is
        the conversation ([{role, content}, ...]) so multi-turn chat carries
        prior turns. Returns the session id immediately — no wait for tokens."""
        sid = session_id or new_session_id()
        if not valid_session_id(sid):
            raise ProviderError("invalid session_id")
        path = self._path(sid)

        # start line, fsync'd, before the thread — so a poll that races the
        # thread still finds a well-formed file. (turns count only — never the
        # message bodies, which stay out of this metadata line.)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "t": "start", "provider": provider.id, "model": model,
                "turns": len(messages or []), "params": _safe_params(params),
                "ts": now(),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

        thread = threading.Thread(
            target=self._run, args=(provider, model, messages, params, path),
            name=f"infer-{sid}", daemon=True)
        thread.start()
        return sid

    def _run(self, provider, model, messages, params, path) -> None:
        sink = FileSink(path)
        try:
            adapter_stats = provider.infer(model, messages, params, sink) or {}
            stats = _final_stats(sink, adapter_stats)
            sink.done(stats)
        except ProviderError as exc:
            sink.error(str(exc))
        except Exception as exc:  # defensive: never let a thread die silently
            sink.error(f"internal error: {exc}")

    def start_agentic(self, provider, model: str, params: dict,
                      session_id: Optional[str], loop_fn) -> str:
        """Like :meth:`start`, but the generation thread runs a caller-supplied
        tool loop instead of one ``provider.infer`` call.

        *loop_fn* is ``loop_fn(sink) -> stats`` and is injected by the caller
        (voice_handlers) so session.py stays free of the tool layer — no
        circular import back into agents.py. The thread uses an
        :class:`AgentFileSink` so the loop can emit ``turn``/``tool_call``/
        ``tool_result`` events beside the streamed ``tok`` lines; the file
        format is otherwise identical, so :meth:`read_since` (and the frontend
        cursor-poll) need no change. Returns the session id immediately."""
        sid = session_id or new_session_id()
        if not valid_session_id(sid):
            raise ProviderError("invalid session_id")
        path = self._path(sid)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "t": "start", "provider": provider.id, "model": model,
                "turns": 1, "params": _safe_params(params), "ts": now(),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        thread = threading.Thread(
            target=self._run_agentic, args=(loop_fn, path),
            name=f"voice-{sid}", daemon=True)
        thread.start()
        return sid

    def _run_agentic(self, loop_fn, path) -> None:
        sink = AgentFileSink(path)
        try:
            stats = loop_fn(sink) or {}
            sink.done(_final_stats(sink, stats))
        except ProviderError as exc:
            sink.error(str(exc))
        except Exception as exc:  # defensive: never let a thread die silently
            sink.error(f"internal error: {exc}")

    def read_since(self, session_id: str, cursor: int = 0) -> dict:
        """Return events appended since *cursor* plus the new cursor and a
        derived status. Never raises on a missing/short file — a poll that beats
        the writer just gets an empty slice."""
        if not valid_session_id(session_id):
            raise ProviderError("invalid session_id")
        path = self._path(session_id)
        if not path.is_file():
            return {"events": [], "cursor": cursor, "status": "unknown"}
        # split on "\n" only — records are newline-delimited by construction
        # (FileSink._write). str.splitlines() also breaks on U+2028/U+2029/
        # U+0085, which json.dumps(ensure_ascii=False) writes literally, so it
        # would fragment and silently drop any record carrying those chars.
        # Drop the single trailing "" that split() leaves after the final "\n"
        # (splitlines does not) — else cursor=len(lines) overshoots by one and
        # buries the next record below the cursor on every subsequent poll.
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
                continue  # a half-written trailing line — next poll gets it
        status, stats = _status_of(lines)
        return {"events": events, "cursor": len(lines), "status": status,
                "stats": stats}

    def list_sessions(self, limit: int = 50) -> list:
        """Newest sessions first, with light header/status metadata."""
        files = sorted(self.dir.glob("*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for p in files:
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
            except OSError:
                continue
            head = _first_json(lines)
            status, _ = _status_of(lines)
            out.append({
                "session_id": p.stem,
                "provider": (head or {}).get("provider"),
                "model": (head or {}).get("model"),
                "ts": (head or {}).get("ts"),
                "status": status,
            })
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _safe_params(params: dict) -> dict:
    """What we're willing to persist about a request — the knobs, never the
    prompt body (persisted separately as tokens/echo is the model's job, and
    we keep the audit-side metadata clean)."""
    if not isinstance(params, dict):
        return {}
    keep = ("temperature", "top_p", "ctx", "num_ctx", "max_tokens", "transport",
            "thinking", "effort")
    return {k: params[k] for k in keep if k in params}


def _final_stats(sink: FileSink, adapter_stats: dict) -> dict:
    """Merge wall-clock timing with any authoritative counters the backend
    reported. Prefer the backend's own eval_count/eval_duration for TPS (exact);
    fall back to token-count / wall-clock elapsed."""
    tokens = sink.count
    thinking = getattr(sink, "think_count", 0)
    elapsed = sink.elapsed()
    tps = None
    ev_count = adapter_stats.get("eval_count")
    ev_dur = adapter_stats.get("eval_duration_s")
    if ev_count and ev_dur:
        tps = round(ev_count / ev_dur, 2)
    elif (tokens + thinking) and elapsed > 0:
        # thinking chunks are real generated tokens — excluding them from the
        # rate would understate throughput on a reasoning model, sometimes to
        # zero when the whole budget went to the scratchpad.
        tps = round((tokens + thinking) / elapsed, 2)
    stats = {"tokens": tokens, "elapsed_s": round(elapsed, 3), "tps": tps}
    if thinking:
        stats["thinking_tokens"] = thinking
    # carry through any extra numeric fields a backend supplied (prompt tokens,
    # cost, etc.) without letting it overwrite our computed ones.
    for k, v in adapter_stats.items():
        stats.setdefault(k, v)
    return stats


def _status_of(lines) -> tuple:
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
    return "running", None


def _first_json(lines) -> Optional[dict]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    return None
