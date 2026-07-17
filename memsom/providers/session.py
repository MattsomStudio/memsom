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
plus a status derived from whether a terminal line is present. Single writer per
file (the generating thread), many readers — so no per-file lock is needed.

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
    same machine, which flush already gives."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.count = 0
        self.t_first: Optional[float] = None
        self.t_last: Optional[float] = None
        self._fh = open(self.path, "a", encoding="utf-8")

    def _write(self, obj: dict, sync: bool = False) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()
        if sync:
            os.fsync(self._fh.fileno())

    def token(self, text: str) -> None:
        if not text:
            return
        t = now()
        if self.t_first is None:
            self.t_first = t
        self.t_last = t
        self.count += 1
        self._write({"t": "tok", "text": text})

    def done(self, stats: dict) -> None:
        self._write({"t": "done", "stats": stats}, sync=True)
        self._fh.close()

    def error(self, message: str) -> None:
        self._write({"t": "error", "error": message}, sync=True)
        self._fh.close()

    def elapsed(self) -> float:
        if self.t_first is None or self.t_last is None:
            return 0.0
        return max(0.0, self.t_last - self.t_first)


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

    def read_since(self, session_id: str, cursor: int = 0) -> dict:
        """Return events appended since *cursor* plus the new cursor and a
        derived status. Never raises on a missing/short file — a poll that beats
        the writer just gets an empty slice."""
        if not valid_session_id(session_id):
            raise ProviderError("invalid session_id")
        path = self._path(session_id)
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
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
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
    elapsed = sink.elapsed()
    tps = None
    ev_count = adapter_stats.get("eval_count")
    ev_dur = adapter_stats.get("eval_duration_s")
    if ev_count and ev_dur:
        tps = round(ev_count / ev_dur, 2)
    elif tokens and elapsed > 0:
        tps = round(tokens / elapsed, 2)
    stats = {"tokens": tokens, "elapsed_s": round(elapsed, 3), "tps": tps}
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
