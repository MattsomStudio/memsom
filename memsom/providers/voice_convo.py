"""Durable multi-turn memory for the voice brain.

The voice tab used to be amnesiac: every ``POST /api/voice/chat`` was a fresh
single-turn generation, so "what GPU do I have?" then "how much VRAM does it
have?" failed — turn 2 had no idea what "it" was. This module threads a
conversation across turns.

Storage — ONE JSON file per conversation at
``<voice_dir>/conversations/<conversation_id>.json``::

    {"conversation_id": "..", "created": "..iso..", "updated": "..iso..",
     "turns": [{"role": "user", "content": ".."},
               {"role": "assistant", "content": ".."}, ...]}

Only the USER text and the FINAL ASSISTANT text are stored — NEVER the
tool-call/tool-result transcript. History stays lean; the model re-calls tools
(recall/web/file) as needed each turn.

Windowing (context/cost guard): the file keeps the FULL history forever, but
only the tail is re-sent to the model each turn. :func:`load_window` caps the
threaded context to the last :data:`WINDOW_TURNS` messages AND ~:data:`WINDOW_TOKENS`
tokens (whichever bites first). Oldest turns drop from the *sent context*, not
from disk.

Writes are atomic (temp-file + ``os.replace``), matching the other durable
files in this package (agent_store, schedule, procman).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

# conversation ids become filenames — fence hard against path traversal, same
# rule the session ids follow.
_CONVO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Sliding-window caps for the THREADED context (not the stored file):
#   * WINDOW_TURNS  — keep at most the last N messages (a "turn" = one
#     {role, content}; a user+assistant exchange is 2). 12 ≈ 6 exchanges.
#   * WINDOW_TOKENS — and if that tail still exceeds this rough budget, drop
#     from the front until it fits. Estimated at ~4 chars/token (no tokenizer
#     dependency — a deliberate over-estimate keeps us safely under real cost).
WINDOW_TURNS = 12
WINDOW_TOKENS = 4000


def new_conversation_id() -> str:
    return uuid.uuid4().hex


def valid_conversation_id(cid) -> bool:
    return bool(cid) and isinstance(cid, str) and bool(_CONVO_ID_RE.match(cid))


def _path(convo_dir: Path, cid: str) -> Path:
    return Path(convo_dir) / f"{cid}.json"


def load_turns(convo_dir, cid: str) -> list:
    """The FULL stored turn list for a conversation ([] if none yet).

    Never raises on a missing or corrupt file — a first turn, or a partially
    written file that lost the race, simply reads as empty history."""
    p = _path(convo_dir, cid)
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    turns = doc.get("turns") if isinstance(doc, dict) else None
    return turns if isinstance(turns, list) else []


def _approx_tokens(turns: list) -> int:
    return sum(len(str(t.get("content", ""))) for t in turns) // 4


def load_window(convo_dir, cid: str, *, max_turns: int = WINDOW_TURNS,
                max_tokens: int = WINDOW_TOKENS) -> list:
    """The windowed TAIL of history to thread into the next turn's context.

    Take the last *max_turns* messages, then trim from the front until the rough
    token estimate fits *max_tokens*. Returns a fresh list of {role, content}
    dicts safe to splice into the messages array."""
    turns = load_turns(convo_dir, cid)
    windowed = [{"role": t.get("role"), "content": t.get("content", "")}
                for t in turns[-max_turns:]
                if isinstance(t, dict) and t.get("role") in ("user", "assistant")]
    while len(windowed) > 1 and _approx_tokens(windowed) > max_tokens:
        windowed = windowed[1:]
    return windowed


def append_turn(convo_dir, cid: str, user_text: str, assistant_text: str) -> None:
    """Append the {user, final-assistant} exchange and persist atomically.

    Load-modify-write: read the current full history, append the two messages,
    write back via temp-file + os.replace. Single-writer-per-conversation in
    practice (one utterance generates at a time), so no lock is needed."""
    convo_dir = Path(convo_dir)
    convo_dir.mkdir(parents=True, exist_ok=True)
    p = _path(convo_dir, cid)

    doc = None
    if p.is_file():
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            doc = None
    if not isinstance(doc, dict) or not isinstance(doc.get("turns"), list):
        doc = {"conversation_id": cid, "created": _now_iso(), "turns": []}

    doc["turns"].append({"role": "user", "content": user_text})
    doc["turns"].append({"role": "assistant", "content": assistant_text})
    doc["updated"] = _now_iso()

    tmp = p.with_suffix(f".tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FinalTextCapture:
    """A pass-through sink wrapper that remembers the LAST turn's streamed text.

    ``run_tool_loop`` streams every token to the sink and emits a ``turn`` event
    at the start of each turn, but returns only stats — not the spoken answer we
    need to persist. This wrapper forwards everything unchanged to the real
    :class:`AgentFileSink` (so the cursor-poll frontend is untouched), while
    resetting its own buffer on each ``turn`` event and accumulating tokens.
    When the loop ends, :attr:`final_text` holds the text of the final turn —
    the tool-decision preambles of earlier turns were reset away, leaving only
    the answer the model actually spoke."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._buf: list = []

    def token(self, text: str) -> None:
        if text:
            self._buf.append(text)
        self._inner.token(text)

    def event(self, obj: dict, sync: bool = False) -> None:
        if isinstance(obj, dict) and obj.get("t") == "turn":
            self._buf = []  # new turn — only the LAST turn's text is the answer
        self._inner.event(obj, sync=sync)

    def __getattr__(self, name):
        # delegate done/error/elapsed/etc. to the wrapped sink
        return getattr(self._inner, name)

    @property
    def final_text(self) -> str:
        return "".join(self._buf).strip()
