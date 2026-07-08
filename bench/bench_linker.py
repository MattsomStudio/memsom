"""bench_linker — the blind LLM link-writer for the graph-arm head-to-head.

The entire integrity claim of the graph benchmark rests on ONE property, and it
is enforced structurally, not by promise: this module never sees the gold. Its
public entrypoint `link()` takes ONLY the memory texts — not the question, not
the `answer_bearing` flags, not the scorer. That signature *is* the anti-cheat
proof: no label can reach the linker because no label is an argument.

It replicates the real memory pipeline's linking step — an LLM reads the notes
and associatively links the ones that are topically related, exactly as it adds
[[wikilinks]] between related memories in Matthew's brain. Here it returns index
pairs instead of writing [[wikilinks]], but the decision is made the same way
and from the same information (text only).

Deterministic: temperature 0 + a content-hash disk cache, so the edge set for a
given text list is stable across runs. Arm-to-arm deltas are signal, not LLM
sampling noise. Delete the cache dir to force a re-link.

Config (env):
  BENCH_LINKER_BACKEND  "openai" | "ollama" (default: openai if OPENAI_API_KEY set,
                        else ollama)
  BENCH_LINKER_MODEL    model id (default gpt-4o for openai, qwen2.5:7b-instruct
                        for ollama)
  BENCH_LINKER_URL      Ollama chat endpoint (default http://localhost:11434/api/chat)
  BENCH_LINKER_CACHE    cache dir (default bench/.linker_cache)

Both backends are gold-blind — the backend only changes WHICH model reads the
texts, never WHAT it sees. The cache key includes the model id, so switching
backends re-links rather than serving another model's edges.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BACKEND = os.environ.get(
    "BENCH_LINKER_BACKEND", "openai" if os.environ.get("OPENAI_API_KEY") else "ollama")
_DEFAULT_MODEL = "gpt-4o" if BACKEND == "openai" else "qwen2.5:7b-instruct"
MODEL = os.environ.get("BENCH_LINKER_MODEL", _DEFAULT_MODEL)
OLLAMA_URL = os.environ.get("BENCH_LINKER_URL", "http://localhost:11434/api/chat")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
CACHE_DIR = Path(os.environ.get(
    "BENCH_LINKER_CACHE", Path(__file__).resolve().parent / ".linker_cache"))

# updated per OpenAI call so a caller (precache/ETA) can read token usage for
# cost estimation; {} until the first live openai call in this process.
LAST_USAGE: dict = {}

_PROMPT = (
    "You are the associative-linking step of a memory system. Below are numbered "
    "memory notes from one person's history. Link ONLY the strongest associations: "
    "for each note, connect it to AT MOST the 2-3 other notes most likely to be "
    "needed together with it when recalling something. MOST PAIRS SHOULD NOT BE "
    "LINKED. Do not link notes merely because they concern the same person or are "
    "loosely on-topic — a link means 'recalling one should specifically pull up the "
    "other'. When unsure, leave it out: a missing weak link costs far less than a "
    "spurious strong one. Judge ONLY by the note contents.\n\n"
    "Return ONLY a JSON array of index pairs, e.g. [[0,3],[1,2]]. No prose, no keys.\n\n"
)


def _cache_key(texts: list[str]) -> str:
    # Key includes MODEL and the PROMPT so tuning either re-links instead of
    # silently serving stale edges from an earlier prompt/model.
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"\x00PROMPT\x00")
    h.update(_PROMPT.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _valid_pairs(raw, n: int) -> list[tuple[int, int]]:
    """Keep only in-range, non-self, de-duped integer pairs."""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in raw or []:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        i, j = p
        if not (isinstance(i, int) and isinstance(j, int)):
            continue
        if not (0 <= i < n and 0 <= j < n) or i == j:
            continue
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            out.append((i, j))
    return out


def _extract(content: str, n: int) -> list[tuple[int, int]]:
    """Pull index pairs from a model response — list, wrapped object, or loose text."""
    raw = None
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            raw = obj
        elif isinstance(obj, dict):
            for v in obj.values():          # model wrapped it, e.g. {"pairs": [...]}
                if isinstance(v, list):
                    raw = v
                    break
    except (ValueError, TypeError):
        pass
    if raw is None:                          # last resort: scrape [i, j] literals
        raw = [[int(a), int(b)]
               for a, b in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", content)]
    return _valid_pairs(raw, n)


def _call_ollama(prompt: str) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:   # raises on connect error
        return json.loads(r.read())["message"]["content"]


def _call_openai(prompt: str) -> str:
    key = os.environ["OPENAI_API_KEY"]                    # KeyError if unset -> loud
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    # Retry with backoff on 429 / transient 5xx — OpenAI rate limits are per-tier
    # and a concurrent precache trips them. Respect Retry-After when present,
    # otherwise exponential (capped). Give up loudly after the last attempt so a
    # persistent failure never masquerades as a null edge set.
    delay = 2.0
    for attempt in range(8):
        try:
            req = urllib.request.Request(OPENAI_URL, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else delay
                time.sleep(wait)
                delay = min(delay * 2, 90)
                continue
            raise
    LAST_USAGE.clear()
    LAST_USAGE.update(resp.get("usage", {}))
    return resp["choices"][0]["message"]["content"]


def link(texts: list[str]) -> list[tuple[int, int]]:
    """Return associative index pairs among `texts`. Gold-blind by signature.

    Raises on a transport error (backend down / missing key) rather than
    returning [] — a silent empty edge set would make the graph arm secretly
    identical to the retrieve arm and produce a fake null. Fail loud instead.
    Preflight the backend before a run.
    """
    n = len(texts)
    if n < 2:
        return []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{_cache_key(texts)}.json"
    if cache.exists():
        return [tuple(p) for p in json.loads(cache.read_text(encoding="utf-8"))]

    prompt = _PROMPT + "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    content = _call_openai(prompt) if BACKEND == "openai" else _call_ollama(prompt)
    pairs = _extract(content, n)
    cache.write_text(json.dumps(pairs), encoding="utf-8")
    return pairs
