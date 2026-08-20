"""prompt_hook — the UserPromptSubmit retrieval hook and its query path.

Three subcommands:

  memsom hook-query <query>   fast retrieval for hooks: warm endpoint first
                              (retrieval/warm.py, served by the running MCP
                              server), in-process BM25 fallback second, and a
                              hard deadline over both — on timeout it prints
                              NOTHING and exits 0, because a stalled hook
                              stalls the whole Claude Code turn.
  memsom hook-prompt          the UserPromptSubmit hook body: reads the hook
                              JSON on stdin, skips short prompts and slash
                              commands, queries k=3, applies the relevance
                              floor and emits the surviving hits as
                              `additionalContext` (<= ~600 bytes).
  memsom hook-stats           summarises the hook log so the floor can be
                              tuned from data.

Modes (`prompt_hook_mode` in `<memory_dir>/.weights/canonical.json` params):
  off     nothing runs, nothing is logged.
  log     every query is logged with its top-3 scores and whether it WOULD
          have injected; nothing is injected.
  inject  log AND inject (the shipped default).
The log is permanent, not a tuning aid: `<memory_dir>/.weights/hook_log.jsonl`,
size-rotated past `prompt_hook_log_max_mb` to hook_log.1.jsonl .. .3.jsonl.

The embedding backend is NEVER cold-loaded from here: the fallback pins
MEMDAG_EMBED_BACKEND=bm25 before the first retrieval import. The warm path
uses whatever backend the MCP server already holds.

Memory dir discovery is `bridge_import.default_memory_dir` — the same walk the
renderer uses ($MEMDAG_BRIDGE_MEMORY_DIR, else the largest
~/.claude/projects/*/memory). No paths are hard-coded.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MIN_PROMPT_CHARS = 12
HOOK_K = 3
MAX_BLOCK_BYTES = 600
MAX_HOOK_CHARS = 90
LOG_NAME = "hook_log.jsonl"
LOG_KEEP = 3
LOG_QUERY_CHARS = 200
HOOK_EVENT = "UserPromptSubmit"


# ---------------------------------------------------------------------------
# Params + memory dir
# ---------------------------------------------------------------------------

def find_memory_dir():
    """The live memory dir, or None when there is none yet (a fresh machine)."""
    from memsom.bridge import bridge_import as bi
    try:
        return Path(bi.default_memory_dir())
    except FileNotFoundError:
        return None


def load_hook_params(memory_dir):
    """The hook's tunables from canonical.json (defaults when absent)."""
    from memsom.lifecycle import forget
    canon = Path(memory_dir) / ".weights" / "canonical.json" if memory_dir else None
    params, _warnings = forget.load_params(canon)
    return {
        "mode": params["prompt_hook_mode"],
        "floor": float(params["prompt_hook_floor"]),
        "deadline_ms": int(params["prompt_hook_deadline_ms"]),
        "log_max_mb": float(params["prompt_hook_log_max_mb"]),
    }


# ---------------------------------------------------------------------------
# Query path
# ---------------------------------------------------------------------------

def _bm25_hits(query, k, clearance):
    """In-process BM25-only retrieval. Pins the backend BEFORE importing the
    retrieval stack so nothing can decide to load a model."""
    os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
    import memsom
    from memsom.retrieval import warm
    conn = memsom.get_connection()
    try:
        return warm.hits_for(conn, query, k=k, clearance=clearance)
    finally:
        conn.close()


_LAST_WORKER = None   # the BM25 worker of the most recent query_hits call


def query_hits(query, k=HOOK_K, clearance="topsecret", deadline_ms=800,
               db_path=None):
    """Returns (hits, source). source in {'warm', 'bm25', 'timeout', 'error'}.
    Never raises; never exceeds the deadline by more than a scheduler tick."""
    from memsom.retrieval import warm
    t_end = time.monotonic() + max(0.05, deadline_ms / 1000.0)

    # 1. warm endpoint (the long-lived MCP server)
    try:
        hits = warm.warm_query(query, k=k, clearance=clearance,
                               deadline_s=t_end - time.monotonic(), db_path=db_path)
        return hits, "warm"
    except socket.timeout:
        return [], "timeout"
    except warm.WarmUnavailable:
        pass
    except Exception:  # noqa: BLE001 — any other failure: fall back, not crash
        pass

    # 2. in-process BM25, bounded by the remaining deadline via a worker thread
    remaining = t_end - time.monotonic()
    if remaining <= 0:
        return [], "timeout"
    box = {}

    def _run():
        try:
            box["hits"] = _bm25_hits(query, k, clearance)
        except Exception as exc:  # noqa: BLE001
            box["error"] = repr(exc)

    global _LAST_WORKER
    th = threading.Thread(target=_run, name="memsom-hook-bm25", daemon=True)
    _LAST_WORKER = th
    th.start()
    th.join(remaining)
    if th.is_alive():
        return [], "timeout"
    if "error" in box:
        return [], "error"
    return box.get("hits", []), "bm25"


# ---------------------------------------------------------------------------
# Block rendering
# ---------------------------------------------------------------------------

def should_skip(prompt) -> bool:
    p = (prompt or "").strip()
    return len(p) < MIN_PROMPT_CHARS or p.startswith("/")


def apply_floor(hits, floor):
    return [h for h in hits if float(h.get("score", 0.0)) >= floor]


def render_block(hits, max_bytes=MAX_BLOCK_BYTES) -> str:
    """`Relevant memories:` + one `- [stem] hook` line per hit, truncated to
    the byte cap on a whole-line boundary (a half line is worse than none)."""
    lines = ["Relevant memories:"]
    for h in hits:
        label = h.get("label") or h.get("stem") or f"mem:{h.get('id')}"
        hook = " ".join((h.get("hook") or "").split())
        if len(hook) > MAX_HOOK_CHARS:
            hook = hook[:MAX_HOOK_CHARS - 1].rstrip() + "…"
        line = f"- [{label}] {hook}" if hook else f"- [{label}]"
        candidate = "\n".join(lines + [line])
        if len(candidate.encode("utf-8")) > max_bytes:
            break
        lines.append(line)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def hook_output(block: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": HOOK_EVENT,
                                   "additionalContext": block}}


# ---------------------------------------------------------------------------
# Log + rotation
# ---------------------------------------------------------------------------

def log_path(memory_dir) -> Path:
    return Path(memory_dir) / ".weights" / LOG_NAME


def rotated_paths(memory_dir):
    """Current log first, then .1 .. .N (only those that exist)."""
    base = log_path(memory_dir)
    out = [base] if base.exists() else []
    for i in range(1, LOG_KEEP + 1):
        p = base.with_name(f"hook_log.{i}.jsonl")
        if p.exists():
            out.append(p)
    return out


def rotate_if_needed(memory_dir, max_mb) -> bool:
    """hook_log.jsonl -> .1 -> .2 -> .3 (dropped) once the live file exceeds
    *max_mb*. Returns True when a rotation happened."""
    base = log_path(memory_dir)
    try:
        size = base.stat().st_size
    except OSError:
        return False
    if size <= max_mb * 1024 * 1024:
        return False
    oldest = base.with_name(f"hook_log.{LOG_KEEP}.jsonl")
    if oldest.exists():
        oldest.unlink()
    for i in range(LOG_KEEP - 1, 0, -1):
        src = base.with_name(f"hook_log.{i}.jsonl")
        if src.exists():
            src.replace(base.with_name(f"hook_log.{i + 1}.jsonl"))
    base.replace(base.with_name("hook_log.1.jsonl"))
    return True


def append_log(memory_dir, record, max_mb) -> None:
    """Append one JSON line; rotate first if the file is over the cap.
    Best-effort: a log failure must never block a prompt."""
    try:
        rotate_if_needed(memory_dir, max_mb)
        p = log_path(memory_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# hook-prompt (pure core + CLI shell)
# ---------------------------------------------------------------------------

def run_prompt_hook(data: dict, *, memory_dir=None, params=None, clearance="topsecret",
                    query_fn=query_hits, now=_now_iso) -> str | None:
    """The whole hook as a function: returns the stdout text to emit (a JSON
    document) or None for silence. Logging happens here in log/inject modes."""
    prompt = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(prompt, str) or should_skip(prompt):
        return None
    if memory_dir is None:
        memory_dir = find_memory_dir()
    if params is None:
        params = load_hook_params(memory_dir)
    mode = params["mode"]
    if mode == "off":
        return None

    t0 = time.perf_counter()
    hits, source = query_fn(prompt, k=HOOK_K, clearance=clearance,
                            deadline_ms=params["deadline_ms"])
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    kept = apply_floor(hits, params["floor"])
    block = render_block(kept) if kept else ""
    would_inject = bool(block)
    injected = would_inject and mode == "inject"

    if memory_dir is not None:
        append_log(memory_dir, {
            "ts": now(),
            "mode": mode,
            "floor": params["floor"],
            "query": prompt.strip()[:LOG_QUERY_CHARS],
            "source": source,
            "ms": elapsed_ms,
            "hits": [{"stem": h.get("label") or h.get("stem"),
                      "score": h.get("score")} for h in hits[:HOOK_K]],
            "would_inject": would_inject,
            "injected": injected,
        }, params["log_max_mb"])

    if not injected:
        return None
    return json.dumps(hook_output(block), ensure_ascii=False)


def _read_stdin_json():
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:  # noqa: BLE001
        return {}


def _exit_now_if_worker_stuck():
    """After a fallback timeout THIS call's BM25 worker may still be inside
    sqlite. Do not wait for it: flush and hard-exit so the hook returns on
    time. Only the worker started by the current command counts."""
    th = _LAST_WORKER
    if th is not None and th.is_alive():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(0)


def _reset_worker():
    global _LAST_WORKER
    _LAST_WORKER = None


def cmd_hook_prompt(args):
    _reset_worker()
    try:
        out = run_prompt_hook(_read_stdin_json(), clearance=args.clearance)
    except Exception as exc:  # noqa: BLE001 — fail silent+open: the prompt must go through
        print(f"[memsom-hook-prompt] error (no context injected): {exc!r}", file=sys.stderr)
        _exit_now_if_worker_stuck()
        return 0
    if out:
        sys.stdout.write(out + "\n")
        sys.stdout.flush()
    _exit_now_if_worker_stuck()
    return 0


# ---------------------------------------------------------------------------
# hook-query CLI
# ---------------------------------------------------------------------------

def cmd_hook_query(args):
    _reset_worker()
    hits, source = query_hits(args.query, k=args.k, clearance=args.clearance,
                              deadline_ms=args.deadline_ms)
    if source == "timeout":
        _exit_now_if_worker_stuck()
        return 0                      # the contract: nothing on stdout, exit 0
    sys.stdout.write(json.dumps({"source": source, "hits": hits}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# hook-stats
# ---------------------------------------------------------------------------

def iter_log_records(memory_dir):
    for p in rotated_paths(memory_dir):
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        yield rec
        except OSError:
            continue


def summarize_log(records, *, top_n=10, bins=10) -> dict:
    """Counts over the log: queries, inject/would-inject rates, surfaced
    stems, a histogram of top-1 scores and the would-inject rate each floor
    candidate would produce (so a floor can be picked from data)."""
    n = 0
    injected = 0
    would = 0
    by_mode = Counter()
    by_source = Counter()
    surfaced = Counter()
    top1 = []
    ms = []
    for r in records:
        n += 1
        by_mode[str(r.get("mode"))] += 1
        by_source[str(r.get("source"))] += 1
        if r.get("injected"):
            injected += 1
        if r.get("would_inject"):
            would += 1
        hits = r.get("hits") or []
        if hits and isinstance(hits[0], dict):
            s = hits[0].get("score")
            if isinstance(s, (int, float)):
                top1.append(float(s))
        if r.get("would_inject"):
            for h in hits:
                if isinstance(h, dict) and h.get("stem"):
                    surfaced[str(h["stem"])] += 1
        if isinstance(r.get("ms"), (int, float)):
            ms.append(float(r["ms"]))
    hist = [0] * bins
    for s in top1:
        idx = min(bins - 1, max(0, int(s * bins)))
        hist[idx] += 1
    floor_sweep = []
    for i in range(0, bins + 1):
        f = i / bins
        k = sum(1 for s in top1 if s >= f)
        floor_sweep.append({"floor": round(f, 2), "would_inject_rate":
                            round(k / n, 3) if n else 0.0})
    ms_sorted = sorted(ms)
    p50 = ms_sorted[len(ms_sorted) // 2] if ms_sorted else None
    p95 = ms_sorted[int(len(ms_sorted) * 0.95)] if ms_sorted else None
    return {
        "queries": n,
        "injected": injected,
        "inject_rate": round(injected / n, 3) if n else 0.0,
        "would_inject": would,
        "would_inject_rate": round(would / n, 3) if n else 0.0,
        "by_mode": dict(by_mode),
        "by_source": dict(by_source),
        "top_stems": surfaced.most_common(top_n),
        "top1_histogram": [{"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": hist[i]}
                           for i in range(bins)],
        "floor_sweep": floor_sweep,
        "ms_p50": p50,
        "ms_p95": p95,
    }


def cmd_hook_stats(args):
    memory_dir = Path(args.memory_dir) if args.memory_dir else find_memory_dir()
    if memory_dir is None:
        print("[hook-stats] no memory dir found", file=sys.stderr)
        return 1
    summary = summarize_log(iter_log_records(memory_dir), top_n=args.top)
    summary["log_files"] = [str(p) for p in rotated_paths(memory_dir)]
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    print(f"hook log: {memory_dir / '.weights' / LOG_NAME}  "
          f"({len(summary['log_files'])} file(s))")
    print(f"queries        : {summary['queries']}")
    print(f"injected       : {summary['injected']}  (rate {summary['inject_rate']})")
    print(f"would inject   : {summary['would_inject']}  (rate {summary['would_inject_rate']})")
    print(f"by mode        : {summary['by_mode']}")
    print(f"by source      : {summary['by_source']}")
    if summary["ms_p50"] is not None:
        print(f"latency ms     : p50 {summary['ms_p50']}  p95 {summary['ms_p95']}")
    print("top surfaced stems:")
    for stem, c in summary["top_stems"]:
        print(f"  {c:5d}  {stem}")
    print("top-1 score histogram:")
    for row in summary["top1_histogram"]:
        print(f"  {row['bin']}  {'#' * min(row['n'], 60)}{' ' if row['n'] else ''}{row['n']}")
    print("would-inject rate by floor:")
    print("  " + "  ".join(f"{r['floor']:.1f}:{r['would_inject_rate']:.2f}"
                           for r in summary["floor_sweep"]))
    return 0


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

def register(sub) -> None:
    q = sub.add_parser("hook-query",
                       help="fast retrieval for hooks: warm endpoint, BM25 fallback, hard deadline")
    q.add_argument("query")
    q.add_argument("--k", type=int, default=HOOK_K)
    q.add_argument("--clearance", default="topsecret")
    q.add_argument("--deadline-ms", type=int, default=800,
                   help="print nothing and exit 0 past this (default 800)")
    q.set_defaults(func=cmd_hook_query)

    p = sub.add_parser("hook-prompt",
                       help="UserPromptSubmit hook: surface top memories as added context")
    p.add_argument("--clearance", default="topsecret")
    p.set_defaults(func=cmd_hook_prompt)

    s = sub.add_parser("hook-stats", help="summarise the prompt-hook log")
    s.add_argument("--memory-dir", default=None)
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_hook_stats)
