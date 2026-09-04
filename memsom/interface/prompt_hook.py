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
        "project_bytes": int(params["prompt_hook_project_bytes"]),
        "project_max": int(params["prompt_hook_project_max"]),
    }


# ---------------------------------------------------------------------------
# Query path
# ---------------------------------------------------------------------------

def _bm25_hits(query, k, clearance):
    """In-process BM25-only retrieval. Pins the backend BEFORE importing the
    retrieval stack so nothing can decide to load a model."""
    from memsom import tuning as memsom_tuning
    memsom_tuning.override("embed.backend", "bm25")
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

    # 1. warm endpoint (the long-lived MCP server). Its budget is capped at
    #    warm.WARM_BUDGET_S (~250 ms) INCLUDING the read: a slow or wedged
    #    endpoint is treated as down and we fall back, never wait it out.
    try:
        hits = warm.warm_query(query, k=k, clearance=clearance,
                               deadline_s=t_end - time.monotonic(), db_path=db_path)
        return hits, "warm"
    except warm.WarmUnavailable:
        pass
    # FAILOPEN: allowed, any other warm-path failure falls back to BM25, never crashes.
    except Exception:
        pass

    # 2. in-process BM25, bounded by the remaining deadline via a worker thread
    remaining = t_end - time.monotonic()
    if remaining <= 0:
        return [], "timeout"
    box = {}

    def _run():
        try:
            box["hits"] = _bm25_hits(query, k, clearance)
        # FAILOPEN: allowed, the worker's error is captured for the joining caller to read.
        except Exception as exc:
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

def is_slash(prompt) -> bool:
    """A slash command is never a retrieval query — skip the hook entirely."""
    return (prompt or "").strip().startswith("/")


def too_short_for_bm25(prompt) -> bool:
    """The 12-char floor gates BM25 ONLY (a long query is what BM25 needs); the
    project alias matcher runs on shorter prompts too (a bare "mspanel?")."""
    return len((prompt or "").strip()) < MIN_PROMPT_CHARS


def should_skip(prompt) -> bool:
    """Back-compat: skip when a slash command OR too short for BM25.  The hook
    body no longer calls this (it splits the two so aliases match short prompts);
    kept because external callers / tests import it."""
    return is_slash(prompt) or too_short_for_bm25(prompt)


def _hook_age_days(iso_date: str) -> int:
    try:
        import datetime as _dt
        return (_dt.date.today() - _dt.date.fromisoformat(iso_date[:10])).days
    # FAILOPEN: an unparseable date is not "stale"; 0 keeps the header non-alarming
    except Exception:
        return 0


def _is_stale(status: str, age: int) -> bool:
    from memsom.bridge import project as _project
    limit = _project.STALE_PARKED_DAYS if (status or "").lower() == "parked" \
        else _project.STALE_ACTIVE_DAYS
    return age > limit


def render_project_block(primary, also, cache) -> str:
    """The injected project block: a header + features tally + Status/Creds/Rules
    block + a sub-note pointer per matched project, then an `also:` trailer."""
    projects = cache.get("projects") or {}
    out = []
    for slug in primary:
        meta = projects.get(slug) or {}
        status = meta.get("status", "active")
        lv = meta.get("last_verified", "")
        age = _hook_age_days(lv)
        verif = f"verified {lv} ({age}d)" if lv else "unverified"
        header = f"[memsom project: {slug} | {status} | {verif} | node {meta.get('path', '')}]"
        if lv and _is_stale(status, age):
            header += f"  STALE ({age}d) — confirm before acting"
        out.append(header)
        if meta.get("features"):
            out.append(f"features: {meta['features']}")
        if meta.get("block"):
            out.append(meta["block"])
        out.append(f"sub-notes: memsom project show {slug} "
                   "--note spec|gotchas|decisions|interface_io|architecture|tests")
    if also:
        out.append("also: " + ", ".join(also))
    return "\n".join(out).strip()


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

_WARM_DOWN_SOURCES = frozenset({"bm25", "timeout", "error"})


def store_health(db_path=None):
    """(configured_backend, degraded_lines) for the store, read-only, ~ms.

    Runs BEFORE the query so it sees the backend the store is configured for
    (env or pin), not the bm25 override `_bm25_hits` installs for the process.
    Any failure (no store yet, locked, odd schema) is ('', []): the hook must
    never block a prompt on its own health check."""
    try:
        import memsom
        from memsom.retrieval import embed as memsom_embed
        from memsom.retrieval import retrieve as memsom_retrieve
        conn = memsom.get_connection(db_path, read_only=True)
    # FAILOPEN: allowed -- a missing/locked store is "unknown", never a hook failure.
    except Exception:
        return "", []
    try:
        return memsom_embed.backend(conn), list(memsom_retrieve.retrieval_warnings(conn))
    # FAILOPEN: allowed -- same contract as above.
    except Exception:
        return "", []
    finally:
        conn.close()


def degraded_lines(source: str, configured_backend: str, store_lines) -> list:
    """The one-line warnings Claude reads. Store-level lines (a model split,
    a recent query-encoder fallback) pass through; a warm-endpoint miss adds
    one line ONLY when the store is configured for dense retrieval -- on a
    bm25 store BM25-only is the design, not a degradation."""
    out = list(store_lines or [])
    if source in _WARM_DOWN_SOURCES and configured_backend not in ("", "bm25"):
        if source == "timeout":
            # The endpoint accepted but did not answer inside the hook budget:
            # with cold-start-on-demand (2026-09-04) that is usually the bge
            # supervisor booting its encoder for THIS query, which keeps loading
            # after we gave up — the next prompt is served dense. Say that, not
            # "down", so the reader does not reconnect a healthy server.
            out.append("⚠️ RETRIEVAL DEGRADED: warm endpoint timed out (encoder "
                       "cold-starting or busy) → BM25-only for this prompt; dense "
                       "recall resumes on the next prompt once it is warm")
        else:
            out.append("⚠️ RETRIEVAL DEGRADED: warm endpoint down → BM25-only for this "
                       "prompt (no dense recall; reconnect the memsom MCP server)")
    return out


def run_prompt_hook(data: dict, *, memory_dir=None, params=None, clearance="topsecret",
                    query_fn=None, now=_now_iso, health_fn=None) -> str | None:
    """The whole hook as a function: returns the stdout text to emit (a JSON
    document) or None for silence. Logging happens here in log/inject modes.

    ``query_fn`` is late-bound to ``query_hits`` ON PURPOSE: a def-time default
    freezes the original function object, so a test's patch of the module
    attribute never reaches the CLI path — which made the CLI test silently
    query the LIVE store (green only on a machine whose brain contains the
    fixture stem, red on CI's empty one)."""
    if query_fn is None:
        query_fn = query_hits
    if health_fn is None:
        health_fn = store_health
    prompt = data.get("prompt") if isinstance(data, dict) else None
    # A slash command is never a query — skip entirely. The 12-char floor is NOT
    # applied here anymore: it gates BM25 only, so an alias ("mspanel?") still
    # matches on a short prompt.
    if not isinstance(prompt, str) or is_slash(prompt):
        return None
    if memory_dir is None:
        memory_dir = find_memory_dir()
    if params is None:
        params = load_hook_params(memory_dir)
    mode = params["mode"]
    if mode == "off":
        return None

    # 1. project auto-load — file-only, ~1 ms, fails open to no-project.
    primary, also, project_block = [], [], ""
    try:
        from memsom.bridge import project as _project
        cache = _project.load_cache(memory_dir) if memory_dir is not None else None
        if cache:
            primary, also = _project.match_projects(prompt, cache, params["project_max"])
            if primary:
                project_block = render_project_block(primary, also, cache)
    # FAILOPEN: a missing/corrupt cache or matcher error just means "no project".
    except Exception:
        primary, also, project_block = [], [], ""
    matched_stems = {f"project_{s}" for s in primary}

    # 2. BM25 retrieval — only when the prompt clears the 12-char floor.
    #    The store's health is read FIRST (before any backend override lands)
    #    so the degraded signal reflects what the store is configured for.
    t0 = time.perf_counter()
    warnings = []
    if too_short_for_bm25(prompt):
        hits, source = [], "short"
    else:
        configured, store_lines = health_fn()
        hits, source = query_fn(prompt, k=HOOK_K, clearance=clearance,
                                deadline_ms=params["deadline_ms"])
        warnings = degraded_lines(source, configured, store_lines)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    kept = apply_floor(hits, params["floor"])
    # a retrieval hit that IS a matched project node is redundant with the block;
    # its sub-notes (project_<slug>_*) are not and stay.
    kept = [h for h in kept if (h.get("label") or h.get("stem")) not in matched_stems]
    mem_block = render_block(kept) if kept else ""
    warn_block = "\n".join(warnings)

    # 3. assemble — the project block precedes the retrieval block; a degraded
    #    warning rides last and is injected even when nothing else matched
    #    (that silence is exactly what let the 2026-09 split go unnoticed).
    block = "\n\n".join(x for x in (project_block, mem_block, warn_block) if x)
    would_inject = bool(block)
    injected = would_inject and mode == "inject"

    should_log = memory_dir is not None and (not too_short_for_bm25(prompt) or primary)
    if should_log:
        append_log(memory_dir, {
            "ts": now(),
            "mode": mode,
            "floor": params["floor"],
            "query": prompt.strip()[:LOG_QUERY_CHARS],
            "source": source,
            "ms": elapsed_ms,
            "hits": [{"stem": h.get("label") or h.get("stem"),
                      "score": h.get("score")} for h in hits[:HOOK_K]],
            "projects": primary,
            "project_bytes": len(project_block.encode("utf-8")),
            "would_inject": would_inject,
            "injected": injected,
            "degraded": warnings,
        }, params["log_max_mb"])

    if not injected:
        return None
    return json.dumps(hook_output(block), ensure_ascii=False)


def _read_stdin_json():
    try:
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    # FAILOPEN: allowed, malformed/absent hook JSON degrades to "no context", never crashes the turn.
    except Exception:
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
    # FAILOPEN: allowed, fail silent+open -- the prompt must go through even on a hook bug.
    except Exception as exc:
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
    project_matched = 0
    proj_surfaced = Counter()
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
        projs = r.get("projects") or []
        if projs:
            project_matched += 1
            for s in projs:
                proj_surfaced[str(s)] += 1
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
        "project_matched": project_matched,
        "project_match_rate": round(project_matched / n, 3) if n else 0.0,
        "top_projects": proj_surfaced.most_common(top_n),
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
    print(f"project match  : {summary['project_matched']}  (rate {summary['project_match_rate']})")
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
