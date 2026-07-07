"""memsom_contradict — the THIRD staleness trigger: cross-source CONTRADICTION.

Supersession (same source_ref re-ingest, memsom_stale.on_reingest_supersede) and
the verification-age sweep (memsom_verify_stale) both need a LINK to fire; neither
sees a NEW, differently-sourced fact that contradicts a live one. So "Sucuri is the
active WAF" and a later, separately-sourced "Cloudflare is the active WAF" both sit
live and both get cited. This detector fills that gap: on ingest it finds the live
nodes closest to the new claim and, when one is judged to contradict it, marks the
OLD node stale and records the link.

Two adjudicator tiers (pluggable):
  - Phase 1 (this file): STRUCTURED — memsom_corroborate.extract_claim on both;
    contradiction iff same (subject, predicate) with a different value. extract_claim
    subjects are type-scoped (a port claim is ("port","is",N)), so this is only
    precise BECAUSE candidates are already topically scoped by the retrieval gate —
    two "port" claims are compared only when their nodes are close, i.e. about the
    same thing. No model, deterministic.
  - Phase 2 (this file): NLI — a small in-process cross-encoder scores
    P(contradiction) on candidates the structured tier didn't resolve. Opt-in via
    $MEMDAG_CONTRADICT_NLI (separate from the detector's own $MEMDAG_CONTRADICT so
    the cheap structured tier can run without the model); heavy-dependency-optional,
    degrades to structured-only when torch/transformers/weights are absent. Callers
    may still inject nli=fn (tests do) to override the default scorer.

Design guardrails:
  - NEVER reuses source_supersedes — freshen()/substitute_fresh() treat that as a
    value-preserving replacement and would silently serve the contradicting node as
    the corrected value. The link lives in its own `contradictions` table.
  - The stale reason is namespaced "contradicted by node N" so other passes'
    _owned()-style clear checks (e.g. verify_stale) never clear a contradiction flag.
  - Marks the OLDER node (newer wins); the new node is never staled.
  - OBSERVE-ONLY by default: the user-facing surfaces (sweep, ingest hook) RECORD a
    contradiction to the table (enforced=0) but do NOT stale anything unless
    $MEMDAG_CONTRADICT_ENFORCE opts in. A precision regression can therefore never
    pollute the brain by default; enforcement is a deliberate per-run choice made
    only after the eval gate is green. (Learned the hard way: an early enforce-by-
    default backfill flagged 110/155 real nodes.)
  - Opt-in: the detector only runs at all when $MEMDAG_CONTRADICT is truthy — keeps
    a cold bridge-import from checking every node before the detector is tuned.

Public API
----------
migrate(conn)                                   idempotent
enabled()                              -> bool  ($MEMDAG_CONTRADICT gate)
detect(conn, new_id, *, k, candidates, nli, nli_threshold, clearance) -> list[(old_id, verdict)]
list_contradictions(conn)              -> list[dict]
register(subparsers) / main(argv)               CLI: contradictions-list
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import memsom
import memsom_schema

REASON_PREFIX = "contradicted by"
_TRUTHY = ("1", "true", "yes", "on")

# NLI semantic tier (Phase 2) — heavy-dependency-optional, mirroring memsom_embed:
# torch + transformers are imported LAZILY only inside the code path, so importing
# this module stays free and memsom keeps its "no required model" property. When the
# libs or weights are absent the tier returns None and the detector runs structured-
# only. Contradiction detection is literally the NLI task (premise/hypothesis ->
# entailment|neutral|CONTRADICTION), so this is a small cross-encoder, NOT an LLM.
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
_NLI = None            # (tokenizer, model, contradiction_idx) process-global, lazy
_NLI_AVAILABLE = None  # cached tri-state import probe
_NLI_WARNED = False


def _nli_model_name():
    return os.environ.get("MEMDAG_CONTRADICT_NLI_MODEL") or DEFAULT_NLI_MODEL


def _nli_threshold():
    """Contradiction-probability cutoff. High by default (precision > recall)."""
    try:
        return float(os.environ.get("MEMDAG_CONTRADICT_NLI_THRESHOLD", "0.85"))
    except ValueError:
        return 0.85


def nli_available():
    """True iff torch + transformers import cleanly. Cached; never raises. Probes
    imports ONLY — no model download / VRAM (that happens on first _load_nli())."""
    global _NLI_AVAILABLE
    if _NLI_AVAILABLE is None:
        try:
            import torch  # noqa: F401,PLC0415
            import transformers  # noqa: F401,PLC0415
            _NLI_AVAILABLE = True
        except Exception:  # noqa: BLE001 — any import/DLL failure -> tier unavailable
            _NLI_AVAILABLE = False
    return _NLI_AVAILABLE


def _load_nli():
    """Lazy-load the NLI cross-encoder once. Reads config.id2label to locate the
    'contradiction' class robustly (models differ in label order)."""
    global _NLI
    if _NLI is None:
        from transformers import (AutoTokenizer,  # noqa: PLC0415
                                  AutoModelForSequenceClassification)
        name = _nli_model_name()
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()
        id2label = model.config.id2label or {}
        contra_idx = next((int(i) for i, lab in id2label.items()
                           if "contradiction" in str(lab).lower()), 0)
        _NLI = (tok, model, contra_idx)
    return _NLI


def _warn_nli_fallback():
    global _NLI_WARNED
    if not _NLI_WARNED:
        _NLI_WARNED = True
        import sys  # noqa: PLC0415
        print("[memsom-contradict] NLI tier requested but the model failed to load; "
              "running structured-only. Check torch/transformers + the model name.",
              file=sys.stderr)


def nli_score(premise, hypothesis):
    """P(*hypothesis* contradicts *premise*) in [0,1], or None if the tier is
    unavailable. Truncates to the model's 512-token window."""
    if not nli_available():
        return None
    try:
        import torch  # noqa: PLC0415
        tok, model, ci = _load_nli()
        with torch.no_grad():
            enc = tok(premise or "", hypothesis or "", truncation=True,
                      max_length=512, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits[0], dim=-1)
            return float(probs[ci])
    except Exception:  # noqa: BLE001 — a scorer failure must never break ingest
        _warn_nli_fallback()
        return None


def _default_nli():
    """The scorer detect() uses when a caller doesn't inject one: the real NLI
    scorer iff the semantic tier is opted in ($MEMDAG_CONTRADICT_NLI) AND loadable;
    otherwise None (structured-only). Kept separate from $MEMDAG_CONTRADICT so a
    user can run the cheap structured tier without the model."""
    if str(os.environ.get("MEMDAG_CONTRADICT_NLI", "")).strip().lower() not in _TRUTHY:
        return None
    if not nli_available():
        return None
    return nli_score


def migrate(conn):
    """Create the contradictions link table (+ index). Idempotent."""
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS contradictions (
    old_id   INTEGER NOT NULL REFERENCES nodes(id),   -- the contradicted (now-stale) node
    new_id   INTEGER NOT NULL REFERENCES nodes(id),   -- the node that contradicted it
    verdict  TEXT NOT NULL,                            -- 'structured' | 'nli'
    judge    TEXT,                                     -- adjudicator label / model name
    score    REAL,                                     -- NLI contradiction prob (NULL for structured)
    reason   TEXT,
    at       TEXT NOT NULL,
    enforced INTEGER NOT NULL DEFAULT 1,               -- 1 = staled the old node; 0 = observe-only
    PRIMARY KEY (old_id, new_id)
  );""")
    # additive for a store created before observe-only shipped (legacy rows enforced)
    memsom_schema.add_column(conn, "contradictions", "enforced", "INTEGER NOT NULL DEFAULT 1")
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS contradict_cursor (
    id           INTEGER PRIMARY KEY CHECK (id = 0),   -- single-row watermark
    last_node_id INTEGER NOT NULL DEFAULT 0,
    swept_at     TEXT
  );""")
    with conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_new "
                     "ON contradictions(new_id)")


def enabled():
    """True iff $MEMDAG_CONTRADICT opts the detector in."""
    return str(os.environ.get("MEMDAG_CONTRADICT", "")).strip().lower() in _TRUTHY


def _enforce_default():
    """Default enforcement for the user-facing surfaces (sweep, ingest hook).
    OBSERVE-ONLY unless $MEMDAG_CONTRADICT_ENFORCE opts in — so a precision
    regression records to the table without ever staling the brain. Enforcement is
    a deliberate choice, not a default."""
    return str(os.environ.get("MEMDAG_CONTRADICT_ENFORCE", "")).strip().lower() in _TRUTHY


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _structured_verdict(new_text, cand_text):
    """Deterministic tier: same (subject, predicate) with a different value.

    Returns (kind, reason, score) or None. Lazy-imports corroborate to avoid an
    import cycle and to keep this file loadable without the corroborate stack."""
    import memsom_corroborate  # noqa: PLC0415
    a = memsom_corroborate.extract_claim(new_text or "")
    b = memsom_corroborate.extract_claim(cand_text or "")
    if not a or not b:
        return None
    if a[0] == b[0] and a[1] == b[1] and a[2] != b[2]:
        return ("structured", f"{a[0]} {a[1]} {b[2]!r} contradicts {a[2]!r}", None)
    return None


def detect(conn, new_id, *, k=5, candidates=None, nli=None, nli_threshold=None,
           clearance="topsecret", enforce=True):
    """Detect live nodes that *new_id* contradicts; record each link, and (when
    *enforce*) mark the older node stale.

    candidates: optional [(id, content), ...] to adjudicate against (injected in
    tests). When None, live topically-close candidates come from the hybrid
    retriever (its BM25 half works with no embedder, so this degrades gracefully).
    nli: optional fn(premise, hypothesis) -> P(contradiction) in [0,1]; used only
    where the structured tier abstains. When None, resolves to _default_nli() —
    the real NLI scorer iff the semantic tier is opted in + loadable, else None.
    enforce: True stales the older node; False is OBSERVE-ONLY — the contradiction
    is recorded (enforced=0) for review but nothing is staled. The user-facing
    surfaces (sweep, ingest hook) default to observe via _enforce_default(); this
    primitive defaults enforce=True so explicit callers get deterministic marking.

    Returns a sorted list of (old_id, verdict_kind) for every DETECTION recorded
    (regardless of enforce). Marks the OLD node, never new_id.
    """
    migrate(conn)
    if nli is None:
        nli = _default_nli()
    if nli_threshold is None:
        nli_threshold = _nli_threshold()
    row = conn.execute(
        "SELECT content, channel FROM nodes WHERE id = ? AND tombstoned = 0",
        (new_id,)).fetchone()
    if row is None:
        return []
    new_text, channel = row
    if not new_text or channel == "agent-derived":
        return []

    if candidates is None:
        try:
            import memsom_retrieve  # noqa: PLC0415
            hits = memsom_retrieve.retrieve(conn, new_text, k=k, clearance=clearance)
            candidates = [(h[0], h[1]) for h in hits]
        except Exception:  # noqa: BLE001 — no retriever/embedder -> nothing to compare
            return []

    marked = []
    ts = _now()
    import memsom_stale  # noqa: PLC0415
    for cid, ctext in candidates:
        if cid == new_id:
            continue
        verdict = _structured_verdict(new_text, ctext)
        if verdict is None and nli is not None:
            try:
                p = nli(ctext, new_text)  # premise=candidate, hypothesis=new claim
            except Exception:  # noqa: BLE001 — a judge failure must not break ingest
                p = None
            if p is not None and p >= nli_threshold:
                verdict = ("nli", f"nli p={p:.2f}", float(p))
        if verdict is None:
            continue
        # The OLDER node loses (newer fact wins), whichever way we're probing. At
        # ingest the probe (new_id) is newest so the candidate loses — same result.
        # But a backfill sweep probes both nodes of a pair; keying on (older, newer)
        # makes both directions converge to ONE record + ONE stale mark instead of
        # cross-flagging both nodes. Node id is monotonic with insertion order.
        old_id, win_id = (cid, new_id) if cid < new_id else (new_id, cid)
        if conn.execute("SELECT 1 FROM contradictions WHERE old_id=? AND new_id=?",
                        (old_id, win_id)).fetchone():
            continue
        kind, reason, score = verdict
        if enforce:
            memsom_stale.mark_stale_cascade(conn, old_id, f"{REASON_PREFIX} node {win_id}")
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO contradictions"
                "(old_id, new_id, verdict, judge, score, reason, at, enforced) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (old_id, win_id, kind, kind, score, reason, ts, 1 if enforce else 0))
        marked.append((old_id, kind))
    return sorted(marked)


def list_contradictions(conn, *, observed_only=False):
    """All recorded contradictions, newest first. observed_only filters to the
    record-but-not-staled rows (enforced=0)."""
    migrate(conn)
    where = " WHERE enforced = 0" if observed_only else ""
    rows = conn.execute(
        "SELECT old_id, new_id, verdict, score, reason, at, enforced "
        f"FROM contradictions{where} ORDER BY at DESC, new_id DESC").fetchall()
    return [{"old_id": r[0], "new_id": r[1], "verdict": r[2], "score": r[3],
             "reason": r[4], "at": r[5], "enforced": bool(r[6])} for r in rows]


# --- batch sweep (bridge/flat-file coverage) ---------------------------------
# The ingest hook only fires on ingest_text (vault sync, MCP, external). Flat-file
# bridge memories go through insert_node and bypass it. This sweep covers them
# WITHOUT loading the NLI model on every Stop-hook render: it runs as an explicit
# CLI/scheduled pass, so the model loads once for the whole run.

def _cursor(conn):
    migrate(conn)
    r = conn.execute("SELECT last_node_id FROM contradict_cursor WHERE id=0").fetchone()
    return r[0] if r else 0


def _set_cursor(conn, nid):
    with conn:
        conn.execute(
            "INSERT INTO contradict_cursor(id, last_node_id, swept_at) VALUES (0,?,?) "
            "ON CONFLICT(id) DO UPDATE SET last_node_id=excluded.last_node_id, "
            "swept_at=excluded.swept_at",
            (nid, _now()))


def _embed_candidate_fn(conn, k):
    """Build a candidate selector over live memory nodes using an IN-MEMORY embedding
    index (bge-m3 loads once), so the sweep covers flat-file bridge memories that are
    NOT in the persistent retrieval index (they're delivered via MEMORY.md, never
    indexed). Returns candidate_fn(probe_id, probe_text) -> [(cid, ctext), ...] or
    None when the embedder is unavailable (caller falls back to retrieve)."""
    try:
        import memsom_embed  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    if not memsom_embed.bge_available():
        return None
    rows = conn.execute(
        "SELECT id, content FROM nodes WHERE tombstoned = 0 "
        "AND channel != 'agent-derived'").fetchall()
    ids, texts, vecs = [], [], []
    for nid, content in rows:
        enc = memsom_embed.encode_doc(content or "")
        if enc and enc.get("dense") is not None:
            ids.append(nid)
            texts.append(content)
            vecs.append(enc["dense"])
    if not vecs:
        return None
    mat = np.asarray(vecs, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms                                   # cosine == dot on unit rows
    pos = {nid: i for i, nid in enumerate(ids)}

    def candidate_fn(pid, _ptext):
        i = pos.get(pid)
        if i is None:
            return []
        sims = mat @ mat[i]
        out = []
        for j in np.argsort(-sims):
            j = int(j)
            if ids[j] == pid:
                continue
            out.append((ids[j], texts[j]))
            if len(out) >= k:
                break
        return out
    return candidate_fn


def sweep(conn, *, limit=None, use_nli=True, k=5, clearance="topsecret",
          backfill=False, nli=None, candidate_fn=None, enforce=None):
    """Probe memory nodes added since the last sweep for cross-source contradictions.

    Incremental by default (id > watermark); backfill=True re-scans from 0 (one-time
    full pass — the first run on an existing store). Candidates come from an
    in-memory embedding index built once (so flat-file bridge memories are covered);
    when the embedder is absent it falls back to detect()'s own retrieve (indexed
    nodes only). Both the embed pass and the NLI model load once for the whole run,
    so this is the right home for flat-file coverage — never per-Stop-hook. limit
    chunks a large run; the deferred remainder is reported, never silently dropped,
    and the watermark only advances over what was probed so the next run resumes.

    enforce defaults to observe-only (_enforce_default); pass True to stale.
    Returns {'probed','contradictions','deferred','from_id','to_id','mode','enforced'}.
    """
    migrate(conn)
    start = 0 if backfill else _cursor(conn)
    rows = conn.execute(
        "SELECT id, content FROM nodes WHERE id > ? AND tombstoned = 0 "
        "AND channel != 'agent-derived' ORDER BY id ASC", (start,)).fetchall()
    total = len(rows)
    if limit is not None:
        rows = rows[:limit]

    # Resolve the scorer ONCE. A non-None nli suppresses detect()'s env-based default;
    # use_nli=False forces structured-only via an always-abstain scorer (also the
    # graceful path when the model is unavailable).
    if nli is None:
        nli = nli_score if (use_nli and nli_available()) else (lambda _p, _h: None)
    if enforce is None:
        enforce = _enforce_default()   # OBSERVE-only unless $MEMDAG_CONTRADICT_ENFORCE

    # Candidate source: injected (tests) > in-memory embedding index > retrieve.
    if candidate_fn is None:
        candidate_fn = _embed_candidate_fn(conn, k)
    mode = "embed" if candidate_fn is not None else "retrieve"

    stats = {"probed": 0, "contradictions": 0,
             "deferred": max(0, total - len(rows)),
             "from_id": start, "to_id": start, "mode": mode,
             "enforced": bool(enforce)}
    for nid, content in rows:
        if candidate_fn is not None:
            marked = detect(conn, nid, k=k, nli=nli, enforce=enforce,
                            candidates=candidate_fn(nid, content))
        else:
            marked = detect(conn, nid, k=k, nli=nli, enforce=enforce, clearance=clearance)
        stats["probed"] += 1
        stats["contradictions"] += len(marked)
        stats["to_id"] = nid
    if rows:
        _set_cursor(conn, rows[-1][0])
    return stats


# --- CLI ---------------------------------------------------------------------

def _cmd_list(args):
    conn = memsom.get_connection()
    try:
        rows = list_contradictions(conn, observed_only=args.observed_only)
    finally:
        conn.close()
    if not rows:
        print("no contradictions recorded.")
        return 0
    print(f"{len(rows)} contradiction(s):")
    for r in rows:
        sc = f" p={r['score']:.2f}" if r["score"] is not None else ""
        state = "enforced" if r["enforced"] else "observed"
        print(f"  {r['at'] or '?':<26} [{state}] old[{r['old_id']}] <- new[{r['new_id']}] "
              f"({r['verdict']}{sc}) {r['reason'] or ''}")
    return 0


def _cmd_sweep(args):
    conn = memsom.get_connection()
    try:
        stats = sweep(conn, limit=args.limit, use_nli=not args.no_nli,
                      backfill=args.backfill,
                      enforce=(True if args.enforce else None))
    finally:
        conn.close()
    tier = "structured-only" if args.no_nli else (
        "structured+NLI" if nli_available() else "structured-only (NLI unavailable)")
    mode = "ENFORCE (staled)" if stats["enforced"] else "observe-only (recorded, not staled)"
    defer = (f", {stats['deferred']} deferred (raise/rerun --limit)"
             if stats["deferred"] else "")
    cov = "" if stats["mode"] == "embed" else " [candidates: retrieve — bridge memories not covered (no embedder)]"
    print(f"[contradict-sweep] {tier}, {mode}: probed {stats['probed']} node(s) "
          f"(id {stats['from_id']}->{stats['to_id']}), "
          f"{stats['contradictions']} contradiction(s){defer}.{cov}")
    return 0


def register(subparsers):
    p = subparsers.add_parser("contradictions-list",
                              help="list recorded cross-source contradictions")
    p.add_argument("--observed-only", action="store_true",
                   help="show only observe-mode rows (recorded, not staled)")
    p.set_defaults(func=_cmd_list)

    s = subparsers.add_parser(
        "contradict-sweep",
        help="batch-scan memories added since the last sweep for contradictions "
             "(covers flat-file bridge memories the ingest hook doesn't)")
    s.add_argument("--limit", type=int, default=None,
                   help="max nodes to probe this run (chunk a large backfill)")
    s.add_argument("--no-nli", action="store_true",
                   help="structured tier only — skip loading the NLI model")
    s.add_argument("--backfill", action="store_true",
                   help="re-scan from the beginning (one-time full pass)")
    s.add_argument("--enforce", action="store_true",
                   help="mark contradictions stale (default: observe-only — record, don't stale)")
    s.set_defaults(func=_cmd_sweep)


def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom-contradict")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
