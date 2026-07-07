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
  - Phase 2: NLI — pass nli=fn where fn(premise, hypothesis) -> P(contradiction);
    used only on candidates the structured tier didn't already resolve.

Design guardrails:
  - NEVER reuses source_supersedes — freshen()/substitute_fresh() treat that as a
    value-preserving replacement and would silently serve the contradicting node as
    the corrected value. The link lives in its own `contradictions` table.
  - The stale reason is namespaced "contradicted by node N" so other passes'
    _owned()-style clear checks (e.g. verify_stale) never clear a contradiction flag.
  - Marks the OLD node (newer wins); the new node is never staled. stale is
    reversible + advisory, so a rare false positive costs a warning, not data.
  - Opt-in: only runs when $MEMDAG_CONTRADICT is truthy — keeps a cold bridge-import
    from checking every node before the detector is tuned.

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
    PRIMARY KEY (old_id, new_id)
  );""")
    with conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_new "
                     "ON contradictions(new_id)")


def enabled():
    """True iff $MEMDAG_CONTRADICT opts the detector in."""
    return str(os.environ.get("MEMDAG_CONTRADICT", "")).strip().lower() in _TRUTHY


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


def detect(conn, new_id, *, k=5, candidates=None, nli=None, nli_threshold=0.85,
           clearance="topsecret"):
    """Detect live nodes that *new_id* contradicts; mark each stale + record the link.

    candidates: optional [(id, content), ...] to adjudicate against (injected in
    tests). When None, live topically-close candidates come from the hybrid
    retriever (its BM25 half works with no embedder, so this degrades gracefully).
    nli: optional fn(premise, hypothesis) -> P(contradiction) in [0,1]; used only
    where the structured tier abstains.

    Returns a sorted list of (old_id, verdict_kind). Marks the OLD node, never new_id.
    """
    migrate(conn)
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
        if conn.execute("SELECT 1 FROM contradictions WHERE old_id=? AND new_id=?",
                        (cid, new_id)).fetchone():
            continue
        kind, reason, score = verdict
        memsom_stale.mark_stale_cascade(conn, cid, f"{REASON_PREFIX} node {new_id}")
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO contradictions"
                "(old_id, new_id, verdict, judge, score, reason, at) "
                "VALUES (?,?,?,?,?,?,?)",
                (cid, new_id, kind, kind, score, reason, ts))
        marked.append((cid, kind))
    return sorted(marked)


def list_contradictions(conn):
    """All recorded contradictions, newest first."""
    migrate(conn)
    rows = conn.execute(
        "SELECT old_id, new_id, verdict, score, reason, at "
        "FROM contradictions ORDER BY at DESC, new_id DESC").fetchall()
    return [{"old_id": r[0], "new_id": r[1], "verdict": r[2],
             "score": r[3], "reason": r[4], "at": r[5]} for r in rows]


# --- CLI ---------------------------------------------------------------------

def _cmd_list(args):
    conn = memsom.get_connection()
    try:
        rows = list_contradictions(conn)
    finally:
        conn.close()
    if not rows:
        print("no contradictions recorded.")
        return 0
    print(f"{len(rows)} contradiction(s):")
    for r in rows:
        sc = f" p={r['score']:.2f}" if r["score"] is not None else ""
        print(f"  {r['at'] or '?':<26} old[{r['old_id']}] <- new[{r['new_id']}] "
              f"({r['verdict']}{sc}) {r['reason'] or ''}")
    return 0


def register(subparsers):
    p = subparsers.add_parser("contradictions-list",
                              help="list recorded cross-source contradictions")
    p.set_defaults(func=_cmd_list)


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
