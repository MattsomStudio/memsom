"""memsom_anticipatory — anticipatory coprocess (Phase 2): retrieval-backed
surprise, warm prefetch cache, and novel-recombination warnings.

THE LOAD-BEARING SECURITY PROPERTY
----------------------------------
The coprocess reads/learns/prefetches ONLY from UNTAINTED memory.  Every
corpus it touches (surprise comparison, prefetch composition, recombination
history) excludes: tombstoned=1, redacted=1, status='quarantined', archived=1,
external-tainted derivations (agent-derived nodes whose integrity label is
EXTERNAL/0), and anything above the caller's clearance.  This mirrors the
spine's pool filters (memsom_cli._build_pool / memsom_retrieve.retrieve).
A poisoned node can therefore never be amplified by prefetch, cited by the
surprise gate, or counted as recombination precedent.

The coprocess also never elevates trust: every node it mints goes through
memsom.derive_node (label = min(parents)) with honest provenance edges, and
warm answers are re-validated against the untainted filter at serve time.

Public API
----------
migrate(conn)
observe(conn, query, answer_node=None)
novelty(answer_text, existing_texts) -> float          (legacy Jaccard, kept)
untainted_sources(conn, clearance) -> rows              (id, content, channel, label, source_ref)
untainted_derived(conn, clearance) -> list[(id, content)]
existing_derived(conn) -> list[(id, content)]           (kept; = untainted_derived)
rank_similar(conn, text, k=8, ...) -> list[(nid, sim)]
surprise(conn, text, k=8, ...) -> float                 (1.0 = fully novel)
surprise_gated_write(conn, question, threshold=0.35, sources=None,
                     clearance="topsecret") -> (node_id, created, score)
surprise_gated(conn, question, threshold=0.35, sources=None)   (kept wrapper)
prefetch(conn, k=3, threshold=0.35, clearance="topsecret", pool_k=8)
    -> list[(query, node_id, created)]
serve_warm(conn, query, clearance="topsecret") -> dict | None
novel_recombination(conn, parent_ids, exclude_node=None, clearance="topsecret")
    -> (is_novel: bool, prior_node_ids: list[int])
status(conn, clearance="topsecret") -> dict

CLI (register(subparsers)):
    observe <query>
    prefetch [--k N] [--threshold F] [--clearance C]
    anticipate-status
main(argv=None)
"""

import math
import sys

import memsom
import memsom_schema
import memsom_rederive
import memsom_quarantine
import memsom_redact
import memsom_confid
import memsom_retrieve


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_PREFETCH_SQL = """CREATE TABLE IF NOT EXISTS prefetch_cache (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  query       TEXT    NOT NULL UNIQUE,
  answer_node INTEGER NOT NULL REFERENCES nodes(id),
  created_at  TEXT    NOT NULL,
  hits        INTEGER NOT NULL DEFAULT 0,
  last_served TEXT
);"""


def migrate(conn):
    """Idempotent: run dependency migrations, create query_log + prefetch_cache."""
    memsom_quarantine.migrate(conn)
    memsom_redact.migrate(conn)
    memsom_confid.migrate(conn)
    memsom_retrieve.migrate(conn)
    memsom_rederive.migrate(conn)  # record_recipe writes to derivation_recipe at mint
    memsom_schema.ensure_table(
        conn,
        "CREATE TABLE IF NOT EXISTS query_log ("
        "  id          INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  ts          TEXT    NOT NULL,"
        "  query       TEXT    NOT NULL,"
        "  answer_node INTEGER"
        ")"
    )
    memsom_schema.ensure_table(conn, _PREFETCH_SQL)


# ---------------------------------------------------------------------------
# Untainted pool — THE security boundary of this module.
# Every read path below goes through these helpers (or _is_untainted_derived).
# ---------------------------------------------------------------------------

def _untainted_clauses(conn, clearance):
    """Shared WHERE fragments mirroring the spine's pool filters.

    tombstoned is ALWAYS excluded; the other filters are applied whenever the
    corresponding column exists (a missing column means the feature module has
    never run, so no node can carry that taint marker — nothing to exclude).

    Delegates to memsom_schema.taint_filter_clauses — the ONE shared
    untainted-pool primitive (same clauses as memsom_cli._build_pool and
    memsom_retrieve._build_retrieve_pool, by construction).
    """
    return memsom_schema.taint_filter_clauses(
        conn, clearance=memsom_confid.parse_conf(clearance))


def untainted_sources(conn, clearance="topsecret"):
    """Live, untainted SOURCE rows in trust order — the only pool the
    coprocess may compose from.  Same shape as memsom.live_sources:
    (id, content, channel, label, source_ref)."""
    clauses, params = _untainted_clauses(conn, clearance)
    sql = (
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE channel != 'agent-derived' AND " + " AND ".join(clauses) +
        " ORDER BY label DESC, id ASC"
    )
    return conn.execute(sql, params).fetchall()


def untainted_derived(conn, clearance="topsecret"):
    """Live, untainted agent-derived nodes — the only answers the coprocess
    may cite, learn from, or serve warm.  Returns list of (id, content).

    label > 0 additionally excludes EXTERNAL-tainted derivations (an
    agent-derived node whose integrity floor is EXTERNAL derives, possibly
    transitively, from an external source — the consolidation gate quarantines
    exactly these; we refuse them here directly as defense in depth)."""
    clauses, params = _untainted_clauses(conn, clearance)
    sql = (
        "SELECT id, content FROM nodes"
        " WHERE channel = 'agent-derived' AND label > 0 AND "
        + " AND ".join(clauses) + " ORDER BY id ASC"
    )
    return conn.execute(sql, params).fetchall()


def existing_derived(conn):
    """Back-compat alias (pre-Phase-2 public API): untainted derived nodes
    at full clearance.  Returns list of (id, content) tuples."""
    return untainted_derived(conn, clearance="topsecret")


def _is_untainted_derived(conn, nid, clearance="topsecret"):
    """True iff *nid* is an agent-derived node passing every untainted filter."""
    clauses, params = _untainted_clauses(conn, clearance)
    row = conn.execute(
        "SELECT 1 FROM nodes WHERE id = ? AND channel = 'agent-derived'"
        " AND label > 0 AND " + " AND ".join(clauses),
        [nid] + params,
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# observe — query log
# ---------------------------------------------------------------------------

def observe(conn, query, answer_node=None):
    """Insert a query-log row. answer_node is nullable/loose — no FK check."""
    with conn:
        conn.execute(
            "INSERT INTO query_log(ts, query, answer_node) VALUES (?,?,?)",
            (memsom.now_iso(), query, answer_node)
        )


# ---------------------------------------------------------------------------
# Legacy Jaccard novelty (kept — pure function, still useful for cheap checks)
# ---------------------------------------------------------------------------

def novelty(answer_text, existing_texts):
    """Return 1 - max_jaccard(answer_text, each existing_text).

    Jaccard = |A ∩ B| / |A ∪ B|;  empty union -> similarity 0 (not nan).
    Empty existing_texts -> 1.0 (everything is novel with no competition).
    Pure function — no conn, no clock.
    """
    if not existing_texts:
        return 1.0
    a_stems = memsom.stems(answer_text)
    max_sim = 0.0
    for t in existing_texts:
        b_stems = memsom.stems(t)
        union = a_stems | b_stems
        if not union:
            sim = 0.0
        else:
            sim = len(a_stems & b_stems) / len(union)
        if sim > max_sim:
            max_sim = sim
    return 1.0 - max_sim


# ---------------------------------------------------------------------------
# Real (semantic) surprise — BM25-IDF-weighted lexical cosine + optional
# Ollama vector cosine, over the untainted corpus only.
# ---------------------------------------------------------------------------

def _tf_map(text):
    """Term-frequency map using the retrieval tokenizer (stems, stopwords out)."""
    m = {}
    for t in memsom_retrieve.tokenize(text):
        m[t] = m.get(t, 0) + 1
    return m


def _lexical_sims(text, corpus):
    """BM25-IDF-weighted TF cosine between *text* and each corpus doc.

    corpus: list[(nid, content)].  Returns {nid: sim} with sim in [0, 1].
    Identical text -> 1.0; fully disjoint stems -> 0.0.  Deterministic, stdlib.
    """
    probe = _tf_map(text)
    if not probe or not corpus:
        return {}
    docs = [(nid, _tf_map(content)) for nid, content in corpus]
    n_docs = len(docs)
    df = {}
    for _, d in docs:
        for term in d:
            df[term] = df.get(term, 0) + 1

    def idf(term):
        # Classic BM25 idf (+1 inside the log keeps it positive at df == N)
        return math.log((n_docs - df.get(term, 0) + 0.5)
                        / (df.get(term, 0) + 0.5) + 1.0)

    def weighted(tfm):
        return {t: tf * idf(t) for t, tf in tfm.items()}

    pw = weighted(probe)
    pn = math.sqrt(sum(v * v for v in pw.values()))
    sims = {}
    for nid, d in docs:
        dw = weighted(d)
        dn = math.sqrt(sum(v * v for v in dw.values()))
        if pn == 0.0 or dn == 0.0:
            sims[nid] = 0.0
            continue
        dot = sum(v * dw.get(t, 0.0) for t, v in pw.items())
        sims[nid] = dot / (pn * dn)
    return sims


def _vector_sims(text, candidates):
    """Ollama-embedding cosine for *text* vs each (nid, content) candidate.

    Degrades to {} silently when Ollama is unreachable — never crashes
    (same pattern as memsom_retrieve.vector_search)."""
    if not candidates:
        return {}
    try:
        qv = memsom_retrieve._call_ollama_embed(text)
    except Exception:
        return {}
    out = {}
    for nid, content in candidates:
        try:
            cv = memsom_retrieve._call_ollama_embed(content)
        except Exception:
            continue
        out[nid] = memsom_retrieve._cosine(qv, cv)
    return out


def _corpus_for_scope(conn, clearance, scope):
    if scope == "derived":
        return untainted_derived(conn, clearance)
    if scope == "sources":
        return [(r[0], r[1]) for r in untainted_sources(conn, clearance)]
    if scope == "all":
        return (untainted_derived(conn, clearance)
                + [(r[0], r[1]) for r in untainted_sources(conn, clearance)])
    raise ValueError(f"unknown scope: {scope!r}")


def rank_similar(conn, text, k=8, clearance="topsecret", corpus=None,
                 scope="derived"):
    """Rank the untainted corpus by similarity to *text*.

    Returns list of (nid, sim) sorted by sim DESC (ties: lower id first),
    up to k entries.  Lexical BM25-IDF cosine always runs (stdlib); the top-k
    lexical candidates are additionally scored by Ollama vector cosine when
    reachable, and each node's similarity is max(lexical, vector).
    """
    if corpus is None:
        corpus = _corpus_for_scope(conn, clearance, scope)
    if not corpus:
        return []
    lex = _lexical_sims(text, corpus)
    if not lex:
        return []
    ranked = sorted(lex.items(), key=lambda x: (-x[1], x[0]))
    by_id = {nid: content for nid, content in corpus}
    cand = [(nid, by_id[nid]) for nid, _ in ranked[:max(k, 1)]]
    vec = _vector_sims(text, cand)
    merged = {nid: max(s, vec.get(nid, 0.0)) for nid, s in lex.items()}
    out = sorted(merged.items(), key=lambda x: (-x[1], x[0]))
    return out[:k]


def surprise(conn, text, k=8, clearance="topsecret", corpus=None,
             scope="derived"):
    """Novelty of *text* against the UNTAINTED corpus.  1.0 = fully novel,
    0.0 = an (effectively) identical answer already exists."""
    ranked = rank_similar(conn, text, k=k, clearance=clearance,
                          corpus=corpus, scope=scope)
    if not ranked:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ranked[0][1]))


# ---------------------------------------------------------------------------
# Surprise-gated write — semantic dedup on the write path
# ---------------------------------------------------------------------------

def surprise_gated_write(conn, question, threshold=0.35, sources=None,
                         clearance="topsecret"):
    """Compose an answer; CITE the existing untainted node if low-surprise,
    else derive a new one (label = min(parents) via memsom.derive_node —
    no trust elevation, honest provenance edges).

    Parameters
    ----------
    conn      : DB connection (migrations must have been run)
    question  : query string
    threshold : novelty cutoff; score < threshold -> cite existing
    sources   : explicit source rows (id, content, channel, label, source_ref);
                default = the untainted source pool at *clearance*
    clearance : confidentiality ceiling for both the default source pool and
                the derived-answer comparison corpus

    Returns (node_id, created: bool, score: float).
    Raises ValueError if no live source yielded any claim (nothing is logged).
    """
    if sources is None:
        sources = untainted_sources(conn, clearance)

    text, used = memsom.compose(question, sources)
    if text is None:
        raise ValueError("no live source yielded any claim")

    corpus = untainted_derived(conn, clearance)
    ranked = rank_similar(conn, text, clearance=clearance, corpus=corpus)
    score = max(0.0, min(1.0, 1.0 - ranked[0][1])) if ranked else 1.0

    if ranked and score < threshold:
        best_id = ranked[0][0]
        observe(conn, question, best_id)
        return (best_id, False, score)

    # ANTICIPATORY-2: take the write lock and re-validate the cited parents
    # against the FULL taint filter before minting. derive_node's frozen recheck
    # only covers `tombstoned`; a redact/quarantine landing between the
    # (out-of-lock) source read above and the mint would otherwise leave a node
    # citing a now-tainted parent that still passes _is_untainted_derived.
    began = not conn.in_transaction
    if began:
        conn.execute("BEGIN IMMEDIATE")
    placeholders = ",".join("?" * len(used))
    clauses, params = memsom_schema.taint_filter_clauses(conn)
    live = {
        r[0] for r in conn.execute(
            f"SELECT id FROM nodes WHERE id IN ({placeholders}) AND "
            + " AND ".join(clauses),
            list(used) + params,
        )
    }
    if set(used) != live:
        if began:
            conn.rollback()
        raise ValueError("a cited source became tainted before mint")

    # derive_node honors the open transaction (commits on its own context exit),
    # so the re-validation above and the insert are one write-locked unit.
    nid, _ = memsom.derive_node(conn, text, used)
    # Prefetch composes deterministically (memsom.compose), so the question is the
    # whole recipe — regeneration replays byte-identically from live parents.
    memsom_rederive.record_recipe(conn, nid, "compose", question=question)
    # ANTICIPATORY-1: derive_node mints at conf_label DEFAULT 0 (PUBLIC). Stamp the
    # high-water confidentiality label at the MINT path — not in each caller — so a
    # node summarising SECRET sources can never be cached/served below clearance.
    # (The interactive `ask` path did this itself; prefetch did not. Fixing it here
    # removes the asymmetry for every caller.)
    memsom_confid.recompute_conf(conn, nid)
    observe(conn, question, nid)
    return (nid, True, score)


def surprise_gated(conn, question, threshold=0.35, sources=None):
    """Pre-Phase-2 public API, preserved.  Now backed by the semantic
    surprise_gated_write at full clearance."""
    return surprise_gated_write(conn, question, threshold=threshold,
                                sources=sources, clearance="topsecret")


# ---------------------------------------------------------------------------
# Prefetch — warm cache of the most-likely-next answers
# ---------------------------------------------------------------------------

def prefetch(conn, k=3, threshold=0.35, clearance="topsecret", pool_k=8):
    """Warm the k most-asked queries from the log.

    For each query (frequency DESC, recency DESC): build the source pool via
    real hybrid retrieval (memsom_retrieve.retrieve — untainted by
    construction; falls back to the full untainted pool when the BM25 index
    is empty), run surprise_gated_write (semantic dedup + honest provenance),
    and store the answer warm in prefetch_cache — but ONLY if the answer node
    itself passes the untainted-derived filter (an external-tainted answer is
    never cached).  Queries with no composable source are skipped.

    Returns list of (query, node_id, created).
    """
    rows = conn.execute(
        "SELECT query, COUNT(*) c, MAX(ts) m"
        " FROM query_log"
        " GROUP BY query"
        " ORDER BY c DESC, m DESC"
        " LIMIT ?",
        (k,)
    ).fetchall()

    results = []
    for query, _count, _ts in rows:
        pool = memsom_retrieve.retrieve(conn, query, k=pool_k,
                                        clearance=clearance)
        if not pool:
            pool = untainted_sources(conn, clearance)
        try:
            node_id, created, _score = surprise_gated_write(
                conn, query, threshold=threshold, sources=pool,
                clearance=clearance)
        except ValueError:
            continue
        if _is_untainted_derived(conn, node_id, clearance):
            with conn:
                conn.execute(
                    "INSERT INTO prefetch_cache(query, answer_node, created_at)"
                    " VALUES (?,?,?)"
                    " ON CONFLICT(query) DO UPDATE SET"
                    "   answer_node = excluded.answer_node,"
                    "   created_at  = excluded.created_at",
                    (query, node_id, memsom.now_iso()),
                )
        results.append((query, node_id, created))
    return results


def serve_warm(conn, query, clearance="topsecret"):
    """Serve a prefetched answer for an exact-match *query*, or None.

    The cached answer node is RE-VALIDATED against the untainted filter at
    serve time: if it has since been revoked/redacted/quarantined/archived,
    or sits above *clearance*, the stale cache row is dropped and None is
    returned — a poisoned answer can never be served warm.

    Returns dict(node_id, content, label, prefetched_at, hits) on a hit;
    hits/last_served are bumped.
    """
    row = conn.execute(
        "SELECT id, answer_node, created_at, hits FROM prefetch_cache"
        " WHERE query = ?",
        (query,)
    ).fetchone()
    if row is None:
        return None
    cache_id, node_id, created_at, hits = row
    if not _is_untainted_derived(conn, node_id, clearance):
        with conn:  # cache hygiene: prefetch_cache is a cache, not history
            conn.execute("DELETE FROM prefetch_cache WHERE id = ?", (cache_id,))
        return None
    node = memsom.get_node(conn, node_id)
    with conn:
        conn.execute(
            "UPDATE prefetch_cache SET hits = hits + 1, last_served = ?"
            " WHERE id = ?",
            (memsom.now_iso(), cache_id),
        )
    return {
        "node_id": node_id,
        "content": node["content"],
        "label": node["label"],
        "prefetched_at": created_at,
        "hits": hits + 1,
    }


# ---------------------------------------------------------------------------
# Novel-recombination detection
# ---------------------------------------------------------------------------

def novel_recombination(conn, parent_ids, exclude_node=None,
                        clearance="topsecret"):
    """Has this EXACT parent-set produced a derived node before?

    Deterministic over the edges history: a prior derivation matches only if
    its parent set EQUALS set(parent_ids).  Matches are restricted to
    untainted derived nodes — a quarantined/redacted/revoked/external-tainted
    prior never counts as precedent (nothing live vouches for the combination,
    and a poisoned node must not surface here).

    Returns (is_novel, prior_node_ids).  exclude_node lets a caller that just
    minted a node ask about its own parent set.
    """
    pset = sorted({int(p) for p in parent_ids})
    if not pset:
        return (True, [])
    qmarks = ",".join("?" * len(pset))
    rows = conn.execute(
        "SELECT child FROM edges GROUP BY child"
        " HAVING COUNT(*) = ?"
        f" AND SUM(CASE WHEN parent IN ({qmarks}) THEN 1 ELSE 0 END) = ?",
        [len(pset)] + pset + [len(pset)],
    ).fetchall()
    prior = sorted(
        c for (c,) in rows
        if c != exclude_node and _is_untainted_derived(conn, c, clearance)
    )
    return (len(prior) == 0, prior)


# ---------------------------------------------------------------------------
# Status — query log + prefetch cache state
# ---------------------------------------------------------------------------

def status(conn, clearance="topsecret", top=5):
    """Snapshot of the coprocess state: query-log stats, top queries, and
    every prefetch_cache row with its serve-time validity."""
    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    distinct = conn.execute(
        "SELECT COUNT(DISTINCT query) FROM query_log").fetchone()[0]
    top_rows = conn.execute(
        "SELECT query, COUNT(*) c, MAX(ts) m FROM query_log"
        " GROUP BY query ORDER BY c DESC, m DESC LIMIT ?",
        (top,)
    ).fetchall()
    cache = []
    for q, nid, created_at, hits, last in conn.execute(
        "SELECT query, answer_node, created_at, hits, last_served"
        " FROM prefetch_cache ORDER BY id"
    ).fetchall():
        cache.append({
            "query": q,
            "answer_node": nid,
            "created_at": created_at,
            "hits": hits,
            "last_served": last,
            "valid": _is_untainted_derived(conn, nid, clearance),
        })
    return {
        "query_log_total": total,
        "distinct_queries": distinct,
        "top_queries": [(q, c, m) for q, c, m in top_rows],
        "cache": cache,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_observe(args):
    conn = memsom.get_connection()
    migrate(conn)
    try:
        observe(conn, args.query)
        print("logged")
    finally:
        conn.close()


def _validate_clearance(args):
    """ANTICIPATORY-CLI-CLEARANCE: mirror CLI-1 — reject a bad --clearance with a
    clean message + exit 1 instead of an uncaught ValueError traceback."""
    try:
        memsom_confid.parse_conf(args.clearance)
    except ValueError as exc:
        print(f"[memsom] invalid --clearance: {exc}", file=sys.stderr)
        sys.exit(1)


def cmd_prefetch(args):
    conn = memsom.get_connection()
    migrate(conn)
    try:
        _validate_clearance(args)  # inside try/finally so a bad value can't leak conn
        results = prefetch(conn, k=args.k, threshold=args.threshold,
                           clearance=args.clearance)
        warm_ids = {r[0] for r in conn.execute(
            "SELECT answer_node FROM prefetch_cache").fetchall()}
        for query, node_id, created in results:
            label = "new" if created else "cached"
            warm = " +warm" if node_id in warm_ids else ""
            print(f'"{query}" -> [{node_id}] ({label}){warm}')
        n_cache = conn.execute(
            "SELECT COUNT(*) FROM prefetch_cache").fetchone()[0]
        print(f"prefetch cache: {n_cache} entr{'y' if n_cache == 1 else 'ies'}")
    finally:
        conn.close()


def cmd_anticipate_status(args):
    conn = memsom.get_connection()
    migrate(conn)
    try:
        _validate_clearance(args)  # inside try/finally so a bad value can't leak conn
        s = status(conn, clearance=args.clearance)
        print(f"query log: {s['query_log_total']} row(s),"
              f" {s['distinct_queries']} distinct quer"
              f"{'y' if s['distinct_queries'] == 1 else 'ies'}")
        for q, c, m in s["top_queries"]:
            print(f'  {c}x  "{q}"  (last {m})')
        if not s["cache"]:
            print("prefetch cache: empty")
        else:
            print(f"prefetch cache: {len(s['cache'])} entr"
                  f"{'y' if len(s['cache']) == 1 else 'ies'}")
            for e in s["cache"]:
                state = "valid" if e["valid"] else "STALE (will not serve)"
                print(f'  [{e["answer_node"]}] "{e["query"]}"'
                      f'  hits={e["hits"]}  prefetched={e["created_at"]}'
                      f'  {state}')
    finally:
        conn.close()


def register(subparsers):
    p_obs = subparsers.add_parser("observe",
                                   help="log a query to the anticipatory query log")
    p_obs.add_argument("query")
    p_obs.set_defaults(func=cmd_observe)

    p_pre = subparsers.add_parser("prefetch",
                                   help="warm the k most-asked answers from the log")
    p_pre.add_argument("--k", type=int, default=3,
                       help="number of top queries to prefetch (default 3)")
    p_pre.add_argument("--threshold", type=float, default=0.35,
                       help="novelty threshold (default 0.35)")
    p_pre.add_argument("--clearance", default="topsecret",
                       help="confidentiality clearance (default: topsecret)")
    p_pre.set_defaults(func=cmd_prefetch)

    p_st = subparsers.add_parser("anticipate-status",
                                  help="show query log + prefetch cache state")
    p_st.add_argument("--clearance", default="topsecret",
                      help="confidentiality clearance (default: topsecret)")
    p_st.set_defaults(func=cmd_anticipate_status)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_anticipatory")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
