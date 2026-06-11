"""memdag_anticipatory — anticipatory coprocess: honest heuristics, no ML.

Logs queries, gates writes on surprise (novelty), and prefetches/warms the
most-asked answers. A low-surprise repeat CITES the existing node instead of
minting a near-duplicate.

Public API
----------
migrate(conn)
observe(conn, query, answer_node=None)
novelty(answer_text, existing_texts) -> float
existing_derived(conn) -> list[(id, content)]
surprise_gated(conn, question, threshold=0.35, sources=None)
    -> (node_id, created: bool, score: float)
prefetch(conn, k=3, threshold=0.35) -> list[(query, node_id, created)]

CLI (register(subparsers)):
    observe <query>
    prefetch [--k N] [--threshold F]
main(argv=None)
"""

import sys

import memdag
import memdag_schema
import memdag_quarantine
import memdag_redact


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(conn):
    """Idempotent: run dependency migrations and create query_log if absent."""
    memdag_quarantine.migrate(conn)
    memdag_redact.migrate(conn)
    memdag_schema.ensure_table(
        conn,
        "CREATE TABLE IF NOT EXISTS query_log ("
        "  id          INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  ts          TEXT    NOT NULL,"
        "  query       TEXT    NOT NULL,"
        "  answer_node INTEGER"
        ")"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def observe(conn, query, answer_node=None):
    """Insert a query-log row. answer_node is nullable/loose — no FK check."""
    with conn:
        conn.execute(
            "INSERT INTO query_log(ts, query, answer_node) VALUES (?,?,?)",
            (memdag.now_iso(), query, answer_node)
        )


def novelty(answer_text, existing_texts):
    """Return 1 - max_jaccard(answer_text, each existing_text).

    Jaccard = |A ∩ B| / |A ∪ B|;  empty union -> similarity 0 (not nan).
    Empty existing_texts -> 1.0 (everything is novel with no competition).
    Pure function — no conn, no clock.
    """
    if not existing_texts:
        return 1.0
    a_stems = memdag.stems(answer_text)
    max_sim = 0.0
    for t in existing_texts:
        b_stems = memdag.stems(t)
        union = a_stems | b_stems
        if not union:
            sim = 0.0
        else:
            sim = len(a_stems & b_stems) / len(union)
        if sim > max_sim:
            max_sim = sim
    return 1.0 - max_sim


def existing_derived(conn):
    """Return live agent-derived nodes that are not redacted and not quarantined.

    These are the only nodes safe to cite as 'already covered'.
    Returns list of (id, content) tuples.
    """
    return conn.execute(
        "SELECT id, content FROM nodes"
        " WHERE channel = 'agent-derived'"
        "   AND tombstoned = 0"
        "   AND redacted = 0"
        "   AND status != 'quarantined'"
        " ORDER BY id ASC"
    ).fetchall()


def _live_sources_filtered(conn):
    """Live sources filtered to status='live' AND redacted=0.

    Returns rows with the same shape as memdag.live_sources:
    (id, content, channel, label, source_ref).
    """
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE tombstoned = 0"
        "   AND channel != 'agent-derived'"
        "   AND status = 'live'"
        "   AND redacted = 0"
        " ORDER BY label DESC, id ASC"
    ).fetchall()


def surprise_gated(conn, question, threshold=0.35, sources=None):
    """Compose an answer; return existing node if low-surprise, else derive new.

    Parameters
    ----------
    conn        : DB connection (migrations must have been run)
    question    : query string
    threshold   : novelty cutoff; score < threshold -> cite existing
    sources     : explicit source rows (shape: id, content, channel, label, source_ref)
                  Default: live_sources filtered by status='live' AND redacted=0

    Returns
    -------
    (node_id, created, score)
      node_id  — id of the node that covers this question
      created  — True if a new node was minted, False if existing was cited
      score    — novelty score (0.0 = duplicate, 1.0 = fully novel)

    Raises
    ------
    ValueError  — if no live source yielded any claim (mirrors ask's refusal;
                  CLI layer translates to sys.exit(1))
    """
    if sources is None:
        sources = _live_sources_filtered(conn)

    text, used = memdag.compose(question, sources)
    if text is None:
        raise ValueError("no live source yielded any claim")

    existing = existing_derived(conn)
    score = novelty(text, [c for _, c in existing])

    if score < threshold and existing:
        # Find the existing node whose content is most similar (argmax Jaccard)
        a_stems = memdag.stems(text)
        best_id = None
        best_sim = -1.0
        for nid, content in existing:
            b_stems = memdag.stems(content)
            union = a_stems | b_stems
            sim = len(a_stems & b_stems) / len(union) if union else 0.0
            if best_id is None or sim > best_sim or (sim == best_sim and nid < best_id):
                best_id = nid
                best_sim = sim
        observe(conn, question, best_id)
        return (best_id, False, score)

    # High surprise: derive a new node
    nid, _ = memdag.derive_node(conn, text, used)
    observe(conn, question, nid)
    return (nid, True, score)


def prefetch(conn, k=3, threshold=0.35):
    """Warm the k most-asked queries from the log.

    Runs surprise_gated for each query (ordered by frequency DESC, recency DESC).
    Skips queries where surprise_gated raises ValueError (no live sources).

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
        try:
            node_id, created, _score = surprise_gated(conn, query, threshold=threshold)
        except ValueError:
            continue
        results.append((query, node_id, created))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_observe(args):
    conn = memdag.get_connection()
    migrate(conn)
    try:
        observe(conn, args.query)
        print("logged")
    finally:
        conn.close()


def cmd_prefetch(args):
    conn = memdag.get_connection()
    migrate(conn)
    try:
        results = prefetch(conn, k=args.k, threshold=args.threshold)
        for query, node_id, created in results:
            label = "new" if created else "cached"
            print(f'"{query}" -> [{node_id}] ({label})')
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
    p_pre.set_defaults(func=cmd_prefetch)


def main(argv=None):
    import argparse
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memdag_anticipatory")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
