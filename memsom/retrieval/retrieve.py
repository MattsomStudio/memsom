"""memsom.retrieval.retrieve — hybrid BM25 + optional dense-vector retrieval.

Backend is pluggable via `embed.backend` / MEMDAG_EMBED_BACKEND
(ollama | bge-m3 | bm25). The opt-in bge-m3 backend fuses dense (1024-dim)
+ learned-sparse + ColBERT late-interaction rerank, degrading to Ollama then
BM25 when absent.

BM25 runs on pure stdlib (no external deps).
Ollama vectors are optional: unreachable or missing embeddings degrade
silently to BM25-only — never a crash.

Schema (new tables, additive):
  postings(term TEXT, node_id INT, tf INT, PRIMARY KEY(term, node_id))
  docstats(node_id INT PRIMARY KEY, length INT)
  embeddings(node_id INT PRIMARY KEY, model TEXT, dim INT, vec BLOB)

Public API
----------
migrate(conn)
tokenize(text) -> list[str]
index_node(conn, nid)
index_all(conn)
bm25(conn, query, k) -> [(nid, score)]
vector_search(conn, query, k) -> [(nid, score)]
retrieve(conn, query, k=8, clearance="topsecret", min_integrity=None,
         exclude_quarantined=True, exclude_redacted=True) -> list[tuple]
pinned_backend(conn) / pin_backend(conn, name)   # the store's own backend
embedding_coverage(conn) -> dict                 # dense reach over live nodes
retrieval_warnings(conn) -> list[str]            # the AI-facing degraded signal
last_query_fallback(conn) -> dict | None         # query-encoder fallback trail

CLI
---
retrieve <query> [--k N] [--clearance C]
reindex
register(subparsers)
main(argv=None)
"""

import argparse
import math
import struct
import sys
import json
import sqlite3

import memsom
from memsom.effects import net as memsom_net
from memsom.storage import schema as memsom_schema
from memsom.kernel import events as memsom_events
from memsom.integrity import confid as memsom_confid
from memsom.retrieval import llm as memsom_llm
from memsom import tuning as memsom_tuning

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "nomic-embed-text"
# 127.0.0.1, not localhost: on Windows `localhost` resolves to BOTH ::1 and
# 127.0.0.1, and a dead Ollama makes the ::1 attempt stall before falling back
# to IPv4 (instant-refuse on Linux, seconds per call on Windows). Ollama binds
# 127.0.0.1 by default, so this is also the more reliable target. Override with
# MEMDAG_EMBED_URL if your Ollama listens elsewhere.
#
# /api/embed (current API), NOT the legacy /api/embeddings: the legacy endpoint
# NEVER truncates and 500s "input too large" on any node past the model context
# window; /api/embed with `truncate: true` truncates server-side instead. An
# explicit MEMDAG_EMBED_URL ending in /api/embeddings still uses the legacy
# body/parse (_call_ollama_embed switches on the resolved path).
DEFAULT_EMBED_URL = "http://127.0.0.1:11434/api/embed"

# BM25 tuning
K1 = 1.2
B = 0.75

# GraphRAG re-ranking: a relevant node linked to a strong seed receives a boost
# of seed_score * GRAPH_DECAY**hops. Decays with distance so a 2-hop neighbor
# gets a quarter of the lift a 1-hop neighbor does.
GRAPH_DECAY = 0.5

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS postings (
  term    TEXT NOT NULL,
  node_id INTEGER NOT NULL,
  tf      INTEGER NOT NULL,
  PRIMARY KEY (term, node_id)
);
CREATE TABLE IF NOT EXISTS docstats (
  node_id INTEGER PRIMARY KEY,
  length  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  node_id INTEGER PRIMARY KEY,
  model   TEXT NOT NULL,
  dim     INTEGER NOT NULL,
  vec     BLOB NOT NULL
);
-- MS-32: a re-index queue. index_node writes a row here whenever the BM25
-- half succeeded but the vector half did not (bge AND Ollama both
-- unavailable/failed) -- previously that state was reported as a plain
-- `True` "indexed" with no trace, and nothing ever revisited the node once
-- the embedder recovered. A successful embed clears the row.
CREATE TABLE IF NOT EXISTS retrieval_degraded (
  node_id     INTEGER PRIMARY KEY,
  reason      TEXT NOT NULL,
  recorded_at TEXT NOT NULL
);
-- The store's OWN embedding backend (key='embed_backend'). Written by the
-- first vector write and by every full `reindex`; read by embed.backend()
-- when neither a CLI override nor MEMDAG_EMBED_BACKEND is set. Closes the
-- 2026-08-31..09-04 split: processes launched without the env var (the
-- Stop-hook importer) silently wrote nomic rows into a bge-m3 store, and
-- `WHERE model = active` then hid every one of them from dense retrieval.
CREATE TABLE IF NOT EXISTS retrieval_meta (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""

# retrieval_degraded row that records the LAST time a query-side encode fell
# back to BM25 (encoder unreachable). Negative so it can never collide with a
# nodes.id and is excluded from the re-index queue (degraded_nodes).
QUERY_DEGRADED_NID = -1
# A query-encode fallback older than this is a trail, not a live warning.
QUERY_DEGRADED_RECENT_S = 900
PIN_KEY = "embed_backend"


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: create postings, docstats, embeddings, retrieval_degraded
    tables if absent."""
    memsom_schema.ensure_table(conn, _SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Tokenize
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """Lowercase alphanumeric tokens, stopwords removed, stems applied.

    Mirrors memsom.stems() discipline but returns a list (with repeats)
    rather than a set, so term-frequency counts are accurate.
    """
    tokens = []
    for w in _alnum_words(text.lower()):
        if w not in memsom.STOP:
            tokens.append(w[:memsom.STEM_WIDTH])  # same stem width as memsom.stems()
    return tokens


def _alnum_words(text: str) -> list:
    """Split text into lowercase alphanumeric tokens."""
    import re
    return re.findall(r"[a-z0-9]+", text)


# ---------------------------------------------------------------------------
# Ollama embedding helpers
# ---------------------------------------------------------------------------

def _embed_model():
    return memsom_tuning.resolve("retrieval.embed_model") or DEFAULT_EMBED_MODEL


def _embed_url():
    return memsom_tuning.resolve("retrieval.embed_url") or DEFAULT_EMBED_URL


def _call_ollama_embed(text: str, timeout: int = None):
    """POST to Ollama's embedding endpoint. Returns list[float] or raises.

    Default endpoint is /api/embed (the current API): the body carries `input`
    plus `truncate: true`, so a node longer than the model's context window is
    truncated server-side rather than 500ing ("input too large"), and the vector
    is read from `embeddings[0]`.

    If MEMDAG_EMBED_URL is explicitly overridden to the LEGACY /api/embeddings
    endpoint, fall back to that endpoint's contract (`prompt` in the body,
    `embedding` in the reply) so an existing override keeps working.

    `timeout` defaults to the retrieval.embed_timeout knob (MEMDAG_EMBED_TIMEOUT,
    default 60s) -- the old hard-coded 10s was too short for a cold model load
    under KEEP_ALIVE=0 (every call reloads the model).
    """
    if timeout is None:
        timeout = int(memsom_tuning.resolve("retrieval.embed_timeout"))
    model = _embed_model()
    url = _embed_url()
    if url.rstrip("/").endswith("/api/embeddings"):
        # Legacy single-input endpoint (explicit MEMDAG_EMBED_URL override only).
        payload = json.dumps(memsom_llm._with_keep_alive(
            {"model": model, "prompt": text}
        )).encode("utf-8")
        raw = memsom_net.fetch(url, data=payload,
                             headers={"Content-Type": "application/json"}, timeout=timeout)
        data = json.loads(raw)
        return data["embedding"]  # list of floats
    # Current /api/embed endpoint: truncate server-side, read embeddings[0].
    payload = json.dumps(memsom_llm._with_keep_alive(
        {"model": model, "input": text, "truncate": True}
    )).encode("utf-8")
    raw = memsom_net.fetch(url, data=payload,
                         headers={"Content-Type": "application/json"}, timeout=timeout)
    data = json.loads(raw)
    return data["embeddings"][0]  # list of floats


def _vec_to_blob(vec: list) -> bytes:
    """Pack list[float] as little-endian float32 BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list:
    """Unpack little-endian float32 BLOB to list[float]."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine(a: list, b: list) -> float:
    """Pure-Python cosine similarity. Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Index a single node
# ---------------------------------------------------------------------------

_WARNED_EMBED_FALLBACK = False


def _warn_embed_fallback(exc: Exception) -> None:
    """MS-32: ONE stderr warning per process when the default (Ollama) embed
    path fails, mirroring embed.py's _warn_fallback for the bge path. Without
    it a degrading Ollama backend -- the DEFAULT install -- never surfaces
    anywhere a human would see it."""
    global _WARNED_EMBED_FALLBACK
    if _WARNED_EMBED_FALLBACK:
        return
    _WARNED_EMBED_FALLBACK = True
    import sys
    print(f"[memsom] Ollama embedding unavailable ({exc}); indexing BM25-only. "
          f"Vectors will be added automatically once Ollama recovers and this "
          f"node is re-indexed. This warning shows once.", file=sys.stderr)


def _record_degraded(conn: sqlite3.Connection, nid: int, reason: str) -> None:
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO retrieval_degraded(node_id, reason, recorded_at)"
            " VALUES (?, ?, ?)",
            (nid, reason, memsom.now_iso()),
        )


def _clear_degraded(conn: sqlite3.Connection, nid: int) -> None:
    migrate(conn)
    with conn:
        conn.execute("DELETE FROM retrieval_degraded WHERE node_id = ?", (nid,))


def degraded_nodes(conn: sqlite3.Connection) -> list:
    """Node ids indexed BM25-only because their vector embed failed -- the
    MS-32 re-index queue. `index_all`/`reindex` can pass this to a follow-up
    pass once the embedder is back. The query-side sentinel row (negative
    id) is a trail, not a node, and is never queued."""
    migrate(conn)
    return [r[0] for r in
            conn.execute("SELECT node_id FROM retrieval_degraded WHERE node_id > 0"
                         " ORDER BY node_id")]


# ---------------------------------------------------------------------------
# Query-side degradation trail + store-level health (the AI-facing signal)
# ---------------------------------------------------------------------------

_LAST_QUERY_FALLBACK = None   # reason str for THIS process's last vector_search, or None


def _record_query_degraded(conn: sqlite3.Connection, reason: str) -> None:
    """Best-effort trail: the active backend's QUERY encoder was unreachable and
    this search ran BM25-only. One sentinel row (INSERT OR REPLACE), so the
    table always shows the most recent fallback. Never raises: a read-only
    connection (features/doctor/warm) just skips the write -- the in-process
    `_LAST_QUERY_FALLBACK` still carries the signal to the caller."""
    global _LAST_QUERY_FALLBACK
    _LAST_QUERY_FALLBACK = reason
    try:
        if not memsom_schema.table_exists(conn, "retrieval_degraded"):
            return
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO retrieval_degraded(node_id, reason, recorded_at)"
                " VALUES (?, ?, ?)",
                (QUERY_DEGRADED_NID, reason, memsom.now_iso()),
            )
    # FAILOPEN: allowed -- a telemetry write must never break a read path (read-only conn, locked db).
    except Exception:
        pass


def last_query_fallback_reason():
    """This process's most recent vector_search fallback reason, or None when
    the last search encoded its query normally."""
    return _LAST_QUERY_FALLBACK


def last_query_fallback(conn: sqlite3.Connection):
    """The stored trail: {'reason', 'recorded_at', 'age_s'} of the most recent
    query-encode fallback on this store, or None. Read-only (no migrate)."""
    if not memsom_schema.table_exists(conn, "retrieval_degraded"):
        return None
    row = conn.execute(
        "SELECT reason, recorded_at FROM retrieval_degraded WHERE node_id = ?",
        (QUERY_DEGRADED_NID,)).fetchone()
    if row is None:
        return None
    age = None
    try:
        from datetime import datetime, timezone
        then = datetime.fromisoformat(row[1])
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - then).total_seconds()
    # FAILOPEN: allowed -- an unparsable timestamp just reports age unknown.
    except Exception:
        pass
    return {"reason": row[0], "recorded_at": row[1], "age_s": age}


def pinned_backend(conn: sqlite3.Connection):
    """The backend this store was embedded under ('ollama'|'bge-m3'|'bm25'),
    or None when nothing has pinned it yet. Read-only (no migrate) so it is
    safe on the read-only connections features/doctor/the hook open."""
    if not memsom_schema.table_exists(conn, "retrieval_meta"):
        return None
    row = conn.execute("SELECT value FROM retrieval_meta WHERE key = ?",
                       (PIN_KEY,)).fetchone()
    return row[0] if row else None


def pin_backend(conn: sqlite3.Connection, name: str) -> None:
    """Record *name* as the store's backend. Called by the first vector write
    (a fresh store adopts whatever first embedded into it) and by every full
    `reindex` (a deliberate switch). A later process that lacks the env var
    then resolves to THIS, not to embed.py's compiled-in default."""
    migrate(conn)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO retrieval_meta(key, value, updated_at) VALUES (?, ?, ?)",
            (PIN_KEY, name, memsom.now_iso()))


def _embeddable_clauses(conn: sqlite3.Connection) -> str:
    """The WHERE body naming exactly the nodes index_node gives a vector to:
    live source nodes that are not agent-derived, redacted or archived. Kept
    next to index_node's own skip rules so coverage can't drift from them."""
    clauses = ["tombstoned = 0", "channel != 'agent-derived'"]
    if memsom_schema.column_exists(conn, "nodes", "redacted"):
        clauses.append("redacted = 0")
    if memsom_schema.column_exists(conn, "nodes", "archived"):
        clauses.append("archived = 0")
    return " AND ".join(clauses)


def embedding_coverage(conn: sqlite3.Connection) -> dict:
    """How much of the embeddable store dense retrieval can actually reach.

    Returns {backend, active_model, total, by_model, covered, dark, split}:
      total     embeddable live nodes (index_node's own rules)
      by_model  {model: rows} counting ONLY rows on those nodes
      covered   rows under the active backend's model (what vector_search reads)
      dark      total - covered: nodes dense retrieval cannot see (stale-model
                rows OR no row at all); 0 on the bm25 backend, where none is expected
      split     more than one model over live nodes -- the 2026-09 bug shape
    Read-only: two indexed SELECTs, no migrate, no model load, no network.
    """
    from memsom.retrieval import embed as memsom_embed
    b = memsom_embed.backend(conn)
    active = memsom_embed.active_model_name(conn)
    if not memsom_schema.table_exists(conn, "nodes"):
        return {"backend": b, "active_model": active, "total": 0, "by_model": {},
                "covered": 0, "dark": 0, "split": False}
    where = _embeddable_clauses(conn)
    total = conn.execute(f"SELECT COUNT(*) FROM nodes WHERE {where}").fetchone()[0]
    by_model = {}
    if memsom_schema.table_exists(conn, "embeddings"):
        for model, n in conn.execute(
                "SELECT e.model, COUNT(*) FROM embeddings e JOIN nodes n ON n.id = e.node_id"
                f" WHERE {where} GROUP BY e.model"):
            by_model[model] = n
    covered = by_model.get(active, 0) if b != "bm25" else 0
    dark = (total - covered) if b != "bm25" else 0
    return {"backend": b, "active_model": active, "total": total, "by_model": by_model,
            "covered": covered, "dark": max(0, dark), "split": len(by_model) > 1}


def retrieval_warnings(conn: sqlite3.Connection) -> list:
    """Short, plain lines for an AI/operator surface (prompt hook, features,
    doctor, the retrieve CLI/MCP tool). Empty on a healthy single-backend
    store whose query encoder answered recently. Read-only and cheap."""
    out = []
    cov = embedding_coverage(conn)
    # bm25 expects no vectors at all: leftover rows under any number of models
    # are inert there, not a degradation.
    if cov["dark"] or (cov["split"] and cov["backend"] != "bm25"):
        stored = ", ".join(f"{m}={n}" for m, n in sorted(cov["by_model"].items())) or "none"
        out.append(
            f"⚠️ RETRIEVAL DEGRADED: {cov['dark']}/{cov['total']} memory nodes are "
            f"dense-dark (active backend {cov['backend']} reads model="
            f"{cov['active_model'] or 'none'}; stored: {stored}); run `memsom reindex`")
    last = last_query_fallback(conn)
    if last and (last["age_s"] is None or last["age_s"] <= QUERY_DEGRADED_RECENT_S):
        out.append(
            f"⚠️ RETRIEVAL DEGRADED: query encoder unreachable at "
            f"{last['recorded_at']} → BM25-only ({last['reason']})")
    return out


def index_node(conn: sqlite3.Connection, nid: int) -> bool:
    """Rebuild postings + docstats for *nid*.  If Ollama is reachable, also
    store an embedding.  Ollama unreachable -> BM25 index is still built.

    No-op for agent-derived or tombstoned nodes (they are not source nodes).
    Returns True if the node was indexed, False if it was skipped (and any
    stale postings purged) — index_all counts on this.
    """
    migrate(conn)
    row = conn.execute(
        "SELECT content, channel, tombstoned FROM nodes WHERE id = ?", (nid,)
    ).fetchone()
    if row is None:
        return False  # unknown id; caller's problem
    content, channel, tombstoned = row
    if tombstoned or channel == "agent-derived":
        deindex_node(conn, nid)  # purge any stale postings for a now-dead node
        return False  # only index live source nodes
    # F-15: never (re)index a redacted node — purge its postings instead so a
    # reindex pass can't resurrect stale term frequencies from wiped content.
    if memsom_schema.column_exists(conn, "nodes", "redacted"):
        r = conn.execute("SELECT redacted FROM nodes WHERE id = ?", (nid,)).fetchone()
        if r and r[0]:
            deindex_node(conn, nid)
            return False
    # COMPACT/F-15: an ARCHIVED (compacted-away) episode is pool-excluded at read
    # time (taint_filter_clauses forces archived = 0), yet if left in postings/
    # docstats it still pollutes the BM25 CORPUS STATS — N, avgdl, df/idf are
    # computed over docstats, so a stale archived chunk skews every score and
    # leaves "duplicate" term mass behind the summary. Treat archived exactly like
    # redacted: purge it from all retrieval structures and never re-index it. This
    # also makes a full `reindex` idempotent w.r.t. compaction (it can't resurrect
    # an archived episode the way index_all's old query would have).
    if memsom_schema.column_exists(conn, "nodes", "archived"):
        r = conn.execute("SELECT archived FROM nodes WHERE id = ?", (nid,)).fetchone()
        if r and r[0]:
            deindex_node(conn, nid)
            return False

    tokens = tokenize(content)
    length = len(tokens)

    # Term-frequency map
    tf_map = {}
    for t in tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    with conn:
        # Remove stale postings / docstats for this node first (full rebuild)
        conn.execute("DELETE FROM postings WHERE node_id = ?", (nid,))
        conn.execute("DELETE FROM docstats WHERE node_id = ?", (nid,))

        # Insert new postings
        if tf_map:
            conn.executemany(
                "INSERT INTO postings(term, node_id, tf) VALUES (?, ?, ?)",
                [(term, nid, tf) for term, tf in tf_map.items()],
            )
        conn.execute(
            "INSERT OR REPLACE INTO docstats(node_id, length) VALUES (?, ?)",
            (nid, length),
        )

    # Optional vector — backend-dispatched; degrade CLEARLY (MS-32), not
    # silently, on any failure: a stderr warning (once per process, same
    # shape as embed.py's bge fallback warning) and a row in
    # retrieval_degraded so a later reindex can find it again once the
    # embedder recovers. index_node still returns True -- BM25 IS indexed,
    # which is the contract index_all counts on -- the degradation is
    # tracked separately, not folded into the boolean.
    from memsom.retrieval import embed as memsom_embed
    # conn-aware: a process with no MEMDAG_EMBED_BACKEND (the Stop-hook
    # importer) resolves to the STORE's pinned backend, not the compiled-in
    # default -- writing under a different model than the reader filters on
    # is exactly the silent split this closes.
    b = memsom_embed.backend(conn)
    if b == "bm25":
        _clear_degraded(conn, nid)
        return True  # BM25-only by request; no vectors stored
    if b == "bge-m3":
        # bge NEVER falls back to the Ollama/nomic MODEL: a nomic row is tagged
        # model='nomic-embed-text' and the bge reader filters WHERE model='bge-m3',
        # so it would be unreadable dead weight. If bge can't embed (no local
        # supervisor AND no torch, or the encode failed), stay BM25-only and TRACK
        # it (MS-32) so a later reindex retries once a path is available.
        if memsom_embed.bge_usable():
            try:
                enc = memsom_embed.encode_doc(content)
                if enc is not None:
                    memsom_embed.store_bge(conn, nid, enc)
                    _clear_degraded(conn, nid)
                    _pin_if_unpinned(conn, b)
                    return True
            # FAILOPEN: swallows -> falls to _record_degraded below; a bge store error must not block ingest (BM25 is already built).
            except Exception as exc:
                _warn_embed_fallback(exc)
        _record_degraded(conn, nid, "bge embed unavailable; BM25-only")
        return True

    # Ollama dense path (backend == 'ollama').
    try:
        vec = _call_ollama_embed(content)
        blob = _vec_to_blob(vec)
        model = _embed_model()
        dim = len(vec)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec)"
                " VALUES (?, ?, ?, ?)",
                (nid, model, dim, blob),
            )
        _clear_degraded(conn, nid)
        _pin_if_unpinned(conn, b)
    # FAILOPEN: warns once and records the node as degraded -- Ollama down/model not pulled/network error (MS-32: the DEFAULT backend) must not block ingest; a later recovery pass can find it via _record_degraded.
    except Exception as exc:
        _warn_embed_fallback(exc)
        _record_degraded(conn, nid, f"ollama embed failed: {exc}")
    return True


def _pin_if_unpinned(conn: sqlite3.Connection, b: str) -> None:
    """First vector write adopts the backend for the store (see pin_backend).
    A pinned store is left alone: switching is `reindex`'s job, not ingest's."""
    try:
        if pinned_backend(conn) is None:
            pin_backend(conn, b)
    # FAILOPEN: allowed -- the pin is a safety net around the write that just succeeded, never a reason to fail it.
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Deindex (F-15): remove a node from all retrieval structures
# ---------------------------------------------------------------------------

def deindex_node(conn: sqlite3.Connection, nid: int) -> None:
    """Remove *nid* from postings, docstats, and embeddings.

    Called from the redact cascade (and from index_node for dead/redacted nodes)
    so a redacted/tombstoned node can never resurface via BM25 or vector ranking.
    No-op (never creates tables) if the retrieval schema is absent.
    """
    if not memsom_schema.table_exists(conn, "postings"):
        return
    from memsom.retrieval import embed as memsom_embed
    with conn:
        conn.execute("DELETE FROM postings WHERE node_id = ?", (nid,))
        conn.execute("DELETE FROM docstats WHERE node_id = ?", (nid,))
        if memsom_schema.table_exists(conn, "embeddings"):
            conn.execute("DELETE FROM embeddings WHERE node_id = ?", (nid,))
        # Purge the bge side-tables in the SAME txn (deindex_bge is bare-exec +
        # table-guarded). Keeps the invariant: anything that drops `embeddings`
        # drops sparse_vecs + colbert_vecs too — no orphaned vectors can resurface.
        memsom_embed.deindex_bge(conn, nid)
        # MS-32: a deindexed node is no longer a re-index candidate either.
        if memsom_schema.table_exists(conn, "retrieval_degraded"):
            conn.execute("DELETE FROM retrieval_degraded WHERE node_id = ?", (nid,))


# ---------------------------------------------------------------------------
# Event subscriptions -- PLAN.md Sec1.5/Sec2.3: "the try-wrapped upward
# imports become subscribers". integrity/ingest.py and integrity/redact.py no
# longer import this module directly; they emit and this module listens,
# which is what keeps integrity (rank 2) from importing retrieval (rank 3).
# ---------------------------------------------------------------------------

def _on_node_ingested(conn, node_id):
    index_node(conn, node_id)


def _on_node_redacted(conn, node_id):
    deindex_node(conn, node_id)


memsom_events.subscribe("node_ingested", _on_node_ingested)
memsom_events.subscribe("node_redacted", _on_node_redacted)


# ---------------------------------------------------------------------------
# Bulk reindex
# ---------------------------------------------------------------------------

def index_all(conn: sqlite3.Connection) -> int:
    """Index every live source node (channel != 'agent-derived').

    Returns the count of nodes indexed.
    """
    migrate(conn)
    # Exclude archived at the query level too (index_node already guards, but the
    # bulk pass should not even visit compacted-away episodes — keeps the count
    # honest and the corpus = exactly the live, retrievable source pool).
    archived_clause = ""
    if memsom_schema.column_exists(conn, "nodes", "archived"):
        archived_clause = " AND archived = 0"
    rows = conn.execute(
        "SELECT id FROM nodes WHERE tombstoned = 0 AND channel != 'agent-derived'"
        + archived_clause + " ORDER BY id"
    ).fetchall()
    # Count what index_node actually indexed, not what the query returned: the
    # visit itself must still happen for redacted nodes (the F-15 purge path),
    # but a purged-and-skipped node is not an "indexed" one.
    n = 0
    for (nid,) in rows:
        if index_node(conn, nid):
            n += 1
    return n


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def bm25(conn: sqlite3.Connection, query: str, k: int = 8) -> list:
    """Classic BM25 over postings/docstats.

    Returns [(nid, score)] sorted by score descending, up to k results.
    Returns [] if no postings exist yet (index_all not called).
    """
    migrate(conn)
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    # Corpus stats
    N = conn.execute("SELECT COUNT(*) FROM docstats").fetchone()[0]
    if N == 0:
        return []
    avg_row = conn.execute("SELECT AVG(length) FROM docstats").fetchone()
    avgdl = avg_row[0] if avg_row[0] is not None else 0.0
    if avgdl == 0.0:
        return []

    # Accumulate per-doc BM25 scores
    scores = {}
    for term in query_tokens:
        # df = number of documents containing this term
        df_row = conn.execute(
            "SELECT COUNT(*) FROM postings WHERE term = ?", (term,)
        ).fetchone()
        df = df_row[0]
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # Fetch tf + doc length for all docs containing this term
        rows = conn.execute(
            "SELECT p.node_id, p.tf, d.length"
            " FROM postings p JOIN docstats d ON d.node_id = p.node_id"
            " WHERE p.term = ?",
            (term,),
        ).fetchall()
        for nid, tf, dl in rows:
            tf_norm = (tf * (K1 + 1.0)) / (
                tf + K1 * (1.0 - B + B * dl / avgdl)
            )
            scores[nid] = scores.get(nid, 0.0) + idf * tf_norm

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:k]


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

def vector_search(conn: sqlite3.Connection, query: str, k: int = 8) -> list:
    """Embed query via the active backend, brute-force cosine over the stored
    DENSE embeddings of that SAME backend.

    Returns [(nid, score)] sorted desc, up to k.
    Returns [] if no embeddings stored or the embedder is down — never crashes.

    Backend dispatch (MEMDAG_EMBED_BACKEND):
      ollama  -> Ollama dense (nomic), model='nomic-embed-text'
      bge-m3  -> FlagEmbedding dense, model='bge-m3'
      bm25    -> [] (BM25-only)

    Collision fix: read ONLY the active backend's rows (`WHERE model = ?`). Two
    backends with different dims (768 nomic vs 1024 bge) can coexist in the
    embeddings table; without this filter _cosine() silently scores every
    mismatched-dim row as 0.0, corrupting ranking after a backend switch.
    """
    global _LAST_QUERY_FALLBACK
    migrate(conn)
    from memsom.retrieval import embed as memsom_embed
    b = memsom_embed.backend(conn)
    if b == "bm25":
        return []
    if b == "bge-m3" and not memsom_embed.dense_enabled():
        return []  # dense signal toggled off on the bge path
    # Every fallback below is RECORDED (sentinel row + in-process reason), so a
    # dead encoder is never again a silent "BM25 felt worse for weeks".
    try:
        if b == "bge-m3":
            if not memsom_embed.bge_usable():
                # bge selected but no encode path -> BM25-only, never a nomic mix
                _record_query_degraded(
                    conn, "bge-m3 has no encode path (supervisor down, FlagEmbedding "
                          "not importable)")
                return []
            enc = memsom_embed.encode_query(query)
            if enc is None:
                _record_query_degraded(conn, "bge-m3 query encode failed")
                return []
            q_vec = enc["dense"]
        else:
            q_vec = _call_ollama_embed(query)
    # FAILOPEN: swallows and returns [] -- embedder down, caller's fusion falls back to BM25-only (recorded).
    except Exception as exc:
        _record_query_degraded(conn, f"{b} query encode raised {type(exc).__name__}: {exc}")
        return []
    _LAST_QUERY_FALLBACK = None

    # Load this backend's stored embeddings ONLY (dim-collision fix).
    rows = conn.execute(
        "SELECT node_id, vec FROM embeddings WHERE model = ?",
        (memsom_embed.active_model_name(conn),),
    ).fetchall()
    if not rows:
        return []

    scored = []
    for nid, blob in rows:
        d_vec = _blob_to_vec(blob)
        sim = _cosine(q_vec, d_vec)
        scored.append((nid, sim))

    scored.sort(key=lambda x: -x[1])
    return scored[:k]


# ---------------------------------------------------------------------------
# Sparse (BGE-M3 learned lexical) retrieval — bge backend only
# ---------------------------------------------------------------------------

def sparse_search(conn: sqlite3.Connection, query: str, k: int = 8) -> list:
    """Score the query's BGE-M3 sparse (lexical) weights against stored doc
    sparse weights by shared-token dot product.

    Returns [(nid, score)] sorted desc, up to k. Returns [] for any non-bge
    backend, when bge is unavailable, or when the encoder fails — so it
    contributes nothing to fusion outside the bge path (degrades cleanly).
    """
    migrate(conn)
    from memsom.retrieval import embed as memsom_embed
    if (memsom_embed.backend(conn) != "bge-m3" or not memsom_embed.sparse_enabled()
            or not memsom_embed.bge_usable()):
        return []
    enc = memsom_embed.encode_query(query)
    if enc is None:
        return []
    q_sparse = enc["sparse"]
    if not memsom_schema.table_exists(conn, "sparse_vecs"):
        return []
    rows = conn.execute(
        "SELECT node_id, weights_json FROM sparse_vecs WHERE model = ?",
        (memsom_embed.active_model_name(conn),),
    ).fetchall()
    scored = []
    for nid, wj in rows:
        try:
            d_sparse = json.loads(wj)
        except (json.JSONDecodeError, TypeError):
            continue
        s = memsom_embed.sparse_dot(q_sparse, d_sparse)
        if s > 0.0:
            scored.append((nid, s))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


# ---------------------------------------------------------------------------
# ColBERT late-interaction RE-RANKER (bge backend only)
# ---------------------------------------------------------------------------

def colbert_rerank(conn: sqlite3.Connection, query: str, candidate_ids: list) -> list:
    """Re-order an ALREADY-POOL-GATED candidate id list by ColBERT MaxSim.

    SECURITY: this only re-orders the ids it is given; it has NO membership
    power. `candidate_ids` must already be the taint/clearance-filtered fused
    window (see retrieve()), so a crafted strong-match per-token vector on an
    excluded node can never enter here.

    Returns [(nid, score)] for EVERY candidate (a candidate with no colbert
    vectors keeps score 0.0 and, via the stable sort, its incoming fused order).
    For any non-bge backend / bge-unavailable, returns the candidates unchanged
    (all score 0.0) so the caller's fused order is preserved.
    """
    if not candidate_ids:
        return []
    from memsom.retrieval import embed as memsom_embed
    if (memsom_embed.backend(conn) != "bge-m3" or not memsom_embed.colbert_enabled()
            or not memsom_embed.bge_usable()):
        return [(nid, 0.0) for nid in candidate_ids]
    enc = memsom_embed.encode_query(query)
    if enc is None or not memsom_schema.table_exists(conn, "colbert_vecs"):
        return [(nid, 0.0) for nid in candidate_ids]
    q_colbert = enc["colbert"]
    placeholders = ",".join("?" * len(candidate_ids))
    rows = conn.execute(
        "SELECT node_id, n_tokens, dim, vecs FROM colbert_vecs"
        f" WHERE model = ? AND node_id IN ({placeholders})",
        [memsom_embed.active_model_name(conn)] + list(candidate_ids),
    ).fetchall()
    have = {}
    for nid, n_tokens, dim, blob in rows:
        d_colbert = memsom_embed.blob_to_colbert(blob, n_tokens, dim)
        have[nid] = memsom_embed.colbert_maxsim(q_colbert, d_colbert)
    # Preserve incoming (fused) order for ties / missing-vector candidates: build
    # scores in candidate order, then stable-sort by score descending.
    scored = [(nid, have.get(nid, 0.0)) for nid in candidate_ids]
    scored.sort(key=lambda x: -x[1])
    return scored


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(*rank_lists, rrf_c: int = 60) -> list:
    """Reciprocal Rank Fusion of N ranked lists (>=1).

    Each list is [(nid, score)]; positions are 1-indexed for RRF.
    Returns merged [(nid, rrf_score)] sorted by rrf_score descending.

    Variadic so bm25 + dense + sparse fuse in one call. `rrf_c` is keyword-only
    (after `*rank_lists`) so a third rank list can't be misread as the constant.
    Two-list callers `_rrf_fuse(a, b)` are numerically identical to the original.
    """
    rrf = {}
    for ranks in rank_lists:
        for rank, (nid, _) in enumerate(ranks, start=1):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (rrf_c + rank)
    return sorted(rrf.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Pool filter (mirrors memsom_retrieve._build_pool)
# ---------------------------------------------------------------------------

def _build_retrieve_pool(
    conn: sqlite3.Connection,
    clearance: int,
    min_integrity,
    exclude_quarantined: bool,
    exclude_redacted: bool,
) -> set:
    """Return set of nids passing all pool filters.

    Taint dimensions come from memsom_schema.taint_filter_clauses — the ONE
    shared untainted-pool primitive (same clauses as memsom_retrieve._build_pool
    and memsom_anticipatory._untainted_clauses, by construction).

    F-15 fail-safe: tombstoned, redacted, and archived are ALWAYS excluded
    regardless of the exclude_* flags (the primitive does not let a caller
    widen them). The flags may only widen the quarantine dimension; they must
    never widen a pool far enough to leak a node's liveness.
    """
    clauses, params = memsom_schema.taint_filter_clauses(
        conn, clearance=clearance,
        include_quarantined=not exclude_quarantined)
    clauses.append("channel != 'agent-derived'")
    if min_integrity is not None:
        clauses.append("label >= ?")
        params.append(min_integrity)

    sql = "SELECT id FROM nodes WHERE " + " AND ".join(clauses) + " ORDER BY id"
    rows = conn.execute(sql, params).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    conn: sqlite3.Connection,
    query: str,
    k: int = 8,
    clearance="topsecret",
    min_integrity=None,
    exclude_quarantined: bool = True,
    exclude_redacted: bool = True,
) -> list:
    """Hybrid BM25+vector retrieve with v1 pool filters.

    Returns up to k rows as (id, content, channel, label, source_ref) tuples,
    ranked by RRF-fused BM25 + vector scores, filtered to the live pool.

    Ollama down -> BM25-only, no crash.
    """
    migrate(conn)
    # RETRIEVE-1: a negative k makes len(pool)+k shrink the candidate window and
    # fused[:k] drop the top hits / return []. Clamp at the API boundary.
    k = max(0, int(k))
    if k == 0:
        return []
    clearance_int = memsom_confid.parse_conf(clearance)

    # Build the pool
    pool = _build_retrieve_pool(
        conn, clearance_int, min_integrity, exclude_quarantined, exclude_redacted
    )
    if not pool:
        return []

    # BM25 + dense + (bge) sparse over the FULL index, then intersect with pool.
    # Scan the whole index (docstats row count), not len(pool)+k: quarantined /
    # above-clearance nodes stay indexed and can outrank pool members in raw
    # BM25/vector — with a small pool the truncated scan could lose every pool
    # member before the intersection. Same fix as retrieve_graph below.
    n_idx = conn.execute("SELECT COUNT(*) FROM docstats").fetchone()[0]
    scan_k = max(n_idx, k)
    bm25_all = bm25(conn, query, k=scan_k)
    vec_all = vector_search(conn, query, k=scan_k)
    sparse_all = sparse_search(conn, query, k=scan_k)  # [] off the bge path

    # Filter each ranked list to pool members only (the trust intersection).
    bm25_filtered = [(nid, s) for nid, s in bm25_all if nid in pool]
    vec_filtered = [(nid, s) for nid, s in vec_all if nid in pool]
    sparse_filtered = [(nid, s) for nid, s in sparse_all if nid in pool]

    # Fuse. Off the bge path sparse_filtered is [] -> numerically identical to
    # the original two-list RRF (pure-zero pool members are omitted by design;
    # retrieval is about relevance, not "give me all nodes").
    fused = _rrf_fuse(bm25_filtered, vec_filtered, sparse_filtered)

    # ColBERT late-interaction re-rank, BOUNDED: take the top-N fused candidates
    # (already pool-gated) and let MaxSim reorder ONLY those before the final
    # top-k slice — cost is N×tokens, not corpus×tokens. No-op off the bge path,
    # where it preserves the fused order exactly.
    from memsom.retrieval import embed as memsom_embed
    if (memsom_embed.backend(conn) == "bge-m3" and memsom_embed.colbert_enabled()
            and memsom_embed.bge_usable()):
        cand_n = min(memsom_embed.colbert_candidates(), len(fused))
        cand_ids = [nid for nid, _ in fused[:cand_n]]
        reranked = colbert_rerank(conn, query, cand_ids)
        top_nids = [nid for nid, _ in reranked[:k]]
        # The window is a RE-RANK bound, not a result cap: if it is narrower
        # than k, backfill from the fused tail in fused order so the caller
        # still gets k results (the tail just isn't MaxSim-reordered).
        if len(top_nids) < k:
            seen = set(top_nids)
            top_nids += [nid for nid, _ in fused
                         if nid not in seen][: k - len(top_nids)]
    else:
        top_nids = [nid for nid, _ in fused[:k]]

    # Fetch rows in fused order
    if not top_nids:
        return []

    nid_to_row = {}
    placeholders = ",".join("?" * len(top_nids))
    rows = conn.execute(
        f"SELECT id, content, channel, label, source_ref FROM nodes WHERE id IN ({placeholders})",
        top_nids,
    ).fetchall()
    for row in rows:
        nid_to_row[row[0]] = row

    result = [nid_to_row[nid] for nid in top_nids if nid in nid_to_row]
    return result


# ---------------------------------------------------------------------------
# Graph-expanded retrieve (GraphRAG-lite over rel_edges / wikilinks)
# ---------------------------------------------------------------------------

def retrieve_graph(
    conn: sqlite3.Connection,
    query: str,
    k: int = 8,
    clearance="topsecret",
    min_integrity=None,
    exclude_quarantined: bool = True,
    exclude_redacted: bool = True,
    hops: int = 1,
) -> list:
    """retrieve() re-ranked by the rel_edges (wikilink) graph.

    A node that is ALREADY relevant to the query but ranks just below the top-k
    cutoff gets promoted if it is linked (within *hops*) to a strong seed hit.
    The graph only RE-ORDERS relevant nodes — it never injects a zero-relevance
    neighbor, because compose() force-includes every pool member (one bullet
    minimum) and an off-topic node would spray noise into the answer.

    Trust is enforced twice, by reuse not reinvention: a candidate must be in the
    `_build_retrieve_pool` set (taint / clearance / min_integrity / non-derived)
    AND survive `memsom_relate.neighborhood` (dead / quarantined / redacted /
    archived / above-clearance all filtered, widest-path integrity floor). So a
    crafted edge to a tainted or above-clearance node cannot leak it into the
    answer pool.

    Empty rel_edges (or Ollama down) degrades to exactly retrieve()'s ranking.
    Returns up to k rows as (id, content, channel, label, source_ref), ranked.
    """
    from memsom.retrieval import relate as memsom_relate
    migrate(conn)
    memsom_relate.migrate(conn)  # idempotent; ensure rel_edges exists
    k = max(0, int(k))
    if k == 0:
        return []
    clearance_int = memsom_confid.parse_conf(clearance)

    pool = _build_retrieve_pool(
        conn, clearance_int, min_integrity, exclude_quarantined, exclude_redacted
    )
    if not pool:
        return []

    # Base relevance over the WHOLE pool (not just top-k) so a sub-cutoff but
    # scored node remains a promotion candidate. Scan the FULL index, not just
    # len(pool): excluded nodes (above-clearance / tainted) can outrank pool
    # members in raw BM25/vector and consume the top-k slots, so a too-small
    # scan would silently drop a relevant pool member from `base` before it ever
    # gets a chance to be graph-promoted. docstats holds the live source count.
    n_idx = conn.execute("SELECT COUNT(*) FROM docstats").fetchone()[0]
    scan_k = max(n_idx, k)
    bm25_all = bm25(conn, query, k=scan_k)
    vec_all = vector_search(conn, query, k=scan_k)
    sparse_all = sparse_search(conn, query, k=scan_k)  # [] off the bge path
    bm25_f = [(nid, s) for nid, s in bm25_all if nid in pool]
    vec_f = [(nid, s) for nid, s in vec_all if nid in pool]
    sparse_f = [(nid, s) for nid, s in sparse_all if nid in pool]
    # Sparse joins the fuse; ColBERT is deliberately NOT layered here — the graph
    # re-rank below is itself a re-ordering, and stacking a second one risks a
    # double-reorder. Off the bge path sparse_f is [] -> identical to the old fuse.
    base = dict(_rrf_fuse(bm25_f, vec_f, sparse_f))  # {nid: rrf_score}, scored pool members
    if not base:
        return []

    # Seeds = what plain retrieve would surface (top-k by base score).
    seeds = [nid for nid, _ in sorted(base.items(), key=lambda x: -x[1])[:k]]

    # Graph boost: a relevant node linked to a strong seed is lifted toward the
    # cutoff. The relevance gate (base.get(n) > 0) is what stops an off-topic
    # neighbor from being added; the pool/neighborhood intersection is the trust
    # gate that stops a tainted/above-clearance neighbor.
    boost = {}
    for s in seeds:
        s_score = base.get(s, 0.0)
        if s_score <= 0.0:
            continue
        for nb in memsom_relate.neighborhood(
            conn, s, hops=hops, min_integrity=0, clearance=clearance_int
        ):
            n = nb["id"]
            if n in pool and base.get(n, 0.0) > 0.0:
                boost[n] = boost.get(n, 0.0) + s_score * (GRAPH_DECAY ** nb["hops"])

    # Re-rank over the SAME candidate set (base's keys); graph only re-orders.
    final = sorted(
        base.keys(), key=lambda nid: -(base[nid] + boost.get(nid, 0.0))
    )
    top_nids = final[:k]
    if not top_nids:
        return []

    placeholders = ",".join("?" * len(top_nids))
    rows = conn.execute(
        f"SELECT id, content, channel, label, source_ref FROM nodes WHERE id IN ({placeholders})",
        top_nids,
    ).fetchall()
    nid_to_row = {row[0]: row for row in rows}
    return [nid_to_row[nid] for nid in top_nids if nid in nid_to_row]


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def _cmd_retrieve(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        results = retrieve(
            conn,
            args.query,
            k=args.k,
            clearance=args.clearance,
        )
        # AI-facing degraded signal on the surface Claude actually reads (the
        # MCP `retrieve` tool runs this command and returns its stdout).
        for line in _retrieve_warnings_after_query(conn):
            print(line)
        if not results:
            print("[memsom-retrieve] no results")
            return
        for row in results:
            nid, content, channel, label, source_ref = row
            ref = source_ref or "(stated directly)"
            # Read-time fact resolution (docs/facts-design.md Phase 3): current
            # value, with drift shown against the version live when this memory
            # was written (as_of = the memory's created_at). Resolve BEFORE
            # snippet() so truncation can't cut a substituted value in half.
            if "[[fact_" in (content or ""):
                created = conn.execute(
                    "SELECT created_at FROM nodes WHERE id = ?", (nid,)).fetchone()
                # bridge.facts (rank 7) cannot be imported directly from
                # here (rank 3); routed through kernel.events (rank 0). A
                # missing subscriber degrades to "text unchanged", matching
                # resolve_ref's own designed behaviour for an unresolvable
                # reference (module docstring, bridge/facts.py).
                _fact_result = {"text": content}
                memsom_events.emit(
                    "resolve_fact_refs", conn=conn, text=content,
                    as_of=created[0] if created else None, result=_fact_result)
                content = _fact_result["text"]
            print(f"[{nid}] {channel:<13} integrity={memsom.NAME[label]:<13} {ref}")
            print(f'      "{memsom.snippet(content)}"')
    finally:
        conn.close()


def _retrieve_warnings_after_query(conn: sqlite3.Connection) -> list:
    """retrieval_warnings() plus THIS query's own fallback (which the stored
    trail already reflects when the write succeeded; the in-process reason
    covers a read-only connection where it could not be written)."""
    lines = retrieval_warnings(conn)
    reason = last_query_fallback_reason()
    if reason and not any("query encoder unreachable" in l for l in lines):
        lines.append(f"⚠️ RETRIEVAL DEGRADED: query encoder unreachable → "
                     f"BM25-only for this query ({reason})")
    return lines


def _cmd_reindex(args):
    conn = memsom.get_connection()
    try:
        migrate(conn)
        from memsom.retrieval import embed as memsom_embed
        n = index_all(conn)
        degraded = degraded_nodes(conn)
        # MS-32: index_all's old line printed identically whether every
        # node got a vector or none did. Report the degraded count
        # separately -- 0 is silent, non-zero says exactly what to expect.
        if degraded:
            print(f"indexed {n} source node(s), {len(degraded)} BM25-only "
                  f"(vector embed degraded -- see stderr; will retry on next "
                  f"reindex)")
        else:
            print(f"indexed {n} source node(s)")
        # A full pass under backend X makes X the store's backend: pin it so a
        # process without the env var keeps writing X (the split fix), then
        # report what dense retrieval can now reach.
        b = memsom_embed.backend(conn)
        pin_backend(conn, b)
        cov = embedding_coverage(conn)
        stored = ", ".join(f"{m}={c}" for m, c in sorted(cov["by_model"].items())) or "none"
        print(f"backend pinned: {b}; dense coverage {cov['covered']}/{cov['total']} "
              f"(stored: {stored})")
        for line in retrieval_warnings(conn):
            print(line)
    finally:
        conn.close()
    # VRAM hygiene: a batch reindex is the canonical place a 2.2GB bge model gets
    # loaded. Unload it after if asked, so it doesn't evict the daily driver.
    if (memsom_tuning.resolve("retrieval.bge_unload") or "").strip().lower() in ("1", "true", "yes", "on"):
        from memsom.retrieval import embed as memsom_embed
        memsom_embed.unload()


# ---------------------------------------------------------------------------
# register / main
# ---------------------------------------------------------------------------

def register(subparsers) -> None:
    """Mount retrieve and reindex sub-commands onto *subparsers*."""
    p_ret = subparsers.add_parser("retrieve", help="hybrid BM25+vector search")
    p_ret.add_argument("query")
    p_ret.add_argument("--k", type=int, default=8, help="max results (default 8)")
    p_ret.add_argument("--clearance", default="topsecret",
                       help="confidentiality clearance level (default: topsecret)")
    p_ret.set_defaults(func=_cmd_retrieve)

    p_ri = subparsers.add_parser("reindex", help="rebuild BM25 postings for all source nodes")
    p_ri.set_defaults(func=_cmd_reindex)


def migrate_and_register(conn, subparsers=None) -> None:
    """Convenience: migrate + optionally register.  Matches CLI pattern."""
    migrate(conn)


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_retrieve")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Enhanced-ask pool (moved from interface/cli.py, Phase 7 -- bridge/obsidian.py
# needs the identical pool for its vault-answer compose path and cannot import
# interface/, rank 8, from bridge's rank 7).
# ---------------------------------------------------------------------------

def _build_pool(conn, clearance_name):
    """Build the filtered source pool for the enhanced ask.

    Filters applied (all stacked on top of the frozen live_sources behaviour):
      - tombstoned = 0
      - channel != 'agent-derived'
      - status != 'quarantined'   (quarantine module layer)
      - redacted = 0              (redact module layer)
      - conf_label <= clearance   (Bell-LaPadula no-read-up)

    Returns list of rows shaped like memsom.live_sources:
    (id, content, channel, label, source_ref).

    Taint dimensions come from memsom_schema.taint_filter_clauses — the ONE
    shared untainted-pool primitive (same clauses as
    memsom_retrieve._build_retrieve_pool and
    memsom_anticipatory._untainted_clauses, by construction).
    """
    clearance = memsom_confid.parse_conf(clearance_name)
    clauses, params = memsom_schema.taint_filter_clauses(conn, clearance=clearance)
    return conn.execute(
        "SELECT id, content, channel, label, source_ref FROM nodes"
        " WHERE channel != 'agent-derived' AND " + " AND ".join(clauses) +
        " ORDER BY label DESC, id ASC",
        params
    ).fetchall()
