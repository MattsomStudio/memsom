"""memsom_embed — opt-in embedding-backend dispatch + BGE-M3 triple fusion.

memsom's retrieval is BM25 (stdlib) plus an OPTIONAL dense vector layer. This
module adds a second, richer backend — **BGE-M3 via FlagEmbedding** — that emits
three complementary signals in ONE encode call:

  - dense     1024-dim sentence vector  (cosine)         -> embeddings table
  - sparse    learned lexical weights   (dot)            -> sparse_vecs table
  - colbert   per-token late-interaction (MaxSim rerank) -> colbert_vecs table

Selection is by env (or the CLI `--embed-backend` flag):

  MEMDAG_EMBED_BACKEND = ollama | bge-m3 | bm25      (default: ollama)

This module is HEAVY-DEPENDENCY-OPTIONAL by construction. FlagEmbedding + torch
+ numpy are imported LAZILY, only inside the bge code path. If any are missing
(the CI box has no GPU), `bge_available()` returns False and every caller falls
back to the existing Ollama / BM25 path — never a crash. Nothing here is
imported at module top beyond stdlib + memsom_schema, so importing memsom_embed
is free and the frozen import graph is untouched.

SECURITY NOTE: this module only computes/stores/scores embeddings. It has NO
membership authority. Which nodes may surface is decided upstream by
memsom_schema.taint_filter_clauses (the pool gate). Sparse + ColBERT signals
only RE-ORDER an already-pool-gated candidate set; a crafted strong-match vector
on an above-clearance or tainted node is excluded at the gate and never reaches
the re-ranker. See memsom_retrieve.retrieve / colbert_rerank for the proof.

Public API
----------
backend() -> str
active_model_name() -> str
bge_available() -> bool
encode_doc(text) -> dict | None        # {'dense','sparse','colbert'}
encode_query(text) -> dict | None
migrate(conn)                          # sparse_vecs + colbert_vecs
store_bge(conn, nid, enc)
deindex_bge(conn, nid)                 # bare execs; safe inside a txn
sparse_dot(q_sparse, d_sparse) -> float
colbert_maxsim(q_colbert, d_colbert) -> float
colbert_to_blob(mat) -> bytes         # fp16 LE row-major
blob_to_colbert(blob, n_tokens, dim) -> list
unload()                               # drop the model, free VRAM
"""

import os
import struct
import sqlite3

import memsom_schema

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "bge-m3", "bm25")
BGE_MODEL_NAME = "bge-m3"          # the `model` tag stored alongside bge vectors
BGE_HF_REPO = "BAAI/bge-m3"        # the FlagEmbedding load path
BGE_DENSE_DIM = 1024
DEFAULT_MAXLEN = 512               # passage/query token cap (FlagEmbedding default)
DEFAULT_COLBERT_CANDIDATES = 100   # ColBERT re-rank window (see memsom_retrieve)

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def backend() -> str:
    """Active embedding backend from MEMDAG_EMBED_BACKEND.

    Unknown / unset -> 'ollama' (back-compat: the historical default path).
    """
    raw = (os.environ.get("MEMDAG_EMBED_BACKEND") or "").strip().lower()
    return raw if raw in VALID_BACKENDS else DEFAULT_BACKEND


def active_model_name() -> str:
    """The `model` tag the active backend writes/reads in the vector tables.

    This is the load-bearing key for the dim-collision fix: vector_search filters
    `WHERE model = active_model_name()`, so 768-dim nomic rows and 1024-dim bge
    rows never get cosine'd against each other.

    bm25 -> '' (no model) so a `WHERE model=''` matches nothing -> BM25-only.
    """
    b = backend()
    if b == "bge-m3":
        return BGE_MODEL_NAME
    if b == "bm25":
        return ""
    # ollama: reuse retrieve's resolver so there's one source of truth.
    import memsom_retrieve  # lazy: avoid import cycle at module load
    return memsom_retrieve._embed_model()


def colbert_candidates() -> int:
    """Re-rank window size from MEMDAG_COLBERT_CANDIDATES (default 100)."""
    raw = os.environ.get("MEMDAG_COLBERT_CANDIDATES")
    if raw is None or not raw.strip():
        return DEFAULT_COLBERT_CANDIDATES
    try:
        v = int(raw)
    except ValueError:
        return DEFAULT_COLBERT_CANDIDATES
    return max(1, v)


def _maxlen() -> int:
    """Token truncation cap from MEMDAG_COLBERT_MAXLEN (default 512).

    Caps ColBERT per-token storage (~2 KB/token fp16) and encode cost. The
    chunker already bounds content; this is the explicit storage valve.
    """
    raw = os.environ.get("MEMDAG_COLBERT_MAXLEN")
    if raw is None or not raw.strip():
        return DEFAULT_MAXLEN
    try:
        v = int(raw)
    except ValueError:
        return DEFAULT_MAXLEN
    return max(1, v)


def _device():
    """Optional explicit device from MEMDAG_BGE_DEVICE (e.g. 'cuda', 'cpu').

    None -> let FlagEmbedding auto-select (cuda if available, else cpu).
    """
    raw = (os.environ.get("MEMDAG_BGE_DEVICE") or "").strip()
    return raw or None


# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_MODEL = None            # process-global BGEM3FlagModel (None until first use)
_BGE_AVAILABLE = None    # cached tri-state import probe


def bge_available() -> bool:
    """True iff FlagEmbedding + torch + numpy import cleanly. Cached; never raises.

    Probes imports ONLY (no model download / VRAM). Lets every caller cheaply
    decide whether to take the bge branch or fall back, including in CI where the
    deps are absent.
    """
    global _BGE_AVAILABLE
    if _BGE_AVAILABLE is None:
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
            import FlagEmbedding  # noqa: F401
            _BGE_AVAILABLE = True
        except Exception:
            _BGE_AVAILABLE = False  # any import/DLL failure -> bge unavailable (never raises)
    return _BGE_AVAILABLE


def _get_model():
    """Load (once) and return the BGEM3FlagModel singleton.

    Cold load is ~2.2 GB fp16 on GPU and 10-30 s — amortized across a batch
    `reindex` or a long-lived MCP/broker process. Raises on failure (callers
    catch and fall back).
    """
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import BGEM3FlagModel
        kwargs = {"use_fp16": True}
        dev = _device()
        if dev:
            kwargs["devices"] = dev
        _MODEL = BGEM3FlagModel(BGE_HF_REPO, **kwargs)
    return _MODEL


def unload() -> None:
    """Drop the model singleton and free VRAM (keep-alive analog).

    Wired to post-`reindex` when MEMDAG_BGE_UNLOAD=1 and to MCP/broker shutdown.
    Safe to call when nothing is loaded.
    """
    global _MODEL
    _MODEL = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # torch missing or cache-clear failed — nothing to free, ignore


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def _encode(text: str) -> dict:
    """Encode one string -> {'dense','sparse','colbert'} of plain-Python types.

    Always encodes a 1-element batch and indexes [0] for deterministic shapes.
    Raises on any failure (callers catch).
    """
    model = _get_model()
    out = model.encode(
        [text],
        max_length=_maxlen(),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    dense = [float(x) for x in out["dense_vecs"][0]]
    # lexical_weights: {token_id_str: weight}; keys may be numpy ints -> str()
    raw_sparse = out["lexical_weights"][0]
    sparse = {str(k): float(v) for k, v in raw_sparse.items()}
    colbert = [[float(x) for x in row] for row in out["colbert_vecs"][0]]
    return {"dense": dense, "sparse": sparse, "colbert": colbert}


_WARNED_FALLBACK = False


def _warn_fallback(op: str, exc: Exception) -> None:
    """Emit ONE stderr warning when bge was requested + importable but failed at
    runtime, then stay quiet (reindex calls encode per-node — no spam).

    Silent degrade-to-default is great for uptime but hides a broken bge setup:
    the caller believes it's on the premium path while quietly getting the default
    backend. A version/API mismatch (FlagEmbedding older than the pinned floor, so
    the `devices=` kwarg or the encode() return shape differs) surfaces right here.
    """
    global _WARNED_FALLBACK
    if _WARNED_FALLBACK:
        return
    _WARNED_FALLBACK = True
    import sys
    print(
        f"[memsom] BGE-M3 backend requested but {op} FAILED "
        f"({type(exc).__name__}: {exc}). Falling back to the default backend — "
        f"retrieval quality is reduced. Verify `pip show FlagEmbedding` (need "
        f">=1.4.0) and that the model is available. This warning shows once.",
        file=sys.stderr,
    )


def encode_doc(text: str):
    """Encode a document. Returns the signal dict, or None on any failure."""
    try:
        return _encode(text)
    except Exception as exc:
        _warn_fallback("document encoding", exc)
        return None


def encode_query(text: str):
    """Encode a query. Returns the signal dict, or None on any failure.

    Separate from encode_doc so a future asymmetric-query instruction can land
    here without touching the doc path.
    """
    try:
        return _encode(text)
    except Exception as exc:
        _warn_fallback("query encoding", exc)
        return None


# ---------------------------------------------------------------------------
# Schema (additive; follows the rel_edges side-table pattern: INTEGER node_id,
# WITHOUT ROWID, join index). node_id is INTEGER (the nodes.id PK), never uuid.
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sparse_vecs (
  node_id      INTEGER NOT NULL REFERENCES nodes(id),
  model        TEXT    NOT NULL,
  weights_json TEXT    NOT NULL,
  PRIMARY KEY (node_id, model)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS colbert_vecs (
  node_id  INTEGER NOT NULL REFERENCES nodes(id),
  model    TEXT    NOT NULL,
  n_tokens INTEGER NOT NULL,
  dim      INTEGER NOT NULL,
  vecs     BLOB    NOT NULL,
  PRIMARY KEY (node_id, model)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_sparse_node  ON sparse_vecs(node_id);
CREATE INDEX IF NOT EXISTS idx_colbert_node ON colbert_vecs(node_id);
"""


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: create sparse_vecs + colbert_vecs (+ indexes) if absent."""
    memsom_schema.ensure_table(conn, _SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def colbert_to_blob(mat) -> bytes:
    """Pack an [n_tokens, dim] matrix as float16 (IEEE binary16) LE row-major.

    Pure stdlib (struct 'e' = half-float) so the storage path needs no numpy —
    keeps the module CI-safe (zero pip deps) and the blob format model-stable.
    """
    flat = [float(x) for row in mat for x in row]
    return struct.pack(f"<{len(flat)}e", *flat)


def blob_to_colbert(blob: bytes, n_tokens: int, dim: int) -> list:
    """Recover the [n_tokens, dim] matrix from a float16 LE blob (stdlib)."""
    if dim <= 0 or n_tokens <= 0:
        return []
    n = len(blob) // 2  # 2 bytes per float16
    flat = struct.unpack(f"<{n}e", blob)
    return [list(flat[i * dim:(i + 1) * dim]) for i in range(n_tokens)]


# ---------------------------------------------------------------------------
# Store / deindex
# ---------------------------------------------------------------------------

def store_bge(conn: sqlite3.Connection, nid: int, enc: dict) -> None:
    """Persist all three bge signals for *nid*, atomically.

    dense -> embeddings(model='bge-m3', dim=1024); sparse -> sparse_vecs;
    colbert -> colbert_vecs (one fp16 BLOB per node). Tagged model='bge-m3' so
    the model-filtered reads never collide with nomic (768-dim) rows.
    """
    migrate(conn)
    import memsom_retrieve  # lazy: reuse the float32 dense packer (cycle-safe)

    dense = enc["dense"]
    dense_blob = memsom_retrieve._vec_to_blob(dense)
    import json
    weights_json = json.dumps(enc["sparse"])
    colbert = enc["colbert"]
    n_tokens = len(colbert)
    dim = len(colbert[0]) if n_tokens else BGE_DENSE_DIM
    colbert_blob = colbert_to_blob(colbert)

    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec)"
            " VALUES (?, ?, ?, ?)",
            (nid, BGE_MODEL_NAME, len(dense), dense_blob),
        )
        conn.execute(
            "INSERT OR REPLACE INTO sparse_vecs(node_id, model, weights_json)"
            " VALUES (?, ?, ?)",
            (nid, BGE_MODEL_NAME, weights_json),
        )
        conn.execute(
            "INSERT OR REPLACE INTO colbert_vecs(node_id, model, n_tokens, dim, vecs)"
            " VALUES (?, ?, ?, ?, ?)",
            (nid, BGE_MODEL_NAME, n_tokens, dim, colbert_blob),
        )


def deindex_bge(conn: sqlite3.Connection, nid: int) -> None:
    """Purge *nid* from sparse_vecs + colbert_vecs. Bare execs (NO `with conn:`)
    so it composes inside an existing transaction (deindex_node's `with conn:`
    or compact's archive txn). Table-guarded: no-op when the schema is absent.
    """
    if memsom_schema.table_exists(conn, "sparse_vecs"):
        conn.execute("DELETE FROM sparse_vecs WHERE node_id = ?", (nid,))
    if memsom_schema.table_exists(conn, "colbert_vecs"):
        conn.execute("DELETE FROM colbert_vecs WHERE node_id = ?", (nid,))


# ---------------------------------------------------------------------------
# Scoring primitives (pure-Python by default; torch path for ColBERT MaxSim)
# ---------------------------------------------------------------------------

def sparse_dot(q_sparse: dict, d_sparse: dict) -> float:
    """Dot product over shared lexical-weight keys. Iterate the smaller dict."""
    if not q_sparse or not d_sparse:
        return 0.0
    if len(d_sparse) < len(q_sparse):
        q_sparse, d_sparse = d_sparse, q_sparse
    total = 0.0
    for tok, w in q_sparse.items():
        other = d_sparse.get(tok)
        if other is not None:
            total += w * other
    return float(total)


def colbert_maxsim(q_colbert, d_colbert) -> float:
    """ColBERT late-interaction score: sum over query tokens of the max dot
    product against any doc token. Vectors are L2-normalized by BGE-M3, so a dot
    is a cosine. Uses torch if available (one batched matmul), else pure Python.
    """
    if not q_colbert or not d_colbert:
        return 0.0
    try:
        import torch
        q = torch.tensor(q_colbert, dtype=torch.float32)
        d = torch.tensor(d_colbert, dtype=torch.float32)
        # [n_q, n_d] sims -> max over doc tokens -> sum over query tokens
        sims = q @ d.T
        return float(sims.max(dim=1).values.sum().item())
    except Exception:
        # Pure-Python fallback (CI-testable, no torch).
        total = 0.0
        for qv in q_colbert:
            best = None
            for dv in d_colbert:
                s = 0.0
                for a, b in zip(qv, dv):
                    s += a * b
                if best is None or s > best:
                    best = s
            if best is not None:
                total += best
        return float(total)
