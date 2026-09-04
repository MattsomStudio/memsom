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

import struct
import sqlite3
import sys
import threading

from memsom.storage import schema as memsom_schema
from memsom import tuning as memsom_tuning

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BACKEND = "ollama"
VALID_BACKENDS = ("ollama", "bge-m3", "bm25")
BGE_MODEL_NAME = "bge-m3"          # the `model` tag stored alongside bge vectors
BGE_HF_REPO = "BAAI/bge-m3"        # the FlagEmbedding load path
BGE_DENSE_DIM = 1024
DEFAULT_MAXLEN = 512               # passage/query token cap (FlagEmbedding default)
DEFAULT_COLBERT_CANDIDATES = 30    # ColBERT re-rank window (see memsom_retrieve); 100 until 2026-09-04, see tuning.py

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_WARNED_PIN_MISMATCH = False


def _warn_pin_mismatch(env_backend: str, pinned: str) -> None:
    """ONE stderr warning per process when an explicit MEMDAG_EMBED_BACKEND /
    --embed-backend disagrees with the backend the store was embedded under.
    The explicit setting still wins (an operator switching on purpose), but
    silently it would split the store again -- the 2026-09 bug."""
    global _WARNED_PIN_MISMATCH
    if _WARNED_PIN_MISMATCH:
        return
    _WARNED_PIN_MISMATCH = True
    import sys
    print(
        f"[memsom] embed.backend={env_backend!r} but this store is pinned to "
        f"{pinned!r}: vectors written now will be invisible to the pinned "
        f"reader (and vice versa). Run `memsom reindex` under the backend you "
        f"want to make it the store's. This warning shows once.",
        file=sys.stderr,
    )


def backend(conn=None) -> str:
    """Active embedding backend.

    Resolution: in-process override (`--embed-backend`) > MEMDAG_EMBED_BACKEND
    > the STORE's pinned backend (retrieval_meta, via *conn*) > 'ollama'.

    The pin is the split fix: a process launched without the env var (the
    Stop-hook importer, an interactive shell) used to fall to the compiled-in
    default and write nomic rows into a bge-m3 store, which the reader's
    `WHERE model = ?` then never saw. With *conn* it now adopts whatever the
    store was embedded under. Without *conn* (display/CLI paths) the old
    env-or-default answer stands.
    """
    raw = (memsom_tuning.resolve("embed.backend") or "").strip().lower()
    pinned = None
    if conn is not None:
        try:
            from memsom.retrieval import retrieve as memsom_retrieve
            pinned = memsom_retrieve.pinned_backend(conn)
        # FAILOPEN: allowed -- an unreadable pin (closed/locked conn) means "unpinned", never a crash on the read path.
        except Exception:
            pinned = None
        if pinned not in VALID_BACKENDS:
            pinned = None
    if raw in VALID_BACKENDS:
        # bm25 writes no vectors, so it can never split the store: the prompt
        # hook pins itself to bm25 on purpose (no model load) and must not warn.
        if pinned is not None and pinned != raw and raw != "bm25":
            _warn_pin_mismatch(raw, pinned)
        return raw
    if pinned is not None:
        return pinned
    return DEFAULT_BACKEND


def active_model_name(conn=None) -> str:
    """The `model` tag the active backend writes/reads in the vector tables.

    This is the load-bearing key for the dim-collision fix: vector_search filters
    `WHERE model = active_model_name()`, so 768-dim nomic rows and 1024-dim bge
    rows never get cosine'd against each other.

    bm25 -> '' (no model) so a `WHERE model=''` matches nothing -> BM25-only.
    """
    b = backend(conn)
    if b == "bge-m3":
        return BGE_MODEL_NAME
    if b == "bm25":
        return ""
    # ollama: reuse retrieve's resolver so there's one source of truth.
    from memsom.retrieval import retrieve as memsom_retrieve
    return memsom_retrieve._embed_model()


def colbert_candidates() -> int:
    """Re-rank window size from MEMDAG_COLBERT_CANDIDATES (default 30; 100 until 2026-09-04)."""
    raw = memsom_tuning.resolve("retrieval.colbert_candidates")
    if isinstance(raw, int):  # unset -> the registered (typed) default
        return max(1, raw)
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
    raw = memsom_tuning.resolve("retrieval.colbert_maxlen")
    if isinstance(raw, int):  # unset -> the registered (typed) default
        return max(1, raw)
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
    raw = (memsom_tuning.resolve("retrieval.bge_device") or "").strip()
    return raw or None


# ---------------------------------------------------------------------------
# Encode path + signal toggles (portable / opt-in)
# ---------------------------------------------------------------------------

def encode_via() -> str:
    """How bge-m3 embeds: 'auto' | 'supervisor' | 'inprocess' (default 'auto')."""
    raw = (memsom_tuning.resolve("retrieval.bge_encode_via") or "").strip().lower()
    return raw if raw in ("auto", "supervisor", "inprocess") else "auto"


def _mode_on(key: str) -> bool:
    """A bge signal toggle (bge_dense/sparse/colbert), default ON. Accepts the
    typed bool default (unset) or a raw env string."""
    v = memsom_tuning.resolve(key)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def dense_enabled() -> bool:
    return _mode_on("retrieval.bge_dense")


def sparse_enabled() -> bool:
    return _mode_on("retrieval.bge_sparse")


def colbert_enabled() -> bool:
    return _mode_on("retrieval.bge_colbert")


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
        # find_spec, NOT import. This probe runs in every process that calls
        # migrate_all / `memsom features` / a retrieve gate -- the MCP server
        # (a Claude Code child) included -- and an import here pulled torch,
        # transformers, sklearn, pandas and pyarrow (~6 s, GBs of address
        # space) into all of them. MEASURED 2026-09-04: while any thread sits
        # in a blocking stdin pipe read (the MCP main loop, always), a
        # C-extension load on another thread blocks until that read returns,
        # so the warm-endpoint request thread hung in this import for minutes;
        # and the same import chain access-violated in pyarrow's DLL init in a
        # long test process. A broken install is still caught: the real
        # import lives in _get_model(), whose callers already fall back.
        try:
            import importlib.util
            _BGE_AVAILABLE = all(
                importlib.util.find_spec(name) is not None
                for name in ("numpy", "torch", "FlagEmbedding"))
        # FAILOPEN: swallows and marks unavailable -- a package whose spec lookup itself errors (broken metadata) means bge is unavailable, never raises.
        except Exception:
            _BGE_AVAILABLE = False
    return _BGE_AVAILABLE


def _supervisor_possible() -> bool:
    """True iff the supervisor path can produce bge signals: bge_url set AND
    (its /health answers now OR a spawn_cmd is configured to bring it up).
    An idle/down supervisor with a spawn_cmd is NOT "no path" — the encode
    will cold-start it. Never imports torch."""
    from memsom.retrieval import bge_client
    if not bge_client.configured():
        return False
    if bge_client.supervisor_reachable():
        return True
    return bool(bge_client.spawn_cmd())


def bge_usable() -> bool:
    """True iff bge signals can be produced by SOME path — in-process torch OR
    the local supervisor (reachable, or spawnable via bge_spawn_cmd). This is
    the gate retrieve.py uses to decide the bge branch, so a box with the
    supervisor but no local torch still gets bge.

    Mode rules (retrieval.bge_encode_via):
      inprocess  -> torch only, never probes the network.
      supervisor -> supervisor only, NEVER imports torch into this process
                    (the MCP server is a Claude Code child; torch must not
                    load there — that is the whole point of the mode).
      auto       -> torch is checked FIRST (a cheap cached import probe, no
                    network): when it is available we never touch the network
                    here, which keeps the test suite (which patches
                    bge_available) hermetic. Only a torch-less box probes.
    """
    via = encode_via()
    if via == "inprocess":
        return bge_available()
    if via == "supervisor":
        return _supervisor_possible()
    if bge_available():
        return True
    return _supervisor_possible()


def _get_model():
    """Load (once) and return the BGEM3FlagModel singleton.

    Cold load is ~2.2 GB fp16 on GPU and 10-30 s — amortized across a batch
    `reindex` or a long-lived MCP/broker process. Raises on failure (callers
    catch and fall back).
    """
    global _MODEL
    if _MODEL is None:
        if ("FlagEmbedding" not in sys.modules
                and threading.current_thread() is not threading.main_thread()):
            # A FIRST torch/FlagEmbedding import off the main thread wedges a
            # process whose main thread is blocked in a stdin pipe read (the
            # MCP server) -- see bge_available(). Refuse; callers fall back
            # (BM25 + degraded signal), which is the honest outcome for a
            # server that was never meant to hold torch anyway.
            raise RuntimeError("in-process bge-m3 load refused off the main thread "
                               "(use the supervisor path: retrieval.bge_encode_via)")
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
    Safe to call when nothing is loaded. Touches torch ONLY if this process
    already imported it — a supervisor-mode process must never import torch
    just to discover it has nothing to free.
    """
    global _MODEL, _QUERY_MEMO
    _MODEL = None
    _QUERY_MEMO = (None, None)
    import sys
    if "torch" not in sys.modules:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # FAILOPEN: swallows and continues -- torch missing or cache-clear failed, nothing to free, ignore.
    except Exception:
        pass


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


_WARNED_SUPERVISOR = False


def _warn_supervisor(op: str) -> None:
    """ONE stderr warning per process when bge_encode_via=supervisor and the
    supervisor could not serve (down and not spawnable / spawn never came up /
    embed failed). In this mode in-process torch is deliberately never tried,
    so the operator must hear that the supervisor path itself is broken."""
    global _WARNED_SUPERVISOR
    if _WARNED_SUPERVISOR:
        return
    _WARNED_SUPERVISOR = True
    import sys
    print(
        f"[memsom] BGE-M3 {op} FAILED: bge_encode_via=supervisor but the local "
        f"supervisor (retrieval.bge_url) did not serve — down and no "
        f"bge_spawn_cmd, spawn never became healthy within bge_spawn_timeout, or "
        f"its /embed errored. In-process torch is disabled in this mode; this "
        f"query degrades to BM25. This warning shows once.",
        file=sys.stderr,
    )


def _dispatch_encode(text: str, op: str):
    """Produce {'dense','sparse','colbert'} for *text*, or None.

    Path (retrieval.bge_encode_via):
      auto / supervisor -> ensure_supervisor(): if bge_url's /health answers —
                           spawning the DETACHED supervisor first when it is down
                           and bge_spawn_cmd is set, and waiting for it — POST
                           /embed to it (no torch in THIS process).
      supervisor        -> that is the ONLY path: on failure warn once, return
                           None (never load torch into this process).
      auto              -> on an unreachable/unspawnable supervisor OR an embed
                           error, fall through to in-process FlagEmbedding.
      inprocess         -> in-process FlagEmbedding (the `bge` pip extra) only.
    The in-process load is SYNCHRONOUS on the calling thread, never a worker
    thread the caller abandons (the 50a5bac shape). On any total failure this
    warns once and returns None, so the caller degrades to BM25 — never raises.
    """
    via = encode_via()
    if via in ("auto", "supervisor"):
        from memsom.retrieval import bge_client
        if bge_client.configured() and bge_client.ensure_supervisor():
            enc = bge_client.encode_http(text)
            if enc is not None:
                return enc
            # supervisor reachable but the embed itself failed.
        if via == "supervisor":
            _warn_supervisor(op)
            return None
    try:
        return _encode(text)
    # FAILOPEN: warns once and returns None -- no bge encode path (supervisor down/unset AND FlagEmbedding not importable or erroring) degrades this to BM25, never crashes ingest/retrieval.
    except Exception as exc:
        _warn_fallback(op, exc)
        return None


def encode_doc(text: str):
    """Encode a document. Returns the signal dict, or None on any failure."""
    return _dispatch_encode(text, "document encoding")


# One-entry memo for the query encoder: retrieve() asks for the SAME query's
# signals up to three times (dense, sparse, colbert). Over the supervisor that
# was three HTTP round trips — and three spawn/cold-start attempts when it was
# down. A failed encode (None) is never memoised, so a retry is a real retry.
_QUERY_MEMO = (None, None)


def _reset_query_memo() -> None:
    """Test-isolation escape hatch."""
    global _QUERY_MEMO
    _QUERY_MEMO = (None, None)


def encode_query(text: str):
    """Encode a query. Returns the signal dict, or None on any failure.

    Separate from encode_doc so a future asymmetric-query instruction can land
    here without touching the doc path.
    """
    global _QUERY_MEMO
    if _QUERY_MEMO[0] == text and _QUERY_MEMO[1] is not None:
        return _QUERY_MEMO[1]
    enc = _dispatch_encode(text, "query encoding")
    _QUERY_MEMO = (text, enc)
    return enc


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
    from memsom.retrieval import retrieve as memsom_retrieve

    import json
    with conn:
        # Each signal is written ONLY when its toggle is on; when off, any existing
        # row for this node is PURGED so toggling a signal off + reindex removes it
        # (leaving a stale disabled vector would keep scoring against it).
        if dense_enabled():
            dense = enc["dense"]
            conn.execute(
                "INSERT OR REPLACE INTO embeddings(node_id, model, dim, vec)"
                " VALUES (?, ?, ?, ?)",
                (nid, BGE_MODEL_NAME, len(dense),
                 memsom_retrieve._vec_to_blob(dense)),
            )
        else:
            conn.execute("DELETE FROM embeddings WHERE node_id = ? AND model = ?",
                         (nid, BGE_MODEL_NAME))
        if sparse_enabled():
            conn.execute(
                "INSERT OR REPLACE INTO sparse_vecs(node_id, model, weights_json)"
                " VALUES (?, ?, ?)",
                (nid, BGE_MODEL_NAME, json.dumps(enc["sparse"])),
            )
        else:
            conn.execute("DELETE FROM sparse_vecs WHERE node_id = ? AND model = ?",
                         (nid, BGE_MODEL_NAME))
        if colbert_enabled():
            colbert = enc["colbert"]
            n_tokens = len(colbert)
            dim = len(colbert[0]) if n_tokens else BGE_DENSE_DIM
            conn.execute(
                "INSERT OR REPLACE INTO colbert_vecs(node_id, model, n_tokens, dim, vecs)"
                " VALUES (?, ?, ?, ?, ?)",
                (nid, BGE_MODEL_NAME, n_tokens, dim, colbert_to_blob(colbert)),
            )
        else:
            conn.execute("DELETE FROM colbert_vecs WHERE node_id = ? AND model = ?",
                         (nid, BGE_MODEL_NAME))


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


def numpy_for_scoring():
    """The numpy module for ColBERT scoring, or None — and NEVER a fresh import
    off the main thread.

    MEASURED 2026-09-04 (py-spy on the live MCP server AND a self-spawned
    repro): `import numpy` from a socketserver request thread of the MCP
    process hangs forever inside the `_multiarray_umath` DLL load
    (create_module), while the same import on the main thread — or from a
    thread of a plain process — takes 80 ms. Every warm-endpoint query then
    wedged in colbert_maxsim, the prompt hook burned its budget and backed off,
    and retrieval read as "warm endpoint down" with a healthy encoder.

    So: an already-imported numpy is used from any thread; a first import is
    only ever attempted on the main thread (the MCP server preloads it there at
    startup, see mcp.preload_numeric). Off the main thread with numpy absent
    this returns None and callers take a numpy-free path.
    """
    mod = sys.modules.get("numpy")
    if mod is not None:
        return mod
    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        import numpy
        return numpy
    # FAILOPEN: allowed -- no numpy on this box means the pure-Python / skip path, never an error.
    except Exception:
        return None


def numpy_scoring_available() -> bool:
    """True iff colbert_maxsim would score with numpy on THIS thread (see
    numpy_for_scoring). The pure-Python fallback is ~100x slower — fine for a
    unit test, far too slow for a 100-candidate rerank inside a hook budget —
    so the rerank is skipped outright when this is False."""
    return numpy_for_scoring() is not None


def colbert_maxsim_blob(q_np, blob: bytes, n_tokens: int, dim: int):
    """MaxSim of a query matrix (numpy float32 [n_q, dim]) against a stored
    fp16-LE colbert blob, decoded with frombuffer -- no Python lists. Returns
    None (caller falls back to blob_to_colbert + colbert_maxsim) when numpy is
    not loaded on this thread or the blob does not match its declared shape."""
    np = numpy_for_scoring()
    if np is None or n_tokens <= 0 or dim <= 0 or len(blob) != 2 * n_tokens * dim:
        return None
    try:
        d = np.frombuffer(blob, dtype="<f2").reshape(n_tokens, dim).astype(np.float32)
        return float((q_np @ d.T).max(axis=1).sum())
    # FAILOPEN: not actually open -- a shape/dtype error returns None and the caller takes the exact list path.
    except Exception:
        return None


def colbert_maxsim(q_colbert, d_colbert) -> float:
    """ColBERT late-interaction score: sum over query tokens of the max dot
    product against any doc token. Vectors are L2-normalized by BGE-M3, so a dot
    is a cosine. Uses numpy if available (one matmul), else pure Python.

    NEVER torch: this runs on the QUERY path of every process — the MCP server
    (a Claude Code child) included — and a torch import there is exactly what
    the supervisor design exists to avoid (2026-09-04). numpy is a cheap CPU
    import; the pure-Python fallback keeps CI at zero deps.
    """
    if not q_colbert or not d_colbert:
        return 0.0
    try:
        np = numpy_for_scoring()
        if np is None:
            raise ImportError("numpy not loaded on this thread")
        q = np.asarray(q_colbert, dtype=np.float32)
        d = np.asarray(d_colbert, dtype=np.float32)
        # [n_q, n_d] sims -> max over doc tokens -> sum over query tokens
        return float((q @ d.T).max(axis=1).sum())
    # FAILOPEN: not actually open -- falls back to an equivalent pure-Python computation (CI-testable, no numpy) instead of raising.
    except Exception:
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
