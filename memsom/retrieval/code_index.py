"""memsom_code_index — a SEPARATE semantic + lexical index over CODE.

Why this exists
---------------
memsom's retrieve.py indexes FACTS and NOTES (bge-m3 / Ollama). Code is a different
beast: identifiers, call structure, thin comments. The 2026-07-20 embedder bench
(project_code_rag_embedder_bench) showed a code embedder (Qwen3-Embedding-4B) beats
bge-m3 by +0.17 MRR on thin code — so the code index is its OWN store, its OWN tables,
its OWN embedder path (retrieval/qwen_embed.py), never mixed into the fact store.

Architecture mirrors retrieve.py exactly:
  - Pure-stdlib BM25 over code_postings / code_docstats (restores exact-identifier
    matching that pure-dense loses).
  - Optional Qwen dense vectors over code_embeddings, RRF-fused with BM25.
  - Chunking: Python AST (functions/classes, raw source with comments) for .py;
    sliding line-windows for everything else.

Optionality (load-bearing — matches memsom's "Ollama is optional" ethos)
  - Opt-in: the whole subsystem is inert unless MEMSOM_CODE_RAG is truthy. migrate()
    still creates the (empty) tables idempotently so migrate_all stays uniform, but
    index/search are no-ops-with-a-message when disabled.
  - No new mandatory dep: qwen_embed is stdlib-only; a down server degrades to BM25.
  - Runtime degrade: Qwen unreachable -> BM25-only index + search, never a crash.

Schema (additive; all IF NOT EXISTS):
  code_chunks(id, repo, path, symbol, kind, start_line, end_line, content,
              content_hash, indexed_at)
  code_postings(term, chunk_id, tf, PK(term, chunk_id))
  code_docstats(chunk_id PK, length)
  code_embeddings(chunk_id, model, dim, vec BLOB, PK(chunk_id, model))
"""
import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import sqlite3

import memsom
from memsom.storage import schema as memsom_schema
from memsom.retrieval import qwen_embed
# Reuse retrieve.py's stdlib helpers — table-agnostic, so no duplication.
from memsom.retrieval.retrieve import (
    tokenize, _rrf_fuse, _cosine, _vec_to_blob, _blob_to_vec, K1, B,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Files worth embedding — code + config, never binaries / data blobs.
CODE_EXTS = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".ps1", ".psm1", ".psd1", ".bat", ".cmd",
    ".go", ".rs", ".java", ".kt", ".c", ".h", ".hpp", ".cc", ".cpp",
    ".rb", ".php", ".lua", ".pl", ".sql", ".r",
    ".html", ".css", ".scss", ".vue", ".svelte",
    ".toml", ".ini", ".cfg", ".yaml", ".yml",
}
SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules", ".venv", "venv",
    "env", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".next", ".cache", "site-packages", ".tox", "target",
}
MAX_FILE_BYTES = 400_000      # skip generated/minified monsters
WINDOW_LINES = 60             # non-.py sliding window size


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------

# The panel writes these flags; the CLI/hook read them fresh every run. A FILE (not
# just the env var) so the memsom panel's settings-tab toggle takes effect immediately
# — an env var only reaches processes started AFTER it was set, which a running shell or
# an already-spawned MCP server never sees. Lives beside the DB so it ports with the store.
_FLAGS_NAME = "code_rag.json"


def _flags_path():
    # Beside the DB file (honors MEMDAG_DB/MEMDAG_HOME) so the flag ports with the store
    # and tests pointed at a temp DB never read the real home-dir flag.
    return memsom.db_path().parent / _FLAGS_NAME


def _read_flags() -> dict:
    """The code_rag.json flag file as a dict; {} if absent/unreadable (fail-open to env)."""
    try:
        with open(_flags_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _enabled() -> bool:
    """The code-RAG is opt-in. Default OFF -> a plain memsom install is unchanged.

    Precedence: the panel-written file flag wins (so the settings-tab toggle is
    authoritative and instant); with no file flag, fall back to MEMSOM_CODE_RAG.
    """
    flags = _read_flags()
    if "enabled" in flags:
        return bool(flags["enabled"])
    return (os.environ.get("MEMSOM_CODE_RAG") or "").strip().lower() in (
        "1", "true", "yes", "on")


def _write_flags(update: dict) -> None:
    """Merge *update* into code_rag.json, atomically. Never clobbers flags this
    process doesn't know about (the panel writes `enabled`/`auto_reindex` here)."""
    flags = _read_flags()
    flags.update(update)
    path = _flags_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(flags, f, indent=2)
    os.replace(tmp, path)


def registered_repos() -> list:
    """The repos the code-RAG indexes, from code_rag.json['repos'].

    Entries are {"path": ..., "name": ...}; a bare string is accepted as a path.
    Deduped by normalized path, order preserved. Empty list = nothing registered
    yet (`code-repos add` / `code-repos discover` fills it)."""
    out, seen = [], set()
    for entry in _read_flags().get("repos") or []:
        if isinstance(entry, str):
            entry = {"path": entry}
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        path = os.path.abspath(os.path.expanduser(entry["path"]))
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        exclude = [str(x).replace("\\", "/").strip("/")
                   for x in (entry.get("exclude") or []) if str(x).strip()]
        out.append({"path": path,
                    "name": entry.get("name") or os.path.basename(path.rstrip("\\/")),
                    "exclude": exclude})
    return out


def add_repo(path: str, name: str = None, exclude: list = None) -> dict:
    """Register a repo. Returns {"added": bool, "repo": {...}} — re-adding an
    already-registered path is a no-op, not an error.

    *exclude* is a list of repo-relative directory prefixes to keep out of the
    index. Git repos rarely need it (gitignore already draws the line); it's for
    non-git trees that mix your code with vendored/downloaded content."""
    path = os.path.abspath(os.path.expanduser(path))
    name = name or os.path.basename(path.rstrip("\\/"))
    exclude = [str(x).replace("\\", "/").strip("/") for x in (exclude or []) if str(x).strip()]
    entry = {"path": path, "name": name}
    if exclude:
        entry["exclude"] = exclude
    repos = registered_repos()
    if any(os.path.normcase(r["path"]) == os.path.normcase(path) for r in repos):
        return {"added": False, "repo": entry}
    repos.append(entry)
    _write_flags({"repos": repos})
    return {"added": True, "repo": entry}


def remove_repo(path_or_name: str) -> bool:
    """Unregister by path OR by name. The repo's chunks stay in the index until
    `code-index-all --prune` (or a manual purge) — unregistering is not deleting."""
    target = os.path.normcase(os.path.abspath(os.path.expanduser(path_or_name)))
    kept = [r for r in registered_repos()
            if os.path.normcase(r["path"]) != target and r["name"] != path_or_name]
    if len(kept) == len(registered_repos()):
        return False
    _write_flags({"repos": kept})
    return True


def discover_repos(root: str, depth: int = 3) -> list:
    """Absolute paths of git repos under *root*, at most *depth* levels deep.
    Doesn't descend INTO a repo once found (no submodule/vendored-clone noise)."""
    root = os.path.abspath(os.path.expanduser(root))
    found = []
    root_depth = root.rstrip("\\/").count(os.sep)
    for dp, dirs, _files in os.walk(root):
        if dp.rstrip("\\/").count(os.sep) - root_depth >= depth:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if ".git" in dirs or os.path.isdir(os.path.join(dp, ".git")):
            found.append(dp)
            dirs[:] = []          # a repo is a leaf for discovery purposes
    return found


def _auto_reindex_enabled() -> bool:
    """The post-commit auto-reindex hook honors this INDEPENDENTLY of the master gate,
    so code-search can stay on while commit-time reindexing is silenced (e.g. to kill
    commit latency). Defaults ON when the feature is enabled, so existing hooks keep
    working with no flag file present."""
    return bool(_read_flags().get("auto_reindex", True))


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS code_chunks (
  id           INTEGER PRIMARY KEY,
  repo         TEXT NOT NULL,
  path         TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  kind         TEXT NOT NULL,
  start_line   INTEGER NOT NULL,
  end_line     INTEGER NOT NULL,
  content      TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  indexed_at   TEXT NOT NULL,
  UNIQUE (repo, path, symbol, start_line)
);
CREATE TABLE IF NOT EXISTS code_postings (
  term     TEXT NOT NULL,
  chunk_id INTEGER NOT NULL,
  tf       INTEGER NOT NULL,
  PRIMARY KEY (term, chunk_id)
);
CREATE TABLE IF NOT EXISTS code_docstats (
  chunk_id INTEGER PRIMARY KEY,
  length   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS code_embeddings (
  chunk_id INTEGER NOT NULL,
  model    TEXT NOT NULL,
  dim      INTEGER NOT NULL,
  vec      BLOB NOT NULL,
  PRIMARY KEY (chunk_id, model)
);
CREATE TABLE IF NOT EXISTS code_files (
  repo       TEXT NOT NULL,
  path       TEXT NOT NULL,
  file_hash  TEXT NOT NULL,
  embedded   INTEGER NOT NULL DEFAULT 0,
  indexed_at TEXT NOT NULL,
  PRIMARY KEY (repo, path)
);
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_path ON code_chunks(repo, path);
CREATE INDEX IF NOT EXISTS idx_code_embeddings_model ON code_embeddings(model);
"""


def migrate(conn: sqlite3.Connection) -> None:
    """Idempotent: create the code-index tables if absent. Harmless when the feature
    is disabled (tables just stay empty)."""
    memsom_schema.ensure_table(conn, _SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _window_chunks(src: str) -> list:
    """Sliding fixed-size line windows for non-AST files (or unparseable .py)."""
    lines = src.splitlines()
    out = []
    for i in range(0, len(lines), WINDOW_LINES):
        block = lines[i:i + WINDOW_LINES]
        seg = "\n".join(block)
        if not seg.strip():
            continue
        out.append({
            "symbol": f"L{i + 1}-{i + len(block)}",
            "kind": "window",
            "start_line": i + 1,
            "end_line": i + len(block),
            "content": seg,
        })
    return out


def chunk_file(path: str) -> list:
    """Return a list of chunk dicts for *path*.

    .py -> one chunk per function / async-function / class (raw source WITH comments +
    docstrings — the bench showed the code embedder reads tokens, so keep the real thing).
    Everything else -> sliding line-windows.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            src = f.read()
    except Exception:
        return []
    if not src.strip():
        return []

    if path.endswith((".py", ".pyi")):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return _window_chunks(src)          # unparseable -> still index it
        out, seen = [], set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seg = ast.get_source_segment(src, n)
                if not seg:
                    continue
                key = (n.name, getattr(n, "lineno", 0))
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "symbol": n.name,
                    "kind": "class" if isinstance(n, ast.ClassDef) else "function",
                    "start_line": getattr(n, "lineno", 0),
                    "end_line": getattr(n, "end_lineno", 0) or getattr(n, "lineno", 0),
                    "content": seg,
                })
        return out or _window_chunks(src)        # module w/ no defs -> window it
    return _window_chunks(src)


def _eligible(path: str) -> bool:
    """Indexable source file: right extension, not a generated monster, not
    inside a junk dir."""
    if os.path.splitext(path)[1].lower() not in CODE_EXTS:
        return False
    parts = os.path.normpath(path).split(os.sep)
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    try:
        return os.path.getsize(path) <= MAX_FILE_BYTES
    except OSError:
        return False


def _git_listed_files(root: str):
    """Everything `git status` would show you: tracked files PLUS untracked ones
    that aren't gitignored. None when *root* isn't a git repo (or git is missing).

    This is the line between "code I write" and "code I downloaded": vendored
    source trees, build output, dependency dirs and model caches are all
    gitignored already, so they never enter the index — and no denylist has to
    be maintained to keep them out. Untracked-but-not-ignored is included on
    purpose: a file you just created is searchable before you commit it.
    """
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"],
            capture_output=True, timeout=120)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    names = out.stdout.decode("utf-8", "replace").split("\0")
    return [os.path.join(root, n.replace("/", os.sep)) for n in names if n]


def _iter_source_files(root: str):
    """Yield indexable source-file paths under *root*.

    Git repo -> ask git what belongs to the project (see _git_listed_files).
    Otherwise -> walk the tree, skipping junk dirs / binaries.
    """
    listed = _git_listed_files(root)
    if listed is not None:
        for p in listed:
            if _eligible(p) and os.path.isfile(p):
                yield p
        return
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in CODE_EXTS:
                continue
            p = os.path.join(dp, f)
            try:
                if os.path.getsize(p) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def deindex_path(conn: sqlite3.Connection, repo: str, rel: str) -> None:
    """Purge every chunk (and its postings/docstats/embeddings) for one file."""
    with conn:
        cids = [r[0] for r in conn.execute(
            "SELECT id FROM code_chunks WHERE repo = ? AND path = ?", (repo, rel))]
        for cid in cids:
            conn.execute("DELETE FROM code_postings WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM code_docstats WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM code_embeddings WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM code_chunks WHERE repo = ? AND path = ?", (repo, rel))
        conn.execute("DELETE FROM code_files WHERE repo = ? AND path = ?", (repo, rel))


def _file_hash(path: str) -> str:
    """Hash of the file's bytes — the "has this changed?" key. Unreadable file
    returns a unique-ish sentinel so it never falsely matches a stored hash."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
    except OSError:
        return "unreadable"
    return h.hexdigest()


def _unchanged_files(conn, repo: str) -> dict:
    """rel path -> (file_hash, embedded) for everything already indexed."""
    return {r[0]: (r[1], r[2]) for r in conn.execute(
        "SELECT path, file_hash, embedded FROM code_files WHERE repo = ?", (repo,))}


def _mark_file(conn, repo: str, rel: str, file_hash: str, embedded: bool) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO code_files(repo, path, file_hash, embedded,"
            " indexed_at) VALUES (?, ?, ?, ?, ?)",
            (repo, rel, file_hash, 1 if embedded else 0, _now()))


def _index_file(conn: sqlite3.Connection, repo: str, root: str, path: str) -> list:
    """Full-file rebuild of BM25 rows for *path*. Returns [(chunk_id, content)] for the
    new chunks so the caller can batch-embed them. Vectors are added separately."""
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    chunks = chunk_file(path)
    new_rows = []
    with conn:
        # Drop the file's old chunks first (handles edits / deletions / renames within it).
        old = [r[0] for r in conn.execute(
            "SELECT id FROM code_chunks WHERE repo = ? AND path = ?", (repo, rel))]
        for cid in old:
            conn.execute("DELETE FROM code_postings WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM code_docstats WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM code_embeddings WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM code_chunks WHERE repo = ? AND path = ?", (repo, rel))

        for ch in chunks:
            cur = conn.execute(
                "INSERT INTO code_chunks(repo, path, symbol, kind, start_line, end_line,"
                " content, content_hash, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (repo, rel, ch["symbol"], ch["kind"], ch["start_line"], ch["end_line"],
                 ch["content"], _content_hash(ch["content"]), _now()))
            cid = cur.lastrowid
            toks = tokenize(ch["content"])
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            if tf:
                conn.executemany(
                    "INSERT INTO code_postings(term, chunk_id, tf) VALUES (?, ?, ?)",
                    [(term, cid, c) for term, c in tf.items()])
            conn.execute(
                "INSERT OR REPLACE INTO code_docstats(chunk_id, length) VALUES (?, ?)",
                (cid, len(toks)))
            new_rows.append((cid, ch["content"]))
    return new_rows


def _store_vectors(conn: sqlite3.Connection, rows: list) -> int:
    """Batch-embed [(chunk_id, content)] via Qwen and store. Returns count stored.
    Server down / disabled -> 0 (BM25-only), never raises."""
    if not rows or not qwen_embed.qwen_available():
        return 0
    vecs = qwen_embed.encode_docs([c for _, c in rows])
    if not vecs:
        return 0
    with conn:
        for (cid, _), v in zip(rows, vecs):
            conn.execute(
                "INSERT OR REPLACE INTO code_embeddings(chunk_id, model, dim, vec)"
                " VALUES (?, ?, ?, ?)",
                (cid, qwen_embed.MODEL_NAME, len(v), _vec_to_blob(v)))
    return len(vecs)


def _git_changed(root: str) -> list:
    """Absolute paths of files touched by HEAD (for the post-commit hook / --changed)."""
    try:
        out = subprocess.run(
            ["git", "-C", root, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    return [os.path.join(root, line.strip().replace("/", os.sep))
            for line in out.stdout.splitlines() if line.strip()]


def index_repo(conn, root: str, repo: str = None, changed: list = None,
               force: bool = False, exclude: list = None) -> dict:
    """Index (or re-index) a repo.

    Returns {repo, files, chunks, vectors, skipped_deleted, unchanged, pruned}.

    changed=None -> full walk. changed=[abs paths] -> only those files (incremental);
    a path in the list that no longer exists is deindexed.

    A file whose bytes hash to what's already stored is SKIPPED — re-embedding
    unchanged code is the expensive part (the embedder runs at single-digit
    chunks/sec), and it's what made a repeat sweep across every project cost as
    much as the first one. `force=True` re-embeds regardless. On a full walk,
    files that have vanished from the repo (deleted, or newly gitignored) are
    pruned so the index can't serve code that no longer exists.
    """
    migrate(conn)
    root = os.path.abspath(root)
    repo = repo or os.path.basename(os.path.normpath(root))

    if changed is None:
        files = list(_iter_source_files(root))
        deleted = []
    else:
        files, deleted = [], []
        for p in changed:
            p = os.path.abspath(p)
            ext = os.path.splitext(p)[1].lower()
            if ext not in CODE_EXTS:
                continue
            (files if os.path.isfile(p) else deleted).append(p)

    if exclude:
        prefixes = tuple(e.replace("\\", "/").strip("/") + "/" for e in exclude if e)
        files = [p for p in files
                 if not os.path.relpath(p, root).replace(os.sep, "/").startswith(prefixes)]

    known = _unchanged_files(conn, repo)
    embedder_up = qwen_embed.qwen_available()
    all_rows, hashed, n_files, n_unchanged = [], [], 0, 0
    for p in files:
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        fh = _file_hash(p)
        prev = known.get(rel)
        # Skip only if the content matches AND we aren't owed vectors we can
        # now actually produce (a BM25-only pass upgrades on the next run).
        if not force and prev and prev[0] == fh and (prev[1] or not embedder_up):
            n_unchanged += 1
            continue
        rows = _index_file(conn, repo, root, p)
        all_rows.extend(rows)
        hashed.append((rel, fh, len(rows)))
        n_files += 1
    for p in deleted:
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        deindex_path(conn, repo, rel)

    n_pruned = 0
    if changed is None:
        seen = {os.path.relpath(p, root).replace(os.sep, "/") for p in files}
        for rel in [r[0] for r in conn.execute(
                "SELECT path FROM code_chunks WHERE repo = ? GROUP BY path", (repo,))]:
            if rel not in seen:
                deindex_path(conn, repo, rel)
                n_pruned += 1

    n_vec = _store_vectors(conn, all_rows)
    # All-or-nothing per sweep: a partial embed leaves `embedded` false so the
    # next run retries those files rather than silently leaving them BM25-only.
    embedded = bool(all_rows) and n_vec == len(all_rows)
    for rel, fh, _n in hashed:
        _mark_file(conn, repo, rel, fh, embedded)

    return {"repo": repo, "files": n_files, "chunks": len(all_rows),
            "vectors": n_vec, "skipped_deleted": len(deleted),
            "unchanged": n_unchanged, "pruned": n_pruned}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def bm25(conn: sqlite3.Connection, query: str, k: int = 8, repo: str = None) -> list:
    """Classic BM25 over code_postings / code_docstats. Returns [(chunk_id, score)]."""
    migrate(conn)
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []
    N = conn.execute("SELECT COUNT(*) FROM code_docstats").fetchone()[0]
    if N == 0:
        return []
    avg_row = conn.execute("SELECT AVG(length) FROM code_docstats").fetchone()
    avgdl = avg_row[0] if avg_row[0] is not None else 0.0
    if avgdl == 0.0:
        return []

    repo_ids = _repo_chunk_ids(conn, repo)
    scores = {}
    for term in query_tokens:
        df = conn.execute(
            "SELECT COUNT(*) FROM code_postings WHERE term = ?", (term,)).fetchone()[0]
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        rows = conn.execute(
            "SELECT p.chunk_id, p.tf, d.length FROM code_postings p"
            " JOIN code_docstats d ON d.chunk_id = p.chunk_id WHERE p.term = ?",
            (term,)).fetchall()
        for cid, tf, dl in rows:
            if repo_ids is not None and cid not in repo_ids:
                continue
            tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avgdl))
            scores[cid] = scores.get(cid, 0.0) + idf * tf_norm
    return sorted(scores.items(), key=lambda x: -x[1])[:k]


def vector_search(conn: sqlite3.Connection, query: str, k: int = 8, repo: str = None) -> list:
    """Qwen dense cosine over code_embeddings (this model's rows only). [(chunk_id, score)].
    Server down / disabled -> [] (silent fallback to BM25)."""
    migrate(conn)
    q_vec = qwen_embed.encode_query(query)
    if not q_vec:
        return []
    repo_ids = _repo_chunk_ids(conn, repo)
    rows = conn.execute(
        "SELECT chunk_id, vec FROM code_embeddings WHERE model = ?",
        (qwen_embed.MODEL_NAME,)).fetchall()
    scored = []
    for cid, blob in rows:
        if repo_ids is not None and cid not in repo_ids:
            continue
        scored.append((cid, _cosine(q_vec, _blob_to_vec(blob))))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def _repo_chunk_ids(conn, repo):
    """Set of chunk ids in *repo*, or None (= no repo filter)."""
    if not repo:
        return None
    return {r[0] for r in conn.execute(
        "SELECT id FROM code_chunks WHERE repo = ?", (repo,))}


def retrieve(conn: sqlite3.Connection, query: str, k: int = 8, repo: str = None) -> list:
    """Hybrid BM25 + Qwen-dense, RRF-fused. Returns up to k rows as
    (repo, path, symbol, start_line, end_line, content) tuples, ranked.
    Qwen down -> BM25-only, no crash."""
    migrate(conn)
    k = max(0, int(k))
    if k == 0:
        return []
    n_idx = conn.execute("SELECT COUNT(*) FROM code_docstats").fetchone()[0]
    if n_idx == 0:
        return []
    scan_k = max(n_idx, k)
    bm25_ranks = bm25(conn, query, k=scan_k, repo=repo)
    vec_ranks = vector_search(conn, query, k=scan_k, repo=repo)
    fused = _rrf_fuse(bm25_ranks, vec_ranks)
    top = [cid for cid, _ in fused[:k]]
    if not top:
        return []
    placeholders = ",".join("?" * len(top))
    rows = conn.execute(
        "SELECT id, repo, path, symbol, start_line, end_line, content"
        f" FROM code_chunks WHERE id IN ({placeholders})", top).fetchall()
    by_id = {r[0]: r[1:] for r in rows}
    return [by_id[cid] for cid in top if cid in by_id]


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def _cmd_code_index(args):
    if not _enabled():
        print("[code-index] disabled — enable the code-RAG (panel settings toggle "
              "or MEMSOM_CODE_RAG=1)")
        return 0
    if args.changed and not _auto_reindex_enabled():
        # The post-commit hook uses --changed; honor the auto-reindex toggle so a commit
        # doesn't reindex when the user has turned that off (search itself stays live).
        print("[code-index] auto-reindex disabled — skipping (--changed)")
        return 0
    conn = memsom.get_connection()
    try:
        migrate(conn)
        changed = None
        if args.changed:
            changed = _git_changed(os.path.abspath(args.path))
            if not changed:
                print("[code-index] --changed: nothing in HEAD to reindex")
                return 0
        warm = qwen_embed.qwen_available()
        if not warm:
            print("[code-index] qwen embedder unreachable — indexing BM25-only "
                  "(dense vectors skipped; re-run when the server is up)")
        stats = index_repo(conn, args.path, repo=args.repo, changed=changed)
        print(f"[code-index] repo={stats['repo']} files={stats['files']} "
              f"chunks={stats['chunks']} vectors={stats['vectors']}"
              + (f" deleted={stats['skipped_deleted']}" if stats['skipped_deleted'] else ""))
        return 0
    finally:
        conn.close()


def _fmt_stats(stats: dict) -> str:
    extra = ""
    if stats.get("unchanged"):
        extra += f" unchanged={stats['unchanged']}"
    if stats.get("pruned"):
        extra += f" pruned={stats['pruned']}"
    if stats.get("skipped_deleted"):
        extra += f" deleted={stats['skipped_deleted']}"
    return (f"files={stats['files']} chunks={stats['chunks']} "
            f"vectors={stats['vectors']}" + extra)


def _cmd_code_index_all(args):
    """Sweep every registered repo. This is the 'index all my projects' entry
    point — cheap to re-run because unchanged files are skipped."""
    if not _enabled():
        print("[code-index] disabled — enable the code-RAG (panel settings toggle "
              "or MEMSOM_CODE_RAG=1)")
        return 0
    repos = registered_repos()
    if not repos:
        print("[code-index] no repos registered — add them first:\n"
              "             memsom code-repos add <path>\n"
              "             memsom code-repos discover <root>")
        return 1
    if not qwen_embed.qwen_available():
        print("[code-index] qwen embedder unreachable — indexing BM25-only "
              "(dense vectors skipped; re-run when the server is up)")
    conn = memsom.get_connection()
    try:
        migrate(conn)
        totals = {"files": 0, "chunks": 0, "vectors": 0, "unchanged": 0, "pruned": 0}
        missing = []
        for r in repos:
            if not os.path.isdir(r["path"]):
                missing.append(r)
                print(f"[code-index] {r['name']}: MISSING ({r['path']}) — skipped")
                continue
            if args.hooks:
                _install_hook(r["path"], quiet=True)
            stats = index_repo(conn, r["path"], repo=r["name"], force=args.force,
                               exclude=r.get("exclude"))
            for k in totals:
                totals[k] += stats.get(k, 0)
            print(f"[code-index] {r['name']:28} {_fmt_stats(stats)}")
        print(f"[code-index] {len(repos) - len(missing)} repo(s): "
              f"files={totals['files']} chunks={totals['chunks']} "
              f"vectors={totals['vectors']} unchanged={totals['unchanged']}"
              + (f" pruned={totals['pruned']}" if totals["pruned"] else ""))
        return 0
    finally:
        conn.close()


def _cmd_code_repos(args):
    if args.action == "list":
        repos = registered_repos()
        if not repos:
            print("[code-repos] none registered")
            return 0
        conn = memsom.get_connection()
        try:
            migrate(conn)
            counts = dict(conn.execute(
                "SELECT repo, COUNT(*) FROM code_chunks GROUP BY repo").fetchall())
        finally:
            conn.close()
        for r in repos:
            mark = "" if os.path.isdir(r["path"]) else "  [MISSING]"
            print(f"{r['name']:28} chunks={counts.get(r['name'], 0):6d}  {r['path']}{mark}")
        return 0

    if args.action == "add":
        if not args.path:
            print("[code-repos] add needs a path")
            return 1
        res = add_repo(args.path, args.name,
                       exclude=[x for x in (args.exclude or "").split(",") if x.strip()])
        print(f"[code-repos] {'added' if res['added'] else 'already registered'}: "
              f"{res['repo']['name']}  {res['repo']['path']}")
        return 0

    if args.action == "remove":
        if not args.path:
            print("[code-repos] remove needs a path or name")
            return 1
        ok = remove_repo(args.path)
        print(f"[code-repos] {'removed' if ok else 'not registered'}: {args.path}")
        return 0 if ok else 1

    if args.action == "discover":
        if not args.path:
            print("[code-repos] discover needs a root directory")
            return 1
        found = discover_repos(args.path, depth=args.depth)
        if not found:
            print(f"[code-repos] no git repos under {args.path}")
            return 0
        for path in found:
            res = add_repo(path)
            print(f"[code-repos] {'added' if res['added'] else 'already registered'}: "
                  f"{res['repo']['name']:28} {path}")
        print(f"[code-repos] {len(found)} repo(s) under {args.path}; "
              f"run `memsom code-index-all` to index them")
        return 0
    return 1


def _cmd_code_search(args):
    if not _enabled():
        print("[code-search] disabled — enable the code-RAG (panel settings toggle "
              "or MEMSOM_CODE_RAG=1)")
        return 0
    conn = memsom.get_connection()
    try:
        migrate(conn)
        results = retrieve(conn, args.query, k=args.k, repo=args.repo)
        if not results:
            print("[code-search] no results")
            return 0
        for repo, path, symbol, start, end, content in results:
            head = content.strip().splitlines()[0] if content.strip() else ""
            print(f"{repo}/{path}:{start}-{end}  {symbol}")
            print(f"      {head[:100]}")
        return 0
    finally:
        conn.close()


# Explicit interpreter + module form, not a bare `memsom` command: memsom is an editable
# install (importable anywhere) but exposes no console script on this machine, and a git
# hook's PATH is minimal. Needs MEMSOM_CODE_RAG=1 in the environment (set user-wide) or it
# no-ops. Overridable via the MEMSOM_PY env var if the interpreter ever moves.
_HOOK_BODY = """#!/bin/sh
# memsom code-RAG auto-reindex (installed by `memsom code-index-hook install`).
# Re-embeds only the files changed by this commit, in the background.
PY="${MEMSOM_PY:-C:/Program Files/Python312/python.exe}"
"$PY" -m memsom.interface.cli code-index "$(git rev-parse --show-toplevel)" --changed >/dev/null 2>&1 &
exit 0
"""


def _install_hook(path: str, quiet: bool = False) -> int:
    repo = os.path.abspath(path)
    gitdir = os.path.join(repo, ".git")
    if not os.path.isdir(gitdir):
        if not quiet:
            print(f"[code-index-hook] not a git repo: {repo}")
        return 1
    hooks = os.path.join(gitdir, "hooks")
    os.makedirs(hooks, exist_ok=True)
    hook = os.path.join(hooks, "post-commit")
    if os.path.exists(hook) and "memsom code-RAG" not in open(hook, encoding="utf-8", errors="replace").read():
        print(f"[code-index-hook] a post-commit hook already exists and isn't ours: {hook}\n"
              f"                  add this line yourself:\n"
              f'                  memsom code-index "$(git rev-parse --show-toplevel)" --changed &')
        return 1
    with open(hook, "w", encoding="utf-8", newline="\n") as f:
        f.write(_HOOK_BODY)
    try:
        os.chmod(hook, 0o755)
    except OSError:
        pass
    if not quiet:
        print(f"[code-index-hook] installed post-commit auto-reindex on {repo}")
    return 0


def _cmd_code_hook_install(args):
    if not _enabled():
        print("[code-index-hook] disabled — set MEMSOM_CODE_RAG=1 first")
        return 0
    return _install_hook(args.path)


# ---------------------------------------------------------------------------
# register / main
# ---------------------------------------------------------------------------

def register(subparsers) -> None:
    """Mount code-index / code-search / code-index-hook onto *subparsers*."""
    p_idx = subparsers.add_parser("code-index", help="index a repo into the code-RAG")
    p_idx.add_argument("path", help="repo root to index")
    p_idx.add_argument("--repo", default=None, help="repo name tag (default: dir name)")
    p_idx.add_argument("--changed", action="store_true",
                       help="only reindex files changed by HEAD (git; for the post-commit hook)")
    p_idx.set_defaults(func=_cmd_code_index)

    p_all = subparsers.add_parser(
        "code-index-all", help="index every registered repo (see `code-repos`)")
    p_all.add_argument("--force", action="store_true",
                       help="re-embed every file, even unchanged ones")
    p_all.add_argument("--hooks", action="store_true",
                       help="also install the post-commit auto-reindex hook on each repo")
    p_all.set_defaults(func=_cmd_code_index_all)

    p_repos = subparsers.add_parser(
        "code-repos", help="manage which repos the code-RAG indexes")
    p_repos.add_argument("action", choices=["list", "add", "remove", "discover"])
    p_repos.add_argument("path", nargs="?", default=None,
                         help="repo root (add/remove) or search root (discover)")
    p_repos.add_argument("--name", default=None, help="repo tag (default: dir name)")
    p_repos.add_argument("--depth", type=int, default=3,
                         help="discover: how many levels to search (default 3)")
    p_repos.add_argument("--exclude", default=None,
                         help="add: comma-separated repo-relative dirs to keep out "
                              "of the index (for non-git trees)")
    p_repos.set_defaults(func=_cmd_code_repos)

    p_srch = subparsers.add_parser("code-search", help="semantic + BM25 search over indexed code")
    p_srch.add_argument("query")
    p_srch.add_argument("--k", type=int, default=8, help="max results (default 8)")
    p_srch.add_argument("--repo", default=None, help="restrict to one repo")
    p_srch.set_defaults(func=_cmd_code_search)

    p_hook = subparsers.add_parser("code-index-hook", help="install/manage the git auto-reindex hook")
    p_hook.add_argument("action", choices=["install"])
    p_hook.add_argument("path", help="repo root")
    p_hook.set_defaults(func=lambda a: _cmd_code_hook_install(a))


def main(argv=None) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(prog="memsom_code_index")
    sub = p.add_subparsers(dest="command", required=True)
    register(sub)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
