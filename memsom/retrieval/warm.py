"""warm — the loopback-only retrieval endpoint the prompt hook queries.

WHY. The UserPromptSubmit hook runs before EVERY prompt and blocks the turn
until it exits, so it has a sub-second budget. A cold memsom process that
imports the bge backend pays a multi-second model load; even BM25-only, a
fresh interpreter + sqlite open is tens to hundreds of milliseconds. The MCP
server is already a long-lived process with a warm backend, so it also serves
retrieval over a tiny local socket, and `memsom hook-query` asks it first and
falls back to in-process BM25 only when it is not there.

TRANSPORT. One TCP listener bound to 127.0.0.1 on an ephemeral port (works the
same on Windows and macOS; AF_UNIX is not portable to Windows Python before
3.12 and named pipes are not portable to macOS). One JSON object per line in,
one JSON object per line out, then the connection closes. The port and a
random per-process token are written to an endpoint file next to the DB
(`<db>.warm.json`, owner-only perms where the OS supports them); the client
reads that file, so the two sides agree on the DB by construction.

SECURITY. Three independent refusals, checked in this order before any work:
  1. bind is 127.0.0.1 only — a non-loopback peer cannot connect at all;
  2. the peer address is re-checked per request (defence in depth: a
     misconfigured bind or a future AF change cannot silently widen it);
  3. the request token must match the process token (constant-time compare).
The endpoint serves ONE method, `retrieve`, through the same pool filters as
`memsom retrieve` (`retrieval.retrieve.retrieve`: taint_filter_clauses +
clearance). No writes, no other tools, no file paths in or out.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import socket
import socketserver
import threading
import time
from pathlib import Path

import memsom
from memsom.integrity import confid as memsom_confid
from memsom.retrieval import retrieve as memsom_retrieve

LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
MAX_REQUEST_BYTES = 64 * 1024
MAX_K = 20
CONNECT_BUDGET_S = 0.25      # loopback accept is ~instant; longer means nobody is home
DEFAULT_CLEARANCE = "topsecret"
ENV_DISABLE = "MEMDAG_WARM_ENDPOINT"   # "0" / "off" disables the listener


def endpoint_file(db_path=None) -> Path:
    """`<db>.warm.json` next to the store the server opened."""
    db = Path(db_path or memsom.db_path())
    return db.with_name(db.name + ".warm.json")


def disabled_by_env() -> bool:
    raw = (os.environ.get(ENV_DISABLE) or "").strip().lower()
    return raw in ("0", "off", "false", "no")


# ---------------------------------------------------------------------------
# Hit computation — shared by the warm server and the in-process fallback
# ---------------------------------------------------------------------------

def _stem_of(source_ref):
    """`memory:<stem>` -> stem; anything else -> None."""
    if source_ref and source_ref.startswith("memory:") \
            and not source_ref.startswith("memory:literal:"):
        return source_ref[len("memory:"):]
    return None


def _hook_line(content: str) -> str:
    """A one-line hook for a hit: the curated index hook, else the
    frontmatter description, else the first prose line."""
    from memsom.bridge import bridge_import as bi
    fm_lines, _body, _had = bi.split_frontmatter(content or "")
    fm = bi.fm_top_level(fm_lines)
    for key in ("index_hook", "description"):
        v = (fm.get(key) or "").strip().strip('"').strip("'")
        if v:
            return " ".join(v.split())
    return memsom.snippet(content or "", width=90)


def coverage_scores(conn, query: str, nids) -> dict:
    """BM25 *coverage* in [0, 1] for each nid: the node's raw BM25 score over
    the best score any document could reach for this query (every term
    present with saturated tf). Backend-independent, query-length-independent,
    so one floor means the same thing on the bm25 fallback and the warm bge
    path. Query terms absent from the corpus still count in the denominator —
    a prompt made of words the store has never seen is NOT relevant to it.
    """
    nids = list(nids)
    if not nids:
        return {}
    memsom_retrieve.migrate(conn)
    terms = set(memsom_retrieve.tokenize(query))
    if not terms:
        return {n: 0.0 for n in nids}
    n_docs = conn.execute("SELECT COUNT(*) FROM docstats").fetchone()[0]
    if n_docs == 0:
        return {n: 0.0 for n in nids}
    avg_row = conn.execute("SELECT AVG(length) FROM docstats").fetchone()
    avgdl = avg_row[0] or 0.0
    if avgdl == 0.0:
        return {n: 0.0 for n in nids}
    k1, b = memsom_retrieve.K1, memsom_retrieve.B
    wanted = set(nids)
    raw = {n: 0.0 for n in nids}
    ceiling = 0.0
    for term in terms:
        df = conn.execute("SELECT COUNT(*) FROM postings WHERE term = ?",
                          (term,)).fetchone()[0]
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        ceiling += idf * (k1 + 1.0)
        if df == 0:
            continue
        rows = conn.execute(
            "SELECT p.node_id, p.tf, d.length FROM postings p "
            "JOIN docstats d ON d.node_id = p.node_id WHERE p.term = ?",
            (term,)).fetchall()
        for nid, tf, dl in rows:
            if nid not in wanted:
                continue
            tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            raw[nid] += idf * tf_norm
    if ceiling <= 0:
        return {n: 0.0 for n in nids}
    return {n: min(1.0, s / ceiling) for n, s in raw.items()}


def hits_for(conn, query: str, k: int = 3, clearance: str = DEFAULT_CLEARANCE) -> list:
    """Top-*k* hits as dicts {id, stem, label, hook, score}, ranked by the
    store's own `retrieve` (pool-filtered), scored by BM25 coverage."""
    k = max(0, min(int(k), MAX_K))
    if k == 0 or not (query or "").strip():
        return []
    rows = memsom_retrieve.retrieve(conn, query, k=k, clearance=clearance)
    if not rows:
        return []
    scores = coverage_scores(conn, query, [r[0] for r in rows])
    out = []
    for nid, content, channel, label, source_ref in rows:
        stem = _stem_of(source_ref)
        out.append({
            "id": nid,
            "stem": stem,
            "label": stem or f"mem:{nid}",
            "channel": channel,
            "hook": _hook_line(content),
            "score": round(scores.get(nid, 0.0), 4),
        })
    return out


# ---------------------------------------------------------------------------
# Request handling (pure; unit-testable without sockets)
# ---------------------------------------------------------------------------

def peer_allowed(peer_ip) -> bool:
    return peer_ip in LOOPBACK_PEERS


def handle_request(raw: bytes, peer_ip: str, token: str, open_conn) -> dict:
    """Decode one request line and answer it. *open_conn* is a zero-arg
    callable returning a sqlite connection (closed here after use).
    Refusals never touch the DB."""
    if not peer_allowed(peer_ip):
        return {"error": "forbidden", "detail": "loopback only"}
    if len(raw) > MAX_REQUEST_BYTES:
        return {"error": "too-large"}
    try:
        req = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return {"error": "bad-json"}
    if not isinstance(req, dict):
        return {"error": "bad-json"}
    got = req.get("token")
    if not isinstance(got, str) or not hmac.compare_digest(got, token):
        return {"error": "unauthorized"}
    if req.get("method") != "retrieve":
        return {"error": "unknown-method"}
    query = req.get("query")
    if not isinstance(query, str):
        return {"error": "bad-request", "detail": "query must be a string"}
    try:
        k = int(req.get("k", 3))
    except (TypeError, ValueError):
        return {"error": "bad-request", "detail": "k must be an integer"}
    clearance = req.get("clearance") or DEFAULT_CLEARANCE
    try:
        memsom_confid.parse_conf(clearance)
    except Exception:  # noqa: BLE001 — invalid clearance is a client error
        return {"error": "bad-request", "detail": "invalid clearance"}
    t0 = time.perf_counter()
    conn = open_conn()
    try:
        hits = hits_for(conn, query, k=k, clearance=clearance)
    finally:
        conn.close()
    return {"hits": hits, "ms": round((time.perf_counter() - t0) * 1000, 1)}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _Handler(socketserver.StreamRequestHandler):
    timeout = 5  # seconds a client may take to send its line

    def handle(self):
        server = self.server
        peer_ip = self.client_address[0]
        try:
            raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        except (OSError, socket.timeout):
            return
        try:
            resp = handle_request(raw, peer_ip, server.token, server.open_conn)
        except Exception as exc:  # noqa: BLE001 — never kill the listener
            resp = {"error": "internal", "detail": exc.__class__.__name__}
        try:
            self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            pass


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False
    request_queue_size = 16

    def __init__(self, addr, token, open_conn):
        self.token = token
        self.open_conn = open_conn
        super().__init__(addr, _Handler)


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    tmp.replace(path)


class WarmServer:
    """Loopback retrieval listener. `start()` binds + writes the endpoint
    file; `stop()` tears both down. Safe to call stop() twice."""

    def __init__(self, db_path=None, open_conn=None, host="127.0.0.1"):
        self.db_path = Path(db_path or memsom.db_path())
        self.open_conn = open_conn or (lambda: memsom.get_connection(self.db_path))
        self.host = host
        self.token = secrets.token_hex(32)
        self._server = None
        self._thread = None
        self.port = None

    @property
    def file(self) -> Path:
        return endpoint_file(self.db_path)

    def start(self) -> "WarmServer":
        if self.host not in LOOPBACK_PEERS:
            raise ValueError(f"refusing to bind warm endpoint off loopback: {self.host!r}")
        self._server = _Server((self.host, 0), self.token, self.open_conn)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="memsom-warm", daemon=True)
        self._thread.start()
        body = {"host": self.host, "port": self.port, "token": self.token,
                "pid": os.getpid(), "db": str(self.db_path), "version": 1}
        self.file.parent.mkdir(parents=True, exist_ok=True)
        _write_private(self.file, json.dumps(body))
        return self

    def stop(self) -> None:
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:  # noqa: BLE001
                pass
        try:
            f = self.file
            if f.exists():
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("token") == self.token:   # only remove OUR file
                    f.unlink()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WarmUnavailable(Exception):
    """No usable endpoint file / connection refused / bad reply — fall back."""


def read_endpoint(db_path=None) -> dict | None:
    f = endpoint_file(db_path)
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("port"), int) \
            or not isinstance(data.get("token"), str):
        return None
    if data.get("host", "127.0.0.1") not in LOOPBACK_PEERS:
        return None   # never follow an endpoint file off loopback
    return data


def warm_query(query: str, k: int = 3, clearance: str = DEFAULT_CLEARANCE,
               deadline_s: float = 0.8, db_path=None) -> list:
    """Ask the warm endpoint. Raises WarmUnavailable on any failure (the caller
    decides whether to fall back); raises socket.timeout when the deadline
    passes so the caller can distinguish 'down' from 'slow'."""
    ep = read_endpoint(db_path)
    if ep is None:
        raise WarmUnavailable("no endpoint file")
    payload = json.dumps({"token": ep["token"], "method": "retrieve",
                          "query": query, "k": k, "clearance": clearance}) + "\n"
    deadline = time.monotonic() + max(0.01, deadline_s)
    # Connect gets a SHORT budget of its own: a live loopback listener accepts
    # in microseconds, while a stale endpoint file (server crashed, port
    # closed) can take a full SYN-retry cycle to be refused on Windows —
    # that must read as "unavailable -> fall back", not eat the whole deadline.
    connect_s = min(CONNECT_BUDGET_S, max(0.01, deadline - time.monotonic()))
    try:
        try:
            s = socket.create_connection((ep.get("host", "127.0.0.1"), ep["port"]),
                                         timeout=connect_s)
        except socket.timeout as exc:
            raise WarmUnavailable("connect timed out") from exc
        with s:
            s.settimeout(max(0.01, deadline - time.monotonic()))
            s.sendall(payload.encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                s.settimeout(max(0.01, deadline - time.monotonic()))
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if len(buf) > 4 * 1024 * 1024:
                    raise WarmUnavailable("reply too large")
    except socket.timeout:
        raise
    except OSError as exc:
        raise WarmUnavailable(f"connect/io failed: {exc.__class__.__name__}") from exc
    try:
        resp = json.loads(buf.decode("utf-8"))
    except ValueError as exc:
        raise WarmUnavailable("bad reply") from exc
    if not isinstance(resp, dict) or "hits" not in resp:
        raise WarmUnavailable(f"endpoint error: {resp.get('error') if isinstance(resp, dict) else '?'}")
    return resp["hits"]
