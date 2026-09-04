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
The endpoint serves `retrieve`, through the same pool filters as
`memsom retrieve` (`retrieval.retrieve.retrieve`: taint_filter_clauses +
clearance), and `ping` (liveness; never touches the DB). No writes, no other
tools, no file paths in or out.
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
from memsom.kernel.frontmatter import split_frontmatter, fm_top_level
from memsom import tuning as memsom_tuning

LOOPBACK_PEERS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
MAX_REQUEST_BYTES = 64 * 1024
MAX_K = 20
CONNECT_BUDGET_S = 0.25      # loopback accept is ~instant; longer means nobody is home
DEFAULT_CLEARANCE = "topsecret"


def endpoint_file(db_path=None) -> Path:
    """`<db>.warm.json` next to the store the server opened."""
    db = Path(db_path or memsom.db_path())
    return db.with_name(db.name + ".warm.json")


def disabled_by_env() -> bool:
    raw = (memsom_tuning.resolve("retrieval.warm_disabled") or "").strip().lower()
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
    fm_lines, _body, _had = split_frontmatter(content or "")
    fm = fm_top_level(fm_lines)
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
    method = req.get("method")
    if method == "ping":
        return {"pong": True, "pid": os.getpid()}      # liveness only; no DB
    if method != "retrieve":
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
    # FAILOPEN: refused (not served) -- an invalid clearance is a client error.
    except Exception:
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
#
# WEDGE HISTORY (2026-08-20). The live MCP server's listener stayed LISTENING
# for hours while connects piled up in CLOSE_WAIT and were never served: the
# hook's connect() succeeded, so the "unavailable -> fall back" branch never
# fired and every prompt burned its whole deadline for nothing. Three layers
# now bound that failure: (1) every connection runs in its own daemon thread
# with a <= 300 ms socket timeout and is closed in `finally`; (2) the accept
# loop is wrapped so a handler/accept exception is logged, never fatal, and a
# loop that exits for any reason is re-entered; (3) the MCP server runs a
# watchdog that pings the endpoint (no DB work) and restarts it on failure,
# while the client caps the warm path at WARM_BUDGET_S and backs off for
# BACKOFF_S after two consecutive failures against a live pid.

CONN_TIMEOUT_S = 0.3         # per-connection recv/send timeout on the server
WARM_BUDGET_S = 0.25         # client: total connect+send+recv budget
BACKOFF_S = 30.0             # client: skip the warm path this long after 2 failures
BACKOFF_AFTER = 2            # consecutive failures before the backoff engages
WATCHDOG_INTERVAL_S = 60.0   # MCP-side self-ping cadence
PING_TIMEOUT_S = 1.0         # watchdog ping budget (off the prompt path)


def _log(msg: str) -> None:
    try:
        import sys
        print(f"[memsom-warm] {msg}", file=sys.stderr, flush=True)
    # FAILOPEN: allowed, a log line must never raise past its daemon thread.
    except Exception:
        pass


class _Handler(socketserver.StreamRequestHandler):
    timeout = CONN_TIMEOUT_S   # applied to the connection socket: bounds recv AND send

    def handle(self):
        server = self.server
        peer_ip = self.client_address[0]
        try:
            try:
                raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
            except (OSError, socket.timeout):
                return            # client connected but never sent a line
            if not raw:
                return
            try:
                resp = handle_request(raw, peer_ip, server.token, server.open_conn)
            # FAILOPEN: allowed, one worker's crash must never take the listener down.
            except Exception as exc:
                _log(f"handler error: {exc!r}")
                resp = {"error": "internal", "detail": exc.__class__.__name__}
            try:
                self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            except (OSError, socket.timeout):
                pass
        finally:
            try:
                self.connection.close()
            except OSError:
                pass


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    block_on_close = False     # do not keep a reference to every finished worker
    allow_reuse_address = False
    request_queue_size = 16

    def __init__(self, addr, token, open_conn):
        self.token = token
        self.open_conn = open_conn
        super().__init__(addr, _Handler)

    def handle_error(self, request, client_address):
        # a worker that blew up outside handle() (thread start, finish...):
        # one log line and carry on — never a traceback dump, never fatal.
        import sys
        exc = sys.exc_info()[1]
        _log(f"request error from {client_address}: {exc!r}")


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    tmp.replace(path)


class WarmServer:
    """Loopback retrieval listener. `start()` binds + writes the endpoint
    file; `stop()` tears both down. Safe to call stop() twice. `restart()`
    is stop()+start() with a fresh port and token (the watchdog's remedy)."""

    def __init__(self, db_path=None, open_conn=None, host="127.0.0.1"):
        self.db_path = Path(db_path or memsom.db_path())
        self.open_conn = open_conn or (lambda: memsom.get_connection(self.db_path))
        self.host = host
        self.token = secrets.token_hex(32)
        self._server = None
        self._thread = None
        self._stopping = threading.Event()
        self.port = None
        self.restarts = 0

    @property
    def file(self) -> Path:
        return endpoint_file(self.db_path)

    def _serve_loop(self, srv):
        """serve_forever, but an exception is logged and the loop re-entered
        until stop() asks for it; the accept loop must outlive any failure."""
        while not self._stopping.is_set() and self._server is srv:
            try:
                srv.serve_forever(poll_interval=0.25)
                return                       # clean shutdown() exit
            # FAILOPEN: allowed, the accept loop must outlive any failure (WEDGE HISTORY above).
            except Exception as exc:
                _log(f"accept loop error, re-entering: {exc!r}")
                time.sleep(0.05)

    def start(self) -> "WarmServer":
        if self.host not in LOOPBACK_PEERS:
            raise ValueError(f"refusing to bind warm endpoint off loopback: {self.host!r}")
        self._stopping.clear()
        self._server = _Server((self.host, 0), self.token, self.open_conn)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._serve_loop, args=(self._server,),
                                        name="memsom-warm", daemon=True)
        self._thread.start()
        self._write_endpoint()
        clear_backoff(self.db_path)          # a fresh listener starts with a clean slate
        return self

    def stop(self) -> None:
        self._stopping.set()
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            # FAILOPEN: allowed, stop() must be safe to call twice on an already-torn-down server.
            except Exception:
                pass
        try:
            f = self.file
            if f.exists():
                data = json.loads(f.read_text(encoding="utf-8"))
                # only remove OUR file: same token, or written by this process
                if data.get("token") == self.token or data.get("pid") == os.getpid():
                    f.unlink()
        # FAILOPEN: allowed, best-effort cleanup -- a missing/unreadable file is not an error.
        except Exception:
            pass

    def restart(self) -> "WarmServer":
        self.stop()
        self.token = secrets.token_hex(32)
        self.restarts += 1
        return self.start()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._server is not None

    def _write_endpoint(self) -> None:
        body = {"host": self.host, "port": self.port, "token": self.token,
                "pid": os.getpid(), "db": str(self.db_path), "version": 1}
        self.file.parent.mkdir(parents=True, exist_ok=True)
        _write_private(self.file, json.dumps(body))

    def ensure_endpoint_file(self) -> bool:
        """Re-adopt the endpoint file when nobody live owns it. Returns True
        when this server (re)wrote it.

        MEASURED 2026-09-04: six MCP servers (one per Claude session) were all
        listening, but the LAST one to start had written `<db>.warm.json` and
        on exit removed it (correctly -- its own pid). The survivors never
        rewrote it, so the prompt hook found no endpoint and ran BM25-only
        for every one of 1,913 logged prompts since 08-20. The watchdog calls
        this each tick: our own intact file -> nothing; a file whose endpoint
        still answers a ping -> leave it (another live server owns it);
        missing/unreadable/dead -> ours.
        """
        if not self.alive():
            return False
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict):
            if data.get("token") == self.token:
                return False
            if endpoint_answers(data):
                return False
        self._write_endpoint()
        return True

    def ping(self, timeout_s: float = PING_TIMEOUT_S) -> bool:
        """Round-trip a `ping` through the real socket with our own token.
        True iff the listener accepted, read and answered within *timeout_s*."""
        if self.port is None:
            return False
        try:
            resp = _roundtrip(self.host, self.port,
                              {"token": self.token, "method": "ping"}, timeout_s)
        except (OSError, ValueError, WarmUnavailable):
            return False
        return isinstance(resp, dict) and resp.get("pong") is True


class WarmWatchdog:
    """Pings a WarmServer every *interval_s* and restarts it when the ping
    fails (the listener is up but not serving — the 2026-08-20 wedge — or the
    accept thread is gone). `check_once()` is the unit of work; `start()` runs
    it on a daemon thread until `stop()`."""

    def __init__(self, server: WarmServer, interval_s: float = WATCHDOG_INTERVAL_S,
                 ping_timeout_s: float = PING_TIMEOUT_S):
        self.server = server
        self.interval_s = interval_s
        self.ping_timeout_s = ping_timeout_s
        self._stop = threading.Event()
        self._thread = None
        self.failures = 0

    def check_once(self) -> bool:
        """True when the endpoint answered; False when it did not and a
        restart was attempted. A healthy server also re-adopts the endpoint
        file if the sibling that wrote it has exited (see ensure_endpoint_file)."""
        if self.server.ping(self.ping_timeout_s):
            try:
                if self.server.ensure_endpoint_file():
                    _log(f"watchdog: re-adopted endpoint file on port {self.server.port}")
            # FAILOPEN: allowed, a failed re-adopt is retried next tick.
            except Exception as exc:
                _log(f"watchdog: endpoint file re-adopt failed: {exc!r}")
            return True
        self.failures += 1
        _log(f"watchdog: ping failed ({self.failures}), restarting endpoint")
        try:
            self.server.restart()
            _log(f"watchdog: endpoint restarted on {self.server.host}:{self.server.port}")
        # FAILOPEN: allowed, a failed restart is logged; the next tick retries.
        except Exception as exc:
            _log(f"watchdog: restart failed: {exc!r}")
        return False

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                self.check_once()
            # FAILOPEN: allowed, the watchdog thread itself must never die.
            except Exception as exc:
                _log(f"watchdog error: {exc!r}")

    def start(self) -> "WarmWatchdog":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="memsom-warm-watchdog",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def endpoint_answers(ep: dict, budget_s: float = None) -> bool:
    """Does the endpoint described by *ep* (an endpoint-file dict) answer a
    ping with its own token? Cheap loopback roundtrip; False on any failure."""
    if not isinstance(ep, dict) or not isinstance(ep.get("port"), int) \
            or not isinstance(ep.get("token"), str):
        return False
    if ep.get("host", "127.0.0.1") not in LOOPBACK_PEERS:
        return False
    try:
        resp = _roundtrip(ep.get("host", "127.0.0.1"), ep["port"],
                          {"token": ep["token"], "method": "ping"},
                          budget_s if budget_s is not None else PING_TIMEOUT_S)
        return bool(resp.get("pong"))
    except (WarmUnavailable, OSError, ValueError):
        return False


class WarmUnavailable(Exception):
    """No usable endpoint file / connection refused / timed out / bad reply —
    fall back. A slow endpoint is treated as DOWN: the hook must not pay for it."""


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


def _roundtrip(host, port, req: dict, budget_s: float) -> dict:
    """One JSON line out, one in, within *budget_s* total. Raises
    WarmUnavailable on connect failure, timeout, short read or bad JSON."""
    payload = (json.dumps(req) + "\n").encode("utf-8")
    deadline = time.monotonic() + max(0.01, budget_s)
    connect_s = min(CONNECT_BUDGET_S, max(0.01, deadline - time.monotonic()))
    try:
        try:
            s = socket.create_connection((host, port), timeout=connect_s)
        except OSError as exc:
            # Connect-phase failure: nobody is home (a refused port, or on
            # Windows a SYN-retry timeout standing in for refused). Flagged
            # so the client does not count it toward the wedge backoff.
            err = WarmUnavailable("connect failed")
            err.connect_phase = True
            raise err from exc
        with s:
            s.settimeout(max(0.01, deadline - time.monotonic()))
            s.sendall(payload)
            buf = b""
            while not buf.endswith(b"\n"):
                s.settimeout(max(0.01, deadline - time.monotonic()))
                chunk = s.recv(65536)
                if not chunk:
                    raise WarmUnavailable("short read")
                buf += chunk
                if len(buf) > 4 * 1024 * 1024:
                    raise WarmUnavailable("reply too large")
    except socket.timeout as exc:
        raise WarmUnavailable("timed out") from exc
    except OSError as exc:
        raise WarmUnavailable(f"connect/io failed: {exc.__class__.__name__}") from exc
    try:
        resp = json.loads(buf.decode("utf-8"))
    except ValueError as exc:
        raise WarmUnavailable("bad reply") from exc
    if not isinstance(resp, dict):
        raise WarmUnavailable("bad reply")
    return resp


# --- client-side backoff (sidecar `<db>.warm.down.json`) ---------------------
#
# The server rewrites the endpoint file atomically and owns it, so the client
# keeps its failure counter in a sidecar: {"port", "pid", "failures", "until"}.
# A counter is only meaningful against the SAME listener, so a different port
# in the endpoint file invalidates it, and start() clears it.

def backoff_file(db_path=None) -> Path:
    f = endpoint_file(db_path)
    return f.with_name(f.name[:-len(".warm.json")] + ".warm.down.json")


def read_backoff(db_path=None) -> dict | None:
    try:
        data = json.loads(backoff_file(db_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def clear_backoff(db_path=None) -> None:
    try:
        backoff_file(db_path).unlink()
    except OSError:
        pass


def pid_alive(pid) -> bool:
    """Best effort; unknown -> True (a failure already happened, the cheap
    assumption is that the server is wedged, not gone)."""
    if not isinstance(pid, int) or pid <= 0:
        return True
    try:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False
            try:
                code = ctypes.c_ulong()
                if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                    return True
                return code.value == 259                # STILL_ACTIVE
            finally:
                k32.CloseHandle(h)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    # FAILOPEN: allowed, an unreadable process state defaults to "alive" (see docstring).
    except Exception:
        return True


def note_warm_failure(ep: dict, db_path=None, *, now=None, alive=pid_alive) -> int:
    """Record one failed warm call against endpoint *ep*. Returns the
    consecutive-failure count. Once it reaches BACKOFF_AFTER and the server
    pid is alive, arms the backoff window (`until`); a dead pid gets no
    backoff — its endpoint file is stale and connect() refuses instantly."""
    now = time.time() if now is None else now
    cur = read_backoff(db_path) or {}
    failures = (int(cur.get("failures", 0)) + 1) if cur.get("port") == ep.get("port") else 1
    body = {"port": ep.get("port"), "pid": ep.get("pid"), "failures": failures,
            "until": 0.0, "ts": now}
    if failures >= BACKOFF_AFTER and alive(ep.get("pid")):
        body["until"] = now + BACKOFF_S
    try:
        _write_private(backoff_file(db_path), json.dumps(body))
    except OSError:
        pass
    return failures


def in_backoff(ep: dict, db_path=None, *, now=None) -> bool:
    """True while a backoff window armed against THIS endpoint is open."""
    cur = read_backoff(db_path)
    if not cur or cur.get("port") != ep.get("port"):
        return False
    now = time.time() if now is None else now
    return float(cur.get("until") or 0.0) > now


def warm_query(query: str, k: int = 3, clearance: str = DEFAULT_CLEARANCE,
               deadline_s: float = WARM_BUDGET_S, db_path=None) -> list:
    """Ask the warm endpoint. Raises WarmUnavailable on ANY failure — no
    endpoint file, refused, timed out, short read, bad reply, or an open
    backoff window — so the caller falls back. The budget is
    min(deadline_s, WARM_BUDGET_S) for connect + send + recv together.
    Timeouts / short reads / bad replies AFTER connect count toward the
    backoff; a failed connect does not (a stale file, not a wedged server)."""
    ep = read_endpoint(db_path)
    if ep is None:
        raise WarmUnavailable("no endpoint file")
    if in_backoff(ep, db_path):
        raise WarmUnavailable("backoff")
    req = {"token": ep["token"], "method": "retrieve",
           "query": query, "k": k, "clearance": clearance}
    budget = min(WARM_BUDGET_S, max(0.01, deadline_s))
    try:
        resp = _roundtrip(ep.get("host", "127.0.0.1"), ep["port"], req, budget)
    except WarmUnavailable as exc:
        if not getattr(exc, "connect_phase", False):
            note_warm_failure(ep, db_path)   # accepted-but-unserved = the wedge
        raise
    if "hits" not in resp:
        note_warm_failure(ep, db_path)
        raise WarmUnavailable(f"endpoint error: {resp.get('error')}")
    clear_backoff(db_path)
    return resp["hits"]
