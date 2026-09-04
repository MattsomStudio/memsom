"""bge_client — stdlib HTTP client for a LOCAL BGE-M3 embedding supervisor,
plus cold-start-on-demand of that supervisor.

memsom's bge-m3 backend can embed one of two ways (see embed._dispatch_encode):
in-process via FlagEmbedding (the `bge` pip extra), or over HTTP to a local
supervisor that owns the torch model. This module is that HTTP client.

WHY A SUPERVISOR AND NOT IN-PROCESS (2026-09-04). The torch/CUDA model is a
~5 GB process that must never live inside memsom's own process on a GPU box:
the MCP server is a Claude Code CHILD, and a heavy GPU child crashes the CLI;
the in-process cold-start attempt (commit 50a5bac) segfaulted. So the model
lives in a DETACHED supervisor process (its own process group, no console, no
pipes back to us) that memsom only talks to over loopback HTTP. When it is
down, `ensure_supervisor()` launches `retrieval.bge_spawn_cmd` detached and
waits for /health; the supervisor keeps its torch backend alive for
`retrieval.bge_idle_ttl` seconds after our last encode (sent as `idle_ttl` on
every /embed) and then idle-kills it to free VRAM. Idle is NOT a degradation —
the next query cold-starts it; only a spawn/embed FAILURE degrades.

PORTABILITY / SECURITY: memsom only ever talks to the LOCAL endpoint in
`retrieval.bge_url` (localhost by default). There is NO mesh, cross-host, or
service-discovery logic here — a machine that wants to reach a supervisor on
another host points bge_url at its own localhost and runs that hop itself
(e.g. an SSH tunnel or a supervisor that also binds loopback). A fresh clone
with nothing listening and no spawn_cmd just fails the fast /health probe and
the caller falls back to in-process torch, then BM25 — never a hang, never a
crash.

Wire contract (bge_service.py / bge_supervisor.py):
  POST /embed {"input": <str>, "idle_ttl": <int>} ->
  {"dense":[[float,...]], "sparse":[{tokid:weight}],
   "colbert_b64":[b64], "colbert_shape":[[n_tokens,dim]]}
colbert is little-endian fp16 (struct 'e'), exactly memsom's own colbert blob
format. GET /health is answered WITHOUT loading the model, so the probe is cheap.

Pure stdlib (urllib via effects/net, Popen via effects/proc) — importing this
never pulls torch/numpy, so memsom's core stays stdlib and the frozen import
graph is untouched.

Public API
----------
configured() -> bool
bge_http_available(force=False) -> bool
supervisor_reachable() -> bool          # cached-positive / fresh-negative probe
ensure_supervisor() -> bool             # reachable, spawning + awaiting it if allowed
encode_http(text) -> dict | None        # {'dense','sparse','colbert'} or None
idle_ttl() / spawn_cmd() / spawn_timeout()
"""
import base64
import json
import shlex
import struct
import sys
import threading
import time

from memsom.effects import net as memsom_net
from memsom.effects import proc as memsom_proc
from memsom import tuning as memsom_tuning

# /health must fail FAST so a box with no supervisor lands on in-process quickly.
# MEASURED 2026-09-04 (Windows 11 loopback): a CLOSED port does NOT refuse
# instantly — the connect sits in SYN retry until this cap (2.01-2.03 s, 3/3
# runs), while a live supervisor answers in ~2 ms. So a negative probe costs the
# whole cap, which is why negatives are cached (_NEG_TTL) and the spawn poll
# uses the shorter SPAWN_PROBE_TIMEOUT.
HEALTH_TIMEOUT = 2
SPAWN_PROBE_TIMEOUT = 0.5   # per attempt inside the spawn poll loop (loopback)
# The embed POST tolerates a cold model load inside the supervisor (~20-40s) and
# matches the supervisor's own 180s backend-proxy timeout.
EMBED_TIMEOUT = 180
# Bound the text we send: the supervisor does not cap token length, and ColBERT
# stores ~2 KB/token, so a pathologically long node would balloon storage. The
# chunker already keeps nodes small; this is the backstop (mirrors qwen_embed CAP).
CAP = 8000
# After a spawn attempt that never became healthy, do not launch again for this
# long: a broken spawn_cmd must not respawn on every query (and every signal of
# every query), and a slow-but-coming supervisor is picked up by the probe.
SPAWN_COOLDOWN_S = 60.0
SPAWN_POLL_S = 0.25

# Cached tri-state health probe (mirrors qwen_embed / embed._BGE_AVAILABLE):
# None = unknown, True/False = last result. A positive is trusted for
# _PROBE_TTL (encode_http's own failure clears it early); a negative only for
# _NEG_TTL, so a supervisor that comes up is seen within seconds while a hot
# loop (reindex, the 3-4 gates of one retrieve) never pays the 2 s closed-port
# cost more than once per window.
_AVAILABLE = None
_AVAILABLE_AT = 0.0
_PROBE_TTL = 30.0
_NEG_TTL = 5.0

# Single-flight spawn state. A burst of concurrent queries (the MCP server's
# warm listener runs one thread per connection) must launch ONE supervisor.
_SPAWN_LOCK = threading.Lock()
_LAST_SPAWN_FAIL_AT = 0.0   # time.monotonic() of the last spawn that never got healthy
_SPAWN_COUNT = 0            # launches this process performed (diagnostics / tests)
_CHILDREN = []              # Popen handles we launched (kept so GC never warns)


def _url():
    return memsom_tuning.resolve("retrieval.bge_url") or ""


def configured() -> bool:
    """True when a non-empty bge_url is set (localhost by default, so normally True)."""
    return bool(_url().strip())


def _health_url():
    u = _url()
    if u.endswith("/embed"):
        return u[: -len("/embed")] + "/health"
    return u.rstrip("/") + "/health"


def _int_knob(key: str, default: int) -> int:
    """A registered int knob as an int: the typed default (unset), a validated
    env/file string, or a native file value."""
    v = memsom_tuning.resolve(key)
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return default


def idle_ttl() -> int:
    """retrieval.bge_idle_ttl, clamped to the supervisor's own 0..86400 range."""
    return max(0, min(86400, _int_knob("retrieval.bge_idle_ttl", 60)))


def spawn_cmd() -> str:
    """retrieval.bge_spawn_cmd, stripped; '' means never spawn."""
    return str(memsom_tuning.resolve("retrieval.bge_spawn_cmd") or "").strip()


def spawn_timeout() -> int:
    return max(1, _int_knob("retrieval.bge_spawn_timeout", 30))


def _reset_probe() -> None:
    """Test-isolation escape hatch: forget the cached /health result AND the
    spawn cooldown so a test can control the next probe/spawn. Production
    never needs it."""
    global _AVAILABLE, _AVAILABLE_AT, _LAST_SPAWN_FAIL_AT, _SPAWN_COUNT
    _AVAILABLE = None
    _AVAILABLE_AT = 0.0
    _LAST_SPAWN_FAIL_AT = 0.0
    _SPAWN_COUNT = 0


def _forget_health() -> None:
    """Drop only the cached /health verdict (an embed just failed against a
    supervisor the cache called healthy), so the next ensure_supervisor()
    re-probes and, if it is really gone, spawns."""
    global _AVAILABLE, _AVAILABLE_AT
    _AVAILABLE = None
    _AVAILABLE_AT = 0.0


def bge_http_available(force: bool = False, timeout=None) -> bool:
    """True iff the local supervisor answers GET /health. Cached (positive
    _PROBE_TTL, negative _NEG_TTL); never raises. Returns False immediately
    when bge_url is unset. *timeout* overrides HEALTH_TIMEOUT for one probe.

    /health is answered by the supervisor WITHOUT loading the model, so this is
    cheap and never drags torch onto the box just to check reachability."""
    global _AVAILABLE, _AVAILABLE_AT
    if not configured():
        return False
    now = time.time()
    if not force and _AVAILABLE is not None:
        age = now - _AVAILABLE_AT
        if age < (_PROBE_TTL if _AVAILABLE else _NEG_TTL):
            return _AVAILABLE
    try:
        memsom_net.fetch(_health_url(), timeout=HEALTH_TIMEOUT if timeout is None else timeout)
        _AVAILABLE = True
    # FAILOPEN: swallows and marks unavailable -- an unreachable/stalled supervisor means the HTTP path is unavailable; the caller falls back to in-process/BM25.
    except Exception:
        _AVAILABLE = False
    _AVAILABLE_AT = now
    return _AVAILABLE


def supervisor_reachable() -> bool:
    """The cached probe under its asymmetric TTLs: a positive is trusted for
    _PROBE_TTL (encode_http's own failure clears it), a negative only for
    _NEG_TTL, so a supervisor that just came up — or was just spawned — is
    seen within seconds without paying the closed-port cost on every call."""
    return bge_http_available()


def _split_cmd(cmd: str) -> list:
    """shlex the spawn command. On Windows backslashes are path separators,
    not escapes: double them first so shlex's posix rules round-trip every
    `\\` (both inside and outside quotes) back to a single one."""
    if sys.platform == "win32":
        return shlex.split(cmd.replace("\\", "\\\\"), posix=True)
    return shlex.split(cmd, posix=True)


def _launch(cmd: str) -> bool:
    """Start *cmd* DETACHED: its own process group / session, no console, no
    stdio pipes back to us (a pipe would tie the torch process's lifetime and
    output to THIS process, which is exactly what must never happen — see
    feedback_isolate_gpu_processes_from_claude_code). The child's env carries
    BGE_PROC_IDLE_SEC=<bge_idle_ttl> so a supervisor launched by memsom idles
    on memsom's knob even before its first /embed. Returns False when the
    executable cannot be started; a wrapper that exits immediately (a .vbs /
    .cmd launcher) is fine — health is what we wait for, not the pid."""
    try:
        argv = _split_cmd(cmd)
    except ValueError:
        return False
    if not argv:
        return False
    kwargs = dict(stdin=memsom_proc.DEVNULL, stdout=memsom_proc.DEVNULL,
                  stderr=memsom_proc.DEVNULL, close_fds=True)
    if sys.platform == "win32":
        kwargs["creationflags"] = (memsom_proc.DETACHED_PROCESS
                                   | memsom_proc.CREATE_NEW_PROCESS_GROUP
                                   | memsom_proc.CREATE_NO_WINDOW)
    else:
        kwargs["start_new_session"] = True
    try:
        child = memsom_proc.popen(argv, env={"BGE_PROC_IDLE_SEC": str(idle_ttl())}, **kwargs)
    except (OSError, ValueError):
        return False
    # Keep the handle: dropping a Popen whose child still runs raises a
    # ResourceWarning at GC. Prune only children that have already exited
    # (launcher wrappers exit at once; spawns are one-per-outage, so this
    # list never grows past a handful).
    _CHILDREN[:] = [c for c in _CHILDREN if c.poll() is None]
    _CHILDREN.append(child)
    return True


def ensure_supervisor() -> bool:
    """True iff the local supervisor answers /health — spawning it first when
    it is down and `retrieval.bge_spawn_cmd` is set, then waiting up to
    `retrieval.bge_spawn_timeout` for it to come up.

    Single-flight: one launch per outage across every thread of this process
    (concurrent callers wait on the lock and then see the healthy probe), and
    a launch that never became healthy is not retried for SPAWN_COOLDOWN_S.
    Never raises. Returns False when bge_url is unset, no spawn_cmd is
    configured (the fresh-clone contract: exactly today's behaviour), the
    command cannot start, or the supervisor did not answer in time — in which
    case the spawn keeps running on its own and a later query is served."""
    global _LAST_SPAWN_FAIL_AT, _SPAWN_COUNT
    if not configured():
        return False
    if supervisor_reachable():
        return True
    cmd = spawn_cmd()
    if not cmd:
        return False
    with _SPAWN_LOCK:
        # A concurrent caller may have spawned and awaited while we waited.
        if bge_http_available(force=True, timeout=SPAWN_PROBE_TIMEOUT):
            return True
        now = time.monotonic()
        if _LAST_SPAWN_FAIL_AT and (now - _LAST_SPAWN_FAIL_AT) < SPAWN_COOLDOWN_S:
            return False
        if _launch(cmd):
            _SPAWN_COUNT += 1
            deadline = now + spawn_timeout()
            while time.monotonic() < deadline:
                time.sleep(SPAWN_POLL_S)
                if bge_http_available(force=True, timeout=SPAWN_PROBE_TIMEOUT):
                    return True
        _LAST_SPAWN_FAIL_AT = time.monotonic()
        return False


def _deserialize_colbert(b64: str, shape) -> list:
    """base64 fp16 LE -> [n_tokens][dim] list of floats (matches embed.blob_to_colbert)."""
    if not b64 or not shape:
        return []
    n_tokens, dim = int(shape[0]), int(shape[1])
    if n_tokens <= 0 or dim <= 0:
        return []
    raw = base64.b64decode(b64)
    flat = struct.unpack(f"<{n_tokens * dim}e", raw)
    return [list(flat[i * dim:(i + 1) * dim]) for i in range(n_tokens)]


def encode_http(text: str):
    """Embed one string over HTTP. Returns {'dense','sparse','colbert'} (the same
    shape embed._encode returns) or None on any error — never raises.

    The supervisor always computes all three signals in one forward pass; which
    ones are STORED/SCORED is decided by the caller's bge_dense/sparse/colbert
    toggles, not here. `idle_ttl` rides on every request: the supervisor takes
    the last real request's value as its idle-kill window."""
    body = json.dumps({"input": text[:CAP], "idle_ttl": idle_ttl()}).encode("utf-8")
    try:
        raw = memsom_net.fetch(_url(), data=body,
                               headers={"Content-Type": "application/json"},
                               timeout=EMBED_TIMEOUT)
        data = json.loads(raw)
        dense = [float(x) for x in data["dense"][0]]
        sparse = {str(k): float(v) for k, v in data["sparse"][0].items()}
        colbert = _deserialize_colbert(
            (data.get("colbert_b64") or [None])[0],
            (data.get("colbert_shape") or [None])[0],
        )
        return {"dense": dense, "sparse": sparse, "colbert": colbert}
    # FAILOPEN: swallows and returns None -- supervisor down/errored or a malformed reply degrades this embed to the in-process/BM25 fallback, never crashes; the cached health verdict is dropped so the next call re-probes (and spawns if it is gone).
    except Exception:
        _forget_health()
        return None
