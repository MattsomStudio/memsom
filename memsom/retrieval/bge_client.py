"""bge_client — stdlib HTTP client for a LOCAL BGE-M3 embedding supervisor.

memsom's bge-m3 backend can embed one of two ways (see embed._dispatch_encode):
in-process via FlagEmbedding (the `bge` pip extra), or over HTTP to a local
supervisor that owns the torch model. This module is that HTTP client.

PORTABILITY / SECURITY: memsom only ever talks to the LOCAL endpoint in
`retrieval.bge_url` (localhost by default). There is NO mesh, cross-host, or
service-discovery logic here — a machine that wants to reach a supervisor on
another host points bge_url at its own localhost and runs that hop itself
(e.g. an SSH tunnel or a supervisor that also binds loopback). A fresh clone
with nothing listening just fails the fast /health probe and the caller falls
back to in-process torch, then BM25 — never a hang, never a crash.

Wire contract (bge_service.py): POST /embed {"input": <str>} ->
  {"dense":[[float,...]], "sparse":[{tokid:weight}],
   "colbert_b64":[b64], "colbert_shape":[[n_tokens,dim]]}
colbert is little-endian fp16 (struct 'e'), exactly memsom's own colbert blob
format. GET /health is answered WITHOUT loading the model, so the probe is cheap.

Pure stdlib (urllib via effects/net) — importing this never pulls torch/numpy,
so memsom's core stays stdlib and the frozen import graph is untouched.

Public API
----------
configured() -> bool
bge_http_available(force=False) -> bool
encode_http(text) -> dict | None      # {'dense','sparse','colbert'} or None
"""
import base64
import json
import struct
import time

from memsom.effects import net as memsom_net
from memsom import tuning as memsom_tuning

# /health must fail FAST so a box with no supervisor lands on in-process in ~1s.
# A refused connection returns instantly; this cap only bounds a listener that
# accepts but stalls.
HEALTH_TIMEOUT = 2
# The embed POST tolerates a cold model load inside the supervisor (~20-40s) and
# matches the supervisor's own 180s backend-proxy timeout.
EMBED_TIMEOUT = 180
# Bound the text we send: the supervisor does not cap token length, and ColBERT
# stores ~2 KB/token, so a pathologically long node would balloon storage. The
# chunker already keeps nodes small; this is the backstop (mirrors qwen_embed CAP).
CAP = 8000

# Cached tri-state health probe (mirrors qwen_embed / embed._BGE_AVAILABLE):
# None = unknown, True/False = last result. Re-checked after _PROBE_TTL so a
# supervisor that comes up later is picked up without a restart.
_AVAILABLE = None
_AVAILABLE_AT = 0.0
_PROBE_TTL = 30.0


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


def _reset_probe() -> None:
    """Test-isolation escape hatch: forget the cached /health result so a test
    can control the next probe. Production never needs it."""
    global _AVAILABLE, _AVAILABLE_AT
    _AVAILABLE = None
    _AVAILABLE_AT = 0.0


def bge_http_available(force: bool = False) -> bool:
    """True iff the local supervisor answers GET /health. Cached for _PROBE_TTL;
    never raises. Returns False immediately when bge_url is unset.

    /health is answered by the supervisor WITHOUT loading the model, so this is
    cheap and never drags torch onto the box just to check reachability."""
    global _AVAILABLE, _AVAILABLE_AT
    if not configured():
        return False
    now = time.time()
    if not force and _AVAILABLE is not None and (now - _AVAILABLE_AT) < _PROBE_TTL:
        return _AVAILABLE
    try:
        memsom_net.fetch(_health_url(), timeout=HEALTH_TIMEOUT)
        _AVAILABLE = True
    # FAILOPEN: swallows and marks unavailable -- an unreachable/stalled supervisor means the HTTP path is unavailable; the caller falls back to in-process/BM25.
    except Exception:
        _AVAILABLE = False
    _AVAILABLE_AT = now
    return _AVAILABLE


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
    toggles, not here."""
    body = json.dumps({"input": text[:CAP]}).encode("utf-8")
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
    # FAILOPEN: swallows and returns None -- supervisor down/errored or a malformed reply degrades this embed to the in-process/BM25 fallback, never crashes.
    except Exception:
        return None
