"""qwen_embed — HTTP client for the Qwen3-Embedding code-RAG embedder.

Talks to the qwen_supervisor (llama.cpp `llama-server --embedding`) over the mesh,
OpenAI-shaped POST /v1/embeddings. Deliberately SEPARATE from retrieval/embed.py:
that module's backend() is a process-global switch for the FACT store (bge-m3 /
Ollama); the code index must never flip it. This client is the code index's own,
private embedder path.

Pure stdlib (urllib only) — no numpy, no torch — so importing it never adds a
dependency and memsom's core stays stdlib. The vector math (MRL truncate + L2
renormalize) is done in plain Python.

Model facts (from the 2026-07-20 bench, project_code_rag_embedder_bench):
  - Qwen3-Embedding-4B, native dim 2560, truncated to 1024 (Matryoshka) — 1024 is
    within noise of native and 2.5x cheaper to store/score.
  - LAST-TOKEN POOLING: every input MUST end with the EOS token <|endoftext|> or
    pooling reads the wrong token (MRR 0.31 -> 0.74). llama.cpp does NOT append it
    automatically, so we do it here. See reference_qwen3_embedding_llamacpp.
  - Query side gets an instruction prefix; document side is raw.

Degrade contract (mirrors embed.bge_available): unreachable server / any error ->
qwen_available() returns False and encode_* return None-filled lists, so the code
index falls back to BM25-only and never crashes.
"""
import json
import os
import time
import urllib.request
import urllib.error

# The model tag stored in code_embeddings.model — read back with WHERE model=? so a
# future re-embed with a different model can coexist (the dim-collision discipline).
MODEL_NAME = "qwen3-embedding-4b"
TARGET_DIM = 1024          # MRL truncation target
CAP = 1600                 # ~512-token window; matches the bench + bge max_length
EOS = "<|endoftext|>"      # Qwen3-Embedding pools the LAST token -> input must end here
# 6 x ~530 tokens (CAP 1600 chars) ~= 3.2k tokens < the server's 4096 ubatch, so a full
# batch clears in ONE forward pass. Bigger batches overflow ubatch -> llama.cpp 400s ->
# the per-item fallback fires and indexing crawls. Keep BATCH*CAP_tokens under QWEN_UBATCH.
BATCH = 6

# Loopback default keeps machine topology (the Nebula mesh IP) out of the repo — set
# MEMSOM_QWEN_URL to the real supervisor address (e.g. the mesh IP) per host.
DEFAULT_URL = "http://127.0.0.1:11437/v1/embeddings"

# Query-side instruction (doc side is raw). Verbatim from the bench harness.
Q_INSTRUCT = ("Instruct: Given a natural-language description of what a code "
              "function does, retrieve the function that implements it.\nQuery: ")

# Cached tri-state availability probe: None=unknown, True/False=last result. Mirrors
# embed._BGE_AVAILABLE. Re-checked after _PROBE_TTL so a server that comes up later is
# picked up without a restart.
_AVAILABLE = None
_AVAILABLE_AT = 0.0
_PROBE_TTL = 30.0


def _url():
    return os.environ.get("MEMSOM_QWEN_URL") or DEFAULT_URL


def _health_url():
    u = _url()
    # strip the /v1/embeddings tail to reach the supervisor's /health
    for tail in ("/v1/embeddings", "/embeddings"):
        if u.endswith(tail):
            return u[: -len(tail)] + "/health"
    return u.rstrip("/") + "/health"


def qwen_available(force: bool = False) -> bool:
    """True if the qwen supervisor answers /health. Cached for _PROBE_TTL; never raises.

    Note: /health is answered by the supervisor WITHOUT spawning llama-server, so this
    is cheap and does not drag the 8 GB model onto the GPU.
    """
    global _AVAILABLE, _AVAILABLE_AT
    now = time.time()
    if not force and _AVAILABLE is not None and (now - _AVAILABLE_AT) < _PROBE_TTL:
        return _AVAILABLE
    try:
        with urllib.request.urlopen(_health_url(), timeout=3) as r:
            _AVAILABLE = (r.status == 200)
    except Exception:
        _AVAILABLE = False
    _AVAILABLE_AT = now
    return _AVAILABLE


def _post(batch, timeout=180):
    """POST a list of strings to /v1/embeddings, return list[list[float]] (raw, native dim)."""
    body = json.dumps({"input": batch, "model": "q"}).encode("utf-8")
    req = urllib.request.Request(
        _url(), data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())["data"]
    return [row["embedding"] for row in data]


def _mrl(vec):
    """Truncate to TARGET_DIM and L2-renormalize (Matryoshka). Returns list[float]."""
    v = vec[:TARGET_DIM]
    norm = sum(x * x for x in v) ** 0.5
    if norm <= 0.0:
        return v
    return [x / norm for x in v]


def _one(text, timeout=180):
    """Embed a single text, hard-truncating on a persistent 400. Keeps EOS last."""
    base = text[: -len(EOS)] if text.endswith(EOS) else text
    for cap in (CAP, 800, 400):
        try:
            return _post([base[:cap] + EOS], timeout=timeout)[0]
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
    raise RuntimeError("qwen 400 even at 400 chars")


def _embed(texts):
    """Batch to the server; return list[list[float]] MRL-truncated + normalized.

    On a batch 400, fall back to per-item (progressive truncation). Any other failure
    -> return None for the whole call (caller degrades to BM25-only).
    """
    if not texts:
        return []
    prepared = [t[:CAP] + EOS for t in texts]
    out = []
    for i in range(0, len(prepared), BATCH):
        batch = prepared[i:i + BATCH]
        try:
            rows = _post(batch)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                return None
            rows = []
            for t in batch:            # split the offending batch
                try:
                    rows.append(_one(t))
                except Exception:
                    return None
        except Exception:
            return None
        out.extend(rows)
    return [_mrl(v) for v in out]


def encode_docs(texts):
    """Embed document texts (raw + EOS). Returns list[list[float]] (len TARGET_DIM) or
    None if the server is down / errored — never raises."""
    if not qwen_available():
        return None
    try:
        return _embed(list(texts))
    except Exception:
        return None


def encode_query(text):
    """Embed one query (instruction prefix + EOS). Returns list[float] or None."""
    if not qwen_available():
        return None
    try:
        vecs = _embed([Q_INSTRUCT + text])
    except Exception:
        return None
    if not vecs:
        return None
    return vecs[0]
