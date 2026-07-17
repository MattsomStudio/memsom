"""VRAM estimation — the "will it fit before I load it" tool.

Two different numbers get confused constantly:

* **measured** VRAM is what nvidia-smi reports right now (handled elsewhere,
  reused from ``telemetry._read_gpu``);
* **estimated** VRAM is what a model *will* cost once loaded at a given context
  length, computed here from the model's architecture metadata — BEFORE you
  commit the load.

The estimate is why the INFERENCE tab's context slider and its VRAM readout are
the same tool: the KV cache grows linearly with context, so moving the slider
must recompute. The formula:

    total = weights + kv_cache + overhead
    weights   = n_params * bytes_per_weight(quant)
    kv_cache  = 2 * n_layers * ctx * n_kv_heads * head_dim * bytes(kv_type)
    head_dim  = embedding_length / n_heads
    overhead  ~ fixed compute/activation buffers

It's an estimate, not a promise — good to roughly ±10-15% for a "fits the
5070's 12 GB?" call, which is all it's for. All inputs come from real model
metadata (Ollama ``/api/show`` model_info or a GGUF header), never a guess.
"""

from __future__ import annotations

import re
from typing import Optional

_MB = 1024 * 1024

# Approximate bytes-per-weight by GGUF quantization tag. GGUF quant names encode
# an effective bits-per-weight (K-quants mix precision across tensors); these are
# the widely-cited effective averages, converted to bytes. Unknown tags fall
# back to FP16 (2.0) — conservative-high, so an unknown never under-warns.
_BITS_PER_WEIGHT = {
    "F32": 32.0, "FP32": 32.0,
    "F16": 16.0, "FP16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q8_1": 8.5, "Q8_K": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.5, "Q5_K_S": 5.5, "Q5_0": 5.5, "Q5_1": 5.5, "Q5_K": 5.5,
    "Q4_K_M": 4.85, "Q4_K_S": 4.58, "Q4_0": 4.55, "Q4_1": 4.8, "Q4_K": 4.85,
    "Q3_K_L": 3.6, "Q3_K_M": 3.45, "Q3_K_S": 3.2, "Q3_K": 3.45,
    "Q2_K": 2.63,
    "IQ4_XS": 4.3, "IQ3_XXS": 3.1, "IQ2_XXS": 2.2,
}

# Bytes per KV-cache element by cache dtype. fp16 is the default (and what most
# servers use unless you explicitly quantize the cache).
_KV_BYTES = {"fp16": 2.0, "f16": 2.0, "q8_0": 1.0, "q8": 1.0, "q4_0": 0.5, "q4": 0.5}

# Flat compute/activation/overhead allowance. Scales a little with context but a
# fixed floor is close enough for the fits-or-not call.
_OVERHEAD_MB_FLOOR = 500.0


def bytes_per_weight(quant: Optional[str]) -> float:
    if not quant:
        return 2.0  # unknown → assume FP16 (conservative-high)
    q = quant.strip().upper()
    if q in _BITS_PER_WEIGHT:
        return _BITS_PER_WEIGHT[q] / 8.0
    # tolerate minor spelling ("Q4_K_M " / "q4_k_m"): try a prefix match
    for tag, bits in _BITS_PER_WEIGHT.items():
        if q.startswith(tag):
            return bits / 8.0
    return 2.0


def parse_params(value) -> Optional[int]:
    """Coerce a parameter count to an int number of weights.

    Accepts an int (already a count), or a human string like "7.6B" / "137M" /
    "70b". Returns None if it can't tell."""
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if not isinstance(value, str):
        return None
    m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*([bBmMkK]?)", value)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    mult = {"b": 1e9, "m": 1e6, "k": 1e3, "": 1.0}[unit]
    return int(num * mult)


def estimate_vram(meta: dict, ctx: int, kv_type: str = "fp16") -> dict:
    """Estimate load-time VRAM in MB for *meta* at context length *ctx*.

    *meta* is the normalized architecture dict an adapter puts on
    ``ModelInfo.meta``:
        n_params, quant, n_layers, n_heads, n_kv_heads, embedding_length
    Missing pieces degrade gracefully: no architecture fields → KV term is 0 and
    the result is flagged ``partial`` so the UI can say "weights only".

    Returns a breakdown dict (all MB) plus ``assumptions`` for display.
    """
    n_params = parse_params(meta.get("n_params"))
    quant = meta.get("quant")
    bpw = bytes_per_weight(quant)

    weights_mb = (n_params * bpw / _MB) if n_params else 0.0

    n_layers = _as_int(meta.get("n_layers"))
    n_kv_heads = _as_int(meta.get("n_kv_heads")) or _as_int(meta.get("n_heads"))
    n_heads = _as_int(meta.get("n_heads")) or n_kv_heads
    embed = _as_int(meta.get("embedding_length"))
    kv_bytes = _KV_BYTES.get((kv_type or "fp16").lower(), 2.0)

    kv_mb = 0.0
    have_kv = bool(n_layers and n_kv_heads and n_heads and embed and ctx)
    if have_kv:
        head_dim = embed / n_heads
        kv_mb = (2 * n_layers * ctx * n_kv_heads * head_dim * kv_bytes) / _MB

    overhead_mb = _OVERHEAD_MB_FLOOR
    total_mb = weights_mb + kv_mb + overhead_mb
    partial = not n_params or not have_kv

    return {
        "total_mb": round(total_mb, 1),
        "weights_mb": round(weights_mb, 1),
        "kv_mb": round(kv_mb, 1),
        "overhead_mb": round(overhead_mb, 1),
        "ctx": ctx,
        "kv_type": kv_type,
        "partial": partial,
        "assumptions": {
            "bytes_per_weight": round(bpw, 3),
            "quant": quant,
            "n_params": n_params,
            "kv_bytes": kv_bytes,
            "have_kv_dims": have_kv,
        },
    }


def _as_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None
