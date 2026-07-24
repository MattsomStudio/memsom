"""Minimal GGUF header reader — just enough architecture to size a launch.

The panel needs three numbers before it can offer honest launch controls for a
local GGUF:

* ``block_count``   — how many layers the model has, which is the real ceiling
  for ``--n-cpu-moe N``. Without it that field is a blind number box and the
  only way to find the limit is to launch and watch it fail.
* ``expert_count``  — whether the model is Mixture-of-Experts at all. Zero or
  absent means every MoE control should be greyed out, not offered.
* ``context_length`` — the model's trained context, the ceiling for ``--ctx-size``.

Reading them means parsing the GGUF metadata block, and the trap there is the
tokenizer: ``tokenizer.ggml.tokens`` is a string array with 30k-260k entries and
can run to megabytes. A naive "parse every key/value" reader materializes all of
it on a 4-second poll. So this reader:

* walks the KV block over an **incrementally grown byte prefix** (1 MB, doubling
  to a cap) rather than reading the whole file;
* **skips** values by computing their length instead of decoding them;
* **stops the moment** the wanted keys are all found — and in practice
  llama.cpp writes ``general.*`` and ``<arch>.*`` before the tokenizer arrays, so
  the answer is almost always inside the first megabyte;
* caches per ``(path, size, mtime)``, so a given file is read at most once.

Format reference: GGUF v2/v3 — magic ``GGUF``, uint32 version, uint64
tensor_count, uint64 kv_count, then kv_count × (string key, uint32 type, value).
v1 used 32-bit lengths and is not produced by anything current; it is refused
rather than mis-parsed.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Optional

# GGUF value type enum -> fixed byte width. STRING (8) and ARRAY (9) are
# variable and handled separately.
_FIXED = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_STRING = 8
_ARRAY = 9

# unpackers for the scalar types we actually read a value out of
_UNPACK = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}

#: the metadata keys worth stopping for, as (suffix -> our field name). Matching
#: on suffix avoids having to read ``general.architecture`` first just to build
#: the real key (`llama.block_count`, `qwen3moe.block_count`, ...).
_WANTED = {
    ".block_count": "n_layers",
    ".expert_count": "n_experts",
    ".expert_used_count": "n_experts_used",
    ".context_length": "ctx_max",
}

_PREFIX_START = 1 << 20        # 1 MB — enough for the arch keys on every real model
_PREFIX_CAP = 64 << 20         # never read more than this looking for them

#: quant tag out of a filename, e.g. "Qwen3-30B-A3B-Q4_K_M.gguf" -> "Q4_K_M".
#: Deliberately filename-derived: these are exactly the keys
#: :mod:`memsom.providers.vram` already indexes, whereas ``general.file_type``
#: is an enum whose numbering has drifted between llama.cpp releases.
_QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:IQ|Q)\d+(?:_[A-Z0-9]+)*|F16|BF16|F32)(?:[-_.]|$)",
    re.IGNORECASE)

_cache: dict = {}


class _Eof(Exception):
    """Ran off the end of the prefix we read — grow it and try again."""


class _Cursor:
    """Read head over a bytes prefix that raises :class:`_Eof` past its end."""

    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if n < 0 or end > len(self.buf):
            raise _Eof
        out = self.buf[self.pos:end]
        self.pos = end
        return out

    def skip(self, n: int) -> None:
        end = self.pos + n
        if n < 0 or end > len(self.buf):
            raise _Eof
        self.pos = end

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")


def quant_from_name(name: str) -> Optional[str]:
    """Quantization tag parsed out of a GGUF filename, uppercased; None if the
    name carries none."""
    m = _QUANT_RE.search(str(name))
    return m.group(1).upper() if m else None


def read_meta(path) -> dict:
    """Architecture facts for one GGUF: ``n_layers``, ``n_experts``,
    ``n_experts_used``, ``ctx_max``, ``quant``.

    Best effort by design — a missing, truncated, or non-GGUF file yields a dict
    with whatever was learned (possibly only ``quant``) rather than raising. The
    panel must never lose a whole model list to one bad file, the same rule
    :func:`memsom.providers.registry.build_registry` follows for a bad spec.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (str(p), st.st_size, int(st.st_mtime))
    hit = _cache.get(key)
    if hit is not None:
        return hit

    meta = {}
    quant = quant_from_name(p.name)
    if quant:
        meta["quant"] = quant

    want = min(_PREFIX_START, st.st_size) or 0
    while True:
        try:
            with open(p, "rb") as fh:
                buf = fh.read(want)
        except OSError:
            break
        found: dict = {}
        try:
            _parse(buf, found)
        except _Eof:
            if want < st.st_size and want < _PREFIX_CAP:
                want = min(want * 2, st.st_size, _PREFIX_CAP)
                continue           # grow the prefix and re-walk
            # whole file (or our cap) walked — keep whatever we did learn
        except (struct.error, ValueError):
            pass                   # not a GGUF, or a header we don't understand
        meta.update(found)
        break

    _cache[key] = meta
    return meta


def _parse(buf: bytes, found: dict) -> None:
    """Walk the KV block over *buf*, filling *found*.

    Fills in place (rather than returning) so a partial answer survives an
    :class:`_Eof` — a model whose tokenizer runs past our cap still yields its
    layer count if that key came first, which it does.
    """
    cur = _Cursor(buf)
    if cur.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = cur.u32()
    if version < 2:
        raise ValueError(f"unsupported GGUF version {version}")
    cur.u64()                      # tensor_count — not needed
    kv_count = cur.u64()

    for _ in range(kv_count):
        key = cur.string()
        vtype = cur.u32()
        # llama.cpp emits general.* then <arch>.* then tokenizer.*, so reaching
        # a tokenizer key means every architecture key that exists has been
        # seen. Stopping here is what keeps a 260k-token array unread. If some
        # writer ever orders it differently we just walk on and skip it.
        if key.startswith("tokenizer.") and "n_layers" in found:
            return
        field = next((f for suffix, f in _WANTED.items() if key.endswith(suffix)),
                     None)
        if field is not None and vtype in _UNPACK:
            found[field] = struct.unpack(_UNPACK[vtype],
                                         cur.take(_FIXED[vtype]))[0]
            continue
        _skip_value(cur, vtype)


def _skip_value(cur: _Cursor, vtype: int) -> None:
    """Advance past one value without decoding it."""
    if vtype in _FIXED:
        cur.skip(_FIXED[vtype])
        return
    if vtype == _STRING:
        cur.skip(cur.u64())
        return
    if vtype == _ARRAY:
        elem = cur.u32()
        count = cur.u64()
        if elem in _FIXED:
            cur.skip(_FIXED[elem] * count)   # one jump, no per-element work
            return
        if elem == _STRING:
            # the tokenizer case: lengths are only knowable one at a time, but
            # we still never materialize a token.
            for _ in range(count):
                cur.skip(cur.u64())
            return
        raise ValueError(f"nested GGUF array type {elem}")
    raise ValueError(f"unknown GGUF value type {vtype}")
