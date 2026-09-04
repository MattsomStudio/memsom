"""memsom.tuning -- the one registry for every tunable knob (PLAN.md Sec2.2).

Sits directly above memsom.kernel in the layer order (.importlinter): every
other package, kernel excepted, may import it. Two bootstrap reads predate
any registry and stay outside it on purpose, not by oversight:

* storage/db.py (MEMDAG_HOME, MEMDAG_DB) -- the store location has to resolve
  before there is a data dir to look for a canonical.json in, so tuning
  cannot be the thing that resolves it without a circular import.
* kernel/paths.py (MEMDAG_BRIDGE_MEMORY_DIR) -- kernel is rank 0 and stdlib
  only; it cannot import upward into this module. default_memory_dir() keeps
  its own read. The knob is still registered here (as "bridge.memory_dir")
  so it is visible to `tuning list` / the panel even though kernel resolves
  it independently.

Both exceptions are named in scripts/env_ratchet.py, which enforces that no
other os.environ/os.getenv site exists anywhere else in the package.

The 13 forget params + memory_budget are NOT here -- they stay owned by
lifecycle/forget.py:load_params (PLAN.md Sec1.7: protected, byte-identical
DEFAULTS, golden parity test). This registry is for the knobs that were
scattered bare os.environ.get() calls before Phase 8; it centralizes the
*read*, not the parsing each call site already did on the raw value.

PERSISTED OVERRIDES (2026-09-04). Resolution is in-process override >
`<store dir>/tuning.json` > env > default. The file is the one place a knob can
be set for EVERY process that opens the store -- the Stop-hook importer, the
prompt hook, and the MCP server (a Claude Code child launched with no
MEMDAG_* shell env) -- and it is what `memsom tuning set` writes and what a
panel json-file provider targets. It sits beside the DB (MEMDAG_DB's parent >
MEMDAG_HOME > ~/.memdag, the same bootstrap read storage/db.py does; tuning
sits BELOW storage in the layer order so it cannot import db_path()) so a test
pointed at a temp DB never reads the real one. Values are validated exactly
like env values (type, bounds, choices); a bad value warns once and falls
through to env/default, never raises.

Every knob carries an explicit `type` -- one of the Python types `int`,
`float`, `bool`, `str`, or the string sentinels `"path"` (a str, semantically
a filesystem path) and `"enum"` (a str restricted to `choices`). Numeric
knobs (`int`/`float`) carry `bounds` -- a generous sanity fence, not a tuning
opinion; the bound only exists so a badly-set operator env var cannot corrupt
the process, not to enforce a "best" value. `resolve()` NEVER raises on a
malformed or out-of-bounds env value: it logs one warning per knob per
process and falls back to `default`. On a VALID env value it still returns
the raw string unmodified, exactly as before Phase 8's ARCH-09 pass -- every
call site keeps doing its own final coercion; this module only decides
whether the raw value is safe to hand to that call site at all.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_LOG = logging.getLogger("memsom.tuning")


@dataclass(frozen=True)
class Knob:
    key: str
    type: type | str
    default: object
    bounds: tuple | None
    source: str   # "env:MEMDAG_..." | "canonical" | "const" -- only "env:" resolves here
    doc: str
    feature: str | None = None
    choices: tuple | None = None   # only meaningful when type == "enum"


REGISTRY: dict[str, Knob] = {}

_lock = threading.Lock()
_overrides: dict[str, object] = {}

_warn_lock = threading.Lock()
_warned_keys: set[str] = set()

_BOOL_TRUE = ("1", "true", "yes")
_BOOL_FALSE = ("0", "false", "no")


def _register(key: str, *, type: type | str = str, default=None, bounds=None,
              source: str, doc: str, feature: str | None = None,
              choices: tuple | None = None) -> None:
    REGISTRY[key] = Knob(key=key, type=type, default=default, bounds=bounds,
                          source=source, doc=doc, feature=feature, choices=choices)


def _coerce(knob: Knob, raw: str):
    """Interpret *raw* as knob.type. Returns the coerced value on success;
    raises ValueError/TypeError on failure -- resolve() catches this, never
    lets it propagate."""
    t = knob.type
    if t is bool:
        s = raw.strip().lower()
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
        raise ValueError(f"not a boolean (expected one of {_BOOL_TRUE + _BOOL_FALSE}): {raw!r}")
    if t is int:
        return int(raw.strip())
    if t is float:
        return float(raw.strip())
    if t == "enum":
        s = raw.strip()
        choices = knob.choices or ()
        for c in choices:
            if s.lower() == str(c).lower():
                return s
        raise ValueError(f"{raw!r} not one of {choices}")
    # str / path: any string is valid, no coercion needed.
    return raw


def _warn_once(key: str, message: str) -> None:
    """Log at most one fail-open warning per knob per process (stdlib
    logging, never print/raise)."""
    with _warn_lock:
        if key in _warned_keys:
            return
        _warned_keys.add(key)
    _LOG.warning(message)


def _clear_warned(key: str | None = None) -> None:
    """Test-isolation escape hatch (mirrors clear_override below): forget
    which knobs have already logged a fail-open warning this process, so a
    test can assert 'exactly one warning' without an earlier test's resolve()
    call having already consumed it. Production call sites never need this."""
    with _warn_lock:
        if key is None:
            _warned_keys.clear()
        else:
            _warned_keys.discard(key)


def _in_bounds(knob: Knob, value) -> bool:
    if knob.bounds is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    lo, hi = knob.bounds
    return lo <= value <= hi


# ---------------------------------------------------------------------------
# Persisted overrides -- <store dir>/tuning.json (see the module docstring)
# ---------------------------------------------------------------------------

PERSISTED_NAME = "tuning.json"
_file_lock = threading.Lock()
_file_cache: dict = {"path": None, "mtime": None, "data": {}, "checked": 0.0}
_FILE_STAT_TTL_S = 1.0   # re-stat at most once a second; a hot loop never pays a stat per resolve
_ABSENT = object()


def persisted_path() -> Path:
    """`<store dir>/tuning.json`. The store dir is MEMDAG_DB's parent, else
    MEMDAG_HOME, else ~/.memdag -- storage/db.py's bootstrap rule, repeated
    here because tuning cannot import upward into storage."""
    db = os.environ.get("MEMDAG_DB")
    if db:
        return Path(db).parent / PERSISTED_NAME
    home = os.environ.get("MEMDAG_HOME")
    return Path(home or (Path.home() / ".memdag")) / PERSISTED_NAME


def _invalidate_persisted() -> None:
    with _file_lock:
        _file_cache.update(path=None, mtime=None, data={}, checked=0.0)


def persisted() -> dict:
    """The tuning.json contents ({} when absent/unreadable/not a dict), cached
    on (path, mtime) so a panel or `tuning set` write is seen on the next
    resolve without a restart. Never raises."""
    p = persisted_path()
    now = time.monotonic()
    with _file_lock:
        c = _file_cache
        if c["path"] == p and (now - c["checked"]) < _FILE_STAT_TTL_S:
            return c["data"]
        try:
            mtime = p.stat().st_mtime_ns
        except OSError:
            mtime = None
        if c["path"] == p and c["mtime"] == mtime:
            c["checked"] = now
            return c["data"]
        data: dict = {}
        if mtime is not None:
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                data = raw if isinstance(raw, dict) else {}
            except (OSError, ValueError):
                data = {}
        c.update(path=p, mtime=mtime, data=data, checked=now)
        return data


def _validate_persisted(knob: Knob, value):
    """A tuning.json value -> the value resolve() returns, or _ABSENT when it
    is malformed/out of bounds (warned once, like a bad env value). A string
    is validated like an env string and returned as-is; a native JSON
    number/bool is type-checked and returned natively (like a default)."""
    if value is None:
        return _ABSENT
    t = knob.type
    if isinstance(value, str):
        try:
            coerced = _coerce(knob, value)
        except (ValueError, TypeError) as exc:
            _warn_once(knob.key, f"{knob.key} (tuning.json={value!r}): {exc}; ignoring the file value")
            return _ABSENT
        if not _in_bounds(knob, coerced):
            _warn_once(knob.key, f"{knob.key} (tuning.json={value!r}) out of bounds {knob.bounds}; "
                                 "ignoring the file value")
            return _ABSENT
        return value
    ok = ((t is bool and isinstance(value, bool))
          or (t is int and isinstance(value, int) and not isinstance(value, bool))
          or (t is float and isinstance(value, (int, float)) and not isinstance(value, bool)))
    if not ok:
        _warn_once(knob.key, f"{knob.key} (tuning.json={value!r}): not a {_type_name(t)}; "
                             "ignoring the file value")
        return _ABSENT
    if not _in_bounds(knob, value):
        _warn_once(knob.key, f"{knob.key} (tuning.json={value!r}) out of bounds {knob.bounds}; "
                             "ignoring the file value")
        return _ABSENT
    return float(value) if t is float else value


def set_persisted(key: str, value) -> Path:
    """Validate *value* for *key* and merge it into tuning.json atomically.
    Raises KeyError for an unknown knob and ValueError for a value the knob
    rejects (type, choices, bounds) -- a persisted knob is never written
    invalid. Strings are stored coerced to the knob's native type."""
    knob = REGISTRY[key]
    if isinstance(value, str):
        value = _coerce(knob, value)
    else:
        if _validate_persisted(knob, value) is _ABSENT:
            raise ValueError(f"{key}: {value!r} is not a valid {_type_name(knob.type)}")
    if not _in_bounds(knob, value):
        raise ValueError(f"{key}: {value!r} out of bounds {knob.bounds}")
    data = dict(persisted())
    data[key] = value
    return _write_persisted(data)


def unset_persisted(key: str) -> Path:
    """Remove *key* from tuning.json (no-op when absent)."""
    data = dict(persisted())
    data.pop(key, None)
    return _write_persisted(data)


def _write_persisted(data: dict) -> Path:
    p = persisted_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    _invalidate_persisted()
    return p


def resolve(key: str, *, conn=None):
    """In-process override > tuning.json > env > default.

    Every real knob here is `source="env:..."`. When the env var is unset,
    `default` is returned as-is (its native type -- a bool/int/float for a
    typed knob, a str otherwise). When it IS set, the raw string is validated
    against the knob's type and bounds; a value that coerces cleanly and (for
    numeric knobs) sits inside `bounds` is returned UNMODIFIED as the raw
    string -- callers keep doing the same coercion they always did. A value
    that fails coercion or falls outside bounds never raises: it logs one
    warning for this key (this process) and falls back to `default`.

    A tuning.json value (see `persisted()`) sits between the in-process
    override and the env: validated the same way, and a bad one falls through
    to the env/default rather than shadowing it.
    """
    knob = REGISTRY[key]
    with _lock:
        if key in _overrides:
            return _overrides[key]
    if not knob.source.startswith("env:"):
        return knob.default
    file_vals = persisted()
    if key in file_vals:
        fv = _validate_persisted(knob, file_vals[key])
        if fv is not _ABSENT:
            return fv
    var = knob.source[len("env:"):]
    raw = os.environ.get(var)
    if raw is None:
        return knob.default

    try:
        value = _coerce(knob, raw)
    except (ValueError, TypeError) as exc:
        _warn_once(key, f"{key} ({var}={raw!r}): {exc}; falling back to default {knob.default!r}")
        return knob.default

    if not _in_bounds(knob, value):
        lo, hi = knob.bounds
        _warn_once(key, f"{key} ({var}={raw!r}) out of bounds [{lo}, {hi}]; "
                         f"falling back to default {knob.default!r}")
        return knob.default

    return raw


def override(key: str, value) -> None:
    """A CLI flag beating env/default for the rest of THIS process -- e.g.
    `--embed-backend` -- without memsom.tuning ever writing os.environ."""
    with _lock:
        _overrides[key] = value


def clear_override(key: str) -> None:
    """Undo override(key, ...): resolve() falls back to env/default again.
    A no-op if the key was never overridden. Test-isolation escape hatch --
    production call sites are one-shot CLI processes that never need it."""
    with _lock:
        _overrides.pop(key, None)


def snapshot(conn=None) -> dict[str, object]:
    return {key: resolve(key, conn=conn) for key in REGISTRY}


def _type_name(t) -> str:
    return t.__name__ if isinstance(t, type) else str(t)


def as_json(conn=None) -> dict[str, dict]:
    out = {}
    file_vals = persisted()
    for key, knob in REGISTRY.items():
        out[key] = {
            "key": key,
            "name": key,
            "type": _type_name(knob.type),
            "default": knob.default,
            "bounds": list(knob.bounds) if knob.bounds is not None else None,
            "choices": list(knob.choices) if knob.choices is not None else None,
            "source": knob.source,
            "doc": knob.doc,
            "feature": knob.feature,
            "value": resolve(key, conn=conn),
            "persisted": file_vals.get(key),
        }
    return out


# ---------------------------------------------------------------------------
# Registrations. Grouped by the module that reads them.
# ---------------------------------------------------------------------------

_register("bridge.author", type=bool, default=True, source="env:MEMDAG_BRIDGE_AUTHOR",
           doc="Stamp bridge-authored nodes as author-channel unless '0'.")
_register("bridge.claude_md_path", type="path", default="", source="env:CLAUDE_MD_PATH",
           doc="Override path to the bundled CLAUDE.md template.")
_register("bridge.hook_policy_path", type="path", default="", source="env:MEMDAG_HOOK_POLICY",
           doc="Override path to the Stop-hook policy file.")
_register("bridge.hook_mode", type="enum", choices=("shadow", "enforcing"),
           default="shadow", source="env:MEMDAG_HOOK_MODE",
           doc="Gate #3 hook arm: 'shadow' (log would-be decisions, deny nothing) "
               "or 'enforcing' (emit real PreToolUse denials). PLAN.md Phase 9 "
               "mandates shadow first.", feature="gate3.hook")
_register("bridge.hook_shadow_log", type="path", default="", source="env:MEMDAG_HOOK_SHADOW_LOG",
           doc="Override path to Gate #3's shadow decision log "
               "(default ~/.claude/gate3_shadow.jsonl).", feature="gate3.hook")
_register("bridge.memory_dir", type="path", default="", source="env:MEMDAG_BRIDGE_MEMORY_DIR",
           doc="Override the discovered Claude memory dir (also read directly "
               "by kernel.paths.default_memory_dir -- see module docstring).")
_register("bridge.index_enabled", type=bool, default=True, source="env:MEMDAG_BRIDGE_INDEX",
           doc="Keep the retrieval index current on bridge import unless '0' "
               "(the kill-switch restores the old write-only behaviour).")
_register("obsidian.vault", type="path", default="", source="env:MEMDAG_OBSIDIAN_VAULT",
           doc="The operator-configured Obsidian vault root.", feature="obsidian")

_register("distill.digest_shrink_floor", type=float, default=0.5, bounds=(0.0, 1.0),
           source="env:MEMDAG_DIGEST_SHRINK_FLOOR",
           doc="Digest shrink floor (0..1); out-of-range or unparsable falls back "
               "to this default (matches digest.py's own SHRINK_FLOOR).", feature="distill")
_register("distill.digest_sections", type=str, default="", source="env:MEMDAG_DIGEST_SECTIONS",
           doc="Comma-separated digest section display order.", feature="distill")
_register("distill.digest_title", type=str, default="# Memory", source="env:MEMDAG_DIGEST_TITLE",
           doc="Digest document title.", feature="distill")
_register("distill.projects_title", type=str, default="# Projects", source="env:MEMDAG_PROJECTS_TITLE",
           doc="projects/INDEX.md document title.", feature="distill")

_register("llm.model", type=str, default="", source="env:MEMDAG_LLM_MODEL",
           doc="Ollama model name for distill/retrieval LLM calls.", feature="llm")
_register("llm.url", type=str, default="", source="env:MEMDAG_LLM_URL",
           doc="Ollama base URL for LLM calls.", feature="llm")
_register("llm.ollama_keep_alive", type=str, default="", source="env:MEMDAG_OLLAMA_KEEP_ALIVE",
           doc="Ollama keep_alive duration string, passed through verbatim "
               "(a numeric string OR a duration like '10m' -- caller decides).",
           feature="llm")
_register("llm.cite_overlap", type=float, default=0.2, bounds=(-1_000_000.0, 1_000_000.0),
           source="env:MEMDAG_LLM_CITE_OVERLAP",
           doc="Minimum citation overlap floor (float); llm.py clamps the final "
               "value to [0.0, 1.0] itself, so this bound is a wide sanity fence "
               "only, not the real domain (test_cite_overlap_floor_env_clamped "
               "relies on out-of-[0,1] values like 2.5/-1 reaching that clamp).",
           feature="llm")

_register("federation.broker_config_path", type="path", default="", source="env:MEMDAG_BROKER_CONFIG",
           doc="Override path to the broker's config file.", feature="gate3.broker")
_register("federation.origin", type=str, default="", source="env:MEMDAG_ORIGIN",
           doc="This machine's federation origin name.", feature="federation.sync")

_register("integrity.channel_ceiling", type=str, default="", source="env:MEMDAG_CHANNEL_CEILING",
           doc="Max channel rank ingest_text will stamp (0-3 or a channel name), "
               "or unset for none. ingest.py validates and raises its own "
               "ValueError on a bad value -- kept as str here, no tuning-level "
               "bounds, since the valid domain is a rank name OR a digit.")
_register("integrity.clearance_ceiling", type=str, default="", source="env:MEMDAG_CLEARANCE_CEILING",
           doc="Max confidentiality rank retrieval may see (0-3 or a rank name). "
               "Same shape as integrity.channel_ceiling -- validated locally.")

_register("mcp.channel_ceiling", type=str, default="", source="env:MEMSOM_MCP_CHANNEL_CEILING",
           doc="MCP server's channel ceiling override (rank name or 0-3).")
_register("mcp.export_dir", type="path", default="", source="env:MEMSOM_MCP_EXPORT_DIR",
           doc="Directory a model-driven export may write into.")

_register("retrieval.warm_disabled", type=str, default="", source="env:MEMDAG_WARM_ENDPOINT",
           doc="'0'/'off'/'false'/'no' disables the loopback warm retrieval "
               "endpoint the prompt hook queries. Kept as str (not bool) -- "
               "'off' isn't in the standard bool vocabulary and warm.py's own "
               "disabled_by_env() already does this exact comparison.")

_register("saveall.userprofile_fallback", type="path", default="", source="env:USERPROFILE",
           doc="Windows subprocess cwd fallback for the detached /saveall child.")

_register("contradict.nli_model", type=str, default="", source="env:MEMDAG_CONTRADICT_NLI_MODEL",
           doc="NLI model name for contradiction detection.", feature="contradict.nli")
_register("contradict.nli_threshold", type=float, default=0.85, bounds=(0.0, 1.0),
           source="env:MEMDAG_CONTRADICT_NLI_THRESHOLD",
           doc="Contradiction-probability cutoff (float, 0..1).", feature="contradict.nli")
_register("contradict.nli_enabled", type=bool, default=False, source="env:MEMDAG_CONTRADICT_NLI",
           doc="Opt in to the NLI contradiction tier.", feature="contradict.nli")
_register("contradict.anchor", type=float, default=0.80, bounds=(0.0, 1.0),
           source="env:MEMDAG_CONTRADICT_ANCHOR",
           doc="Anchor-similarity threshold for contradiction candidates (float, 0..1).",
           feature="contradict.nli")
_register("contradict.enabled", type=bool, default=False, source="env:MEMDAG_CONTRADICT",
           doc="Opt in to the structured contradiction detector.")
_register("contradict.enforce", type=bool, default=False, source="env:MEMDAG_CONTRADICT_ENFORCE",
           doc="Enforce (not just record) contradiction sweeps.")

_register("retrieval.embed_url", type=str, default="", source="env:MEMDAG_EMBED_URL",
           doc="Ollama embeddings endpoint, used by doctor and retrieve.")
_register("retrieval.embed_model", type=str, default="", source="env:MEMDAG_EMBED_MODEL",
           doc="Ollama embeddings model, used by doctor and retrieve.")
_register("retrieval.embed_timeout", type=int, default=60, bounds=(1, 3600),
           source="env:MEMDAG_EMBED_TIMEOUT",
           doc="Timeout (seconds) for one Ollama embedding HTTP call. The old "
               "hard-coded 10s was too short for a cold model load under "
               "KEEP_ALIVE=0 (every call reloads the model).",
           feature="retrieval.dense")
_register("lifecycle.verify_stale_days", type=int, default=21, bounds=(0, 36500),
           source="env:MEMDAG_VERIFY_STALE_DAYS",
           doc="Staleness threshold in days (int); 0 turns the pass off "
               "(bridge_render's test_... relies on 0 being in-bounds).")

_register("code_rag.enabled", type=bool, default=False, source="env:MEMSOM_CODE_RAG",
           doc="Opt in to the code-RAG index.", feature="code_rag")
_register("code_rag.qwen_url", type=str, default="", source="env:MEMSOM_QWEN_URL",
           doc="Qwen embedding supervisor URL.", feature="code_rag")

_register("embed.backend", type="enum", choices=("ollama", "bge-m3", "bm25"),
           default="", source="env:MEMDAG_EMBED_BACKEND",
           doc="Embedding backend: ollama | bge-m3 | bm25. Unknown/unset falls "
               "back to embed.py's own DEFAULT_BACKEND ('ollama').",
           feature="retrieval.bge")
_register("retrieval.colbert_candidates", type=int, default=100, bounds=(1, 1_000_000),
           source="env:MEMDAG_COLBERT_CANDIDATES",
           doc="ColBERT re-rank window size (int).", feature="retrieval.colbert")
_register("retrieval.colbert_maxlen", type=int, default=512, bounds=(1, 1_000_000),
           source="env:MEMDAG_COLBERT_MAXLEN",
           doc="ColBERT passage/query token cap (int).", feature="retrieval.colbert")
_register("retrieval.bge_device", type=str, default="", source="env:MEMDAG_BGE_DEVICE",
           doc="Force the BGE-M3 device (cuda/cpu); unset auto-selects.", feature="retrieval.bge")
_register("retrieval.bge_unload", type=bool, default=False, source="env:MEMDAG_BGE_UNLOAD",
           doc="Unload the BGE-M3 model after a batch reindex.", feature="retrieval.bge")
# --- BGE-M3 encode path + signal toggles (portable / opt-in) --------------
# bge_url ships as LOCALHOST, never a mesh/host IP: memsom only ever talks to a
# LOCAL embedding supervisor. A fresh clone with no supervisor running just
# fails the fast /health probe and falls back to in-process torch (the `bge`
# pip extra), and if that is not installed, to BM25 — see embed._dispatch_encode.
_register("retrieval.bge_url", type=str, default="http://127.0.0.1:11435/embed",
           source="env:MEMDAG_BGE_URL",
           doc="LOCAL BGE-M3 embedding supervisor endpoint (POST /embed). Used only "
               "when bge_encode_via allows the supervisor AND its /health answers; "
               "otherwise memsom embeds in-process (torch) or degrades to BM25. "
               "Localhost by default — memsom contains no mesh/cross-host logic.",
           feature="retrieval.bge")
_register("retrieval.bge_encode_via", type="enum",
           choices=("auto", "supervisor", "inprocess"),
           default="auto", source="env:MEMDAG_BGE_ENCODE_VIA",
           doc="How bge-m3 embeds: 'auto' (prefer the local supervisor if bge_url's "
               "/health answers, else in-process torch, else BM25), 'supervisor' "
               "(prefer the HTTP supervisor), 'inprocess' (FlagEmbedding only, never "
               "probe the supervisor).", feature="retrieval.bge")
# --- supervisor cold-start-on-demand + idle keep-alive (2026-09-04) ----------
# The supervisor is a DETACHED process (never a child of the caller): memsom
# launches bge_spawn_cmd only when bge_url's /health is down, waits up to
# bge_spawn_timeout for it, and every /embed carries idle_ttl so the supervisor
# knows how long to hold the torch backend after memsom's last encode. A fresh
# clone leaves bge_spawn_cmd empty and behaves exactly as before (no spawn).
_register("retrieval.bge_idle_ttl", type=int, default=60, bounds=(0, 86400),
           source="env:MEMDAG_BGE_IDLE_TTL",
           doc="Idle keep-alive (seconds) the local BGE-M3 supervisor holds its torch "
               "backend after memsom's last encode; sent as `idle_ttl` on every "
               "/embed (last request wins) and as BGE_PROC_IDLE_SEC to a supervisor "
               "memsom spawns. 0 = never idle-kill.", feature="retrieval.bge")
_register("retrieval.bge_spawn_cmd", type=str, default="", source="env:MEMDAG_BGE_SPAWN_CMD",
           doc="Command line memsom launches DETACHED (own process group, no console, "
               "no pipes) when bge_url's /health is down, e.g. the supervisor's "
               "launcher script. Empty = never spawn (a fresh clone falls back to "
               "in-process torch / BM25 as before). One launch per outage; a broken "
               "command is retried only after a cooldown.", feature="retrieval.bge")
_register("retrieval.bge_spawn_timeout", type=int, default=30, bounds=(1, 300),
           source="env:MEMDAG_BGE_SPAWN_TIMEOUT",
           doc="Seconds to wait for a spawned supervisor's /health before giving the "
               "query up as degraded (the spawn keeps running; the next query is "
               "dense).", feature="retrieval.bge")
_register("retrieval.bge_dense", type=bool, default=True, source="env:MEMDAG_BGE_DENSE",
           doc="Store + score the BGE-M3 dense vector. Off -> no dense signal on the "
               "bge path (reindex to apply).", feature="retrieval.bge")
_register("retrieval.bge_sparse", type=bool, default=True, source="env:MEMDAG_BGE_SPARSE",
           doc="Store + score the BGE-M3 sparse (learned-lexical) weights. Off -> no "
               "sparse signal on the bge path (reindex to apply).", feature="retrieval.bge")
_register("retrieval.bge_colbert", type=bool, default=True, source="env:MEMDAG_BGE_COLBERT",
           doc="Store + score the BGE-M3 ColBERT late-interaction vectors. Off -> no "
               "colbert re-rank on the bge path (reindex to apply).", feature="retrieval.bge")

_register("storage.sync_extra_markers", type=str, default="", source="env:MEMSOM_EXTRA_SYNC_MARKERS",
           doc="Comma-separated extra file-sync marker names, visibility only -- "
               "kernel.syncguard (rank 0) reads MEMSOM_EXTRA_SYNC_MARKERS itself "
               "(cannot import tuning upward). Same pattern as bridge.memory_dir.")
_register("remote.action_gate_mode", type="enum", choices=("shadow", "enforcing"),
           default="shadow", source="env:MEMDAG_REMOTE_ACTION_GATE_MODE",
           doc="Remote mutate calls' action-gate (capgate.check_capability) mode: "
               "'shadow' (log the verdict, deny nothing beyond the capability "
               "table) or 'enforcing'. PLAN.md Phase 10 mandates shadow first, "
               "same schedule as bridge.hook_mode.", feature="remote.server")
_register("remote.export_dir", type="path", default="", source="env:MEMSOM_REMOTE_EXPORT_DIR",
           doc="Directory a remote export tool call may write into.")
_register("remote.tls_cert", type="path", default="", source="env:MEMSOM_REMOTE_TLS_CERT",
           doc="Optional self-signed cert for remote serve (mesh already encrypts).",
           feature="remote.server")
_register("remote.tls_key", type="path", default="", source="env:MEMSOM_REMOTE_TLS_KEY",
           doc="Private key matching remote.tls_cert.", feature="remote.server")

_register("telemetry.consolidation_dir", type="path", default="", source="env:MEMSOM_CONSOLIDATION_DIR",
           doc="Override where the weekly consolidation sweep's dated reports "
               "live (default ~/.claude/consolidation); telemetry.last_consolidation "
               "reads the newest report's mtime from here.", feature="telemetry")
_register("telemetry.episodic_db", type="path", default="", source="env:MEMSOM_EPISODIC_DB",
           doc="Override the episodic sessions archive path (default "
               "~/.claude/episodic/sessions.db) telemetry.sessions counts from.",
           feature="telemetry")


# ---------------------------------------------------------------------------
# CLI -- `memsom tuning list|get|set|unset`
#
# The 13 forget params + memory_budget are NOT registered here (module
# docstring). `tuning set` writes the persisted override file
# (`<store dir>/tuning.json`, see the module docstring), which every process
# that opens this store -- hook, Stop-hook importer, the env-less MCP child --
# reads on its next resolve. `tuning unset` removes the file value so the
# env/default applies again.
# ---------------------------------------------------------------------------

def _cmd_tuning_list(args) -> None:
    if getattr(args, "json", False):
        print(json.dumps(as_json()))
        return
    file_vals = persisted()
    for key in sorted(REGISTRY):
        knob = REGISTRY[key]
        tag = " [tuning.json]" if key in file_vals else ""
        print(f"{key:<32} {resolve(key)!r:<20} type={_type_name(knob.type):<6} "
              f"default={knob.default!r} source={knob.source}{tag}")


def _cmd_tuning_get(args) -> None:
    if args.key not in REGISTRY:
        raise SystemExit(f"unknown knob: {args.key!r}")
    print(resolve(args.key))


def _cmd_tuning_set(args) -> None:
    if args.key not in REGISTRY:
        raise SystemExit(f"unknown knob: {args.key!r}")
    try:
        path = set_persisted(args.key, args.value)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"refused: {exc}")
    print(f"{args.key} = {resolve(args.key)!r}  (persisted in {path})")


def _cmd_tuning_unset(args) -> None:
    if args.key not in REGISTRY:
        raise SystemExit(f"unknown knob: {args.key!r}")
    path = unset_persisted(args.key)
    print(f"{args.key} = {resolve(args.key)!r}  (removed from {path})")


def register(sub) -> None:
    p = sub.add_parser("tuning", help="report/inspect/persist every tunable knob")
    tsub = p.add_subparsers(dest="tuning_command", required=True)
    p_list = tsub.add_parser("list", help="list every registered knob")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_tuning_list)
    p_get = tsub.add_parser("get", help="resolve one knob")
    p_get.add_argument("key")
    p_get.set_defaults(func=_cmd_tuning_get)
    p_set = tsub.add_parser("set", help="persist a knob in <store dir>/tuning.json "
                                        "(beats env; seen by every process on the store)")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_set.set_defaults(func=_cmd_tuning_set)
    p_unset = tsub.add_parser("unset", help="remove a knob from tuning.json (env/default applies again)")
    p_unset.add_argument("key")
    p_unset.set_defaults(func=_cmd_tuning_unset)
