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
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Knob:
    key: str
    type: type
    default: object
    bounds: tuple | None
    source: str   # "env:MEMDAG_..." | "canonical" | "const" -- only "env:" resolves here
    doc: str
    feature: str | None = None


REGISTRY: dict[str, Knob] = {}

_lock = threading.Lock()
_overrides: dict[str, object] = {}


def _register(key: str, *, type: type = str, default=None, bounds=None,
              source: str, doc: str, feature: str | None = None) -> None:
    REGISTRY[key] = Knob(key=key, type=type, default=default, bounds=bounds,
                          source=source, doc=doc, feature=feature)


def resolve(key: str, *, conn=None):
    """In-process override > env > default. Every knob here is `source="env:..."`;
    the raw string is returned unmodified (or `default`) -- callers keep doing
    the same coercion they always did, on a value read from one place now."""
    knob = REGISTRY[key]
    with _lock:
        if key in _overrides:
            return _overrides[key]
    if knob.source.startswith("env:"):
        var = knob.source[len("env:"):]
        raw = os.environ.get(var)
        if raw is not None:
            if knob.bounds is not None:
                try:
                    numeric = float(raw)
                except ValueError:
                    return knob.default
                lo, hi = knob.bounds
                if not (lo <= numeric <= hi):
                    raise ValueError(f"{key} ({var}={raw!r}) out of bounds [{lo}, {hi}]")
            return raw
    return knob.default


def override(key: str, value) -> None:
    """A CLI flag beating env/default for the rest of THIS process -- e.g.
    `--embed-backend` -- without memsom.tuning ever writing os.environ."""
    with _lock:
        _overrides[key] = value


def snapshot(conn=None) -> dict[str, object]:
    return {key: resolve(key, conn=conn) for key in REGISTRY}


def as_json(conn=None) -> dict[str, dict]:
    out = {}
    for key, knob in REGISTRY.items():
        out[key] = {
            "key": key,
            "name": key,
            "type": knob.type.__name__ if isinstance(knob.type, type) else str(knob.type),
            "default": knob.default,
            "bounds": list(knob.bounds) if knob.bounds is not None else None,
            "source": knob.source,
            "doc": knob.doc,
            "feature": knob.feature,
            "value": resolve(key, conn=conn),
        }
    return out


# ---------------------------------------------------------------------------
# Registrations. Grouped by the module that reads them.
# ---------------------------------------------------------------------------

_register("bridge.author", default="1", source="env:MEMDAG_BRIDGE_AUTHOR",
           doc="Stamp bridge-authored nodes as author-channel unless '0'.")
_register("bridge.claude_md_path", default="", source="env:CLAUDE_MD_PATH",
           doc="Override path to the bundled CLAUDE.md template.")
_register("bridge.hook_policy_path", default="", source="env:MEMDAG_HOOK_POLICY",
           doc="Override path to the Stop-hook policy file.")
_register("bridge.memory_dir", default="", source="env:MEMDAG_BRIDGE_MEMORY_DIR",
           doc="Override the discovered Claude memory dir (also read directly "
               "by kernel.paths.default_memory_dir -- see module docstring).")
_register("obsidian.vault", default="", source="env:MEMDAG_OBSIDIAN_VAULT",
           doc="The operator-configured Obsidian vault root.", feature="obsidian")

_register("distill.digest_shrink_floor", default="", source="env:MEMDAG_DIGEST_SHRINK_FLOOR",
           doc="Digest shrink floor override (float).", feature="distill")
_register("distill.digest_sections", default="", source="env:MEMDAG_DIGEST_SECTIONS",
           doc="Comma-separated digest section display order.", feature="distill")
_register("distill.digest_title", default="# Memory", source="env:MEMDAG_DIGEST_TITLE",
           doc="Digest document title.", feature="distill")

_register("llm.model", default="", source="env:MEMDAG_LLM_MODEL",
           doc="Ollama model name for distill/retrieval LLM calls.", feature="llm")
_register("llm.url", default="", source="env:MEMDAG_LLM_URL",
           doc="Ollama base URL for LLM calls.", feature="llm")
_register("llm.ollama_keep_alive", default="", source="env:MEMDAG_OLLAMA_KEEP_ALIVE",
           doc="Ollama keep_alive duration string, passed through verbatim.", feature="llm")
_register("llm.cite_overlap", default="", source="env:MEMDAG_LLM_CITE_OVERLAP",
           doc="Minimum citation overlap floor (float).", feature="llm")

_register("federation.broker_config_path", default="", source="env:MEMDAG_BROKER_CONFIG",
           doc="Override path to the broker's config file.", feature="gate3.broker")
_register("federation.origin", default="", source="env:MEMDAG_ORIGIN",
           doc="This machine's federation origin name.", feature="federation.sync")

_register("integrity.channel_ceiling", default="", source="env:MEMDAG_CHANNEL_CEILING",
           doc="Max channel rank ingest_text will stamp (0-3), or unset for none.")
_register("integrity.clearance_ceiling", default="", source="env:MEMDAG_CLEARANCE_CEILING",
           doc="Max confidentiality rank retrieval may see (0-3 or a rank name).")

_register("mcp.channel_ceiling", default="", source="env:MEMSOM_MCP_CHANNEL_CEILING",
           doc="MCP server's channel ceiling override.")
_register("mcp.export_dir", default="", source="env:MEMSOM_MCP_EXPORT_DIR",
           doc="Directory a model-driven export may write into.")

_register("saveall.userprofile_fallback", default="", source="env:USERPROFILE",
           doc="Windows subprocess cwd fallback for the detached /saveall child.")

_register("contradict.nli_model", default="", source="env:MEMDAG_CONTRADICT_NLI_MODEL",
           doc="NLI model name for contradiction detection.", feature="contradict.nli")
_register("contradict.nli_threshold", default="0.85", source="env:MEMDAG_CONTRADICT_NLI_THRESHOLD",
           doc="Contradiction-probability cutoff (float).", feature="contradict.nli")
_register("contradict.nli_enabled", default="", source="env:MEMDAG_CONTRADICT_NLI",
           doc="Opt in to the NLI contradiction tier.", feature="contradict.nli")
_register("contradict.anchor", default="0.80", source="env:MEMDAG_CONTRADICT_ANCHOR",
           doc="Anchor-similarity threshold for contradiction candidates (float).",
           feature="contradict.nli")
_register("contradict.enabled", default="", source="env:MEMDAG_CONTRADICT",
           doc="Opt in to the structured contradiction detector.")
_register("contradict.enforce", default="", source="env:MEMDAG_CONTRADICT_ENFORCE",
           doc="Enforce (not just record) contradiction sweeps.")

_register("retrieval.embed_url", default="", source="env:MEMDAG_EMBED_URL",
           doc="Ollama embeddings endpoint, used by doctor and retrieve.")
_register("retrieval.embed_model", default="", source="env:MEMDAG_EMBED_MODEL",
           doc="Ollama embeddings model, used by doctor and retrieve.")
_register("lifecycle.verify_stale_days", default="", source="env:MEMDAG_VERIFY_STALE_DAYS",
           doc="Staleness threshold in days (int).")

_register("code_rag.enabled", default="", source="env:MEMSOM_CODE_RAG",
           doc="Opt in to the code-RAG index.", feature="code_rag")
_register("code_rag.qwen_url", default="", source="env:MEMSOM_QWEN_URL",
           doc="Qwen embedding supervisor URL.", feature="code_rag")

_register("embed.backend", default="", source="env:MEMDAG_EMBED_BACKEND",
           doc="Embedding backend: ollama | bge-m3 | bm25.", feature="retrieval.bge")
_register("retrieval.colbert_candidates", default="", source="env:MEMDAG_COLBERT_CANDIDATES",
           doc="ColBERT re-rank window size (int).", feature="retrieval.colbert")
_register("retrieval.colbert_maxlen", default="", source="env:MEMDAG_COLBERT_MAXLEN",
           doc="ColBERT passage/query token cap (int).", feature="retrieval.colbert")
_register("retrieval.bge_device", default="", source="env:MEMDAG_BGE_DEVICE",
           doc="Force the BGE-M3 device (cuda/cpu); unset auto-selects.", feature="retrieval.bge")
_register("retrieval.bge_unload", default="", source="env:MEMDAG_BGE_UNLOAD",
           doc="Unload the BGE-M3 model after a batch reindex.", feature="retrieval.bge")


# ---------------------------------------------------------------------------
# CLI -- `memsom tuning list|get|set`
#
# The 13 forget params + memory_budget are NOT registered here (module
# docstring) -- `tuning set` therefore refuses every key today, honestly: it
# has zero canonical-sourced knobs to write, not a bug. `tuning list`/`get`
# already cover the ~35 knobs that used to be scattered os.environ.get()
# calls, which is the ratchet this phase actually closes.
# ---------------------------------------------------------------------------

def _cmd_tuning_list(args) -> None:
    import json
    if getattr(args, "json", False):
        print(json.dumps(as_json()))
        return
    for key in sorted(REGISTRY):
        knob = REGISTRY[key]
        print(f"{key:<32} {resolve(key)!r:<20} default={knob.default!r} "
              f"source={knob.source}")


def _cmd_tuning_get(args) -> None:
    if args.key not in REGISTRY:
        raise SystemExit(f"unknown knob: {args.key!r}")
    print(resolve(args.key))


def _cmd_tuning_set(args) -> None:
    if args.key not in REGISTRY:
        raise SystemExit(f"unknown knob: {args.key!r}")
    knob = REGISTRY[args.key]
    raise SystemExit(
        f"refused: {args.key!r} is {knob.source} -- env-sourced knobs are "
        f"read-only through this API (PLAN.md Sec2.2); set ${knob.source[4:]} "
        f"instead. No canonical-block (file-writable) knob is registered yet.")


def register(sub) -> None:
    p = sub.add_parser("tuning", help="report/inspect every tunable knob")
    tsub = p.add_subparsers(dest="tuning_command", required=True)
    p_list = tsub.add_parser("list", help="list every registered knob")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_tuning_list)
    p_get = tsub.add_parser("get", help="resolve one knob")
    p_get.add_argument("key")
    p_get.set_defaults(func=_cmd_tuning_get)
    p_set = tsub.add_parser("set", help="write a canonical-block knob (none registered yet)")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_set.set_defaults(func=_cmd_tuning_set)
