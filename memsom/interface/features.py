"""memsom.interface.features -- the Features registry composition root (PLAN.md Sec2.1).

kernel.features fixes the SHAPE (FeatureStatus, the Feature protocol); this
module is where the shape gets filled in, because answering "is retrieval.bge
available" or "is the broker configured" means reaching into retrieval,
federation, and bridge -- rank 3/6/7 -- which only rank 8 (interface) may do
downward. In-tree features register in _REGISTRANTS below (a static list);
external packages register through the existing `memsom.commands`
entry-point group (memsom/interface/cli.py:_register_plugin_commands), whose
contract widens from "register a CLI subcommand" to "register a Feature" --
see register_external().

Every probe here is CHEAP by construction: no model load, no network call
(the one apparent exception, retrieval.bge's availability check, is an
import-only probe cached by embed.py itself -- see its docstring). A feature
that needs a real reachability check (ollama, the qwen supervisor) is exactly
what `memsom doctor` already does elsewhere; features.py answers "is this
capability wired at all", not "is the far end up right now".
"""

from __future__ import annotations

import importlib.util

from memsom.kernel.features import FeatureStatus, VALID_STATES
from memsom.storage import schema as memsom_schema
from memsom import tuning as memsom_tuning


def _status(name, state, detail="", *, required=False, knobs=None) -> FeatureStatus:
    assert state in VALID_STATES, state
    return FeatureStatus(name=name, state=state, detail=detail, since=None,
                          required=required, knobs=list(knobs or []))


def _safe(name, fn) -> FeatureStatus:
    """Run one probe; a raise is reported as `error`, NEVER swallowed into
    `active` or dropped -- kernel.features' own vocabulary rule (Sec2.1/2.3)."""
    try:
        return fn()
    # FAILOPEN: allowed, error IS the reportable state (kernel.features vocabulary)
    except Exception as exc:  # noqa: BLE001
        return _status(name, "error", f"status() raised: {exc!r}")


# ---------------------------------------------------------------------------
# retrieval.*
# ---------------------------------------------------------------------------

def _degraded_count(conn) -> int:
    """MS-32's retrieval_degraded re-index queue: nodes whose vector embed
    failed and were indexed BM25-only. A CHEAP signal (one indexed SELECT, no
    model load, no network call -- Sec2.1's status() rule) that Phase 6 already
    built and left unread; this is the "last encode call actually failed"
    evidence retrieval.dense/retrieval.bge were missing."""
    if conn is None:
        return 0
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_degraded'"
    ).fetchone():
        return 0
    # node_id > 0: the negative sentinel is the query-encoder trail, not a node.
    return conn.execute(
        "SELECT COUNT(*) FROM retrieval_degraded WHERE node_id > 0").fetchone()[0]


def _coverage_degradation(conn):
    """The 2026-09 split shape, as a detail string or None: live nodes whose
    vector is under a model the active backend never reads (dense-dark), or a
    recent query-encode fallback. Read-only, two indexed SELECTs."""
    if conn is None:
        return None
    from memsom.retrieval import retrieve as memsom_retrieve
    cov = memsom_retrieve.embedding_coverage(conn)
    parts = []
    if cov["split"] or cov["dark"]:
        stored = ", ".join(f"{m}={n}" for m, n in sorted(cov["by_model"].items())) or "none"
        parts.append(f"{cov['dark']}/{cov['total']} live node(s) dense-dark "
                     f"(reads model={cov['active_model'] or 'none'}; stored: {stored})"
                     " -- run `memsom reindex`")
    last = memsom_retrieve.last_query_fallback(conn)
    if last and (last["age_s"] is None
                 or last["age_s"] <= memsom_retrieve.QUERY_DEGRADED_RECENT_S):
        parts.append(f"query encoder unreachable at {last['recorded_at']}: {last['reason']}")
    return "; ".join(parts) or None


def _pin_note(conn):
    if conn is None:
        return ""
    from memsom.retrieval import retrieve as memsom_retrieve
    try:
        pinned = memsom_retrieve.pinned_backend(conn)
    # FAILOPEN: allowed -- status() must report, never raise, on an odd store.
    except Exception:
        pinned = None
    return f"; store pinned to {pinned!r}" if pinned else "; store not yet pinned"


def _retrieval_dense(conn):
    from memsom.retrieval import embed as memsom_embed
    backend = memsom_embed.backend(conn)
    if backend != "ollama":
        return _status("retrieval.dense", "disabled",
                       f"embed.backend={backend!r} (ollama not selected)",
                       knobs=["embed.backend"])
    knobs = ["embed.backend", "retrieval.embed_url", "retrieval.embed_model"]
    n = _degraded_count(conn)
    if n:
        return _status("retrieval.dense", "degraded",
                       f"{n} node(s) BM25-only -- vector embed failing "
                       "(see retrieval_degraded)", knobs=knobs)
    cov = _coverage_degradation(conn)
    if cov:
        return _status("retrieval.dense", "degraded", cov, knobs=knobs)
    return _status("retrieval.dense", "active",
                   "ollama selected; reachability is checked live by `memsom doctor`"
                   + _pin_note(conn), knobs=knobs)


def _retrieval_bge(conn):
    from memsom.retrieval import embed as memsom_embed
    backend = memsom_embed.backend(conn)
    if backend != "bge-m3":
        return _status("retrieval.bge", "disabled",
                       f"embed.backend={backend!r} (bge-m3 not selected)",
                       knobs=["embed.backend", "retrieval.bge_device"])
    knobs = ["embed.backend", "retrieval.bge_device"]
    via = memsom_embed.encode_via()
    if via == "supervisor":
        # Supervisor mode never imports torch into this process (the MCP server
        # is a Claude Code child); the encode path is the local supervisor,
        # reachable now or spawnable on demand. A cheap /health probe, no model.
        from memsom.retrieval import bge_client
        knobs = ["embed.backend", "retrieval.bge_encode_via", "retrieval.bge_url",
                 "retrieval.bge_spawn_cmd", "retrieval.bge_spawn_timeout",
                 "retrieval.bge_idle_ttl"]
        if not bge_client.configured():
            return _status("retrieval.bge", "degraded",
                           "bge_encode_via=supervisor but retrieval.bge_url is unset",
                           knobs=knobs)
        if bge_client.supervisor_reachable():
            path_note = f"supervisor reachable at {bge_client._url()}"
        elif bge_client.spawn_cmd():
            path_note = (f"supervisor idle at {bge_client._url()}; spawned on demand "
                         f"by bge_spawn_cmd (cold start on the next query)")
        else:
            return _status("retrieval.bge", "degraded",
                           f"supervisor unreachable at {bge_client._url()} and no "
                           "bge_spawn_cmd to start it -- queries run BM25-only",
                           knobs=knobs)
    elif not memsom_embed.bge_available():
        if via == "auto" and memsom_embed._supervisor_possible():
            from memsom.retrieval import bge_client
            path_note = f"torch not importable; supervisor path via {bge_client._url()}"
        else:
            return _status("retrieval.bge", "absent",
                           "FlagEmbedding/torch/numpy not importable",
                           required=False, knobs=["embed.backend", "retrieval.bge_device"])
    else:
        path_note = "bge-m3 selected and importable"
    n = _degraded_count(conn)
    if n:
        return _status("retrieval.bge", "degraded",
                       f"{n} node(s) BM25-only -- bge encode (and Ollama "
                       "fall-through) failing (see retrieval_degraded)", knobs=knobs)
    cov = _coverage_degradation(conn)
    if cov:
        return _status("retrieval.bge", "degraded", cov, knobs=knobs)
    return _status("retrieval.bge", "active", path_note + _pin_note(conn), knobs=knobs)


def _retrieval_colbert():
    from memsom.retrieval import embed as memsom_embed
    if memsom_embed.backend() != "bge-m3":
        return _status("retrieval.colbert", "disabled", "requires embed.backend=bge-m3",
                       knobs=["retrieval.colbert_candidates", "retrieval.colbert_maxlen"])
    return _status("retrieval.colbert", "active", "colbert re-rank window active",
                   knobs=["retrieval.colbert_candidates", "retrieval.colbert_maxlen"])


def _retrieval_ppr():
    # retrieve_graph is a pure function over rel_edges -- always present, never
    # network/model-dependent; it degrades to plain retrieve() on empty edges.
    from memsom.retrieval import retrieve as memsom_retrieve
    if not hasattr(memsom_retrieve, "retrieve_graph"):
        return _status("retrieval.ppr", "absent", "retrieve_graph not found")
    return _status("retrieval.ppr", "active",
                   "graph re-rank over rel_edges (wikilinks); degrades to plain "
                   "retrieve() on empty edges")


# ---------------------------------------------------------------------------
# code_rag (+qwen)
# ---------------------------------------------------------------------------

def _code_rag():
    from memsom.retrieval import code_index as memsom_code_index
    if not memsom_code_index._enabled():
        return _status("code_rag", "disabled", "MEMSOM_CODE_RAG / panel flag not set",
                       knobs=["code_rag.enabled", "code_rag.qwen_url"])
    qwen_url = memsom_tuning.resolve("code_rag.qwen_url")
    detail = f"enabled; qwen supervisor url={qwen_url or '(default)'}"
    return _status("code_rag", "active", detail,
                   knobs=["code_rag.enabled", "code_rag.qwen_url"])


# ---------------------------------------------------------------------------
# contradict.nli
# ---------------------------------------------------------------------------

def _contradict_nli():
    from memsom.lifecycle import contradict as memsom_contradict
    if not memsom_contradict.enabled():
        return _status("contradict.nli", "disabled", "$MEMDAG_CONTRADICT not opted in",
                       knobs=["contradict.enabled", "contradict.nli_enabled"])
    if str(memsom_tuning.resolve("contradict.nli_enabled")).strip().lower() not in (
            "1", "true", "yes", "on"):
        return _status("contradict.nli", "active",
                       "structured tier only ($MEMDAG_CONTRADICT_NLI not opted in)",
                       knobs=["contradict.enabled", "contradict.nli_enabled"])
    if not memsom_contradict.nli_available():
        return _status("contradict.nli", "absent", "torch/transformers not importable",
                       knobs=["contradict.nli_enabled", "contradict.nli_model"])
    return _status("contradict.nli", "active", "semantic NLI tier importable",
                   knobs=["contradict.nli_enabled", "contradict.nli_model",
                          "contradict.nli_threshold"])


# ---------------------------------------------------------------------------
# gate3.*
# ---------------------------------------------------------------------------

def _gate3_hook():
    from memsom.bridge import hook as memsom_hook
    mode = memsom_hook.hook_mode()
    path = memsom_hook.hook_policy_path()
    policy_detail = "built-in default policy" if path is None else f"override policy: {path}"
    return _status("gate3.hook", "active", f"{policy_detail}; mode={mode}",
                   knobs=["bridge.hook_policy_path", "bridge.hook_mode",
                          "bridge.hook_shadow_log"])


def _gate3_broker():
    from memsom.federation import broker as memsom_broker
    path = memsom_broker.default_config_path()
    if not path.exists():
        # ARCH-15: the broker module is present (no dependency missing); a
        # missing config file is operator state -> `disabled`, not `absent`.
        return _status("gate3.broker", "disabled",
                       f"not configured: no broker config at {path}",
                       knobs=["federation.broker_config_path"])
    return _status("gate3.broker", "active", f"config present: {path}",
                   knobs=["federation.broker_config_path"])


# ---------------------------------------------------------------------------
# obsidian / federation.sync / bridge.sync
# ---------------------------------------------------------------------------

def _obsidian():
    vault = memsom_tuning.resolve("obsidian.vault")
    if not vault:
        return _status("obsidian", "disabled", "no vault configured",
                       knobs=["obsidian.vault"])
    return _status("obsidian", "active", f"vault={vault}", knobs=["obsidian.vault"])


def _federation_sync(conn):
    from memsom.federation import federation as memsom_federation
    if conn is None:
        return _status("federation.sync", "active",
                       "core present; trusted-origin count needs a DB connection")
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trusted_origins'"
    ).fetchone():
        return _status("federation.sync", "disabled",
                       "no trusted origins registered (table absent)")
    origins = conn.execute("SELECT origin FROM trusted_origins").fetchall()
    if not origins:
        return _status("federation.sync", "disabled", "no trusted origins registered")
    return _status("federation.sync", "active", f"{len(origins)} trusted origin(s)")


def _bridge_sync():
    return _status("bridge.sync", "active",
                   "core bridge (claude-sync, bridge-render) always present")


# ---------------------------------------------------------------------------
# llm / anticipatory / distill
# ---------------------------------------------------------------------------

def _llm():
    return _status("llm", "active",
                   "opt-in per call (`ask --llm`); reachability checked at call time",
                   knobs=["llm.model", "llm.url", "llm.ollama_keep_alive",
                          "llm.cite_overlap"])


def _anticipatory():
    from memsom.lifecycle import anticipatory  # noqa: F401 -- presence probe only
    return _status("anticipatory", "active", "pure heuristic, no external dependency")


def _distill():
    return _status("distill", "active",
                   "export-training / distill-plan always available (writes files, "
                   "spawns nothing)", knobs=["llm.model"])


# ---------------------------------------------------------------------------
# remote.* -- Phase 10 deployment modes not yet shipped
# ---------------------------------------------------------------------------

def _remote_server():
    import memsom
    from memsom.storage import settings as memsom_settings
    settings = memsom_settings.load_settings(memsom.DATA_DIR)
    if settings.get("mode") != "server":
        return _status("remote.server", "disabled", "mode != 'server' (see `memsom setup`)",
                       knobs=["remote.action_gate_mode"])
    bind = settings.get("bind") or "(auto-discovered)"
    gate_mode = memsom_tuning.resolve("remote.action_gate_mode")
    return _status("remote.server", "active",
                   f"bind={bind} port={settings.get('port')} action_gate={gate_mode}",
                   knobs=["remote.action_gate_mode"])


def _remote_client():
    import memsom
    from memsom.storage import settings as memsom_settings
    settings = memsom_settings.load_settings(memsom.DATA_DIR)
    if settings.get("mode") != "client":
        return _status("remote.client", "disabled", "mode != 'client' (see `memsom setup`)",
                       knobs=[])
    url = settings.get("remote_server_url") or "(unset)"
    return _status("remote.client", "active", f"server={url}", knobs=[])


# ---------------------------------------------------------------------------
# telemetry -- the panel's contract surface (PROMOTE-Q11-PANEL.md B1)
# ---------------------------------------------------------------------------

_TELEMETRY_KEYS = frozenset((
    "generated", "last_consolidation", "totals", "tier", "types", "hist",
    "top_access", "scatter", "growth", "stale", "budget", "sessions",
    "thresholds", "graph",
))


def _telemetry(conn):
    from memsom.interface import telemetry as memsom_telemetry  # noqa: F401 -- presence
    if conn is None:
        return _status("telemetry", "active",
                       "module present; full payload needs a DB connection",
                       knobs=["telemetry.consolidation_dir"])
    if not memsom_schema.column_exists(conn, "nodes", "forget_rs"):
        return _status("telemetry", "degraded",
                       "forget_* columns absent: run `memsom bridge-render` once",
                       knobs=["telemetry.consolidation_dir"])
    return _status("telemetry", "active",
                   "panel contract surface reachable (forget_* columns present)",
                   knobs=["telemetry.consolidation_dir"])


# ---------------------------------------------------------------------------
# panel -- external package, attaches via the memsom.commands entry-point group
# ---------------------------------------------------------------------------

_MEMSOM_HOOK_SUBS = ("bridge-render", "hook-prompt", "hook-pre", "hook-post")


def _claude_hooks(home=None):
    """Are the Claude Code hooks that call memsom actually runnable?

    The failure this catches (2026-09-02, live): a reinstall moved memsom.exe
    from the user-site Scripts dir to the system one, settings.json kept the
    old absolute path, and every session end printed a Stop-hook error
    instead of rendering MEMORY.md. `memsom doctor` / `memsom features` now
    say so and name the fix (`memsom wire-claude`), which re-points every
    stale memsom hook in place.
    """
    from memsom.bridge import wire_claude as wc
    path = wc.settings_path(home)
    if not path.exists():
        return _status("claude.hooks", "disabled",
                       f"not wired: no {path} (run `memsom wire-claude`)")
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    hooks = data.get("hooks") if isinstance(data, dict) else None
    ours, stale = [], []
    for event, groups in (hooks or {}).items():
        for g in groups if isinstance(groups, list) else []:
            for h in (g.get("hooks") or []) if isinstance(g, dict) else []:
                cmd = h.get("command") if isinstance(h, dict) else None
                if cmd and any(s in cmd for s in _MEMSOM_HOOK_SUBS):
                    ours.append(f"{event}: {cmd}")
                    if wc.is_stale_command(cmd):
                        stale.append(f"{event} -> {wc.command_head(cmd) or cmd!r}")
    if not ours:
        return _status("claude.hooks", "disabled",
                       f"no memsom hook in {path} (run `memsom wire-claude`)")
    if stale:
        return _status("claude.hooks", "error",
                       "stale hook executable, session-end render is failing: "
                       + "; ".join(stale) + " -- run `memsom wire-claude`")
    return _status("claude.hooks", "active", f"{len(ours)} memsom hook(s) resolve in {path}")


def _panel():
    if importlib.util.find_spec("memsom_panel") is None:
        return _status("panel", "absent", "memsom-panel not installed")
    return _status("panel", "active", "memsom-panel importable")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRANTS = {
    "retrieval.dense": lambda conn: _retrieval_dense(conn),
    "retrieval.bge": lambda conn: _retrieval_bge(conn),
    "retrieval.colbert": lambda conn: _retrieval_colbert(),
    "retrieval.ppr": lambda conn: _retrieval_ppr(),
    "code_rag": lambda conn: _code_rag(),
    "contradict.nli": lambda conn: _contradict_nli(),
    "gate3.hook": lambda conn: _gate3_hook(),
    "gate3.broker": lambda conn: _gate3_broker(),
    "obsidian": lambda conn: _obsidian(),
    "federation.sync": lambda conn: _federation_sync(conn),
    "bridge.sync": lambda conn: _bridge_sync(),
    "llm": lambda conn: _llm(),
    "anticipatory": lambda conn: _anticipatory(),
    "distill": lambda conn: _distill(),
    "remote.server": lambda conn: _remote_server(),
    "remote.client": lambda conn: _remote_client(),
    "telemetry": lambda conn: _telemetry(conn),
    "panel": lambda conn: _panel(),
    "claude.hooks": lambda conn: _claude_hooks(),
}

_EXTERNAL: dict[str, object] = {}


def register_external(name, probe) -> None:
    """External packages (memsom-panel et al.) register through the existing
    `memsom.commands` entry-point group -- see cli.py:_register_plugin_commands.
    A plugin's entry point calls this instead of (or in addition to) mounting
    a subcommand; *probe* is `probe(conn) -> FeatureStatus`."""
    _EXTERNAL[name] = probe


def migrate(conn) -> None:
    """Idempotent: the feature_status table records state TRANSITIONS (the
    number that catches a multi-day silent degrade -- Sec2.1)."""
    memsom_schema.ensure_table(conn, """CREATE TABLE IF NOT EXISTS feature_status (
    name       TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    detail     TEXT,
    since      TEXT NOT NULL
  );""")


def record_transitions(conn, statuses: dict[str, FeatureStatus]) -> None:
    """Called from migrate_all: write a row per feature whose state CHANGED
    since the last observation, stamping `since` with now(). Never raises --
    a write failure here must not block the migration it is riding along."""
    import memsom
    try:
        migrate(conn)
        now = memsom.now_iso()
        with conn:
            for name, st in statuses.items():
                row = conn.execute(
                    "SELECT state FROM feature_status WHERE name=?", (name,)).fetchone()
                if row is None or row[0] != st["state"]:
                    conn.execute(
                        "INSERT INTO feature_status(name, state, detail, since) "
                        "VALUES (?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                        "state=excluded.state, detail=excluded.detail, since=excluded.since",
                        (name, st["state"], st["detail"], now))
    # FAILOPEN: allowed, a diagnostic side-write must never block migrate_all
    except Exception:  # noqa: BLE001
        pass


def all_statuses(conn=None) -> dict[str, FeatureStatus]:
    out = {}
    for name, probe in _REGISTRANTS.items():
        out[name] = _safe(name, lambda probe=probe: probe(conn))
    for name, probe in _EXTERNAL.items():
        out[name] = _safe(name, lambda probe=probe: probe(conn))
    if conn is not None:
        rows = conn.execute(
            "SELECT name, since FROM feature_status") if _table_exists(conn) else []
        since_by_name = {n: s for n, s in rows}
        for name, st in out.items():
            if name in since_by_name:
                st["since"] = since_by_name[name]
    return out


def _table_exists(conn) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_status'"
    ).fetchone() is not None


def _remote_server_features_block(statuses: dict) -> dict | None:
    """Client mode only (Sec3.6): fetch the SERVER's own `/features` block so
    `memsom features --json` shows BOTH blocks -- what this machine can do,
    and what the machine it talks to can do. Best-effort: an unreachable
    server degrades to a small error dict, never an exception -- a features
    probe must never crash the whole command."""
    import json
    from memsom.effects import net as memsom_net

    client = statuses.get("remote.client")
    if client is None or client["state"] != "active":
        return None
    import memsom
    from memsom.storage import settings as memsom_settings
    settings = memsom_settings.load_settings(memsom.DATA_DIR)
    url = (settings.get("remote_server_url") or "").rstrip("/")
    if not url:
        return {"error": "remote.client active but remote_server_url is unset"}
    token = settings.get("remote_device_token", "")
    try:
        body = memsom_net.fetch(f"{url}/features",
                                headers={"Authorization": f"Bearer {token}"}, timeout=3)
        return json.loads(body.decode("utf-8"))
    except (memsom_net.NetworkError, ValueError) as exc:
        return {"error": f"unreachable: {exc}"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_features(args):
    import json
    import memsom
    conn = None
    try:
        conn = memsom.get_connection(read_only=True)
    # FAILOPEN: allowed, no store yet -- probes that need a conn degrade
    except Exception:
        conn = None
    try:
        statuses = all_statuses(conn)
    finally:
        if conn is not None:
            conn.close()

    if getattr(args, "json", False):
        payload = dict(statuses)
        remote_block = _remote_server_features_block(statuses)
        if remote_block is not None:
            payload["_remote_server_features"] = remote_block
        print(json.dumps(payload, indent=2))
        return

    exit_bad = False
    for name in sorted(statuses):
        st = statuses[name]
        marker = "!" if st["state"] in ("error",) else " "
        print(f"{marker} {name:<20} {st['state']:<9} {st['detail']}")
        if st["required"] and st["state"] == "absent":
            exit_bad = True
    if exit_bad:
        raise SystemExit(1)


def register(sub) -> None:
    p = sub.add_parser("features", help="report every optional capability's live state")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_features)
