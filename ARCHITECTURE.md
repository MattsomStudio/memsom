# memsom — Architecture

A provenance-aware memory store for AI agents — *"version control for machine knowledge."* Snapshot at the current HEAD: 45+ runtime modules / ~19k LOC / ~995 tests. Pure-stdlib Python over a single SQLite file (the only required dependency); Ollama is optional and degrades gracefully.

## The mental model

Most memory systems store facts and relate-to links. memsom also stores **came-from** links — a derivation DAG — so every answer can be traced to its sources, revoked, blamed, and gated. Three conceptual tiers:

```
APPS       taint · blame · revocation · trust algebra · GraphRAG · anticipatory · weights · vault sync
SUBSTRATE  the derivation DAG   <- the product
STORAGE    SQLite (single file, stdlib sqlite3)   <- commodity, swappable
```

Two **orthogonal security axes**, both enforced structurally — by channel/provenance, never by inspecting content (an attacker controls content, never the channel it arrives on):

| Axis | Model | Direction | Column | Meaning |
|---|---|---|---|---|
| Integrity | Biba low-water-mark | `min()` floor | `label` 0–3 | how trustworthy — derived ≤ weakest parent |
| Confidentiality | Bell-LaPadula | `max()` ceiling | `conf_label` 0–3 | how secret — derived ≥ most-secret parent |

Integrity flows **down** (one external source poisons everything derived from it); confidentiality flows **up** (one secret source makes all descendants secret). Content can lower the integrity floor, never raise it.

## Data model (SQLite)

The **frozen core** owns two tables — `memsom/__init__.py` + `test_memsom.py` are kept byte-identical across every feature build (the trust anchor; baseline re-anchored at the 2026-07-01 memdag→memsom rename, which only renamed the file + its `import`):

```sql
nodes(id, content, channel, label, source_ref, created_at,
      tombstoned, tombstoned_at, revoke_reason)
   CHECK channel IN ('endorsed','user','agent-derived','external')
   CHECK label BETWEEN 0 AND 3
edges(child, parent)                 -- came-from / provenance (the DAG)
```

Feature modules extend `nodes` **additively** via `memsom_schema.add_column` (never altering frozen behavior): `content_hash, status, quarantine_reason/at, redacted/at, redact_reason, archived/at, conf_label, uuid, origin, obsidian_path/mtime`. Each owns its own tables: `rel_edges` (relates-to/wikilinks), `postings/docstats/embeddings` (retrieval index), `elevations` (audited integrity-elevation log), `gate_log`, `claims/claim_assertions/corroborations/independence_roots`, `query_log/prefetch_cache`, `trusted_origins`, `redaction_log`.

**Two edge types, deliberately separate:**
- `edges` (**came-from**): causal, one-hop, set by `derive_node` — and by the fact
  layer's `depends_on:` (a measurement *derives from* the hardware it was taken on),
  so retiring a fact stales its dependents through the same cascade. Carries the
  Biba integrity floor.
- `rel_edges` (**relates-to**): associative/bidirectional, set by `relate` and Obsidian/memory wikilinks. Navigated by `neighborhood()` with the same floor-propagation discipline but orthogonal to derivation — and deliberately invisible to the cascade (association must never propagate staleness or revocation).

## Channel / integrity model

```
RANK = endorsed:3 > user:2 > agent-derived:1 > external:0
```

Channel is stamped by the transport/adapter, **never inferred from content**. `insert_node` stamps `label = RANK[channel]`. `derive_node` mints `agent-derived` nodes with `label = min(parent labels)` — the laundering-proof property: you cannot wash external content up to `user` by summarizing it. (In an injection benchmark, integrity held: laundering 0.00 / gated-ASR 0.00 across 96 attacks, at ~0 tokens and single-digit milliseconds per write.)

## Frozen core — `memsom/__init__.py`

- `insert_node(content, channel, label=None, source_ref)` — source nodes; label from channel.
- `derive_node(content, parent_ids)` — mints `agent-derived`, `label=min(parents)`, writes provenance edges, all under `BEGIN IMMEDIATE` so a concurrent revoke can't race the liveness check.
- `revoke_cascade(seed, reason)` — tombstones a node and all transitive descendants (recursive CTE, `UNION` dedupes cycles). **First-death-wins**: rows/edges/content survive, liveness is filtered at read time.
- `compose(question, sources)` — **deterministic, LLM-free** composer: keyword-matched sentences → bulleted answer with `[mem:id|channel]` citations. Same inputs → byte-identical output. Returns `(None, [])` if no live source → never an unprovenanced answer.

## Layers (module map)

**Schema (`memsom_schema`)** — `add_column`/`ensure_table` (idempotent DDL), `PRAGMA user_version` versioned migrations, and **`taint_filter_clauses` — the ONE shared "untainted pool" WHERE-fragment** every read path inherits (`tombstoned=0 AND status!='quarantined' AND redacted=0 AND archived=0 AND conf_label<=clearance`). Single source of truth = no pool drift.

**Integrity enforcement:**
- `memsom_ingest` — the real write path: chunking, content-hash dedup (channel-aware), caller-layer guards **F-13** (`MEMDAG_CHANNEL_CEILING` caps ingest rank) and **F-14** (label dictated solely by channel, never a caller-supplied value).
- `memsom_recompute` — order-independent multi-hop re-flooring (`effective_labels`, memoized DFS, O(V+E)).
- `memsom_gate` — `check_action(node, required_floor)`: **the only enforcement point** (read paths never gate; the Windows-MIC / CaMeL action-boundary pattern). Names the weakest-leaf culprit on deny; logs every call.
- `memsom_trust` — lattice `meet`/`join` + **audited `elevate`** (manual only; force-gated on the provenance floor, not the channel string).
- `memsom_corroborate` — content-free trust *lift*: k independent **registered** roots asserting the same structured claim mint a lift node, **capped at agent-derived(1)**, fail-closed, auto-dropped if any asserting source is revoked.

**Confidentiality (`memsom_confid`)** — Bell-LaPadula: `classify` (manual roots), `recompute_conf` (`max(parents)`, high-water-mark), order-independent `recompute_conf_all` (Gauss-Seidel fixpoint). Clearance filters results, not integrity conduction.

**Lifecycle:**
- `memsom_redact` — destroys payload (`content=''`) but **preserves DAG shape** (edges/label/dates survive → blame still works), transitive cascade, and **F-15** purges the retrieval index so a redacted node can't resurface via BM25/vector.
- `memsom_quarantine` — consolidation gate: external-tainted agent-derived nodes get `status='quarantined'`; promotion requires a live endorsed ancestor. External taint can never silently promote.
- `memsom_compact` — consolidation engine: groups live episodes, mints a summary (`label=min`), archives members (edges preserved for blame).
- `memsom_heal` — invariant checker/rebuilder (dangling edges, stored-vs-effective integrity/conf mismatch, orphaned live children).
- `memsom_blame` — DFS trace from a node to all root sources; immutable history shows tombstoned/redacted state; clearance suppresses content but keeps metadata for audit.
- `memsom_federation` — multi-machine sync: `export/import_changeset`, **first-death-wins** (monotonic — importing a stale live copy can't resurrect a tombstone), **trusted-origin allowlist** (default-deny; untrusted imports clamp channel→external + a conf floor, edges origin-authenticated on UUIDs actually inserted).

**Retrieval & answering:**
- `memsom_retrieve` — hybrid **BM25 (pure stdlib) + optional Ollama vectors**, RRF-fused, pool-filtered. `retrieve_graph` = **GraphRAG-lite**: re-ranks the retrieved pool by the wikilink graph — a relevant note linked from a strong hit is boosted into the top-k, without ever widening past the trusted pool (`base.keys() ⊆ pool`).
- `memsom_anticipatory` — surprise-gated writes (cite existing on low novelty) + prefetch cache. Reads/learns **only from untainted memory** — which is why taint had to ship before it.
- `memsom_llm` — **opt-in** Ollama compose behind a **citation firewall**: every line must carry a valid `[mem:id|channel]` tag validated against real sources, else it falls back to deterministic `compose`. Guarantees per-line provenance (not semantic faithfulness — a documented boundary).
- `memsom_distill` / `memsom_reflex` — provenance-gated training export: only untainted + consolidated memory is eligible to bake into weights.

**Surfaces:**
- `memsom_cli` — unified CLI (75+ subcommands), `migrate_all` (every module's idempotent migration + versioned steps), enhanced `ask` orchestrating `--retrieve / --graph / --anticipate / --llm`.
- `memsom_mcp` — stdio MCP server (JSON-RPC 2.0, 17 tools), all diagnostics to stderr. The tool→argv mapping and end-to-end dispatch of every tool are pinned by tests (`tests/test_memsom_mcp.py`).
- `memsom_obsidian` — vault integration: `sync_vault` (notes → nodes, `[[wikilinks]]` → `rel_edges`), `export_note`, `watch_vault`. A note's frontmatter `memsom-channel` can only **lower** integrity (`min(default, declared)`) — closing the write→re-ingest laundering loop.
- `memsom.bridge.facts` — the fact layer: single-source-of-truth values (`type: fact` memories with `value`/`unit`/`last-verified`) referenced as `[[fact_<stem>]]` from other memories and resolved **at read time** (digest = current value; retrieve = drift vs the referencing memory's age; retired = last known, flagged). The supersede chain is the value history (`fact-log`); `fact-set` edits the file, never the DB. `depends_on:` between facts materializes into `edges` so the stale cascade covers real derivation. Core rule: memories are immutable history, facts carry the lifecycle, all reconciliation happens at read.
- `memsom.interface.panel` — the live tuning + telemetry panel (`memsom panel --profile <host-profile.json>`): a loopback-only stdlib HTTP server (hard bind refusal off loopback, Host-header allowlist, JSON+Origin CSRF checks, sha256-pinned inline script under a `default-src 'none'` CSP) over four knob providers — canonical.json runtime params, generic JSON key-paths, `set KEY=VALUE` env files, Windows scheduled-task cadences (degrades to a copyable elevated command when the token can't write) — with bounds validation that rejects (never clamps) and a two-phase JSONL audit log (intent line gates the write; a pending line with no result is a crash marker). Machine-specific knob inventory lives in the host profile, never in the repo. `memsom.interface.telemetry` samples RAM (working set AND commit charge — `GetPerformanceInfo`), GPU, disk growth, TCP service probes, and Syncthing REST at request time; psutil is an optional extra (`pip install memsom[panel]`), never a core dependency.
- `memsom.providers.kernels` — persistent CLI kernels (panel Phase 4): a kernel is a durable panel object wrapping a Claude Code/Codex session — the CLI's own transcript carries the context, so kernels survive app close, panel restart, and reboot. Prompts are per-run headless resumes (`claude --resume <ptr> -p --output-format stream-json`, prompt on stdin) streamed into the inference sessions dir (`krn-` prefixed) so the existing `/api/inference` cursor read path serves them unchanged. `session_ptr` moves ONLY on a run's `result` event — never by scanning the project dir (recency ≠ lineage; an auto-scan once adopted an unrelated live session and forked it). Per-kernel lock → 409 busy; interactive-terminal exclusivity is enforced app-side (the panel can't see the sessiond daemon by design). The desktop-app half — the `memsom-sessiond` durable-terminal daemon (PTYs that outlive the app; WS attach + ring replay) and the KERNELS rail/terminal bridge — lives in the `memsom-agentic-os` repo (v0.5.0). Lifecycle contract: sessiond dies at logoff (sessions surface as `lost` tombstones, never ghosts); kernels survive everything.
- **Runtime params (`forget.load_params`)** — the forgetting layer's 13 compute params + `memory_budget` are read per-run from the store's `.weights/canonical.json` `params` block (tolerant merge over DEFAULTS; degenerate values rejected with caller-logged warnings). `bridge-render` computes with them and `digest`/`audit`/`dashboard` resolve the budget from the same source — one knob file drives the whole render path. The golden DEFAULTS dict stays byte-identical to the original `mem_weights.py` (parity-tested, including under param overrides).
- `memsom_config` (MCP client wiring), `bootstrap.py` (one-command install), `memsom_chats` (chat import).

## End-to-end: `ask "X" --retrieve --graph --clearance public`

1. `cmd_ask` → `migrate_all`, validate clearance, `_build_pool` via `taint_filter_clauses` (drops tombstoned/quarantined/redacted/archived/above-clearance).
2. `retrieve_graph`: BM25+vector over the pool → seed top-k → `neighborhood(hops)` boosts linked-and-relevant nodes (re-rank within the pool only).
3. (optional) `surprise_gated_write`: cite existing if low-novelty, else continue.
4. `compose` (deterministic) **or** `llm_compose` (citation-firewalled) → `(text, used_ids)`.
5. `derive_node(text, used_ids)`: new `agent-derived` node, `label=min(used)`, provenance edges, `conf=max(used)` — under a write lock.
6. The answer carries `[mem:id|channel]` citations and is itself now blamable, revocable, and gateable.

## Load-bearing invariants

- Frozen core is byte-identical across all builds; features are additive-only.
- One taint primitive (`taint_filter_clauses`) feeds every read pool.
- Enforcement is **action-time only** (`check_action`); reads are transparent.
- History is immutable: tombstone (revoke) / redact (wipe payload, keep shape) / archive (compact) / quarantine (gate) — never an in-place mutation or hard delete by disuse.
- Integrity min-floor + confidentiality max-ceiling, both content-independent.
- Deterministic by default; the LLM is opt-in and firewalled.

## Honest boundaries

- Federation origin is honor-system (transport-authenticated in single-operator deployments); signed changesets are the documented next step if untrusted multi-party federation enters scope.
- The LLM firewall guarantees per-line provenance, not semantic faithfulness.
- Corroboration claim-extraction is deterministic (hashes/IPs/ports/semver/key=value), not prose-semantic.
- `retrieve_graph` boosts linked-relevant notes; it does not yet pull pure-context (zero-lexical-overlap) neighbors — deliberate, because `compose` force-includes every pool member.
