# memsom — Architecture

A provenance-aware memory store for AI agents — *"version control for machine knowledge."* Snapshot at the current HEAD: 45+ runtime modules / ~19k LOC / ~1350 tests. Pure-stdlib Python over a single SQLite file (the only required dependency); Ollama is optional and degrades gracefully.

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

The **frozen core** owns two tables — `memsom/__init__.py` is a lazy facade since the Phase-2 core split; the frozen-core BEHAVIOUR (`python -m memsom seed|ask|explain|revoke|dump`, `test_memsom.py`) is what stays byte-identical across every feature build (the trust anchor; baseline re-anchored at the 2026-07-01 memdag→memsom rename, which only renamed the file + its `import`):

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

## Frozen core — `memsom/integrity/dag.py` (insert_node, derive_node, revoke_cascade) + `memsom/kernel/compose.py` (compose), re-exported lazily by `memsom/__init__.py`

- `insert_node(content, channel, label=None, source_ref)` — source nodes; label from channel.
- `derive_node(content, parent_ids)` — mints `agent-derived`, `label=min(parents)`, writes provenance edges, all under `BEGIN IMMEDIATE` so a concurrent revoke can't race the liveness check.
- `revoke_cascade(seed, reason)` — tombstones a node and all transitive descendants (recursive CTE, `UNION` dedupes cycles). **First-death-wins**: rows/edges/content survive, liveness is filtered at read time.
- `compose(question, sources)` — **deterministic, LLM-free** composer: keyword-matched sentences → bulleted answer with `[mem:id|channel]` citations. Same inputs → byte-identical output. Returns `(None, [])` if no live source → never an unprovenanced answer.

## Import layering (the 9-rank contract)

Enforced by `lint-imports` (`.importlinter`, a `layers` contract — the whole thing is one binary green/red check, no `independence`/sibling contracts to drift). Top imports down, never up, no skip is a violation except into `memsom.kernel` (stdlib-only, rank 0, reachable from anywhere):

```
memsom.interface  >  memsom.bridge  >  memsom.federation  >  memsom.distill  >
memsom.lifecycle  >  memsom.retrieval  >  memsom.integrity  >
{memsom.effects | memsom.storage}  >  memsom.tuning  >  memsom.kernel
```

`effects` and `storage` are tied siblings directly on `kernel`; no import either way, ENFORCED by the `|` independence form in `.importlinter` (`:` would allow sibling imports). `memsom.tuning` sits one rank above `kernel` — every other package may import it downward to centralize env reads; `kernel` cannot import it (kernel is stdlib-only) and keeps its own two bootstrap-only env reads instead (`MEMDAG_HOME`/`MEMDAG_DB` in `storage/db.py`, `MEMDAG_BRIDGE_MEMORY_DIR` in `kernel/paths.py` — both named exceptions in `tuning.py`'s own docstring and enforced by `scripts/env_ratchet.py`).

Invariants the layering exists to hold, each with its own gate script under `scripts/` (control-tested: a planted violation must turn the gate red):
- **Seam direction** (`seam-direction` contract) — `memsom` never imports `memsom_panel`; the panel imports memsom, one-way. memsom lints clean on a machine with no panel installed at all — `memsom_panel` is deliberately not a root package to `.importlinter`.
- **Frozen core** — `memsom/__init__.py` imports no feature subpackage (interface/bridge/federation/distill/lifecycle/retrieval/integrity).
- **One connection owner** (`gate_readpool.py`) — DB connections come only from `memsom.storage.db.get_connection`; no bare `sqlite3.connect(` outside `storage/db.py`.
- **One taint primitive** — every read pool over `nodes` goes through `memsom.storage.schema.taint_filter_clauses`; no hand-rolled WHERE clause.
- **One write path** (`gate_writeowner.py`) — new nodes are minted only via `ingest`; `insert_node` has exactly one caller module outside it (`interface/cli.py`'s frozen demo seed).
- **Knobs, not bare env reads** (`env_ratchet.py`) — every tunable is `memsom.tuning.resolve("<key>")`, registered with type/default/bounds/doc; the exceptions above are the only bare `os.environ` sites in the package.
- **Adjudicated fail-open** (`failopen_annotations.py`) — a bare `except:`/`except Exception:` must carry an immediately-preceding `# FAILOPEN: <decision>` line naming why the broad catch is deliberate.
- **RMW guarded** (`writer_census.py`) — a function that SELECTs from `nodes`/`edges` and later writes the same tables opens `BEGIN IMMEDIATE` or carries the census annotation.
- **Effects boxed** (`effects_ratchet.py`, `upward_imports.py`) — subprocess/urllib live under `memsom/effects/` (one importer each); every layer calls them (10 importers above integrity: broker, features, ingest, saveall, compact, doctor, code_index, llm, qwen_embed, retrieve) — the boxing is that only `effects/proc.py` and `effects/net.py` import subprocess/urllib, not that no one above integrity reaches `effects`. Raw TCP listeners (the panel-facing `interface/serve.py`, the prompt hook's `retrieval/warm.py`) are the documented exception to the *subprocess/urllib* half of this rule — they are the transport itself, not a side effect a higher layer reaches past; a listening socket is not what `effects/net.py` (outbound fetch) or `effects/proc.py` (child processes) exist to box.
- **Features are additive** — an optional capability registers in `memsom/interface/features.py`'s `_REGISTRANTS` with a `status()` probe reporting `active | degraded | disabled | error | absent`, never raising through (`_safe` wraps every probe).

## Layers (module map)

**Schema (`memsom_schema`)** — `add_column`/`ensure_table` (idempotent DDL), `PRAGMA user_version` versioned migrations, and **`taint_filter_clauses` — the ONE shared "untainted pool" WHERE-fragment** every read path inherits (`tombstoned=0 AND status!='quarantined' AND redacted=0 AND archived=0 AND conf_label<=clearance`). Single source of truth = no pool drift.

**Integrity enforcement:**
- `memsom_ingest` — the real write path: chunking, content-hash dedup (channel-aware), caller-layer guards **F-13** (`MEMDAG_CHANNEL_CEILING` caps ingest rank) and **F-14** (label dictated solely by channel, never a caller-supplied value).
- `memsom_recompute` — order-independent multi-hop re-flooring (`effective_labels`, memoized DFS, O(V+E)).
- `memsom_gate` — `check_action(node, required_floor)`: an **advisory node-integrity
  oracle** (CLI `check-action` / MCP tool only — zero internal callers, MS-40); read
  paths never gate. Names the weakest-leaf culprit on deny; logs every call. Runtime
  enforcement of Gate #3 (what a session may *do*) is `memsom_capgate.check_capability`,
  interposed by the broker and the native-tool hooks before a consequential call is
  forwarded — the Windows-MIC / CaMeL action-boundary pattern lives there, not here.
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
- `memsom_mcp` — stdio MCP server (JSON-RPC 2.0, 19 tools), all diagnostics to stderr. The tool→argv mapping and end-to-end dispatch of every tool are pinned by tests (`tests/test_memsom_mcp.py`).
- `memsom_obsidian` — vault integration: `sync_vault` (notes → nodes, `[[wikilinks]]` → `rel_edges`), `export_note`, `watch_vault`. A note's frontmatter `memsom-channel` can only **lower** integrity (`min(default, declared)`) — closing the write→re-ingest laundering loop.
- `memsom.bridge.facts` — the fact layer: single-source-of-truth values (`type: fact` memories with `value`/`unit`/`last-verified`) referenced as `[[fact_<stem>]]` from other memories and resolved **at read time** (digest = current value; retrieve = drift vs the referencing memory's age; retired = last known, flagged). The supersede chain is the value history (`fact-log`); `fact-set` edits the file, never the DB. `depends_on:` between facts materializes into `edges` so the stale cascade covers real derivation. Core rule: memories are immutable history, facts carry the lifecycle, all reconciliation happens at read.
- `memsom_panel` (external package, attached via the `memsom.commands` plugin entry-point group — see "Seam direction" under Import layering above; **not** a module in this repo) — the live tuning + telemetry panel (`memsom panel --profile <host-profile.json>`): a loopback-only stdlib HTTP server (hard bind refusal off loopback, Host-header allowlist, JSON+Origin CSRF checks, sha256-pinned inline script under a `default-src 'none'` CSP) over four knob providers — canonical.json runtime params, generic JSON key-paths, `set KEY=VALUE` env files, Windows scheduled-task cadences (degrades to a copyable elevated command when the token can't write) — with bounds validation that rejects (never clamps) and a two-phase JSONL audit log (intent line gates the write; a pending line with no result is a crash marker). Machine-specific knob inventory lives in the host profile, never in the repo. Panel-facing memory telemetry is `memsom.interface.telemetry` (`build_telemetry(memory_dir=None, *, conn=None)`, `load_weights(conn=None)`, `default_memory_dir`), the frozen contract pinned by `tests/test_panel_contract.py`; `memsom.interface.dashboard` is a deprecation shim over it (removed one release after the panel switches). Host telemetry (RAM, GPU, disk, TCP probes, Syncthing) lives in `memsom_panel/telemetry.py`, not here; this store stays stdlib-only.
- **Runtime params (`forget.load_params`)** — the forgetting layer's 13 compute params + `memory_budget` / `memory_max_lines` are read per-run from the store's `.weights/canonical.json` `params` block (tolerant merge over DEFAULTS; degenerate values rejected with caller-logged warnings). `bridge-render` computes with them and `digest`/`audit`/`index-stats` resolve the budget from the same source — one knob file drives the whole render path. The golden DEFAULTS dict stays byte-identical to the original `mem_weights.py` (parity-tested, including under param overrides).
- `memsom_config` (MCP client wiring), `bootstrap.py` (one-command install), `memsom_chats` (chat import).
- **Prompt hook + warm endpoint (`interface/prompt_hook.py`, `retrieval/warm.py`).** A `UserPromptSubmit` hook (`memsom hook-prompt`) runs before every prompt, so it has a sub-second budget and must never load a model. `hook-query` therefore asks the long-lived MCP server first: `serve_stdio` starts a `WarmServer` — one TCP listener bound to `127.0.0.1` on an ephemeral port, one JSON line in / one out, port + a per-process random token published in `<db>.warm.json` (owner-only perms). It refuses non-loopback peers (bind AND a per-request peer check), a wrong token (constant-time compare), and every method but `retrieve`, which goes through the same `retrieve()` pool filters (taint + clearance) as the CLI. **Wedge hardening (2026-08-20, after a live listener sat LISTENING for hours with connects piling up in CLOSE_WAIT and never served, so the hook burned its whole deadline on 28/37 prompts):** each connection runs in its own daemon thread with a 300 ms socket timeout and is closed in `finally`; the accept loop is wrapped so a handler/accept exception is logged and the loop re-entered; `ping` answers liveness with no DB work; `serve_stdio` runs a `WarmWatchdog` that self-pings every 60 s and `restart()`s the listener (new port + token, endpoint file rewritten) on failure, and removes the endpoint file on every exit path. Client side, `warm_query` caps the whole warm round-trip at `WARM_BUDGET_S` (250 ms, read included) and treats timeout/short read/bad reply as DOWN (fall back, never wait); after two consecutive post-connect failures against a live server pid it writes `<db>.warm.down.json` and skips the warm path for `BACKOFF_S` (30 s) — a refused connect does not count (stale file, not a wedge), a different port invalidates the counter, and a success or `start()` clears it. If the file is absent, the connect is refused, the server is slow, or a backoff window is open, the hook falls back to in-process BM25 with `MEMDAG_EMBED_BACKEND=bm25` pinned before the first retrieval import, and a deadline (`prompt_hook_deadline_ms`, default 800) bounds both paths — on timeout it prints nothing and exits 0. Hits are scored by **BM25 coverage** (raw score over the best any document could reach for the query, so it is in [0,1] and comparable across backends) and only hits at or above `prompt_hook_floor` are emitted as `hookSpecificOutput.additionalContext` (`Relevant memories:` + `- [stem] hook`, <= ~600 bytes). Modes (`prompt_hook_mode`): `off` = nothing runs or logs; `log` = every query + top-3 scores + would-inject appended to `<memory_dir>/.weights/hook_log.jsonl`, nothing injected; `inject` (default) = log AND inject. The log is permanent and size-rotated (`prompt_hook_log_max_mb`, 3 generations); `memsom hook-stats` summarises it. Short prompts (< 12 chars) and slash commands are skipped without a log line. Packaging: the repo root is a Claude Code plugin (`.claude-plugin/plugin.json` pointing `skills` at `claude/skills/`, `hooks/hooks.json` with Stop → `bridge-render` and UserPromptSubmit → `hook-prompt` in exec form, `.mcp.json` → `memsom-mcp`) and its own single-entry marketplace (`.claude-plugin/marketplace.json`, `source: "./"`); `wire-claude` wires the same prompt hook into `settings.json` for non-plugin installs, text-compare-upgrading a stale entry in place.

**The memory index (`bridge_import` → `digest` → `bridge_render`):**
- **Layout.** The memory dir is flat, with ONE subtree: `projects/`. A project is a directory `projects/<slug>/` holding its parent overview `project_<slug>.md` plus any subprojects `project_<slug>_<sub>.md`; a loose `projects/project_<x>.md` is a standalone project. `bridge_import.iter_memory_files` is the single walker (top level, `projects/`, `projects/*/` — depth 2, nothing deeper; the generated `MEMORY.md` and any `INDEX.md` are never imported) and every consumer in the package (`bridge_import`, `bridge.consolidate`, `interface.audit`) goes through it. Node identity stays the **basename** (`bridge_path`), so moving a file between the two levels is an edit of the same memory (forget state carried), not a new one — which is also why filenames must be unique across both levels: the walker raises `DuplicateMemoryStem` rather than let one file shadow the other. The importer stamps `memory_subdir` (`projects` or `projects/<slug>`) on files under the tree so the digest can link them relative to whichever index it renders.
- **First-run scaffold.** `bridge_import.scaffold_memory_dir` (called by `wire-claude` on the largest existing memory dir and by every `bridge-render`) creates — only if absent — `projects/`, an empty `projects/INDEX.md`, and `.weights/canonical.json` seeded with the forgetting defaults + panel params. It never overwrites: canonical.json stays owned by whatever weights layer the user runs; memsom only reads it afterwards. The managed CLAUDE.md block (protocol v2, `bridge.claude.CANONICAL`) documents the layout and both caps; `claude-sync` replaces any older block text in place, so existing installs pick it up on the next render.
- **Two caps, not one.** `MEMORY.md` must fit `memory_budget` bytes AND `memory_max_lines` lines (canonical.json `params`; fallbacks 16384 / 180). The byte cap alone never caught the consumer's ~200-line read limit silently truncating the file. The shed loop satisfies both; each shed entry in `.weights/shed.json` records `reason: "budget"` (bytes were over at drop time) or `"lines"`, and the manifest carries `lines` / `max_lines` next to `rendered_bytes` / `budget`.
- **Reserved live state.** Among droppable (non-pinned) entries, anything under `## Live state` or with `type: fact` is shed LAST — drop order is `(is_live_state, rs)`. A stale reference note losing its slot costs less than a live number going dark. Live state is also **exempt from tier** in `_select_hot`: `fact_*` files are channel `user`, so a cold tier (RS≈0 — nobody re-reads a number they can see in the index) used to drop them as `cold` before the shed-last ordering ever ran (2026-08-20). Now only the byte/line budget can shed a live-state entry; `pin:` still works as before. `memsom audit` accepts `type: fact`.
- **Projects split.** `project_*` stems are excluded from `MEMORY.md` entirely and rendered by `digest.render_projects_index` into `projects/INDEX.md` (written atomically by `bridge_render` after a successful main write; no byte cap). One group per `projects/<slug>/` dir: a `### <parent index line>` headline (or `### <slug> (no parent overview)` when `project_<slug>.md` is missing, so the gap is visible) followed by the subprojects as indented lines; loose files follow under `## Standalone`. Every line carries a ` [Parked]` / ` [Closed]` tag (Active untagged); status = the file's `status:`, else the parent's `status:` for a subproject, else the forget tier (hot / warm / cold). Order: Active first, then RS desc (groups sort by the parent's status/RS). Links are relative to `projects/` (`project_x.md`, `<slug>/project_x.md`, or `../project_x.md` for a legacy flat file). Membership does not require a `section:` — being a project memory is enough. `MEMORY.md` carries one synthetic pointer line under `## Personal projects`; the importer mirrors it back as a literal node on the next import, and the digest drops that copy so the line never duplicates and disappears on its own once no project memories exist.
- **The bridge keeps the retrieval index current.** `insert_node` (frozen core) writes a node and nothing else; postings/docstats/embeddings are built only by `retrieval.retrieve.index_node`. The bridge importer never called it, so bridge-imported memory was invisible to `memsom retrieve` until a manual rebuild (116 of 3065 nodes indexed on the live store; every query returned the same stray chunk). Now `bridge_import.sync_index` runs after each import transaction commits (index/deindex open their own `with conn:`, so they cannot run inside the importer's `BEGIN IMMEDIATE` — same rule as the stale cascade): every created node is `index_node`d, every superseded or swept node `deindex_node`d, counted in the stats as `indexed` / `deindexed`. `index_node` degrades to BM25-only when no embedding backend is reachable, so this is unconditional; `MEMDAG_BRIDGE_INDEX=0` restores the old write-only behaviour. `memsom reindex` remains the full rebuild. The `add` CLI path indexes best-effort the same way `ingest` does; `corroborate` mints agent-derived nodes, which `index_node` skips by design.
- **Audit: withdrawn is not orphan.** `memsom audit` reads the render's `.weights/shed.json` receipt: a file whose reason is `unsectioned` (or whose own frontmatter says `section: none` / `index: false`) is reported as INFO `withdrawn`, never as an orphan; `budget` / `lines` / `projects` reasons are explained absences and are skipped.
- **Explicit withdrawal.** The section-resolution chain is curated `MEMORY.md` line > the file's own `section:` frontmatter > the value already stamped on the live node (the fallback that keeps a budget-evicted memory filed). A file that says `section: none` (case-insensitive) or `index: false` overrides all three: its stored section is cleared on re-import and the digest excludes it (`reason: unsectioned`) — the one deliberate way to take a memory out of the index without deleting it.
- **Anti-creep (four shipped defaults; `forget.PANEL_PARAM_DEFAULTS`, `bridge_import.born_unindexed`, `digest.shed_section_budgets`, `bridge/consolidate.py`, `interface/index_stats.py`).** The Feedback section reached 141 pinned lines because every new `feedback_*` file was born with its own index line and nothing merged. (1) **Born unindexed**: a NEW feedback file (no stored node, or one already held) that resolves to Feedback with no `why_own_line:` and no curated `MEMORY.md` line is imported with its section cleared and `index_pending: needs_cluster` stamped on the node; the digest reports it as `needs_cluster`, the audit as INFO `needs-cluster`. Clusters (`feedback_cluster_*`) are exempt; already-indexed nodes are never retroactively unfiled; `feedback_born_unindexed=false` disables it. (2) **Per-section budget** (`section_budgets`, default `{"Feedback": 7168}`): before the global byte/line loop, a section over its cap sheds newest first (`COALESCE(forget_first_seen, created_at)` desc, ties RS asc), pinned or not — plain entries before `why_own_line:` ones, literals and clusters never — with reason `section_budget`. (3) **Merge proposer**: `consolidate-feedback` (BM25 over the store's own postings, restricted to live cluster nodes) and `consolidate-projects` (closed/cold subprojects → a row in the parent's `## Threads` table, `index: false` on the file, which `project_entries` honours) — dry-run by default, JSON receipt in `.weights/consolidate_proposals.json`, never deletes. (4) **Visible counts**: `write_live` returns `sections` (`digest.section_stats`), `bridge-render` prints `feedback: N lines / B bytes (budget X)`, `shed.json` and `audit --json` carry the per-section table, `index-stats` reads it on demand.
- **Duplicate stems self-heal.** The importer no longer raises on two files sharing a basename (additive sync — robocopy /E, rsync --update — leaves the pre-move copy behind on the other machine, which froze every render): `resolve_duplicates` picks the canonical copy (byte-identical → deepest path; differing → newest mtime) and `heal_duplicates` deletes the identical leftover or moves the older differing copy to `.weights/dup_quarantine/<stem>.<mtime_ns>.md`, counted as `dedup` / `quarantined`. `DuplicateMemoryStem` is raised only when that quarantine write fails.
- **Hook paths are absolute.** `wire_claude.resolve_exe` writes the running console script's absolute path (then the interpreter-sibling script, then `shutil.which`, then `"<python>" -m memsom.interface.cli`) and upgrades a bare `"memsom"` entry in place; a bare name only works when the venv is on the PATH Claude Code was launched with. The plugin's `hooks/hooks.json` must stay machine-agnostic, so it keeps the bare name and the README documents the `pipx`/symlink constraint.

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
- Enforcement is **action-time only**: Gate #3 (`check_capability`, via the broker
  and the native-tool hooks) gates what a session may *do*; `check_action` is an
  advisory node-integrity oracle, invoked manually, never auto-interposed; reads
  are transparent.
- History is immutable: tombstone (revoke) / redact (wipe payload, keep shape) / archive (compact) / quarantine (gate) — never an in-place mutation or hard delete by disuse.
- Integrity min-floor + confidentiality max-ceiling, both content-independent.
- Deterministic by default; the LLM is opt-in and firewalled.

## Honest boundaries

- Federation origin is honor-system (transport-authenticated in single-operator deployments); signed changesets are the documented next step if untrusted multi-party federation enters scope.
- The LLM firewall guarantees per-line provenance, not semantic faithfulness.
- Corroboration claim-extraction is deterministic (hashes/IPs/ports/semver/key=value), not prose-semantic.
- `retrieve_graph` boosts linked-relevant notes; it does not yet pull pure-context (zero-lexical-overlap) neighbors — deliberate, because `compose` force-includes every pool member.
