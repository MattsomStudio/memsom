# BUILT.md — changelog for the 2026-06-11 overnight build

All modules are stdlib-only Python 3.12, Windows-compatible. No pip deps anywhere.
The frozen demo #1 suite (TestEndToEndDemo, 18 tests) is green; the full sweep
adds 80+ tests across 13 modules + CLI + MCP.

## Module table

| Module                | File                     | Capability                                                     | Test file                     | Tests |
|-----------------------|--------------------------|----------------------------------------------------------------|-------------------------------|-------|
| schema helpers        | memsom/storage/schema.py         | Idempotent ADD COLUMN / CREATE TABLE IF NOT EXISTS helpers     | test_memsom_schema.py         | 8     |
| recompute             | memsom/retrieval/recompute.py      | Multi-hop integrity label recompute; elevation fixed-points    | test_memsom_recompute.py      | 10    |
| redact                | memsom/integrity/redact.py         | Payload destruction (content=''); rows/edges/dates survive     | test_memsom_redact.py         | 10    |
| quarantine            | memsom/integrity/quarantine.py     | Consolidation gate; manual quarantine; promote with gate check | test_memsom_quarantine.py     | 10    |
| confid                | memsom/integrity/confid.py         | Bell-LaPadula conf axis (MAX high-water-mark); classify        | test_memsom_confid.py         | 10    |
| trust                 | memsom/integrity/trust.py          | Integrity lattice (meet/join); audited manual elevation        | test_memsom_trust.py          | 10    |
| blame                 | memsom/interface/blame.py          | Git-blame: trace any node to its root sources                  | test_memsom_blame.py          | 8     |
| federation            | memsom/federation/federation.py     | Cross-machine sync; first-death-wins monotonic; UUID backfill  | test_memsom_federation.py     | 10    |
| relate                | memsom/retrieval/relate.py         | Associative rel_edges; BFS neighborhood with integrity floor   | test_memsom_relate.py         | 10    |
| anticipatory          | memsom/lifecycle/anticipatory.py   | Surprise-gating (Jaccard novelty); query_log; prefetch         | test_memsom_anticipatory.py   | 10    |
| distill               | memsom/distill/distill.py        | Provenance-filtered JSONL export; distill_plan runner stub     | test_memsom_distill.py        | 10    |
| heal                  | memsom/lifecycle/heal.py           | Invariant check (5 violation kinds); rebuild-derived           | test_memsom_heal.py           | 10    |
| llm                   | memsom/distill/llm.py            | Opt-in Ollama LLM path; citation firewall; LlmUnavailable      | test_memsom_llm.py            | 10    |
| **CLI**               | **memsom/interface/cli.py**        | **Unified 38-subcommand CLI; enhanced ask**                    | **test_memsom_cli.py**        | **10**|
| **MCP server**        | **memsom/interface/mcp.py**        | **stdio JSON-RPC 2.0 MCP server; 12 tools; --selfcheck**       | **test_memsom_mcp.py**        | **9** |

## Schema additions (all additive, all defaulted — demo #1 byte-identical)

### New columns on `nodes`

| Column            | Type / Default                     | Added by          |
|-------------------|------------------------------------|-------------------|
| `redacted`        | INTEGER NOT NULL DEFAULT 0         | memsom_redact     |
| `redacted_at`     | TEXT                               | memsom_redact     |
| `redact_reason`   | TEXT                               | memsom_redact     |
| `status`          | TEXT NOT NULL DEFAULT 'live'       | memsom_quarantine |
| `quarantine_reason` | TEXT                             | memsom_quarantine |
| `quarantined_at`  | TEXT                               | memsom_quarantine |
| `conf_label`      | INTEGER NOT NULL DEFAULT 0         | memsom_confid     |
| `uuid`            | TEXT (nullable)                    | memsom_federation |
| `origin`          | TEXT (nullable)                    | memsom_federation |

### New tables

| Table           | Owner              | Purpose |
|-----------------|--------------------|---------|
| `rel_edges`     | memsom_relate      | Associative (relates-to) edges, separate from provenance edges |
| `query_log`     | memsom_anticipatory| Anticipatory coprocess: query history for novelty/prefetch |
| `elevations`    | memsom_trust       | Audit trail for every manual integrity label elevation |

### New index

| Index            | Table   | Purpose |
|------------------|---------|---------|
| `idx_nodes_uuid` | nodes   | Unique federation identity (NULLs OK under SQLite UNIQUE) |

## Design deviations from the spec

- **recompute fixed-point vs raw transitive CTE**: The spec mentioned a raw
  transitive CTE for recompute. The implementation uses an iterative post-order
  DFS with a memo dict and in-progress set. This handles diamond graphs, cycles,
  and 1500-deep chains without hitting the Python recursion limit — something a
  CTE alone cannot guarantee for arbitrary depths.

- **`elevated_by` vs reserved `by`**: The elevations table uses `elevated_by` as
  the column name (not `by`) because `by` is a reserved word in some SQLite
  contexts and would require quoting everywhere. The Python API exposes it as
  `'by'` in the result dict (mapped at fetch time).

- **`content=''` vs NULL for redaction**: `nodes.content` is NOT NULL in the
  frozen schema and cannot be relaxed additively. Redaction stores `content=''`
  plus `redacted=1`. The redacted flag is the authoritative marker; the empty
  string prevents any NOT NULL violation and means a redacted node composes
  nothing even if the redacted column is ignored.

- **distill never executes ollama**: The spec mentioned "may attempt ollama
  create". The implementation detects ollama via `shutil.which` but never spawns
  it. The fine-tune is documented as the one manual GPU step. This is intentional.

- **Single `LlmUnavailable`**: Both Ollama-unreachable and citation-firewall
  failures raise `LlmUnavailable`. One exception type = one fallback path.
  The CLI catches it, prints the warning, and falls back to deterministic compose.

## Honest boundaries

- **GPU fine-tune**: `export-training` generates the JSONL; `distill-plan`
  writes the runner stub and config. The actual QLoRA fine-tune is manual.
- **Dangling edges**: `check` and `rebuild-derived` REPORT dangling edges but
  never delete them. Rows and edges always survive (invariant #3).
- **LLM is opt-in**: `--llm` only. The default `ask` path is 100% deterministic
  and has zero network calls.
- **Redaction propagation**: redaction events propagate as priority records and
  scrub stale changesets on any machine that holds the record; a cold machine that
  never received the record still gets content from a stale changeset —
  deletion-vs-immutability limit, encoded as a known-limit test
  (`test_known_limit_cold_machine_stale_changeset`).

## Biba-fatigue v1 (2026-06-11)

### New modules

| Module                 | File                        | Test file                        | Tests |
|------------------------|-----------------------------|----------------------------------|-------|
| profile                | memsom/interface/profile.py           | test_memsom_profile.py           | 13    |
| gate                   | memsom/integrity/gate.py              | test_memsom_gate.py              | 11    |
| corroborate            | memsom/integrity/corroborate.py       | test_memsom_corroborate.py       | 12    |

### New tables (all additive — no nodes-table columns touched)

| Table                | Owner               | Purpose |
|----------------------|---------------------|---------|
| `gate_log`           | memsom_gate         | One row per check_action() call: node, required, floor, decision, culprit, ts |
| `independence_roots` | memsom_corroborate  | Registered independence roots; unregistered sources earn zero credit |
| `claims`             | memsom_corroborate  | Structured (subject, predicate, value) triples with extractor version |
| `claim_assertions`   | memsom_corroborate  | Maps (claim_id, node_id) -> independence_root; PK prevents double-assertion |
| `corroborations`     | memsom_corroborate  | Maps (claim_id, lift_node_id) with k_used, roots_count, ts |

### Design deviations

- **`check-action` not `check`**: `check` is already registered by `memsom_heal`. Collision avoided; programmatic function is still `check_action` (no hyphen).
- **Lift node elevation fixed-point**: The lift node minted by `corroborate()` gets an `elevations` row marking it as a fixed point so `recompute_all` cannot claw label 1 back down to `min(parents)=0`.
- **Lift-drop is native revoke-cascade**: The lift node is a child of every asserting node. Revoking any asserting node cascades the tombstone to the lift automatically — no special teardown path.
- **MCP tool count 10 -> 12**: `profile` and `check_action` added; `SERVER_VERSION` bumped to `0.3.0`.

## Spine v1 (2026-06-11)

### New modules

| Module          | File                   | Capability                                                                                   | Test file                   | Tests |
|-----------------|------------------------|----------------------------------------------------------------------------------------------|-----------------------------|-------|
| ingest          | memsom/interface/ingest.py       | Write path: channel-stamped adapters (file/dir/url/text), SHA-256 dedup, auto-chunking       | test_memsom_ingest.py       | 33    |
| retrieve        | memsom/retrieval/retrieve.py     | Hybrid BM25 (stdlib) + optional Ollama-vector (RRF fusion); degrades silently to BM25-only   | test_memsom_retrieve.py     | 35    |
| compact         | memsom/lifecycle/compact.py      | Edge-preserving compaction: derive_node from episodes, archive (never delete), integrity gate| test_memsom_compact.py      | 22    |

### CLI wiring (memsom/interface/cli.py)

- `migrate_all()` now runs `memsom_ingest.migrate`, `memsom_retrieve.migrate`, `memsom_compact.migrate` — guarantees `content_hash`, `postings`, and `archived` exist on every CLI path.
- `_build_pool()` adds `AND archived = 0` — DEFAULT 0 means no behaviour change until something is actually compacted; all existing CLI tests stay green.
- `ask` gains two opt-in flags: `--retrieve` (use ranked retrieval pool instead of all-live) and `--topk N` (default 8). Without `--retrieve`, ask is byte-identical to prior behaviour.
- New subcommands mounted: `ingest`, `ingest-dir`, `ingest-url`, `ingest-text`, `retrieve`, `reindex`, `compact`, `archived-list`.

### MCP wiring (memsom/interface/mcp.py)

- Two new tools added: `retrieve` (ranked hits) and `ingest_text` (stamp + store).
- MCP tool count: 12 → **14**.
- `--selfcheck` exits 0.

### New schema columns (additive; all DEFAULT-safe)

| Column         | Type / Default              | Added by         |
|----------------|-----------------------------|------------------|
| `content_hash` | TEXT (nullable)             | memsom_ingest    |
| `archived`     | INTEGER NOT NULL DEFAULT 0  | memsom_compact   |
| `archived_at`  | TEXT                        | memsom_compact   |

### New tables

| Table       | Owner           | Purpose |
|-------------|-----------------|---------|
| `postings`  | memsom_retrieve | BM25 term→doc inverted index (term, node_id, tf) |
| `docstats`  | memsom_retrieve | Per-document token count for BM25 normalization |
| `embeddings`| memsom_retrieve | Optional Ollama float32 vectors (node_id, model, dim, vec BLOB) |

### New index

| Index                    | Table | Purpose |
|--------------------------|-------|---------|
| `idx_nodes_content_hash` | nodes | Fast dedup lookup by SHA-256 hash |

### Test counts (Spine v1 additions)

| Test file              | New tests | Notes |
|------------------------|-----------|-------|
| test_memsom_ingest.py  | 33        | Migration, dedup, chunking, file/dir/url adapters, CLI register, frozen-core compat |
| test_memsom_retrieve.py| 35        | BM25, vector (mocked), RRF, pool filters, CLI smoke |
| test_memsom_compact.py | 22        | Grouping, extractive summary, archiving, edges, integrity gate |
| test_memsom_cli.py     | +4        | Retrieve-flag tests (unchanged path, ranked pool, empty-pool exit-1, parse smoke) |
| test_memsom_mcp.py     | +2        | ingest_text stores node, retrieve returns hits; tool count 12→14 |

Full suite after Spine v1: **339 tests, 0 failures**.

## Regression guarantee

The frozen TestEndToEndDemo suite (`test_memsom.py`, 18 tests) runs against
`memsom/__init__.py` only. It was byte-identical to the 2026-06-12 demo through the
memdag era; the byte-identity anchor was **re-baselined at the 2026-07-01
memdag→memsom rename** (the file was renamed and its `import` updated, nothing
else). From that commit forward `memsom/__init__.py`/`test_memsom.py` stay byte-identical.
It MUST stay green. Command:

```powershell
python -W error::DeprecationWarning -m unittest discover -s <repo> -p "test_memsom.py" -t <repo> -v
```

Full sweep:

```powershell
python -W error::DeprecationWarning -m unittest discover -s <repo> -p "test_memsom*.py" -t <repo> -v
```

> Keep `memdag.db` out of any synced or backup trees so private memories are not replicated.

---

## Operator-class hardening (2026-06-11)

Closes the remaining operator-class findings from the adversarial audit
(`_build/AUDIT.md`). `memsom/__init__.py` and `test_memsom.py` stayed frozen; F-14 was
fixed at the caller layer. Suite: **372 passing, 0 failures** (was 357).

- **Bypass-2G/2H — archived-parent conf laundering (HIGH).**
  `memsom_confid.classify()` now refuses to LOWER `conf_label` on an *archived*
  node (frozen post-compaction), and `sources_for_clearance()` excludes
  `archived=1` rows. A SECRET-derived compact summary can no longer be dragged
  to PUBLIC by declassifying its archived sources + recompute. *Chosen over a
  stored high-water floor because the floor would break the legitimate
  tombstone-drop semantics pinned by `TestTombstonedParentExcluded`.*
  Regression: `test_memsom_confid.TestArchivedConfLaunderingBlocked`.

- **F-08 — stepwise elevation bypass (HIGH).** `memsom_trust.elevate()` keys the
  external→endorsed force gate on the node's IMMUTABLE `channel` (rank 0) rather
  than its current label, so `0→1→2→3` can no longer reach ENDORSED without
  `--force`; the reaching-ENDORSED step records `forced=1`.
  Regression: `test_memsom_trust.TestStepwiseElevationBlocked`.

- **F-15 — stale retrieval index after redaction (MEDIUM).** Added
  `memsom_retrieve.deindex_node()` (postings + docstats + embeddings), called
  from the `memsom_redact` cascade and from `index_node()` for dead/redacted
  nodes. `retrieve()` now ALWAYS excludes tombstoned + redacted + archived
  regardless of the `exclude_*` flags (fail-safe — flags may widen, never leak
  liveness). Regression: `test_memsom_retrieve.test_redaction_deindexes_and_never_surfaces`.

- **F-14 — channel/label mismatch at the API (MEDIUM).** Frozen `insert_node()`
  still accepts a mismatched `label`, so enforcement lives at the caller layer:
  `memsom_ingest.authoritative_label(channel)` pins a source node's label to
  `RANK[channel]`, used by the CLI `add` path and `ingest_text`. No entry point
  can stamp a mismatched label. Regression: `test_memsom_ingest.TestChannelLabelLock`,
  `test_memsom_cli.test_add_stamps_channel_label`.

- **F-16 — explain shows empty snippet for redacted node (LOW).** The CLI
  `explain` bridge now prints the `memsom_redact.describe()` `[REDACTED <date>:
  <reason>]` marker for a redacted node. Regression:
  `test_memsom_cli.test_explain_redacted_node_shows_marker`.

- **F-13 — entry points accept caller-declared `endorsed` (MEDIUM).** This is BY
  DESIGN for a single-user tool — the operator is the trust authority — so the
  default is permissive. Added an OPTIONAL guard: env var
  `MEMDAG_CHANNEL_CEILING` (channel name or 0-3). When set, the CLI/MCP/ingest
  stamping entry points (`add`, `ingest`, `ingest-text`, MCP `ingest_text`)
  refuse any channel whose rank exceeds the ceiling; `ingest-url` is already
  hard-locked to `external` and is always under any ceiling. Enforced once in
  `memsom_ingest.enforce_channel_ceiling()` (shared by all paths).
  Regression: `test_memsom_ingest.TestChannelCeiling`.

---

## Anticipatory (Phase 2) — retrieval-backed coprocess (2026-06-11)

Turns the `memsom_anticipatory` stub (Jaccard novelty, observe-only prefetch)
into a real coprocess on top of the spine. `memsom/__init__.py` / `test_memsom.py`
stayed frozen; default `ask` (no `--anticipate`) is byte-identical. Suite:
**394 passing, 0 failures** (was 372).

**THE security property (why taint precedes anticipatory):** the coprocess
reads/learns/prefetches ONLY from untainted memory. Every corpus it touches
goes through `_untainted_clauses()` — mirroring the spine's pool filters —
which excludes `tombstoned=1`, `redacted=1`, `status='quarantined'`,
`archived=1`, above-clearance (`conf_label`), plus `label > 0` on the derived
corpus so EXTERNAL-tainted derivations (the ones the consolidation gate
quarantines) are refused directly as defense in depth. It never elevates
trust: every mint goes through the frozen `memsom.derive_node`
(label = min(parents), honest provenance edges), and warm answers are
RE-validated against the same filter at serve time. Proven by
`test_memsom_anticipatory.TestCoprocessNeverTouchesTainted` (six tests:
poisoned sources out of the pool; poisoned derived answers vanish from the
surprise corpus and are never cited; above-clearance answers invisible below
clearance; external-tainted derivations never cited, cached, or served warm —
even via a forced cache row; prefetch never composes from a quarantined
source; recombination never lists a tainted prior).

- **Real surprise + surprise-gated writes** (`surprise`, `rank_similar`,
  `surprise_gated_write`). Novelty against the untainted corpus via
  BM25-IDF-weighted TF cosine over `memsom_retrieve.tokenize` stems (stdlib,
  deterministic), refined by Ollama vector cosine on the top-k lexical
  candidates when reachable (per-node sim = max(lexical, vector); Ollama down
  -> lexical-only, never a crash). `surprise_gated_write` is SEMANTIC dedup on
  the write path — a reworded question whose composition differs in bytes
  (different content hash) still CITES the existing node instead of minting a
  duplicate (`TestSemanticDedupBeyondExactHash`). Old API preserved:
  `surprise_gated` delegates; legacy Jaccard `novelty()` kept. CLI: `ask
  --anticipate` upgraded in place (also clearance-aware now — the derived
  comparison corpus previously ignored clearance).

- **Real prefetch (warm cache)** (`prefetch`, `serve_warm`; additive
  `prefetch_cache` table: query UNIQUE, answer_node, created_at, hits,
  last_served). Top-k queries by `query_log` frequency+recency; pool built by
  real `memsom_retrieve.retrieve()` (untainted by construction; falls back to
  the full untainted pool when the BM25 index is empty); answers minted/cited
  via `surprise_gated_write` and cached ONLY if the answer node itself passes
  the untainted-derived filter. `ask --anticipate` serves an exact-match warm
  hit ("served WARM from prefetch cache", with prefetch timestamp + hits) and
  logs the hit to `query_log`. Stale/poisoned cache rows are dropped at serve
  time, never served.

- **Episode-recombination warning** (`novel_recombination(parent_ids)`).
  Deterministic over the edges history: a prior derivation counts only if its
  parent set EQUALS the probe set (subset/superset are different
  combinations) and the prior node is itself untainted. `ask --anticipate`
  prints "new inference: this combination of sources has not been seen
  before." on a first-time parent-set, or lists the precedent node ids.

- **Observe + status surface.** `observe` unchanged; new `status()` +
  `anticipate-status` CLI showing query-log totals, top queries, and every
  cache row with its serve-time validity (stale rows flagged "STALE (will not
  serve)"). `prefetch` CLI gained `--clearance` and a cache summary line.

- **Scoped down, deliberately:** warm hits are exact-string query matches (no
  fuzzy matching — deterministic, no false serves); derived-node embeddings
  are computed on the fly for the top-k candidates rather than persisted in
  the `embeddings` table (keeps the source-only index invariant); surprise's
  vector arm is uncalibrated max(lex, vec) — threshold tuning on real corpora
  is future work.

## Phase 3 — weights/distillation pipeline (2026-06-11, `_weights\`)

**Thesis tested:** does a reflex-distilled adapter add anything over RAG-ing
the DAG?  Per the 2026-06-03 pivot, NO knowledge was distilled — only the
house response SCHEMA (Verdict / Evidence with `[mem:id|channel]` cites /
Integrity floor / Next move, plus cite-or-refuse).  CLS framing: slow
weights = schema, fast DAG = episodes.

- **`memsom/lifecycle/reflex.py`** (+10 tests, suite now 404) — `export-reflex` CLI:
  reflex-shaped chat pairs from UNTAINTED CONSOLIDATED memory only.
  Eligibility = the full distill gate (alive, unredacted, unquarantined,
  label floor, no live external ancestor — same CTE) AND consolidation
  signature (all immediate parents archived by compact). `assert_clean()`
  enforces the poison gate at export; 4 seeded poison cases (external-tainted
  / tombstoned / quarantined / redacted) verifiably never reach the JSONL.
- **Pipeline** (`_weights\`): `build_dataset.py` (own DB, 63 train + 10
  held-out) → `train_reflex_planb.py` (QLoRA r=16, 127 s on the 5070) →
  `merge_streaming.py` → `gguf_export.ps1` → ollama `memsom-reflex` (q4_k_m)
  + `memsom-reflex-q8`.
- **EOS saga closed:** root cause of the bug that survived 3 nmap retrains is
  in `tok_surgery.py` — Foundation-Sec-8B is a continued-PRETRAIN base whose
  chat tokens (`<|eot_id|>`, headers, pad) are UNTRAINED rows (embed norms
  ~0.02 vs 0.67; pad row all-zero). Surgery (copy `<|end_of_text|>` rows +
  mean-init) + LoRA on lm_head/embed_tokens fixed it: adapter stops on
  `<|eot_id|>` every probe.
- **⚠ Env regression (NOT memsom):** unsloth/triton training is broken
  box-wide since ~06-03 (`KeyError: 'cubin'` in rms_layernorm backward —
  repros on the KNOWN-GOOD `train_nmap.py`; suspect the GPU driver
  reinstall).  Plan-B trainer (plain transformers+peft+bnb, no triton) is the
  workaround and lives in `_weights\`.
- **Eval verdict** (10 held-out probes, strict/lenient, `eval_results.json`):
  adapter 1.00/1.00 · base+prompt 0.40 · qwen3-30b+prompt+RAG 0.80 strict /
  1.00 lenient on the five surface metrics (cite format + refusal wording
  were surface-only misses) — but 0/10 on the added integrity-value
  substance check: it maps the channel into the Integrity line instead of
  the integrity NAME (adapter 10/10). Honest call: **RAG+prompt on the strong base still
  carries knowledge and nearly all schema; the adapter buys deterministic
  format compliance + the integrity-vocabulary reflex on an 8B with no
  prompt overhead.**  The security contribution (only untainted consolidated
  memory can bake) holds regardless.
- **Known limitation:** GGUF/quant artifacts keep schema+cites+stop but the
  Verdict line degrades under quantization (63-example overfit adapter);
  fp16 merged + PEFT runtime verified perfect (`test_merged.py`).

## Packaging + Ollama VRAM hygiene (2026-06-11)

`memsom/__init__.py` / `test_memsom.py` stayed frozen. Suite: **416 passing, 0
failures** (was 404; +12 in `test_memsom_keepalive.py`).

- **pip-installable, flat layout kept** (`pyproject.toml`, hatchling). The
  wheel ships all 23 `memsom*.py` files as top-level modules (allowlist
  `only-include` — the hatchling equivalent of setuptools `py-modules`), so
  the frozen `import memsom` contract is untouched. `external_fallback.txt`
  is force-included at wheel root so `memsom seed --offline` works from
  site-packages (`memsom.FALLBACK` resolves beside `memsom/__init__.py`). Tests,
  `_build/`, `_weights/`, `*.db`, demo scripts are excluded from sdist and
  wheel by construction. Console entry point: `memsom = memsom_cli:main`.
  Verified in a throwaway venv: editable install, built-wheel install,
  `memsom --help` / `seed --offline` / `ask` against a temp `MEMDAG_DB`.
- **CI**: `.github/workflows/tests.yml` — Python 3.12 on ubuntu + windows,
  stdlib-only (no pip installs), full `unittest discover` sweep plus the
  frozen-core gate as a separate step. README gained the badge + an
  `## Install` section.
- **Ollama keep_alive knob (opt-in VRAM hygiene).** Every memsom call to the
  Ollama API — `memsom_llm.llm_compose` (/api/generate),
  `memsom_compact._llm_summarize` (/api/generate),
  `memsom_retrieve._call_ollama_embed` (/api/embeddings) — routes its body
  through ONE shared helper, `memsom_llm._with_keep_alive()`. **Shipped
  default: keep_alive is OMITTED** — memsom defers to Ollama's own native
  warm-keep, which is what downstream users want. Setting
  `MEMDAG_OLLAMA_KEEP_ALIVE=0` makes the model unload from VRAM immediately
  after every call (Matt's 12GB-card config, so memsom never evicts his
  daily-driver model); any duration string (e.g. `10m`) holds it warm longer.
  `memsom/__init__.py`'s only urlopen is the seed-article fetch (not Ollama) — frozen
  file untouched. Graceful Ollama-down fallbacks unchanged (regression-tested).
- **Note for Matt's machines:** set `MEMDAG_OLLAMA_KEEP_ALIVE=0` as a user env
  var on the PC and Mac to get unload-after-use locally without baking it into
  the shipped default. With that set + Ollama running, the unmocked embed paths
  load+unload `nomic-embed-text` per call, so the local sweep is slower (~95s
  vs ~15s) — that's the hygiene working, not a bug. Unset (the default), the
  local sweep is fast again; CI is unaffected either way (no Ollama → instant
  connection-refused fallback).

## `verify-stale` wired into the shipped CLI + MCP surface (2026-07-06)

`memsom/__init__.py` / `test_memsom.py` stayed frozen. Suite: **885 passing, 0
failures** (+5 net new — 4 in `test_memsom_verify_stale.py`, +1 net in
`test_memsom_mcp.py`).

- **No new core logic, no new schema — this is a wiring fix.**
  `memsom/integrity/verify_stale.py` (verification-age staleness over `memory:`-sourced
  notes, built and tested long ago — 14 tests) has had a working
  `register(sub)` mounting CLI verb `verify-stale` the whole time, but
  `memsom/interface/cli.py` never imported or registered the module. It was only
  reachable standalone or silently inside `memsom/bridge/bridge_render.py`'s
  internal pipeline (wrapped in a bare `try/except`, failures never
  surfaced). Closed the gap: `memsom/interface/cli.py` now imports and registers it
  (one line, same pattern as every other feature module); no migration
  needed — it operates entirely on columns `memsom/integrity/stale.py` already owns.
- **MCP tool added:** `verify_stale` (underscored per the `check_action` /
  `ingest_text` convention for multi-word tool names), dispatching to
  `verify-stale [--apply]`. Tool count 16 → 17.
- **Naming note:** deliberately NOT named `consolidate` — that CLI
  subcommand/MCP tool already exists (`memsom_quarantine.consolidate()`, the
  taint/quarantine gate) and means something unrelated. Skill folder name
  matches the CLI verb 1:1 (`claude/skills/verify-stale/`), same convention
  as the `audit` skill.
- **New skill:** `claude/skills/verify-stale/SKILL.md`, alongside the
  existing `audit`/`memdash`/`recall`/`saveall` productized skills.
- **Test counts:**

  | Test file                      | New tests | Notes |
  |---------------------------------|-----------|-------|
  | tests/test_memsom_verify_stale.py | 4       | `TestCLI`: `register()` mounts `verify-stale --apply`, `--apply` defaults False, dry-run makes no writes, `--apply` marks |
  | tests/test_memsom_mcp.py        | net +1    | tool-count assertion bumped 16→17; new `test_tools_call_verify_stale_apply_marks_and_dry_run_does_not` (real DB-effect check, not just no-crash) |

- **Scope, deliberately not built:** no LLM narration layer
  (`memsom/distill/llm.py` is a well-isolated, proven opt-in pattern for this if
  wanted later, but the deterministic sweep alone satisfies the gap — no new
  LLM dependency for v1); no expansion of the scan beyond `memory:`-sourced
  bridge notes to the whole DAG store (a natural phase-2, not needed to
  close the wiring gap); `memsom/bridge/bridge_render.py`'s existing silent
  try/except around its internal `verify-stale` call is untouched (separate,
  deliberate fail-safe design).
