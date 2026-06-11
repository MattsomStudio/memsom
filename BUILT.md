# BUILT.md — changelog for the 2026-06-11 overnight build

All modules are stdlib-only Python 3.12, Windows-compatible. No pip deps anywhere.
The frozen demo #1 suite (TestEndToEndDemo, 18 tests) is green; the full sweep
adds 80+ tests across 13 modules + CLI + MCP.

## Module table

| Module                | File                     | Capability                                                     | Test file                     | Tests |
|-----------------------|--------------------------|----------------------------------------------------------------|-------------------------------|-------|
| schema helpers        | memdag_schema.py         | Idempotent ADD COLUMN / CREATE TABLE IF NOT EXISTS helpers     | test_memdag_schema.py         | 8     |
| recompute             | memdag_recompute.py      | Multi-hop integrity label recompute; elevation fixed-points    | test_memdag_recompute.py      | 10    |
| redact                | memdag_redact.py         | Payload destruction (content=''); rows/edges/dates survive     | test_memdag_redact.py         | 10    |
| quarantine            | memdag_quarantine.py     | Consolidation gate; manual quarantine; promote with gate check | test_memdag_quarantine.py     | 10    |
| confid                | memdag_confid.py         | Bell-LaPadula conf axis (MAX high-water-mark); classify        | test_memdag_confid.py         | 10    |
| trust                 | memdag_trust.py          | Integrity lattice (meet/join); audited manual elevation        | test_memdag_trust.py          | 10    |
| blame                 | memdag_blame.py          | Git-blame: trace any node to its root sources                  | test_memdag_blame.py          | 8     |
| federation            | memdag_federation.py     | Cross-machine sync; first-death-wins monotonic; UUID backfill  | test_memdag_federation.py     | 10    |
| relate                | memdag_relate.py         | Associative rel_edges; BFS neighborhood with integrity floor   | test_memdag_relate.py         | 10    |
| anticipatory          | memdag_anticipatory.py   | Surprise-gating (Jaccard novelty); query_log; prefetch         | test_memdag_anticipatory.py   | 10    |
| distill               | memdag_distill.py        | Provenance-filtered JSONL export; distill_plan runner stub     | test_memdag_distill.py        | 10    |
| heal                  | memdag_heal.py           | Invariant check (5 violation kinds); rebuild-derived           | test_memdag_heal.py           | 10    |
| llm                   | memdag_llm.py            | Opt-in Ollama LLM path; citation firewall; LlmUnavailable      | test_memdag_llm.py            | 10    |
| **CLI**               | **memdag_cli.py**        | **Unified 38-subcommand CLI; enhanced ask**                    | **test_memdag_cli.py**        | **10**|
| **MCP server**        | **memdag_mcp.py**        | **stdio JSON-RPC 2.0 MCP server; 12 tools; --selfcheck**       | **test_memdag_mcp.py**        | **9** |

## Schema additions (all additive, all defaulted — demo #1 byte-identical)

### New columns on `nodes`

| Column            | Type / Default                     | Added by          |
|-------------------|------------------------------------|-------------------|
| `redacted`        | INTEGER NOT NULL DEFAULT 0         | memdag_redact     |
| `redacted_at`     | TEXT                               | memdag_redact     |
| `redact_reason`   | TEXT                               | memdag_redact     |
| `status`          | TEXT NOT NULL DEFAULT 'live'       | memdag_quarantine |
| `quarantine_reason` | TEXT                             | memdag_quarantine |
| `quarantined_at`  | TEXT                               | memdag_quarantine |
| `conf_label`      | INTEGER NOT NULL DEFAULT 0         | memdag_confid     |
| `uuid`            | TEXT (nullable)                    | memdag_federation |
| `origin`          | TEXT (nullable)                    | memdag_federation |

### New tables

| Table           | Owner              | Purpose |
|-----------------|--------------------|---------|
| `rel_edges`     | memdag_relate      | Associative (relates-to) edges, separate from provenance edges |
| `query_log`     | memdag_anticipatory| Anticipatory coprocess: query history for novelty/prefetch |
| `elevations`    | memdag_trust       | Audit trail for every manual integrity label elevation |

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
| profile                | memdag_profile.py           | test_memdag_profile.py           | 13    |
| gate                   | memdag_gate.py              | test_memdag_gate.py              | 11    |
| corroborate            | memdag_corroborate.py       | test_memdag_corroborate.py       | 12    |

### New tables (all additive — no nodes-table columns touched)

| Table                | Owner               | Purpose |
|----------------------|---------------------|---------|
| `gate_log`           | memdag_gate         | One row per check_action() call: node, required, floor, decision, culprit, ts |
| `independence_roots` | memdag_corroborate  | Registered independence roots; unregistered sources earn zero credit |
| `claims`             | memdag_corroborate  | Structured (subject, predicate, value) triples with extractor version |
| `claim_assertions`   | memdag_corroborate  | Maps (claim_id, node_id) -> independence_root; PK prevents double-assertion |
| `corroborations`     | memdag_corroborate  | Maps (claim_id, lift_node_id) with k_used, roots_count, ts |

### Design deviations

- **`check-action` not `check`**: `check` is already registered by `memdag_heal`. Collision avoided; programmatic function is still `check_action` (no hyphen).
- **Lift node elevation fixed-point**: The lift node minted by `corroborate()` gets an `elevations` row marking it as a fixed point so `recompute_all` cannot claw label 1 back down to `min(parents)=0`.
- **Lift-drop is native revoke-cascade**: The lift node is a child of every asserting node. Revoking any asserting node cascades the tombstone to the lift automatically — no special teardown path.
- **MCP tool count 10 -> 12**: `profile` and `check_action` added; `SERVER_VERSION` bumped to `0.3.0`.

## Spine v1 (2026-06-11)

### New modules

| Module          | File                   | Capability                                                                                   | Test file                   | Tests |
|-----------------|------------------------|----------------------------------------------------------------------------------------------|-----------------------------|-------|
| ingest          | memdag_ingest.py       | Write path: channel-stamped adapters (file/dir/url/text), SHA-256 dedup, auto-chunking       | test_memdag_ingest.py       | 33    |
| retrieve        | memdag_retrieve.py     | Hybrid BM25 (stdlib) + optional Ollama-vector (RRF fusion); degrades silently to BM25-only   | test_memdag_retrieve.py     | 35    |
| compact         | memdag_compact.py      | Edge-preserving compaction: derive_node from episodes, archive (never delete), integrity gate| test_memdag_compact.py      | 22    |

### CLI wiring (memdag_cli.py)

- `migrate_all()` now runs `memdag_ingest.migrate`, `memdag_retrieve.migrate`, `memdag_compact.migrate` — guarantees `content_hash`, `postings`, and `archived` exist on every CLI path.
- `_build_pool()` adds `AND archived = 0` — DEFAULT 0 means no behaviour change until something is actually compacted; all existing CLI tests stay green.
- `ask` gains two opt-in flags: `--retrieve` (use ranked retrieval pool instead of all-live) and `--topk N` (default 8). Without `--retrieve`, ask is byte-identical to prior behaviour.
- New subcommands mounted: `ingest`, `ingest-dir`, `ingest-url`, `ingest-text`, `retrieve`, `reindex`, `compact`, `archived-list`.

### MCP wiring (memdag_mcp.py)

- Two new tools added: `retrieve` (ranked hits) and `ingest_text` (stamp + store).
- MCP tool count: 12 → **14**.
- `--selfcheck` exits 0.

### New schema columns (additive; all DEFAULT-safe)

| Column         | Type / Default              | Added by         |
|----------------|-----------------------------|------------------|
| `content_hash` | TEXT (nullable)             | memdag_ingest    |
| `archived`     | INTEGER NOT NULL DEFAULT 0  | memdag_compact   |
| `archived_at`  | TEXT                        | memdag_compact   |

### New tables

| Table       | Owner           | Purpose |
|-------------|-----------------|---------|
| `postings`  | memdag_retrieve | BM25 term→doc inverted index (term, node_id, tf) |
| `docstats`  | memdag_retrieve | Per-document token count for BM25 normalization |
| `embeddings`| memdag_retrieve | Optional Ollama float32 vectors (node_id, model, dim, vec BLOB) |

### New index

| Index                    | Table | Purpose |
|--------------------------|-------|---------|
| `idx_nodes_content_hash` | nodes | Fast dedup lookup by SHA-256 hash |

### Test counts (Spine v1 additions)

| Test file              | New tests | Notes |
|------------------------|-----------|-------|
| test_memdag_ingest.py  | 33        | Migration, dedup, chunking, file/dir/url adapters, CLI register, frozen-core compat |
| test_memdag_retrieve.py| 35        | BM25, vector (mocked), RRF, pool filters, CLI smoke |
| test_memdag_compact.py | 22        | Grouping, extractive summary, archiving, edges, integrity gate |
| test_memdag_cli.py     | +4        | Retrieve-flag tests (unchanged path, ranked pool, empty-pool exit-1, parse smoke) |
| test_memdag_mcp.py     | +2        | ingest_text stores node, retrieve returns hits; tool count 12→14 |

Full suite after Spine v1: **339 tests, 0 failures**.

## Regression guarantee

The frozen TestEndToEndDemo suite (`test_memdag.py`, 18 tests) runs against
`memdag.py` only and is byte-identical to the 2026-06-12 demo. It MUST stay
green. Command:

```powershell
python -W error::DeprecationWarning -m unittest discover -s C:\Users\you\memdag -p "test_memdag.py" -t C:\Users\you\memdag -v
```

Full sweep:

```powershell
python -W error::DeprecationWarning -m unittest discover -s C:\Users\you\memdag -p "test_memdag*.py" -t C:\Users\you\memdag -v
```

> Keep `memdag.db` out of Syncthing-synced trees — same rule as `sessions.db`.

---

## Operator-class hardening (2026-06-11)

Closes the remaining operator-class findings from the adversarial audit
(`_build/AUDIT.md`). `memdag.py` and `test_memdag.py` stayed frozen; F-14 was
fixed at the caller layer. Suite: **372 passing, 0 failures** (was 357).

- **Bypass-2G/2H — archived-parent conf laundering (HIGH).**
  `memdag_confid.classify()` now refuses to LOWER `conf_label` on an *archived*
  node (frozen post-compaction), and `sources_for_clearance()` excludes
  `archived=1` rows. A SECRET-derived compact summary can no longer be dragged
  to PUBLIC by declassifying its archived sources + recompute. *Chosen over a
  stored high-water floor because the floor would break the legitimate
  tombstone-drop semantics pinned by `TestTombstonedParentExcluded`.*
  Regression: `test_memdag_confid.TestArchivedConfLaunderingBlocked`.

- **F-08 — stepwise elevation bypass (HIGH).** `memdag_trust.elevate()` keys the
  external→endorsed force gate on the node's IMMUTABLE `channel` (rank 0) rather
  than its current label, so `0→1→2→3` can no longer reach ENDORSED without
  `--force`; the reaching-ENDORSED step records `forced=1`.
  Regression: `test_memdag_trust.TestStepwiseElevationBlocked`.

- **F-15 — stale retrieval index after redaction (MEDIUM).** Added
  `memdag_retrieve.deindex_node()` (postings + docstats + embeddings), called
  from the `memdag_redact` cascade and from `index_node()` for dead/redacted
  nodes. `retrieve()` now ALWAYS excludes tombstoned + redacted + archived
  regardless of the `exclude_*` flags (fail-safe — flags may widen, never leak
  liveness). Regression: `test_memdag_retrieve.test_redaction_deindexes_and_never_surfaces`.

- **F-14 — channel/label mismatch at the API (MEDIUM).** Frozen `insert_node()`
  still accepts a mismatched `label`, so enforcement lives at the caller layer:
  `memdag_ingest.authoritative_label(channel)` pins a source node's label to
  `RANK[channel]`, used by the CLI `add` path and `ingest_text`. No entry point
  can stamp a mismatched label. Regression: `test_memdag_ingest.TestChannelLabelLock`,
  `test_memdag_cli.test_add_stamps_channel_label`.

- **F-16 — explain shows empty snippet for redacted node (LOW).** The CLI
  `explain` bridge now prints the `memdag_redact.describe()` `[REDACTED <date>:
  <reason>]` marker for a redacted node. Regression:
  `test_memdag_cli.test_explain_redacted_node_shows_marker`.

- **F-13 — entry points accept caller-declared `endorsed` (MEDIUM).** This is BY
  DESIGN for a single-user tool — the operator is the trust authority — so the
  default is permissive. Added an OPTIONAL guard: env var
  `MEMDAG_CHANNEL_CEILING` (channel name or 0-3). When set, the CLI/MCP/ingest
  stamping entry points (`add`, `ingest`, `ingest-text`, MCP `ingest_text`)
  refuse any channel whose rank exceeds the ceiling; `ingest-url` is already
  hard-locked to `external` and is always under any ceiling. Enforced once in
  `memdag_ingest.enforce_channel_ceiling()` (shared by all paths).
  Regression: `test_memdag_ingest.TestChannelCeiling`.

---

## Anticipatory (Phase 2) — retrieval-backed coprocess (2026-06-11)

Turns the `memdag_anticipatory` stub (Jaccard novelty, observe-only prefetch)
into a real coprocess on top of the spine. `memdag.py` / `test_memdag.py`
stayed frozen; default `ask` (no `--anticipate`) is byte-identical. Suite:
**394 passing, 0 failures** (was 372).

**THE security property (why taint precedes anticipatory):** the coprocess
reads/learns/prefetches ONLY from untainted memory. Every corpus it touches
goes through `_untainted_clauses()` — mirroring the spine's pool filters —
which excludes `tombstoned=1`, `redacted=1`, `status='quarantined'`,
`archived=1`, above-clearance (`conf_label`), plus `label > 0` on the derived
corpus so EXTERNAL-tainted derivations (the ones the consolidation gate
quarantines) are refused directly as defense in depth. It never elevates
trust: every mint goes through the frozen `memdag.derive_node`
(label = min(parents), honest provenance edges), and warm answers are
RE-validated against the same filter at serve time. Proven by
`test_memdag_anticipatory.TestCoprocessNeverTouchesTainted` (six tests:
poisoned sources out of the pool; poisoned derived answers vanish from the
surprise corpus and are never cited; above-clearance answers invisible below
clearance; external-tainted derivations never cited, cached, or served warm —
even via a forced cache row; prefetch never composes from a quarantined
source; recombination never lists a tainted prior).

- **Real surprise + surprise-gated writes** (`surprise`, `rank_similar`,
  `surprise_gated_write`). Novelty against the untainted corpus via
  BM25-IDF-weighted TF cosine over `memdag_retrieve.tokenize` stems (stdlib,
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
  real `memdag_retrieve.retrieve()` (untainted by construction; falls back to
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
