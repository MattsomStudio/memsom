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
| **CLI**               | **memdag_cli.py**        | **Unified 30-subcommand CLI; enhanced ask**                    | **test_memdag_cli.py**        | **10**|
| **MCP server**        | **memdag_mcp.py**        | **stdio JSON-RPC 2.0 MCP server; 10 tools; --selfcheck**       | **test_memdag_mcp.py**        | **9** |

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
