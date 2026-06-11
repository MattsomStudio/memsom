# memdag (working name)

> **One-liner — Matt's words, pending.** Do not ship the pitch without it.

A derivation-DAG memory store for AI agents. Every memory is a node; edges mean
**came-from** (provenance), not relates-to. Trust is stamped by **channel** at
write time (`endorsed > user > agent-derived > external`), derived answers carry
`min(parent labels)` (Biba low-water-mark), and revoking a source tombstones
everything derived from it — while history stays fully explainable.

```
APPS  taint(quarantine) · blame · redact · trust/elevate · relate(GraphRAG-safe)
      · anticipatory · distill · heal · federation · conf-labels · MCP
SUBSTRATE   the derivation DAG          <- this repo
STORAGE     SQLite                      <- commodity, swappable
```

Two entry points:
- `memdag.py` — **frozen core slice** (do not touch; the original explain/revoke vertical slice, kept byte-identical as a regression anchor)
- `memdag_cli.py` — **full surface** (all 13 feature modules, 30 subcommands)

## Quickstart (frozen core)

Python 3.12+, stdlib only. No install.

```powershell
python memdag.py seed            # stamp 3 sources: user fact, endorsed vault note, external article
python memdag.py ask "How should I configure Nebula?"
python memdag.py explain 4      # provenance tree with channels, labels, dates
python memdag.py revoke 3 --reason "untrusted source retracted"        # dry run: blast radius
python memdag.py revoke 3 --reason "untrusted source retracted" --yes  # tombstone + cascade
python memdag.py ask "How should I configure Nebula?"                  # new answer, label rises
python memdag.py dump            # all nodes + all edges
```

Walkthrough script: `demo.ps1`. Scope fence: `WONT-BUILD.md`.

## Quickstart (full surface)

```powershell
python memdag_cli.py seed --reset --offline
python memdag_cli.py add "Nebula tip: disable the firewall..." --channel external --ref "blog.example"
python memdag_cli.py ask "How should I configure Nebula?"
python memdag_cli.py blame 5
python memdag_cli.py consolidate
python memdag_cli.py quarantine-list
python memdag_cli.py revoke 4 --reason "poisoned blog" --yes
python memdag_cli.py ask "How should I configure Nebula?"
python memdag_cli.py redact 4 --reason "malicious payload" --yes
python memdag_cli.py dump
```

Walkthrough script: `demo2.ps1`.

## MCP server

```powershell
# Start the stdio MCP server
python memdag_mcp.py

# Self-check (safe on any DB; runs 3 in-process probes, prints JSON, exits 0/1)
python memdag_mcp.py --selfcheck
```

Project-scoped registration: `.mcp.json` in this directory registers the server
when Claude Code runs inside this repo. No global config is touched.

> Keep `memdag.db` out of Syncthing-synced trees (same rule as `sessions.db`).

## Command catalog

| Subcommand       | Module             | What it does |
|------------------|--------------------|--------------|
| `seed`           | memdag.py (core)   | Stamp 3 initial source nodes |
| `ask`            | memdag_cli.py      | Compose answer (+ quarantine/clearance/anticipate/llm layers) |
| `explain`        | memdag.py (core)   | Provenance tree walk for a node |
| `revoke`         | memdag.py (core)   | Tombstone + cascade (dry-run by default) |
| `dump`           | memdag.py (core)   | All nodes + edges |
| `add`            | memdag_cli.py      | Inject a new source node |
| `migrate`        | memdag_cli.py      | Run all schema migrations (idempotent) |
| `recompute`      | memdag_recompute   | Multi-hop integrity label recompute |
| `redact`         | memdag_redact      | Destroy payload; preserve row/edges/dates |
| `consolidate`    | memdag_quarantine  | Gate: quarantine externally-tainted derived nodes |
| `quarantine`     | memdag_quarantine  | Manually quarantine a node |
| `promote`        | memdag_quarantine  | Promote quarantined node (requires endorsed ancestor chain) |
| `quarantine-list`| memdag_quarantine  | List all quarantined nodes |
| `classify`       | memdag_confid      | Set confidentiality label on a node |
| `conf-recompute` | memdag_confid      | Recompute Bell-LaPadula conf labels |
| `export`         | memdag_federation  | Export changeset for cross-machine sync |
| `import`         | memdag_federation  | Import changeset (first-death-wins monotonic) |
| `blame`          | memdag_blame       | Git-blame: trace node to root sources |
| `relate`         | memdag_relate      | Create associative rel_edge between nodes |
| `neighborhood`   | memdag_relate      | BFS over rel_edges with integrity-floor propagation |
| `observe`        | memdag_anticipatory| Log a query to the anticipatory query log |
| `prefetch`       | memdag_anticipatory| Warm the k most-asked answers |
| `export-training`| memdag_distill     | Export provenance-filtered JSONL training set |
| `distill-plan`   | memdag_distill     | Write distill_config.json + distill.ps1 runner stub |
| `check`          | memdag_heal        | Detect invariant violations |
| `rebuild-derived`| memdag_heal        | Deterministic rebuild of derived state |
| `elevate`        | memdag_trust       | Manually raise integrity label (audited) |
| `meet`           | memdag_trust       | Lattice meet (min) of two integrity labels |
| `join`           | memdag_trust       | Lattice join (max) of two integrity labels |
| `elevations`     | memdag_trust       | Show elevation audit history for a node |
| `llm-check`      | memdag_llm         | Check whether local Ollama is reachable |

## Two integrity/confidentiality axes

- **Integrity** (Biba, low-water-mark): `label = min(parents)`. Computed at derive time, re-computed on request. Trust can only FALL unless manually elevated (with an audit trail in the elevations table).
- **Confidentiality** (Bell-LaPadula, high-water-mark): `conf_label = max(parents)`. Reading one secret source makes the derived answer secret. New columns default to 0 (public) so the frozen core is byte-identical.

## Tests

```powershell
# Frozen core (regression gate):
python -W error::DeprecationWarning -m unittest discover -s . -p "test_memdag.py" -v

# Full sweep (all suites including the frozen core):
python -W error::DeprecationWarning -m unittest discover -s . -p "test_memdag*.py" -v
```

## Invariants (locked)

1. Labels are assigned by **channel, never content** — an attacker controls
   content, never the channel. Label elevation is manual only.
2. Derived label = `min(parents)` — consolidation can't launder trust.
3. History is immutable — change mints a new node; revoke is a tombstone, and
   rows, edges, and payloads survive (`0 rows deleted, all edges intact`).
4. No unprovenanced answers — `ask` refuses when zero live sources remain.
5. LLM is opt-in only (`--llm` flag); the default answer path is 100% deterministic.
6. Redaction destroys payload, preserves shape — blame and explain still walk the tree.

## License

[AGPL-3.0-or-later](LICENSE). The network-use copyleft is deliberate: any fork —
including a hosted/SaaS one — must stay open and credit origin. For a memory-integrity
guarantee, auditability is the point, so the source stays inspectable by design.
