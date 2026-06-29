# memdag (working name)

[![tests](https://github.com/MattsomStudio/memdag/actions/workflows/tests.yml/badge.svg)](https://github.com/MattsomStudio/memdag/actions/workflows/tests.yml)

> **memdag is auditable, revocable memory with poison-proof answers.**

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

## For beta testers — start here

**👉 Full walkthrough: [TESTERS.md](TESTERS.md)** — a guided onboarding (install →
scripted mission → use it on your own data → try to break it), with expected
outcomes at every step and how to report bugs.

Setup is one command:

1. **Accept the GitHub invite** to `MattsomStudio/memdag`, then clone:
   ```
   git clone https://github.com/MattsomStudio/memdag.git
   cd memdag
   ```
2. **Run the bootstrap** — it installs memdag in an isolated env, installs Ollama +
   the embedding model, creates your database, optionally seeds from your own chat
   history, and wires your MCP client:
   ```
   python3 bootstrap.py          # macOS / Linux
   py bootstrap.py               # Windows
   ```
3. **Restart your MCP client** and ask it to use the `memdag` tools.

Then follow **[TESTERS.md](TESTERS.md)** to verify it worked and run the guided
mission. The notes below are reference detail for that guide.

### What gets set up
- **Your data lives in `~/.memdag/`** (the database + an isolated virtualenv).
  Nothing leaves your machine, and none of the author's data is in the repo. To
  remove everything: delete `~/.memdag/` and remove the server from your client.
- **Ollama is required** for semantic search — the bootstrap installs it and pulls
  `nomic-embed-text` (~275 MB, runs on CPU). If that step fails, memdag still works
  on keyword (BM25) search until you install Ollama manually; the bootstrap prints
  the exact command.
- **Client configs are merged, not overwritten** — your other MCP servers are kept
  and the file is backed up to `*.bak` first. The launch command is written as an
  absolute path (GUI clients don't inherit your shell `PATH`). Want to wire it
  yourself? `python3 bootstrap.py --print-only` prints the snippets and writes nothing.

### Per client
- **Claude Code (CLI):** wired via `claude mcp add --scope user` when the `claude`
  CLI is present; otherwise `~/.claude.json` is edited.
- **Claude Desktop:** `claude_desktop_config.json` (macOS / Windows / Linux). Desktop
  chats live server-side, so seeding there needs a manual export —
  `memdag ingest-chats --file <exported.json>`.
- **Codex:** `~/.codex/config.toml` (`[mcp_servers.memdag]`).

### If the bootstrap fails (manual setup)
The bootstrap is just a convenience wrapper. You can do every step by hand:
```
# 1. install memdag into an isolated environment
pipx install .                      # or: python3 -m venv ~/.memdag/venv && ~/.memdag/venv/bin/pip install .

# 2. install Ollama + the embedding model (https://ollama.com/download)
ollama pull nomic-embed-text

# 3. create the database
memdag init                         # creates ~/.memdag/memdag.db, prints its path

# 4. (optional) seed from your own chat history
memdag ingest-chats                 # asks before reading anything

# 5. print the exact client-config snippet and paste it yourself
memdag wire-config --print-only
```
Use the **absolute path** to the `memdag-mcp` executable in your client config —
GUI clients (Claude Desktop) don't inherit your shell `PATH`, so a bare
`memdag-mcp` fails with `spawn ENOENT`. `memdag wire-config` fills it in for you.

### Uninstall
```
rm -rf ~/.memdag                    # database + virtualenv
pipx uninstall memdag               # if installed via pipx
```
Then remove the `memdag` entry from your client config (a `*.bak` of the original
sits next to it).

### Found a bug?
Run `memdag doctor` and paste its output into a new GitHub issue (the bug-report
template prompts for it). It captures your OS, Python, Ollama status, DB path, and a
server self-check — everything needed to reproduce. See **[TESTERS.md](TESTERS.md)**
for the full reporting flow.

---

Two entry points:
- `memdag.py` — **frozen core slice** (do not touch; the original explain/revoke vertical slice, kept byte-identical as a regression anchor)
- `memdag_cli.py` — **full surface** (all feature modules + the friend-beta commands: `init`, `ingest-chats`, `doctor`, `wire-config`)

## Install

Python 3.12+, **zero runtime dependencies** (stdlib only — the optional Ollama
integration is reached over urllib and is never a pip dep). The repo keeps its
flat module layout; the wheel ships every `memdag*.py` as a top-level module.

```powershell
# dev install (editable) — from the repo root
pip install -e .

# you now have the `memdag` console command (same surface as memdag_cli.py)
memdag --help
memdag seed --offline
memdag ask "What is SQLite?"
```

The DB lands beside `memdag.py` by default; override with the `MEMDAG_DB` env
var. Running straight from the repo without installing still works exactly as
before (`python memdag_cli.py ...`).

> **VRAM hygiene knob:** by default memdag leaves Ollama's `keep_alive`
> alone — the model stays warm per Ollama's own setting. On a shared or
> small-VRAM card, set `MEMDAG_OLLAMA_KEEP_ALIVE=0` to make the model unload
> from VRAM immediately after every call (so memdag won't squat the GPU
> between queries); any Ollama duration string (e.g. `10m`) holds it warm
> longer.

## Quickstart (frozen core)

Python 3.12+, stdlib only. No install.

```powershell
python memdag.py seed            # stamp 3 sources: user fact, endorsed vault note, external article
python memdag.py ask "What is SQLite?"
python memdag.py explain 4      # provenance tree with channels, labels, dates
python memdag.py revoke 3 --reason "untrusted source retracted"        # dry run: blast radius
python memdag.py revoke 3 --reason "untrusted source retracted" --yes  # tombstone + cascade
python memdag.py ask "What is SQLite?"                  # new answer, label rises
python memdag.py dump            # all nodes + all edges
```

Walkthrough script: `demo.ps1`. Scope fence: `WONT-BUILD.md`.

## Quickstart (full surface)

```powershell
python memdag_cli.py seed --reset --offline
python memdag_cli.py add "SQLite tip: always enable WAL mode for read concurrency" --channel external --ref "blog.example"
python memdag_cli.py ask "What is SQLite?"
python memdag_cli.py blame 5
python memdag_cli.py consolidate
python memdag_cli.py quarantine-list
python memdag_cli.py revoke 4 --reason "poisoned blog" --yes
python memdag_cli.py ask "What is SQLite?"
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

> Keep `memdag.db` out of any synced or backup trees so private memories are not replicated.

## Claude Code memory loop

Beyond the MCP tools, memdag ships the always-on memory loop for the Claude Code
CLI — a write skill, a Stop hook that keeps the loaded index current, and a managed
block in your `CLAUDE.md`. `bootstrap.py` wires all of it (step `[6/6]`); to (re)do
it by hand:

```powershell
memdag wire-claude              # install skills + Stop hook + CLAUDE.md block
memdag wire-claude --print-only # show what it would do, write nothing
memdag wire-claude --with-gate  # also wire the opt-in Gate #3 taint hooks
```

How the loop works:

- **Capture** — the bundled `/saveall` skill sweeps a conversation and writes each
  fact as one file in your memory dir (`~/.claude/projects/<project>/memory/`).
- **Regenerate** — on session end the Stop hook runs `memdag bridge-render`, which
  re-imports those files, runs the forgetting pass, and rewrites the always-loaded
  `MEMORY.md` index from the store via the fail-safe digest writer (a bad render
  never blanks the file). `MEMORY.md` is **generated** — edit the per-fact files,
  not the index.
- **Instruct** — `memdag claude-sync` keeps a delimited, memdag-managed block in
  `~/.claude/CLAUDE.md` (the memory protocol) current. It only ever rewrites the
  block between its markers; everything else in your `CLAUDE.md` is yours.

Safety: `wire-claude` mirrors the MCP wiring contract — it backs up before writing,
merges (never overwrites your other hooks), is idempotent, and **refuses to
overwrite an existing same-named skill** without `--force`. Set
`MEMDAG_DIGEST_TITLE` for the `MEMORY.md` H1 and `MEMDAG_BRIDGE_AUTHOR=0` on extra
machines that should mirror without re-rendering.

> `/recall` ships too — it drives `memdag retrieve` (hybrid BM25 + local nomic
> vectors) over the store, so it searches everything ingested, not just the loaded
> MEMORY.md. The author's GPU/vault-coupled BGE retrieval engine is intentionally
> **not** included; `memdag retrieve` is the portable substrate.

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
| `profile`        | memdag_profile     | Leaf-origin provenance histogram (display-only; floor gates, profile never does) |
| `check-action`   | memdag_gate        | Action-time integrity gate: allow/deny by floor (the ONLY gate; audited in gate_log) |
| `gate-log`       | memdag_gate        | Show recent gate decisions (audit log) |
| `register-root`  | memdag_corroborate | Register an independence root for corroboration |
| `assert-claim`   | memdag_corroborate | Assert a structured claim under a registered root |
| `corroborate`    | memdag_corroborate | Mint a lift node when k independent roots agree |
| `claims-list`    | memdag_corroborate | List all claims and their corroboration status |
| `roots-list`     | memdag_corroborate | List all registered independence roots |
| `ingest`         | memdag_ingest      | Ingest a single file (channel stamped by caller; SHA-256 dedup; auto-chunked) |
| `ingest-dir`     | memdag_ingest      | Ingest all `*.md` (or `--glob`) files under a directory tree |
| `ingest-url`     | memdag_ingest      | Fetch a URL (GET) and ingest the body (channel always `external`) |
| `ingest-text`    | memdag_ingest      | Ingest raw text at a declared channel |
| `retrieve`       | memdag_retrieve    | Hybrid BM25 + optional-Ollama-vector ranked retrieval |
| `reindex`        | memdag_retrieve    | Rebuild BM25 postings for all live source nodes |
| `compact`        | memdag_compact     | Consolidate related episodes into a semantic node (edge-preserving, archived) |
| `archived-list`  | memdag_compact     | List all archived (compacted) episodes |
| `bridge-render`  | memdag_bridge_render | Regenerate MEMORY.md from the store (the Stop-hook command; fail-safe) |
| `claude-sync`    | memdag_claude      | Seed/refresh the memdag-managed memory block in CLAUDE.md |
| `wire-claude`    | memdag_wire_claude | Install the Claude Code memory loop (skills + Stop hook + CLAUDE.md) |

## Spine v1

### Ingestion adapters

Channel is stamped by the **adapter/transport**, never inferred from content. Three file-based and one raw-text adapter are available:

| Subcommand      | What it ingests                                              | Channel |
|-----------------|--------------------------------------------------------------|---------|
| `ingest <path>` | Single UTF-8 file (chunked if > 1 200 chars)                | caller declares |
| `ingest-dir <dir> --channel <c>` | All `*.md` (or `--glob`) files under a directory | caller declares |
| `ingest-url <url>` | HTTP GET response body                                  | always `external` |
| `ingest-text <text> --channel <c>` | Raw text string                              | caller declares |

Content is SHA-256-deduplicated (`content_hash` column, normalised whitespace). Duplicate content reuses the existing live node — no duplicate rows. Long text is auto-chunked on paragraph/sentence boundaries; each chunk is a separate node with `source_ref="…#chunkN"`. Every newly ingested node is automatically indexed for retrieval (best-effort — ingest never crashes if the retrieve module is absent or Ollama is down).

### Hybrid retrieval

BM25 is **pure stdlib** — works offline with zero external services. Ollama `nomic-embed-text` vectors are **optional**: Ollama unreachable → silent fallback to BM25-only, never a crash. Results are fused with Reciprocal Rank Fusion (RRF).

```powershell
# Build (or rebuild) the BM25 postings index
python memdag_cli.py reindex

# Query — returns ranked (id, content, channel, label, source_ref) rows
python memdag_cli.py retrieve "How does SQLite WAL mode work?"

# Opt-in retrieval inside ask: uses retrieve() to build the source pool
python memdag_cli.py ask --retrieve --topk 5 "How does SQLite WAL mode work?"
```

Without `--retrieve`, `ask` is **unchanged** — it uses all live sources exactly as before. Every existing test still passes.

### Edge-preserving compaction

`compact` consolidates groups of related live episodes into a single semantic (agent-derived) node, preserving every edge for provenance:

- The semantic node is minted via `derive_node()` — **edges point back to every episode**, label = `min(parents)` (no trust laundering).
- Episodes are **archived** (`archived = 1`), **never deleted** — rows and edges survive; `explain` and `blame` still walk them.
- `archived-list` shows all archived episodes.
- The quarantine integrity gate (`consolidate`) runs automatically after compaction.

```powershell
python memdag_cli.py compact                         # similarity grouping (Jaccard)
python memdag_cli.py compact --group-by claim        # group by corroboration claim
python memdag_cli.py compact --llm                   # Ollama summary (extractive fallback)
python memdag_cli.py archived-list
```

## Two integrity/confidentiality axes

- **Integrity** (Biba, low-water-mark): `label = min(parents)`. Computed at derive time, re-computed on request. Trust can only FALL unless manually elevated (with an audit trail in the elevations table).
- **Confidentiality** (Bell-LaPadula, high-water-mark): `conf_label = max(parents)`. Reading one secret source makes the derived answer secret. New columns default to 0 (public) so the frozen core is byte-identical.

## Biba-fatigue v1

### FLOOR vs PROFILE

The **floor** (`label` column = `min(parents)`) is the only thing that gates action decisions. It is enforced by `check-action` and nowhere else.

The **profile** is a display-only leaf-origin histogram. A derived node with 44 endorsed/user leaves and 1 external leaf reads as:

```
floor: EXTERNAL (gates) | provenance: 44 of 45 leaves endorsed/user, 1 external [mem:N] - inspect
```

...instead of a flat `EXTERNAL` scare label on every answer that touched the internet once. The fatigue fix does not weaken the gate — the floor still controls action decisions exactly as before.

`ask` and `explain` both append the profile line automatically (display-only; neither call `check_action`).

### Action gate

READ paths are always free: `ask`, `explain`, `profile`, `blame` never gate. They always return the answer regardless of integrity level.

`check-action <id> --require <level>` is the single enforcement point. It:
- reads the stored floor (never recomputes),
- compares floor >= required,
- writes one row to `gate_log` (audited),
- exits 0 on ALLOW, 2 on DENY,
- names the weakest live leaf ancestor (culprit) on DENY.

### Corroboration v1

Registered independence roots only. An assertion only earns credit if its `independence_root` is registered — unregistered and open-web sources get zero credit (fail-closed).

`k`-of-`n` distinct roots lifts `external(0)` -> `agent-derived(1)` **only**. The cap is load-bearing and never exceeded. The lift is a normal child node, so revoking any corroborating node cascades the lift away natively — no special teardown path needed.

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
