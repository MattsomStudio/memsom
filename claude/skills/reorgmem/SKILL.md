---
name: reorgmem
description: Maintenance sweep over the structured project-memory nodes — runs the deterministic `memsom project reorg` checks + `memsom audit`, applies the no-judgment mechanical fixes, and walks the content findings (broken links, stale specs, loose files, feature/decision drift) with you. Trigger when Matt says "/reorgmem", "reorg memory", "clean up the project memory", after a big migration, or when a session's project block shows `reorgmem: N open (Sunday)`. Returns a report of what was fixed and what needs a call.
---

# /reorgmem — structured project-memory maintenance

The problem this solves: a project node and its six sub-notes drift. Links go
dangling, a feature ships but its spec never gets the `Changes` line, a
sync-conflict copy lands next to the real file, loose `project_<slug>_*` files
pile up outside the fixed set. `/reorgmem` is the periodic sweep that catches all
of that. The **deterministic half** lives in code (`memsom project reorg`) so the
Sunday sweep can run it with no model; **this skill** is the interactive half that
also handles the findings needing judgment.

**Store safety:** every `memsom project …` and `memsom audit` command defaults
`--memory-dir` to the **live PC store**. That's what you want for a real reorg.
Pass `--memory-dir <copy>` only when rehearsing against a copy.

## Step 1 — Gather the findings

Two entry points:

- **Fresh run:** `memsom project reorg --json` (optionally `--project <slug>`).
  It runs every `project check` schema finding **plus** the maintenance checks —
  sub-note presence/kind, log caps, dangling wikilinks, missing `[[fact_*]]` refs,
  `Rules & gates` ⊆ the architecture note, sub-note count drift, and
  `*.sync-conflict-*` copies. Also run `memsom audit --json` for the store-wide
  view (orphans, budget, nested frontmatter).
- **Continuing the Sunday sweep:** if `<memory>/.weights/reorgmem_pending.json`
  exists and is non-empty, start from it — the headless sweep already applied the
  mechanical fixes and left every content finding there (per project). Don't
  re-scan; read that file. (A matched project's prompt block shows
  `reorgmem: N open (Sunday)` when this file has findings for it.)

Each finding carries `fix: mechanical | content`.

## Step 2 — Apply the mechanical fixes

`fix: mechanical` = no judgment. Apply them with:

```
memsom project reorg --apply [--project <slug>]
```

That does exactly two things, safely:
- **Sync-conflict copies:** an identical copy is deleted; a divergent *log* note
  (`## Entries`) is union-merged by entry ID and the copy deleted; a divergent
  *non-log* file is **kept** and surfaced as a content finding (a human picks).
- **Sub-note counts:** the node's `## Sub-notes` wikilinks get fresh `— N` counts.

Nothing else is touched. This is the same code path `--sweep` runs headless.

## Step 3 — Walk the content findings (the judgment half)

For each `fix: content` finding, propose the fix, show the diff, apply on Matt's
"go". Never guess a content edit; never delete.

| Finding | What to do |
|---|---|
| `reorg-link-broken` | the `[[stem]]` points at nothing — fix the target, or drop the link if the referenced memory is genuinely gone |
| `reorg-fact-missing` | create the `fact_*` file (see the fact-authoring recipe) or correct the ref |
| `reorg-subnote-cap` | a log note is over its cap — fold superseded entries into a `## History` section (keep the newest live), and if History itself exceeds ~60 lines move it to a vault artifact + a `## Pointers` line |
| `project-schema` (spec.stale) | a feature shipped/changed but its spec note's `## Changes` is older than a log entry naming it — bring the spec current with `memsom project spec <slug> <feat> --set behaviour "…" --why "…"` |
| `project-schema` (features/left/needs-matt) | the node's `### Left`/`### Next`/`### Needs Matt` disagree with the feature statuses — fix the status (`project feature`) or the Status bullet; an `active-decision` with no answer stays a `### Needs Matt` question for Matt |
| `project-loose-file` | a stray `project_<slug>_*` file outside the fixed set — **absorb, never delete**: add `index: false` + a `## Absorbed <date>` bullet pointing at the node, or if it's wrong/obsolete `memsom tombstone <stem> --reason "folded into project_<slug> <date>"` |
| `project-nested-frontmatter` | flatten the frontmatter (the importer ignores keys nested under `metadata:`) |
| `project-alias-clash` / `project-creds-value` | ERROR — an alias two nodes claim (drop it from both or rename) / a secret in `## Creds` (replace with a pointer) |

**Two model-assisted checks the deterministic pass does NOT adjudicate** (do these
by reading, in an interactive run):
- **decisions.coverage:** grep the project's bound memory + vault files for
  decision language (`DECIDED`, `Matt's call`, `OPTION \d`, `go with`, `we
  decided`); every such commitment should have a `decisions` entry. Missing one →
  add it with `memsom project log <slug> decisions …`.
- **tests.drift:** `git ls-files tests/` from the project's `dir_pc`/`dir_mac` vs
  the `tests` note, both directions — a test file with no `T-` entry, or a `T-`
  entry whose path is gone.

## Step 4 — Scale it

Inline when it's small: **< 10 findings and < 15 files**. Otherwise spawn a
subagent with this skill + the relevant slice of the plan
(`~/.claude/plans/greedy-sauteeing-pnueli.md`) as its brief, one project at a
time, and review its diffs before applying. Migrations are always subagent-scale.

## Step 5 — Verify and report

After applying, re-run `memsom project check <slug>` per touched project — it must
be clean (or only the known cosmetic loose-file WARNs). Then report:

```
Reorg — <slug> (+ others):
Mechanical (applied):
- <fix> — <target>
Content (applied on your go):
- <fix> — <target>
Needs your call:
- <question>
Clean: project check <slug> → clean
```

## The Sunday sweep (headless, no model)

The weekly consolidation task runs `memsom project reorg --sweep`: it applies
ONLY the mechanical fixes and writes every content finding to
`<memory>/.weights/reorgmem_pending.json` (+ a line in `reorgmem_log.jsonl`).
Nothing content-bearing is ever applied unattended. The next interactive
`/reorgmem` picks up from that pending file (Step 1).
