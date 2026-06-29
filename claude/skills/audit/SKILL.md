---
name: audit
description: Deterministic structural integrity audit of the Claude memory store (the flat per-fact files that feed memdag and the generated MEMORY.md index). Trigger on /audit, "audit my memory", "check memory integrity", "is my memory store healthy", "any broken links or orphans in memory". Catches orphan files, dead MEMORY.md links, broken frontmatter, bad types, broken wikilinks, and MEMORY.md budget breaches. No LLM, read-only.
---

# /audit — memory structural integrity check

Run the auditor and interpret its output. It is **read-only** — safe anytime.

```
memdag audit --json
```

Parse the JSON and summarize for the user; or drop `--json` to show the human
report verbatim. Pass a path as the first argument to audit a specific memory dir;
with no argument it auto-detects `~/.claude/projects/<project>/memory`.

## What it checks

ERROR-class (exit 1): **orphan-file** (live in the store but absent from MEMORY.md
and not a deliberate cold demote — the index and the store disagree),
**dead-index-link** (MEMORY.md links a file that's missing),
**frontmatter-missing** (no name / description / type).

WARN-class: **bad-type** (type not in user/feedback/project/reference),
**budget-breach** (MEMORY.md over the 16384-byte cap).

INFO-class: **pending-import** (file on disk not yet imported — run
`memdag bridge-render`), **broken-wikilink** (a `[[link]]` with no target — *legal
by design*, just a heads-up).

## Interpreting for the user

- **Lead with the ERROR count.** Zero errors = the store is structurally sound; say
  so plainly.
- **orphan-file** and **dead-index-link** are the actionable ones. Because MEMORY.md
  is **generated** by `memdag bridge-render`, the fix is never to hand-edit the
  index — for an orphan, re-run `memdag bridge-render` (it re-renders from the
  store); for a dead link, restore the file or let the next render drop the line.
- **pending-import** just means a freshly written file hasn't been imported yet —
  run `memdag bridge-render` and it clears.
- INFO is noise-by-design; mention the count, don't enumerate unless asked.
- Flag budget headroom when it's tight (near the 16384 cap).

## Why it's report-only

The forgetting layer checks whether a memory is still *used*; the staleness pass
checks whether its claims still match reality. This audit is the third leg — whether
the store is structurally *sound*. It never writes: the correct fix for every
finding is to edit the per-fact file (or re-render), so there's nothing safe to
auto-patch on the generated index.

## Related

- `/saveall` — the write path this audits the output of.
- `memdag bridge-render` — regenerates MEMORY.md from the store (clears orphan /
  pending-import findings).
