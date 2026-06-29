---
name: saveall
description: Sweep the current conversation for memory-worthy content (facts, preferences, project state, references) and persist each as one file in the Claude memory store, then let memdag regenerate the always-loaded MEMORY.md index. Trigger when the user says "/saveall", "save everything", "save this session", or finishes a meaty session and wants to lock in the learnings. Returns a summary of what was saved.
---

# /saveall — end-of-session memory sweep

When invoked, do a full sweep of the conversation, identify everything worth
keeping for future sessions, write each piece to the memory store as its own file,
and report back. The goal: nothing valuable is lost to a future `/clear`, and
nothing ephemeral pollutes long-term memory.

The flat per-fact files are the **live input**. memdag re-imports them and
regenerates `MEMORY.md` (the always-loaded index) on session end via the
`memdag bridge-render` Stop hook — so you never hand-edit `MEMORY.md`; you write
the per-fact files and let it regenerate.

## Step 0 — Where memory lives

The memory directory is `~/.claude/projects/<project>/memory/` (the `<project>`
segment is machine-specific; it's the dir that holds `MEMORY.md` plus the per-fact
`*.md` files). If `$MEMDAG_BRIDGE_MEMORY_DIR` is set, use that. Each memory is a
**flat file in this directory** — never create a `memory/` subdirectory.

Read the current `MEMORY.md` first so you don't duplicate something already saved.

## Step 1 — Scan and bucket

Go through the session and bucket each save-worthy item into one type:

| Type | What it is | File prefix |
|---|---|---|
| **user** | who the user is — role, skills, durable preferences, context | `user_<topic>.md` |
| **personal** | private / self-reflection notes about the user (also endorsed/pinned) | `personal_<topic>.md` |
| **feedback** | corrections, validated approaches, "from now on do X" | `feedback_<topic>.md` |
| **project** | milestone, status change, decision, deadline, who's doing what | `project_<topic>.md` |
| **reference** | external pointers — "the docs are at X", "issues go in tracker Y" | `reference_<topic>.md` |

**Save to memory if** it's a quick fact / status / preference / pointer that
should drive behavior or load as context next session.

**Don't save:** ephemeral task state ("where we left off"), in-progress reasoning,
or raw transcript. Distill into the actual learning, or skip it.

## Step 2 — Write each item

Each memory is one file with this frontmatter:

```markdown
---
name: <short-kebab-case-slug>
description: <one specific line; used to judge relevance on recall>
type: user | personal | feedback | project | reference
source: user | session:<YYYY-MM-DD>
salience: <0.00-1.00>
---

<the fact.>
```

- **`source`** is provenance: `user` when the user stated it directly, else
  `session:<YYYY-MM-DD>` (or `external:<ref>` when distilled from a specific source).
- **feedback / project bodies** lead with the rule/fact, then a `**Why:**` line,
  then a `**How to apply:**` line.
- Link related memories in the body with `[[their-name]]` (the bare slug). A link to
  a memory that doesn't exist yet is fine — it marks one worth writing later.
- **`salience`** (0.00–1.00) seeds how strongly to encode — it slows how fast a
  memory decays out of the index. Score it affect-first:
  `salience = 0.5*affect + 0.3*surprise + 0.2*source`, rounded to 2 dp.
  - *affect:* emotional/identity weight of the content (a real decision, a hard
    lesson → high; routine config/parameter → low).
  - *surprise:* novelty vs what memory already holds (genuinely new → high; a
    restatement → low).
  - *source:* user-stated → high; inferred by you → lower.
  When unsure, use **0.30** (don't inflate — saturating salience makes it useless).

## Step 3 — Update vs create, and the interference check

- **Prefer updating** an existing file over creating a near-duplicate. If a topic
  already has a file, edit it.
- **Interference check:** for each new memory, find the nearest existing one. If it
  both heavily overlaps AND genuinely contradicts the new fact (memory says X, this
  says NOT-X — an IP changed, a tool was replaced, a status flipped):
  1. add `supersedes: [[<existing-slug>]]` to the NEW file's frontmatter,
  2. keep both files (never delete/edit the old one at write time),
  3. surface it in the report and ask the user how to resolve. Never auto-resolve.
  - Guard: similarity ≠ contradiction. "uses tool X" and "tool X is v2.1" overlap
    but are compatible — that's an update, not a conflict.

## Deleting a memory

If a memory turns out to be wrong or obsolete, don't just delete the file (its node
stays live in the store and keeps rendering). Use the sanctioned path, which revokes
the node (auditable, cascades to anything derived from it) and removes the file:

```
memdag tombstone <stem> --reason "why"
```

Pinned `user_`/`feedback_`/`personal_` memories are refused unless you pass `--force`.

## Step 4 — Don't hand-edit MEMORY.md

`MEMORY.md` is generated from the store. After you write the per-fact files, the
Stop hook (`memdag bridge-render`) re-imports them and rewrites `MEMORY.md`. You may
run it yourself to see the result immediately:

```
memdag bridge-render
```

Keep the rendered index lean — the budget is **16,384 bytes** (it loads in full
every session). The forgetting layer drops unused `project_`/`reference_` lines
automatically; `user_`/`feedback_`/`personal_` are pinned and never auto-dropped.

## Step 5 — Report back

After all writes, give a tight, scannable summary:

```
Saved to memory:
- <file> — <one-line description>
- <file> — <one-line description>

Skipped (not save-worthy):
- <thing> — <reason>

⚠️ Conflicts to resolve:
- <new-file> supersedes <old-file> which said "<old claim>"   (only if any)
```

## Notes on judgment

- Two separate topics → two files. One coherent thing → one file.
- When you're unsure where something goes, ask rather than guessing — one quick
  clarification beats a misfiled memory.
