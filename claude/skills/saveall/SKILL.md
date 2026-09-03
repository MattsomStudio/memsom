---
name: saveall
description: Sweep the current conversation for memory-worthy content (facts, preferences, project state, references) and persist each as one file in the Claude memory store, then let memsom regenerate the always-loaded MEMORY.md index. Trigger when the user says "/saveall", "save everything", "save this session", or finishes a meaty session and wants to lock in the learnings. Returns a summary of what was saved.
---

# /saveall — end-of-session memory sweep

When invoked, do a full sweep of the conversation, identify everything worth
keeping for future sessions, write each piece to the memory store as its own file,
and report back. The goal: nothing valuable is lost to a future `/clear`, and
nothing ephemeral pollutes long-term memory.

The per-fact files are the **live input**. memsom re-imports them and
regenerates `MEMORY.md` (the always-loaded index) on session end via the
`memsom bridge-render` Stop hook — so you never hand-edit `MEMORY.md`; you write
the per-fact files and let it regenerate.

## Step 0 — Where memory lives

The memory directory is `~/.claude/projects/<project>/memory/` (the `<project>`
segment is machine-specific; it's the dir that holds `MEMORY.md` plus the per-fact
`*.md` files). If `$MEMDAG_BRIDGE_MEMORY_DIR` is set, use that.

Layout — **flat, with one exception**:

```
memory/<type>_<topic>.md                         every non-project memory (flat)
memory/projects/<slug>/project_<slug>.md         a project's parent overview
memory/projects/<slug>/project_<slug>_<sub>.md   its subprojects (any number)
memory/projects/project_<x>.md                   a standalone project (no subprojects)
memory/projects/INDEX.md                         GENERATED — never write it
```

Project rule: **append to or update the existing parent/subproject file when one
exists; create a new subproject file only for a genuinely new thread; never create
a sibling of an existing project at the top level.** Filenames must be unique
across all levels. Create no other subdirectory.

A project may instead be a **structured node** (`kind: project-node`): one parent
carrying What / Status / Features / Rules / Creds / Where, six fixed sub-notes
(spec, gotchas, decisions, interface_io, architecture, tests), and one spec note
per feature. Never hand-write files for such a project — route through the
`memsom project` CLI (Step 3.5).

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

## Step 3.5 — Project routing (structured project nodes)

Before you write a `project_*` file, check whether the project it belongs to is a
**structured node**: `memsom project list` (each row is a `kind: project-node`
project). If it is, you do NOT hand-write a file — every project fact goes through
the `memsom project` CLI, which keeps the node, the sub-notes, the feature list and
the spec notes in sync. (The CLI defaults `--memory-dir` to the live store.)

**1 — Bind the project.** Match a session fact to a node when the conversation
names one of its aliases (`memsom project show <slug>` → frontmatter `aliases:`) OR
edits a path under one of its `dir_pc` / `dir_mac` / `dir_droplet`. A fact can bind
to more than one node; route a copy to each.

**2 — Bucket each fact to the right sub-note / command:**

| The fact is… | Route it to |
|---|---|
| a bug WITH its cause | `project log <slug> gotchas "**symptom**" --cause … --fix … --where file:line` — a fix with no known cause is a `### Left` item, not a gotcha |
| a decision (see the detector below) | `project log <slug> decisions "**decision**" --rejected "alt:why" --why … --source user\|session:<date>` |
| an Edit/Write under `tests/**` | `project log <slug> tests "\`tests/path\`" --covers … --run "<cmd>"` |
| an endpoint / CLI / IPC / MCP contract change | edit the `interface_io` note (`project show <slug> --note interface_io` then update) |
| a `never …` / `always …` / `before X …` rule | the node `## Rules & gates` (`project status … ` is not it — edit the node) AND the `architecture` note Invariants/Gates |
| a milestone reached / blocked | `project status <slug> --done "(MEASURED) …"` / `--next` / `--left` / `--ask` |
| a feature shipped / started / dropped | `project feature <slug> <feature-id> --status implemented\|planned\|active-decision\|archived` (implemented needs `--evidence "(MEASURED) …"`) |
| a change to what a feature DOES (signature, limit, default, contract) | `project spec <slug> <feature-id> --set behaviour "…" --why "…"` **in the same sweep** — a build that changed behaviour without touching the spec is a `/reorgmem` red |
| >40 lines of narrative design | a vault artifact + one `## Pointers` line on the node |

**3 — Decision detector (precise, to keep decisions clean):** log a `decisions`
entry ONLY on **user commit language** — "go with", "decided", "not Y", "drop",
"keep", "option N", or a yes to a numbered option — or an assistant proposal the
user accepted within ~3 turns. An assistant-only recommendation is NOT a decision:
it goes to `### Needs Matt` as a question (`project status <slug> --ask "…"`).

**4 — Evidence discipline:** `### Done` bullets and `--evidence` carry a
`(MEASURED)` / `(DERIVED)` tag. Bump `last-verified` (`project status <slug>
--verified`) only when you actually re-checked the claim against the repo, not just
because you touched the node.

**Hard rule:** never create a loose `project_<slug>*` file for a project that has a
node — that throws a `project check` loose-file WARN and the fact escapes the
structured render. If the fact doesn't fit any bucket, it probably belongs in a
flat `reference_*` / `feedback_*` file, not the project.

## Deleting a memory

If a memory turns out to be wrong or obsolete, don't just delete the file (its node
stays live in the store and keeps rendering). Use the sanctioned path, which revokes
the node (auditable, cascades to anything derived from it) and removes the file:

```
memsom tombstone <stem> --reason "why"
```

Pinned `user_`/`feedback_`/`personal_` memories are refused unless you pass `--force`.

## Step 4 — Don't hand-edit MEMORY.md

`MEMORY.md` is generated from the store. After you write the per-fact files, the
Stop hook (`memsom bridge-render`) re-imports them and rewrites `MEMORY.md`. You may
run it yourself to see the result immediately:

```
memsom bridge-render
```

Keep the rendered index lean — it must fit **both** `memory_budget` bytes and
`memory_max_lines` lines (`memory/.weights/canonical.json` → `params`; defaults
16384 / 180 — read the file, don't assume). `project_` files never render into
MEMORY.md; they render into the generated `projects/INDEX.md` (one group per
project, subprojects nested, tagged Active / Parked / Closed from `status:`).
The forgetting layer drops unused `reference_` lines automatically; `user_` /
`feedback_` / `personal_` are pinned and never auto-dropped; `## Live state` and
`fact_` entries are shed last. A file with `section: none` is deliberately out.

**Feedback is the exception to "one file, one line".** A new lesson goes INTO the
body of the matching `feedback_cluster_*` file (one bullet: `- [[stem]] — rule`,
or just the rule). A brand-new standalone `feedback_*` file is imported
*unindexed* (shed reason `needs_cluster`) unless its frontmatter carries
`why_own_line: <reason>`; the Feedback section also has its own byte budget
(`section_budgets` in the same params file) that pinning does not exempt.

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
