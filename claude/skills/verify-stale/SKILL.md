---
name: verify-stale
description: Flag state-bearing memory notes whose verification age has lapsed (deterministic, no LLM). Trigger on /verify-stale, "check for stale memory claims", "which memories need re-verification", "run the staleness sweep", "is anything in memory out of date".
---

# /verify-stale — verification-age staleness sweep

Run the sweep and report its output. **Dry-run by default** — safe anytime; nothing
is written unless `--apply` is passed.

```
memsom verify-stale          # dry-run: reports what WOULD change
memsom verify-stale --apply  # commits the marks/clears
```

## What it flags

A note is a candidate only if it's **state-bearing** — its content describes a
status that can go out of date ("not deployed", "unfinished", "in progress", or
similar), or it carries an explicit `DUE:`/`deadline:` date. Stable facts (preferences,
identity, historical decisions) are never candidates, no matter how old.

A candidate note gets marked **stale** once either is true:
- its `DUE:`/`deadline:` date has passed (regardless of how recently it was touched), or
- it hasn't been re-verified (`last-verified:` frontmatter, or file mtime as a
  fallback) in more than `MEMDAG_VERIFY_STALE_DAYS` days (default **21**).

Only `user`-channel notes (the `project_`/`reference_` kind — status snapshots,
not durable rules) are candidates. `endorsed` notes (`user_`/`personal_`/`feedback_`
— pinned facts and standing preferences) are never freshly marked, though a leftover
flag on one still clears on a run.

## Output

```
[verify-stale] DRY-RUN (no writes)  threshold=21d
  scanned : 42 memory file nodes
  marked  : 2 -> stale
  cleared : 1 (re-verified / no longer stale)
  stale   : project_wol_openwrt, project_xray_reality_tunnel
```

- **marked** — newly flagged stale this run.
- **cleared** — previously flagged, now re-verified or no longer state-bearing
  (only clears flags this sweep itself owns — a staleness flag from source
  supersession, e.g. a re-ingested changed source, is left alone).
- **stale** — bare stems (`project_x`, not `memory:project_x`) of every note
  currently flagged, for a quick look at what needs attention.

## Interpreting for the user

- Staleness is **disclosure, not enforcement** — a stale note still answers `ask`
  queries; it's flagged, not excluded. (`ask --fresh-only` opts into exclusion,
  `ask --prefer-fresh` opts into silently swapping in a fresher version if one
  exists — neither is this skill's concern.)
- Nothing here gates reads or gets in the way — the point is surfacing what's gone
  stale so a human (or a future re-ingest) can catch it up.
- Recommend `--apply` once the dry-run output looks right; there's no harm in
  running dry-run repeatedly first.

## Related

- `/audit` — the structural-integrity neighbor (orphan files, dead links, budget).
  Different axis entirely: audit checks the store is *sound*, this checks claims
  are *current*. No overlap.
- `memsom stale-cascade` / `memsom freshen` — the underlying cascade primitives
  this sweep calls into (`memsom_stale.py`): a source re-ingest fires the same
  cascade this sweep does, just on a different trigger (content-hash change vs.
  verification age).
