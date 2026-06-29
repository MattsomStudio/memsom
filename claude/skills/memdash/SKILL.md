---
name: memdash
description: Build and open an HTML telemetry dashboard for the Claude memory system — visualizes the forgetting-layer weights, hot/cold tier split, access counts, accessibility (RS) distribution, store growth, demote-risk watchlist, the section/wikilink relationship graph, and the MEMORY.md budget as interactive charts. Trigger on /memdash, "memory dashboard", "visualize my memory", "show memory telemetry", "graph the memory weights". Read-only; never writes to the store.
---

# /memdash — memory telemetry dashboard

Generate a self-contained HTML dashboard from the memdag store's live data and open
it in the browser. **Read-only** — it reads the `forget_*` node columns, the
generated `MEMORY.md`, and (optionally) the episodic session archive; it never
mutates the store. Safe anytime.

```
memdag dashboard
```

On a `/memdash` invocation, **build AND open it** — run the command as above (it
writes `~/Desktop/memory-telemetry.html` and launches the browser). Flags:
`--no-open` (build only, for scripted/headless use), `--out PATH` (custom file).

## Data sources

- **`~/.memdag/memdag.db`** — live forgetting telemetry from the `forget_*` columns:
  `forget_rs` (RS / accessibility, shown as weight), `forget_count`,
  `forget_tier` (hot = in MEMORY.md / cold = demoted), `forget_first_seen`,
  `forget_last_used`. Pinned = endorsed channel (`user_`/`feedback_`/`personal_`)
  or a frontmatter `pin`. Override the path with `$MEMDAG_DB`.
- **`MEMORY.md`** (the generated index) — byte size vs the 16384 budget cap.
- **episodic `sessions.db`** (optional) — session count; the card is omitted if the
  archive isn't present.

## What it shows

- **Stat cards** — total / hot / cold / pinned, MEMORY.md budget gauge, session count.
- **Tier doughnut** (hot vs cold) and **by-type** bar.
- **Accessibility (RS) histogram** — buckets under the 0.2 demote floor flagged.
- **RS vs access-count scatter** — every memory, colored by tier, demote floor drawn.
- **Demote-risk watchlist** — unpinned memories with the lowest RS / longest idle.
- **Store growth** (cumulative by first-seen) and **most-accessed** top 15.
- **Relationship graph** — MEMORY.md sections are parent hubs, memories are siblings,
  `[[wikilinks]]` are cross-links (D3 force layout).

## Interpreting for the user

- **Budget gauge is the one to watch** — if it's red (>90%), the next `/saveall`
  may breach and the forgetting layer will demote candidates; `/audit` lists them.
- The **demote-risk watchlist** is the actionable list — those fall out of MEMORY.md
  next. To keep one, pin it (`user_`/`feedback_` are auto-protected).

## Related

- `/audit` — structural integrity (orphans, dead links). Different layer: `/audit`
  checks *correctness*, `/memdash` visualizes *telemetry*.
- `memdag bridge-render` — the Stop hook that writes the `forget_*` data this charts.
