---
name: recall
description: Search the memdag memory store by meaning, not just by what's loaded in MEMORY.md. Trigger when the user asks things like "what did we figure out about X", "recall that conversation about Y", "search my memory for Z", "what do I have on W", "remember when we worked on…", or any request to find prior content that isn't in the always-loaded MEMORY.md index. Runs memdag's hybrid (BM25 + local vector) retrieval over the whole store and returns ranked hits with their provenance.
---

# recall — semantic search over the memdag store

`MEMORY.md` is only the always-loaded *index* — the hot, pinned, recently-used
memories. The full store holds far more (cold-demoted facts, ingested chat history,
ingested docs). `/recall` searches all of it.

## Query — one command

```
memdag retrieve "QUERY" --k 8
```

`retrieve` is **hybrid**: BM25 keyword ranking fused (RRF) with local vector
similarity (Ollama `nomic-embed-text`). If Ollama is unreachable it silently
degrades to BM25-only — still useful, just no paraphrase matching. Each hit prints
its node id, channel, integrity label, source ref, and a snippet.

- `--k N` widens/narrows the result count (default 8).
- To pull the full provenance of a promising hit: `memdag explain <id>`.
- To trace a hit back to its root sources: `memdag blame <id>`.

## What's actually searchable

`retrieve` searches the **memdag store** — one corpus, everything that's been
ingested:

- facts captured via `/saveall`,
- past chat history, if ingested: `memdag ingest-chats`,
- notes / docs / a folder of markdown, if ingested: `memdag ingest-dir <dir>`.

If a search comes up empty, it's usually because the content was never ingested —
suggest the relevant `ingest-*` command rather than assuming it isn't there. After
a large ingest, `memdag reindex` rebuilds the BM25 postings.

## Presenting results

- Lead with what the snippet says, not "I found N hits."
- Cite each hit's source ref / channel so the user can locate it, and note the
  integrity label — a hit floored at `external` is weaker-trust than a `user` /
  `endorsed` one.
- If hits look thin, say so and widen with `--k`, or rephrase (prose queries lean
  on vectors; exact identifiers lean on keyword). Never fabricate a hit.
- If results are clearly keyword-only (Ollama down), mention it — a paraphrase
  query may do better once the local embedder is back.

## Related

- `/saveall` — writes the facts this searches.
- `memdag explain` / `memdag blame` — provenance of a specific hit.
