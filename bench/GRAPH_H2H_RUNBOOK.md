# Graph self-ablation — runbook (memsom-retrieve vs memsom-graph)

Everything is built and static-checked. This is the exact sequence for tomorrow.
Nothing here has been run yet.

## What's wired
- `bench/bench_linker.py` — blind LLM link-writer. `link(texts) -> [(i,j)]`.
  Sees ONLY the texts (no question, no `answer_bearing`). temp 0 + disk cache
  (`bench/.linker_cache/`) so edges are reproducible across both arms/runs.
- `adapters/memsom_adapter.py` — `graph`/`hops` flags. On the graph arm it
  captures each `add`'s node id, then before `ask` runs the linker over the
  item's evidence and writes the pairs as `wikilink` rel_edges, then asks with
  `--graph --hops N`. `graph=False` is the untouched retrieve baseline.
- `run_headtohead.py` — `make_adapter` now passes `graph`/`hops` from `--config`.

Both arms are `--system memsom`; they differ ONLY by `--config`. Poison selection
is deterministic (stride pick, no RNG), so both arms see identical items+poison.

## Preflight (do first, or the graph arm fails loud — by design)
1. Ollama up and the linker model present:
   `ollama list | grep qwen2.5:7b-instruct`  (or set BENCH_LINKER_MODEL)
   `curl -s http://localhost:11434/api/tags >/dev/null && echo ok`
2. Warm the embedder (your rule for sweeps) before the big run.
3. Pick the corpus config — THE LANDMINE: the graph only re-ranks WITHIN the
   retrieved candidate set. If `--topk` >= sources/item, the graph cannot change
   recall and you get a meaningless zero. Run a big haystack: high `--max-evidence`
   (full sessions), modest `--topk 8`. Set `$DATASET` to the LongMemEval file.

## Run — two arms, identical everything but the flag

```bash
cd C:/Users/you/memsom/bench
REPO=C:/Users/you/memsom
DATASET=<path-to-longmemeval-oracle.json>   # <-- fill in
ME=20            # max-evidence: big enough that candidates >> topk (the landmine)
COMMON="--repo $REPO --dataset $DATASET --max-evidence $ME --topk 8 --rate 0.5 --workers 6 --judge"

# Arm 1 — retrieve baseline (graph OFF)
python run_headtohead.py --system memsom $COMMON \
  --config '{}' \
  --run-root ./h2h/retrieve --out ./h2h/retrieve.json

# Arm 2 — graph (blind edges + ask --graph)
python run_headtohead.py --system memsom $COMMON \
  --config '{"graph": true, "hops": 2}' \
  --run-root ./h2h/graph --out ./h2h/graph.json
```

Heads-up: estimate runtime on a small `--max-items 20` smoke first, then the full
run. Keep the embedder warm between arms.

## Compare
- Headline delta = graph.json evidence-recall − retrieve.json evidence-recall.
- Also read the judge score delta (expected smaller — graph helps retrieval, not
  composition).
- Pre-registered bet: +3 recall, +1 judge, ~25% chance flat-to-negative.
- REPORT THE RESULT EITHER WAY. A null in a clean ablation is a real finding.

## Anti-cheat verification (run before trusting a positive result)
1. Code wall: `link()`'s only argument is `texts`. Confirm nothing passes the
   question or `answer_bearing` into it — `grep -n answer_bearing adapters/memsom_adapter.py`
   should show it captured and dropped in `add()`, never forwarded.
2. Edge sanity: on a couple of items, dump `rel_edges` and confirm edges connect
   topically-related turns, not "answer turn wired to everything."

## Deferred
- `mem0-graph` arm (neo4j/kuzu backend) — not built. Add later for the true
  head-to-head; the self-ablation above stands on its own.
