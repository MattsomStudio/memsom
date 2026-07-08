"""precache_links — pre-build the blind-linker edge cache for the graph arm.

Runs bench_linker.link() over every LongMemEval item's text list, mirroring
EXACTLY what memsom_adapter feeds the linker at ask-time: the evidence turns in
from_longmemeval order, plus the single poison turn for poisoned items (see
run_headtohead._work lines 69-73). Because bench_linker caches by a hash of that
text list, populating it here means the graph arm hits cache tomorrow instead of
paying per-item LLM latency — and the edges are identical, so the run stays
reproducible.

Lock (dataset, max_evidence, rate) to the SAME values the graph arm will use, or
the hashes won't match and it re-links. Gold-blind: only evidence/poison TEXT is
ever passed to the linker — never the question, answer_bearing, or gold.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import bench_linker
from dataset import from_longmemeval
from poison import select_poisoned


def item_texts(item: dict, poison_ids: set) -> list[str]:
    """Reproduce the adapter's self._texts for one item, in add() order."""
    texts = [ev["text"] for ev in item["evidence"]]
    if item["id"] in poison_ids and item.get("poison"):
        texts.append(item["poison"]["text"])
    return texts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="precache_links")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-evidence", type=int, default=20)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N items (ETA sample)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    items, _ = from_longmemeval(args.dataset, max_items=args.limit,
                                max_evidence=args.max_evidence)
    poison_ids = select_poisoned(items, args.rate)

    # Mirror run_headtohead EXACTLY: every item runs a "clean" task (never
    # poisoned) AND a "poisoned" task (poison only for poison_ids). So the linker
    # sees the evidence-only list for ALL items, plus the evidence+poison list for
    # poison_ids items. Dedupe by cache key so identical lists (a non-poison item's
    # two tasks) are linked once. Missing the clean variant of poisoned items was
    # the cache-miss bug that 429-stormed the first graph-arm run.
    targets = {}
    for is_p in (lambda _id: False, lambda _id: _id in poison_ids):
        for it in items:
            texts = [ev["text"] for ev in it["evidence"]]
            if is_p(it["id"]) and it.get("poison"):
                texts.append(it["poison"]["text"])
            targets.setdefault(bench_linker._cache_key(texts), texts)
    targets = list(targets.values())
    print(f"[precache] backend={bench_linker.BACKEND} model={bench_linker.MODEL} "
          f"items={len(items)} distinct_text_lists={len(targets)} "
          f"max_evidence={args.max_evidence} rate={args.rate} workers={args.workers}",
          flush=True)

    t0 = time.time()
    edges, in_tok, out_tok, failed = [], 0, 0, 0

    def work(texts):
        pairs = bench_linker.link(texts)
        return len(pairs), dict(bench_linker.LAST_USAGE)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, t): i for i, t in enumerate(targets)}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                ne, usage = f.result()
            except Exception as e:   # one stubborn 429 must not kill the whole run
                failed += 1
                print(f"  ITEM FAILED ({type(e).__name__}: {e}) — re-run to retry",
                      flush=True)
                continue
            edges.append(ne)
            in_tok += usage.get("prompt_tokens", 0)
            out_tok += usage.get("completion_tokens", 0)
            if done % 25 == 0 or done == len(targets):
                print(f"  {done}/{len(targets)}  {time.time()-t0:.0f}s  "
                      f"mean_edges/list={sum(edges)/max(1,len(edges)):.1f}", flush=True)

    el, n = time.time() - t0, len(targets)
    linked = len(edges)
    print(f"[precache] DONE {linked}/{n} items linked in {el:.0f}s "
          f"({failed} failed — re-run to retry stragglers)")
    if linked:
        print(f"  edges: total={sum(edges)} mean/item={sum(edges)/linked:.1f} "
              f"zero-edge items={sum(1 for e in edges if e == 0)}")
    if in_tok or out_tok:
        cost = in_tok / 1e6 * 2.50 + out_tok / 1e6 * 10.0   # gpt-4o pricing
        print(f"  tokens: in={in_tok} out={out_tok}  ~${cost:.2f} at gpt-4o "
              f"({in_tok/n:.0f} in / {out_tok/n:.0f} out per item)")


if __name__ == "__main__":
    main()
