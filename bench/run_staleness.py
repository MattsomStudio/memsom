r"""run_staleness — the staleness head-to-head orchestrator.

Per item, per system, on an isolated store:
  1. seed v1 under a stable source_ref        (memsom: ingest-text --ref)
  2. ask  -> derive answer A1                  (capture its node id)
  3. update the source to v2 (same ref)        (memsom: supersede + cascade;
                                                 RAG/Mem0: just add the new version)
  4. attribution: is A1 now flagged affected?  (memsom: exact; others: n/a)
  5. ask  -> A2; measure fresh/stale serving + whether staleness is flagged

Headline = attribution (memsom answers exactly; no-provenance systems are n/a,
never scored 0). Serving is the honest secondary (memsom and a good RAG often
TIE on fresh_serve — only memsom can attribute + flag).

Run (env-safe core, no mem0/API):
  python run_staleness.py --repo C:\Users\you\memsom --run-root C:\Users\you\stale_runs \
      --systems memsom,memsom-bm25,rag --out C:\Users\you\stale_runs\fixture.json
LongMemEval knowledge-update scale:
  ... --dataset C:\path\longmemeval_s.json --max-items 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import dataset_stale
import score_stale
from score_stale import score_stale_item


def _make_adapter(name: str, repo: str):
    from adapters.memsom_adapter import MemsomAdapter
    from adapters.rag_adapter import RagAdapter
    if name == "memsom":
        return MemsomAdapter(repo=repo, no_embed=False)
    if name == "memsom-fresh":
        return MemsomAdapter(repo=repo, no_embed=False, fresh_only=True)
    if name == "memsom-prefer-fresh":
        return MemsomAdapter(repo=repo, no_embed=False, prefer_fresh=True)
    if name == "memsom-bm25":
        return MemsomAdapter(repo=repo, no_embed=True)
    if name == "memsom-bm25-fresh":
        return MemsomAdapter(repo=repo, no_embed=True, fresh_only=True)
    if name == "memsom-bm25-prefer":
        return MemsomAdapter(repo=repo, no_embed=True, prefer_fresh=True)
    if name == "rag":
        return RagAdapter()
    if name == "mem0":
        from adapters.mem0_adapter import Mem0Adapter  # env-fragile: imported lazily
        return Mem0Adapter()
    raise SystemExit(f"unknown system: {name}")


def run_system(name: str, items: list[dict], repo: str, run_root: str,
               haystack: bool = False, topk: int = 8) -> dict:
    adapter = _make_adapter(name, repo)
    scores = []
    for i, item in enumerate(items):
        item_dir = os.path.join(run_root, name, f"item_{i:04d}")
        adapter.reset(item_dir)
        # HAYSTACK: seed the item's other turns as noise FIRST, so retrieval must
        # find the right node among dozens. This is the pressure that separates
        # --prefer-fresh (substitute the buried fresh head) from --fresh-only
        # (exclude, which fails when the fresh node never made top-k).
        if haystack:
            for di, dtext in enumerate(item.get("distractors", [])):
                adapter.seed(dtext, item["channel"], f"{item['id']}:d{di}.md")
        adapter.seed(item["evidence_v1"], item["channel"], item.get("source_ref"))
        a1 = adapter.ask(item["question"], topk=topk)

        if item.get("kind") == "updated" and item.get("update_text"):
            adapter.update(item["source_ref"], item["update_text"], item["channel"])

        # attribution verdict on the PRE-update answer node (post-update state)
        affected = adapter.stale_attribution(a1.composed_node_id)

        a2 = adapter.ask(item["question"], topk=topk)
        flagged = "[STALE]" in (a2.raw or "")

        s = score_stale_item(item, a2, affected, flagged, adapter.has_provenance)
        scores.append(s)
        print(f"  [{name}] {item['id']:<14} kind={s.kind:<8} "
              f"affected={s.affected} fresh_clean={s.fresh_clean} "
              f"served_stale={s.served_stale} flagged={s.flagged}")
    return score_stale.aggregate(scores)


def _fmt(v):
    if v is None:
        return "  n/a"
    if isinstance(v, float):
        return f"{v:5.2f}"
    return str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_staleness")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--systems", default="memsom,memsom-fresh,memsom-prefer-fresh,rag",
                    help="comma list: memsom,memsom-fresh,memsom-prefer-fresh,memsom-bm25,rag,mem0")
    ap.add_argument("--dataset", default=None,
                    help="LongMemEval json (knowledge-update); default = bundled fixture")
    ap.add_argument("--fixture", default=None,
                    help="path to a curated staleness fixture json (e.g. staleness_curated.json)")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--haystack", action="store_true",
                    help="seed each item's other turns as distractors (retrieval pressure)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.dataset:
        items, report = dataset_stale.from_longmemeval_update(args.dataset, args.max_items)
        print(f"LongMemEval knowledge-update: used {report['used']}, "
              f"skipped {report['skipped_total']} {report['skipped']}")
    elif args.fixture:
        items = dataset_stale.load_stale_fixture(args.fixture)
        if args.max_items:
            items = items[: args.max_items]
    else:
        items = dataset_stale.load_stale_fixture()
        if args.max_items:
            items = items[: args.max_items]
        print(f"fixture: {len(items)} items "
              f"({sum(1 for it in items if it['kind']=='updated')} updated, "
              f"{sum(1 for it in items if it['kind']=='control')} control)")

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    results = {}
    for name in systems:
        print(f"\n=== {name} ===")
        results[name] = run_system(name, items, args.repo, args.run_root,
                                   haystack=args.haystack, topk=args.topk)

    cols = ["attribution_recall", "attribution_fpr",
            "fresh_present_rate", "fresh_clean_rate", "served_stale_rate",
            "flagged_rate", "refusal_rate"]
    print("\n" + "=" * 92)
    print(f"{'system':<14}" + "".join(f"{c.replace('_',' ')[:13]:>14}" for c in cols))
    print("-" * 92)
    for name in systems:
        r = results[name]
        print(f"{name:<14}" + "".join(f"{_fmt(r.get(c)):>14}" for c in cols))
    print("=" * 92)
    print("attribution n/a = system has no provenance edges to attribute over "
          "(NOT a score of 0).")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"systems": results, "n": len(items)}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
