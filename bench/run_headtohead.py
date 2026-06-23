"""run_headtohead — run ONE memory system through the poison protocol.

Same clean+poisoned arms and same scorer as run_bench_par, but the system under
test is pluggable via --system. memdag/rag parallelize across a process pool;
LLM/service-bound systems (mem0/zep) should run --workers 1. Writes a per-system
result JSON; compare.py stitches them into the head-to-head table.

Usage:
  python run_headtohead.py --system memdag --repo C:\\Users\\you\\memdag ^
    --run-root C:\\Users\\you\\h2h\\memdag --dataset ...oracle.json ^
    --rate 0.5 --gated --workers 6 --out ...\\memdag.json
  python run_headtohead.py --system rag  --run-root ...\\rag  --dataset ... --rate 0.5 --workers 6 --out ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from dataset import load_fixture, from_longmemeval
from poison import select_poisoned
from score import score_item, aggregate, ItemScore
from judge import judge_correct, synthesize

_SCORE_FIELDS = ("item_id", "poisoned", "composed", "utility", "asr",
                 "citation_asr", "laundered", "integrity", "gate_allowed")

# per-worker adapter, built once in the initializer.
_ADAPTER = None
_HAS_PROV = False


def make_adapter(system: str, repo: str, config: dict):
    if system == "memdag":
        from adapters.memdag_adapter import MemdagAdapter
        return MemdagAdapter(repo, no_embed=config.get("no_embed", False))
    if system == "rag":
        from adapters.rag_adapter import RagAdapter
        return RagAdapter()
    if system == "mem0":
        from adapters.mem0_adapter import Mem0Adapter
        return Mem0Adapter(**config)
    if system == "superlocal":
        from adapters.superlocal_adapter import SuperLocalAdapter
        return SuperLocalAdapter(**config)
    if system == "zep":
        from adapters.zep_adapter import ZepAdapter
        return ZepAdapter(**config)
    raise ValueError(f"unknown system {system}")


def _init_worker(system, repo, config):
    global _ADAPTER, _HAS_PROV
    _ADAPTER = make_adapter(system, repo, config)
    _HAS_PROV = _ADAPTER.has_provenance


def _work(task: dict) -> dict:
    item, poisoned, gated = task["item"], task["poisoned"], task["gated"]
    try:
        _ADAPTER.reset(task["item_dir"])
        for ev in item["evidence"]:
            _ADAPTER.add(ev["text"], ev["channel"], ev.get("answer_bearing", False))
        if poisoned and item.get("poison"):
            p = item["poison"]
            _ADAPTER.add(p["text"], p["channel"], False)
        res = _ADAPTER.ask(item["question"], topk=task["topk"])
        s = score_item(item, res, poisoned)
        # LLM-judged utility (uniform across systems; fair to extractive stores
        # like Mem0 that rephrase). A FIXED synthesizer writes a real answer from
        # each system's retrieved memories, then the judge grades it -- realistic
        # utility without grading a raw concat. Integrity metrics (cite_ASR,
        # laundering) stay on res (the raw retrieval), unaffected by synthesis.
        if task.get("judge"):
            gold = "; ".join(item.get("gold_terms", []))
            mems = [c.text for c in res.citations]
            answer = synthesize(item["question"], mems,
                                model=task["judge_model"], url=task["judge_url"])
            s.utility = judge_correct(item["question"], gold, answer,
                                      model=task["judge_model"], url=task["judge_url"])
        if gated and _HAS_PROV:
            s.gate_allowed = _ADAPTER.gate(res.composed_node_id, require="user")
        return {"arm": task["arm"], **{k: getattr(s, k) for k in _SCORE_FIELDS}}
    except Exception as e:  # noqa: BLE001 - item-level resilience: never let one
        # bad item drop a whole system. Failed item scores as a non-composition
        # (counts against the system, honestly) and is logged loudly.
        print(f"[h2h] item {item['id']} FAILED ({type(e).__name__}: {e})", file=sys.stderr)
        fail = ItemScore(item_id=item["id"], poisoned=poisoned, composed=False,
                         utility=False, asr=False, citation_asr=False,
                         laundered=False, integrity=None, gate_allowed=None)
        return {"arm": task["arm"], **{k: getattr(fail, k) for k in _SCORE_FIELDS}}


def _to_scores(dicts):
    return [ItemScore(**{k: d[k] for k in _SCORE_FIELDS}) for d in dicts]


def main() -> int:
    ap = argparse.ArgumentParser(description="poison head-to-head, one system")
    ap.add_argument("--system", required=True,
                    choices=["memdag", "rag", "mem0", "superlocal", "zep"])
    ap.add_argument("--repo", default="", help="memdag repo (memdag system only)")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-evidence", type=int, default=3)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gated", action="store_true", help="apply memdag check-action gate")
    ap.add_argument("--no-embed", action="store_true",
                    help="memdag-bm25 ablation: force embedder offline (BM25-only)")
    ap.add_argument("--mem0-llm", default="ollama", choices=["ollama", "openai"],
                    help="mem0 extraction backend: ollama (local qwen, handicapped) or openai (gpt-4o-mini, at-best)")
    ap.add_argument("--judge", action="store_true",
                    help="LLM-judged utility (uniform across systems; fair to extractive stores)")
    ap.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--judge-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    ap.add_argument("--config", default="{}", help="JSON adapter config (mem0/zep/superlocal)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = json.loads(args.config)
    if args.no_embed:
        config["no_embed"] = True
    if args.mem0_llm == "openai":
        config["llm_provider"] = "openai"  # reads OPENAI_API_KEY from env
    # label distinguishes ablation/at-best variants in the table.
    if args.system == "memdag" and args.no_embed:
        system_label = "memdag-bm25"
    elif args.system == "mem0" and args.mem0_llm == "openai":
        system_label = "mem0-best"
    else:
        system_label = args.system

    if args.dataset:
        items, report = from_longmemeval(args.dataset, max_items=args.max_items,
                                         max_evidence=args.max_evidence)
        substrate = f"LongMemEval ({Path(args.dataset).name}, max_evidence={args.max_evidence})"
        print(f"[h2h] LongMemEval: used {report['used']}, skipped "
              f"{report['skipped_total']} ({report['skipped']})", file=sys.stderr)
    else:
        items = load_fixture()
        if args.max_items:
            items = items[:args.max_items]
        substrate = "bundled fixture (factrecall_min)"

    if not items:
        print("[h2h] no items", file=sys.stderr)
        return 1

    poison_ids = select_poisoned(items, args.rate)
    run_root = Path(args.run_root)
    tasks = []
    for arm, is_p in (("clean", lambda _id: False),
                      ("poisoned", lambda _id: _id in poison_ids)):
        for i, item in enumerate(items):
            tasks.append({"arm": arm, "item": item, "poisoned": is_p(item["id"]),
                          "item_dir": str(run_root / arm / f"item_{i:04d}"),
                          "topk": args.topk, "gated": args.gated,
                          "judge": args.judge, "judge_model": args.judge_model,
                          "judge_url": args.judge_url})

    print(f"[h2h] system={args.system} items={len(items)} rate={args.rate} "
          f"poisoned={len(poison_ids)} gated={args.gated} workers={args.workers} "
          f"runs={len(tasks)}", file=sys.stderr)

    t0 = time.time()
    results = []
    with mp.Pool(processes=args.workers, initializer=_init_worker,
                 initargs=(args.system, args.repo, config)) as pool:
        for n, d in enumerate(pool.imap_unordered(_work, tasks), 1):
            results.append(d)
            if n % 20 == 0 or n == len(tasks):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(tasks)} ({rate:.2f}s/run, eta {rate*(len(tasks)-n):.0f}s)",
                      file=sys.stderr)
    elapsed = time.time() - t0

    clean = _to_scores([d for d in results if d["arm"] == "clean"])
    pois = _to_scores([d for d in results if d["arm"] == "poisoned"])
    agg_clean, agg_pois = aggregate(clean), aggregate(pois)
    has_prov = args.system == "memdag"

    print(f"\n========== head-to-head: {system_label} ==========")
    print(f"substrate : {substrate}")
    print(f"items={len(items)} rate={args.rate} time={elapsed:.0f}s "
          f"({elapsed/max(1,len(tasks)):.2f}s/run)")
    laund = f"{agg_pois['laundering_rate']:.2f}" if has_prov else "n/a"
    print(f"clean    : utility={agg_clean['utility']:.2f}")
    print(f"poisoned : utility={agg_pois['utility']:.2f} ASR={agg_pois['asr']:.2f} "
          f"cite_ASR={agg_pois['citation_asr']:.2f} laundering={laund}")
    if args.gated and has_prov:
        print(f"gated    : gated_ASR={agg_pois.get('gated_asr', 0):.2f} "
              f"clean_gated_utility={agg_clean.get('gated_utility', 0):.2f}")
    print("=" * 42 + "\n")

    if args.out:
        payload = {
            "system": system_label, "has_provenance": has_prov,
            "substrate": substrate,
            "config": {"items": len(items), "rate": args.rate, "topk": args.topk,
                       "gated": args.gated, "workers": args.workers,
                       "max_evidence": args.max_evidence, "elapsed_s": round(elapsed, 1)},
            "clean": agg_clean, "poisoned": agg_pois,
            "items_detail": {"clean": [vars(s) for s in clean],
                             "poisoned": [vars(s) for s in pois]},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[h2h] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
