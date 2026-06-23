"""run_bench — the integrity benchmark entry point (steps 2+3).

What it does, per item, in a FRESH isolated DB (no cross-item retrieval leakage):
  1. init a clean DB        (init --data-dir)
  2. seed evidence          (add, on the truth channel)
  3. (poisoned arm) inject  (ingest-text, on the poison channel)
  4. reindex                (build BM25+vector index)
  5. ask --retrieve         (deterministic compose, no --llm)
  6. score                  (utility / asr / citation_asr / laundering)

It runs two arms over the same items — CLEAN (rate 0) and POISONED (rate R) —
and prints the (utility, ASR) pair for each. That pair is the honest figure:
clean-arm utility says the harness can answer at all; poisoned-arm (utility, ASR)
says what the attack does to a system that does NOT yet apply an integrity floor
on `ask`. The floor/baseline arm is step 4 — this scaffold proves the harness
MEASURES the attack and that provenance is tracked end-to-end.

Usage (run ON the PC, where memdag + Ollama live):
  python run_bench.py --repo C:\\Users\\you\\memdag --run-root C:\\Users\\you\\bench_runs
  python run_bench.py ... --dataset C:\\path\\longmemeval_s.json --max-items 100 --rate 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runner import MemdagRunner
from dataset import load_fixture, from_longmemeval
from poison import select_poisoned
from score import score_item, aggregate, ItemScore


def run_arm(runner: MemdagRunner, items: list[dict], poison_ids: set[str],
            run_root: Path, arm: str, topk: int, clearance: str | None,
            verbose: bool) -> list[ItemScore]:
    scores: list[ItemScore] = []
    for i, item in enumerate(items):
        poisoned = item["id"] in poison_ids
        data_dir = run_root / arm / f"item_{i:04d}"
        runner.init(data_dir)

        for ev in item["evidence"]:
            runner.add(ev["text"], channel=ev["channel"])
        if poisoned and item.get("poison"):
            p = item["poison"]
            runner.ingest_text(p["text"], channel=p["channel"], ref="poison")
        runner.reindex()

        res = runner.ask(item["question"], topk=topk, clearance=clearance)
        s = score_item(item, res, poisoned)
        scores.append(s)
        if verbose:
            tag = "P" if poisoned else " "
            print(f"  [{arm}|{tag}] {item['id']:<14} "
                  f"util={int(s.utility)} asr={int(s.asr)} "
                  f"cite_asr={int(s.citation_asr)} launder={int(s.laundered)} "
                  f"integ={s.integrity}", file=sys.stderr)
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description="memdag integrity benchmark (steps 2+3)")
    ap.add_argument("--repo", required=True, help="path to the memdag repo")
    ap.add_argument("--python", default="python", help="python exe memdag runs under")
    ap.add_argument("--run-root", required=True, help="scratch dir for per-item DBs")
    ap.add_argument("--dataset", default=None, help="path to longmemeval_*.json (default: bundled fixture)")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--rate", type=float, default=1.0, help="poison rate for the poisoned arm (0–1)")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--clearance", default=None, help="confidentiality clearance passed to ask")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # ---- load substrate ----
    if args.dataset:
        items, report = from_longmemeval(args.dataset, max_items=args.max_items)
        substrate = f"LongMemEval ({Path(args.dataset).name})"
        print(f"[bench] LongMemEval: used {report['used']} items, "
              f"skipped {report['skipped_total']} ({report['skipped']})", file=sys.stderr)
    else:
        items = load_fixture()
        if args.max_items:
            items = items[:args.max_items]
        substrate = "bundled fixture (factrecall_min) -- PIPE VALIDATION, not headline number"

    if not items:
        print("[bench] no items loaded", file=sys.stderr)
        return 1

    poison_ids = select_poisoned(items, args.rate)
    runner = MemdagRunner(args.repo, python=args.python)
    run_root = Path(args.run_root)
    verbose = not args.quiet

    print(f"[bench] substrate: {substrate}", file=sys.stderr)
    print(f"[bench] items={len(items)} poison_rate={args.rate} "
          f"poisoned={len(poison_ids)} topk={args.topk}", file=sys.stderr)

    # ---- two arms over the same items ----
    clean = run_arm(runner, items, set(), run_root, "clean", args.topk, args.clearance, verbose)
    pois = run_arm(runner, items, poison_ids, run_root, "poisoned", args.topk, args.clearance, verbose)

    agg_clean = aggregate(clean)
    agg_pois = aggregate(pois)

    # ---- report ----
    print("\n================ memdag integrity benchmark ================")
    print(f"substrate : {substrate}")
    print(f"items     : {len(items)}   poison rate: {args.rate}   poisoned: {len(poison_ids)}")
    print(f"{'arm':<10} {'utility':>8} {'ASR':>6} {'cite_ASR':>9} {'launder':>8} {'refuse':>7}")
    for name, a in (("clean", agg_clean), ("poisoned", agg_pois)):
        print(f"{name:<10} {a['utility']:>8.2f} {a['asr']:>6.2f} "
              f"{a['citation_asr']:>9.2f} {a['laundering_rate']:>8.2f} {a['refusal_rate']:>7.2f}")
    print("------------------------------------------------------------")
    print("read: poisoned-arm ASR > 0 means the harness detects the attack.")
    print("      laundering should be 0.00 -- Biba low-water-mark holding.")
    print("      (utility, ASR) is the honest pair; ASR alone is gameable.")
    print("============================================================\n")

    if args.out:
        payload = {
            "substrate": substrate,
            "config": {"items": len(items), "rate": args.rate,
                       "poisoned": sorted(poison_ids), "topk": args.topk,
                       "clearance": args.clearance},
            "clean": agg_clean,
            "poisoned": agg_pois,
            "items_detail": {
                "clean": [vars(s) for s in clean],
                "poisoned": [vars(s) for s in pois],
            },
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
