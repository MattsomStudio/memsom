"""run_bench_par — parallel integrity benchmark (Path A speed-up).

Same measurement as run_bench_fast.py, run across a process POOL. Each item is
embedded by memdag's reindex via serial Ollama round-trips (~1s/node); those
round-trips leave the GPU mostly idle, so overlapping many items keeps Ollama
busy and collapses wall-clock.

Why PROCESSES, not threads: each item sets os.environ["MEMDAG_DB"] to its own
DB. That env var is process-global -- threads would clobber each other and write
into the wrong database. Separate processes each get their own os.environ, so the
isolation is automatic. (Same reason you isolate untrusted work into processes,
not shared threads.) On Windows multiprocessing uses spawn, so each worker
imports memdag_cli once in an initializer and reuses it across its items.

Usage:
  python run_bench_par.py --repo C:\\Users\\you\\memdag ^
    --run-root C:\\Users\\you\\bench_par ^
    --dataset C:\\Users\\you\\lme_data\\longmemeval_oracle.json ^
    --rate 0.5 --workers 6 --out C:\\Users\\you\\bench_par\\result.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

from dataset import load_fixture, from_longmemeval
from poison import select_poisoned
from score import score_item, aggregate, ItemScore
from runner import parse_ask

# per-worker handle to memdag_cli, set once in the pool initializer.
_CLI = None
_SCORE_FIELDS = ("item_id", "poisoned", "composed", "utility", "asr",
                 "citation_asr", "laundered", "integrity")


def _init_worker(repo: str):
    global _CLI
    sys.path.insert(0, repo)
    import memdag_cli  # noqa: E402
    _CLI = memdag_cli


def _call(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            _CLI.main(argv)
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001 - contain per-item blowups
            print(f"[driver-error] {type(e).__name__}: {e}")
    return buf.getvalue()


def _work(task: dict) -> dict:
    """Run one item end-to-end in this worker process. Returns score+arm dict."""
    item = task["item"]
    poisoned = task["poisoned"]
    item_dir = task["item_dir"]
    topk = task["topk"]
    clearance = task["clearance"]

    shutil.rmtree(item_dir, ignore_errors=True)
    _call(["init", "--data-dir", item_dir])
    os.environ["MEMDAG_DB"] = os.path.join(item_dir, "memdag.db")

    for ev in item["evidence"]:
        _call(["add", ev["text"], "--channel", ev["channel"]])
    if poisoned and item.get("poison"):
        p = item["poison"]
        _call(["ingest-text", p["text"], "--channel", p["channel"], "--ref", "poison"])
    _call(["reindex"])

    argv = ["ask", item["question"], "--retrieve", "--topk", str(topk)]
    if clearance:
        argv += ["--clearance", clearance]
    res = parse_ask(_call(argv))
    s = score_item(item, res, poisoned)
    return {"arm": task["arm"], **{k: getattr(s, k) for k in _SCORE_FIELDS}}


def _to_scores(dicts: list[dict]) -> list[ItemScore]:
    return [ItemScore(**{k: d[k] for k in _SCORE_FIELDS}) for d in dicts]


def main() -> int:
    ap = argparse.ArgumentParser(description="memdag integrity benchmark (parallel)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-evidence", type=int, default=3,
                    help="cap evidence turns/item (fewer nodes -> faster reindex)")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--clearance", default=None)
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset:
        items, report = from_longmemeval(args.dataset, max_items=args.max_items,
                                         max_evidence=args.max_evidence)
        substrate = f"LongMemEval ({Path(args.dataset).name}, max_evidence={args.max_evidence})"
        print(f"[bench] LongMemEval: used {report['used']}, skipped "
              f"{report['skipped_total']} ({report['skipped']})", file=sys.stderr)
    else:
        items = load_fixture()
        if args.max_items:
            items = items[:args.max_items]
        substrate = "bundled fixture (factrecall_min) -- PIPE VALIDATION, not headline number"

    if not items:
        print("[bench] no items loaded", file=sys.stderr)
        return 1

    poison_ids = select_poisoned(items, args.rate)
    run_root = Path(args.run_root)

    # one task per (arm, item); both arms share the pool so all cores stay busy.
    tasks = []
    for arm, get_pois in (("clean", lambda _id: False),
                          ("poisoned", lambda _id: _id in poison_ids)):
        for i, item in enumerate(items):
            tasks.append({
                "arm": arm, "item": item, "poisoned": get_pois(item["id"]),
                "item_dir": str(run_root / arm / f"item_{i:04d}"),
                "topk": args.topk, "clearance": args.clearance,
            })

    print(f"[bench] substrate: {substrate}", file=sys.stderr)
    print(f"[bench] items={len(items)} rate={args.rate} poisoned={len(poison_ids)} "
          f"topk={args.topk} workers={args.workers} runs={len(tasks)}", file=sys.stderr)

    t0 = time.time()
    results: list[dict] = []
    with mp.Pool(processes=args.workers, initializer=_init_worker, initargs=(args.repo,)) as pool:
        for n, d in enumerate(pool.imap_unordered(_work, tasks), 1):
            results.append(d)
            if n % 20 == 0 or n == len(tasks):
                rate = (time.time() - t0) / n
                eta = rate * (len(tasks) - n)
                print(f"  {n}/{len(tasks)} done ({rate:.2f}s/run, eta {eta:.0f}s)", file=sys.stderr)
    elapsed = time.time() - t0

    clean = _to_scores([d for d in results if d["arm"] == "clean"])
    pois = _to_scores([d for d in results if d["arm"] == "poisoned"])
    agg_clean, agg_pois = aggregate(clean), aggregate(pois)

    print("\n================ memdag integrity benchmark (parallel) ================")
    print(f"substrate : {substrate}")
    print(f"items     : {len(items)}   poison rate: {args.rate}   poisoned: {len(poison_ids)}")
    print(f"time      : {elapsed:.0f}s for {len(tasks)} runs, {args.workers} workers "
          f"({elapsed/max(1,len(tasks)):.2f}s/run wall, effective)")
    print(f"{'arm':<10} {'utility':>8} {'ASR':>6} {'cite_ASR':>9} {'launder':>8} {'refuse':>7}")
    for name, a in (("clean", agg_clean), ("poisoned", agg_pois)):
        print(f"{name:<10} {a['utility']:>8.2f} {a['asr']:>6.2f} "
              f"{a['citation_asr']:>9.2f} {a['laundering_rate']:>8.2f} {a['refusal_rate']:>7.2f}")
    print("-----------------------------------------------------------------------")
    print("read: poisoned ASR>0 = harness detects the attack. launder=0.00 = Biba holds.")
    print("=======================================================================\n")

    if args.out:
        payload = {
            "substrate": substrate,
            "config": {"items": len(items), "rate": args.rate,
                       "poisoned": sorted(poison_ids), "topk": args.topk,
                       "workers": args.workers, "max_evidence": args.max_evidence,
                       "clearance": args.clearance, "elapsed_s": round(elapsed, 1)},
            "clean": agg_clean, "poisoned": agg_pois,
            "items_detail": {"clean": [vars(s) for s in clean],
                             "poisoned": [vars(s) for s in pois]},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    mp.freeze_support()  # Windows spawn safety
    raise SystemExit(main())
