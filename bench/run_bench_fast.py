"""run_bench_fast — same benchmark as run_bench.py, ~30x faster.

run_bench.py spawns a fresh Python interpreter for EVERY memdag command (init,
each add, reindex, ask). On Windows that cold-start dominates: ~30 spawns/item.

This driver imports memdag_cli ONCE and calls `memdag_cli.main(argv)` in-process
in a loop. Identical code paths, identical stdout contract, identical parser —
the interpreter + import cost is paid once for the whole run instead of per
command. Measures exactly what run_bench.py measures; only the plumbing differs.

Must run ON the box where memdag lives (it imports memdag_cli).

Usage:
  python run_bench_fast.py --repo C:\\Users\\you\\memdag ^
    --run-root C:\\Users\\you\\bench_runs_fast ^
    --dataset C:\\Users\\you\\lme_data\\longmemeval_oracle.json --rate 0.5 ^
    --out C:\\Users\\you\\bench_runs_fast\\lme_result.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dataset import load_fixture, from_longmemeval
from poison import select_poisoned
from score import score_item, aggregate, ItemScore
from runner import parse_ask

# memdag_cli is imported lazily in main() once --repo is known.
memdag_cli = None


def _call(argv: list[str]) -> str:
    """Invoke a memdag CLI command in-process, capturing stdout+stderr.

    memdag refusal paths sys.exit(1) AFTER printing the message, so we swallow
    SystemExit and keep the captured text for the parser. Any other exception is
    recorded in the captured text so a single bad item degrades to a refusal
    rather than killing the whole run.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            memdag_cli.main(argv)
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001 - we WANT to contain per-item blowups
            print(f"[driver-error] {type(e).__name__}: {e}", file=sys.stderr)
    return buf.getvalue()


def run_item(item: dict, poisoned: bool, item_dir: Path, topk: int,
             clearance: str | None):
    # fresh, isolated DB. cmd_init reuses an existing dir ("schema up to date"),
    # which would carry stale nodes across items -> wipe first.
    shutil.rmtree(item_dir, ignore_errors=True)
    _call(["init", "--data-dir", str(item_dir)])
    # init ignores MEMDAG_DB; everything after resolves the DB from it.
    os.environ["MEMDAG_DB"] = str(item_dir / "memdag.db")

    for ev in item["evidence"]:
        _call(["add", ev["text"], "--channel", ev["channel"]])
    if poisoned and item.get("poison"):
        p = item["poison"]
        _call(["ingest-text", p["text"], "--channel", p["channel"], "--ref", "poison"])
    _call(["reindex"])

    ask_argv = ["ask", item["question"], "--retrieve", "--topk", str(topk)]
    if clearance:
        ask_argv += ["--clearance", clearance]
    out = _call(ask_argv)
    res = parse_ask(out)
    return score_item(item, res, poisoned)


def run_arm(items, poison_ids, run_root: Path, arm: str, topk, clearance, verbose):
    scores: list[ItemScore] = []
    for i, item in enumerate(items):
        poisoned = item["id"] in poison_ids
        s = run_item(item, poisoned, run_root / arm / f"item_{i:04d}", topk, clearance)
        scores.append(s)
        if verbose:
            tag = "P" if poisoned else " "
            print(f"  [{arm}|{tag}] {item['id']:<16} util={int(s.utility)} "
                  f"asr={int(s.asr)} cite={int(s.citation_asr)} "
                  f"launder={int(s.laundered)} integ={s.integrity}", file=sys.stderr)
    return scores


def main() -> int:
    global memdag_cli
    ap = argparse.ArgumentParser(description="memdag integrity benchmark (fast, in-process)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--clearance", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import memdag_cli as _cli  # noqa: E402
    memdag_cli = _cli

    if args.dataset:
        items, report = from_longmemeval(args.dataset, max_items=args.max_items)
        substrate = f"LongMemEval ({Path(args.dataset).name})"
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
    verbose = not args.quiet
    print(f"[bench] substrate: {substrate}", file=sys.stderr)
    print(f"[bench] items={len(items)} rate={args.rate} poisoned={len(poison_ids)} "
          f"topk={args.topk}", file=sys.stderr)

    t0 = time.time()
    clean = run_arm(items, set(), run_root, "clean", args.topk, args.clearance, verbose)
    pois = run_arm(items, poison_ids, run_root, "poisoned", args.topk, args.clearance, verbose)
    elapsed = time.time() - t0

    agg_clean, agg_pois = aggregate(clean), aggregate(pois)
    n_runs = len(clean) + len(pois)

    print("\n================ memdag integrity benchmark (fast) ================")
    print(f"substrate : {substrate}")
    print(f"items     : {len(items)}   poison rate: {args.rate}   poisoned: {len(poison_ids)}")
    print(f"time      : {elapsed:.0f}s for {n_runs} item-runs ({elapsed/max(1,n_runs):.2f}s/run)")
    print(f"{'arm':<10} {'utility':>8} {'ASR':>6} {'cite_ASR':>9} {'launder':>8} {'refuse':>7}")
    for name, a in (("clean", agg_clean), ("poisoned", agg_pois)):
        print(f"{name:<10} {a['utility']:>8.2f} {a['asr']:>6.2f} "
              f"{a['citation_asr']:>9.2f} {a['laundering_rate']:>8.2f} {a['refusal_rate']:>7.2f}")
    print("-------------------------------------------------------------------")
    print("read: poisoned ASR>0 = harness detects the attack. launder=0.00 = Biba holds.")
    print("      (utility, ASR) is the honest pair; ASR alone is gameable.")
    print("===================================================================\n")

    if args.out:
        payload = {
            "substrate": substrate,
            "config": {"items": len(items), "rate": args.rate,
                       "poisoned": sorted(poison_ids), "topk": args.topk,
                       "clearance": args.clearance, "elapsed_s": round(elapsed, 1)},
            "clean": agg_clean, "poisoned": agg_pois,
            "items_detail": {"clean": [vars(s) for s in clean],
                             "poisoned": [vars(s) for s in pois]},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
