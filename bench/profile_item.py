"""profile_item — split the per-item wall-clock into phases.

Answers the only question that matters before optimizing: where do the ~8s/item
go? init(migrations) / adds / ingest / reindex(embeds) / ask(embed+retrieve).
CPU-bound phases (rmtree, init) are reliable even under GPU contention; the
Ollama-bound phases (reindex, ask) read high while the main run is live -- look
at the RATIO, not the absolute, for those.

Usage:
  python profile_item.py --repo C:\\Users\\you\\memdag ^
    --run-root C:\\Users\\you\\bench_profile ^
    --dataset C:\\Users\\you\\lme_data\\longmemeval_oracle.json --n 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
import time
from pathlib import Path

from dataset import from_longmemeval

memdag_cli = None


def _call(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            memdag_cli.main(argv)
        except SystemExit:
            pass
    return buf.getvalue()


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def profile_item(item, item_dir: Path, topk=8):
    ph = {}
    ph["rmtree"] = _timed(lambda: shutil.rmtree(item_dir, ignore_errors=True))
    ph["init"] = _timed(lambda: _call(["init", "--data-dir", str(item_dir)]))
    os.environ["MEMDAG_DB"] = str(item_dir / "memdag.db")

    def do_adds():
        for ev in item["evidence"]:
            _call(["add", ev["text"], "--channel", ev["channel"]])
    ph["adds"] = _timed(do_adds)
    ph["n_adds"] = len(item["evidence"])

    p = item["poison"]
    ph["ingest"] = _timed(lambda: _call(["ingest-text", p["text"], "--channel", p["channel"], "--ref", "poison"]))
    ph["reindex"] = _timed(lambda: _call(["reindex"]))
    ph["ask"] = _timed(lambda: _call(["ask", item["question"], "--retrieve", "--topk", str(topk)]))
    return ph


def main():
    global memdag_cli
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    import memdag_cli as _cli
    memdag_cli = _cli

    items, _ = from_longmemeval(args.dataset, max_items=args.n)
    run_root = Path(args.run_root)

    rows = []
    for i, item in enumerate(items):
        ph = profile_item(item, run_root / f"item_{i:04d}")
        rows.append(ph)
        print(f"item {i}: " + "  ".join(
            f"{k}={ph[k]:.2f}s" for k in ("rmtree", "init", "adds", "ingest", "reindex", "ask")), file=sys.stderr)

    print("\n==== phase breakdown (mean over {} items) ====".format(len(rows)))
    keys = ["rmtree", "init", "adds", "ingest", "reindex", "ask"]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    total = sum(means.values())
    for k in keys:
        print(f"  {k:<8} {means[k]:6.2f}s  ({100*means[k]/total:4.1f}%)")
    print(f"  {'TOTAL':<8} {total:6.2f}s")
    print(f"\n  (adds = {rows[0]['n_adds']} nodes; reindex+ask are Ollama-bound and inflated "
          f"while the main run is live)")


if __name__ == "__main__":
    main()
