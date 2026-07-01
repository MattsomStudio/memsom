"""perf_per_write — the tokens/latency-per-write column.

Measures the COST OF WRITING a memory for one system: generative-LLM tokens
consumed + wall latency, averaged over K writes. Run uncontended (single process,
sequential) so latency is clean -- token counts are contention-independent anyway.

The contrast this column makes concrete:
  memsom / memsom-bm25 / RAG : 0 LLM tokens per write (deterministic / embeddings)
  Mem0 / Zep                 : thousands of tokens per write (per-write extraction)
memsom-bm25 is the zero-on-both anchor (no LLM, no embedder).

Usage:
  python perf_per_write.py --system memsom --repo C:\\Users\\you\\memsom ^
    --run-root C:\\Users\\you\\perf\\memsom --dataset ...oracle.json --k 40 --out ...
  python perf_per_write.py --system memsom --no-embed ...   (memsom-bm25)
  python perf_per_write.py --system rag ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from run_headtohead import make_adapter
from dataset import load_fixture, from_longmemeval


def _gather_texts(dataset, k, max_evidence):
    """K short memory texts to write -- real evidence turns if a dataset is given."""
    texts = []
    if dataset:
        items, _ = from_longmemeval(dataset, max_items=k, max_evidence=max_evidence)
        for it in items:
            texts += [e["text"] for e in it["evidence"]]
    else:
        for it in load_fixture():
            texts += [e["text"] for e in it["evidence"]]
    # pad by cycling if we came up short
    if not texts:
        texts = ["sample memory text for write-cost measurement."]
    while len(texts) < k:
        texts += texts
    return texts[:k]


def main() -> int:
    ap = argparse.ArgumentParser(description="per-write token/latency cost")
    ap.add_argument("--system", required=True,
                    choices=["memsom", "rag", "mem0", "superlocal", "zep"])
    ap.add_argument("--repo", default="")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--max-evidence", type=int, default=3)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--k", type=int, default=40, help="number of writes to time")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = json.loads(args.config)
    if args.no_embed:
        config["no_embed"] = True
    label = "memsom-bm25" if (args.system == "memsom" and args.no_embed) else args.system

    texts = _gather_texts(args.dataset, args.k, args.max_evidence)
    adapter = make_adapter(args.system, args.repo, config)
    adapter.reset(str(Path(args.run_root) / "perf_item"))

    tok0 = getattr(adapter, "llm_tokens", 0)
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        adapter.add(t, "user", False)
        lat.append(time.perf_counter() - t0)
    tokens = getattr(adapter, "llm_tokens", 0) - tok0

    lat_mean = statistics.mean(lat)
    out = {
        "system": label,
        "k": len(texts),
        "latency_per_write_s": round(lat_mean, 4),
        "latency_per_write_ms": round(lat_mean * 1000, 1),
        "latency_p50_ms": round(statistics.median(lat) * 1000, 1),
        "llm_tokens_total": tokens,
        "llm_tokens_per_write": round(tokens / max(1, len(texts)), 1),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[perf] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
