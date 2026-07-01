"""compare — stitch per-system result JSONs into the head-to-head table.

Each input is a run_headtohead.py --out file. Baselines are listed first and
memsom last, so the table reads as the punchline: everyone else's poison lands
and can't be contained; memsom's is provably contained (laundering 0.00) and
gateable to ~0 ASR with clean utility intact.

Usage:
  python compare.py result1.json result2.json ...
  python compare.py --dir C:\\Users\\you\\h2h
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*", help="run_headtohead result JSON files")
    ap.add_argument("--dir", default=None, help="dir of *.json results")
    ap.add_argument("--perf-dir", default=None, help="dir of perf_per_write *.json")
    args = ap.parse_args()

    paths = list(args.results)
    if args.dir:
        paths += glob.glob(os.path.join(args.dir, "*.json"))
    if not paths:
        ap.error("no result files given")

    # perf map: system -> {tokens/write, latency/write} from perf_per_write.py
    perf = {}
    if args.perf_dir:
        for pp in glob.glob(os.path.join(args.perf_dir, "*.json")):
            d = json.loads(open(pp, encoding="utf-8").read())
            if "llm_tokens_per_write" in d:
                perf[d["system"]] = d

    rows = []
    for p in sorted(set(paths)):
        d = json.loads(open(p, encoding="utf-8").read())
        c, po = d["clean"], d["poisoned"]
        rows.append({
            "system": d["system"],
            "has_prov": d.get("has_provenance", False),
            "clean_util": c["utility"],
            "pois_util": po["utility"],
            "asr": po["asr"],
            "cite_asr": po["citation_asr"],
            "laundering": po["laundering_rate"] if d.get("has_provenance") else "n/a",
            "gated_asr": po.get("gated_asr", "n/a"),
            "gated_clean_util": c.get("gated_utility", "n/a"),
            "tok_per_write": perf.get(d["system"], {}).get("llm_tokens_per_write", "n/a"),
            "lat_per_write_ms": perf.get(d["system"], {}).get("latency_per_write_ms", "n/a"),
        })

    # baselines first, memsom (the punchline) last
    rows.sort(key=lambda r: (r["has_prov"], r["system"]))

    hdr = ["system", "clean util", "pois util", "ASR", "cite_ASR",
           "laundering", "gated ASR", "gated clean util",
           "tok/write", "lat/write ms"]
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join([
            r["system"], _fmt(r["clean_util"]), _fmt(r["pois_util"]),
            _fmt(r["asr"]), _fmt(r["cite_asr"]), _fmt(r["laundering"]),
            _fmt(r["gated_asr"]), _fmt(r["gated_clean_util"]),
            _fmt(r["tok_per_write"]), _fmt(r["lat_per_write_ms"]),
        ]) + " |")
    print("\nread: ASR/cite_ASR = poison reached the answer (higher = worse).")
    print("      laundering 0.00 = integrity floor held (memsom only has the property).")
    print("      gated ASR = poison still landing AFTER the action gate (memsom).")
    print("      tok/write = generative-LLM tokens per memory write (memsom = 0).\n")


if __name__ == "__main__":
    main()
