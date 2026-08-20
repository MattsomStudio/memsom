"""run_three_arm — deterministic overnight three-arm LongMemEval S benchmark.

Runs three arms, each as a SEPARATE subprocess (arm1 imports the live memsom
package, arms 2/3 import the refactor tree's package -- a single process can't
hold both), then merges the per-arm result JSONs into ONE comparison file plus a
printed markdown summary.

Arms:
  arm1  pre_refactor      repo=~\\memsom                config={}                 (live baseline, graph off)
  arm2  refactor_graph_on repo=~\\memsom-refactor-work  config={"graph":true,"hops":2}
  arm3  refactor_flat     repo=~\\memsom-refactor-work  config={}

This driver is PLAIN CODE -- no LLM decides anything. It self-sequences: arm1
runs immediately; arms 2/3 wait behind a refactor-readiness gate (sentinel file
or a phase(11) commit), polling every 5 min up to an 8-hour ceiling. Launch it
before bed and it finishes the run when the refactor greens.

Launch the full overnight run:
  python ~\\memsom\\bench\\run_three_arm.py

Smoke (arm1 only, 3 items, local qwen judge -- no OpenAI needed, no refactor):
  python ~\\memsom\\bench\\run_three_arm.py --smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LIVE_REPO = str(Path.home() / "memsom")
REFACTOR_REPO = str(Path.home() / "memsom-refactor-work")
DEFAULT_DATASET = str(Path.home() / "lme_data" / "longmemeval_oracle.json")
RUN_ROOT = Path.home() / "lme_bench_run"
SENTINEL = RUN_ROOT / "REFACTOR_DONE"
RESULTS = RUN_ROOT / "three_arm_results.json"

ARMS = [
    {"arm": "pre_refactor", "repo": LIVE_REPO, "config": {}},
    {"arm": "refactor_graph_on", "repo": REFACTOR_REPO, "config": {"graph": True, "hops": 2}},
    {"arm": "refactor_flat", "repo": REFACTOR_REPO, "config": {}},
]

_HERE = Path(__file__).resolve().parent


def refactor_ready() -> tuple[bool, str]:
    """(a) sentinel file exists, OR (b) git log shows a phase(11) commit."""
    if SENTINEL.exists():
        return True, f"sentinel {SENTINEL}"
    try:
        out = subprocess.run(["git", "-C", REFACTOR_REPO, "log", "--oneline", "-40"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            if "phase(11)" in line:
                return True, f"phase(11) commit: {line.strip()}"
    except Exception as e:  # noqa: BLE001
        return False, f"git check failed: {type(e).__name__}"
    return False, "no sentinel, no phase(11) commit"


def run_arm(spec: dict, *, dataset: str, judge_provider: str,
            max_items: int | None, out: Path) -> dict | None:
    """Run one arm as a subprocess. Returns the parsed result JSON, or None."""
    home = RUN_ROOT / spec["arm"]
    argv = [sys.executable, str(_HERE / "run_arm.py"),
            "--arm", spec["arm"], "--repo", spec["repo"],
            "--config", json.dumps(spec["config"]),
            "--dataset", dataset, "--memdag-home", str(home),
            "--judge-provider", judge_provider, "--out", str(out)]
    if max_items is not None:
        argv += ["--max-items", str(max_items)]
    print(f"\n=== launching arm '{spec['arm']}' (repo={spec['repo']}, "
          f"config={spec['config']}) ===", flush=True)
    # stream child output straight through; the child never prints the key.
    proc = subprocess.run(argv, cwd=str(_HERE))
    if proc.returncode != 0 or not out.exists():
        print(f"!!! arm '{spec['arm']}' failed (rc={proc.returncode})", flush=True)
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def _fmt_pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x)


def print_summary(results: list[dict]) -> str:
    """Build + print the markdown comparison table. Returns the text."""
    lines = []
    lines.append("\n# Three-arm LongMemEval S — judged accuracy\n")
    hdr = ["arm", "judge", "n", "overall", "empty", "reliable", "wall"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in results:
        rel = "NO -- THROTTLED" if r.get("unreliable") else "yes"
        lines.append("| " + " | ".join([
            r["arm"], r.get("judge_provider", "?"), str(r["n"]),
            _fmt_pct(r["accuracy"]),
            f"{r['empty_answers']+r['judge_errors']} ({_fmt_pct(r['empty_rate'])})",
            rel, f"{r['wall_time_s']:.0f}s",
        ]) + " |")

    # by-type comparison
    all_types = []
    for r in results:
        for t in r.get("by_type", {}):
            if t not in all_types:
                all_types.append(t)
    all_types.sort()
    lines.append("\n## By question type (accuracy)\n")
    thdr = ["question type"] + [r["arm"] for r in results]
    lines.append("| " + " | ".join(thdr) + " |")
    lines.append("|" + "|".join(["---"] * len(thdr)) + "|")
    for t in all_types:
        row = [t]
        for r in results:
            bt = r.get("by_type", {}).get(t)
            row.append(f"{_fmt_pct(bt['accuracy'])} (n={bt['n']})" if bt else "-")
        lines.append("| " + " | ".join(row) + " |")

    for r in results:
        if r.get("unreliable"):
            lines.append(f"\n> **FLAG:** arm '{r['arm']}' empty-answer rate "
                         f"{_fmt_pct(r['empty_rate'])} exceeds "
                         f"{_fmt_pct(r['empty_threshold'])} -- result UNRELIABLE, "
                         "judge was likely throttled. Do not trust its score.")
    text = "\n".join(lines)
    print(text, flush=True)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="three-arm overnight LME benchmark")
    ap.add_argument("--smoke", action="store_true",
                    help="arm1 only, 3 items, local qwen judge, no refactor gate")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "local"])
    ap.add_argument("--poll-seconds", type=int, default=300, help="readiness poll interval")
    ap.add_argument("--poll-ceiling-hours", type=float, default=8.0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    if args.smoke:
        print("[three-arm] SMOKE: arm1 only, 3 items, local qwen judge", flush=True)
        out = RUN_ROOT / "smoke_pre_refactor.json"
        r = run_arm(ARMS[0], dataset=args.dataset, judge_provider="local",
                    max_items=3, out=out)
        if r is None:
            print("[three-arm] smoke FAILED", flush=True)
            return 1
        results.append(r)
        (RUN_ROOT / "smoke_results.json").write_text(
            json.dumps({"arms": results}, indent=2), encoding="utf-8")
        print_summary(results)
        print(f"\n[three-arm] smoke result -> {RUN_ROOT / 'smoke_results.json'}", flush=True)
        return 0

    # --- FULL RUN ---
    # arm1 runs immediately (live repo is always ready).
    out1 = RUN_ROOT / "pre_refactor.json"
    r1 = run_arm(ARMS[0], dataset=args.dataset, judge_provider=args.judge_provider,
                 max_items=args.max_items, out=out1)
    if r1 is not None:
        results.append(r1)

    # arms 2/3 gate on refactor readiness; poll up to the ceiling.
    ready, why = refactor_ready()
    deadline = time.time() + args.poll_ceiling_hours * 3600
    while not ready and time.time() < deadline:
        remain = (deadline - time.time()) / 60
        print(f"[three-arm] refactor NOT ready ({why}); polling again in "
              f"{args.poll_seconds//60} min (>{remain:.0f} min left)", flush=True)
        time.sleep(args.poll_seconds)
        ready, why = refactor_ready()

    if not ready:
        print(f"[three-arm] refactor never became ready within "
              f"{args.poll_ceiling_hours}h ({why}); arms 2/3 SKIPPED.", flush=True)
    else:
        print(f"[three-arm] refactor READY ({why}); running arms 2/3", flush=True)
        for spec in ARMS[1:]:
            out = RUN_ROOT / f"{spec['arm']}.json"
            r = run_arm(spec, dataset=args.dataset, judge_provider=args.judge_provider,
                        max_items=args.max_items, out=out)
            if r is not None:
                results.append(r)

    summary = print_summary(results)
    RESULTS.write_text(json.dumps({"arms": results, "summary_md": summary}, indent=2),
                       encoding="utf-8")
    print(f"\n[three-arm] merged results -> {RESULTS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
