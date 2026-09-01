"""run_arm — run ONE arm of the three-arm LongMemEval judged-accuracy benchmark.

This is a per-arm SUBPROCESS entry, invoked once per arm by run_three_arm.py.
It must be a separate process because arm1 imports memsom from the live repo
while arms 2/3 import a DIFFERENT memsom package from the refactor tree -- a
single Python process cannot hold two memsom packages.

Pipeline per item (all local + $0 except the correctness judge):
  ingest evidence -> memsom ask (deterministic retrieve/compose)
    -> synthesize an answer with LOCAL qwen (free), date-injected
    -> JUDGE correctness (OpenAI gpt-4o serial, or local qwen fallback)
    -> score, aggregate.

Only the correctness JUDGE call hits OpenAI (~1 call/question). The synthesizer
stays local. Judge calls are SERIAL with retry/backoff (openai_judge._chat) so a
Tier-1 rate limit can't throttle us into empty answers.

Isolation: MEMDAG_HOME is pinned to a throwaway root for the WHOLE process (belt),
and each item gets its own MEMDAG_DB under that root (suspenders).

Usage (driven by run_three_arm.py; not typically run by hand):
  python run_arm.py --arm pre_refactor --repo ~\\memsom \
    --config "{}" --dataset ...oracle.json --memdag-home C:\\...\\lme_bench_run\\pre_refactor \
    --judge-provider openai --max-items 3 --out ...\\pre_refactor.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ensure bench/ is importable regardless of cwd
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _build_dated_memories(citations, evidence) -> list[str]:
    """Prefix each retrieved citation with its session date [YYYY/MM/DD].

    The date is looked up from the ingested evidence by matching the citation
    text back to the evidence turn it came from (exact, then substring). Turns
    with no known date are passed through undated -- honest, no fabrication.
    """
    date_by_text: dict[str, str] = {}
    for ev in evidence:
        d = ev.get("date") or ""
        if d:
            date_by_text[ev["text"].strip().lower()] = d
    out = []
    for c in citations:
        ct = c.text.strip()
        low = ct.lower()
        date = date_by_text.get(low, "")
        if not date:
            for etext, d in date_by_text.items():
                if low and (low in etext or etext in low):
                    date = d
                    break
        out.append(f"[{date}] {ct}" if date else ct)
    return out


def _judge_openai(question: str, gold: str, answer: str) -> tuple[bool, str]:
    """Serial OpenAI judge. Returns (correct, status).
    status in {ok, empty, judge_error}. Reuses openai_judge's exact prompt and
    its _chat (which already has 429/5xx backoff, serial by construction)."""
    import openai_judge
    if not answer or not answer.strip():
        return False, "empty"
    prompt = (f"Question: {question}\n"
              f"Reference (correct) answer: {gold}\n"
              f"Candidate answer: {answer}\n\n"
              "Does the candidate answer contain the correct answer to the question, "
              "consistent with the reference answer? Ignore extra or contradictory "
              "text -- answer YES if the correct information is present, NO if it is "
              "absent or wrong. Reply with exactly one word: YES or NO.")
    try:
        text = openai_judge._chat(prompt).upper()
    except Exception:
        return False, "judge_error"
    for tok in text.replace("\n", " ").split():
        t = tok.strip(".,:;!*\"'")
        if t.startswith("YES"):
            return True, "ok"
        if t.startswith("NO"):
            return False, "ok"
    return text.startswith("Y"), "ok"


def _judge_local(question: str, gold: str, answer: str,
                 model: str, url: str) -> tuple[bool, str]:
    import judge as local_judge
    if not answer or not answer.strip():
        return False, "empty"
    try:
        ok = local_judge.judge_correct(question, gold, answer, model=model, url=url)
        return bool(ok), "ok"
    except Exception:
        return False, "judge_error"


def main() -> int:
    ap = argparse.ArgumentParser(description="one arm of the three-arm LME benchmark")
    ap.add_argument("--arm", required=True, help="arm label, e.g. pre_refactor")
    ap.add_argument("--repo", required=True, help="memsom repo for THIS arm")
    ap.add_argument("--config", default="{}", help="JSON adapter config (graph/hops)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--memdag-home", required=True,
                    help="throwaway MEMDAG_HOME + per-item store root for this arm")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "local"])
    ap.add_argument("--synth-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--synth-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--judge-model", default="qwen2.5:7b-instruct",
                    help="local judge model (ignored for --judge-provider openai)")
    ap.add_argument("--judge-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--empty-threshold", type=float, default=0.05,
                    help="empty/error rate above which the arm is flagged unreliable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # --- belt: pin MEMDAG_HOME to a throwaway root for the WHOLE process so a
    # missing-env fallback can never touch the real ~/.memdag.
    home = Path(args.memdag_home)
    home.mkdir(parents=True, exist_ok=True)
    os.environ["MEMDAG_HOME"] = str(home)

    config = json.loads(args.config)

    # --- judge provider selection (key is loaded into env HERE, in the judge
    # subprocess only, and never logged).
    judge_provider = args.judge_provider
    judge_label = ""
    if judge_provider == "openai":
        from _keyfile import load_openai_key
        if load_openai_key():
            import openai_judge
            judge_label = f"openai:{os.environ.get('OAI_MODEL', 'gpt-4o')}"
            print(f"[arm {args.arm}] judge=OpenAI ({os.environ.get('OAI_MODEL','gpt-4o')}), "
                  "key loaded from disk into env (not logged)", file=sys.stderr)
        else:
            judge_provider = "local"
            judge_label = f"local:{args.judge_model}"
            print(f"[arm {args.arm}] OpenAI key file absent -> falling back to LOCAL "
                  f"qwen judge ({args.judge_model})", file=sys.stderr)
    else:
        judge_label = f"local:{args.judge_model}"

    # --- load items (all types, no poison, date-carrying evidence)
    from dataset import from_longmemeval_all
    from adapters.memsom_adapter import MemsomAdapter
    import judge as local_synth   # synthesizer is ALWAYS local (free)

    items, report = from_longmemeval_all(args.dataset, max_items=args.max_items,
                                         max_evidence=args.max_evidence)
    print(f"[arm {args.arm}] items={report['used']} by_type={report['used_by_type']} "
          f"skipped={report['skipped_total']} ({report['skipped']})", file=sys.stderr)
    if not items:
        print(f"[arm {args.arm}] no items", file=sys.stderr)
        return 1

    adapter = MemsomAdapter(args.repo, graph=config.get("graph", False),
                            hops=config.get("hops", 2))

    store_root = home / "items"
    correct = 0
    empty_answers = 0
    judge_errors = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # type -> [n, correct]
    detail = []

    t0 = time.time()
    for i, item in enumerate(items):
        item_dir = str(store_root / f"item_{i:04d}")
        status = "ok"
        is_correct = False
        answer = ""
        try:
            adapter.reset(item_dir)
            for ev in item["evidence"]:
                adapter.add(ev["text"], ev["channel"], ev.get("answer_bearing", False))
            res = adapter.ask(item["question"], topk=args.topk)

            mems = _build_dated_memories(res.citations, item["evidence"])
            # date-injection: today's date line threads into the synth question
            qdate = item.get("question_date", "")
            synth_q = (f"Today's date is {qdate}.\n\n{item['question']}"
                       if qdate else item["question"])
            answer = local_synth.synthesize(synth_q, mems,
                                             model=args.synth_model, url=args.synth_url)

            if judge_provider == "openai":
                is_correct, status = _judge_openai(item["question"], item["gold"], answer)
            else:
                is_correct, status = _judge_local(item["question"], item["gold"], answer,
                                                  model=args.judge_model, url=args.judge_url)
        except Exception as e:  # noqa: BLE001 - one bad item never drops the arm
            print(f"[arm {args.arm}] item {item['id']} FAILED "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            status = "judge_error"

        if status == "empty":
            empty_answers += 1
        elif status == "judge_error":
            judge_errors += 1
        if is_correct:
            correct += 1
        qt = item.get("question_type", "?")
        by_type[qt][0] += 1
        by_type[qt][1] += int(is_correct)
        detail.append({"id": item["id"], "type": qt, "correct": is_correct,
                       "status": status, "answer_len": len(answer or "")})

        n = i + 1
        if n % 25 == 0 or n == len(items):
            rate = (time.time() - t0) / n
            print(f"  [{args.arm}] {n}/{len(items)} acc={correct/n:.3f} "
                  f"empty={empty_answers} jerr={judge_errors} "
                  f"({rate:.2f}s/q, eta {rate*(len(items)-n):.0f}s)", file=sys.stderr)

    elapsed = time.time() - t0
    n = len(items)
    empty_rate = (empty_answers + judge_errors) / n if n else 0.0
    unreliable = empty_rate > args.empty_threshold

    payload = {
        "arm": args.arm,
        "repo": args.repo,
        "config": config,
        "judge_provider": judge_label,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "empty_answers": empty_answers,
        "judge_errors": judge_errors,
        "empty_rate": empty_rate,
        "empty_threshold": args.empty_threshold,
        "unreliable": unreliable,
        "wall_time_s": round(elapsed, 1),
        "by_type": {t: {"n": v[0], "correct": v[1],
                        "accuracy": v[1] / v[0] if v[0] else 0.0}
                    for t, v in sorted(by_type.items())},
        "coverage": report,
        "items": detail,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flag = "  ***UNRELIABLE (judge throttled?)***" if unreliable else ""
    print(f"\n[arm {args.arm}] DONE n={n} acc={correct/n:.4f} "
          f"empty_rate={empty_rate:.3f}{flag}  ({elapsed:.0f}s)", file=sys.stderr)
    print(f"[arm {args.arm}] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
