"""consumer_eval — same real-corpus questions, same retrieved context, same judge;
the ONLY variable is the consumer that writes the answer. Proves the audit thesis:
the recall system's retrieval is fine; the benchmark's low utility was the weak
local synthesizer, which the real system replaces with Claude.

  --mode weak   : qwen2.5:7b answers from each item's retrieved context, then judge
  --mode judge  : judge an externally-produced answers file (Claude's answers)
Both use the SAME qwen2.5:7b judge as the original 0.66 benchmark, so utilities are
directly comparable.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import re  # noqa: E402
from judge import _chat, synthesize, judge_correct, _DEFAULT_MODEL, _DEFAULT_URL  # noqa: E402


def synth_with(question, context, model, url):
    """Synthesize an answer with an arbitrary model; strip qwen3 <think>…</think>."""
    block = "\n".join(f"- {m}" for m in context)
    prompt = ("Answer the question using ONLY the memories below. Be concise. If they "
              "conflict, prefer the most recent.\n\n"
              f"Question: {question}\n\nMemories:\n{block}\n\nAnswer:")
    try:
        out = _chat(prompt, model, url, timeout=240)
    except Exception:
        return ""
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="qa_real.json (questions + gold + context)")
    ap.add_argument("--mode", choices=["weak", "judge", "synth"], required=True)
    ap.add_argument("--synth-model", default=_DEFAULT_MODEL, help="answerer model for --mode synth")
    ap.add_argument("--answers", help="answers json [{question,gold,answer}] for --mode judge")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    qa = json.loads(Path(args.qa).read_text(encoding="utf-8"))

    if args.mode == "synth":
        answers = []
        for it in qa:
            ans = synth_with(it["question"], it["context"], args.synth_model, _DEFAULT_URL)
            answers.append({"question": it["question"], "gold": it["gold"], "answer": ans})
            print(f"  [{len(answers)}/{len(qa)}] synthesized ({len(ans)} chars)", file=sys.stderr)
        Path(args.out).write_text(json.dumps(answers, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[synth] wrote {len(answers)} answers via {args.synth_model} -> {args.out}", file=sys.stderr)
        return 0

    rows = []
    if args.mode == "weak":
        for it in qa:
            ans = synthesize(it["question"], it["context"], model=_DEFAULT_MODEL, url=_DEFAULT_URL)
            ok = bool(judge_correct(it["question"], it["gold"], ans, model=_DEFAULT_MODEL, url=_DEFAULT_URL))
            rows.append({"question": it["question"], "gold": it["gold"], "answer": ans,
                         "ok": ok, "source_in_topk": it.get("source_in_topk")})
            print(f"  [{len(rows)}/{len(qa)}] ok={ok}", file=sys.stderr)
        label = "weak consumer (qwen2.5:7b)"
    else:
        ans_map = {a["question"]: a for a in json.loads(Path(args.answers).read_text(encoding="utf-8"))}
        for it in qa:
            a = ans_map.get(it["question"])
            ans = (a or {}).get("answer", "")
            ok = bool(judge_correct(it["question"], it["gold"], ans, model=_DEFAULT_MODEL, url=_DEFAULT_URL))
            rows.append({"question": it["question"], "gold": it["gold"], "answer": ans,
                         "ok": ok, "source_in_topk": it.get("source_in_topk")})
            print(f"  [{len(rows)}/{len(qa)}] ok={ok}", file=sys.stderr)
        label = "Claude consumer"

    util = sum(1 for r in rows if r["ok"]) / len(rows) if rows else 0.0
    # also report utility restricted to items where retrieval actually surfaced the source
    ret = [r for r in rows if r.get("source_in_topk")]
    util_ret = sum(1 for r in ret if r["ok"]) / len(ret) if ret else 0.0
    print("\n" + "=" * 56)
    print(f"  CONSUMER EVAL — {label}   (real corpus, n={len(rows)})")
    print("=" * 56)
    print(f"  utility (all):              {util:.4f}")
    print(f"  utility (source retrieved): {util_ret:.4f}  (n={len(ret)})")
    print("=" * 56)
    if args.out:
        Path(args.out).write_text(json.dumps({"label": label, "utility": util,
                                              "utility_source_retrieved": util_ret,
                                              "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
