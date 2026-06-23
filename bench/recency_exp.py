"""recency_exp — isolate ONE lever for knowledge-update accuracy: does presenting
the retrieved memories date-tagged + ordered oldest->newest (with 'latest wins')
beat the plain unordered concat? Same retrieval, same judge, same items — only the
synthesis PRESENTATION changes. Reuses the cached embeddings so it's fast.

This is a legit recall-system improvement: the real /recall output already carries
per-hit dates; ordering+tagging them by recency is how a downstream answerer should
consume them. We re-measure the plain baseline in the same run for a fair delta.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import from_longmemeval                       # noqa: E402
from judge import _chat, synthesize, judge_correct, _DEFAULT_MODEL, _DEFAULT_URL  # noqa: E402

RRF_C = 60


def cos_mat(q, M):
    qn = q / (np.linalg.norm(q) or 1.0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn


def sparse_dot(a, b):
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def maxsim(q, d):
    return 0.0 if q.size == 0 or d.size == 0 else float((q @ d.T).max(axis=1).mean())


def rrf(scores):
    out = np.zeros(len(scores))
    for r, i in enumerate(np.argsort(-scores), 1):
        out[i] = 1.0 / (RRF_C + r)
    return out


def recency_synth(question, dated_mems, model, url):
    """dated_mems: list of (date, text) sorted oldest->newest.
    v2 — SCOPED recency: only same-fact conflicts defer to the latest date, and
    off-topic memories are to be ignored. v1's blunt 'latest always wins' let a
    recent but irrelevant pooled-haystack distractor hijack the answer (10 regressions)."""
    block = "\n".join(f"- [{d}] {t}" for d, t in dated_mems)
    prompt = (
        "Answer the question using ONLY the dated memories below (ordered oldest to "
        "newest). Some memories may be about unrelated topics — IGNORE those entirely. "
        "Only when two RELEVANT memories give different values for the exact thing the "
        "question asks does the one with the LATEST date win. Otherwise just answer from "
        "the relevant memory. Be concise.\n\n"
        f"Question: {question}\n\nMemories (oldest to newest):\n{block}\n\nAnswer:"
    )
    try:
        return _chat(prompt, model, url)
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--arm", default="rrf_equal")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="item offset (to reach non-knowledge-update types)")
    ap.add_argument("--per-item", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, _ = from_longmemeval(args.dataset, max_items=args.max_items, max_evidence=args.max_evidence)
    raw = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    raw_type = {e.get("question_id", ""): e.get("question_type", "?") for e in raw}
    # (qid, turn_content) -> session date, so each evidence turn knows its recency
    raw_date = {}
    for e in raw:
        qid = e.get("question_id", ""); dates = e.get("haystack_dates", []) or []
        for si, sess in enumerate(e.get("haystack_sessions", [])):
            d = dates[si] if si < len(dates) else ""
            for turn in sess:
                c = (turn.get("content") or "").strip()
                if c:
                    raw_date[(qid, c)] = d

    corpus_text, corpus_item, corpus_gold, corpus_date = [], [], [], []
    seen = {}
    for it in items:
        for ev in it["evidence"]:
            key = (it["id"], ev["text"])
            if key in seen:
                continue
            seen[key] = len(corpus_text)
            corpus_text.append(ev["text"]); corpus_item.append(it["id"])
            corpus_gold.append(bool(ev.get("answer_bearing")))
            corpus_date.append(raw_date.get((it["id"], ev["text"]), ""))
    N = len(corpus_text)
    gold_idx = {}
    for i, (iid, g) in enumerate(zip(corpus_item, corpus_gold)):
        if g:
            gold_idx.setdefault(iid, set()).add(i)
    items = [it for it in items if gold_idx.get(it["id"])]

    cache = Path(args.cache)
    # pickle is safe here: these caches are written by audit_optimize.py on this
    # same PC (self-produced local artifacts, never an untrusted source).
    with open(cache / f"emb_{N}.pkl", "rb") as f:
        blob = pickle.load(f)
    nom_doc, bge_d, bge_s, bge_c = blob["nom"], blob["bd"], blob["bs"], blob["bc"]
    with open(cache / f"qemb_{N}_{len(items)}.pkl", "rb") as f:
        qb = pickle.load(f)
    q_d, q_s, q_c = qb["d"], qb["s"], qb["c"]

    def ranking(qi):
        s_bd = cos_mat(q_d[qi], bge_d)
        s_sp = np.array([sparse_dot(q_s[qi], bge_s[i]) for i in range(N)])
        s_cb = np.array([maxsim(q_c[qi], bge_c[i]) for i in range(N)])
        return np.argsort(-(rrf(s_bd) + rrf(s_sp) + rrf(s_cb))).tolist()

    memo_file = cache / "recency_memo.json"
    memo = json.loads(memo_file.read_text()) if memo_file.exists() else {}

    def cached_judge(tag, question, mems_repr, gold, ans_fn):
        key = hashlib.sha1((tag + "|" + question + "|" + "|".join(mems_repr)).encode()).hexdigest()
        if key in memo:
            return memo[key]
        ans = ans_fn()
        ok = bool(judge_correct(question, gold, ans, model=_DEFAULT_MODEL, url=_DEFAULT_URL))
        memo[key] = ok
        return ok

    plain, recen = [], []
    per_item_f = open(args.per_item, "w", encoding="utf-8") if args.per_item else None
    window = list(range(args.start, min(args.start + args.n, len(items))))
    t0 = time.time()
    for c, qi in enumerate(window, 1):
        it = items[qi]
        rk = ranking(qi)[:args.topk]   # qi is the ABSOLUTE item index (query-embed aligned)
        gold = "; ".join(it["gold_terms"]); q = it["question"]
        texts = [corpus_text[i] for i in rk]
        dated = sorted([(corpus_date[i] or "", corpus_text[i]) for i in rk], key=lambda x: x[0])
        ok_p = cached_judge("plain", q, sorted(texts), gold,
                            lambda: synthesize(q, texts))
        ok_r = cached_judge("recency2", q, [f"[{d}] {t}" for d, t in dated], gold,
                            lambda: recency_synth(q, dated, _DEFAULT_MODEL, _DEFAULT_URL))
        plain.append(1.0 if ok_p else 0.0); recen.append(1.0 if ok_r else 0.0)
        if per_item_f:
            per_item_f.write(json.dumps({"id": it["id"], "qtype": raw_type.get(it["id"], "?"),
                                         "plain": bool(ok_p), "recency": bool(ok_r)}) + "\n")
            per_item_f.flush()
        if c % 10 == 0:
            memo_file.write_text(json.dumps(memo))
            print(f"   {c}/{len(window)} ({(time.time()-t0)/c:.1f}s/q)", file=sys.stderr)
    if per_item_f:
        per_item_f.close()
    memo_file.write_text(json.dumps(memo))

    up, ur = sum(plain) / len(plain), sum(recen) / len(recen)
    print("\n" + "=" * 50)
    print(f"  RECENCY EXPERIMENT  arm={args.arm} top-{args.topk} n={len(plain)}")
    print("=" * 50)
    print(f"  plain synthesis   utility = {up:.4f}")
    print(f"  recency synthesis utility = {ur:.4f}")
    print(f"  DELTA (recency - plain)   = {ur-up:+.4f}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
