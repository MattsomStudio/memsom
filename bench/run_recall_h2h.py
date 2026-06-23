"""run_recall_h2h — nomic vs BGE-M3 retrieval head-to-head for Matthew's recall system.

Mirrors the memdag bench's principle (one substrate, same judge, fair-both-ways,
report losses) but the metric is RETRIEVAL DISCRIMINATION, which is the actual
variable between the old backend (nomic, dense-only) and the new one (BGE-M3,
dense+sparse+colbert triple fusion).

Why a shared haystack: the memdag per-item harness gives each item only its own
evidence, so top-k >= evidence -> retrieval is trivial and two embedders look
identical. Here we POOL every LongMemEval evidence turn into one corpus; each
question must surface ITS answer-bearing evidence above all the distractors. That
is what separates a better embedder from a worse one.

Three arms, identical items / queries / top-k / judge:
  nomic       — Ollama nomic-embed-text, dense cosine (the old system)
  bge-dense   — BGE-M3 dense vector only (isolates EMBEDDING-QUALITY gain)
  bge-triple  — BGE-M3 dense+sparse+colbert, RRF-fused (isolates FUSION/ColBERT gain)

Retrieval metrics (deterministic, no LLM): hit@1, hit@5, MRR, recall@k over each
item's answer-bearing evidence. Optional --judge adds LLM-judged answer utility
(synthesize from top-k + qwen2.5 judge) on the first --judge-n items, matching the
memdag utility metric.

Run on the PC (nomic Ollama + BGE service + LongMemEval all local there):
  python run_recall_h2h.py --dataset C:\\Users\\you\\lme_data\\longmemeval_oracle.json \\
    --max-items 120 --topk 10 --judge --judge-n 60 --out C:\\Users\\you\\recall_h2h.json
"""
from __future__ import annotations
import argparse, base64, json, math, sys, time, urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import from_longmemeval                       # noqa: E402
from judge import synthesize, judge_correct                # noqa: E402

NOMIC_URL = "http://localhost:11434/api/embed"
NOMIC_MODEL = "nomic-embed-text"
BGE_URL = "http://127.0.0.1:11435/embed"
RRF_C = 60


# ---------- embedding backends ----------
def nomic_embed(texts, is_query):
    pre = "search_query: " if is_query else "search_document: "
    body = json.dumps({"model": NOMIC_MODEL, "input": [pre + t for t in texts]}).encode()
    req = urllib.request.Request(NOMIC_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        embs = json.loads(r.read())["embeddings"]
    return [np.asarray(e, dtype=np.float32) for e in embs]


def bge_embed(texts):
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(BGE_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    dense = [np.asarray(d, dtype=np.float32) for d in resp["dense"]]
    sparse = resp["sparse"]
    colbert = []
    for b64, (n, d) in zip(resp["colbert_b64"], resp["colbert_shape"]):
        arr = np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32).reshape(int(n), int(d))
        colbert.append(arr)
    return dense, sparse, colbert


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


# ---------- scoring ----------
def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(a @ b / (na * nb))


def sparse_dot(a, b):
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def maxsim(q, d):
    if q.size == 0 or d.size == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).mean())


def rrf_ranks(scores):
    """scores: list of floats over corpus -> {idx: rrf_term} by descending score."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {idx: 1.0 / (RRF_C + r) for r, idx in enumerate(order, 1)}


# ---------- metrics ----------
def item_metrics(ranked_idxs, gold_set, topk):
    """ranked_idxs: corpus indices best-first. gold_set: set of answer-bearing corpus idxs."""
    top = ranked_idxs[:topk]
    hit1 = 1.0 if top and top[0] in gold_set else 0.0
    hit5 = 1.0 if any(i in gold_set for i in top[:5]) else 0.0
    recall = (sum(1 for i in top if i in gold_set) / len(gold_set)) if gold_set else 0.0
    mrr = 0.0
    for rank, i in enumerate(ranked_idxs, 1):
        if i in gold_set:
            mrr = 1.0 / rank
            break
    return hit1, hit5, mrr, recall


def aggregate(rows):
    n = len(rows)
    if not n:
        return {}
    keys = ("hit1", "hit5", "mrr", "recall")
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=120)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-n", type=int, default=60)
    ap.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--judge-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, report = from_longmemeval(args.dataset, max_items=args.max_items,
                                     max_evidence=args.max_evidence)
    print(f"[h2h] LongMemEval used={report['used']} skipped={report['skipped_total']}", file=sys.stderr)
    if not items:
        print("[h2h] no items", file=sys.stderr); return 1

    # ---- build the shared haystack: every evidence turn, tagged by source item + gold flag ----
    corpus_text, corpus_item, corpus_gold = [], [], []
    seen = {}
    for it in items:
        for ev in it["evidence"]:
            t = ev["text"]
            key = (it["id"], t)
            if key in seen:
                continue
            seen[key] = len(corpus_text)
            corpus_text.append(t)
            corpus_item.append(it["id"])
            corpus_gold.append(bool(ev.get("answer_bearing")))
    N = len(corpus_text)
    # gold corpus indices per item (its answer-bearing evidence)
    gold_idx = {}
    for i, (iid, g) in enumerate(zip(corpus_item, corpus_gold)):
        if g:
            gold_idx.setdefault(iid, set()).add(i)
    items = [it for it in items if gold_idx.get(it["id"])]   # need >=1 gold target
    print(f"[h2h] corpus={N} texts | scoring questions={len(items)} | topk={args.topk}", file=sys.stderr)

    # ---- embed the corpus once per backend ----
    t0 = time.time()
    print("[h2h] embedding corpus: nomic ...", file=sys.stderr)
    nom_doc = []
    for i, part in batched(corpus_text, 64):
        nom_doc.extend(nomic_embed(part, is_query=False))
        print(f"   nomic {i+len(part)}/{N}", file=sys.stderr)
    print("[h2h] embedding corpus: bge (dense+sparse+colbert) ...", file=sys.stderr)
    bge_d, bge_s, bge_c = [], [], []
    for i, part in batched(corpus_text, 12):
        d, s, c = bge_embed(part)
        bge_d.extend(d); bge_s.extend(s); bge_c.extend(c)
        print(f"   bge {i+len(part)}/{N}", file=sys.stderr)
    print(f"[h2h] corpus embedded in {time.time()-t0:.0f}s", file=sys.stderr)

    rows = {"nomic": [], "bge-dense": [], "bge-triple": []}
    util = {"nomic": [], "bge-dense": [], "bge-triple": []}

    for qn, it in enumerate(items, 1):
        gold = gold_idx[it["id"]]
        q = it["question"]
        qnom = nomic_embed([q], is_query=True)[0]
        qd, qs, qc = bge_embed([q]); qd, qs, qc = qd[0], qs[0], qc[0]

        # nomic: dense cosine
        s_nom = [cos(qnom, nom_doc[i]) for i in range(N)]
        rank_nom = sorted(range(N), key=lambda i: -s_nom[i])
        # bge-dense: dense cosine on bge vectors
        s_bd = [cos(qd, bge_d[i]) for i in range(N)]
        rank_bd = sorted(range(N), key=lambda i: -s_bd[i])
        # bge-triple: RRF over dense + sparse + colbert
        rd = rrf_ranks(s_bd)
        rs = rrf_ranks([sparse_dot(qs, bge_s[i]) for i in range(N)])
        rc = rrf_ranks([maxsim(qc, bge_c[i]) for i in range(N)])
        fused = [rd.get(i, 0) + rs.get(i, 0) + rc.get(i, 0) for i in range(N)]
        rank_bt = sorted(range(N), key=lambda i: -fused[i])

        for name, ranking in (("nomic", rank_nom), ("bge-dense", rank_bd), ("bge-triple", rank_bt)):
            h1, h5, mrr, rec = item_metrics(ranking, gold, args.topk)
            rows[name].append({"hit1": h1, "hit5": h5, "mrr": mrr, "recall": rec})

        if args.judge and qn <= args.judge_n:
            goldterms = "; ".join(it["gold_terms"])
            for name, ranking in (("nomic", rank_nom), ("bge-dense", rank_bd), ("bge-triple", rank_bt)):
                mems = [corpus_text[i] for i in ranking[:args.topk]]
                ans = synthesize(q, mems, model=args.judge_model, url=args.judge_url)
                ok = judge_correct(q, goldterms, ans, model=args.judge_model, url=args.judge_url)
                util[name].append(1.0 if ok else 0.0)

        if qn % 10 == 0 or qn == len(items):
            print(f"   scored {qn}/{len(items)} ({(time.time()-t0)/qn:.1f}s/q)", file=sys.stderr)

    result = {"systems": {}, "config": {"corpus": N, "questions": len(items), "topk": args.topk,
                                        "judged": min(args.judge_n, len(items)) if args.judge else 0}}
    for name in rows:
        agg = aggregate(rows[name])
        if args.judge and util[name]:
            agg["utility"] = round(sum(util[name]) / len(util[name]), 4)
        result["systems"][name] = agg

    # ---- print the head-to-head table ----
    print("\n" + "=" * 64)
    print(f"  RECALL BACKEND HEAD-TO-HEAD  (corpus={N}, q={len(items)}, top-{args.topk})")
    print("=" * 64)
    cols = ["hit1", "hit5", "mrr", "recall"] + (["utility"] if args.judge else [])
    print(f"  {'system':<12}" + "".join(f"{c:>9}" for c in cols))
    for name in ("nomic", "bge-dense", "bge-triple"):
        a = result["systems"][name]
        print(f"  {name:<12}" + "".join(f"{a.get(c, 0):>9.4f}" for c in cols))
    print("-" * 64)
    base = result["systems"]["nomic"]
    bt = result["systems"]["bge-triple"]
    print("  DELTA bge-triple - nomic:" + "".join(
        f"{c}={bt.get(c,0)-base.get(c,0):+.4f}  " for c in cols))
    print("=" * 64 + "\n")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[h2h] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
