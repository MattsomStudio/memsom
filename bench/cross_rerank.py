"""cross_rerank — the one genuinely-stronger lever left: a CROSS-ENCODER reranker
(bge-reranker-v2-m3, joint query-doc attention) over the baseline's top-K.

Bi-encoder fusion (dense/sparse/colbert/bm25) ties at ~0.72 hit@1 and a logistic
reranker over their scores can't beat it. A cross-encoder is a different, stronger
model. This measures whether it reaches +0.1 hit@1, and times the added latency so
the "no slower" trade is explicit (it IS extra compute — that's the honest caveat).

Baseline = the 4-way equal RRF (fts/BM25 + BGE dense+sparse+colbert) from cache.
Rerank top-K candidates by the cross-encoder, measure hit@1 + per-query rerank ms.
"""
from __future__ import annotations
import argparse, json, math, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memdag\bench"; REPO = r"C:\Users\you\memdag"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from dataset import from_longmemeval                       # noqa: E402
from memdag_retrieve import tokenize                       # noqa: E402

RRF_C = 60; K1 = 1.2; B = 0.75


def rrf_term(s):
    o = np.argsort(-s); out = np.zeros(len(s)); out[o] = 1.0 / (RRF_C + np.arange(1, len(s) + 1)); return out


def sparse_dot(a, b):
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def cos_all(q, M):
    qn = q / (np.linalg.norm(q) or 1.0); Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn


def bm25_setup(corpus):
    postings = defaultdict(dict); doclen = np.zeros(len(corpus))
    for i, t in enumerate(corpus):
        toks = tokenize(t); doclen[i] = len(toks); tf = defaultdict(int)
        for w in toks:
            tf[w] += 1
        for w, c in tf.items():
            postings[w][i] = c
    N = len(corpus); avgdl = doclen.mean() if N else 0.0
    idf = {w: math.log((N - len(d) + 0.5) / (len(d) + 0.5) + 1.0) for w, d in postings.items()}
    return postings, doclen, avgdl, idf


def bm25_q(q, postings, doclen, avgdl, idf, N):
    s = np.zeros(N)
    if avgdl == 0:
        return s
    for term in set(tokenize(q)):
        if term in postings:
            wid = idf[term]
            for nid, tf in postings[term].items():
                s[nid] += wid * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doclen[nid] / avgdl))
    return s


def gold_rank(order, gold):
    for r, i in enumerate(order, 1):
        if i in gold:
            return r
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=192)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=20, help="how many baseline candidates to cross-encode")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, _ = from_longmemeval(args.dataset, max_items=args.max_items, max_evidence=args.max_evidence)
    corpus, citem, cgold, seen = [], [], [], {}
    for it in items:
        for ev in it["evidence"]:
            key = (it["id"], ev["text"])
            if key in seen:
                continue
            seen[key] = len(corpus); corpus.append(ev["text"]); citem.append(it["id"]); cgold.append(bool(ev.get("answer_bearing")))
    N = len(corpus)
    gold_idx = {}
    for i, (iid, g) in enumerate(zip(citem, cgold)):
        if g:
            gold_idx.setdefault(iid, set()).add(i)
    items = [it for it in items if gold_idx.get(it["id"])]

    # self-produced local cache — safe pickle.
    blob = pickle.load(open(Path(args.cache) / f"emb_{N}.pkl", "rb"))
    bge_d, bge_s, bge_c = blob["bd"], blob["bs"], blob["bc"]
    doc_starts = np.zeros(N, dtype=np.int64); off = 0
    for i in range(N):
        doc_starts[i] = off; off += bge_c[i].shape[0]
    BIGC = np.vstack(bge_c).astype(np.float32)
    postings, doclen, avgdl, idf = bm25_setup(corpus)
    qcache = pickle.load(open(Path(args.cache) / "qcache.pkl", "rb"))

    print("[cross] loading bge-reranker-v2-m3 (transformers, fast tokenizer) ...", file=sys.stderr, flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", use_fast=True)
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16 if _dev == "cuda" else torch.float32).to(_dev).eval()

    class _R:
        def compute_score(self, pairs, normalize=True):
            with torch.no_grad():
                inp = _tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(_dev)
                logits = _model(**inp).logits.view(-1).float().cpu().numpy()
            return (1.0 / (1.0 + np.exp(-logits))).tolist() if normalize else logits.tolist()
    reranker = _R()

    base_h1 = cross_h1 = 0
    rerank_ms = []
    for it in items:
        gold = gold_idx[it["id"]]
        q = it["question"]
        v = qcache[__import__("hashlib").sha1(q.encode()).hexdigest()]
        qd, qs, qc = v["qd"], v["qs"], v["qc"]
        s_d = cos_all(qd, bge_d)
        s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)], float)
        s_cb = np.maximum.reduceat(qc @ BIGC.T, doc_starts, axis=1).mean(axis=0).astype(float)
        s_fts = bm25_q(q, postings, doclen, avgdl, idf, N)
        base = rrf_term(s_fts) + rrf_term(s_sp) + rrf_term(s_d) + rrf_term(s_cb)
        base_order = np.argsort(-base).tolist()
        base_h1 += 1.0 if (base_order and base_order[0] in gold) else 0.0
        # cross-encode the top-K baseline candidates
        cand = base_order[:args.topk]
        pairs = [[q, corpus[i]] for i in cand]
        t0 = time.time()
        scores = reranker.compute_score(pairs, normalize=True)
        rerank_ms.append((time.time() - t0) * 1000.0)
        if not isinstance(scores, list):
            scores = [scores]
        reordered = [c for _, c in sorted(zip(scores, cand), key=lambda x: -x[0])]
        cross_h1 += 1.0 if (reordered and reordered[0] in gold) else 0.0

    n = len(items)
    print("\n" + "=" * 64)
    print(f"  CROSS-ENCODER RERANK (bge-reranker-v2-m3, top-{args.topk})  n={n}  hit@1")
    print("=" * 64)
    print(f"  baseline 4-way RRF hit@1 = {base_h1/n:.4f}")
    print(f"  + cross-encoder rerank   = {cross_h1/n:.4f}")
    print(f"  GAIN = {(cross_h1-base_h1)/n:+.4f}   (target >= +0.1: {'MET' if (cross_h1-base_h1)/n>=0.1 else 'no'})")
    print(f"  added latency/query: mean={np.mean(rerank_ms):.0f}ms median={np.median(rerank_ms):.0f}ms")
    print("=" * 64)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base_h1": base_h1 / n, "cross_h1": cross_h1 / n, "gain": (cross_h1 - base_h1) / n,
            "rerank_ms_mean": float(np.mean(rerank_ms)), "topk": args.topk, "n": n,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
