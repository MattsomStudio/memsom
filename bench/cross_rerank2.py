"""cross_rerank2 — cross-encoder reranking BLENDED with the first-stage score
(standard production rerank practice), selected by query train/test so the config
is validated held-out, not fit to hit +0.1.

Score top-50 candidates with bge-reranker-v2-m3 ONCE per query (one model pass),
then evaluate, from those cached scores, a grid of (depth, lambda) where final rank
= RRF(cross-encoder) + lambda*RRF(first-stage), over the top-`depth` candidates.
Pick the best (depth,lambda) on the TRAIN half by hit@1, report it on the held-out
TEST half. The cross-encoder is frozen (no fitting); the only selected knobs are
depth + lambda. Reports full grid for transparency.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memsom\bench"; REPO = r"C:\Users\you\memsom"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from dataset import from_longmemeval                       # noqa: E402
from memsom.retrieval.retrieve import tokenize

RRF_C = 60; K1 = 1.2; B = 0.75
SCORE_K = 50                       # how many baseline candidates we cross-encode (once)
DEPTHS = [10, 20, 30, 50]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]


def rrf_term(s):
    o = np.argsort(-s); out = np.zeros(len(s)); out[o] = 1.0 / (RRF_C + np.arange(1, len(s) + 1)); return out


def rrf_over(order):
    """RRF term keyed by position in `order` (list of ids)."""
    return {nid: 1.0 / (RRF_C + r) for r, nid in enumerate(order, 1)}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=192)
    ap.add_argument("--max-evidence", type=int, default=6)
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

    blob = pickle.load(open(Path(args.cache) / f"emb_{N}.pkl", "rb"))   # self-produced local cache
    bge_d, bge_s, bge_c = blob["bd"], blob["bs"], blob["bc"]
    doc_starts = np.zeros(N, dtype=np.int64); off = 0
    for i in range(N):
        doc_starts[i] = off; off += bge_c[i].shape[0]
    BIGC = np.vstack(bge_c).astype(np.float32)
    postings, doclen, avgdl, idf = bm25_setup(corpus)
    qcache = pickle.load(open(Path(args.cache) / "qcache.pkl", "rb"))

    print("[cross2] loading bge-reranker-v2-m3 ...", file=sys.stderr, flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", use_fast=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    def cross_score(q, docs):
        with torch.no_grad():
            inp = tok([[q, d] for d in docs], padding=True, truncation=True, return_tensors="pt", max_length=512).to(dev)
            return model(**inp).logits.view(-1).float().cpu().numpy()

    # per query: baseline top-SCORE_K candidates + their cross scores + base rrf-over-candidates
    QD = []
    t0 = time.time()
    for it in items:
        gold = gold_idx[it["id"]]; q = it["question"]
        v = qcache[hashlib.sha1(q.encode()).hexdigest()]
        s_d = cos_all(v["qd"], bge_d)
        s_sp = np.array([sparse_dot(v["qs"], bge_s[i]) for i in range(N)], float)
        s_cb = np.maximum.reduceat(v["qc"] @ BIGC.T, doc_starts, axis=1).mean(axis=0).astype(float)
        s_fts = bm25_q(q, postings, doclen, avgdl, idf, N)
        base = rrf_term(s_fts) + rrf_term(s_sp) + rrf_term(s_d) + rrf_term(s_cb)
        base_order = np.argsort(-base).tolist()
        cand = base_order[:SCORE_K]
        cs = cross_score(q, [corpus[i] for i in cand])
        QD.append({"gold": gold, "cand": cand, "cross": cs,
                   "base_h1": 1.0 if base_order[0] in gold else 0.0})
    print(f"[cross2] scored {len(QD)} queries in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    def hit1_blend(qd, depth, lam):
        cand = qd["cand"][:depth]
        cross = qd["cross"][:depth]
        cross_order = [cand[j] for j in np.argsort(-cross)]
        rc = rrf_over(cross_order)                       # cross-encoder RRF
        rb = {nid: 1.0 / (RRF_C + r) for r, nid in enumerate(cand, 1)}   # first-stage RRF (cand already base-ordered)
        fused = sorted(cand, key=lambda i: -(rc[i] + lam * rb[i]))
        return 1.0 if fused[0] in qd["gold"] else 0.0

    n = len(QD); half = n // 2
    base_all = np.mean([qd["base_h1"] for qd in QD])

    # full-data grid (transparency)
    print("\n" + "=" * 70)
    print(f"  CROSS-ENCODER + FIRST-STAGE BLEND grid (ALL n={n}), baseline hit@1={base_all:.4f}")
    print("=" * 70)
    print("  depth\\lam " + "".join(f"{l:>8}" for l in LAMBDAS))
    grid = {}
    for d in DEPTHS:
        row = []
        for l in LAMBDAS:
            h = np.mean([hit1_blend(qd, d, l) for qd in QD]); grid[(d, l)] = h; row.append(h)
        print(f"  d={d:<3}    " + "".join(f"{x:>8.4f}" for x in row))
    print(f"  (gain = cell - {base_all:.4f})")

    # honest selection: pick (depth,lambda) on TRAIN, report on TEST
    train, test = QD[:half], QD[half:]
    base_te = np.mean([qd["base_h1"] for qd in test])
    best = max(((d, l) for d in DEPTHS for l in LAMBDAS),
               key=lambda dl: np.mean([hit1_blend(qd, *dl) for qd in train]))
    tr_gain = np.mean([hit1_blend(qd, *best) for qd in train]) - np.mean([qd["base_h1"] for qd in train])
    te_learn = np.mean([hit1_blend(qd, *best) for qd in test])
    print("\n" + "=" * 70)
    print(f"  selected on TRAIN: depth={best[0]} lambda={best[1]}  (train gain {tr_gain:+.4f})")
    print(f"  HELD-OUT TEST: baseline={base_te:.4f}  blended={te_learn:.4f}  GAIN={te_learn-base_te:+.4f}")
    print(f"  target >= +0.1 : {'MET' if te_learn-base_te >= 0.1 else 'no'}")
    print("=" * 70)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base_all": base_all, "grid": {f"{d}_{l}": grid[(d, l)] for d in DEPTHS for l in LAMBDAS},
            "selected": {"depth": best[0], "lambda": best[1]},
            "test_baseline": base_te, "test_blended": te_learn, "test_gain": te_learn - base_te,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
