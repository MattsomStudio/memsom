"""ship_rerank — train + measure a reranker over EXACTLY recall_bge.py's signals.

recall_bge.py fuses FTS(keyword, BM25-like) + BGE dense + BGE sparse + BGE colbert.
So the shippable reranker may use ONLY those four — no nomic, nothing the engine
can't compute at query time. Baseline = the 4-way equal RRF recall_bge actually
ships. 5-fold CV for the honest held-out gain; then a final fit on all data whose
weights/mu/sd are exported for the engine. Reranker cost at query time = a dot
product over ~50 candidates from signals already computed -> not slower.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, math, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memdag\bench"; REPO = r"C:\Users\you\memdag"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from dataset import from_longmemeval                       # noqa: E402
from memdag_retrieve import tokenize                       # noqa: E402

BGE_URL = "http://127.0.0.1:11435/embed"
RRF_C = 60; K1 = 1.2; B = 0.75
import urllib.request


def bge_embed(texts):
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(BGE_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    dense = [np.asarray(d, dtype=np.float32) for d in resp["dense"]]
    colbert = [np.frombuffer(base64.b64decode(b), dtype="<f2").astype(np.float32).reshape(int(n), int(d))
               for b, (n, d) in zip(resp["colbert_b64"], resp["colbert_shape"])]
    return dense, resp["sparse"], colbert


def cos_mat(q, M):
    qn = q / (np.linalg.norm(q) or 1.0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn


def sparse_dot(a, b):
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def rrf_term(s):
    o = np.argsort(-s); out = np.zeros(len(s)); out[o] = 1.0 / (RRF_C + np.arange(1, len(s) + 1)); return out


def rank_of(s):
    o = np.argsort(-s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); return r


def znorm(x):
    x = np.asarray(x, float); sd = x.std(); return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, l2=1.0, iters=1500, lr=0.3):
    w = np.zeros(X.shape[1]); pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
    sw = np.where(y == 1, neg / pos, 1.0); sw /= sw.mean()
    for _ in range(iters):
        p = sigmoid(X @ w)
        w -= lr * (X.T @ (sw * (p - y)) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y))
    return w


def gold_rank(order, gold):
    for r, i in enumerate(order, 1):
        if i in gold:
            return r
    return 0


def build_bm25(corpus):
    postings = defaultdict(dict); doclen = np.zeros(len(corpus))
    for i, t in enumerate(corpus):
        toks = tokenize(t); doclen[i] = len(toks)
        tf = defaultdict(int)
        for w in toks:
            tf[w] += 1
        for w, c in tf.items():
            postings[w][i] = c
    N = len(corpus); avgdl = doclen.mean() if N else 0.0
    idf = {w: math.log((N - len(d) + 0.5) / (len(d) + 0.5) + 1.0) for w, d in postings.items()}
    return postings, doclen, avgdl, idf


def bm25_scores(q, postings, doclen, avgdl, idf, N):
    s = np.zeros(N)
    if avgdl == 0.0:
        return s
    for term in set(tokenize(q)):
        if term not in postings:
            continue
        wid = idf[term]
        for nid, tf in postings[term].items():
            s[nid] += wid * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doclen[nid] / avgdl))
    return s


# feature order is FIXED and shipped with the weights. Signals: fts(bm25), dense, sparse, colbert.
FEAT_NAMES = ["bias",
              "z_fts", "z_dense", "z_sparse", "z_colbert",
              "rrf_fts", "rrf_dense", "rrf_sparse", "rrf_colbert",
              "ir_fts", "ir_dense", "ir_sparse", "ir_colbert",
              "max_z", "agree_top5", "agree_top1"]


def feats(s_fts, s_d, s_sp, s_cb, idxs):
    sigs = (s_fts, s_d, s_sp, s_cb)
    zs = [znorm(s) for s in sigs]; rs = [rrf_term(s) for s in sigs]; ks = [rank_of(s) for s in sigs]
    X = []
    for i in idxs:
        zv = [z[i] for z in zs]; rk = [k[i] for k in ks]
        a5 = sum(1 for r in rk if r <= 5) / 4.0; a1 = sum(1 for r in rk if r == 1) / 4.0
        X.append([1.0] + zv + [r[i] for r in rs] + [1.0 / k[i] for k in ks] + [max(zv), a5, a1])
    return np.asarray(X, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=192)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--cand", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1.0)
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
    postings, doclen, avgdl, idf = build_bm25(corpus)

    qfile = Path(args.cache) / "qcache.pkl"
    qcache = pickle.load(open(qfile, "rb")) if qfile.exists() else {}

    def get_q(q):
        key = hashlib.sha1(q.encode()).hexdigest()
        if key in qcache:
            d = qcache[key]; return d["qd"], d["qs"], d["qc"]
        qd, qs, qc = bge_embed([q]); qd, qs, qc = qd[0], qs[0], qc[0]
        qcache[key] = {"nom": None, "qd": qd, "qs": qs, "qc": qc}; return qd, qs, qc

    Q = []
    for it in items:
        gold = gold_idx[it["id"]]
        qd, qs, qc = get_q(it["question"])
        s_d = cos_mat(qd, bge_d)
        s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)], float)
        s_cb = np.maximum.reduceat(qc @ BIGC.T, doc_starts, axis=1).mean(axis=0).astype(float)
        s_fts = bm25_scores(it["question"], postings, doclen, avgdl, idf, N)
        # recall_bge ships the 4-way equal RRF(fts,sparse,dense,colbert)
        base = rrf_term(s_fts) + rrf_term(s_sp) + rrf_term(s_d) + rrf_term(s_cb)
        base_rank = gold_rank(np.argsort(-base).tolist(), gold)
        cand = set()
        for s in (s_fts, s_d, s_sp, s_cb):
            cand.update(np.argsort(-s)[:args.cand].tolist())
        cand = sorted(cand)
        Q.append({"gold": gold, "X": feats(s_fts, s_d, s_sp, s_cb, cand), "cand": cand, "base_rank": base_rank})

    base_h1 = sum(1 for q in Q if q["base_rank"] == 1) / len(Q)
    fg, fl = [], []
    for f in range(args.folds):
        tr = [i for i in range(len(Q)) if i % args.folds != f]; te = [i for i in range(len(Q)) if i % args.folds == f]
        Xtr = np.vstack([Q[i]["X"] for i in tr])
        ytr = np.concatenate([[1.0 if c in Q[i]["gold"] else 0.0 for c in Q[i]["cand"]] for i in tr])
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1; mu[0] = 0; sd[0] = 1
        w = fit_logistic((Xtr - mu) / sd, ytr, l2=args.l2)
        h1b = h1l = 0
        for i in te:
            q = Q[i]; h1b += 1.0 if q["base_rank"] == 1 else 0.0
            p = sigmoid(((q["X"] - mu) / sd) @ w)
            order = [q["cand"][j] for j in np.argsort(-p)]
            h1l += 1.0 if gold_rank(order, q["gold"]) == 1 else 0.0
        n = len(te); fg.append((h1l - h1b) / n); fl.append(h1l / n)
        print(f"  fold {f}: base={h1b/n:.4f} learned={h1l/n:.4f} d={(h1l-h1b)/n:+.4f}", file=sys.stderr, flush=True)

    g = np.array(fg)
    print("\n" + "=" * 62)
    print("  recall_bge RERANKER (fts+dense+sparse+colbert) — hit@1")
    print("=" * 62)
    print(f"  shipped 4-way RRF hit@1   = {base_h1:.4f}")
    print(f"  learned (CV mean)         = {np.mean(fl):.4f}")
    print(f"  HELD-OUT GAIN mean={g.mean():+.4f} std={g.std():.4f} min={g.min():+.4f} max={g.max():+.4f}")
    print("=" * 62)

    Xall = np.vstack([q["X"] for q in Q])
    yall = np.concatenate([[1.0 if c in q["gold"] else 0.0 for c in q["cand"]] for q in Q])
    mu = Xall.mean(0); sd = Xall.std(0); sd[sd == 0] = 1; mu[0] = 0; sd[0] = 1
    w = fit_logistic((Xall - mu) / sd, yall, l2=args.l2)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "cv_gain_mean": float(g.mean()), "cv_gain_std": float(g.std()),
            "base_h1": base_h1, "learned_cv_h1": float(np.mean(fl)),
            "weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "feat_names": FEAT_NAMES, "cand": args.cand, "l2": args.l2, "rrf_c": RRF_C,
        }, indent=2))
        print(f"[ship] wrote {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
