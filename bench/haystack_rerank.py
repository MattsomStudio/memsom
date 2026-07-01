"""haystack_rerank — the UNSATURATED test: per-question retrieval over the full
~550-turn LongMemEval haystack, where the baseline is weak and a reranker can earn
a real gain. Uses recall_bge's 4 shippable signals (fts/BM25 + BGE dense/sparse/
colbert). Per-question embeddings cached to disk so tuning re-runs are cheap.
Reranker trained/evaluated by leave-question-out CV (folds over questions).

Gold = turns flagged has_answer in the question's own haystack. hit@1 = is the
top-ranked turn answer-bearing.

Usage (PC): python haystack_rerank.py --dataset C:\\Users\\you\\lme_data\\longmemeval_s_cleaned.json
            --max-q 30 --cache C:\\Users\\you\\h2h_cache\\haystack
"""
from __future__ import annotations
import argparse, base64, json, math, pickle, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memsom\bench"; REPO = r"C:\Users\you\memsom"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from memsom_retrieve import tokenize                       # noqa: E402

BGE_URL = "http://127.0.0.1:11435/embed"
RRF_C = 60; K1 = 1.2; B = 0.75


def bge_embed(texts):
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(BGE_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read())
    dense = [np.asarray(d, dtype=np.float32) for d in resp["dense"]]
    colbert = [np.frombuffer(base64.b64decode(b), dtype="<f2").astype(np.float32).reshape(int(n), int(d))
               for b, (n, d) in zip(resp["colbert_b64"], resp["colbert_shape"])]
    return dense, resp["sparse"], colbert


def bge_embed_many(texts, batch=12):
    D, S, C = [], [], []
    for i in range(0, len(texts), batch):
        d, s, c = bge_embed(texts[i:i + batch]); D.extend(d); S.extend(s); C.extend(c)
    return np.vstack(D).astype(np.float32), S, C


def cos_all(q, M):
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


FEAT_NAMES = ["bias", "z_fts", "z_dense", "z_sparse", "z_colbert",
              "rrf_fts", "rrf_dense", "rrf_sparse", "rrf_colbert",
              "ir_fts", "ir_dense", "ir_sparse", "ir_colbert", "max_z", "agree5", "agree1"]


def feats(s_fts, s_d, s_sp, s_cb, idxs):
    sigs = (s_fts, s_d, s_sp, s_cb)
    zs = [znorm(s) for s in sigs]; rs = [rrf_term(s) for s in sigs]; ks = [rank_of(s) for s in sigs]
    X = []
    for i in idxs:
        zv = [z[i] for z in zs]; rk = [k[i] for k in ks]
        X.append([1.0] + zv + [r[i] for r in rs] + [1.0 / k[i] for k in ks]
                 + [max(zv), sum(r <= 5 for r in rk) / 4.0, sum(r == 1 for r in rk) / 4.0])
    return np.asarray(X, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-q", type=int, default=30)
    ap.add_argument("--cand", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)

    Q = []
    used = 0
    t0 = time.time()
    for e in data:
        if used >= args.max_q:
            break
        # build this question's corpus + gold from its own haystack
        corpus, gold = [], set()
        for sess in e["haystack_sessions"]:
            for turn in sess:
                c = (turn.get("content") or "").strip()
                if not c:
                    continue
                if turn.get("has_answer"):
                    gold.add(len(corpus))
                corpus.append(c)
        if not gold or len(corpus) < 20:
            continue
        qid = e["question_id"]; question = e["question"]
        # per-question embedding cache (self-produced local pickle — safe)
        cf = cache / f"q_{qid}.pkl"
        if cf.exists():
            blob = pickle.load(open(cf, "rb"))
            bge_d, bge_s, bge_c, qd, qs, qc = blob["d"], blob["s"], blob["c"], blob["qd"], blob["qs"], blob["qc"]
        else:
            bge_d, bge_s, bge_c = bge_embed_many(corpus)
            qd, qs, qc = bge_embed([question]); qd, qs, qc = qd[0], qs[0], qc[0]
            pickle.dump({"d": bge_d, "s": bge_s, "c": bge_c, "qd": qd, "qs": qs, "qc": qc}, open(cf, "wb"))
        N = len(corpus)
        doc_starts = np.zeros(N, dtype=np.int64); off = 0
        for i in range(N):
            doc_starts[i] = off; off += bge_c[i].shape[0]
        BIGC = np.vstack(bge_c).astype(np.float32)
        s_d = cos_all(qd, bge_d)
        s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)], float)
        s_cb = np.maximum.reduceat(qc @ BIGC.T, doc_starts, axis=1).mean(axis=0).astype(float)
        postings, doclen, avgdl, idf = bm25_setup(corpus)
        s_fts = bm25_q(question, postings, doclen, avgdl, idf, N)
        base = rrf_term(s_fts) + rrf_term(s_sp) + rrf_term(s_d) + rrf_term(s_cb)
        base_rank = gold_rank(np.argsort(-base).tolist(), gold)
        cand = set()
        for s in (s_fts, s_d, s_sp, s_cb):
            cand.update(np.argsort(-s)[:args.cand].tolist())
        cand = sorted(cand)
        Q.append({"qid": qid, "gold": gold & set(cand), "gold_full": gold,
                  "X": feats(s_fts, s_d, s_sp, s_cb, cand), "cand": cand,
                  "base_rank": base_rank, "N": N})
        used += 1
        if used % 5 == 0:
            print(f"  embedded/scored {used} questions ({(time.time()-t0)/used:.1f}s/q)", file=sys.stderr, flush=True)

    n = len(Q)
    print(f"[haystack] questions={n} avg_corpus={np.mean([q['N'] for q in Q]):.0f}", file=sys.stderr, flush=True)
    base_h1 = sum(1 for q in Q if q["base_rank"] == 1) / n
    cand_miss = sum(1 for q in Q if not q["gold"]) / n

    fg, fl = [], []
    for f in range(args.folds):
        tr = [i for i in range(n) if i % args.folds != f]; te = [i for i in range(n) if i % args.folds == f]
        Xtr = np.vstack([Q[i]["X"] for i in tr])
        ytr = np.concatenate([[1.0 if c in Q[i]["gold"] else 0.0 for c in Q[i]["cand"]] for i in tr])
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1; mu[0] = 0; sd[0] = 1
        w = fit_logistic((Xtr - mu) / sd, ytr, l2=args.l2)
        h1b = h1l = 0
        for i in te:
            q = Q[i]; h1b += 1.0 if q["base_rank"] == 1 else 0.0
            p = sigmoid(((q["X"] - mu) / sd) @ w)
            order = [q["cand"][j] for j in np.argsort(-p)]
            h1l += 1.0 if gold_rank(order, q["gold_full"]) == 1 else 0.0
        m = len(te); fg.append((h1l - h1b) / m); fl.append(h1l / m)
        print(f"  fold {f}: base={h1b/m:.4f} learned={h1l/m:.4f} d={(h1l-h1b)/m:+.4f}", file=sys.stderr, flush=True)

    g = np.array(fg)
    print("\n" + "=" * 64)
    print(f"  HAYSTACK (per-question, ~550 distractors)  n={n}  hit@1")
    print("=" * 64)
    print(f"  baseline 4-way RRF hit@1 = {base_h1:.4f}   (candidate gold-miss={cand_miss:.3f})")
    print(f"  learned reranker (CV)    = {np.mean(fl):.4f}")
    print(f"  HELD-OUT GAIN mean={g.mean():+.4f} std={g.std():.4f} min={g.min():+.4f} max={g.max():+.4f}")
    print(f"  target >= +0.1 : {'MET' if g.mean() >= 0.1 else 'not yet'}")
    print("=" * 64)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base_h1": base_h1, "learned_cv_h1": float(np.mean(fl)),
            "cv_gain_mean": float(g.mean()), "cv_gain_std": float(g.std()),
            "n": n, "cand": args.cand, "l2": args.l2,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
