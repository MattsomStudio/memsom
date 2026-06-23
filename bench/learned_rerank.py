"""learned_rerank — does a tiny learned reranker over {dense,sparse,colbert,nomic}
score-geometry beat the shipped equal-RRF by >=0.1 hit@1, HELD-OUT?

Why this and not more fusion: combo_h2h showed every static fusion ties (~0.67
hit@1) but the per-query ORACLE ceiling is 0.87 — the signals are complementary,
equal-RRF averages that away. A reranker that learns, per query, which signal to
trust from the SCORE GEOMETRY (margins/ranks) can capture some of that ceiling.

Honesty: weights are fit on the TRAIN half only; the +0.1 claim is judged on the
HELD-OUT TEST half. No gold leaks into test scoring. Hand-rolled logistic
regression (numpy, no sklearn dep). Reranker cost at query time = a dot product
over ~50 candidates using the THREE signals already computed -> not slower.

Usage (PC): python learned_rerank.py --dataset ...oracle.json --max-items 192 --cache C:\\Users\\you\\h2h_cache
"""
from __future__ import annotations
import argparse, base64, hashlib, json, pickle, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import from_longmemeval                       # noqa: E402

NOMIC_URL = "http://localhost:11434/api/embed"; NOMIC_MODEL = "nomic-embed-text"
BGE_URL = "http://127.0.0.1:11435/embed"; RRF_C = 60


def nomic_embed(texts, is_query):
    pre = "search_query: " if is_query else "search_document: "
    body = json.dumps({"model": NOMIC_MODEL, "input": [pre + t for t in texts]}).encode()
    req = urllib.request.Request(NOMIC_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return [np.asarray(e, dtype=np.float32) for e in json.loads(r.read())["embeddings"]]


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


def rrf_term(scores):
    order = np.argsort(-scores)
    out = np.zeros(len(scores))
    out[order] = 1.0 / (RRF_C + np.arange(1, len(scores) + 1))
    return out


def rank_of(scores):
    order = np.argsort(-scores)
    rank = np.empty(len(scores), dtype=np.float64)
    rank[order] = np.arange(1, len(scores) + 1)
    return rank


def znorm(x):
    x = np.asarray(x, dtype=np.float64); sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def feats(s_nom, s_d, s_sp, s_cb, idxs):
    """Per-candidate feature row from per-query score geometry (no gold used)."""
    zn, zd, zs, zc = znorm(s_nom), znorm(s_d), znorm(s_sp), znorm(s_cb)
    rn, rd, rs, rc = rrf_term(s_nom), rrf_term(s_d), rrf_term(s_sp), rrf_term(s_cb)
    kn, kd, ks, kc = rank_of(s_nom), rank_of(s_d), rank_of(s_sp), rank_of(s_cb)
    X = []
    for i in idxs:
        X.append([1.0, zn[i], zd[i], zs[i], zc[i],
                  rn[i], rd[i], rs[i], rc[i],
                  1.0 / kn[i], 1.0 / kd[i], 1.0 / ks[i], 1.0 / kc[i]])
    return np.asarray(X, dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, l2=1.0, iters=800, lr=0.3):
    """Plain full-batch gradient descent with class balancing (golds are rare)."""
    w = np.zeros(X.shape[1])
    pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
    sw = np.where(y == 1, neg / pos, 1.0)           # upweight the rare positive class
    sw = sw / sw.mean()
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (sw * (p - y)) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * g
    return w


def hit1(rank): return 1.0 if rank == 1 else 0.0


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
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cand", type=int, default=20, help="top-N per signal unioned into the candidate set")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, _ = from_longmemeval(args.dataset, max_items=args.max_items, max_evidence=args.max_evidence)
    corpus_text, corpus_item, corpus_gold = [], [], []
    seen = {}
    for it in items:
        for ev in it["evidence"]:
            key = (it["id"], ev["text"])
            if key in seen:
                continue
            seen[key] = len(corpus_text)
            corpus_text.append(ev["text"]); corpus_item.append(it["id"]); corpus_gold.append(bool(ev.get("answer_bearing")))
    N = len(corpus_text)
    gold_idx = {}
    for i, (iid, g) in enumerate(zip(corpus_item, corpus_gold)):
        if g:
            gold_idx.setdefault(iid, set()).add(i)
    items = [it for it in items if gold_idx.get(it["id"])]
    print(f"[learned] corpus={N} questions={len(items)} cand_union=top{args.cand}", file=sys.stderr, flush=True)

    # self-produced local cache (numpy arrays / ragged colbert / sparse dicts) — safe pickle.
    cache = Path(args.cache); cfile = cache / f"emb_{N}.pkl"
    if not cfile.exists():
        print(f"[learned] FATAL: no cache {cfile}", file=sys.stderr); return 1
    blob = pickle.load(open(cfile, "rb"))
    nom_doc, bge_d, bge_s, bge_c = blob["nom"], blob["bd"], blob["bs"], blob["bc"]
    doc_starts = np.zeros(N, dtype=np.int64); off = 0
    for i in range(N):
        doc_starts[i] = off; off += bge_c[i].shape[0]
    BIGC = np.vstack(bge_c).astype(np.float32)

    qfile = cache / "qcache.pkl"
    qcache = pickle.load(open(qfile, "rb")) if qfile.exists() else {}
    q_new = 0

    def get_q(q):
        nonlocal q_new
        key = hashlib.sha1(q.encode()).hexdigest()
        if key in qcache:
            d = qcache[key]; return d["nom"], d["qd"], d["qs"], d["qc"]
        qnom = nomic_embed([q], True)[0]
        qd, qs, qc = bge_embed([q]); qd, qs, qc = qd[0], qs[0], qc[0]
        qcache[key] = {"nom": qnom, "qd": qd, "qs": qs, "qc": qc}; q_new += 1
        return qnom, qd, qs, qc

    # precompute per-query score arrays + candidate sets
    Q = []
    t0 = time.time()
    for qn, it in enumerate(items):
        gold = gold_idx[it["id"]]
        qnom, qd, qs, qc = get_q(it["question"])
        s_nom = cos_mat(qnom, nom_doc); s_d = cos_mat(qd, bge_d)
        s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)], dtype=np.float64)
        sims = qc @ BIGC.T
        s_cb = np.maximum.reduceat(sims, doc_starts, axis=1).mean(axis=0).astype(np.float64)
        cand = set()
        for s in (s_nom, s_d, s_sp, s_cb):
            cand.update(np.argsort(-s)[:args.cand].tolist())
        cand = sorted(cand)
        # baseline rrf_equal ranking (full corpus) for the apples-to-apples floor
        rrf_eq = rrf_term(s_d) + rrf_term(s_sp) + rrf_term(s_cb)
        base_order = np.argsort(-rrf_eq).tolist()
        Q.append({"gold": gold, "s": (s_nom, s_d, s_sp, s_cb), "cand": cand,
                  "base_rank": gold_rank(base_order, gold)})
        if (qn + 1) % 25 == 0 or qn + 1 == len(items):
            print(f"   scored {qn+1}/{len(items)} ({(time.time()-t0)/(qn+1):.2f}s/q)", file=sys.stderr, flush=True)
    if q_new:
        pickle.dump(qcache, open(qfile, "wb"))

    half = len(Q) // 2
    train, test = Q[:half], Q[half:]

    # build train matrix
    Xtr, ytr = [], []
    for q in train:
        X = feats(*q["s"], q["cand"])
        y = np.array([1.0 if i in q["gold"] else 0.0 for i in q["cand"]])
        Xtr.append(X); ytr.append(y)
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    # standardize features (store mu/sd to apply at test + ship in the engine)
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
    mu[0] = 0.0; sd[0] = 1.0                       # leave bias column alone
    Xtr_s = (Xtr - mu) / sd
    w = fit_logistic(Xtr_s, ytr)

    def eval_split(split, name):
        h1_base = h1_learn = 0
        for q in split:
            # baseline
            h1_base += hit1(q["base_rank"])
            # learned: rerank candidates by predicted prob, tie-break by rrf
            X = (feats(*q["s"], q["cand"]) - mu) / sd
            p = sigmoid(X @ w)
            order = [q["cand"][j] for j in np.argsort(-p)]
            h1_learn += hit1(gold_rank(order, q["gold"]))
        n = len(split)
        print(f"  {name:<6} n={n:3d}  rrf_equal hit1={h1_base/n:.4f}  learned hit1={h1_learn/n:.4f}  "
              f"Δ={(h1_learn-h1_base)/n:+.4f}")
        return h1_base / n, h1_learn / n

    print("\n" + "=" * 64)
    print("  LEARNED RERANKER vs SHIPPED rrf_equal  (hit@1)")
    print("=" * 64)
    tr = eval_split(train, "TRAIN")
    te = eval_split(test, "TEST")
    print("=" * 64)
    print(f"  HELD-OUT TEST gain = {te[1]-te[0]:+.4f}  (target >= +0.1)")
    print("  weights:", np.round(w, 3).tolist())
    print("=" * 64)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "train": {"base": tr[0], "learned": tr[1]},
            "test": {"base": te[0], "learned": te[1]},
            "weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "feat_names": ["bias", "z_nom", "z_dense", "z_sparse", "z_colbert",
                           "rrf_nom", "rrf_dense", "rrf_sparse", "rrf_colbert",
                           "inv_rank_nom", "inv_rank_dense", "inv_rank_sparse", "inv_rank_colbert"],
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
