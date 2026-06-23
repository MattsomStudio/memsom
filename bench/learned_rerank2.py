"""learned_rerank2 — richer features + 5-fold CV for an honest stable gain estimate.

v1 got +0.083 on one held-out half (noisy at n=96). v2 adds cross-signal
agreement features (gold is the doc several signals concur on) and reports the
mean held-out gain over 5 deterministic folds, so the +0.1 claim isn't one lucky
split. Reranker cost is unchanged: a dot product over ~50 candidates from the
three signals already computed -> not slower, no downside.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, pickle, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import from_longmemeval                       # noqa: E402

NOMIC_URL = "http://localhost:11434/api/embed"; NOMIC_MODEL = "nomic-embed-text"
BGE_URL = "http://127.0.0.20:11435/embed"; RRF_C = 60


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
    order = np.argsort(-scores); out = np.zeros(len(scores))
    out[order] = 1.0 / (RRF_C + np.arange(1, len(scores) + 1)); return out


def rank_of(scores):
    order = np.argsort(-scores); rank = np.empty(len(scores), dtype=np.float64)
    rank[order] = np.arange(1, len(scores) + 1); return rank


def znorm(x):
    x = np.asarray(x, dtype=np.float64); sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


FEAT_NAMES = ["bias", "z_nom", "z_dense", "z_sparse", "z_colbert",
              "rrf_nom", "rrf_dense", "rrf_sparse", "rrf_colbert",
              "inv_rank_nom", "inv_rank_dense", "inv_rank_sparse", "inv_rank_colbert",
              "max_z", "agree_top5", "agree_top1", "rrf_equal"]


def feats(s_nom, s_d, s_sp, s_cb, idxs):
    zn, zd, zs, zc = znorm(s_nom), znorm(s_d), znorm(s_sp), znorm(s_cb)
    rn, rd, rs, rc = rrf_term(s_nom), rrf_term(s_d), rrf_term(s_sp), rrf_term(s_cb)
    kn, kd, ks, kc = rank_of(s_nom), rank_of(s_d), rank_of(s_sp), rank_of(s_cb)
    rrf_eq = rd + rs + rc
    X = []
    for i in idxs:
        ranks = (kn[i], kd[i], ks[i], kc[i])
        agree5 = sum(1 for r in ranks if r <= 5) / 4.0
        agree1 = sum(1 for r in ranks if r == 1) / 4.0
        maxz = max(zn[i], zd[i], zs[i], zc[i])
        X.append([1.0, zn[i], zd[i], zs[i], zc[i],
                  rn[i], rd[i], rs[i], rc[i],
                  1.0 / kn[i], 1.0 / kd[i], 1.0 / ks[i], 1.0 / kc[i],
                  maxz, agree5, agree1, rrf_eq[i]])
    return np.asarray(X, dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, l2=1.0, iters=1200, lr=0.3):
    w = np.zeros(X.shape[1])
    pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
    sw = np.where(y == 1, neg / pos, 1.0); sw = sw / sw.mean()
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (sw * (p - y)) / len(y) + l2 * np.r_[0.0, w[1:]] / len(y)
        w -= lr * g
    return w


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
    ap.add_argument("--cand", type=int, default=25)
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
    print(f"[learned2] corpus={N} q={len(items)} cand=top{args.cand} folds={args.folds}", file=sys.stderr, flush=True)

    # self-produced local cache — safe pickle.
    cache = Path(args.cache); cfile = cache / f"emb_{N}.pkl"
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

    Q = []
    cand_miss = 0
    for it in items:
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
        if not (gold & set(cand)):
            cand_miss += 1
        rrf_eq = rrf_term(s_d) + rrf_term(s_sp) + rrf_term(s_cb)
        base_rank = gold_rank(np.argsort(-rrf_eq).tolist(), gold)
        Q.append({"gold": gold, "X": feats(s_nom, s_d, s_sp, s_cb, cand),
                  "cand": cand, "base_rank": base_rank})
    if q_new:
        pickle.dump(qcache, open(qfile, "wb"))
    print(f"[learned2] candidate-set gold miss = {cand_miss}/{len(Q)}", file=sys.stderr, flush=True)

    # ---- 5-fold CV: deterministic modulo folds (reproducible, no RNG) ----
    base_h1 = sum(1 for q in Q if q["base_rank"] == 1) / len(Q)
    fold_gains, fold_learn = [], []
    for f in range(args.folds):
        test_i = [i for i in range(len(Q)) if i % args.folds == f]
        train_i = [i for i in range(len(Q)) if i % args.folds != f]
        Xtr = np.vstack([Q[i]["X"] for i in train_i])
        ytr = np.concatenate([[1.0 if c in Q[i]["gold"] else 0.0 for c in Q[i]["cand"]] for i in train_i])
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd == 0] = 1.0; mu[0] = 0.0; sd[0] = 1.0
        w = fit_logistic((Xtr - mu) / sd, ytr, l2=args.l2)
        h1b = h1l = 0
        for i in test_i:
            q = Q[i]
            h1b += 1.0 if q["base_rank"] == 1 else 0.0
            p = sigmoid(((q["X"] - mu) / sd) @ w)
            order = [q["cand"][j] for j in np.argsort(-p)]
            h1l += 1.0 if gold_rank(order, q["gold"]) == 1 else 0.0
        n = len(test_i)
        fold_gains.append((h1l - h1b) / n); fold_learn.append(h1l / n)
        print(f"  fold {f}: n={n} base={h1b/n:.4f} learned={h1l/n:.4f} Δ={(h1l-h1b)/n:+.4f}", file=sys.stderr, flush=True)

    g = np.array(fold_gains)
    print("\n" + "=" * 64)
    print(f"  LEARNED RERANKER — {args.folds}-fold CV (held-out hit@1)")
    print("=" * 64)
    print(f"  baseline rrf_equal hit@1 (full)  = {base_h1:.4f}")
    print(f"  learned hit@1 (CV mean)          = {np.mean(fold_learn):.4f}")
    print(f"  HELD-OUT GAIN  mean={g.mean():+.4f}  std={g.std():.4f}  min={g.min():+.4f}  max={g.max():+.4f}")
    print(f"  target >= +0.1 :  {'MET' if g.mean() >= 0.1 else 'not yet'}")
    print("=" * 64)

    # final model on ALL data (to ship), with standardization params
    Xall = np.vstack([q["X"] for q in Q])
    yall = np.concatenate([[1.0 if c in q["gold"] else 0.0 for c in q["cand"]] for q in Q])
    mu = Xall.mean(axis=0); sd = Xall.std(axis=0); sd[sd == 0] = 1.0; mu[0] = 0.0; sd[0] = 1.0
    w = fit_logistic((Xall - mu) / sd, yall, l2=args.l2)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "cv_gain_mean": float(g.mean()), "cv_gain_std": float(g.std()),
            "cv_learned_mean": float(np.mean(fold_learn)), "base_h1": base_h1,
            "weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "feat_names": FEAT_NAMES, "cand": args.cand, "l2": args.l2,
        }, indent=2))
        print(f"[learned2] wrote {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
