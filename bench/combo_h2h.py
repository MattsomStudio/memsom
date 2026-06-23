"""combo_h2h — fast all-combos retrieval head-to-head over the cached BGE/nomic embeddings.

Reuses the embedding cache audit_optimize.py wrote (emb_<N>.pkl) so NO corpus
re-embedding. Caches query embeddings too, so the 2nd run onward is instant.

Speed fix vs audit_optimize: ColBERT maxsim was 860 tiny per-doc numpy matmuls
per query (BLAS-thread oversubscription -> ~20s/q). Here it's ONE matmul
qc @ BIGC.T over a concatenated doc-token matrix, then np.maximum.reduceat to
segment-max per doc -> ~vectorized, ~0.05s/q.

Combos tested (all share the SAME embeddings — only fusion differs):
  nomic                 dense cosine, old backend (floor)
  dense / sparse / colbert            each BGE signal alone
  dense+sparse / dense+colbert / sparse+colbert      RRF pairs
  rrf_equal             RRF(dense,sparse,colbert) equal              [SHIPPED]
  rrf_weighted          RRF colbert*2, dense*1.5, sparse*1
  wsum_z                z-normed 0.45*colbert+0.35*dense+0.20*sparse
  rerankK               dense+sparse RRF -> top-K -> colbert reorder (K=10,20,40)
  rerank_tie            triple RRF -> top-20 -> colbert + 0.3*first-stage

Honest selection: metrics reported on TRAIN (first half) and TEST (held-out
second half) separately; a +0.1 claim must hold on TEST.

Usage (PC):  python combo_h2h.py --dataset ...oracle.json --max-items 150 --cache C:\\Users\\you\\h2h_cache
"""
from __future__ import annotations
import argparse, base64, hashlib, json, pickle, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import from_longmemeval                       # noqa: E402

NOMIC_URL = "http://localhost:11434/api/embed"; NOMIC_MODEL = "nomic-embed-text"
BGE_URL = "http://127.0.0.1:11435/embed"; RRF_C = 60


# ---------- embedding (only for queries; corpus comes from cache) ----------
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


# ---------- scoring primitives ----------
def cos_mat(q, M):
    qn = q / (np.linalg.norm(q) or 1.0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn


def sparse_dot(a, b):
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def rrf_from_scores(scores):
    order = np.argsort(-scores)
    out = np.zeros(len(scores))
    out[order] = 1.0 / (RRF_C + np.arange(1, len(scores) + 1))
    return out


def znorm(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


# ---------- metrics ----------
def metrics_from_rank(ranking, gold_set, topk):
    top = ranking[:topk]
    h1 = 1.0 if len(top) and top[0] in gold_set else 0.0
    h5 = 1.0 if any(i in gold_set for i in top[:5]) else 0.0
    rec = sum(1 for i in top if i in gold_set) / len(gold_set) if gold_set else 0.0
    mrr = 0.0
    for r, i in enumerate(ranking, 1):
        if i in gold_set:
            mrr = 1.0 / r; break
    return h1, h5, mrr, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=150)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--per-item", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, report = from_longmemeval(args.dataset, max_items=args.max_items, max_evidence=args.max_evidence)
    # rebuild corpus in the EXACT order audit_optimize used so the cache aligns
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
    print(f"[combo] corpus={N} questions={len(items)} topk={args.topk}", file=sys.stderr, flush=True)

    # pickle is safe here: emb_<N>.pkl / qcache.pkl are self-produced local artifacts
    # written AND read only by these bench scripts on the PC (numpy arrays + ragged
    # colbert ndarrays + sparse dicts — not JSON-friendly), never an untrusted source.
    cache = Path(args.cache)
    cfile = cache / f"emb_{N}.pkl"
    if not cfile.exists():
        print(f"[combo] FATAL: no embedding cache at {cfile} — run audit_optimize first", file=sys.stderr)
        return 1
    with open(cfile, "rb") as f:
        blob = pickle.load(f)
    nom_doc, bge_d, bge_s, bge_c = blob["nom"], blob["bd"], blob["bs"], blob["bc"]
    assert nom_doc.shape[0] == N and bge_d.shape[0] == N and len(bge_s) == N and len(bge_c) == N, "cache/corpus mismatch"

    # concatenate colbert doc tokens into one matrix + per-doc segment offsets
    doc_starts = np.zeros(N, dtype=np.int64)
    off = 0
    for i in range(N):
        doc_starts[i] = off
        off += bge_c[i].shape[0]
    BIGC = np.vstack(bge_c).astype(np.float32)          # (sum_tokens, 1024)
    print(f"[combo] BIGC={BIGC.shape} (concatenated colbert tokens)", file=sys.stderr, flush=True)

    # query-embedding cache
    qfile = cache / "qcache.pkl"
    qcache = pickle.load(open(qfile, "rb")) if qfile.exists() else {}
    q_new = 0

    def get_q(q):
        nonlocal q_new
        key = hashlib.sha1(q.encode()).hexdigest()
        if key in qcache:
            d = qcache[key]
            return d["nom"], d["qd"], d["qs"], d["qc"]
        qnom = nomic_embed([q], True)[0]
        qd, qs, qc = bge_embed([q]); qd, qs, qc = qd[0], qs[0], qc[0]
        qcache[key] = {"nom": qnom, "qd": qd, "qs": qs, "qc": qc}; q_new += 1
        return qnom, qd, qs, qc

    # fusion definitions: name -> function(scores_dict) -> ranking (list of corpus idx)
    def rankings(s_nom, s_d, s_sp, s_cb):
        r_d, r_sp, r_cb = rrf_from_scores(s_d), rrf_from_scores(s_sp), rrf_from_scores(s_cb)
        out = {}
        out["nomic"] = np.argsort(-s_nom).tolist()
        out["dense"] = np.argsort(-s_d).tolist()
        out["sparse"] = np.argsort(-s_sp).tolist()
        out["colbert"] = np.argsort(-s_cb).tolist()
        out["dense+sparse"] = np.argsort(-(r_d + r_sp)).tolist()
        out["dense+colbert"] = np.argsort(-(r_d + r_cb)).tolist()
        out["sparse+colbert"] = np.argsort(-(r_sp + r_cb)).tolist()
        out["rrf_equal"] = np.argsort(-(r_d + r_sp + r_cb)).tolist()           # SHIPPED
        out["rrf_weighted"] = np.argsort(-(1.5 * r_d + 1.0 * r_sp + 2.0 * r_cb)).tolist()
        out["wsum_z"] = np.argsort(-(0.45 * znorm(s_cb) + 0.35 * znorm(s_d) + 0.20 * znorm(s_sp))).tolist()
        # rerank variants: first-stage candidate set -> reorder by colbert
        for K in (10, 20, 40):
            firststage = r_d + r_sp
            cand = np.argsort(-firststage)[:K].tolist()
            cand_sorted = sorted(cand, key=lambda i: -s_cb[i])
            candset = set(cand)
            rest = [i for i in np.argsort(-firststage).tolist() if i not in candset]
            out[f"rerank{K}"] = cand_sorted + rest
        # rerank_tie: triple-RRF candidates, reorder by colbert + small first-stage tiebreak
        triple = r_d + r_sp + r_cb
        cand = np.argsort(-triple)[:20].tolist()
        fs_norm = znorm(triple)
        cand_sorted = sorted(cand, key=lambda i: -(znorm(s_cb)[i] + 0.3 * fs_norm[i]))
        candset = set(cand)
        rest = [i for i in np.argsort(-triple).tolist() if i not in candset]
        out["rerank_tie"] = cand_sorted + rest
        return out

    ARMS = ["nomic", "dense", "sparse", "colbert", "dense+sparse", "dense+colbert",
            "sparse+colbert", "rrf_equal", "rrf_weighted", "wsum_z",
            "rerank10", "rerank20", "rerank40", "rerank_tie"]
    half = len(items) // 2
    # accumulate metrics separately for train (first half) and test (second half)
    acc = {split: {a: np.zeros(4) for a in ARMS} for split in ("train", "test", "all")}
    cnt = {"train": 0, "test": 0, "all": 0}
    per_item_f = open(args.per_item, "w", encoding="utf-8") if args.per_item else None
    t0 = time.time()
    for qn, it in enumerate(items):
        gold = gold_idx[it["id"]]
        qnom, qd, qs, qc = get_q(it["question"])
        s_nom = cos_mat(qnom, nom_doc)
        s_d = cos_mat(qd, bge_d)
        s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)], dtype=np.float64)
        # vectorized colbert maxsim: one matmul + segment-max per doc
        sims = qc @ BIGC.T                                   # (nq_tokens, sum_tokens)
        seg_max = np.maximum.reduceat(sims, doc_starts, axis=1)   # (nq_tokens, N)
        s_cb = seg_max.mean(axis=0).astype(np.float64)           # (N,)
        ranks = rankings(s_nom, s_d, s_sp, s_cb)
        split = "train" if qn < half else "test"
        rec = {"id": it["id"], "gold_n": len(gold), "ranks": {}}
        for a in ARMS:
            m = np.array(metrics_from_rank(ranks[a], gold, args.topk))
            acc[split][a] += m; acc["all"][a] += m
            # gold rank for the log
            gr = next((r for r, i in enumerate(ranks[a], 1) if i in gold), 0)
            rec["ranks"][a] = gr
        cnt[split] += 1; cnt["all"] += 1
        if per_item_f:
            per_item_f.write(json.dumps(rec) + "\n")
        if (qn + 1) % 25 == 0 or qn + 1 == len(items):
            print(f"   {qn+1}/{len(items)} ({(time.time()-t0)/(qn+1):.2f}s/q)", file=sys.stderr, flush=True)
    if per_item_f:
        per_item_f.close()
    if q_new:
        pickle.dump(qcache, open(qfile, "wb"))
        print(f"[combo] cached {q_new} new query embeddings", file=sys.stderr, flush=True)

    def table(split):
        n = cnt[split] or 1
        print(f"\n{'='*72}\n  COMBO H2H — {split.upper()}  (q={cnt[split]}, top-{args.topk})\n{'='*72}")
        print(f"  {'arm':<16}{'hit1':>9}{'hit5':>9}{'mrr':>9}{'recall':>9}")
        base = acc[split]["rrf_equal"] / n
        rows = []
        for a in ARMS:
            v = acc[split][a] / n
            rows.append((a, v))
            print(f"  {a:<16}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}{v[3]:>9.4f}")
        print("-" * 72)
        # deltas vs shipped rrf_equal on hit1
        best = max(rows, key=lambda r: r[1][0])
        print(f"  SHIPPED rrf_equal hit1={base[0]:.4f} | BEST {best[0]} hit1={best[1][0]:.4f} "
              f"(Δ={best[1][0]-base[0]:+.4f})")
        return {a: (acc[split][a] / n).tolist() for a in ARMS}

    res = {"train": table("train"), "test": table("test"), "all": table("all"),
           "config": {"corpus": N, "questions": len(items), "topk": args.topk, "half": half}}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\n[combo] wrote {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
