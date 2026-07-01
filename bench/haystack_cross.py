"""haystack_cross — the m3 cross-encoder on the UNSATURATED haystack (per-question,
~550 distractors), where a strong reranker should help most. Reuses the cached
per-question embeddings (haystack q_*.pkl); rebuilds corpus text deterministically
(same iteration order -> aligns with the cached signal vectors). Frozen reranker ->
full-data hit@1 gain is unbiased (no train/test needed). m3 is light (~1.2GB)."""
from __future__ import annotations
import argparse, base64, json, math, pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memsom\bench"; REPO = r"C:\Users\you\memsom"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from memsom_retrieve import tokenize                       # noqa: E402

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
    Nn = len(corpus); avgdl = doclen.mean() if Nn else 0.0
    idf = {w: math.log((Nn - len(d) + 0.5) / (len(d) + 0.5) + 1.0) for w, d in postings.items()}
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
    ap.add_argument("--max-q", type=int, default=30)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--cache", required=True)         # the haystack cache dir
    ap.add_argument("--device", default="auto")        # auto|cpu|cuda
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cache = Path(args.cache)

    print("[hc] loading bge-reranker-v2-m3 ...", file=sys.stderr, flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", use_fast=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    def cross(q, docs, batch=32):
        out = []
        for i in range(0, len(docs), batch):
            inp = tok([[q, d] for d in docs[i:i+batch]], padding=True, truncation=True,
                      return_tensors="pt", max_length=512).to(dev)
            with torch.no_grad():
                out.extend(model(**inp).logits.view(-1).float().cpu().numpy().tolist())
        return out

    base_h1 = cross_h1 = 0; used = 0; ms = []
    t0 = time.time()
    for e in data:
        if used >= args.max_q:
            break
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
        cf = cache / f"q_{e['question_id']}.pkl"
        if not cf.exists():
            continue
        blob = pickle.load(open(cf, "rb"))   # self-produced local cache
        bge_d, bge_s, bge_c, qd, qs, qc = blob["d"], blob["s"], blob["c"], blob["qd"], blob["qs"], blob["qc"]
        N = len(corpus)
        if bge_d.shape[0] != N:
            print(f"[hc] skip {e['question_id']}: corpus/embed mismatch {N} vs {bge_d.shape[0]}", file=sys.stderr); continue
        doc_starts = np.zeros(N, dtype=np.int64); off = 0
        for i in range(N):
            doc_starts[i] = off; off += bge_c[i].shape[0]
        BIGC = np.vstack(bge_c).astype(np.float32)
        s_d = cos_all(qd, bge_d)
        s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)], float)
        s_cb = np.maximum.reduceat(qc @ BIGC.T, doc_starts, axis=1).mean(axis=0).astype(float)
        postings, doclen, avgdl, idf = bm25_setup(corpus)
        s_fts = bm25_q(e["question"], postings, doclen, avgdl, idf, N)
        base = rrf_term(s_fts) + rrf_term(s_sp) + rrf_term(s_d) + rrf_term(s_cb)
        order = np.argsort(-base).tolist()
        base_h1 += 1.0 if order[0] in gold else 0.0
        cand = order[:args.topk]
        t1 = time.time()
        sc = cross(e["question"], [corpus[i] for i in cand]); ms.append((time.time()-t1)*1000)
        pick = cand[int(np.argmax(sc))]
        cross_h1 += 1.0 if pick in gold else 0.0
        used += 1
        if used % 10 == 0:
            print(f"   {used}/{args.max_q} base={base_h1/used:.3f} cross={cross_h1/used:.3f}", file=sys.stderr, flush=True)

    n = used
    print("\n" + "=" * 64)
    print(f"  HAYSTACK + m3 CROSS-ENCODER (per-q ~550 distractors, top-{args.topk})  n={n}  hit@1")
    print("=" * 64)
    print(f"  baseline 4-way RRF hit@1 = {base_h1/n:.4f}")
    print(f"  + cross-encoder rerank   = {cross_h1/n:.4f}")
    print(f"  GAIN = {(cross_h1-base_h1)/n:+.4f}   (target >= +0.1: {'MET' if (cross_h1-base_h1)/n>=0.1 else 'no'})")
    print(f"  added latency/query: mean={np.mean(ms):.0f}ms")
    print("=" * 64)
    if args.out:
        Path(args.out).write_text(json.dumps({"base_h1": base_h1/n, "cross_h1": cross_h1/n,
                                               "gain": (cross_h1-base_h1)/n, "n": n,
                                               "latency_ms": float(np.mean(ms)), "topk": args.topk}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
