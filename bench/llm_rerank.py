"""llm_rerank — strongest reranker: the local LLM (qwen2.5) picks, from the
baseline's top-K, the single passage that answers the question. Listwise, directly
hit@1-targeted. Settles whether +0.1 is crossable at all (it costs an LLM call/
query -> slower; that's the disclosed trade vs the 'no slower' constraint).

Baseline = 4-way equal RRF from cache. Reports honest full-data hit@1 gain.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pickle, re, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memsom\bench"; REPO = r"C:\Users\you\memsom"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from dataset import from_longmemeval                       # noqa: E402
from memsom_retrieve import tokenize                       # noqa: E402

RRF_C = 60; K1 = 1.2; B = 0.75
LLM_URL = "http://localhost:11434/api/chat"; LLM_MODEL = "qwen2.5:7b-instruct"


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


def llm_pick(question, passages, timeout=60):
    """Ask the LLM which passage (1..K) best answers the question. Returns 0-based
    index, or -1 on parse failure (caller keeps baseline order)."""
    block = "\n".join(f"[{i+1}] {p[:400]}" for i, p in enumerate(passages))
    prompt = ("You are ranking retrieved passages. Which single passage best answers the "
              "question? Reply with ONLY its number.\n\n"
              f"Question: {question}\n\nPassages:\n{block}\n\nBest passage number:")
    body = json.dumps({"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "options": {"temperature": 0}}).encode()
    try:
        req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read()).get("message", {}).get("content", "")
        m = re.search(r"\d+", txt)
        if m:
            k = int(m.group()) - 1
            if 0 <= k < len(passages):
                return k
    except Exception:
        pass
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=192)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=8)
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

    base_h1 = llm_h1 = 0; ms = []
    t0 = time.time()
    for qn, it in enumerate(items, 1):
        gold = gold_idx[it["id"]]; q = it["question"]
        v = qcache[hashlib.sha1(q.encode()).hexdigest()]
        s_d = cos_all(v["qd"], bge_d)
        s_sp = np.array([sparse_dot(v["qs"], bge_s[i]) for i in range(N)], float)
        s_cb = np.maximum.reduceat(v["qc"] @ BIGC.T, doc_starts, axis=1).mean(axis=0).astype(float)
        s_fts = bm25_q(q, postings, doclen, avgdl, idf, N)
        base = rrf_term(s_fts) + rrf_term(s_sp) + rrf_term(s_d) + rrf_term(s_cb)
        order = np.argsort(-base).tolist()
        base_h1 += 1.0 if order[0] in gold else 0.0
        cand = order[:args.topk]
        t1 = time.time()
        k = llm_pick(q, [corpus[i] for i in cand]); ms.append((time.time() - t1) * 1000)
        pick = cand[k] if k >= 0 else cand[0]      # parse fail -> keep baseline #1
        llm_h1 += 1.0 if pick in gold else 0.0
        if qn % 40 == 0:
            print(f"   {qn}/{len(items)} base={base_h1/qn:.3f} llm={llm_h1/qn:.3f} ({(time.time()-t0)/qn:.1f}s/q)", file=sys.stderr, flush=True)

    n = len(items)
    print("\n" + "=" * 64)
    print(f"  LLM LISTWISE RERANK (qwen2.5, top-{args.topk})  n={n}  hit@1")
    print("=" * 64)
    print(f"  baseline 4-way RRF hit@1 = {base_h1/n:.4f}")
    print(f"  + LLM rerank             = {llm_h1/n:.4f}")
    print(f"  GAIN = {(llm_h1-base_h1)/n:+.4f}   (target >= +0.1: {'MET' if (llm_h1-base_h1)/n>=0.1 else 'no'})")
    print(f"  added latency/query: mean={np.mean(ms):.0f}ms")
    print("=" * 64)
    if args.out:
        Path(args.out).write_text(json.dumps({"base_h1": base_h1/n, "llm_h1": llm_h1/n,
                                               "gain": (llm_h1-base_h1)/n, "latency_ms": float(np.mean(ms)),
                                               "topk": args.topk, "n": n}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
