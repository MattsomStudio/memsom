"""full500_eval — the COMPLETE benchmark (all 500 LongMemEval questions, ALL types,
not the poison-restricted 192 subset), with memdag's REAL shipped baseline
(BM25 + nomic-dense equal RRF) + an m3 cross-encoder rerank. No BGE service needed
(nomic via Ollama, up). Harder question types (multi-session, temporal) are where a
reranker pays off, so this is the honest place a >=+0.1 gain could legitimately live.

Frozen reranker -> full-data hit@1 gain is unbiased. Caches nomic corpus + query
embeddings so re-runs are cheap.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pickle, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memdag\bench"; REPO = r"C:\Users\you\memdag"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from memdag_retrieve import tokenize                       # noqa: E402

RRF_C = 60; K1 = 1.2; B = 0.75
NOMIC_URL = "http://localhost:11434/api/embed"; NOMIC_MODEL = "nomic-embed-text"


def nomic_embed(texts, is_query, batch=64):
    pre = "search_query: " if is_query else "search_document: "
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": NOMIC_MODEL, "input": [pre + t for t in texts[i:i+batch]]}).encode()
        req = urllib.request.Request(NOMIC_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            out.extend(json.loads(r.read())["embeddings"])
    return np.asarray(out, dtype=np.float32)


def load_all(path, max_evidence=6):
    """ALL question types; evidence = answer-bearing turns (gold) + a little context."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for e in raw:
        ans_turns, ctx = [], []
        for sess in e.get("haystack_sessions", []):
            for t in sess:
                c = (t.get("content") or "").strip()
                if not c:
                    continue
                (ans_turns if t.get("has_answer") else ctx).append(c)
        if not ans_turns:
            continue
        ev = [{"text": t, "ab": True} for t in ans_turns]
        ev += [{"text": t, "ab": False} for t in ctx[: max(0, max_evidence - len(ev))]]
        items.append({"id": e.get("question_id", f"q{len(items)}"),
                      "question": e["question"], "type": e.get("question_type", "?"), "evidence": ev})
    return items


def rrf_term(s):
    o = np.argsort(-s); out = np.zeros(len(s)); out[o] = 1.0 / (RRF_C + np.arange(1, len(s) + 1)); return out


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
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items = load_all(args.dataset, args.max_evidence)
    corpus, citem, cgold, seen = [], [], [], {}
    for it in items:
        for ev in it["evidence"]:
            key = (it["id"], ev["text"])
            if key in seen:
                continue
            seen[key] = len(corpus); corpus.append(ev["text"]); citem.append(it["id"]); cgold.append(ev["ab"])
    N = len(corpus)
    gold_idx = {}
    for i, (iid, g) in enumerate(zip(citem, cgold)):
        if g:
            gold_idx.setdefault(iid, set()).add(i)
    items = [it for it in items if gold_idx.get(it["id"])]
    print(f"[full500] questions={len(items)} corpus={N}", file=sys.stderr, flush=True)

    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"nomic_full_{N}.pkl"     # self-produced local cache
    if cf.exists():
        blob = pickle.load(open(cf, "rb")); nom_doc = blob["doc"]; qcache = blob["q"]
        print("[full500] loaded nomic cache", file=sys.stderr, flush=True)
    else:
        print("[full500] embedding corpus with nomic ...", file=sys.stderr, flush=True)
        t0 = time.time(); nom_doc = nomic_embed(corpus, False)
        print(f"[full500] embedding {len(items)} queries ...", file=sys.stderr, flush=True)
        qcache = {}
        qs = [it["question"] for it in items]
        qe = nomic_embed(qs, True)
        for it, e in zip(items, qe):
            qcache[hashlib.sha1(it["question"].encode()).hexdigest()] = e
        pickle.dump({"doc": nom_doc, "q": qcache}, open(cf, "wb"))
        print(f"[full500] embedded+cached in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    postings, doclen, avgdl, idf = bm25_setup(corpus)

    print("[full500] loading bge-reranker-v2-m3 ...", file=sys.stderr, flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", use_fast=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    except RuntimeError as exc:
        print(f"[full500] {dev} load failed ({exc}); falling back to CPU", file=sys.stderr, flush=True)
        dev = "cpu"; model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3", dtype=torch.float32).to(dev).eval()

    def cross(q, docs, batch=16):
        out = []
        for i in range(0, len(docs), batch):
            inp = tok([[q, d] for d in docs[i:i+batch]], padding=True, truncation=True,
                      return_tensors="pt", max_length=512).to(dev)
            with torch.no_grad():
                out.extend(model(**inp).logits.view(-1).float().cpu().numpy().tolist())
        return out

    base_h1 = cross_h1 = 0; ms = []
    by_type = defaultdict(lambda: [0, 0, 0])   # type -> [n, base_hits, cross_hits]
    t0 = time.time()
    for qn, it in enumerate(items, 1):
        gold = gold_idx[it["id"]]; q = it["question"]
        qv = qcache[hashlib.sha1(q.encode()).hexdigest()]
        s_nom = cos_all(qv, nom_doc)
        s_bm = bm25_q(q, postings, doclen, avgdl, idf, N)
        base = rrf_term(s_bm) + rrf_term(s_nom)            # memdag shipped
        order = np.argsort(-base).tolist()
        bh = 1.0 if order[0] in gold else 0.0
        base_h1 += bh
        cand = order[:args.topk]
        t1 = time.time(); sc = cross(q, [corpus[i] for i in cand]); ms.append((time.time()-t1)*1000)
        pick = cand[int(np.argmax(sc))]
        ch = 1.0 if pick in gold else 0.0
        cross_h1 += ch
        bt = by_type[it["type"]]; bt[0] += 1; bt[1] += bh; bt[2] += ch
        if qn % 50 == 0:
            print(f"   {qn}/{len(items)} base={base_h1/qn:.3f} cross={cross_h1/qn:.3f} ({(time.time()-t0)/qn:.2f}s/q)", file=sys.stderr, flush=True)

    n = len(items)
    print("\n" + "=" * 70)
    print(f"  FULL-500 (all types) — memdag BM25+nomic  vs  + m3 cross-encoder  (top-{args.topk})")
    print(f"  n={n}  corpus={N}  hit@1")
    print("=" * 70)
    print(f"  memdag baseline (BM25+nomic RRF) = {base_h1/n:.4f}")
    print(f"  + cross-encoder rerank           = {cross_h1/n:.4f}")
    print(f"  GAIN = {(cross_h1-base_h1)/n:+.4f}   (target >= +0.1: {'MET' if (cross_h1-base_h1)/n>=0.1 else 'no'})")
    print(f"  added latency/query: mean={np.mean(ms):.0f}ms")
    print("-" * 70)
    print("  by question type:")
    for t, (cnt, bh, ch) in sorted(by_type.items(), key=lambda x: -x[1][0]):
        print(f"    {t:<26} n={cnt:<4} base={bh/cnt:.3f} cross={ch/cnt:.3f} d={(ch-bh)/cnt:+.3f}")
    print("=" * 70)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base_h1": base_h1/n, "cross_h1": cross_h1/n, "gain": (cross_h1-base_h1)/n,
            "n": n, "corpus": N, "latency_ms": float(np.mean(ms)),
            "by_type": {t: {"n": c[0], "base": c[1]/c[0], "cross": c[2]/c[0]} for t, c in by_type.items()},
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
