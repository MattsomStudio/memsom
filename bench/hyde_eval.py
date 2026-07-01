"""hyde_eval — HyDE (Hypothetical Document Embeddings) as a FIRST-STAGE recall
boost, the untried lever that targets the volume types reranking can't help
(temporal, multi-session): the LLM writes a hypothetical answer passage, we embed
it with nomic and retrieve with THAT (closes the question<->answer-passage gap),
fuse with the BM25+nomic baseline, then optionally m3 cross-encode.

Reuses the full500 nomic corpus cache. Caches HyDE generations. Reports hit@1
overall + by type for: baseline / +HyDE / +HyDE+cross.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pickle, sys, time, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np

BENCH = r"C:\Users\you\memsom\bench"; REPO = r"C:\Users\you\memsom"
sys.path.insert(0, BENCH); sys.path.insert(0, REPO)
from memsom_retrieve import tokenize                       # noqa: E402

RRF_C = 60; K1 = 1.2; B = 0.75
NOMIC_URL = "http://localhost:11434/api/embed"; NOMIC_MODEL = "nomic-embed-text"
LLM_URL = "http://localhost:11434/api/chat"; LLM_MODEL = "qwen2.5:7b-instruct"


def nomic_embed_one(text, is_query=True):
    pre = "search_query: " if is_query else "search_document: "
    body = json.dumps({"model": NOMIC_MODEL, "input": [pre + text]}).encode()
    req = urllib.request.Request(NOMIC_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return np.asarray(json.loads(r.read())["embeddings"][0], dtype=np.float32)


def llm_hyde(question, timeout=60):
    prompt = ("Write a short, plausible passage (1-3 sentences) that would directly answer "
              "this question, as if recalled from a past conversation. Do not hedge.\n\n"
              f"Question: {question}\n\nPassage:")
    body = json.dumps({"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "options": {"temperature": 0}}).encode()
    try:
        req = urllib.request.Request(LLM_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (json.loads(r.read()).get("message", {}).get("content", "") or "").strip()
    except Exception:
        return ""


def load_all(path, max_evidence=6):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for e in raw:
        ans, ctx = [], []
        for sess in e.get("haystack_sessions", []):
            for t in sess:
                c = (t.get("content") or "").strip()
                if not c:
                    continue
                (ans if t.get("has_answer") else ctx).append(c)
        if not ans:
            continue
        ev = [{"text": t, "ab": True} for t in ans] + [{"text": t, "ab": False} for t in ctx[:max(0, max_evidence-len(ans))]]
        items.append({"id": e.get("question_id", f"q{len(items)}"), "question": e["question"],
                      "type": e.get("question_type", "?"), "evidence": ev})
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
    ap.add_argument("--cross", action="store_true")
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
    print(f"[hyde] questions={len(items)} corpus={N}", file=sys.stderr, flush=True)

    cache = Path(args.cache)
    blob = pickle.load(open(cache / f"nomic_full_{N}.pkl", "rb"))   # self-produced local cache (full500)
    nom_doc, qcache = blob["doc"], blob["q"]
    postings, doclen, avgdl, idf = bm25_setup(corpus)

    # HyDE cache: question -> (hypothetical passage, its nomic embedding)
    hfile = cache / f"hyde_{N}.pkl"
    hyde = pickle.load(open(hfile, "rb")) if hfile.exists() else {}
    new = 0
    t0 = time.time()
    for qn, it in enumerate(items, 1):
        k = hashlib.sha1(it["question"].encode()).hexdigest()
        if k not in hyde:
            doc = llm_hyde(it["question"])
            emb = nomic_embed_one(doc, is_query=False) if doc else None
            hyde[k] = {"doc": doc, "emb": None if emb is None else emb.tolist()}
            new += 1
            if new % 50 == 0:
                print(f"   HyDE gen {new} ({(time.time()-t0)/new:.1f}s/q)", file=sys.stderr, flush=True)
    if new:
        pickle.dump(hyde, open(hfile, "wb"))
        print(f"[hyde] generated {new} hypotheticals", file=sys.stderr, flush=True)

    reranker = None
    if args.cross:
        print("[hyde] loading m3 cross-encoder ...", file=sys.stderr, flush=True)
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        rtok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3", use_fast=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            rmodel = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-v2-m3", dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        except RuntimeError:
            dev = "cpu"; rmodel = AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-v2-m3", dtype=torch.float32).to(dev).eval()

        def reranker(q, docs, batch=16):
            out = []
            for i in range(0, len(docs), batch):
                inp = rtok([[q, d] for d in docs[i:i+batch]], padding=True, truncation=True,
                           return_tensors="pt", max_length=512).to(dev)
                with torch.no_grad():
                    out.extend(rmodel(**inp).logits.view(-1).float().cpu().numpy().tolist())
            return out

    agg = defaultdict(lambda: [0, 0, 0, 0])   # type -> [n, base, hyde, hyde_cross]
    tot = [0, 0, 0, 0]
    for it in items:
        gold = gold_idx[it["id"]]; q = it["question"]
        qv = qcache[hashlib.sha1(q.encode()).hexdigest()]
        s_nom = cos_all(qv, nom_doc)
        s_bm = bm25_q(q, postings, doclen, avgdl, idf, N)
        base = rrf_term(s_bm) + rrf_term(s_nom)
        base_order = np.argsort(-base).tolist()
        b = 1.0 if base_order[0] in gold else 0.0
        # + HyDE signal
        h = hyde[hashlib.sha1(q.encode()).hexdigest()]
        if h["emb"] is not None:
            s_hy = cos_all(np.asarray(h["emb"], dtype=np.float32), nom_doc)
            hyfused = rrf_term(s_bm) + rrf_term(s_nom) + rrf_term(s_hy)
        else:
            hyfused = base
        hy_order = np.argsort(-hyfused).tolist()
        hh = 1.0 if hy_order[0] in gold else 0.0
        # + cross-encoder on HyDE candidates
        hc = hh
        if reranker is not None:
            cand = hy_order[:args.topk]
            sc = reranker(q, [corpus[i] for i in cand])
            hc = 1.0 if cand[int(np.argmax(sc))] in gold else 0.0
        a = agg[it["type"]]; a[0] += 1; a[1] += b; a[2] += hh; a[3] += hc
        tot[0] += 1; tot[1] += b; tot[2] += hh; tot[3] += hc

    n = tot[0]
    print("\n" + "=" * 78)
    print(f"  HyDE FIRST-STAGE BOOST (full {n} q)  hit@1   [+cross={args.cross}]")
    print("=" * 78)
    print(f"  baseline (BM25+nomic)       = {tot[1]/n:.4f}")
    print(f"  + HyDE                      = {tot[2]/n:.4f}   (Δ {(tot[2]-tot[1])/n:+.4f})")
    if args.cross:
        print(f"  + HyDE + cross-encoder      = {tot[3]/n:.4f}   (Δ {(tot[3]-tot[1])/n:+.4f})")
    print(f"  target >= +0.1: {'MET' if max(tot[2]-tot[1], tot[3]-tot[1])/n >= 0.1 else 'no'}")
    print("-" * 78)
    print(f"  {'type':<26}{'n':>4}{'base':>9}{'+hyde':>9}{'+h+cross':>10}{'best_d':>9}")
    for t, a in sorted(agg.items(), key=lambda x: -x[1][0]):
        c = a[0]; bestd = max(a[2]-a[1], a[3]-a[1]) / c
        print(f"  {t:<26}{c:>4}{a[1]/c:>9.3f}{a[2]/c:>9.3f}{a[3]/c:>10.3f}{bestd:>+9.3f}")
    print("=" * 78)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base": tot[1]/n, "hyde": tot[2]/n, "hyde_cross": tot[3]/n if args.cross else None,
            "n": n, "by_type": {t: {"n": a[0], "base": a[1]/a[0], "hyde": a[2]/a[0],
                                    "hyde_cross": a[3]/a[0]} for t, a in agg.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
