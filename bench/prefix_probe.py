"""prefix_probe — isolate the effect of nomic task-prefixes on retrieval RANK.

Faithful 1:1 with the bench: imports dataset.from_longmemeval so the items are
byte-identical to the head-to-head run. The ONLY variable is whether we send
nomic-embed-text its intended task prefixes:

  RAW  (what memsom_retrieve._call_ollama_embed does today): prompt = text
  PFX  (what bench/adapters/rag_adapter does):
         documents -> "search_document: " + text
         query     -> "search_query: "    + text

Recall@k is saturated on this substrate (<=7 docs, topk=8) so we measure RANK:
  clean arm  (evidence only): does the ANSWER-BEARING turn rank above context?
  poison arm (evidence+poison): does true evidence outrank the poison?

No LLM synth, no judge -- pure retrieval geometry. Same Ollama endpoint memsom
and rag both use. Run from the bench dir so `import dataset` resolves.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request

import dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import memsom_retrieve as mr  # exact tokenize + BM25 constants + RRF

EMBED_URL = os.environ.get("MEMDAG_EMBED_URL") or "http://localhost:11434/api/embeddings"
EMBED_MODEL = os.environ.get("MEMDAG_EMBED_MODEL") or "nomic-embed-text"
TOPK = 8

_cache: dict[tuple[str, str], list] = {}


def embed(text: str, kind: str) -> list:
    """kind in {'raw','doc','query'} -> prefix policy."""
    if kind == "raw":
        prompt = text
    elif kind == "doc":
        prompt = "search_document: " + text
    elif kind == "query":
        prompt = "search_query: " + text
    else:
        raise ValueError(kind)
    key = (kind, text)
    if key in _cache:
        return _cache[key]
    body = json.dumps({"model": EMBED_MODEL, "prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(EMBED_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        vec = json.loads(r.read())["embedding"]
    _cache[key] = vec
    return vec


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def vec_rank(qvec, doc_vecs):
    """doc_vecs: list[vec] indexed by doc position. Returns [(idx, score)] desc."""
    scored = [(i, cosine(qvec, v)) for i, v in enumerate(doc_vecs)]
    scored.sort(key=lambda x: -x[1])
    return scored


def bm25_rank(query, doc_texts):
    """memsom's exact BM25 (mr.tokenize, mr.K1, mr.B) over a per-item corpus.

    Returns [(idx, score)] desc. Mirrors memsom_retrieve.bm25 math on a small
    in-memory corpus instead of the sqlite postings table.
    """
    qterms = set(mr.tokenize(query))
    tok = [mr.tokenize(t) for t in doc_texts]
    lengths = [len(t) for t in tok]
    N = len(doc_texts)
    avgdl = (sum(lengths) / N) if N else 0.0
    if not qterms or avgdl == 0.0:
        return [(i, 0.0) for i in range(N)]
    tf = [{} for _ in tok]
    for i, toks in enumerate(tok):
        for w in toks:
            tf[i][w] = tf[i].get(w, 0) + 1
    scores = [0.0] * N
    for term in qterms:
        df = sum(1 for m in tf if term in m)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for i in range(N):
            f = tf[i].get(term, 0)
            if not f:
                continue
            denom = f + mr.K1 * (1.0 - mr.B + mr.B * lengths[i] / avgdl)
            scores[i] += idf * (f * (mr.K1 + 1.0)) / denom
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return ranked


def order_to_tags(order, tags):
    """order: [(idx, score)] -> tags in that order."""
    return [tags[i] for i, _ in order]


def first_rank(order_tags, predicate):
    for i, tag in enumerate(order_tags, start=1):
        if predicate(tag):
            return i
    return None


MODES = ("vec_raw", "vec_pfx", "hyb_raw", "hyb_pfx")


def _accumulate(acc, arm, order_tags):
    a = acc[arm]
    ab_r = first_rank(order_tags, lambda t: t == "AB")
    a["n"] += 1
    a["top1_ab"] += 1 if order_tags and order_tags[0] == "AB" else 0
    a["mrr_ab"] += (1.0 / ab_r) if ab_r else 0.0
    if arm == "pois":
        poison_r = first_rank(order_tags, lambda t: t == "POISON")
        if ab_r and poison_r and ab_r < poison_r:
            a["ab_above_poison"] += 1
        if order_tags and order_tags[0] == "POISON":
            a["top1_poison"] += 1


def run(items):
    """One pass; produces clean+poison rank metrics for all four MODES."""
    acc = {m: {arm: {"n": 0, "top1_ab": 0, "mrr_ab": 0.0,
                     "ab_above_poison": 0, "top1_poison": 0}
               for arm in ("clean", "pois")}
           for m in MODES}

    for it in items:
        q_raw = embed(it["question"], "raw")
        q_pfx = embed(it["question"], "query")

        ev_tags = ["AB" if ev.get("answer_bearing") else "CTX" for ev in it["evidence"]]
        ev_txt = [ev["text"] for ev in it["evidence"]]
        ev_raw = [embed(t, "raw") for t in ev_txt]
        ev_pfx = [embed(t, "doc") for t in ev_txt]
        if not any(t == "AB" for t in ev_tags):
            continue

        # ----- clean arm: evidence only -----
        bm = bm25_rank(it["question"], ev_txt)
        vraw = vec_rank(q_raw, ev_raw)
        vpfx = vec_rank(q_pfx, ev_pfx)
        orders = {
            "vec_raw": vraw,
            "vec_pfx": vpfx,
            "hyb_raw": mr._rrf_fuse(bm, vraw),
            "hyb_pfx": mr._rrf_fuse(bm, vpfx),
        }
        for m in MODES:
            _accumulate(acc[m], "clean", order_to_tags(orders[m], ev_tags))

        # ----- poison arm: evidence + poison -----
        if it.get("poison"):
            p_tags = ev_tags + ["POISON"]
            p_txt = ev_txt + [it["poison"]["text"]]
            p_raw = ev_raw + [embed(it["poison"]["text"], "raw")]
            p_pfx = ev_pfx + [embed(it["poison"]["text"], "doc")]
            bmp = bm25_rank(it["question"], p_txt)
            vrawp = vec_rank(q_raw, p_raw)
            vpfxp = vec_rank(q_pfx, p_pfx)
            orders = {
                "vec_raw": vrawp,
                "vec_pfx": vpfxp,
                "hyb_raw": mr._rrf_fuse(bmp, vrawp),
                "hyb_pfx": mr._rrf_fuse(bmp, vpfxp),
            }
            for m in MODES:
                _accumulate(acc[m], "pois", order_to_tags(orders[m], p_tags))

    return acc


def _rate(d, key):
    return d[key] / d["n"] if d["n"] else 0.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\you\lme_data\longmemeval_oracle.json"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    items, report = dataset.from_longmemeval(path, max_items=limit)
    print(f"[probe] items={report['used']} skipped={report['skipped_total']} topk={TOPK}")

    t0 = time.time()
    acc = run(items)
    print(f"[probe] {time.time()-t0:.0f}s  embed_cache={len(_cache)}")
    print()
    print(f"{'mode':<10}{'clean_top1_ab':>15}{'clean_mrr_ab':>14}"
          f"{'pois_top1_poison':>18}{'pois_ab>poison':>16}")
    print("-" * 73)
    for m in MODES:
        c, p = acc[m]["clean"], acc[m]["pois"]
        print(f"{m:<10}{_rate(c,'top1_ab'):>15.3f}{_rate(c,'mrr_ab'):>14.3f}"
              f"{_rate(p,'top1_poison'):>18.3f}{_rate(p,'ab_above_poison'):>16.3f}")
    print()
    out = {m: {"clean": {k: _rate(acc[m]["clean"], k) for k in ("top1_ab", "mrr_ab")},
               "pois": {k: _rate(acc[m]["pois"], k)
                        for k in ("top1_ab", "mrr_ab", "top1_poison", "ab_above_poison")}}
           for m in MODES}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
