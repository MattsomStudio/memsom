"""audit_optimize — find where recall accuracy is lost, then optimize fusion.

Builds on run_recall_h2h but adds: (1) on-disk embedding cache so every trial
after the first is cheap, (2) per-item gold-rank logging across SEVERAL fusion
variants, (3) a disk-memoized LLM judge keyed on (question, top-k text set) so
sweeping top-k / arms never re-judges an identical context.

Fusion variants (bge signals dense/sparse/colbert, identical embeddings):
  bge_dense       dense cosine only
  rrf_equal       RRF(dense,sparse,colbert) equal weights        [the shipped one]
  rrf_weighted    RRF with dense=2,colbert=2,sparse=1
  wsum            normalized 0.4*dense+0.2*sparse+0.4*colbert     [BGE-M3 recommended]
  rerank          RRF(dense,sparse) -> top-50 -> colbert MaxSim rerank  [real recall_bge shape]
plus nomic (dense cosine, the old backend) as the floor.

Usage (PC):
  python audit_optimize.py --dataset ...oracle.json --max-items 200 \\
    --cache C:\\Users\\you\\h2h_cache --per-item C:\\Users\\you\\h2h_items.jsonl \\
    --judge --judge-arm rrf_equal --judge-topk 10 --judge-n 80
"""
from __future__ import annotations
import argparse, base64, hashlib, json, pickle, sys, time, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import from_longmemeval                # noqa: E402
from judge import synthesize, judge_correct         # noqa: E402

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


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i:i + n]


def cos_mat(q, M):
    qn = q / (np.linalg.norm(q) or 1.0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return Mn @ qn


def sparse_dot(a, b):
    if len(b) < len(a):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))


def maxsim(q, d):
    if q.size == 0 or d.size == 0:
        return 0.0
    return float((q @ d.T).max(axis=1).mean())


def rrf_from_scores(scores):
    order = np.argsort(-scores)
    out = np.zeros(len(scores))
    for r, i in enumerate(order, 1):
        out[i] = 1.0 / (RRF_C + r)
    return out


def norm01(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rank_all_variants(qnom, nom_doc, qd, qs, qc, bge_d, bge_s, bge_c, N):
    s_nom = cos_mat(qnom, nom_doc)
    s_bd = cos_mat(qd, bge_d)
    s_sp = np.array([sparse_dot(qs, bge_s[i]) for i in range(N)])
    s_cb = np.array([maxsim(qc, bge_c[i]) for i in range(N)])
    variants = {}
    variants["nomic"] = np.argsort(-s_nom).tolist()
    variants["bge_dense"] = np.argsort(-s_bd).tolist()
    variants["rrf_equal"] = np.argsort(-(rrf_from_scores(s_bd) + rrf_from_scores(s_sp) + rrf_from_scores(s_cb))).tolist()
    variants["rrf_weighted"] = np.argsort(-(2 * rrf_from_scores(s_bd) + 1 * rrf_from_scores(s_sp) + 2 * rrf_from_scores(s_cb))).tolist()
    variants["wsum"] = np.argsort(-(0.4 * norm01(s_bd) + 0.2 * norm01(s_sp) + 0.4 * norm01(s_cb))).tolist()
    # rerank: dense+sparse RRF -> top-50 candidates -> reorder by colbert maxsim
    cand = np.argsort(-(rrf_from_scores(s_bd) + rrf_from_scores(s_sp)))[:50].tolist()
    cand_sorted = sorted(cand, key=lambda i: -s_cb[i])
    rest = [i for i in range(N) if i not in set(cand)]
    rest_sorted = sorted(rest, key=lambda i: -(rrf_from_scores(s_bd)[i] + rrf_from_scores(s_sp)[i]))
    variants["rerank"] = cand_sorted + rest_sorted
    return variants


def gold_rank(ranking, gold_set):
    for r, i in enumerate(ranking, 1):
        if i in gold_set:
            return r
    return 0  # not found


def metrics_from_rank(ranking, gold_set, topk):
    top = ranking[:topk]
    h1 = 1.0 if top and top[0] in gold_set else 0.0
    h5 = 1.0 if any(i in gold_set for i in top[:5]) else 0.0
    rec = sum(1 for i in top if i in gold_set) / len(gold_set) if gold_set else 0.0
    gr = gold_rank(ranking, gold_set)
    mrr = 1.0 / gr if gr else 0.0
    return h1, h5, mrr, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--max-evidence", type=int, default=6)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--per-item", default=None)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-arm", default="rrf_equal")
    ap.add_argument("--judge-topk", type=int, default=10)
    ap.add_argument("--judge-n", type=int, default=80)
    ap.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--judge-url", default="http://localhost:11434/api/chat")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    items, report = from_longmemeval(args.dataset, max_items=args.max_items, max_evidence=args.max_evidence)
    print(f"[audit] used={report['used']} skipped={report['skipped_total']}", file=sys.stderr)
    corpus_text, corpus_item, corpus_gold, qtype = [], [], [], {}
    seen = {}
    # capture qtype per item id from raw (from_longmemeval drops it); re-read raw quickly
    raw = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    raw_type = {e.get("question_id", ""): e.get("question_type", "?") for e in raw}
    for it in items:
        qtype[it["id"]] = raw_type.get(it["id"], "?")
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
    print(f"[audit] corpus={N} questions={len(items)} topk={args.topk}", file=sys.stderr)

    # ---- embedding cache ----
    # pickle is safe here: this cache is written AND read only by this script on
    # the PC (self-produced local artifact, never an untrusted source). Used over
    # JSON because the payload is numpy arrays + colbert ndarrays + sparse dicts.
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    cfile = cache / f"emb_{N}.pkl"
    if cfile.exists():
        print("[audit] loading cached embeddings", file=sys.stderr)
        with open(cfile, "rb") as f:
            blob = pickle.load(f)
        nom_doc, bge_d, bge_s, bge_c = blob["nom"], blob["bd"], blob["bs"], blob["bc"]
    else:
        t0 = time.time()
        print("[audit] embedding corpus (nomic)…", file=sys.stderr)
        nom_doc = []
        for i, part in batched(corpus_text, 64):
            nom_doc.extend(nomic_embed(part, False))
        nom_doc = np.vstack(nom_doc)
        print("[audit] embedding corpus (bge)…", file=sys.stderr)
        bge_d, bge_s, bge_c = [], [], []
        for i, part in batched(corpus_text, 12):
            d, s, c = bge_embed(part); bge_d.extend(d); bge_s.extend(s); bge_c.extend(c)
            if i % 120 == 0:
                print(f"   bge {i}/{N}", file=sys.stderr)
        bge_d = np.vstack(bge_d)
        with open(cfile, "wb") as f:
            pickle.dump({"nom": nom_doc, "bd": bge_d, "bs": bge_s, "bc": bge_c}, f)
        print(f"[audit] embedded+cached in {time.time()-t0:.0f}s", file=sys.stderr)

    # ---- precompute ALL query embeddings up front (batched) ----
    # The first run crashed at ~q60: per-question embedding calls competed with the
    # qwen judge on the GPU and one hit a transient timeout that wasn't wrapped,
    # killing the whole run. Doing every query embed up front (a) removes that
    # contention entirely (embeds finish before any judging starts) and (b) is
    # faster (batched). Cached like the corpus so retries are instant.
    qcache = cache / f"qemb_{N}_{len(items)}.pkl"
    if qcache.exists():
        with open(qcache, "rb") as f:
            qb = pickle.load(f)
        q_nom, q_d, q_s, q_c = qb["nom"], qb["d"], qb["s"], qb["c"]
    else:
        questions = [it["question"] for it in items]
        q_nom = []
        for _i, part in batched(questions, 64):
            q_nom.extend(nomic_embed(part, True))
        q_nom = np.vstack(q_nom)
        q_d, q_s, q_c = [], [], []
        for _i, part in batched(questions, 12):
            d, s, c = bge_embed(part); q_d.extend(d); q_s.extend(s); q_c.extend(c)
        q_d = np.vstack(q_d)
        with open(qcache, "wb") as f:
            pickle.dump({"nom": q_nom, "d": q_d, "s": q_s, "c": q_c}, f)
        print("[audit] query embeddings cached", file=sys.stderr)

    # ---- judge memo ----
    memo_file = cache / "judge_memo.json"
    memo = json.loads(memo_file.read_text()) if memo_file.exists() else {}

    def judged(question, mems, gold):
        key = hashlib.sha1(("|".join([question] + sorted(mems))).encode()).hexdigest()
        if key in memo:
            return memo[key]
        ans = synthesize(question, mems, model=args.judge_model, url=args.judge_url)
        ok = bool(judge_correct(question, gold, ans, model=args.judge_model, url=args.judge_url))
        memo[key] = ok
        return ok

    ARMS = ["nomic", "bge_dense", "rrf_equal", "rrf_weighted", "wsum", "rerank"]
    agg = {a: {"hit1": 0, "hit5": 0, "mrr": 0.0, "recall": 0.0} for a in ARMS}
    util = {a: [] for a in ARMS}
    per_item_f = open(args.per_item, "w", encoding="utf-8") if args.per_item else None
    t0 = time.time(); failed = 0
    for qn, it in enumerate(items, 1):
        try:
            gold = gold_idx[it["id"]]; q = it["question"]
            qnom = q_nom[qn - 1]; qd = q_d[qn - 1]; qs = q_s[qn - 1]; qc = q_c[qn - 1]
            variants = rank_all_variants(qnom, nom_doc, qd, qs, qc, bge_d, bge_s, bge_c, N)
            rec = {"id": it["id"], "qtype": qtype.get(it["id"], "?"), "gold_n": len(gold), "ranks": {}}
            for a in ARMS:
                h1, h5, mrr, r = metrics_from_rank(variants[a], gold, args.topk)
                agg[a]["hit1"] += h1; agg[a]["hit5"] += h5; agg[a]["mrr"] += mrr; agg[a]["recall"] += r
                rec["ranks"][a] = gold_rank(variants[a], gold)
            if args.judge and qn <= args.judge_n:
                goldterms = "; ".join(it["gold_terms"])
                mems = [corpus_text[i] for i in variants[args.judge_arm][:args.judge_topk]]
                rec["util"] = judged(q, mems, goldterms)
                util[args.judge_arm].append(1.0 if rec["util"] else 0.0)
            if per_item_f:
                per_item_f.write(json.dumps(rec) + "\n"); per_item_f.flush()
        except Exception as e:  # one bad question must never kill the whole run
            failed += 1
            print(f"[audit] q{qn} {it.get('id')} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        if qn % 10 == 0:
            memo_file.write_text(json.dumps(memo))   # incremental: a crash never loses judge work
        if qn % 20 == 0 or qn == len(items):
            print(f"   {qn}/{len(items)} ({(time.time()-t0)/qn:.1f}s/q, failed={failed})", file=sys.stderr)
    if per_item_f:
        per_item_f.close()
    memo_file.write_text(json.dumps(memo))

    n = len(items)
    print("\n" + "=" * 70)
    print(f"  AUDIT/SWEEP  corpus={N} q={n} top-{args.topk}  judge_arm={args.judge_arm}@{args.judge_topk}")
    print("=" * 70)
    print(f"  {'arm':<14}{'hit1':>9}{'hit5':>9}{'mrr':>9}{'recall':>9}{'utility':>9}")
    for a in ARMS:
        u = f"{sum(util[a])/len(util[a]):.4f}" if util[a] else "-"
        print(f"  {a:<14}{agg[a]['hit1']/n:>9.4f}{agg[a]['hit5']/n:>9.4f}{agg[a]['mrr']/n:>9.4f}{agg[a]['recall']/n:>9.4f}{u:>9}")
    print("=" * 70 + "\n")
    if args.out:
        Path(args.out).write_text(json.dumps({a: {k: agg[a][k]/n for k in agg[a]} for a in ARMS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
