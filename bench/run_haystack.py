"""run_haystack — memsom on FULL LongMemEval-S haystack, Mem0's config.

Per question: fresh DB, BULK-insert all haystack turns (one migrate, one txn --
NOT the per-turn CLI add, which re-runs migrate_all 550x), embed once, retrieve
top-200, answer+judge with gpt-4o. No poison. Resumable JSONL. Records carry
stage timings (Windows buffers stdout, so timing lives in the file).

Usage: python run_haystack.py <out.jsonl> [limit] [topk]
"""
from __future__ import annotations
import json, os, sys, tempfile, time

import openai_judge
from adapters.memsom_adapter import MemsomAdapter

DATA = r"C:\Users\you\lme_data\longmemeval_s_cleaned.json"
REPO = r"C:\Users\you\memsom"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\you\h2h\haystack.jsonl"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
TOPK = int(sys.argv[3]) if len(sys.argv) > 3 else 200

data = json.load(open(DATA, encoding="utf-8"))
if LIMIT:
    data = data[:LIMIT]
done = set()
if os.path.exists(OUT):
    for line in open(OUT, encoding="utf-8"):
        try: done.add(json.loads(line)["id"])
        except Exception: pass

md = MemsomAdapter(REPO)
import memsom
from memsom.retrieval import retrieve as memsom_retrieve
from memsom.interface import cli as memsom_cli
from memsom.interface import ingest as memsom_ingest
LABEL = memsom_ingest.authoritative_label("user")
print(f"[haystack] {len(data)} q, {len(done)} done, topk={TOPK}", flush=True)

for n, e in enumerate(data, 1):
    qid = e["question_id"]
    if qid in done: continue
    q = e["question"]; gold = str(e.get("answer", "")); qtype = e.get("question_type", "?")
    turns = [(t.get("content") or "").strip()
             for s in e.get("haystack_sessions", []) for t in s if (t.get("content") or "").strip()]
    rec = {"id": qid, "type": qtype, "n_turns": len(turns)}
    try:
        d = tempfile.mkdtemp(prefix="hs_")
        md.reset(d)                                  # inits DB + sets MEMDAG_DB
        conn = memsom.get_connection()
        memsom_cli.migrate_all(conn)                 # ONCE
        t = time.time()
        with conn:
            for txt in turns:
                memsom.insert_node(conn, txt, "user", label=LABEL, source_ref=None)
        rec["ingest_s"] = round(time.time() - t, 1)
        t = time.time(); memsom_retrieve.index_all(conn); rec["embed_s"] = round(time.time() - t, 1)
        t = time.time(); rows = memsom_retrieve.retrieve(conn, q, k=TOPK); rec["retr_s"] = round(time.time() - t, 1)
        conn.close()
        mems = [r[1] for r in rows]; rec["n_retrieved"] = len(mems)
        t = time.time(); ans = openai_judge.synthesize(q, mems); rec["synth_s"] = round(time.time() - t, 1)
        t = time.time(); rec["ok"] = bool(openai_judge.judge_correct(q, gold, ans)); rec["judge_s"] = round(time.time() - t, 1)
    except Exception as ex:
        rec["ok"] = False; rec["error"] = f"{type(ex).__name__}: {ex}"
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  {n}/{len(data)} {qid} ok={rec.get('ok')} turns={rec.get('n_turns')} "
          f"ingest={rec.get('ingest_s')} embed={rec.get('embed_s')} synth={rec.get('synth_s')}", flush=True)

rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
rows = [r for r in rows if r["id"] in {e["question_id"] for e in data}]
ov = sum(1 for r in rows if r.get("ok")) / len(rows) if rows else 0
print(f"\n[haystack] n={len(rows)} OVERALL={ov:.3f}  (Mem0 ref 0.944)", flush=True)
