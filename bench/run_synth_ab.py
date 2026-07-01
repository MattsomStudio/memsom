"""run_synth_ab — isolate the citation-REPRESENTATION effect on judged util.

Same synthesizer + judge as the head-to-head (judge.synthesize / judge_correct).
Three payloads per clean item, content held as identical as possible:

  A  md_frag   : memsom's actual sentence-fragment citations (what the harness fed)
  B  md_joined : the SAME memsom content, regrouped by [mem:N] into whole nodes
  C  rag       : RAG's whole-turn citations (reference)

A vs B isolates the split itself with zero content difference. Saves every
answer + verdict + payload so divergent items can be re-judged by a stronger
model and hand-read (the 7B judge is not trusted on disputes).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict

import dataset
from adapters.memsom_adapter import MemsomAdapter
from adapters.rag_adapter import RagAdapter
from judge import synthesize, judge_correct

REPO = r"C:\Users\you\memsom"
PATH = r"C:\Users\you\lme_data\longmemeval_oracle.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\you\h2h\synth_ab.json"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5:7b-instruct")
JUDGE_URL = "http://localhost:11434/api/chat"

items, _ = dataset.from_longmemeval(PATH, max_items=N, max_evidence=3)
md = MemsomAdapter(REPO)
rag = RagAdapter()
ARMS = ("md_frag", "md_joined", "md_fullnode", "rag")
results = []

for n, it in enumerate(items, 1):
    gold = "; ".join(it["gold_terms"])

    d = tempfile.mkdtemp(prefix="md_")
    md.reset(d)
    for e in it["evidence"]:
        md.add(e["text"], e["channel"], e.get("answer_bearing", False))
    rmd = md.ask(it["question"], topk=8)
    md_cites = rmd.citations or []
    md_frag = [c.text for c in md_cites]
    bynode = defaultdict(list)
    for c in md_cites:
        bynode[c.node_id].append(c.text)
    md_joined = [" ".join(v) for v in bynode.values()]

    # md_fullnode: the FULL content of memsom's retrieved nodes (bypasses the
    # lossy compose render). retrieve() returns (id, content, channel, label, ref).
    import memsom, memsom_retrieve  # lazy: MemsomAdapter put repo on sys.path
    _conn = memsom.get_connection()  # uses MEMDAG_DB set by md.reset
    try:
        _rows = memsom_retrieve.retrieve(_conn, it["question"], k=8)
        md_fullnode = [r[1] for r in _rows]
    finally:
        _conn.close()

    d2 = tempfile.mkdtemp(prefix="rag_")
    rag.reset(d2)
    for e in it["evidence"]:
        rag.add(e["text"], e["channel"], e.get("answer_bearing", False))
    rrag = rag.ask(it["question"], topk=8)
    rag_mems = [c.text for c in (rrag.citations or [])]

    payloads = {"md_frag": md_frag, "md_joined": md_joined,
                "md_fullnode": md_fullnode, "rag": rag_mems}
    rec = {"id": it["id"], "q": it["question"], "gold": gold,
           "n_nodes": len(bynode), "n_frag": len(md_frag)}
    for arm in ARMS:
        ans = synthesize(it["question"], payloads[arm], model=JUDGE_MODEL, url=JUDGE_URL)
        ok = bool(judge_correct(it["question"], gold, ans, model=JUDGE_MODEL, url=JUDGE_URL))
        rec[arm + "_ok"] = ok
        rec[arm + "_ans"] = ans
        rec[arm + "_mems"] = payloads[arm]
    results.append(rec)
    if n % 10 == 0:
        print(f"  ...{n}/{len(items)}", flush=True)

util = {arm: sum(1 for r in results if r[arm + "_ok"]) / len(results) for arm in ARMS}
divergent = [r["id"] for r in results if r["md_frag_ok"] != r["md_joined_ok"]]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"n": len(results), "judge": JUDGE_MODEL, "util": util,
               "divergent_frag_vs_joined": divergent, "items": results},
              f, indent=2)

print(json.dumps({"n": len(results), "judge": JUDGE_MODEL, "util": util,
                  "n_divergent_frag_vs_joined": len(divergent)}, indent=2))
