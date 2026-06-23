"""citation_diff — what does memdag.ask hand the synthesizer vs RAG?

Clean arm only (no poison). Same evidence in, diff the res.citations out --
that list is exactly what run_headtohead.synthesize() consumes.
"""
from __future__ import annotations

import os
import sys
import tempfile

import dataset
from adapters.memdag_adapter import MemdagAdapter
from adapters.rag_adapter import RagAdapter

REPO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\you\memdag"
PATH = r"C:\Users\you\lme_data\longmemeval_oracle.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4

items, _ = dataset.from_longmemeval(PATH, max_items=N, max_evidence=3)


def show(tag, res, evidence):
    cites = res.citations or []
    print(f"  [{tag}] composed={res.composed} used={res.used} "
          f"considered={res.considered} n_citations={len(cites)} "
          f"node={getattr(res,'composed_node_id',None)} integ={getattr(res,'integrity',None)}")
    for i, c in enumerate(cites):
        t = (c.text or "").replace("\n", " ")
        print(f"      cite{i} ch={getattr(c,'channel',None)}: {t[:110]}")
    if not cites:
        print("      (no citations passed to synthesizer)")


md = MemdagAdapter(REPO)
rag = RagAdapter()

for it in items:
    ev = it["evidence"]
    n_ab = sum(1 for e in ev if e.get("answer_bearing"))
    print(f"\n=== {it['id']}  evidence={len(ev)} (answer_bearing={n_ab}) ===")
    print(f"  Q: {it['question'][:120]}")

    d = tempfile.mkdtemp(prefix="md_")
    md.reset(d)
    for e in ev:
        md.add(e["text"], e["channel"], e.get("answer_bearing", False))
    show("memdag", md.ask(it["question"], topk=8), ev)

    d2 = tempfile.mkdtemp(prefix="rag_")
    rag.reset(d2)
    for e in ev:
        rag.add(e["text"], e["channel"], e.get("answer_bearing", False))
    show("rag", rag.ask(it["question"], topk=8), ev)
