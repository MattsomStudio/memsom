import json, tempfile, time
import openai_judge
from adapters.memsom_adapter import MemsomAdapter

e = json.load(open(r"C:\Users\you\lme_data\longmemeval_s_cleaned.json", encoding="utf-8"))[0]
turns = [(t.get("content") or "").strip() for s in e["haystack_sessions"] for t in s if (t.get("content") or "").strip()]
print("turns:", len(turns))
md = MemsomAdapter(r"C:\Users\you\memsom")
import memsom
from memsom.retrieval import retrieve as memsom_retrieve
d = tempfile.mkdtemp(prefix="prof_"); md.reset(d)
t=time.time(); [md.add(x,"user") for x in turns]; print(f"add-loop (CLI x{len(turns)}): {time.time()-t:.1f}s")
t=time.time(); md._call(["reindex"]); print(f"reindex(embed): {time.time()-t:.1f}s")
conn=memsom.get_connection(); t=time.time(); rows=memsom_retrieve.retrieve(conn,e["question"],k=200); print(f"retrieve top200: {time.time()-t:.1f}s ({len(rows)} rows)"); conn.close()
mems=[r[1] for r in rows]
t=time.time(); ans=openai_judge.synthesize(e["question"],mems); print(f"gpt-4o synth: {time.time()-t:.1f}s")
t=time.time(); ok=openai_judge.judge_correct(e["question"],e["answer"],ans); print(f"gpt-4o judge: {time.time()-t:.1f}s ok={ok}")
