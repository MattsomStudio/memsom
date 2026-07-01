@echo off
REM Full 192-item poison head-to-head: memsom-vec, memsom-bm25, RAG, mem0.
REM memsom/RAG use global python (sqlite_vec); mem0 uses the isolated venv.
REM All judged (LLM utility), rate 0.5, gated where provenance exists.
cd /d C:\Users\you\memsom\bench

set REPO=C:\Users\you\memsom
set DS=C:\Users\you\lme_data\longmemeval_oracle.json
set GP=python
set VP=C:\Users\you\bench_venv\Scripts\python.exe
set OUT=C:\Users\you\h2h\results
set PERF=C:\Users\you\h2h\perf
set RUNS=C:\Users\you\h2h\runs

if not exist %OUT% mkdir %OUT%
if not exist %PERF% mkdir %PERF%
if not exist %RUNS% mkdir %RUNS%

echo ============ memsom-vec ============
%GP% run_headtohead.py --system memsom --repo %REPO% --run-root %RUNS%\memsom --dataset %DS% --rate 0.5 --gated --judge --workers 6 --out %OUT%\memsom.json
echo ============ memsom-bm25 ============
%GP% run_headtohead.py --system memsom --no-embed --repo %REPO% --run-root %RUNS%\memsom_bm25 --dataset %DS% --rate 0.5 --gated --judge --workers 6 --out %OUT%\memsom_bm25.json
echo ============ rag ============
%GP% run_headtohead.py --system rag --run-root %RUNS%\rag --dataset %DS% --rate 0.5 --judge --workers 6 --out %OUT%\rag.json
echo ============ mem0 (venv) ============
%VP% run_headtohead.py --system mem0 --run-root %RUNS%\mem0 --dataset %DS% --rate 0.5 --judge --workers 1 --out %OUT%\mem0.json

echo ============ per-write perf ============
%GP% perf_per_write.py --system memsom --repo %REPO% --run-root %PERF%\memsom_run --dataset %DS% --k 40 --out %PERF%\memsom.json
%GP% perf_per_write.py --system memsom --no-embed --repo %REPO% --run-root %PERF%\memsom_bm25_run --dataset %DS% --k 40 --out %PERF%\memsom_bm25.json
%GP% perf_per_write.py --system rag --run-root %PERF%\rag_run --dataset %DS% --k 40 --out %PERF%\rag.json
%VP% perf_per_write.py --system mem0 --run-root %PERF%\mem0_run --dataset %DS% --k 40 --out %PERF%\mem0.json

echo ============ HEAD-TO-HEAD TABLE ============
%GP% compare.py --dir %OUT% --perf-dir %PERF%
echo ============ DONE ============
