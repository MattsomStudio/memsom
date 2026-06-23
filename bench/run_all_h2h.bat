@echo off
REM Full 192-item poison head-to-head: memdag-vec, memdag-bm25, RAG, mem0.
REM memdag/RAG use global python (sqlite_vec); mem0 uses the isolated venv.
REM All judged (LLM utility), rate 0.5, gated where provenance exists.
cd /d C:\Users\you\memdag\bench

set REPO=C:\Users\you\memdag
set DS=C:\Users\you\lme_data\longmemeval_oracle.json
set GP=python
set VP=C:\Users\you\bench_venv\Scripts\python.exe
set OUT=C:\Users\you\h2h\results
set PERF=C:\Users\you\h2h\perf
set RUNS=C:\Users\you\h2h\runs

if not exist %OUT% mkdir %OUT%
if not exist %PERF% mkdir %PERF%
if not exist %RUNS% mkdir %RUNS%

echo ============ memdag-vec ============
%GP% run_headtohead.py --system memdag --repo %REPO% --run-root %RUNS%\memdag --dataset %DS% --rate 0.5 --gated --judge --workers 6 --out %OUT%\memdag.json
echo ============ memdag-bm25 ============
%GP% run_headtohead.py --system memdag --no-embed --repo %REPO% --run-root %RUNS%\memdag_bm25 --dataset %DS% --rate 0.5 --gated --judge --workers 6 --out %OUT%\memdag_bm25.json
echo ============ rag ============
%GP% run_headtohead.py --system rag --run-root %RUNS%\rag --dataset %DS% --rate 0.5 --judge --workers 6 --out %OUT%\rag.json
echo ============ mem0 (venv) ============
%VP% run_headtohead.py --system mem0 --run-root %RUNS%\mem0 --dataset %DS% --rate 0.5 --judge --workers 1 --out %OUT%\mem0.json

echo ============ per-write perf ============
%GP% perf_per_write.py --system memdag --repo %REPO% --run-root %PERF%\memdag_run --dataset %DS% --k 40 --out %PERF%\memdag.json
%GP% perf_per_write.py --system memdag --no-embed --repo %REPO% --run-root %PERF%\memdag_bm25_run --dataset %DS% --k 40 --out %PERF%\memdag_bm25.json
%GP% perf_per_write.py --system rag --run-root %PERF%\rag_run --dataset %DS% --k 40 --out %PERF%\rag.json
%VP% perf_per_write.py --system mem0 --run-root %PERF%\mem0_run --dataset %DS% --k 40 --out %PERF%\mem0.json

echo ============ HEAD-TO-HEAD TABLE ============
%GP% compare.py --dir %OUT% --perf-dir %PERF%
echo ============ DONE ============
