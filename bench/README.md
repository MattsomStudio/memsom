# memsom integrity benchmark

Measures the property memsom actually claims: **deterministic channel-provenance
that a poison memory cannot launder up the integrity lattice.** Not "memsom never
shows bad data" — it's "memsom tracks where data came from precisely enough that a
floor can exclude it without reading its content."

## Why these metrics

| metric | question it answers |
|---|---|
| `utility` | gold term in the composed answer — can it still answer? |
| `asr` | poison term in the composed answer — did the attack land? |
| `citation_asr` | was a poison **node** actually cited? (stricter) |
| `laundering` | poison cited *and* answer integrity > injection channel — Biba failed |
| `refuse` | how often it composed nothing |

ASR alone is gameable (refuse everything → ASR 0). The honest figure is the
**(utility, ASR) pair**.

## Run it

On the PC (memsom + Ollama live there):

```bat
cd C:\Users\you\memsom\bench

:: pipe validation against the bundled fixture (no download needed)
python run_bench.py --repo C:\Users\you\memsom --run-root C:\Users\you\bench_runs --rate 1.0 --out C:\Users\you\bench_runs\fixture_result.json

:: the headline number: real LongMemEval substrate
python run_bench.py --repo C:\Users\you\memsom --run-root C:\Users\you\bench_runs ^
  --dataset C:\path\to\longmemeval_s.json --max-items 200 --rate 0.5 ^
  --out C:\Users\you\bench_runs\lme_result.json
```

## What it does / does not prove yet

- **Proven (steps 2+3):** the harness measures the attack (ASR 0→1 under poison),
  utility is measurable, and **laundering holds at 0.00** — when external poison
  enters the pool the composed answer's integrity correctly drops to `external`
  (Biba low-water-mark), end-to-end through the shipping CLI. The provenance
  invariant is now *checked*, not asserted.
- **Not yet (step 4):** `ask` applies no integrity floor by default, so default
  ASR is high — that's expected. Next is a **gated arm** (floor excludes
  `external`) → ASR should drop to ~0 while utility holds, because we already
  proved the provenance to gate on is tracked correctly. That delta vs baselines
  (vanilla RAG / mem0 / Zep / SuperLocalMemory) is the killer figure.
- **Falsifiability (also step 4):** include attacks that *should* beat memsom —
  federation honor-system origin, cold-machine pre-redaction residual — and
  report where it loses.

## Isolation contract (memsom quirk)

`init --data-dir DIR` ignores `MEMDAG_DB` and builds `DIR/memdag.db`; every other
command honors `MEMDAG_DB`. The runner inits a fresh DB per item, then pins
`MEMDAG_DB` via the subprocess env (never shell `set`), so each item is an
independent trial with zero cross-item retrieval leakage.

## Files

- `runner.py` — CLI subprocess wrapper + `ask` output parser
- `dataset.py` — normalized item loader (bundled fixture + LongMemEval adapter)
- `poison.py` — deterministic poison selection by rate
- `score.py` — the four metrics + aggregation
- `run_bench.py` — orchestrator (clean vs poisoned arms)
- `fixtures/factrecall_min.json` — tiny built-in set for pipe validation
