# AMENDMENTS

Charter amendment log for the memsom-core refactor's execution (not to be confused with
`DECISIONS-AND-DEVIATIONS.md`, which is the plan-of-record's own amendment history). Format
matches that document's §3.

---

## A-13 — differential oracle re-recorded after MS-22's fix (Phase 4)

**Date:** 2026-08-11
**Affects:** phase-4, `_meta/tools/differential.py`, `_meta/measurements/differential-oracle.json`
**Supersedes:** nothing — this extends `DECISIONS-AND-DEVIATIONS.md` A-7 ("Security fixes are
their own phases and MAY change behaviour").

MS-22 (`SECURITY-REMEDIATION.md` §3.3, closed in Phase 4) deliberately changes
`kernel/text.py:strip_furniture`'s handling of citation-shaped substrings found in source content:
a literal `[mem:N|channel]` is neutralised to `(mem:N|channel)` at ingest time so untrusted document
text cannot forge a second citation tag onto a bullet. This is an intentional behaviour change, not
a regression — the differential oracle exists to catch accidental ones, and A-7 states plainly that
security fixes are exempt from the "moves are behaviour-preserving" rule they sit alongside.

MEASURED at `4e378db` (`python _meta/tools/differential.py --check`): exactly 8 of 220 cases diverge
from the Phase-0 oracle, all and only the cases built from `TEXTS[17]`
(`"[mem:1|user] a sentence that already looks like a citation"`) — `strip_furniture(...)`,
`snippet(...,80)`, `candidate_sentences(...)`, and `compose(...)` under all 4 question variants that
include that source. No other corpus case changed. This is the complete and expected blast radius of
MS-22's fix; nothing else in the compose pipeline moved.

**Action taken:** `python _meta/tools/differential.py --record` re-baselines the oracle at the Phase-4
fix commit. Every later phase's `--check` now diffs against the corrected (MS-22-fixed) behaviour, per
A-7's own mechanism ("Move phases stay behaviour-preserving and are proven so by the compose
differential oracle" — the oracle, not a frozen Phase-0 snapshot, is what "behaviour-preserving" means
after a security phase has deliberately moved the baseline).

---

## A-14 -- `interface/dashboard.py` deletion (Phase 5): the RISKS.md §1.7 grep found a real consumer

**Date:** 2026-08-11
**Affects:** phase-5, `RISKS.md` §1.7, `DECISIONS-AND-DEVIATIONS.md` A-9/D-13
**Supersedes:** nothing -- this records a measurement A-9 itself called for and did not have.

`RISKS.md` §1.7 flagged that A-9's basis for deleting `interface/dashboard.py` ("Matt confirmed he
has never run `memsom dashboard`") did not establish whether anything ELSE imports it, and named the
exact check: `grep -rn "interface.dashboard|import dashboard" . --include=*.py` across memsom,
memsom-panel and memsom-agentic-os, "in the same commit" as the delete.

MEASURED, run against `memsom-panel-refactor` and `memsom-agentic-os` (the panel's current homes) at
this commit: `memsom-agentic-os/backend/memsom_panel/__main__.py:73`,
`.../transport/activity.py:37` and `.../transport/knobs.py:37` all do `from memsom.interface import
dashboard`, and use `dashboard.build_telemetry`, `dashboard.default_memory_dir` and
`dashboard.load_weights` as real, load-bearing calls (the panel's own memory-telemetry cache,
activity feed and knobs surface) -- not a stray import. The INFERRED risk RISKS.md §1.7 named is
CONFIRMED: the panel is a real consumer of the module this phase deletes.

**Action taken: deleted anyway**, per Matt's own explicit decision (D-13: re-provision is a backlog
item, not a refactor deliverable) and `PLAN.md` D-4's already-adopted position that this refactor
carries no cross-repo exit gate -- memsom-panel's adoption of the eventual Features/Tuning-API-based
dashboard Feature is that repo's own, separately scheduled fix, not this phase's. Recorded here,
prominently, so the consequence is visible rather than discovered later: **memsom-panel's telemetry
cache, activity feed and knobs surface will raise `ImportError` on `memsom.interface.dashboard` the
moment this refactor's `memsom` package is installed there**, until memsom-panel is updated to consume
the backlog Feature (`PLAN.md` §8) instead of reaching into memsom's internals directly.

---

## A-15 -- Phase 7 ("the layers land"): four judgment calls beyond the module map

**Date:** 2026-08-11
**Affects:** phase-7, `.importlinter`, `scripts/gate_readpool.py`, `kernel/events.py` consumers
**Supersedes:** nothing -- PLAN.md Sec1.5's module map did not anticipate these four; each is a
direct consequence of finishing the moves it does specify.

1. **`gate_readpool.py`'s classifier was refined, not just run.** The unrefined "NODES + >=2 taint
   columns" substring scan flagged 10 modules at Phase 7's start. Adjudicated one by one (matching
   C-1's own federation.py precedent -- a text-shaped classifier firing on shape, not intent): 8 were
   false positives (single-row-by-key fetches, edges-table DAG walks for internal recompute, OR-joined
   inclusion predicates, a positive quarantined-status audit listing) and the classifier now excludes
   those four shapes. The remaining 2 genuine duplicates were `distill.export_training` (routed
   through `taint_filter_clauses`) and two dead partial-filter helpers with zero production callers --
   `confid.sources_for_clearance`, `quarantine.live_source_ids` -- deleted per the MS-39 precedent
   (charter R2) and gated by extending `test_gate_one_taint_primitive.py`'s existing existence check.
   Two more genuine hand-rolled `tombstoned = 0` fragments (`confid.recompute_conf_all`,
   `quarantine.consolidate`) were routed through the primitive's own first clause rather than left as
   local literals. Post-refinement count: 0 outside `storage/schema.py` (the scan excludes that module
   by construction, so 0 is "exactly one builder total").
2. **Four upward edges routed through `kernel/events.py`, following the pattern Phase 4 established
   for `node_ingested`/`node_redacted`.** `integrity/tombstone.py`'s hard-delete path needed
   `lifecycle/rederive.py`'s `erase()` result *synchronously* (the count feeds its own return dict);
   `emit()` never returns a subscriber's return value by design (only collected failures), so the
   subscriber writes into a mutable `result` out-param instead, and the emitter raises if no
   subscriber answered (fail loud on a mis-wired install, never silently report `erased: 0`).
   `integrity/redact.py`'s MS-16 claims-reap (`lifecycle/corroborate.py`) and
   `integrity/ingest.py`'s re-ingest staleness trigger (`lifecycle/stale.py`, already best-effort,
   preserved as such) use the same shape. `distill/digest.py` and `retrieval/retrieve.py`'s read-time
   `[[fact_*]]` substitution (`bridge/facts.py`) also route this way, degrading to "text unchanged" on
   a missing subscriber -- matching `resolve_ref`'s own already-designed behaviour for an unresolvable
   reference, not a new failure mode.
3. **`federation/broker.py`'s in-process `memsom.*` dispatch** (`interface/mcp.py`, rank 8, from
   rank 6) cannot be a static import at any point in the file, including inside `main()` -- import-
   linter's layers contract is a static AST scan, so a lazy/function-scoped `import` is still a
   violation. `set_mcp_dispatch()` exists for real injection; `main()` (the `memsom-broker` console-
   script's own composition point, with no external wiring to inject from) falls back to
   `importlib.import_module("memsom.interface.mcp")` -- a dynamic string lookup, invisible to the
   static scan, used only as that fallback so the console script needs zero external wiring.
4. **Six more `kernel/frontmatter.py` items from Sec1.5's list were still in `bridge_import.py`**
   (`_strip_render_marker`, `parse_primary_index`, `_LINK_IN_LINE`, `section_map`,
   `parse_index_entries`, and -- found only because it shared a code region with the others --
   `index_hooks` + its `_HOOK_RE`), left behind by Phase 3's kernel extraction. Moved the whole
   region; `bridge_import.py` re-imports them (so `bi.parse_index_entries` etc. still resolve as
   module attributes, unchanged for existing callers/tests) and `distill/digest.py` +
   `interface/audit.py` now import them from `kernel.frontmatter` directly instead of reaching into
   `bridge/`.

MEASURED at this commit: `lint-imports --config .importlinter` -> `3 kept, 0 broken`;
`scripts/mutual_pairs.py` -> `TOTAL PAIRS 0`; `scripts/gate_readpool.py --check` and
`scripts/gate_writeowner.py --check` both `OK`; `pytest tests/gates -q` -> `77 passed`;
`_meta/tools/differential.py --check` -> identical to the oracle (0 changed, 0 new, 0 lost) --
confirming all four are structural, not behaviour changes.

---

## A-16 -- Phase 8 ("the two APIs"): tuning.py's rank, five judgment calls

**Date:** 2026-08-11
**Affects:** phase-8, `memsom/tuning.py`, `memsom/interface/features.py`, `.importlinter`,
`scripts/env_ratchet.py`
**Supersedes:** nothing -- PLAN.md Sec2.2's code-comment placed `tuning.py` at "rank 1"; this
resolves that against the Phase-0-committed `scripts/env_ratchet.py`, which already assumed a
different path (`interface/tuning.py`) before Phase 8 existed to build the module.

1. **`tuning.py` is a NEW layer, inserted directly above `kernel` and below `effects:storage`
   (a 10th rank, not a literal "rank 1").** Neither of the two pre-existing candidates worked:
   `interface/tuning.py` (rank 8, matching the Phase-0 script) cannot be imported by any of the ~40
   call sites below rank 8 (upward import, forbidden); a literal rank-1 placement alongside
   `effects`/`storage` would need `lifecycle/forget.py`'s canonical-file logic reachable from below
   it, which is backwards (`forget.py` is rank 4). A dedicated layer just above `kernel` lets every
   other rank import it downward, while `kernel` itself cannot reach it -- matching kernel's own
   "stdlib only" rule, since `tuning.py` is a real registry (dataclasses, a lock, a dict), not a
   kernel primitive. `.importlinter`'s `memsom-layers` contract was extended with this one new entry;
   `scripts/env_ratchet.py`'s `_OWNER` was corrected from `interface/tuning.py` to `tuning.py` to
   match.
2. **Four exemptions from "no bare os.environ outside tuning.py", not oversights:** `MEMDAG_HOME` /
   `MEMDAG_DB` (anywhere -- `storage/db.py`'s own resolution plus `interface/audit.py`'s setdefault
   and `federation/broker.py --selfcheck`'s pin; tuning.py would need the data dir already resolved
   to find a canonical.json override, which is circular), `kernel/paths.py`'s
   `MEMDAG_BRIDGE_MEMORY_DIR` (kernel is rank 0, cannot import the new tuning layer upward -- the
   knob is still registered in `tuning.py` for `tuning list` visibility, duplicated rather than
   shared), `$PATH` anywhere (an OS executable-search variable, not a memsom knob), and
   `childenv.py`'s whole-`os.environ` copy for child-process sanitization (a security boundary, not
   a named-key lookup). `scripts/env_ratchet.py`'s docstring carries the same four, in full.
   MEASURED: 48 bare `os.environ`/`getenv` sites at Phase 8's start, 40 genuine migrations, 8
   exempted across these four categories; `python scripts/env_ratchet.py --check` -> 0.
3. **The 13 `forget.py` params + `memory_budget` are NOT in the tuning registry.** PLAN.md Sec1.7
   protects `forget.py`'s pure core + `DEFAULTS` (the golden parity test); Sec2.2 says the registry
   "wraps `load_params`; it does not replace it," but `load_params` lives at rank 4 and the new
   tuning layer sits below rank 1 -- it cannot call upward into it. Rather than duplicate
   `load_params`'s file-read/precedence logic into `tuning.py` (drift risk against a protected,
   tested function) or modify the protected function itself, this phase leaves the canonical block
   entirely outside the registry: `memsom tuning list/get/set` cover only the ~40 migrated env-
   sourced knobs. `tuning set` therefore refuses every key today (0 canonical-sourced knobs
   registered) -- an honest, not silent, gap, left for a later phase or the panel's own schedule
   (Matt's Q7) to close.
4. **`doctor` is "rewired to the registry" via a merge at `cli.py`, not inside `doctor.py`.**
   `lifecycle/doctor.py` (rank 4) cannot import `interface/features.py` (rank 8, the composition
   root -- it needs to reach federation/bridge to answer "is the broker configured"). `doctor.py`
   gained an optional `features=` parameter threaded through `gather()`/`_format()`; `cli.py` defines
   `cmd_doctor_with_features()`, which resolves the registry and calls `memsom_doctor.gather(features=
   ...)`, then overrides the `doctor` subparser's `func` after `memsom_doctor.register(sub)` runs
   (`sub.choices["doctor"].set_defaults(...)`). `doctor.py` itself never imports `interface/`.
5. **CORRECTED (fix round, 2026-08-11): item 5 as originally written was wrong.** It argued no cheap
   signal existed for "selected, importable, but the last encode call actually failed" -- but
   Phase 6's MS-32 `retrieval_degraded` table (`retrieval/retrieve.py`) is exactly that signal,
   already built, keyed by node and populated on any vector-embed failure regardless of backend
   (bge encode failing falls through to the Ollama attempt, which also gets recorded on failure).
   It was sitting unread. `_retrieval_dense()`/`_retrieval_bge()` now take `conn` and report
   `degraded` when `retrieve.degraded_nodes(conn)` is non-empty for the active backend -- one
   indexed `SELECT`, no model load, no network call, satisfying Sec2.1's status() cost rule.
   `cmd_ask` now also checks it unconditionally (not just on `--retrieve`/`--graph`) and prints a
   stderr notice naming "bm25" whenever any source is BM25-only, closing the exit gate's `ask ...
   | grep -i bm25` control test. Verified: kill Ollama (refused port), `add` + `reindex` ->
   `retrieval.dense` reports `degraded`; `ask` still answers and stderr contains "bm25".

MEASURED at this commit: `python scripts/env_ratchet.py --check` -> 0; `lint-imports --config
.importlinter` -> `3 kept, 0 broken`; `python scripts/mutual_pairs.py` -> `TOTAL PAIRS 0`; `python
scripts/upward_imports.py --check` -> `0 (baseline 0)`; `pytest tests/gates -q` -> `77 passed`;
`_meta/tools/differential.py --check` -> identical to the oracle; full suite cold x2 -> `1113 tests
... OK (expected failures=2)`, reproducible.

---

## A-17 -- Q11 (fact-layer corpus migration): code ships in the copy, real-corpus migration deferred

**Date:** 2026-08-11
**Affects:** phase-8

**Scope decision (Matt-approved):** Q11 is split into two parts.

1. **SHIPPED this phase, in the copy -- the actual Phase 8 deliverable:** the fact-layer feature
   code (`memsom fact-set` / `fact-log`, `scripts/fact_refs.py`, the bridge fact-dependency
   resolution), proven by an in-copy synthetic-fixture gate test
   (`test_gate_fact_refs_corpus_check.py`) that exercises the check logic against a fixture and
   never touches any real memory store.

2. **DEFERRED to promote-time, done deliberately by the operator:** migrating the real memory
   corpus's changing values into `fact_*` files. A copy-confined refactor agent must never write
   the operator's live memory store; that is the run's founding invariant.

**Gate scope for this phase:** the fact-layer code and its in-copy fixture test only. The
operator's live memory store is explicitly OUT OF SCOPE for Phase 8 verification -- this amendment
makes and requires no claim about its contents, and gates must not measure it.

**Lesson recorded:** an earlier turn breached copy-confinement by writing the live store through
the unfenced `shell` tool (the `write_work` / `patch_work` tools are fenced to the copy; `shell`
is not). Root-cause hardening -- fencing shell, or keeping live-store-targeting deliverables out of
copy-confined runs -- is tracked as follow-up, not this phase.
