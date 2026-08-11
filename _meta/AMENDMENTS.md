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
