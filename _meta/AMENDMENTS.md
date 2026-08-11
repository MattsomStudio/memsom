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
