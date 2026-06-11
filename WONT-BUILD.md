# Things We Will Not Build (yet)

Scope freeze adopted 2026-06-10. This file converts scope creep from a
willpower problem into a paperwork problem: adding anything below requires
deleting it from this list first, in its own commit, with a reason.

## Out of the demo slice (until demo #1 has shipped)

_(This section is now empty — everything below either shipped or was moved to the parking lot.)_

## Moved off this list — built 2026-06-11

- Multi-hop label recompute — demo #1 shipped; elevation needs descendants to
  re-floor — `memdag_recompute`
- Consolidation gates / quarantine — demo #2 closer — `memdag_quarantine` + `demo2.ps1`
- Redaction — Guarantee #6 second mode is now code: content destroyed, shape
  survives — `memdag_redact`
- LLM in answer path — opt-in `--llm` only; default stays deterministic; citation
  firewall falls back on any uncited claim — `memdag_llm`
- Anticipatory coprocess — heuristic surprise-gating, honest no-ML — `memdag_anticipatory`
- LoRA/weights distillation — active to the GPU boundary; provenance-filtered
  exports, fine-tune stays the one manual step — `memdag_distill`
- Trust algebra — lattice + audited elevation — `memdag_trust`
- Federation + the deletion bug — death is monotonic cross-machine,
  first-death-wins, no resurrection — `memdag_federation`
- Bell-LaPadula confidentiality labels — orthogonal conf axis, MAX not min —
  `memdag_confid`

## Still out of scope (parking lot / never)

- Any UI — the CLI is the demo
- Touching `~/.claude/episodic/` — that is the recall engine, not the DAG
- Enterprise anything
