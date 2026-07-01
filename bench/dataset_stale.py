"""dataset_stale — load STALENESS benchmark items in one normalized schema.

Distinct from dataset.py (poison). A staleness item is a fact under a stable
source_ref that gets a LEGITIMATE same-channel UPDATE (the deadline moved, the
on-call changed) — not an external poison. The harness seeds v1, derives an
answer, applies the update, then measures two families:

  attribution  — does the system know the derived answer now depends on a
                 CHANGED source? (memsom: exact via cascade; others: n/a)
  serving      — after the update, does the re-asked answer reflect v2 (fresh),
                 v1 (stale), and is staleness flagged?

Normalized item schema:
  {
    "id":            str,
    "kind":          "updated" | "control",   # control = no update (precision)
    "question":      str,
    "source_ref":    str,                      # stable id so re-ingest supersedes
    "channel":       str,                      # same channel for v1 and the update
    "evidence_v1":   str,                      # the original fact
    "v1_gold_terms": [str, ...],               # old answer terms (stale-serve signal)
    "update_text":   str | None,               # v2 fact (None for control)
    "v2_gold_terms": [str, ...] | None         # new answer terms (fresh-serve / gold)
  }

Two sources:
  1. fixtures/staleness_min.json — runs immediately, proves the pipe + a real
     attribution/precision number on a clean curated set.
  2. LongMemEval knowledge-update via from_longmemeval_update — scale. HONEST
     LIMIT: LME does not structurally label the OLD value, so its stale_serve
     signal is unavailable; we drive attribution (we control the update) and
     fresh_serve (gold = the LME answer) and SAY SO (no v1_gold -> stale_serve n/a).
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "staleness_min.json"


def load_stale_fixture(path: str | Path = FIXTURE) -> list[dict]:
    """Load a curated staleness fixture. A top-level `distractors` pool (if present)
    is attached to every item's `distractors` key, so --haystack mode seeds the
    same noise for each item uniformly."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pool = data.get("distractors", [])
    items = data["items"]
    for it in items:
        it.setdefault("distractors", list(pool))
    return items


_UPDATE_TYPE = "knowledge-update"


def from_longmemeval_update(path: str | Path, max_items: int | None = None,
                            max_distractors: int = 30
                            ) -> tuple[list[dict], dict]:
    """Adapt LongMemEval knowledge-update entries into the staleness schema.

    We CONTROL the update (seed an early turn as v1 at a synthetic source_ref,
    then re-ingest the answer-bearing turn as v2 at the SAME ref), so attribution
    is exact regardless of LME's semantics. gold (v2) = the LME answer.

    HAYSTACK: every OTHER turn in the item's sessions is carried as a `distractors`
    list. The orchestrator (in --haystack mode) seeds those as noise sources under
    their own refs, so retrieval must FIND the right node among ~dozens — the
    pressure that makes --prefer-fresh (substitute the buried fresh head via the
    supersedes edge) diverge from --fresh-only (exclude, which fails when the fresh
    node never made top-k). Serving is scored by recall of the EXACT seeded v1/v2
    strings against the answer's citations (id-free, no old-value needed).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[dict] = []
    skipped: dict[str, int] = {}

    for entry in raw:
        if entry.get("question_type") != _UPDATE_TYPE:
            skipped[entry.get("question_type", "unknown")] = (
                skipped.get(entry.get("question_type", "unknown"), 0) + 1)
            continue
        gold = str(entry.get("answer", "")).strip()
        if not gold:
            skipped["empty-answer"] = skipped.get("empty-answer", 0) + 1
            continue

        answer_turns, context_turns = [], []
        for session in entry.get("haystack_sessions", []):
            for turn in session:
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                (answer_turns if turn.get("has_answer") else context_turns).append(content)
        if not answer_turns:
            skipped["no-answer-turn"] = skipped.get("no-answer-turn", 0) + 1
            continue

        qid = entry.get("question_id", f"lme-upd-{len(items)}")
        # v1 = an earlier context turn (the pre-update state we control); v2 = the
        # answer-bearing turn (current truth). If no distinct context turn exists,
        # synthesize a neutral v1 so the update still fires.
        v1 = context_turns[0] if context_turns else f"(prior note re: {entry['question'].rstrip('?')})"
        # distractors = every remaining turn (other answer-bearing turns + the rest
        # of the context), bounded. These become the haystack noise.
        distractors = (answer_turns[1:] + context_turns[1:])[:max_distractors]
        items.append({
            "id": qid,
            "kind": "updated",
            "question": entry["question"],
            "source_ref": f"lme:{qid}.md",
            "channel": "user",
            "evidence_v1": v1,
            "v1_gold_terms": [],                 # no clean old VALUE -> serving scored by seeded-text recall
            "update_text": answer_turns[0],
            "v2_gold_terms": [gold],
            "distractors": distractors,
        })
        if max_items and len(items) >= max_items:
            break

    report = {"used": len(items), "skipped": skipped,
              "skipped_total": sum(skipped.values())}
    return items, report
