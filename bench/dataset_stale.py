"""dataset_stale — load STALENESS benchmark items in one normalized schema.

Distinct from dataset.py (poison). A staleness item is a fact under a stable
source_ref that gets a LEGITIMATE same-channel UPDATE (the deadline moved, the
on-call changed) — not an external poison. The harness seeds v1, derives an
answer, applies the update, then measures two families:

  attribution  — does the system know the derived answer now depends on a
                 CHANGED source? (memdag: exact via cascade; others: n/a)
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
    return json.loads(Path(path).read_text(encoding="utf-8"))["items"]


_UPDATE_TYPE = "knowledge-update"


def from_longmemeval_update(path: str | Path, max_items: int | None = None
                            ) -> tuple[list[dict], dict]:
    """Adapt LongMemEval knowledge-update entries into the staleness schema.

    We CONTROL the update (seed an early turn as v1 at a synthetic source_ref,
    then re-ingest the answer-bearing turn as v2 at the SAME ref), so attribution
    is exact regardless of LME's semantics. gold (v2) = the LME answer, giving a
    real fresh_serve number. v1_gold is left empty -> the scorer reports
    stale_serve as n/a for these items (documented honest limit, not faked).
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
        items.append({
            "id": qid,
            "kind": "updated",
            "question": entry["question"],
            "source_ref": f"lme:{qid}.md",
            "channel": "user",
            "evidence_v1": v1,
            "v1_gold_terms": [],                 # LME gives no clean old value -> stale_serve n/a
            "update_text": answer_turns[0],
            "v2_gold_terms": [gold],
        })
        if max_items and len(items) >= max_items:
            break

    report = {"used": len(items), "skipped": skipped,
              "skipped_total": sum(skipped.values())}
    return items, report
