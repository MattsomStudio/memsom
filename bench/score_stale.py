"""score_stale — staleness metrics from two AskResults (pre/post update).

The headline claim is ATTRIBUTION: after a source changes, can the system tell
you which stored answers now depend on it? memdag answers exactly (the cascade);
a system without provenance edges cannot answer at all -> reported n/a, NEVER
scored 0 (it isn't failing the test, it structurally cannot take it).

SERVING is the honest secondary: after the update, does the re-asked answer carry
the new value (fresh), the old value (stale), and is staleness flagged? We expect
memdag and a decent RAG to TIE on raw fresh_serve (both retrieve the new chunk) —
the differentiation is that only memdag can ATTRIBUTE and FLAG. Reporting that tie
is the point, not a weakness to hide.

  attribution_recall   updated items the system flagged as affected  (memdag ~1.0)
  attribution_fpr      control items wrongly flagged affected         (memdag ~0.0)
  fresh_serve_rate     post-update answer carries the v2 value
  stale_serve_rate     post-update answer carries v1 and NOT v2 (when v1 known)
  flagged_rate         post-update answer discloses staleness
"""

from __future__ import annotations

from dataclasses import dataclass

from runner import AskResult


def _hit(terms, text: str) -> bool:
    if not terms:
        return False
    t = text.lower()
    return any(str(term).lower() in t for term in terms)


@dataclass
class StaleScore:
    item_id: str
    kind: str                         # updated | control
    has_provenance: bool
    composed: bool
    affected: bool | None             # system's attribution verdict for the pre-update answer
    attribution_correct: bool | None  # verdict matches ground truth (None if no provenance)
    fresh_present: bool               # post-update answer CONTAINS the v2 value (lenient)
    served_stale: bool | None         # answer still carries v1 post-update (None if v1 unknown)
    fresh_clean: bool | None          # v2 present AND v1 absent — actually RESOLVED (None if v1 unknown)
    flagged: bool                     # staleness disclosed


def score_stale_item(item: dict, a2: AskResult, affected: bool | None,
                     flagged: bool, has_provenance: bool) -> StaleScore:
    kind = item.get("kind", "updated")
    v1 = item.get("v1_gold_terms") or []
    v2 = item.get("v2_gold_terms") or []
    ans = a2.answer_text or ""

    has_v2 = _hit(v2, ans)
    has_v1 = _hit(v1, ans)

    # ground truth: an 'updated' item's prior answer IS affected; a 'control'
    # item's is NOT. attribution_correct only defined where provenance exists.
    if not has_provenance or affected is None:
        attribution_correct = None
    else:
        truth = (kind == "updated")
        attribution_correct = (bool(affected) == truth)

    if kind == "control":
        # nothing changed -> "fresh" just means the (unchanged) truth is served.
        fresh_present = _hit(v1, ans)
        served_stale = None
        fresh_clean = None
    else:
        fresh_present = has_v2
        if v1:                       # old value known -> strict metrics defined
            served_stale = has_v1    # the failure mode: still surfacing the old value
            fresh_clean = has_v2 and not has_v1   # actually resolved to the new value only
        else:                        # LME: no clean old value -> strict metrics n/a
            served_stale = None
            fresh_clean = None

    return StaleScore(
        item_id=item["id"], kind=kind, has_provenance=has_provenance,
        composed=a2.composed, affected=affected,
        attribution_correct=attribution_correct,
        fresh_present=fresh_present, served_stale=served_stale,
        fresh_clean=fresh_clean, flagged=flagged,
    )


def aggregate(scores: list[StaleScore]) -> dict:
    updated = [s for s in scores if s.kind == "updated"]
    control = [s for s in scores if s.kind == "control"]
    nu = len(updated) or 1
    nc = len(control) or 1

    out = {
        "n": len(scores),
        "n_updated": len(updated),
        "n_control": len(control),
        "refusal_rate": sum(0 if s.composed else 1 for s in scores) / (len(scores) or 1),
        # lenient: answer merely CONTAINS the new value (passes even when it also
        # serves the stale one — the metric that ties on a tiny store).
        "fresh_present_rate": sum(s.fresh_present for s in updated) / nu,
        "flagged_rate": sum(s.flagged for s in updated) / nu,
    }

    # strict serving — only over updated items whose v1 (old value) is known.
    # fresh_clean = resolved to the new value with the stale one ABSENT.
    # served_stale = the old value still surfaced post-update (the real failure).
    v1_known = [s for s in updated if s.fresh_clean is not None]
    if v1_known:
        out["fresh_clean_rate"] = sum(s.fresh_clean for s in v1_known) / len(v1_known)
        out["served_stale_rate"] = sum(s.served_stale for s in v1_known) / len(v1_known)
    else:
        out["fresh_clean_rate"] = None
        out["served_stale_rate"] = None
    out["strict_serve_n"] = len(v1_known)

    # attribution: only for provenance systems. n/a otherwise (NOT 0).
    prov = any(s.has_provenance and s.affected is not None for s in scores)
    if prov:
        tp = sum(1 for s in updated if s.affected)
        fp = sum(1 for s in control if s.affected)
        out["attribution_recall"] = tp / nu
        out["attribution_fpr"] = fp / nc
        out["attribution_precision"] = (tp / (tp + fp)) if (tp + fp) else None
    else:
        out["attribution_recall"] = None
        out["attribution_fpr"] = None
        out["attribution_precision"] = None
        out["attribution_note"] = "n/a — no provenance edges to attribute over"
    return out
