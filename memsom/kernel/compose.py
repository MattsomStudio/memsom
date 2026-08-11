"""memsom.kernel.compose -- deterministic answer composition.

Moved out of memsom/__init__.py (Phase 2, the core split). PROTECTED
(PLAN.md Sec1.7): compose's behaviour must stay byte-identical across every
phase -- proved by _meta/tools/differential.py.
"""

from memsom.kernel.text import candidate_sentences, stems


def compose(question, sources):
    """Pure + deterministic: same inputs -> byte-identical answer. No LLM, no clock."""
    keys = stems(question)
    bullets, used = [], []
    for sid, content, channel, _label, _ref in sources:  # label DESC, id ASC = trust order
        cands = candidate_sentences(content)
        if not cands:  # nothing survived the filter: first non-empty line, capped
            first = next((l.strip() for l in content.splitlines() if l.strip()), "")
            if first:
                cands = [first[:200].rstrip(".:") + "."]
        scored = [(sum(1 for k in keys if k in s.lower()), pos, s)
                  for pos, s in enumerate(cands)]
        top = sorted([t for t in scored if t[0] > 0], key=lambda t: (-t[0], t[1]))  # keep all keyword-matching sentences (was [:2])
        picked = [s for _, _, s in sorted(top, key=lambda t: t[1])]
        if not picked and cands:  # every live source contributes >=1 claim,
            picked = [cands[0]]   # so revoking any source is visible by construction
        if picked:
            bullets += [f"- {s} [mem:{sid}|{channel}]" for s in picked]
            used.append(sid)
    if not bullets:
        return None, []
    text = f"Q: {question}\nA (composed from {len(used)} live sources):\n" + "\n".join(bullets)
    return text, used
