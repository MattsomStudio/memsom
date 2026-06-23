"""poison — decide which items get poisoned, at a controlled rate.

Deterministic selection (no RNG): items are sorted by id and poisoned by a
stride derived from the rate, so a given (dataset, rate) always poisons the same
items. Reproducibility is the whole point of a benchmark — a run you can't
re-run isn't evidence.

The actual injection (ingest-text on the poison channel) lives in run_bench so
the DB lifecycle stays in one place; this module only answers "is item i
poisoned at rate r?".
"""

from __future__ import annotations


def select_poisoned(items: list[dict], rate: float) -> set[str]:
    """Return the set of item ids to poison at the given rate (0.0–1.0)."""
    if rate <= 0:
        return set()
    if rate >= 1:
        return {it["id"] for it in items if it.get("poison")}

    poisonable = [it for it in sorted(items, key=lambda x: x["id"]) if it.get("poison")]
    if not poisonable:
        return set()
    # evenly spaced pick of ceil(rate*N) items
    k = max(1, round(rate * len(poisonable)))
    stride = len(poisonable) / k
    chosen = {poisonable[min(len(poisonable) - 1, int(i * stride))]["id"] for i in range(k)}
    return chosen
