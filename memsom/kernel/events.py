"""memsom.kernel.events -- synchronous pub/sub with no swallow (PLAN.md Sec1.5, Sec2.3).

The try-wrapped upward imports this replaces (integrity reaching into retrieval
to reindex/deindex a node) always caught every exception and moved on, which is
exactly memsom's dominant defect class: a failure that reaches the user as "it
worked" instead of "it degraded" or "it failed" (Sec2.3). This primitive makes
that impossible structurally rather than by convention: `emit` NEVER swallows a
subscriber's exception -- it collects every one and hands the list back to the
caller, which is the one place that knows whether "the index update failed" is
optional-degrade or a fact that must be reported.

It also kills the layering violation: a rank-2 module (integrity) used to
`from memsom.retrieval import ...` directly (rank 3, upward). Now integrity
only imports kernel.events (rank 0, downward) and emits a named event; the
rank-3 module subscribes to it. Nothing above kernel ever appears in a rank-2
module's import list because of this pattern.
"""

_SUBSCRIBERS = {}


def subscribe(event, fn):
    """Register *fn* to run on emit(event, **payload). Returns fn (decorator-usable)."""
    _SUBSCRIBERS.setdefault(event, []).append(fn)
    return fn


def unsubscribe(event, fn):
    """Remove *fn* from *event*'s subscriber list. No-op if not registered."""
    subs = _SUBSCRIBERS.get(event)
    if subs and fn in subs:
        subs.remove(fn)


def clear(event=None):
    """Remove all subscribers for *event*, or every event if None. Test-only escape hatch."""
    if event is None:
        _SUBSCRIBERS.clear()
    else:
        _SUBSCRIBERS.pop(event, None)


def emit(event, **payload):
    """Call every subscriber to *event* synchronously, in registration order.

    A subscriber that raises does NOT stop the others and is NOT swallowed:
    every exception is collected and returned as a list of
    (subscriber, exception) pairs. An empty list means every subscriber
    succeeded (or there were none). The caller decides what "failure" means
    for its own event -- this primitive never decides that for it.
    """
    failures = []
    for fn in list(_SUBSCRIBERS.get(event, ())):
        try:
            fn(**payload)
        except Exception as exc:  # noqa: BLE001 -- collected, never swallowed
            failures.append((fn, exc))
    return failures
