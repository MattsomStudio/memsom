"""interleave -- a reusable forced-interleaving harness (two connections, a
barrier, N trials), generalised from
`Desktop/memsom-rebuild-2026-07-31/01-pentest/poc/conc_redact_derive_toctou.py`
(PLAN.md Phase 0).

That PoC hand-rolled this pattern once, inline, to prove MS-06 (the redact/
derive TOCTOU race): monkeypatch a function on the module under test so it
pauses mid-body at the exact point the real race window opens, let a second
thread on a second connection run through that window, then resume and
inspect what landed. `tests/gates/test_gate_redact_derive_taint.py`'s own
`test_redact_cascade_is_atomic_against_a_concurrent_derive` already
re-implements the same shape inline -- this module is that shape, written
once, so the NEXT writer_census-flagged race (Phase 6's writer census, this
phase's own scripts/writer_census.py) does not re-derive it a third time.

USAGE
-----
    from interleave import ForcedWindow

    real = target_module.cascade_set
    window = ForcedWindow(target_module, "cascade_set", real)

    def racer():
        window.wait_for_open()
        ... do the concurrent write on a second connection ...
        window.signal_done()

    t = threading.Thread(target=racer, daemon=True)
    t.start()
    with window:                    # patches in, restores on exit even if
        victim_call()               # victim_call raises
    t.join(15)

`ForcedWindow.natural_rate(trials, victim_fn, racer_fn, detect_fn)` is the
uninstrumented-stress-loop half of the PoC (`natural_rate()`): report how
often the race bites with NO monkeypatch, because "a low natural rate does
not clear the defect -- it sets how often it bites, not whether it exists"
(the PoC's own words, worth keeping verbatim).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class ForcedWindow:
    """Pause a real function exactly where a race window opens, on request.

    `real_fn` is wrapped so that once it has done its work, it sets
    `opened` and blocks until the racer calls `signal_done()`. The wrapper
    is installed as `getattr(module, attr)` only inside the `with` block and
    is ALWAYS restored on exit, including on exception -- a harness that
    leaves a patched function behind on a failure poisons every test that
    runs after it in the same process.
    """

    def __init__(self, module, attr: str, real_fn, timeout: float = 10.0):
        self.module = module
        self.attr = attr
        self.real_fn = real_fn
        self.timeout = timeout
        self._opened = threading.Event()
        self._done = threading.Event()

    def _instrumented(self, *args, **kwargs):
        result = self.real_fn(*args, **kwargs)
        self._opened.set()
        self._done.wait(self.timeout)
        return result

    def __enter__(self):
        setattr(self.module, self.attr, self._instrumented)
        return self

    def __exit__(self, exc_type, exc, tb):
        setattr(self.module, self.attr, self.real_fn)
        return False

    def wait_for_open(self) -> bool:
        """Called from the racer thread: block until the victim has entered
        the window. Returns False on timeout (the victim never got there)."""
        return self._opened.wait(self.timeout)

    def signal_done(self) -> None:
        """Called from the racer thread once its concurrent write has
        landed: release the victim to resume past the window."""
        self._done.set()


@dataclass
class NaturalRateResult:
    trials: int
    hits: int
    errors: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.hits / self.trials if self.trials else 0.0


def natural_rate(trials, setup_fn, victim_fn, racer_fn, detect_fn) -> NaturalRateResult:
    """Run `victim_fn` and `racer_fn` concurrently, uninstrumented, `trials`
    times, starting them from a shared barrier so neither gets a head start.

    setup_fn()   -> context passed to victim_fn/racer_fn/detect_fn
    victim_fn(ctx, barrier)
    racer_fn(ctx, barrier)
    detect_fn(ctx) -> bool   (did this trial land in the bad state)
    """
    hits = 0
    errors: list[str] = []
    for _ in range(trials):
        ctx = setup_fn()
        barrier = threading.Barrier(2)
        result: dict = {}

        def run(fn, key):
            try:
                fn(ctx, barrier)
            except Exception as exc:  # noqa: BLE001 -- report, never swallow
                result[key] = repr(exc)

        threads = [threading.Thread(target=run, args=(victim_fn, "victim_error")),
                  threading.Thread(target=run, args=(racer_fn, "racer_error"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)

        for k in ("victim_error", "racer_error"):
            if k in result:
                errors.append(result[k])

        if detect_fn(ctx):
            hits += 1

    return NaturalRateResult(trials=trials, hits=hits, errors=errors)
