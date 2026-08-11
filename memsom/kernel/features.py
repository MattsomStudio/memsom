"""memsom.kernel.features -- types and the registry protocol only (PLAN.md Sec2.1).

Rank 0: stdlib only, no registry, no probes. The registry itself (import-heavy,
needs to reach every rank to ask "is retrieval.bge available", "is the broker
configured") lives in memsom.interface.features, the composition root for this
surface -- exactly like memsom.tuning, a Feature's status() implementation is
free to live wherever its subsystem does; this module only fixes the shape
every implementation agrees on.

Vocabulary is load-bearing (charter R3, "unknown != empty != allowed"):
  disabled  operator turned it off                    (never `active`)
  absent    a dependency is missing                    (never `active`)
  degraded  dependency present, backend down/reduced    (never `active`)
  error     status() itself raised -- always reported this way, NEVER `active`
  active    the feature is present, enabled, and working
"""

from __future__ import annotations

from typing import Protocol, TypedDict


class FeatureStatus(TypedDict):
    name: str
    state: str        # "active" | "degraded" | "absent" | "disabled" | "error"
    detail: str
    since: str | None  # ISO, first observation of THIS state (feature_status table)
    required: bool     # a required feature may not be absent; the CLI exits non-zero
    knobs: list[str]   # tuning.py REGISTRY keys this feature owns


class Feature(Protocol):
    name: str

    def migrate(self, conn) -> None: ...

    def status(self, conn) -> FeatureStatus: ...   # cheap: no model load, no network call

    def register_cli(self, subparsers) -> None: ...  # optional


VALID_STATES = ("active", "degraded", "absent", "disabled", "error")
