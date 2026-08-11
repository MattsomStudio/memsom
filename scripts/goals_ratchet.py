#!/usr/bin/env python3
r"""goals_ratchet -- the goals file's violations may never grow while the
refactor is in flight.

Brought home from `memsom-agentic-os/scripts/goals_ratchet.py` (Phase 0, A9).
That script ratchets memsom's `.importlinter-goals` from OUTSIDE this repo,
which means memsom's own CI cannot enforce it without a sibling checkout of
another repo. This is the same ratchet, scoped to memsom alone: no
`MEMSOM_REPO`, no panel baseline, cwd is always this repo's root.

THE ARGUMENT
------------
`.importlinter-goals` is deliberately unwired from `.importlinter` -- it is RED
by design (see its own header) and blocking CI on it would make CI permanently
red. But nothing then stops the violation count growing while the refactor is
in flight; the goals file's own instruction ("watch the violation count
shrink") is a manual habit, and this engagement does not trust those.

THE TWO WAYS THIS RATCHET GOES VACUOUS, both defended against here:

1. Invoking `python -m importlinter.cli lint-imports` instead of the console
   script. MEASURED on this machine: the module form prints NOTHING and exits
   0 -- a permanently, silently green gate. This script calls the console
   script by name and hard-fails if it is not on PATH.
2. Defaulting to "no increase" when the output is unparseable or the config is
   missing. A ratchet that cannot find its number must not pass.

SCOPE is the set of violating EDGES, per contract -- not a whole-file count.
An aggregate moves when you add a new aspiration to the goals file, which
teaches you to re-baseline on sight; a per-contract edge set does not move
just because a sibling contract gained one, and a new violation names the
edge in the failure message instead of just moving a number.

BASELINE taken fresh from the tool at Phase 0, 2026-08-10, at this repo's
`7862fa8` (before this phase's own commit): `lint-imports --config
.importlinter-goals` -- 26 edges under the layering contract, 4 under
acyclic-siblings. Re-derive with `python scripts/goals_ratchet.py --json`.

RE-BASELINED at Phase 2, 2026-08-10: `retrieval/recompute.py` moved to
`integrity/recompute.py` (its own docstring already called it "multi-hop
INTEGRITY recompute" -- it was retrieval-layer only by location, not by
domain). Removes corroborate/trust -> retrieval.recompute (2 edges, both
already in the Phase-0 baseline, now intra-integrity) and absorbs the one
edge Phase 1's MS-03 fix added (stale -> retrieval.recompute) without
reopening that CRITICAL -- it never reaches this BASELINE at all, because
by the time this commit lands stale.py imports integrity.recompute, a
same-layer call. 26 -> 24.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ".importlinter-goals"

BASELINE = {
    "memsom internal layering (REFACTOR TARGET)": [
        "memsom.bridge.bridge_import -> memsom.interface.ingest",
        "memsom.bridge.chats -> memsom.interface.ingest",
        "memsom.bridge.obsidian -> memsom.interface.cli",
        "memsom.bridge.obsidian -> memsom.interface.ingest",
        "memsom.distill.digest -> memsom.bridge.bridge_import",
        "memsom.distill.digest -> memsom.bridge.facts",
        "memsom.federation.broker -> memsom.interface.ingest",
        "memsom.federation.broker -> memsom.interface.mcp",
        "memsom.integrity.contradict -> memsom.bridge.bridge_import",
        "memsom.integrity.contradict -> memsom.retrieval.embed",
        "memsom.integrity.contradict -> memsom.retrieval.retrieve",
        "memsom.integrity.corroborate -> memsom.retrieval.rederive",
        "memsom.integrity.gate -> memsom.interface.blame",
        "memsom.integrity.redact -> memsom.bridge.bridge_import",
        "memsom.integrity.redact -> memsom.retrieval.retrieve",
        "memsom.integrity.stale -> memsom.retrieval.rederive",
        "memsom.integrity.tombstone -> memsom.bridge.bridge_import",
        "memsom.integrity.tombstone -> memsom.retrieval.rederive",
        "memsom.integrity.verify_stale -> memsom.bridge.bridge_import",
        "memsom.lifecycle.compact -> memsom.distill.llm",
        "memsom.lifecycle.reflex -> memsom.distill.distill",
        "memsom.retrieval.rederive -> memsom.lifecycle.compact",
        "memsom.retrieval.retrieve -> memsom.bridge.facts",
        "memsom.retrieval.retrieve -> memsom.distill.llm",
    ],
    "acyclic_siblings - same-level packages never import each other": [
        "memsom.distill.digest -> memsom.bridge.bridge_import",
        "memsom.federation.broker -> memsom.interface.ingest",
        "memsom.federation.broker -> memsom.interface.mcp",
        "memsom.federation.federation -> memsom.integrity.redact",
    ],
}

_SUMMARY_RE = re.compile(r"Contracts: (\d+) kept, (\d+) broken\.")
#: `\s+` rather than a single space -- import-linter indents `forbidden`
#: contracts with three spaces and `layers`/`independence` ones with one.
_CHAIN_RE = re.compile(r"^-\s+(\S+) -> (\S+)", re.M)
_PAIR_RE = re.compile(r"^(\S+) is not allowed to import", re.M)
_HEADING_UNDERLINE_RE = re.compile(r"^-{4,}\s*$")
_ROSTER_RE = re.compile(r"^(.+?) (KEPT|BROKEN)\s*$", re.M)


def _annotate(level: str, msg: str) -> None:
    print(f"::{level}::{msg}")


def _sections(out: str) -> dict:
    """`{contract name: [raw line, ...]}` for every BROKEN contract.

    Section names come from the ROSTER import-linter prints at the top
    (`<name> KEPT` / `<name> BROKEN`); a heading only opens a section if it is
    a name from that roster, not merely "a line with dashes under it" (the
    summary line `Analyzed N files, M dependencies.` also has one).
    """
    roster = {name.strip(): verdict for name, verdict in _ROSTER_RE.findall(out)}
    broken = {n for n, v in roster.items() if v == "BROKEN"}
    lines = out.splitlines()
    sections: dict = {}
    current = None
    for i, line in enumerate(lines):
        name = line.strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if name in broken and _HEADING_UNDERLINE_RE.match(nxt):
            current = name
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    missing = broken - set(sections)
    if missing:
        raise SystemExit(
            f"::error::contracts reported BROKEN with no detail section: "
            f"{sorted(missing)}. The report shape changed; refusing to report "
            "zero edges for a contract import-linter says is broken.")
    return sections


def measure() -> dict:
    if shutil.which("lint-imports") is None:
        raise SystemExit(
            "::error::`lint-imports` console script not found. Do NOT fall "
            "back to `python -m importlinter.cli lint-imports` -- MEASURED: it "
            "prints nothing and exits 0, a permanently, silently green gate."
        )
    cfg = ROOT / CONFIG
    if not cfg.is_file():
        raise SystemExit(f"::error::{cfg} is missing. A ratchet that cannot "
                         "find its config must fail, not pass.")
    # encoding/errors explicit: import-linter's banner uses bytes the default
    # console codec on this box (cp1252) cannot decode; without this the
    # subprocess reader dies and stdout comes back None, read as "no violations".
    proc = subprocess.run(["lint-imports", "--config", CONFIG],
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)  # rich's ANSI progress spinner
    if not _SUMMARY_RE.search(out):
        raise SystemExit(
            "::error::could not parse lint-imports output (no 'Contracts: N "
            f"kept, M broken.' line). Refusing to assume no increase.\n{out[-2000:]}")
    kept, broken = (int(x) for x in _SUMMARY_RE.search(out).groups())

    edges: dict = {}
    for name, body in _sections(out).items():
        text = "\n".join(body)
        seen = []
        for importer, imported in _CHAIN_RE.findall(text):
            edge = f"{importer} -> {imported}"
            if edge not in seen:
                seen.append(edge)
        edges[name] = sorted(seen)

    pairs = len(_PAIR_RE.findall(out))
    if pairs and not any(edges.values()):
        raise SystemExit(
            f"::error::parsed {pairs} 'is not allowed to import' headings but "
            "ZERO edges. import-linter's report shape changed; this tool would "
            "otherwise report the refactor as finished.")
    return {"contracts_kept": kept, "contracts_broken": broken,
            "pair_headings": pairs, "edges": edges}


def _compare(got: dict) -> bool:
    """True on failure. Prints one line per contract and names every delta."""
    failed = False
    seen = got["edges"]
    for contract in sorted(set(BASELINE) | set(seen)):
        want = set(BASELINE.get(contract, []))
        have = set(seen.get(contract, []))
        if contract not in BASELINE:
            print(f"  [NEW ] {contract}  {len(have)} edge(s)")
            _annotate("error",
                      f"contract {contract!r} is not in BASELINE. Adding a "
                      "contract to the goals file re-baselines here IN THE "
                      "SAME COMMIT, naming the contract and its edge count.")
            failed = True
            continue
        if contract not in seen:
            print(f"  [GONE] {contract}")
            _annotate("error",
                      f"contract {contract!r} is in BASELINE but lint-imports "
                      "never reported it. Either it went GREEN (delete the row "
                      "here and say so) or it was deleted from the goals file.")
            failed = True
            continue
        added = sorted(have - want)
        removed = sorted(want - have)
        mark = "ok " if not added and not removed else ("UP " if added else "DOWN")
        print(f"  [{mark}] {contract}  {len(have)} edge(s) (baseline {len(want)})")
        for e in sorted(have):
            flag = "  <- NEW" if e in set(added) else ""
            print(f"           {e}{flag}")
        if added:
            _annotate("error",
                      f"{contract!r} grew {len(added)} edge(s): "
                      f"{', '.join(added)}. Remove the import -- do NOT add it "
                      "to BASELINE.")
            failed = True
        if removed:
            _annotate("error",
                      f"{contract!r} lost {len(removed)} edge(s): "
                      f"{', '.join(removed)}. Record it in BASELINE in the "
                      "same commit, or the floor drifts above the count and "
                      "silently accepts the violation coming back.")
            failed = True
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    got = measure()
    if args.json:
        print(json.dumps(got, indent=2))
        return 0
    failed = _compare(got)
    print("\n[goals-ratchet] " + ("FAILED" if failed else "at baseline"))
    print("[goals-ratchet] SCOPE: violating EDGES per contract, endpoints "
          "identified by their full dotted module name (memsom has no layer "
          "packages to strip). A module renamed or moved between subpackages "
          "reads as one edge leaving and one arriving.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
