"""runner — thin, deterministic wrapper around the memsom CLI for benchmarking.

The whole harness talks to memsom ONLY through its CLI (memsom.interface.cli). That is
deliberate: the benchmark must measure the shipping product's behaviour, not poke
at internals that could drift. If a metric can't be derived from CLI output, it
isn't a property the product actually exposes.

Isolation contract (verified against memsom 2026-06-16):
  - `init --data-dir DIR`  builds a fresh DB at DIR/memdag.db. It IGNORES MEMDAG_DB.
  - every other command (add, ingest-text, reindex, ask, retrieve, dump) resolves
    its DB from `MEMDAG_DB` (falling back to repo/memdag.db). So we init once into
    a per-item dir, then pin MEMDAG_DB at that dir's memdag.db for all ops.
  - we pass MEMDAG_DB via the subprocess env dict, never via shell `set`, so there
    is zero dependency on the host shell's quoting (Windows cmd eats trailing
    spaces on `set VAR=val & ...`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# ---- parsed shapes -------------------------------------------------------

@dataclass
class Citation:
    node_id: int
    channel: str
    text: str


@dataclass
class AskResult:
    composed: bool                       # did memsom compose, or refuse?
    answer_text: str                     # concatenation of all answer bullet texts
    citations: list[Citation] = field(default_factory=list)
    used: int = 0                        # sources used count from the trailer
    considered: int = 0
    composed_node_id: int | None = None
    integrity: str | None = None         # integrity label of the composed node
    raw: str = ""                        # full stdout+stderr, kept for debugging


# integrity lattice order (low -> high). Biba low-water-mark: a composed node's
# label is the MIN of its parents, so a poison node on `external` should drag any
# answer that uses it down to EXTERNAL. "Laundering" = that demotion failing.
INTEGRITY_ORDER = {"external": 0, "agent-derived": 1, "user": 2, "endorsed": 3}
# the CLI prints labels upper-cased (USER, EXTERNAL, ...); normalise on read.


# The `ask --retrieve` (deterministic, no --llm) output contract:
#   Q: <question>
#   A (composed from K live sources):
#   - <text> [mem:ID|channel]
#   ...
#   stored as node [N] | integrity: LABEL (floor of P parents) | sources
#       considered: X, used: Y, excluded: ...
# Refusal: "[memsom] no live sources - refusing to compose ..."
_BULLET = re.compile(r"^-\s+(?P<text>.*?)\s*\[mem:(?P<id>\d+)\|(?P<chan>[a-z-]+)\]\s*$")
_TRAILER = re.compile(
    r"stored as node \[(?P<node>\d+)\].*?integrity:\s*(?P<integ>[A-Z-]+).*?"
    r"considered:\s*(?P<considered>\d+),\s*used:\s*(?P<used>\d+)",
    re.S,
)


def parse_ask(out: str) -> "AskResult":
    """Parse `ask --retrieve` stdout+stderr into an AskResult.

    Shared by the subprocess runner and the in-process driver so both measure
    exactly the same thing from exactly the same text.
    """
    res = AskResult(composed=False, answer_text="", raw=out)
    if "refusing to compose" in out or "no live sources" in out:
        return res

    cites: list[Citation] = []
    texts: list[str] = []
    for line in out.splitlines():
        m = _BULLET.match(line.strip())
        if m:
            texts.append(m.group("text"))
            cites.append(Citation(int(m.group("id")), m.group("chan"), m.group("text")))

    t = _TRAILER.search(out)
    if t:
        res.composed_node_id = int(t.group("node"))
        res.integrity = t.group("integ").lower()
        res.considered = int(t.group("considered"))
        res.used = int(t.group("used"))

    res.composed = bool(cites) or t is not None
    res.citations = cites
    res.answer_text = " ".join(texts)
    return res


class MemsomRunner:
    def __init__(self, repo: str, python: str = "python", data_dir: str | None = None):
        self.repo = Path(repo)
        self.python = python
        # The flat memsom_cli.py died in the package restructure; the full-stack
        # CLI is the memsom.interface.cli module (console script `memsom`).
        self.cli = ["-m", "memsom.interface.cli"]
        self.data_dir: Path | None = Path(data_dir) if data_dir else None

    # -- lifecycle --------------------------------------------------------

    def init(self, data_dir: str | Path) -> None:
        """Create a fresh, isolated DB. Wipes any prior dir at that path."""
        self.data_dir = Path(data_dir)
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir, ignore_errors=True)
        # init resolves its own path from --data-dir and ignores MEMDAG_DB.
        self._run(["init", "--data-dir", str(self.data_dir)], pin_db=False)

    @property
    def db_path(self) -> Path:
        assert self.data_dir is not None, "call init() first"
        return self.data_dir / "memdag.db"

    # -- writes -----------------------------------------------------------

    def add(self, text: str, channel: str, ref: str | None = None) -> None:
        args = ["add", text, "--channel", channel]
        if ref:
            args += ["--ref", ref]
        self._run(args)

    def ingest_text(self, text: str, channel: str, ref: str | None = None) -> None:
        args = ["ingest-text", text, "--channel", channel]
        if ref:
            args += ["--ref", ref]
        self._run(args)

    def reindex(self) -> None:
        self._run(["reindex"])

    # -- read / compose ---------------------------------------------------

    def ask(self, question: str, topk: int = 8, clearance: str | None = None,
            llm: bool = False) -> AskResult:
        args = ["ask", question, "--retrieve", "--topk", str(topk)]
        if clearance:
            args += ["--clearance", clearance]
        if llm:
            args += ["--llm"]
        out = self._run(args)
        return self._parse_ask(out)

    # -- internals --------------------------------------------------------

    def _run(self, args: list[str], pin_db: bool = True) -> str:
        env = dict(os.environ)
        if pin_db:
            env["MEMDAG_DB"] = str(self.db_path)
        proc = subprocess.run(
            [self.python, *self.cli, *args],
            cwd=str(self.repo), env=env,
            capture_output=True, text=True,
        )
        # memsom prints answers to stdout and status to stderr; we want both.
        return (proc.stdout or "") + (proc.stderr or "")

    def _parse_ask(self, out: str) -> AskResult:
        return parse_ask(out)
