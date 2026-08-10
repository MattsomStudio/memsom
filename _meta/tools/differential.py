#!/usr/bin/env python3
"""differential -- the perfect oracle, repointed at memsom's own `compose` (PLAN.md Phase 0).

REPOINTED, NOT REWRITTEN: this is the panel refactor's differential harness
(same shape -- record/check, sha256 digest, JSON oracle, exceptions ARE
recorded behaviour) with the corpus and import target swapped for memsom's
frozen-core text pipeline. That harness lived at
$REFACTOR/_meta/tools/differential.py against a DIFFERENT repo
(memsom-agentic-os/backend/memsom_panel) entirely -- this one lives inside
the memsom checkout itself, next to the code it proves identical, for the
same reason goals_ratchet.py was brought home in this same phase: a gate
that needs a sibling checkout to run is a gate that silently stops running
the day that checkout is missing.

Later phases' exit-gate commands that say `$REFACTOR/_meta/tools/
differential.py --check` should read this file's path instead
(`_meta/tools/differential.py`, run from the memsom repo root) --
a path correction, not a scope change; PLAN.md Phase 3/7 still mean
"run the compose oracle".

`compose` is documented as "Pure + deterministic: same inputs -> byte-
identical answer. No LLM, no clock." -- exactly the property this oracle
exists to hold the refactor to across the Phase 2 split into kernel/compose.py.

USAGE
  python _meta/tools/differential.py --record     # write the oracle (tag HEAD first)
  python _meta/tools/differential.py --check      # diff current behaviour against it
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ORACLE = Path(__file__).resolve().parent.parent / "measurements" / "differential-oracle.json"

sys.path.insert(0, str(REPO))


def _outcome(fn, *args, **kwargs):
    """Value or exception, both recorded. An exception IS behaviour.

    `set`/`frozenset` results are sorted before repr: their iteration order
    depends on Python's per-process string hash seed, not on memsom's
    logic, and an oracle that flags PYTHONHASHSEED as a behaviour change is
    noise, not a finding.
    """
    try:
        value = fn(*args, **kwargs)
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        return {"ok": True, "value": repr(value)}
    except Exception as exc:  # noqa: BLE001 -- recording, not handling
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Corpora. Adversarial values first.
# ---------------------------------------------------------------------------
TEXTS = [
    "", "   ", "\n\n\n", "a", "a" * 500,
    "Hello. This is a normal sentence. And another one!",
    "No terminal punctuation here",
    "Line one\nLine two\nLine three.",
    "- bullet one\n- bullet two\n* bullet three",
    "> a quoted blockquote line",
    "```code fence\nx = 1\n```",
    "# Heading\n\nBody text follows the heading.",
    "Question? Answer! Statement.",
    "Unicode: café naïve résumé. 中文句子。",
    "Emoji test \U0001F600 in the middle of a sentence.",
    "Tabs\tand\tmultiple   spaces.",
    "Trailing punctuation....",
    "A very very very very very very very very very very very very long "
    "single sentence that keeps going and going without stopping for a "
    "very long time indeed until eventually it does stop here.",
    "[mem:1|user] a sentence that already looks like a citation",
    "sql injection attempt' OR '1'='1",
    "<script>alert(1)</script> embedded markup",
]

QUESTIONS = ["", "what is the CA passphrase?", "How does Nebula hole punching work?",
             "unrelated query with no keyword overlap", "a" * 50]

CHANNELS = ["user", "endorsed", "external", "agent-derived"]


def _sources_variants():
    variants = []
    variants.append([])
    for i, (text, ch) in enumerate(zip(TEXTS, CHANNELS * 10)):
        variants.append([(i + 1, text, ch, 0, None)])
    variants.append([(1, TEXTS[5], "user", 0, None),
                      (2, TEXTS[6], "external", 0, "https://evil/x"),
                      (3, "", "user", 0, None)])
    return variants


def build() -> dict:
    import memsom

    rec: dict[str, dict] = {}

    for t in TEXTS:
        rec[f"stems({t!r})"] = _outcome(memsom.stems, t)
        rec[f"prose_lines({t!r})"] = _outcome(lambda x: list(memsom.prose_lines(x)), t)
        rec[f"strip_furniture({t!r})"] = _outcome(memsom.strip_furniture, t)
        rec[f"snippet({t!r},80)"] = _outcome(memsom.snippet, t, 80)
        rec[f"candidate_sentences({t!r})"] = _outcome(
            lambda x: list(memsom.candidate_sentences(x)), t)

    for q in QUESTIONS:
        for sources in _sources_variants():
            key = f"compose({q!r},{sources!r})"
            rec[key] = _outcome(memsom.compose, q, sources)

    return rec


def _head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    try:
        rec = build()
    except Exception:
        traceback.print_exc()
        raise SystemExit("::error::the corpus could not be built -- refusing to "
                         "record or check a partial oracle")

    digest = hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode("utf-8")).hexdigest()

    if args.record:
        ORACLE.parent.mkdir(parents=True, exist_ok=True)
        ORACLE.write_text(json.dumps(
            {"oracle_commit": _head(), "cases": len(rec), "digest": digest,
             "outcomes": rec}, indent=2, sort_keys=True), encoding="utf-8")
        print(f"recorded {len(rec)} cases at {_head()[:8]}  digest {digest[:16]}")
        print(f"wrote {ORACLE}")
        return 0

    if not args.check:
        ap.print_help()
        return 2

    if not ORACLE.exists():
        raise SystemExit(f"::error::no oracle at {ORACLE}. Run --record first.")
    old = json.loads(ORACLE.read_text(encoding="utf-8"))
    base = old["outcomes"]

    added = sorted(set(rec) - set(base))
    removed = sorted(set(base) - set(rec))
    changed = sorted(k for k in set(rec) & set(base) if rec[k] != base[k])

    print(f"oracle    : {old['oracle_commit'][:8]}  ({old['cases']} cases)")
    print(f"now       : {_head()[:8]}  ({len(rec)} cases)")
    print(f"changed   : {len(changed)}")
    print(f"new case  : {len(added)}")
    print(f"lost case : {len(removed)}")
    for k in changed[:40]:
        print(f"  DIFF {k}\n       was {base[k]}\n       now {rec[k]}")
    for k in removed[:20]:
        print(f"  GONE {k}")

    if changed or removed:
        print("::error::behaviour differs from the oracle. In a refactor that is "
              "a bug by definition -- every case here is supposed to be identical.")
        return 1
    print("\n[differential] identical to the oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
