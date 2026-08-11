"""GATE for MS-40 -- ARCHITECTURE.md must not claim check_action is the only
enforcement point (SECURITY-REMEDIATION.md MS-40, PLAN.md Phase 9).

check_action has zero internal callers (CLI `check-action` / MCP tool only);
real runtime enforcement of Gate #3 is memsom_capgate.check_capability via the
broker and native-tool hooks. PLAN.md Phase 9 requires deciding this in one of
two ways: interpose check_action on memsom's own mutating actions, or correct
the doc. Phase 9 took the doc-correction path -- this test pins it so the doc
cannot silently regress back to the false claim.
"""

from pathlib import Path

ARCHITECTURE_MD = Path(__file__).resolve().parents[2] / "ARCHITECTURE.md"


def test_check_action_not_claimed_as_only_enforcement_point():
    text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    assert "the only enforcement point" not in text.lower()


def test_check_action_documented_as_advisory():
    text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    assert "advisory node-integrity" in text
    assert "check_capability" in text
