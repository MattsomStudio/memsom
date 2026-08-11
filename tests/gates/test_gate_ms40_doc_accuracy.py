"""GATE for MS-40 -- neither ARCHITECTURE.md nor any live code surface (module
docstring, function docstring, CLI help text, MCP tool description) may claim
check_action is the only enforcement point (SECURITY-REMEDIATION.md MS-40,
PLAN.md Phase 9).

check_action has zero internal callers (CLI `check-action` / MCP tool only);
real runtime enforcement of Gate #3 is memsom_capgate.check_capability via the
broker and native-tool hooks. PLAN.md Phase 9 requires deciding this in one of
two ways: interpose check_action on memsom's own mutating actions, or correct
the doc. Phase 9 took the doc-correction path -- this test pins it (across the
doc AND the code surfaces an operator or agent actually reads) so none of them
can silently regress back to the false claim.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_MD = ROOT / "ARCHITECTURE.md"

_FALSE_CLAIM_FRAGMENTS = ("the only enforcement", "the only gate", "the only place the floor")


def _assert_no_false_claim(text, where):
    lowered = text.lower()
    for frag in _FALSE_CLAIM_FRAGMENTS:
        assert frag not in lowered, f"{where} still claims check_action is {frag!r}"


def test_check_action_not_claimed_as_only_enforcement_point():
    text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    _assert_no_false_claim(text, "ARCHITECTURE.md")


def test_check_action_documented_as_advisory():
    text = ARCHITECTURE_MD.read_text(encoding="utf-8")
    assert "advisory node-integrity" in text
    assert "check_capability" in text


def test_gate_module_does_not_claim_only_enforcement():
    text = (ROOT / "memsom" / "integrity" / "gate.py").read_text(encoding="utf-8")
    _assert_no_false_claim(text, "memsom/integrity/gate.py")


def test_mcp_tool_description_does_not_claim_only_gate():
    text = (ROOT / "memsom" / "interface" / "mcp.py").read_text(encoding="utf-8")
    _assert_no_false_claim(text, "memsom/interface/mcp.py")


def test_cli_help_does_not_claim_only_gate():
    out = subprocess.run(
        [sys.executable, "-m", "memsom.interface.cli", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    _assert_no_false_claim(out.stdout + out.stderr, "memsom --help")


def test_mcp_selfcheck_tool_listing_does_not_claim_only_gate():
    out = subprocess.run(
        [sys.executable, "-m", "memsom.interface.mcp", "--selfcheck"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    _assert_no_false_claim(out.stdout + out.stderr, "mcp --selfcheck tool listing")
