"""What a run is allowed to touch — the target rules, unit level.

The end-to-end half (a refusal is audited, the run survives it, the declaration
rides on the head line, an approval EDIT is re-checked) lives in
test_provider_agents.py, because those are properties of the runner around the
check rather than of the check itself.
"""

from __future__ import annotations

import pytest

from memsom.providers import scope
from memsom.providers.tools.registry import BUILTIN_TOOLS


def _fetch(url: str) -> dict:
    return {"url": url}


# ---------------------------------------------------------------------------
# default-open
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://example.com/a",
    "http://10.4.4.4:8080/x",
    "https://192.168.2.1/",          # the home LAN: read-only GETs are ALLOWED
])
def test_an_undeclared_scope_lets_everything_through(url):
    """Default-open is the contract, not an accident of implementation. Every
    graph saved before scope existed must behave exactly as it did, or the
    feature is a breaking change wearing a safety feature's clothes.

    The home-LAN case is deliberate. The standing rule permits read-only GETs
    there and `http_fetch` IS a read-only GET — a blanket deny would break
    permitted use, and a guardrail that blocks the normal case is one that gets
    switched off wholesale within a day.
    """
    assert scope.check({}, "http_fetch", _fetch(url)) == ""
    assert scope.check(None, "http_fetch", _fetch(url)) == ""


def test_a_tool_with_no_target_is_never_refused():
    assert scope.check({"hosts": ["nothing.example"]}, "state_set",
                       {"key": "k", "value": "v"}) == ""


# ---------------------------------------------------------------------------
# a declared scope
# ---------------------------------------------------------------------------


def test_a_declared_host_scope_refuses_a_host_outside_it():
    declared = {"hosts": ["*.example.com"]}
    refusal = scope.check(declared, "http_fetch", _fetch("https://evil.test/x"))
    assert refusal
    assert "evil.test" in refusal and "declared scope" in refusal


def test_a_declared_scope_allows_what_it_names():
    declared = {"hosts": ["*.example.com", "api.thing.io"]}
    assert scope.check(declared, "http_fetch",
                       _fetch("https://a.example.com/x")) == ""
    assert scope.check(declared, "http_fetch",
                       _fetch("https://api.thing.io/v1")) == ""


def test_a_cidr_entry_matches_by_network_not_by_string():
    """The bug this pins is a string-prefix match: "10.0.0.0/24" must admit
    10.0.0.5 and REFUSE 10.0.0.50 even though one spelling starts the other."""
    declared = {"hosts": ["10.0.0.0/24"]}
    assert scope.check(declared, "http_fetch", _fetch("http://10.0.0.5/")) == ""
    assert scope.check(declared, "http_fetch", _fetch("http://10.0.0.50/")) == ""
    assert scope.check(declared, "http_fetch", _fetch("http://10.0.1.5/"))


def test_a_port_does_not_smuggle_a_host_past_the_check():
    declared = {"hosts": ["good.example"]}
    assert scope.check(declared, "http_fetch",
                       _fetch("http://good.example:8443/x")) == ""
    assert scope.check(declared, "http_fetch",
                       _fetch("http://bad.example:80/x"))


# ---------------------------------------------------------------------------
# the seatbelt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,label", [
    ("http://127.0.0.1:11434/api/chat", "ollama"),
    ("http://127.0.0.1:8080/completion", "llama.cpp"),
    ("http://127.0.0.1:7788/api/agents/run", "the panel's OWN api"),
    ("http://[::1]:7788/", "loopback v6"),
])
def test_the_seatbelt_denies_loopback_with_no_scope_declared(url, label):
    """Default-open everywhere EXCEPT the control plane.

    The panel's own API is the sharp one: an agent that can POST there can start
    runs, resume paused ones and approve its own gates. There is no legitimate
    agent reason to reach it, so this is the one place the default flips.
    """
    refusal = scope.check({}, "http_fetch", _fetch(url))
    assert refusal, f"{label}: reachable with no scope declared"
    assert "127.0.0.0/8" in refusal or "::1" in refusal


def test_link_local_metadata_is_denied():
    refusal = scope.check({}, "http_fetch",
                          _fetch("http://169.254.169.254/latest/meta-data/"))
    assert refusal and "169.254.0.0/16" in refusal


def test_an_explicit_declaration_can_unbuckle_the_seatbelt():
    """A SEATBELT, not a wall — and the difference is the whole design.

    A wall gets worked around and then resented; the capability already exists
    from a terminal, so a rule that cannot be switched off deliberately is the
    theatre the lab guardrail rule warns about. Naming the target is how you take
    responsibility for it.
    """
    assert scope.check({"hosts": ["127.0.0.1"]}, "http_fetch",
                       _fetch("http://127.0.0.1:11434/api/chat")) == ""
    assert scope.check({"hosts": ["127.0.0.0/8"]}, "http_fetch",
                       _fetch("http://127.0.0.1:7788/")) == ""
    # and a declaration that names something ELSE does not unbuckle it
    assert scope.check({"hosts": ["example.com"]}, "http_fetch",
                       _fetch("http://127.0.0.1:7788/"))


def test_a_hostname_that_resolves_into_the_seatbelt_is_still_refused(monkeypatch):
    """Without this, the seatbelt is decorative: point evil.test at 127.0.0.1
    and a string-only host check waves it straight through.

    Honest limit, and it is in the docstring too: urllib re-resolves when it
    connects, so DNS rebinding with a short TTL still wins. This raises the cost
    of the naive attack rather than closing the hole.
    """
    monkeypatch.setattr(scope, "_resolved_ips",
                        lambda host: [scope.ipaddress.ip_address("127.0.0.1")])
    refusal = scope.check({}, "http_fetch", _fetch("http://rebind.test/x"))
    assert refusal and "127.0.0.0/8" in refusal


def test_a_name_that_resolves_somewhere_harmless_is_untouched(monkeypatch):
    monkeypatch.setattr(scope, "_resolved_ips",
                        lambda host: [scope.ipaddress.ip_address("93.184.216.34")])
    assert scope.check({}, "http_fetch", _fetch("http://example.test/x")) == ""


def test_a_dead_resolver_does_not_refuse_the_call(monkeypatch):
    """A DNS failure must not become a scope refusal. The fetch will fail on its
    own a moment later with a message that says what actually went wrong."""
    monkeypatch.setattr(scope, "_resolved_ips", lambda host: [])
    assert scope.check({}, "http_fetch", _fetch("http://nowhere.test/x")) == ""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def test_a_path_scope_refuses_a_read_outside_it(tmp_path):
    inside = tmp_path / "work"
    inside.mkdir()
    declared = {"paths": [str(inside)]}
    assert scope.check(declared, "file_read",
                       {"path": str(inside / "notes.md")}) == ""
    refusal = scope.check(declared, "file_read",
                          {"path": str(tmp_path / "secrets.env")})
    assert refusal and "declared scope" in refusal


def test_a_path_scope_is_not_fooled_by_dot_dot(tmp_path):
    inside = tmp_path / "work"
    inside.mkdir()
    declared = {"paths": [str(inside)]}
    assert scope.check(declared, "file_read",
                       {"path": str(inside / ".." / "secrets.env")})


# ---------------------------------------------------------------------------
# no silent gaps
# ---------------------------------------------------------------------------


def test_every_builtin_tool_type_is_in_the_scope_table():
    """A tool added later must not quietly acquire an unchecked target.

    This is the test that makes the table trustworthy: without it, `_TARGETS`
    is a snapshot of what somebody remembered in July, and the failure mode is
    silent — a new tool with a `url` argument would simply never be checked and
    nothing anywhere would say so.
    """
    missing = sorted(set(BUILTIN_TOOLS) - set(scope._TARGETS))
    assert not missing, (
        f"builtin tools with no scope decision: {missing}. Add each to "
        f"scope._TARGETS — (arg, kind), None for no target, or UNSCOPED.")
    stale = sorted(set(scope._TARGETS) - set(BUILTIN_TOOLS))
    assert not stale, f"scope._TARGETS names tools that no longer exist: {stale}"


def test_shell_is_declared_unscoped_rather_than_forgotten():
    """`shell` reaches anything a target rule forbids by spelling it `curl`.

    Pretending otherwise would be the theatre this whole module is trying not to
    be, so it is written down, it is surfaced in the run log at start, and its
    real bound is the approval gate it already defaults to.
    """
    assert scope._TARGETS["shell"] == scope.UNSCOPED
    assert scope.check({"hosts": ["nothing.example"]}, "shell",
                       {"command": "curl https://evil.test"}) == ""
    assert scope.unscoped_tools(["shell", "http_fetch", "file_read"]) == ["shell"]
