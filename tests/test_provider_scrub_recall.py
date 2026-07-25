"""How much does `scrub` actually catch? A labelled corpus, and a number.

`_SECRET_PATTERNS` was written for PRECISION and never measured for RECALL. The
docstring is honest about the trade — "a false positive silently mangles a
legitimate answer, and an agent whose output is quietly corrupted is worse than
one that leaked a string nobody had budgeted for" — but a redactor whose miss
rate nobody has ever computed is a redactor nobody can size. "It catches the
common shapes" is a feeling, not a property.

So this file is the instrument before it is a gate. Every entry is one labelled
example; the tests turn them into recall and precision and assert floors, so the
numbers can only ever go up by accident and never down.

**The negatives matter more than the positives.** Recall is easy to buy and the
bill arrives silently, in mangled output nobody traces back here. The hard
negatives below (a git SHA, a UUID, a base64 payload, a hex digest) are what make
a recall number mean anything, and they are why this file asserts precision at a
hard 1.00 rather than "high".

Measured 2026-07-25: **recall 35% → 100%, precision 100% before and after.** The
first number is the one worth keeping: the pattern list was believed to cover the
common cases and caught nine of twenty-six.

Two things the corpus found that no amount of reading had:

* `db_password=…` was NOT redacted. The pattern opened with `\\b`, and a word
  boundary does not fire between `_` and `password` — so every key name with a
  prefix, which is what they are actually called in a shell, walked straight
  through a rule everyone believed covered them.
* Scrubbing was not idempotent: a second pass over already-scrubbed text reported
  fresh hits for `[REDACTED]` placeholders, inflating a number somebody reads.

A general high-entropy rule was written and measured against this corpus rather
than dismissed on principle — at threshold 3.5 it cost 8 false positives and
added **0** catches the shapes did not already have. See `_SECRET_PATTERNS`.

Values are synthetic. Nothing here has ever been a live credential.
"""

from __future__ import annotations

import pytest

from memsom.providers.lc_runtime import _REDACTED, _scrub_text

#: (label, text, the substring that MUST NOT survive)
#:
#: Grouped by whether the shape was covered when the corpus was first written,
#: so the miss list is legible rather than implied.
POSITIVES = [
    # ---- covered before this corpus existed -------------------------------
    ("openai-style key",
     "use sk-" "abcdefghijklmnopqrstuvwx0123456789 to auth",
     "sk-" "abcdefghijklmnopqrstuvwx0123456789"),
    ("anthropic key",
     "ANTHROPIC_API_KEY=sk-ant-api03-" "AAAAbbbbCCCCddddEEEEffffGGGGhhhh",
     "sk-ant-api03-" "AAAAbbbbCCCCddddEEEEffffGGGGhhhh"),
    ("stripe restricted key",
     "rk-live-0123456789abcdefghijklmnopqrs",
     "rk-live-0123456789abcdefghijklmnopqrs"),
    ("aws access key id",
     "aws_access_key_id AKIA" "IOSFODNN7EXAMPLE here",
     "AKIA" "IOSFODNN7EXAMPLE"),
    ("aws temporary key id",
     "ASIAY34FZKBOKMUTVV7A was minted by sts",
     "ASIAY34FZKBOKMUTVV7A"),
    ("bearer token",
     "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789",
     "abcdefghijklmnopqrstuvwxyz0123456789"),
    ("pem private key",
     "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n"
     "-----END RSA PRIVATE KEY-----",
     "MIIEowIBAAKCAQEA"),
    ("openssh private key",
     "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n"
     "-----END OPENSSH PRIVATE KEY-----",
     "b3BlbnNzaC1rZXktdjEAAAAA"),
    ("password assignment",
     'db_password = "hunter2horse battery"',
     "hunter2horse"),
    ("api_key assignment",
     "api_key: 9f8e7d6c5b4a39281706",
     "9f8e7d6c5b4a39281706"),

    # ---- the documented misses --------------------------------------------
    ("bare jwt",
     "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
     "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkogRCJ9."
     "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c expired",
     "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
    ("jwt with an empty signature (alg none)",
     "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhbm9uIn0.",
     "eyJhbGciOiJub25lIn0"),
    ("github classic pat",
     "clone with ghp_" "16CharsAndThenSomeMore0123456789abcd",
     "ghp_" "16CharsAndThenSomeMore0123456789abcd"),
    ("github oauth token",
     "gho_16CharsAndThenSomeMore0123456789abcd",
     "gho_16CharsAndThenSomeMore0123456789abcd"),
    ("github server-to-server token",
     "ghs_" "16CharsAndThenSomeMore0123456789abcd",
     "ghs_" "16CharsAndThenSomeMore0123456789abcd"),
    ("github fine-grained pat",
     "github_pat_" "11ABCDEFG0abcdefghijkl_"
     "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ",
     "github_pat_" "11ABCDEFG0abcdefghijkl_"),
    ("slack bot token",
     "xoxb-" "123456789012-1234567890123-abcdefghijklmnopqrstuvwx",
     "xoxb-" "123456789012-1234567890123-abcdefghijklmnopqrstuvwx"),
    ("slack user token",
     "xoxp-" "123456789012-123456789012-123456789012-abcdef0123456789",
     "xoxp-" "123456789012-123456789012-123456789012-abcdef0123456789"),
    ("slack app-level token",
     "xapp-" "1-A012BCDEFGH-1234567890123-abcdef0123456789abcdef",
     "xapp-" "1-A012BCDEFGH-1234567890123-abcdef0123456789abcdef"),
    ("google api key",
     "maps key AIza" "SyA0123456789abcdefghijklmnopqrstuv here",
     "AIza" "SyA0123456789abcdefghijklmnopqrstuv"),
    ("sendgrid key",
     "SG.abcdefghijklmnopqrstuv.0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHI",
     "SG.abcdefghijklmnopqrstuv."),
    ("npm token",
     "npm_0123456789abcdefghijklmnopqrstuvwxyzAB",
     "npm_0123456789abcdefghijklmnopqrstuvwxyzAB"),
    ("pypi token",
     "pypi-AgEIcHlwaS5vcmcCJDAwMDAwMDAwLTAwMDAtMDAwMA",
     "pypi-AgEIcHlwaS5vcmcCJDAwMDAwMDAwLTAwMDAtMDAwMA"),
    ("password inside a url",
     "psql postgres://admin:s3cr3tpassw0rd@db.internal:5432/app",
     "s3cr3tpassw0rd"),
    ("password inside an https url",
     "git clone https://matt:ghp_" "notatoken0123456789@github.com/x/y.git",
     "ghp_" "notatoken0123456789"),
    ("basic auth header",
     "Authorization: Basic bWF0dDpodW50ZXIyaHVudGVyMg==",
     "bWF0dDpodW50ZXIyaHVudGVyMg=="),
]

#: (label, text) — must come back BYTE-IDENTICAL with zero hits.
#:
#: This half is the expensive half to get right and the reason the pattern list
#: stays shape-based. Every one of these is something an agent legitimately says.
NEGATIVES = [
    ("git commit sha", "fixed in 9f2c4a1b8e7d6c5b4a39281706f5e4d3c2b1a098"),
    ("sha256 digest",
     "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("uuid", "run id 3f2504e0-4f89-11d3-9a0c-0305e82c3301"),
    ("semver and hex colour", "v1.24.0 shipped, accent #a0f2c4"),
    # Generic username on purpose: `scripts/scrub_gate.py` scans this tree for
    # author-identifying tokens and correctly flagged the real one when this
    # corpus first landed. What the case is testing is the backslashes and the
    # length, neither of which needs a real account name.
    ("windows path",
     r"C:\Users\dev\AppData\Local\Temp\build\artifacts\notes.md"),
    ("base64 of ordinary text",
     "decoded from VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wZWQgb3ZlciB0aGUgZG9n"),
    ("data uri fragment",
     "src=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"),
    ("the word password with no value", "reset your password from the login page"),
    ("password with a redaction placeholder", "password = [REDACTED]"),
    ("prose about bearer tokens",
     "send a Bearer token in the Authorization header"),
    ("AKIA as a prefix", "AKIA is the prefix AWS uses for long-lived keys"),
    ("jwt header alone, not a jwt", "the header decodes to eyJhbGciOiJIUzI1NiJ9"),
    ("npm config variable", "set npm_config_registry to the mirror"),
    # These two exist because a control pass proved the pair above did NOT test
    # what it looked like it tested: `npm_config_registry` ends in more word
    # characters, so the trailing `\b` already refuses it and the length floor was
    # never exercised. These end cleanly, so the floor is the only thing standing
    # between them and a mangled sentence.
    ("npm package name", "the npm_check tool reports outdated deps"),
    ("pypi mirror name", "install from the pypi-mirror we host"),
    ("a normal https url", "see https://github.com/anthropics/claude-code/issues"),
    ("a postgres url with no password", "postgres://reader@db.internal:5432/app"),
    ("sk- in prose", "the sk-ish naming convention is unfortunate"),
    ("long identifier", "AgentRunnerCheckpointRetentionEnforcementPolicy"),
    ("hex bytes in a hexdump", "00000000  7f 45 4c 46 02 01 01 00  00 00 00 00"),
    ("a long file hash line",
     "d41d8cd98f00b204e9800998ecf8427e  emptyfile.txt"),
    ("no secrets at all", "the run finished in 3.4s with 12 tool calls"),
]


def _redacted(text: str, secret: str) -> bool:
    cleaned, hits = _scrub_text(text)
    return hits > 0 and secret not in cleaned


def recall() -> tuple:
    """(caught, total, [labels missed])."""
    missed = [label for label, text, secret in POSITIVES
              if not _redacted(text, secret)]
    return len(POSITIVES) - len(missed), len(POSITIVES), missed


def precision_failures() -> list:
    """Negatives the scrubber touched. Each one is mangled legitimate output."""
    out = []
    for label, text in NEGATIVES:
        cleaned, hits = _scrub_text(text)
        if hits or cleaned != text:
            out.append((label, cleaned))
    return out


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def test_precision_is_perfect_on_the_hard_negatives():
    """Precision first, and it is a HARD 1.00 rather than a floor.

    This is the constraint the pattern list was designed around and it does not
    get traded for recall: a false positive silently corrupts an answer nobody
    will trace back to the scrubber, while a false negative leaks a string that
    at least still looks like itself downstream. One entropy heuristic would
    lift recall a long way and break several of these on the same afternoon.
    """
    failures = precision_failures()
    assert not failures, "\n".join(
        f"  {label}: {cleaned!r}" for label, cleaned in failures)


def test_recall_is_measured_and_does_not_regress():
    """The number this file exists to produce.

    A floor rather than an exact figure so adding a new credential shape to the
    corpus is cheap — the point is that recall can never quietly fall, not that
    it is frozen. If this fails because somebody added positives faster than
    patterns, that is the corpus doing its job.
    """
    caught, total, missed = recall()
    assert caught == total, (
        f"recall {caught}/{total} = {caught / total:.0%}; missed:\n"
        + "\n".join(f"  {label}" for label in missed))


@pytest.mark.parametrize("label,text,secret",
                         POSITIVES, ids=[p[0] for p in POSITIVES])
def test_each_known_shape_is_redacted(label, text, secret):
    """Per-shape, so a failure names the shape instead of a percentage."""
    cleaned, hits = _scrub_text(text)
    assert hits > 0, f"{label}: nothing matched"
    assert secret not in cleaned, f"{label}: value survived in {cleaned!r}"


@pytest.mark.parametrize("label,text", NEGATIVES, ids=[n[0] for n in NEGATIVES])
def test_each_benign_string_is_untouched(label, text):
    cleaned, hits = _scrub_text(text)
    assert (cleaned, hits) == (text, 0), f"{label}: false positive"


def test_a_redaction_keeps_the_key_name_legible():
    """The one deliberate asymmetry in the output: assignment-shaped secrets
    keep their key and lose only the value, because "something was redacted
    here" is more useful to a reader than a hole."""
    cleaned, _hits = _scrub_text("api_key: 9f8e7d6c5b4a39281706")
    assert cleaned == f"api_key: {_REDACTED}"


def test_several_secrets_in_one_string_are_all_caught():
    text = ("export GITHUB_TOKEN=ghp_" "16CharsAndThenSomeMore0123456789abcd\n"
            "export SLACK=xoxb-" "123456789012-1234567890123-abcdefghijklmnopqrstuvwx")
    cleaned, hits = _scrub_text(text)
    assert hits >= 2
    assert "ghp_16Chars" not in cleaned and "xoxb-1234" not in cleaned
