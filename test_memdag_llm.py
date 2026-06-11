#!/usr/bin/env python3
"""Tests for memdag_llm — the opt-in LLM answer path.

Run:
  python -W error::DeprecationWarning -m unittest discover \
      -s C:\\Users\\you\\memdag -p test_memdag_llm.py \
      -t C:\\Users\\you\\memdag -v

No real Ollama needed: urllib.request.urlopen is patched in all network tests.
No DB needed: llm_compose operates purely on source-row tuples passed in.
"""

import io
import json
import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

warnings.simplefilter("error", DeprecationWarning)

# Make sure the module directory is importable regardless of cwd
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import memdag
import memdag_llm
from memdag_llm import LlmUnavailable, llm_compose, ping, resolve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal file-like object returned by mocked urlopen.

    Supports both the context-manager protocol (with urlopen(...) as resp:)
    and .read() for the raw bytes.
    """

    def __init__(self, body: dict, status: int = 200):
        self._data = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Fixture — two live source rows (same style as TestCompose.SRC in test_memdag.py)
# ---------------------------------------------------------------------------

SOURCES = [
    (1, "Nebula needs a lighthouse node with a public IP for hole punching.", "endorsed", 3, None),
    (2, "Use UDP port 4242 for nebula tunnels on every host.", "user", 2, None),
]

QUESTION = "How should I configure Nebula?"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLlmCompose(unittest.TestCase):

    def _make_response(self, response_text):
        return FakeResponse({"response": response_text})

    # ------------------------------------------------------------------
    # 1. Happy path: citations preserved in response
    # ------------------------------------------------------------------
    def test_up_citations_preserved(self):
        good_answer = (
            "- Lighthouse with public IP required. [mem:1|endorsed]\n"
            "- Use UDP 4242 everywhere. [mem:2|user]"
        )
        captured_req = {}

        def fake_urlopen(req, timeout=None):
            # Capture the request for later inspection
            captured_req["req"] = req
            return FakeResponse({"response": good_answer})

        with patch("memdag_llm.urllib.request.urlopen", side_effect=fake_urlopen):
            text, used = llm_compose(QUESTION, SOURCES)

        # used must be the sorted list of IDs the LLM cited
        self.assertEqual(used, [1, 2])
        # Both citation tags must appear in the returned text
        self.assertIn("[mem:1|endorsed]", text)
        self.assertIn("[mem:2|user]", text)
        # The header must mark this as LLM-composed
        self.assertIn("LLM-composed", text)

        # Verify the request body contained the model name and the deterministic bullets
        req_body = json.loads(captured_req["req"].data.decode("utf-8"))
        m, _ = resolve()
        self.assertEqual(req_body["model"], m)
        # Both [mem:1| and [mem:2| deterministic bullets must be in the prompt
        self.assertIn("[mem:1|", req_body["prompt"])
        self.assertIn("[mem:2|", req_body["prompt"])

    # ------------------------------------------------------------------
    # 2. Ollama down -> LlmUnavailable; fallback is deterministic compose
    # ------------------------------------------------------------------
    def test_down_raises_and_fallback_identical(self):
        import urllib.error as _ue

        with patch("memdag_llm.urllib.request.urlopen",
                   side_effect=_ue.URLError("connection refused")):
            with self.assertRaises(LlmUnavailable):
                llm_compose(QUESTION, SOURCES)

        # Caller falls back to deterministic compose — must contain same citations
        det_text, det_used = memdag.compose(QUESTION, SOURCES)
        self.assertIn("[mem:1|", det_text)
        self.assertIn("[mem:2|", det_text)

    # ------------------------------------------------------------------
    # 3. Response has bullets but no [mem: tags -> LlmUnavailable
    # ------------------------------------------------------------------
    def test_stripped_citations_rejected(self):
        bad_answer = (
            "- Lighthouse with public IP required.\n"
            "- Use UDP 4242 everywhere."
        )
        with patch("memdag_llm.urllib.request.urlopen",
                   return_value=FakeResponse({"response": bad_answer})):
            with self.assertRaises(LlmUnavailable):
                llm_compose(QUESTION, SOURCES)

    # ------------------------------------------------------------------
    # 4. Invented citation (ID not in source set) -> LlmUnavailable
    # ------------------------------------------------------------------
    def test_invented_citation_rejected(self):
        bad_answer = (
            "- Lighthouse with public IP required. [mem:1|endorsed]\n"
            "- Extra claim from nowhere. [mem:999|external]"
        )
        with patch("memdag_llm.urllib.request.urlopen",
                   return_value=FakeResponse({"response": bad_answer})):
            with self.assertRaises(LlmUnavailable):
                llm_compose(QUESTION, SOURCES)

    # ------------------------------------------------------------------
    # 5. One good cited bullet + one uncited claim line -> LlmUnavailable
    # ------------------------------------------------------------------
    def test_uncited_claim_line_rejected(self):
        bad_answer = (
            "- Lighthouse with public IP required. [mem:1|endorsed]\n"
            "- bonus claim with no tag at all"
        )
        with patch("memdag_llm.urllib.request.urlopen",
                   return_value=FakeResponse({"response": bad_answer})):
            with self.assertRaises(LlmUnavailable):
                llm_compose(QUESTION, SOURCES)

    # ------------------------------------------------------------------
    # 6. Env-var resolution (MEMDAG_LLM_MODEL / MEMDAG_LLM_URL)
    # ------------------------------------------------------------------
    def test_env_resolution(self):
        custom_model = "custom-model:test"
        custom_url = "http://localhost:9999/api/generate"

        orig_model = os.environ.pop("MEMDAG_LLM_MODEL", None)
        orig_url = os.environ.pop("MEMDAG_LLM_URL", None)
        os.environ["MEMDAG_LLM_MODEL"] = custom_model
        os.environ["MEMDAG_LLM_URL"] = custom_url

        try:
            m, url = resolve()
            self.assertEqual(m, custom_model)
            self.assertEqual(url, custom_url)

            # Verify the request body uses the env model
            captured_req = {}

            def fake_urlopen(req, timeout=None):
                captured_req["req"] = req
                good_answer = (
                    "- Lighthouse with public IP required. [mem:1|endorsed]\n"
                    "- Use UDP 4242 everywhere. [mem:2|user]"
                )
                return FakeResponse({"response": good_answer})

            with patch("memdag_llm.urllib.request.urlopen", side_effect=fake_urlopen):
                llm_compose(QUESTION, SOURCES)

            req_body = json.loads(captured_req["req"].data.decode("utf-8"))
            self.assertEqual(req_body["model"], custom_model)
        finally:
            # Restore env
            os.environ.pop("MEMDAG_LLM_MODEL", None)
            os.environ.pop("MEMDAG_LLM_URL", None)
            if orig_model is not None:
                os.environ["MEMDAG_LLM_MODEL"] = orig_model
            if orig_url is not None:
                os.environ["MEMDAG_LLM_URL"] = orig_url

    # ------------------------------------------------------------------
    # 7. Empty sources -> ValueError (not LlmUnavailable)
    # ------------------------------------------------------------------
    def test_no_sources_is_valueerror_not_unavailable(self):
        with self.assertRaises(ValueError):
            llm_compose(QUESTION, [])

    # ------------------------------------------------------------------
    # 8. <think>...</think> block stripped before citation check
    # ------------------------------------------------------------------
    def test_think_block_stripped(self):
        answer_with_think = (
            "<think>internal reasoning the user should never see</think>\n"
            "- Lighthouse with public IP required. [mem:1|endorsed]\n"
            "- Use UDP 4242 everywhere. [mem:2|user]"
        )
        with patch("memdag_llm.urllib.request.urlopen",
                   return_value=FakeResponse({"response": answer_with_think})):
            text, used = llm_compose(QUESTION, SOURCES)

        self.assertEqual(used, [1, 2])
        self.assertNotIn("internal", text)
        self.assertIn("[mem:1|endorsed]", text)


class TestPing(unittest.TestCase):
    """Tests for the ping() helper."""

    def test_ping_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("memdag_llm.urllib.request.urlopen", return_value=mock_resp):
            self.assertTrue(ping())

    def test_ping_false_on_exception(self):
        import urllib.error as _ue
        with patch("memdag_llm.urllib.request.urlopen",
                   side_effect=_ue.URLError("refused")):
            self.assertFalse(ping())


class TestResolve(unittest.TestCase):
    """Tests for resolve() parameter/env priority."""

    def setUp(self):
        self._old_model = os.environ.pop("MEMDAG_LLM_MODEL", None)
        self._old_url = os.environ.pop("MEMDAG_LLM_URL", None)

    def tearDown(self):
        os.environ.pop("MEMDAG_LLM_MODEL", None)
        os.environ.pop("MEMDAG_LLM_URL", None)
        if self._old_model is not None:
            os.environ["MEMDAG_LLM_MODEL"] = self._old_model
        if self._old_url is not None:
            os.environ["MEMDAG_LLM_URL"] = self._old_url

    def test_explicit_args_win(self):
        os.environ["MEMDAG_LLM_MODEL"] = "env-model"
        m, url = resolve(model="explicit-model", base_url="http://explicit/api/generate")
        self.assertEqual(m, "explicit-model")
        self.assertEqual(url, "http://explicit/api/generate")

    def test_defaults_used_when_nothing_set(self):
        m, url = resolve()
        self.assertEqual(m, memdag_llm.DEFAULT_MODEL)
        self.assertEqual(url, memdag_llm.DEFAULT_URL)


class TestRegister(unittest.TestCase):
    """Smoke test: register() mounts a subcommand without crashing."""

    def test_register_adds_llm_check(self):
        import argparse
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        memdag_llm.register(sub)
        # Parsing "llm-check" must not raise
        args = p.parse_args(["llm-check"])
        self.assertEqual(args.command, "llm-check")
        self.assertTrue(callable(args.func))


class TestMigrate(unittest.TestCase):
    """migrate(conn) must be a no-op (no DB interaction needed)."""

    def test_migrate_is_noop(self):
        # Pass a MagicMock — if migrate() tries DB calls it will blow up
        mock_conn = MagicMock()
        result = memdag_llm.migrate(mock_conn)
        self.assertIsNone(result)
        mock_conn.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
