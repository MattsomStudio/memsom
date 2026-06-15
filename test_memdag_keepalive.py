#!/usr/bin/env python3
"""Tests for the Ollama keep_alive VRAM-hygiene contract.

Shipped default: memdag does NOT touch keep_alive — the request body OMITS
the key entirely so Ollama applies its own native default (model stays warm).
When MEMDAG_OLLAMA_KEEP_ALIVE is set, every memdag call to the Ollama API
(/api/generate via memdag_llm and memdag_compact, /api/embeddings via
memdag_retrieve) must stamp that value into the request body — 0 = unload
immediately after the call, "10m" = hold warm longer.

No real Ollama needed: urllib.request.urlopen is patched in all network
tests (same FakeResponse pattern as test_memdag_llm.py).  No DB needed.

Run:
  python -W error::DeprecationWarning -m unittest discover \
      -s <repo> -p test_memdag_keepalive.py \
      -t <repo> -v
"""

import json
import os
import sys
import unittest
import urllib.error
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)

# Make sure the module directory is importable regardless of cwd
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import memdag_compact
import memdag_llm
import memdag_retrieve
from memdag_llm import LlmUnavailable, keep_alive, llm_compose


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal file-like object returned by mocked urlopen."""

    def __init__(self, body: dict, status: int = 200):
        self._data = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class EnvIsolation(unittest.TestCase):
    """Base: save/restore MEMDAG_OLLAMA_KEEP_ALIVE around every test."""

    def setUp(self):
        self._old = os.environ.pop("MEMDAG_OLLAMA_KEEP_ALIVE", None)

    def tearDown(self):
        os.environ.pop("MEMDAG_OLLAMA_KEEP_ALIVE", None)
        if self._old is not None:
            os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = self._old


SOURCES = [
    (1, "Nebula needs a lighthouse node with a public IP for hole punching.",
     "endorsed", 3, None),
    (2, "Use UDP port 4242 for nebula tunnels on every host.", "user", 2, None),
]
QUESTION = "How should I configure Nebula?"
GOOD_ANSWER = (
    "- Lighthouse with public IP required. [mem:1|endorsed]\n"
    "- Use UDP 4242 everywhere. [mem:2|user]"
)


def _capture_urlopen(captured, body):
    """Return a fake urlopen that records the Request and replies with body."""
    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        return FakeResponse(body)
    return fake_urlopen


# ---------------------------------------------------------------------------
# 1. The shared helper itself
# ---------------------------------------------------------------------------

class TestKeepAliveHelper(EnvIsolation):

    def test_unset_returns_none(self):
        # Shipped default: defer to Ollama's native keep_alive.
        self.assertIsNone(keep_alive())

    def test_blank_env_returns_none(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "   "
        self.assertIsNone(keep_alive())

    def test_zero_becomes_int_zero(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "0"
        self.assertEqual(keep_alive(), 0)
        self.assertIsInstance(keep_alive(), int)

    def test_duration_string_passed_through(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "10m"
        self.assertEqual(keep_alive(), "10m")

    def test_numeric_string_becomes_int(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "300"
        self.assertEqual(keep_alive(), 300)
        self.assertIsInstance(keep_alive(), int)


class TestWithKeepAlive(EnvIsolation):
    """The shared body-stamping helper."""

    def test_unset_leaves_body_untouched(self):
        body = memdag_llm._with_keep_alive({"model": "m"})
        self.assertNotIn("keep_alive", body)

    def test_zero_stamps_int_zero(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "0"
        body = memdag_llm._with_keep_alive({"model": "m"})
        self.assertEqual(body["keep_alive"], 0)


# ---------------------------------------------------------------------------
# 2. /api/generate via memdag_llm.llm_compose
# ---------------------------------------------------------------------------

class TestGenerateCarriesKeepAlive(EnvIsolation):

    def _body_for_llm_compose(self):
        captured = {}
        with patch("memdag_llm.urllib.request.urlopen",
                   side_effect=_capture_urlopen(captured,
                                                {"response": GOOD_ANSWER})):
            llm_compose(QUESTION, SOURCES)
        return json.loads(captured["req"].data.decode("utf-8"))

    def test_default_omits_keep_alive(self):
        body = self._body_for_llm_compose()
        self.assertNotIn("keep_alive", body)

    def test_zero_unloads_immediately(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "0"
        body = self._body_for_llm_compose()
        self.assertEqual(body["keep_alive"], 0)

    def test_duration_override_keeps_warm(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "10m"
        body = self._body_for_llm_compose()
        self.assertEqual(body["keep_alive"], "10m")

    def test_ollama_down_fallback_unchanged(self):
        with patch("memdag_llm.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaises(LlmUnavailable):
                llm_compose(QUESTION, SOURCES)


# ---------------------------------------------------------------------------
# 3. /api/embeddings via memdag_retrieve._call_ollama_embed
# ---------------------------------------------------------------------------

class TestEmbeddingsCarriesKeepAlive(EnvIsolation):

    def _body_for_embed(self):
        captured = {}
        with patch("memdag_retrieve.urllib.request.urlopen",
                   side_effect=_capture_urlopen(captured,
                                                {"embedding": [0.1, 0.2]})):
            vec = memdag_retrieve._call_ollama_embed("probe text")
        self.assertEqual(vec, [0.1, 0.2])  # reply still parsed correctly
        return json.loads(captured["req"].data.decode("utf-8"))

    def test_default_omits_keep_alive(self):
        body = self._body_for_embed()
        self.assertNotIn("keep_alive", body)

    def test_zero_unloads_immediately(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "0"
        body = self._body_for_embed()
        self.assertEqual(body["keep_alive"], 0)

    def test_duration_override_keeps_warm(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "10m"
        body = self._body_for_embed()
        self.assertEqual(body["keep_alive"], "10m")

    def test_embed_failure_still_raises_for_silent_degrade(self):
        # vector_search/_vector_sims rely on _call_ollama_embed raising so
        # they can degrade to BM25-only; keep_alive must not swallow that.
        with patch("memdag_retrieve.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            with self.assertRaises(Exception):
                memdag_retrieve._call_ollama_embed("probe text")


# ---------------------------------------------------------------------------
# 4. /api/generate via memdag_compact._llm_summarize
# ---------------------------------------------------------------------------

class TestCompactSummaryCarriesKeepAlive(EnvIsolation):

    ROWS = [(1, "Nebula lighthouse must have a public IP."),
            (2, "Use UDP 4242 for nebula tunnels.")]

    def _body_for_summary(self):
        captured = {}
        with patch("urllib.request.urlopen",
                   side_effect=_capture_urlopen(
                       captured, {"response": "- consolidated bullet"})):
            memdag_compact._llm_summarize(self.ROWS)
        return json.loads(captured["req"].data.decode("utf-8"))

    def test_default_omits_keep_alive(self):
        body = self._body_for_summary()
        self.assertNotIn("keep_alive", body)

    def test_zero_unloads_immediately(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "0"
        body = self._body_for_summary()
        self.assertEqual(body["keep_alive"], 0)

    def test_duration_override_keeps_warm(self):
        os.environ["MEMDAG_OLLAMA_KEEP_ALIVE"] = "10m"
        body = self._body_for_summary()
        self.assertEqual(body["keep_alive"], "10m")


if __name__ == "__main__":
    unittest.main()
