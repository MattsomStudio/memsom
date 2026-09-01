"""Repo-wide pytest safety net (Phase 0, A5.1) + session-wide test hygiene.

Closes, for the whole suite, the hole `tests/gates/conftest.py` (ported from
the pentest) already closes for `tests/gates/` alone: nothing here may ever
touch `~/.memdag/memdag.db`, the real store. See tests/_isolation.py for why
the pin has to be an import-time side effect rather than a fixture body.

Also: the bridge importer now indexes every node it creates (retrieve.index_node),
and index_node asks the active embedding backend for a vector.  With a live
Ollama on the box that is ~1s per node — enough to turn the suite from ~2.5 min
into ~4.5 min, and it makes every memory-importing test depend on the network.

So: every test runs BM25-only (MEMDAG_EMBED_BACKEND=bm25) unless its module is
one that exercises the embedding paths on purpose — those keep whatever the
environment (or the test itself) sets.  Modules that manage the variable in
their own setUp/tearDown are unaffected: the fixture restores the prior value.
"""

from __future__ import annotations

import os

import pytest

from tests import _isolation  # import-time pin; must run before any test module
                               # in this process gets a chance to import memsom


@pytest.fixture(scope="session", autouse=True)
def _memsom_isolation_guard():
    """Assert the pin took, once, before any test body runs."""
    _isolation.assert_pinned()
    yield


def _real_db_stat():
    try:
        st = _isolation.REAL_DB_PATH.stat()
        return (st.st_size, st.st_mtime_ns)
    except FileNotFoundError:
        return None


def pytest_sessionstart(session):
    """Record the real store's stat before a single test has run.

    This is the automated form of the standing exit gate's point 4 ("record
    before AND after; must be identical") — baked into every local run rather
    than a step someone has to remember to run by hand.
    """
    session.config._memsom_real_db_before = _real_db_stat()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_memsom_real_db_before", None)
    after = _real_db_stat()
    if before != after:
        # Not a normal assert: sessionfinish exceptions can be swallowed by
        # some pytest plugins, and this must be impossible to miss.
        msg = (
            f"\n\n::error::REAL STORE CHANGED DURING THE TEST RUN: "
            f"{_isolation.REAL_DB_PATH} was {before} before, {after} after. "
            "The isolation pin was bypassed somewhere in this run.\n\n"
        )
        os.write(2, msg.encode("utf-8", "replace"))
        session.exitstatus = 1


# Modules that test the vector / backend paths themselves (they patch
# _call_ollama_embed, toggle MEMDAG_EMBED_BACKEND, or assert on `embeddings`).
EMBEDDING_AWARE = {
    "test_code_index", "test_doctor", "test_init", "test_memsom_anticipatory",
    "test_memsom_cli", "test_memsom_compact", "test_memsom_contradict",
    "test_memsom_embed", "test_memsom_graphrag", "test_memsom_keepalive",
    "test_memsom_retrieve",
}


@pytest.fixture(autouse=True)
def _bm25_only_unless_embedding_aware(request):
    if request.module.__name__.rsplit(".", 1)[-1] in EMBEDDING_AWARE:
        yield
        return
    prev = os.environ.get("MEMDAG_EMBED_BACKEND")
    os.environ["MEMDAG_EMBED_BACKEND"] = "bm25"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("MEMDAG_EMBED_BACKEND", None)
        else:
            os.environ["MEMDAG_EMBED_BACKEND"] = prev
