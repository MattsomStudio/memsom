"""Session-wide test hygiene.

The bridge importer now indexes every node it creates (retrieve.index_node),
and index_node asks the active embedding backend for a vector.  With a live
Ollama on the box that is ~1s per node — enough to turn the suite from ~2.5 min
into ~4.5 min, and it makes every memory-importing test depend on the network.

So: every test runs BM25-only (MEMDAG_EMBED_BACKEND=bm25) unless its module is
one that exercises the embedding paths on purpose — those keep whatever the
environment (or the test itself) sets.  Modules that manage the variable in
their own setUp/tearDown are unaffected: the fixture restores the prior value.
"""
import os

import pytest

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
