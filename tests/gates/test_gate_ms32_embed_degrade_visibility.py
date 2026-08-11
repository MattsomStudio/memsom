"""GATE for MS-32 -- a momentary embedder outage silently, permanently demotes
a node.

`index_node` used to swallow both the bge AND Ollama failure paths with a
bare `except Exception: pass`, then return True regardless. `index_all`
printed "indexed N source node(s)" whether every node got a vector or none
did, and nothing ever re-attempted the vector once the embedder recovered.
The bge path already warned once per process on fallback; the DEFAULT Ollama
path -- what most installs actually run -- printed nothing at all.

CONTROL-TESTED: on the pre-fix tree, `retrieval_degraded` does not exist and
a failed Ollama embed leaves zero trace anywhere queryable.
"""

import memsom
from memsom.retrieval import retrieve as memsom_retrieve


def test_a_failed_embed_is_queued_and_warned(conn, monkeypatch, capsys):
    nid = memsom.insert_node(conn, "content that needs a vector embed", "user")
    conn.commit()

    def _boom(text):
        raise RuntimeError("ollama: connection refused")

    monkeypatch.setattr(memsom_retrieve, "_call_ollama_embed", _boom)
    # Force the default/no-bge path deterministically.
    from memsom.retrieval import embed as memsom_embed
    monkeypatch.setattr(memsom_embed, "backend", lambda: "ollama")

    ok = memsom_retrieve.index_node(conn, nid)
    assert ok is True, "BM25 half must still succeed despite the vector failure"

    queued = memsom_retrieve.degraded_nodes(conn)
    assert nid in queued, (
        f"a failed embed left no trace in the MS-32 re-index queue: {queued!r}")

    captured = capsys.readouterr()
    assert captured.err.strip(), (
        "a failed Ollama embed (the DEFAULT backend) produced no stderr "
        "warning -- MS-32's exact asymmetry with the bge path, which does warn")


def test_recovery_clears_the_queue(conn, monkeypatch):
    nid = memsom.insert_node(conn, "content that recovers on retry", "user")
    conn.commit()

    from memsom.retrieval import embed as memsom_embed
    monkeypatch.setattr(memsom_embed, "backend", lambda: "ollama")

    def fails_once(text, _state={"n": 0}):
        _state["n"] += 1
        if _state["n"] == 1:
            raise RuntimeError("ollama: connection refused")
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(memsom_retrieve, "_call_ollama_embed", fails_once)

    memsom_retrieve.index_node(conn, nid)
    assert nid in memsom_retrieve.degraded_nodes(conn), "precondition: queued after failure"

    memsom_retrieve.index_node(conn, nid)
    assert nid not in memsom_retrieve.degraded_nodes(conn), (
        "a node that successfully re-embedded was not cleared from the "
        "re-index queue")


def test_control_a_clean_embed_never_touches_the_queue(conn, monkeypatch):
    """GREEN control: proves the queue mechanism doesn't fire on the happy path."""
    nid = memsom.insert_node(conn, "content that embeds cleanly", "user")
    conn.commit()

    from memsom.retrieval import embed as memsom_embed
    monkeypatch.setattr(memsom_embed, "backend", lambda: "ollama")
    monkeypatch.setattr(memsom_retrieve, "_call_ollama_embed",
                        lambda text: [0.1, 0.2, 0.3])

    memsom_retrieve.index_node(conn, nid)
    assert nid not in memsom_retrieve.degraded_nodes(conn)
