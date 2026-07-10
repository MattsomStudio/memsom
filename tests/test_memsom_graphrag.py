#!/usr/bin/env python3
"""Tests for retrieve_graph — GraphRAG-lite re-ranking over rel_edges.

The vector (Ollama) layer is pinned OFF so BM25 alone determines ranking and the
"B ranks just below the cutoff, gets promoted via a link to A" scenario is exact.

Run:  python -m unittest discover -s . -p test_memsom_graphrag.py
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("error", DeprecationWarning)

import memsom
from memsom.integrity import confid as memsom_confid
from memsom.integrity import quarantine as memsom_quarantine
from memsom.integrity import redact as memsom_redact
from memsom.retrieval import relate as memsom_relate
from memsom.retrieval import retrieve as memsom_retrieve
from memsom.storage import schema as memsom_schema


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        memsom_retrieve.migrate(self.conn)
        memsom_confid.migrate(self.conn)
        memsom_quarantine.migrate(self.conn)
        memsom_redact.migrate(self.conn)
        memsom_relate.migrate(self.conn)
        # Pin the vector layer off -> deterministic BM25-only ranking (also the
        # supported "Ollama down" degradation path).
        self._patcher = patch(
            "memsom.retrieval.retrieve._call_ollama_embed", side_effect=Exception("offline"))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel="user"):
        with self.conn:
            nid = memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])
        memsom_retrieve.index_node(self.conn, nid)
        return nid

    def ids(self, rows):
        return [r[0] for r in rows]


# Build a deterministic ranking A > F > B over the term "quux":
#   A: short doc, many hits   -> rank 1 (a seed)
#   F: medium doc, fewer hits -> rank 2
#   B: long doc, one hit      -> rank 3 (just below a k=2 cutoff)
_A = "quux quux quux quux quux alpha"
_F = "quux quux quux beta"
_B = "quux gamma delta epsilon zeta eta theta iota kappa lambda"


class TestPromotion(Base):
    def test_link_promotes_subcutoff_relevant_node(self):
        a = self.add(_A)
        f = self.add(_F)
        b = self.add(_B)

        plain = self.ids(memsom_retrieve.retrieve(self.conn, "quux", k=2))
        self.assertIn(a, plain)
        self.assertNotIn(b, plain)  # B ranks below the k=2 cutoff

        # Link B to the top seed A; B should now be promoted into the top-2.
        memsom_relate.relate(self.conn, a, b, kind="wikilink")
        graph = self.ids(memsom_retrieve.retrieve_graph(self.conn, "quux", k=2))
        self.assertIn(b, graph, "linked, relevant node was not promoted")
        self.assertIn(a, graph)

    def test_promotion_survives_excluded_high_ranker(self):
        # Regression: an above-clearance node that OUTRANKS the relevant pool
        # member in raw BM25 must not consume the scan window and silently drop
        # the pool member from `base` before it can be graph-promoted. (Found via
        # e2e: secret node outranked B at public clearance, B never promoted.)
        a = self.add(_A)
        self.add(_F)
        b = self.add(_B)
        secret = self.add("quux quux quux quux secret omega")  # strong + SECRET
        with self.conn:
            self.conn.execute("UPDATE nodes SET conf_label = 2 WHERE id = ?", (secret,))
        memsom_relate.relate(self.conn, a, b, kind="wikilink")
        graph = self.ids(memsom_retrieve.retrieve_graph(
            self.conn, "quux", k=2, clearance="public"))
        self.assertIn(b, graph, "promotion blocked by an excluded high-ranker")
        self.assertNotIn(secret, graph)  # and the secret node still never leaks

    def test_unlinked_b_stays_excluded(self):
        # Without the edge, retrieve_graph must match plain retrieve exactly.
        a = self.add(_A); self.add(_F); b = self.add(_B)
        plain = self.ids(memsom_retrieve.retrieve(self.conn, "quux", k=2))
        graph = self.ids(memsom_retrieve.retrieve_graph(self.conn, "quux", k=2))
        self.assertEqual(graph, plain)
        self.assertNotIn(b, graph)
        self.assertIn(a, graph)


class TestNoInjection(Base):
    def test_irrelevant_neighbor_never_injected(self):
        # Z shares no terms with the query -> zero base score. Even linked to the
        # top hit, it must NOT enter the answer pool (compose force-bullets it).
        a = self.add(_A)
        self.add(_F)
        z = self.add("completely unrelated content about sailboats and weather")
        memsom_relate.relate(self.conn, a, z, kind="wikilink")
        graph = self.ids(memsom_retrieve.retrieve_graph(self.conn, "quux", k=8))
        self.assertNotIn(z, graph)


class TestTrustGate(Base):
    """A crafted edge must not leak a tainted / above-clearance / derived node."""

    def _link_then_graph(self, a, b, clearance="topsecret", k=8):
        memsom_relate.relate(self.conn, a, b, kind="wikilink")
        return self.ids(
            memsom_retrieve.retrieve_graph(self.conn, "quux", k=k, clearance=clearance))

    def test_above_clearance_neighbor_not_leaked(self):
        a = self.add(_A)
        b = self.add(_B)
        with self.conn:  # mark B SECRET (conf_label 2)
            self.conn.execute("UPDATE nodes SET conf_label = 2 WHERE id = ?", (b,))
        # Ask at PUBLIC clearance: B is above clearance and must never appear,
        # even though it is relevant AND linked to the top hit.
        graph = self._link_then_graph(a, b, clearance="public")
        self.assertNotIn(b, graph)
        self.assertIn(a, graph)

    def test_quarantined_neighbor_not_leaked(self):
        a = self.add(_A)
        b = self.add(_B)
        with self.conn:
            self.conn.execute("UPDATE nodes SET status = 'quarantined' WHERE id = ?", (b,))
        graph = self._link_then_graph(a, b)
        self.assertNotIn(b, graph)

    def test_redacted_neighbor_not_leaked(self):
        a = self.add(_A)
        b = self.add(_B)
        memsom_redact.redact_node(self.conn, b, "test", cascade=False)
        graph = self._link_then_graph(a, b)
        self.assertNotIn(b, graph)

    def test_tombstoned_neighbor_not_leaked(self):
        a = self.add(_A)
        b = self.add(_B)
        memsom.revoke_cascade(self.conn, b, "test")
        graph = self._link_then_graph(a, b)
        self.assertNotIn(b, graph)

    def test_agent_derived_neighbor_not_leaked(self):
        # An exported (agent-derived) note can carry wikilink edges; it must never
        # be promoted into the SOURCE pool (sources are non-derived by contract).
        a = self.add(_A)
        with self.conn:
            d = memsom.insert_node(self.conn, _B, "agent-derived", memsom.RANK["agent-derived"])
        memsom_retrieve.index_node(self.conn, d)  # no-op for agent-derived
        graph = self._link_then_graph(a, d)
        self.assertNotIn(d, graph)


class TestPoolSubsetInvariant(Base):
    """The load-bearing safety proof: retrieve_graph output is ALWAYS a subset of
    the trusted retrieval pool, even under hostile edges to excluded nodes.
    Concretely this guards the in-loop trust gate `if n in pool and base.get(n)>0`
    (memsom_retrieve.py) — removing that gate makes these assertions fail. The
    output is built from base.keys() (itself a pool subset), so a tainted/
    above-clearance neighbor can never reach the returned rows.
    """

    def test_output_subset_of_pool_under_hostile_edges(self):
        a = self.add(_A)                       # strong, clean seed
        above = self.add(_F)                   # relevant but will be above-clearance
        quar = self.add("quux quux delta")     # relevant but quarantined
        with self.conn:
            self.conn.execute("UPDATE nodes SET conf_label = 3 WHERE id = ?", (above,))
            self.conn.execute("UPDATE nodes SET status = 'quarantined' WHERE id = ?", (quar,))
            d = memsom.insert_node(self.conn, _B, "agent-derived",
                                   memsom.RANK["agent-derived"])  # relevant but derived
        # Hostile edges from the clean seed to every excluded node.
        for bad in (above, quar, d):
            memsom_relate.relate(self.conn, a, bad, kind="wikilink")

        clr = memsom_confid.parse_conf("public")
        pool = memsom_retrieve._build_retrieve_pool(
            self.conn, clr, None, True, True)
        out = set(self.ids(
            memsom_retrieve.retrieve_graph(self.conn, "quux", k=8, clearance="public")))
        self.assertTrue(out <= pool, f"output {out} escaped the trusted pool {pool}")
        for bad in (above, quar, d):
            self.assertNotIn(bad, out)


class TestGraceful(Base):
    def test_empty_rel_edges_equals_retrieve(self):
        self.add(_A); self.add(_F); self.add(_B)
        for k in (1, 2, 3):
            self.assertEqual(
                self.ids(memsom_retrieve.retrieve_graph(self.conn, "quux", k=k)),
                self.ids(memsom_retrieve.retrieve(self.conn, "quux", k=k)),
                f"graph diverged from retrieve with no edges (k={k})",
            )

    def test_zero_k_returns_empty(self):
        self.add(_A)
        self.assertEqual(memsom_retrieve.retrieve_graph(self.conn, "quux", k=0), [])

    def test_empty_pool_returns_empty(self):
        # No source nodes -> empty.
        self.assertEqual(memsom_retrieve.retrieve_graph(self.conn, "quux", k=8), [])


class TestAskIntegration(Base):
    def test_ask_graph_cites_promoted_node(self):
        from memsom.interface import cli as memsom_cli
        a = self.add(_A)
        self.add(_F)
        b = self.add(_B)
        memsom_relate.relate(self.conn, a, b, kind="wikilink")
        # cmd_ask opens its own connection; commit our writes first.
        self.conn.commit()
        import argparse
        args = argparse.Namespace(
            question="quux", clearance="topsecret", anticipate=False, threshold=0.35,
            llm=False, model=None, retrieve=True, graph=True, hops=1, topk=2,
        )
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            memsom_cli.cmd_ask(args)
        out = buf.getvalue()
        self.assertIn(f"[mem:{b}|", out, "promoted node not cited in the answer")


if __name__ == "__main__":
    unittest.main()
