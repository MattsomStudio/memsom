#!/usr/bin/env python3
"""Tests for memdag_corroborate — corroboration v1.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s C:\\Users\\you\\memdag -p test_memdag_corroborate.py \
    -t C:\\Users\\you\\memdag -v
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memdag
import memdag_recompute
import memdag_corroborate


class Base(unittest.TestCase):
    """Temp-DB base class — mirrors test_memdag_recompute.py Base exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memdag.get_connection()
        memdag_corroborate.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memdag.insert_node(self.conn, content, channel, memdag.RANK[channel])

    def mk_ext(self, content="external node content"):
        """Insert an external node and return its id."""
        return self.add(content, "external")


# ---------------------------------------------------------------------------
# 1. test_lift_external_to_agent_derived_only
# ---------------------------------------------------------------------------

class TestLiftExternalToAgentDerivedOnly(Base):
    def test_lift_external_to_agent_derived_only(self):
        # Register two independent roots
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")

        # Two external nodes, each asserting the same claim under different roots
        x1 = self.mk_ext("listening on port 4242")
        x2 = self.mk_ext("service running on port 4242")

        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-B")

        lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(lift)

        node = memdag.get_node(self.conn, lift)
        self.assertEqual(node["channel"], "agent-derived")
        self.assertEqual(node["label"], 1)
        # The cap invariant: NEVER user(2) or endorsed(3)
        self.assertNotIn(node["label"], (2, 3))

        # Edges from lift to both asserting nodes must exist
        edges = self.conn.execute(
            "SELECT parent FROM edges WHERE child=? ORDER BY parent", (lift,)
        ).fetchall()
        parents = {r[0] for r in edges}
        self.assertIn(x1, parents)
        self.assertIn(x2, parents)


# ---------------------------------------------------------------------------
# 2. test_below_k_no_lift
# ---------------------------------------------------------------------------

class TestBelowKNoLift(Base):
    def test_below_k_no_lift(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")

        x1 = self.mk_ext("port 4242 open")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")

        # Only one root registered — below k=2 threshold
        result = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNone(result)

        # No agent-derived node should have been minted
        agent_count = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]
        self.assertEqual(agent_count, 0)


# ---------------------------------------------------------------------------
# 3. test_unregistered_root_fail_closed
# ---------------------------------------------------------------------------

class TestUnregisteredRootFailClosed(Base):
    def test_unregistered_root_fail_closed(self):
        # 'open-web' is never registered — assert_claim must reject it
        x1 = self.mk_ext("port 4242 service")
        triple = ("port", "is", "4242")

        with self.assertRaises(ValueError):
            memdag_corroborate.assert_claim(self.conn, x1, triple, "open-web")

        # Nothing was recorded
        assertion_count = self.conn.execute(
            "SELECT COUNT(*) FROM claim_assertions"
        ).fetchone()[0]
        self.assertEqual(assertion_count, 0)

        # Even if we register one OTHER root and assert under it, we still can't
        # hit k=2 because the rejected assertion gave NO credit
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        x2 = self.mk_ext("port 4242 also mentioned")
        cid = memdag_corroborate.assert_claim(self.conn, x2, triple, "root-A")

        result = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 4. test_same_root_counts_once
# ---------------------------------------------------------------------------

class TestSameRootCountsOnce(Base):
    def test_same_root_counts_once(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")

        # Two nodes, BOTH under root-A
        x1 = self.mk_ext("port 4242 node 1")
        x2 = self.mk_ext("port 4242 node 2")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-A")

        # live_root_count should be 1 (same root, not two)
        self.assertEqual(memdag_corroborate.live_root_count(self.conn, cid), 1)

        # corroborate at k=2 must return None
        result = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 5. test_idempotent_no_double_mint
# ---------------------------------------------------------------------------

class TestIdempotentNoDoubleMint(Base):
    def test_idempotent_no_double_mint(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")

        x1 = self.mk_ext("port 4242 first source")
        x2 = self.mk_ext("port 4242 second source")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-B")

        lift1 = memdag_corroborate.corroborate(self.conn, cid, k=2)
        lift2 = memdag_corroborate.corroborate(self.conn, cid, k=2)

        # Same id both times — idempotent
        self.assertIsNotNone(lift1)
        self.assertEqual(lift1, lift2)

        # Only one agent-derived node minted
        agent_count = self.conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE channel='agent-derived'"
        ).fetchone()[0]
        self.assertEqual(agent_count, 1)


# ---------------------------------------------------------------------------
# 6. test_revoke_corroborator_drops_lift
# ---------------------------------------------------------------------------

class TestRevokeCorraboratorDropsLift(Base):
    def test_revoke_corroborator_drops_lift(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")

        x1 = self.mk_ext("port 4242 A")
        x2 = self.mk_ext("port 4242 B")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-B")

        lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(lift)
        self.assertEqual(memdag.get_node(self.conn, lift)["tombstoned"], 0)

        # Revoking x1 must cascade-tombstone the lift (it's a child of x1)
        memdag.revoke_cascade(self.conn, x1, "source retracted")

        self.assertEqual(memdag.get_node(self.conn, lift)["tombstoned"], 1)

        # Now only root-B is live — below k=2, corroborate returns None
        result = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNone(result)

        # recompute_all must not resurrect the lift
        changes = memdag_recompute.recompute_all(self.conn)
        self.assertEqual(changes, [])
        self.assertEqual(memdag.get_node(self.conn, lift)["tombstoned"], 1)


# ---------------------------------------------------------------------------
# 7. test_remint_after_revoke_when_still_at_k
# ---------------------------------------------------------------------------

class TestRemintAfterRevokeWhenStillAtK(Base):
    def test_remint_after_revoke_when_still_at_k(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")
        memdag_corroborate.register_root(self.conn, "root-C", by="test")

        x1 = self.mk_ext("port 4242 A")
        x2 = self.mk_ext("port 4242 B")
        x3 = self.mk_ext("port 4242 C")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-B")
        memdag_corroborate.assert_claim(self.conn, x3, triple, "root-C")

        old_lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(old_lift)

        # Revoking x3: old_lift becomes tombstoned (it's a child of x3)
        memdag.revoke_cascade(self.conn, x3, "retracted")
        self.assertEqual(memdag.get_node(self.conn, old_lift)["tombstoned"], 1)

        # Still have root-A (x1) and root-B (x2) live -> can re-mint at k=2
        new_lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(new_lift)

        # New node, not the old tombstoned one
        self.assertNotEqual(new_lift, old_lift)

        # New lift has label 1
        self.assertEqual(memdag.get_node(self.conn, new_lift)["label"], 1)

        # Old lift remains tombstoned (immutable history)
        self.assertEqual(memdag.get_node(self.conn, old_lift)["tombstoned"], 1)


# ---------------------------------------------------------------------------
# 8. test_cap_holds_with_high_integrity_corroborators
# ---------------------------------------------------------------------------

class TestCapHoldsWithHighIntegrityCorroborators(Base):
    def test_cap_holds_with_high_integrity_corroborators(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")

        # Asserting nodes are user(2) and endorsed(3) — much higher than external(0)
        u = self.add("user node: port 4242", "user")
        e = self.add("endorsed node: port 4242", "endorsed")

        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, u, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, e, triple, "root-B")

        lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(lift)

        node = memdag.get_node(self.conn, lift)
        # Cap: label == 1 EXACTLY — never raised to match higher-integrity parents
        self.assertEqual(node["label"], 1)
        self.assertNotIn(node["label"], (2, 3))


# ---------------------------------------------------------------------------
# 9. test_lift_survives_recompute_all
# ---------------------------------------------------------------------------

class TestLiftSurvivesRecomputeAll(Base):
    def test_lift_survives_recompute_all(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")

        x1 = self.mk_ext("port 4242 A")
        x2 = self.mk_ext("port 4242 B")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-B")

        lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(lift)

        # The critical invariant: after recompute_all, lift is still label=1
        # (the elevations row is what prevents the claw-back to min(parents)=0)
        memdag_recompute.recompute_all(self.conn)

        node = memdag.get_node(self.conn, lift)
        self.assertEqual(node["label"], 1,
                         "lift was clawed back by recompute_all — elevations row missing?")
        self.assertEqual(node["tombstoned"], 0)


# ---------------------------------------------------------------------------
# 10. test_downstream_refloor_through_lift
# ---------------------------------------------------------------------------

class TestDownstreamRefloorthroughLift(Base):
    def test_downstream_refloor_through_lift(self):
        memdag_corroborate.register_root(self.conn, "root-A", by="test")
        memdag_corroborate.register_root(self.conn, "root-B", by="test")

        x1 = self.mk_ext("port 4242 A")
        x2 = self.mk_ext("port 4242 B")
        triple = ("port", "is", "4242")
        cid = memdag_corroborate.assert_claim(self.conn, x1, triple, "root-A")
        memdag_corroborate.assert_claim(self.conn, x2, triple, "root-B")

        lift = memdag_corroborate.corroborate(self.conn, cid, k=2)
        self.assertIsNotNone(lift)
        self.assertEqual(memdag.get_node(self.conn, lift)["label"], 1)

        # Derive a node from the lift — should get label 1 (the lift's label)
        d, d_label = memdag.derive_node(self.conn, "derived from lift", [lift])
        self.assertEqual(d_label, 1)

        # After revoking a corroborator, the lift is tombstoned
        memdag.revoke_cascade(self.conn, x1, "retracted")
        self.assertEqual(memdag.get_node(self.conn, lift)["tombstoned"], 1)

        # Trying to derive from the tombstoned lift must raise ValueError
        with self.assertRaises(ValueError):
            memdag.derive_node(self.conn, "should fail", [lift])


# ---------------------------------------------------------------------------
# 11. test_extract_claim_patterns
# ---------------------------------------------------------------------------

class TestExtractClaimPatterns(Base):
    def test_extract_claim_patterns(self):
        # sha256: 64 hex chars
        sha = "a" * 64
        self.assertEqual(
            memdag_corroborate.extract_claim(f"hash is {sha}"),
            ("sha256", "is", sha.lower())
        )

        # sha256: uppercase input -> lowercased output
        sha_upper = "A" * 64
        result = memdag_corroborate.extract_claim(f"hash: {sha_upper}")
        self.assertEqual(result, ("sha256", "is", sha_upper.lower()))

        # ipv4
        self.assertEqual(
            memdag_corroborate.extract_claim("server at 192.168.1.107"),
            ("ipv4", "is", "192.168.1.107")
        )

        # port
        self.assertEqual(
            memdag_corroborate.extract_claim("listening on port 4242"),
            ("port", "is", "4242")
        )

        # semver
        self.assertEqual(
            memdag_corroborate.extract_claim("nebula 1.9.5 released"),
            ("version", "is", "1.9.5")
        )

        # key=value
        self.assertEqual(
            memdag_corroborate.extract_claim("cipher = aes"),
            ("cipher", "=", "aes")
        )

        # Plain prose -> None
        self.assertIsNone(
            memdag_corroborate.extract_claim("no structure here at all")
        )

        # IP-vs-semver ordering: IP wins (ipv4 checked before semver)
        result = memdag_corroborate.extract_claim("host 10.0.0.1 connected")
        self.assertEqual(result, ("ipv4", "is", "10.0.0.1"))
        self.assertNotEqual(result[0], "version")


# ---------------------------------------------------------------------------
# 12. test_register_root_idempotent_and_validates
# ---------------------------------------------------------------------------

class TestRegisterRootIdempotentAndValidates(Base):
    def test_register_root_idempotent_and_validates(self):
        # First registration returns True
        first = memdag_corroborate.register_root(self.conn, "root-A", by="test")
        self.assertTrue(first)

        # Second registration of the same root returns False (idempotent)
        second = memdag_corroborate.register_root(self.conn, "root-A", by="test")
        self.assertFalse(second)

        # Empty root raises ValueError
        with self.assertRaises(ValueError):
            memdag_corroborate.register_root(self.conn, "", by="test")

        # Whitespace-only root raises ValueError
        with self.assertRaises(ValueError):
            memdag_corroborate.register_root(self.conn, "   ", by="test")


if __name__ == "__main__":
    unittest.main()
