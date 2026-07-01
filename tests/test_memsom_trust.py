#!/usr/bin/env python3
"""Tests for memsom_trust — trust algebra: integrity lattice + audited elevation.

Run:
  python -W error::DeprecationWarning -m unittest discover \
    -s <repo> -p test_memsom_trust.py \
    -t <repo> -v
"""

import contextlib
import io
import os
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("error", DeprecationWarning)

import memsom
import memsom_recompute
import memsom_trust


class Base(unittest.TestCase):
    """Temp-DB base class — mirrors test_memsom.py Base exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "sub" / "test.db"
        os.environ["MEMDAG_DB"] = str(self.db)
        self.conn = memsom.get_connection()
        # Ensure trust migration applied so all tests can use the table
        memsom_trust.migrate(self.conn)

    def tearDown(self):
        self.conn.close()
        os.environ.pop("MEMDAG_DB", None)
        self.tmp.cleanup()

    def add(self, content, channel):
        with self.conn:
            return memsom.insert_node(self.conn, content, channel, memsom.RANK[channel])


# ---------------------------------------------------------------------------
# 1. test_audit_row_written
# ---------------------------------------------------------------------------

class TestAuditRowWritten(Base):
    def test_audit_row_written(self):
        x = self.add("external data", "external")   # label 0

        result = memsom_trust.elevate(self.conn, x, "user", "verified source", "matt")

        # Return dict shape
        self.assertEqual(result["node"], x)
        self.assertEqual(result["from"], 0)
        self.assertEqual(result["to"], 2)
        self.assertFalse(result["forced"])

        # Audit row
        rows = memsom_trust.elevations_for(self.conn, x)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["from_label"], 0)
        self.assertEqual(r["to_label"], 2)
        self.assertEqual(r["reason"], "verified source")
        self.assertEqual(r["by"], "matt")
        self.assertEqual(r["forced"], 0)
        # ISO-8601 timestamp
        self.assertRegex(r["ts"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

        # Stored label updated
        node = memsom.get_node(self.conn, x)
        self.assertEqual(node["label"], 2)


# ---------------------------------------------------------------------------
# 2. test_descendants_refloor_upward
# ---------------------------------------------------------------------------

class TestDescendantsRefloored(Base):
    def test_descendants_refloor_upward(self):
        x = self.add("external source", "external")        # label 0
        d1, _ = memsom.derive_node(self.conn, "d1 text", [x])  # label 0
        d2, _ = memsom.derive_node(self.conn, "d2 text", [d1]) # label 0

        result = memsom_trust.elevate(self.conn, x, "user", "trust elevation", "matt")

        # changed_descendants must contain both d1 and d2 with old=0, new=2
        changed = result["changed_descendants"]
        changed_map = {nid: (old, new) for nid, old, new in changed}
        self.assertIn(d1, changed_map)
        self.assertIn(d2, changed_map)
        self.assertEqual(changed_map[d1], (0, 2))
        self.assertEqual(changed_map[d2], (0, 2))

        # Confirm stored labels
        self.assertEqual(memsom.get_node(self.conn, d1)["label"], 2)
        self.assertEqual(memsom.get_node(self.conn, d2)["label"], 2)


# ---------------------------------------------------------------------------
# 3. test_elevated_derived_node_not_clawed_back
# ---------------------------------------------------------------------------

class TestElevatedDerivedNodeFixedPoint(Base):
    def test_elevated_derived_node_not_clawed_back(self):
        x = self.add("external source", "external")        # label 0
        d, _ = memsom.derive_node(self.conn, "derived", [x])  # label 0

        # Elevate the derived node itself (from 0 to 2)
        memsom_trust.elevate(self.conn, d, "user", "manual trust", "matt")
        self.assertEqual(memsom.get_node(self.conn, d)["label"], 2)

        # Now call recompute_all directly — d has an elevation row so it's a fixed point
        changes = memsom_recompute.recompute_all(self.conn)
        changed_ids = [nid for nid, _, _ in changes]
        self.assertNotIn(d, changed_ids)

        # Label must still be 2
        self.assertEqual(memsom.get_node(self.conn, d)["label"], 2)


# ---------------------------------------------------------------------------
# 4. test_external_to_endorsed_blocked_then_forced
# ---------------------------------------------------------------------------

class TestExternalToEndorsedPolicy(Base):
    def test_external_to_endorsed_blocked_then_forced(self):
        x = self.add("external data", "external")   # label 0

        # Without force: must raise ValueError
        with self.assertRaises(ValueError) as ctx:
            memsom_trust.elevate(self.conn, x, "endorsed", "jump attempt", "matt")
        self.assertIn("blocked", str(ctx.exception))

        # Label unchanged
        self.assertEqual(memsom.get_node(self.conn, x)["label"], 0)
        # No audit row written on the failed attempt
        self.assertEqual(len(memsom_trust.elevations_for(self.conn, x)), 0)

        # With force: must succeed and record forced=1
        result = memsom_trust.elevate(self.conn, x, "endorsed", "forced jump", "matt",
                                      force=True)
        self.assertEqual(result["to"], 3)
        rows = memsom_trust.elevations_for(self.conn, x)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["forced"], 1)
        self.assertEqual(memsom.get_node(self.conn, x)["label"], 3)


# ---------------------------------------------------------------------------
# 4b. F-08: stepwise elevation can't launder external -> endorsed past the gate
# ---------------------------------------------------------------------------

class TestStepwiseElevationBlocked(Base):
    """Operator-class regression for F-08 (AUDIT 2026-06-11).

    The force gate is keyed on the IMMUTABLE channel, so 0->1->2->3 cannot
    reach ENDORSED without --force the way the old current-label check allowed.
    """

    def test_stepwise_to_endorsed_blocked_without_force(self):
        x = self.add("external content", "external")  # channel external, label 0
        # 0->1 and 1->2 are allowed (still below ENDORSED)
        memsom_trust.elevate(self.conn, x, 1, "step 1", "attacker")
        memsom_trust.elevate(self.conn, x, 2, "step 2", "attacker")
        self.assertEqual(memsom.get_node(self.conn, x)["label"], 2)
        # 2->3 on an external-channel node is the force-requiring step
        with self.assertRaises(ValueError) as ctx:
            memsom_trust.elevate(self.conn, x, 3, "step 3", "attacker")
        self.assertIn("blocked", str(ctx.exception))
        # Label never reached ENDORSED
        self.assertEqual(memsom.get_node(self.conn, x)["label"], 2)
        # And no forced=1 row was silently written
        rows = memsom_trust.elevations_for(self.conn, x)
        self.assertTrue(all(r["forced"] == 0 for r in rows))
        self.assertEqual(max(r["to_label"] for r in rows), 2)

    def test_stepwise_to_endorsed_with_force_records_forced(self):
        x = self.add("external content", "external")
        memsom_trust.elevate(self.conn, x, 1, "step 1", "matt")
        memsom_trust.elevate(self.conn, x, 2, "step 2", "matt")
        result = memsom_trust.elevate(self.conn, x, 3, "step 3", "matt", force=True)
        self.assertEqual(result["to"], 3)
        self.assertTrue(result["forced"])
        self.assertEqual(memsom.get_node(self.conn, x)["label"], 3)
        # The reaching-ENDORSED step is audited as forced=1
        rows = memsom_trust.elevations_for(self.conn, x)
        endorsed_rows = [r for r in rows if r["to_label"] == 3]
        self.assertEqual(len(endorsed_rows), 1)
        self.assertEqual(endorsed_rows[0]["forced"], 1)


# ---------------------------------------------------------------------------
# 4c. TRUST-1: a DERIVED node whose provenance floor is external(0) must not
# launder to ENDORSED without --force, even though its channel is 'agent-derived'
# (chan_rank 1).  The old channel-keyed gate let this through with forced=0.
# ---------------------------------------------------------------------------

class TestDerivedFromExternalForceGate(Base):
    def test_derived_from_external_to_endorsed_blocked(self):
        x = self.add("external data", "external")          # label 0
        d1, lbl = memsom.derive_node(self.conn, "derived", [x])  # agent-derived, label 0
        self.assertEqual(lbl, 0, "node derived from external must floor to 0")

        # Direct 0->3 on the derived node must be BLOCKED without force...
        with self.assertRaises(ValueError) as ctx:
            memsom_trust.elevate(self.conn, d1, "endorsed", "launder", "attacker")
        self.assertIn("blocked", str(ctx.exception))
        self.assertEqual(memsom.get_node(self.conn, d1)["label"], 0)
        self.assertEqual(len(memsom_trust.elevations_for(self.conn, d1)), 0,
                         "no audit row on the blocked attempt")

        # ...and the stepwise walk 0->1 then 1->3 must ALSO be blocked at the
        # ENDORSED hop (parents' floor stays 0 regardless of the node's own label).
        memsom_trust.elevate(self.conn, d1, 1, "step 1", "attacker")
        with self.assertRaises(ValueError):
            memsom_trust.elevate(self.conn, d1, 3, "step 3", "attacker")
        self.assertEqual(memsom.get_node(self.conn, d1)["label"], 1)
        rows = memsom_trust.elevations_for(self.conn, d1)
        self.assertTrue(all(r["forced"] == 0 for r in rows))
        self.assertLess(max(r["to_label"] for r in rows), 3,
                        "derived-from-external node never reached ENDORSED unforced")

    def test_derived_from_user_to_endorsed_allowed(self):
        # Control: a node derived from a USER source (floor 2) is a 2->3 jump,
        # NOT external->endorsed, so it needs no force.
        u = self.add("user fact", "user")                  # label 2
        d, lbl = memsom.derive_node(self.conn, "derived", [u])   # label 2
        self.assertEqual(lbl, 2)
        result = memsom_trust.elevate(self.conn, d, "endorsed", "ok", "matt")
        self.assertEqual(result["to"], 3)
        self.assertFalse(result["forced"])


# ---------------------------------------------------------------------------
# 5. test_lattice_laws
# ---------------------------------------------------------------------------

class TestLatticeLaws(Base):
    def test_lattice_laws(self):
        vals = [0, 1, 2, 3]

        for a in vals:
            # Idempotent
            self.assertEqual(memsom_trust.meet(a, a), a, f"meet idempotent fail a={a}")
            self.assertEqual(memsom_trust.join(a, a), a, f"join idempotent fail a={a}")

            for b in vals:
                # Commutative
                self.assertEqual(memsom_trust.meet(a, b), memsom_trust.meet(b, a),
                                 f"meet commutative fail a={a} b={b}")
                self.assertEqual(memsom_trust.join(a, b), memsom_trust.join(b, a),
                                 f"join commutative fail a={a} b={b}")

                # Absorption: join(a, meet(a, b)) == a
                self.assertEqual(memsom_trust.join(a, memsom_trust.meet(a, b)), a,
                                 f"absorption fail a={a} b={b}")

                for c in vals:
                    # Associative
                    self.assertEqual(
                        memsom_trust.meet(a, memsom_trust.meet(b, c)),
                        memsom_trust.meet(memsom_trust.meet(a, b), c),
                        f"meet associative fail a={a} b={b} c={c}"
                    )
                    self.assertEqual(
                        memsom_trust.join(a, memsom_trust.join(b, c)),
                        memsom_trust.join(memsom_trust.join(a, b), c),
                        f"join associative fail a={a} b={b} c={c}"
                    )


# ---------------------------------------------------------------------------
# 6. test_validation
# ---------------------------------------------------------------------------

class TestValidation(Base):
    def test_unknown_id(self):
        with self.assertRaises(ValueError):
            memsom_trust.elevate(self.conn, 9999, "user", "r", "matt")

    def test_tombstoned_node(self):
        x = self.add("data", "external")
        memsom.revoke_cascade(self.conn, x, "revoked")
        with self.assertRaises(ValueError):
            memsom_trust.elevate(self.conn, x, "user", "r", "matt")

    def test_same_label(self):
        x = self.add("user data", "user")   # label 2
        with self.assertRaises(ValueError) as ctx:
            memsom_trust.elevate(self.conn, x, "user", "r", "matt")
        self.assertIn("already at that label", str(ctx.exception))

    def test_downward_elevation(self):
        x = self.add("user data", "user")   # label 2
        with self.assertRaises(ValueError) as ctx:
            memsom_trust.elevate(self.conn, x, "external", "r", "matt")
        self.assertIn("lowering", str(ctx.exception))

    def test_bad_label_name(self):
        x = self.add("data", "external")
        with self.assertRaises(ValueError):
            memsom_trust.elevate(self.conn, x, "superduper", "r", "matt")

    def test_meet_out_of_range(self):
        with self.assertRaises(ValueError):
            memsom_trust.meet(0, 4)
        with self.assertRaises(ValueError):
            memsom_trust.meet(-1, 0)

    def test_join_out_of_range(self):
        with self.assertRaises(ValueError):
            memsom_trust.join(3, 5)
        with self.assertRaises(ValueError):
            memsom_trust.join(0, -1)


# ---------------------------------------------------------------------------
# 7. test_cli
# ---------------------------------------------------------------------------

class TestCli(Base):
    def run_main(self, *argv, expect_exit=None):
        buf = io.StringIO()
        if expect_exit is not None:
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(SystemExit) as cm:
                    memsom_trust.main(list(argv))
            self.assertEqual(cm.exception.code, expect_exit)
        else:
            with contextlib.redirect_stdout(buf):
                memsom_trust.main(list(argv))
        return buf.getvalue()

    def test_cli_elevate(self):
        x = self.add("external data", "external")
        out = self.run_main("elevate", str(x), "--to", "user",
                            "--reason", "verified", "--by", "matt")
        self.assertIn("EXTERNAL -> USER", out)

    def test_cli_meet(self):
        out = self.run_main("meet", "endorsed", "external")
        self.assertIn("EXTERNAL", out)

    def test_cli_join(self):
        out = self.run_main("join", "user", "external")
        self.assertIn("USER", out)

    def test_cli_meet_int(self):
        # meet(3, 1) = 1 = AGENT-DERIVED
        out = self.run_main("meet", "3", "1")
        self.assertIn("AGENT-DERIVED", out)

    def test_cli_elevations_empty(self):
        x = self.add("data", "external")
        out = self.run_main("elevations", str(x))
        self.assertIn("no elevations", out)

    def test_cli_elevations_after_elevate(self):
        x = self.add("external data", "external")
        self.run_main("elevate", str(x), "--to", "user",
                      "--reason", "verified", "--by", "matt")
        out = self.run_main("elevations", str(x))
        self.assertIn("EXTERNAL -> USER", out)

    def test_cli_elevate_error_exits_1(self):
        # Unknown node -> exit 1
        self.run_main("elevate", "9999", "--to", "user",
                      "--reason", "r", "--by", "matt", expect_exit=1)


if __name__ == "__main__":
    unittest.main()
